"""Versioned runtime parameters for compiled DY evaluator bundles.

The public parameter names in this module are deliberately independent of the
symbols used by Symbolica evaluators.  In particular, users override
``alpha_s`` while evaluators receive one common strong-coupling input.  The
UFO ``GC_*`` couplings are then formed symbolically from that single input, so
related couplings can never acquire independent rounding errors.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal, InvalidOperation, localcontext
from functools import lru_cache
from typing import Any


RUNTIME_PARAMETER_REGISTRY_KEY = "runtime_parameter_registry"
RUNTIME_PARAMETER_REGISTRY_VERSION = 2
H_NORMALISATION_FORMULA = "2*exp(2)/sqrt(pi)"


@lru_cache(maxsize=32, typed=True)
def h_normalisation(decimal_digit_precision: int | None) -> float | Decimal:
    """Evaluate the exact formula, without serializing a large coefficient.

    Direct expression evaluation supports precision-aware built-in constants;
    this Symbolica release's saved arbitrary-precision evaluators do not.
    Keep the result as an input shared by every term, never a generated float
    coefficient. The mathematical pi here is independent of coupling overrides.
    """
    from symbolica import E

    if decimal_digit_precision is not None and (
        type(decimal_digit_precision) is not int or decimal_digit_precision < 1
    ):
        raise DYRuntimeParameterError("H normalization precision must be positive.")
    digits = 32 if decimal_digit_precision is None else decimal_digit_precision
    value = E(H_NORMALISATION_FORMULA).evaluate({}, digits + 20)
    if isinstance(value, (tuple, list)):
        value, imaginary = value
        if imaginary != 0:
            raise DYRuntimeParameterError("H normalization must be real.")
    result = Decimal(str(value))
    if not result.is_finite() or result <= 0:
        raise DYRuntimeParameterError("H normalization must be finite and positive.")
    return float(result) if decimal_digit_precision is None else result

# Long enough for all currently supported fallback precisions, while remaining
# a decimal string in JSON rather than crossing a binary-float boundary.
PI_DECIMAL = (
    "3.141592653589793238462643383279502884197169399375105820974944592307816406"
    "286208998628034825342117067982148086513282306647093844609550582231725359"
)

PARAMETER_ORDER = (
    "m_top",
    "muv",
    "lambda_sq",
    "mur_sq",
    "alpha_s",
    "alpha_ew_inverse",
    "zmin",
    "zmax",
    "pi",
)

# These are the real-valued inputs actually handed to compiled and saved
# Symbolica evaluators, in deterministic order after the kinematic inputs.
EVALUATOR_PARAMETER_ORDER = (
    "m_top",
    "muv",
    "lambda_sq",
    "mur_sq",
    "zmin",
    "zmax",
    "pi",
    "strong_coupling",
    "electromagnetic_coupling",
    "h_normalisation",
)

EVALUATOR_SYMBOLS = {
    "m_top": "dy_m_top",
    "muv": "mUV",
    "lambda_sq": "Lambdasq",
    "mur_sq": "mursq",
    "zmin": "zmin",
    "zmax": "zmax",
    "pi": "dy_pi",
    "strong_coupling": "dy_G",
    "electromagnetic_coupling": "dy_ee",
    "h_normalisation": "dy_h_normalisation",
}

DERIVED_EXPRESSIONS = {
    "strong_coupling": "2*sqrt(alpha_s*pi)",
    "electromagnetic_coupling": "2*sqrt(pi/alpha_ew_inverse)",
    "GC_1": "-i*electromagnetic_coupling/3",
    "GC_10": "-strong_coupling",
    "GC_11": "i*strong_coupling",
    "GC_12": "i*strong_coupling^2",
    "h_normalisation": H_NORMALISATION_FORMULA,
}

_BUILTIN_DEFAULTS = {
    "alpha_s": "0.118",
    "alpha_ew_inverse": "132.507",
    "zmin": "0",
    "zmax": "1",
    "pi": PI_DECIMAL,
}

DY_COUPLING_NORMALISATION_CONVENTION = "-e^4*Q_d^2*g_s^2"


class DYRuntimeParameterError(ValueError):
    """Raised for malformed registries or invalid runtime overrides."""


def dy_coupling_normalisation_factor(
    alpha_s: float | Decimal | str,
    alpha_ew_inverse: float | Decimal | str,
    pi: float | Decimal | str,
) -> float | Decimal:
    """Return the DY coupling/sign factor omitted from generated integrands.

    Decimal inputs (including exact decimal strings) keep the calculation in
    decimal arithmetic.  Native numeric inputs return a binary float.  The
    GeV^-2-to-pb unit conversion is deliberately not part of this factor.
    """

    precise = any(
        isinstance(value, (Decimal, str))
        for value in (alpha_s, alpha_ew_inverse, pi)
    )
    if precise:
        values = tuple(
            Decimal(exact_decimal_string(value, name=name))
            for value, name in (
                (alpha_s, "alpha_s"),
                (alpha_ew_inverse, "alpha_ew_inverse"),
                (pi, "pi"),
            )
        )
        alpha_s_value, inverse_alpha_value, pi_value = values
    else:
        alpha_s_value = float(alpha_s)
        inverse_alpha_value = float(alpha_ew_inverse)
        pi_value = float(pi)

    if alpha_s_value <= 0 or inverse_alpha_value <= 0 or pi_value <= 0:
        raise DYRuntimeParameterError(
            "DY coupling normalisation requires positive alpha_s, "
            "alpha_ew_inverse, and pi."
        )
    if precise:
        significant_digits = max(
            len(value.as_tuple().digits)
            for value in (
                alpha_s_value,
                inverse_alpha_value,
                pi_value,
            )
        )
        with localcontext() as context:
            context.prec = max(64, significant_digits + 16)
            factor = +(
                -(4 * pi_value / inverse_alpha_value) ** 2
                * (Decimal(1) / Decimal(9))
                * (4 * pi_value * alpha_s_value)
            )
        if not factor.is_finite():
            raise DYRuntimeParameterError(
                "DY coupling normalisation is not finite."
            )
    else:
        factor = -(
            (4.0 * pi_value / inverse_alpha_value) ** 2
            * (1.0 / 9.0)
            * (4.0 * pi_value * alpha_s_value)
        )
        if not math.isfinite(factor):
            raise DYRuntimeParameterError(
                "DY coupling normalisation is not finite."
            )
    return factor


def exact_decimal_string(value: Any, *, name: str) -> str:
    """Validate *value* as a finite decimal and preserve its decimal spelling."""
    if isinstance(value, bool):
        raise DYRuntimeParameterError(
            f"DY runtime parameter '{name}' must be a finite decimal value."
        )
    if isinstance(value, Decimal):
        text = str(value)
    elif isinstance(value, int):
        text = str(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise DYRuntimeParameterError(
                f"DY runtime parameter '{name}' must be finite."
            )
        text = repr(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise DYRuntimeParameterError(
                f"DY runtime parameter '{name}' cannot be empty."
            )
    else:
        raise DYRuntimeParameterError(
            f"DY runtime parameter '{name}' must be a number or decimal string."
        )
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:
        raise DYRuntimeParameterError(
            f"DY runtime parameter '{name}' is not a valid decimal: {value!r}."
        ) from exc
    if not parsed.is_finite():
        raise DYRuntimeParameterError(
            f"DY runtime parameter '{name}' must be finite."
        )
    return text


def build_runtime_parameter_registry(
    defaults: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical JSON-serialisable registry for a new bundle."""
    merged: dict[str, Any] = dict(_BUILTIN_DEFAULTS)
    merged.update(dict(defaults))
    missing = [name for name in PARAMETER_ORDER if name not in merged]
    if missing:
        raise DYRuntimeParameterError(
            "Missing DY runtime parameter defaults: " + ", ".join(missing)
        )
    exact_defaults = {
        name: exact_decimal_string(merged[name], name=name)
        for name in PARAMETER_ORDER
    }
    registry = {
        "version": RUNTIME_PARAMETER_REGISTRY_VERSION,
        "parameter_order": list(PARAMETER_ORDER),
        "defaults": exact_defaults,
        "derived_expressions": dict(DERIVED_EXPRESSIONS),
        "evaluator_parameter_order": list(EVALUATOR_PARAMETER_ORDER),
        "evaluator_symbols": dict(EVALUATOR_SYMBOLS),
    }
    # Apply the same semantic checks used for integration-time overrides.
    resolve_runtime_parameter_values(registry, {}, decimal_digit_precision=64)
    return registry


def validate_runtime_parameter_registry(
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and detach a registry loaded from bundle metadata."""
    if not isinstance(registry, Mapping):
        raise DYRuntimeParameterError(
            "DY runtime parameter registry must be a JSON object."
        )
    version = registry.get("version")
    if (
        type(version) is not int
        or version not in (1, RUNTIME_PARAMETER_REGISTRY_VERSION)
    ):
        raise DYRuntimeParameterError(
            "Unsupported DY runtime parameter registry version "
            f"{version!r}; expected 1 or {RUNTIME_PARAMETER_REGISTRY_VERSION}."
        )
    # Version 1 bundles retain their original ABI and frozen h coefficients.
    evaluator_names = tuple(
        name for name in EVALUATOR_PARAMETER_ORDER
        if version >= 2 or name != "h_normalisation"
    )
    expected_symbols = {name: EVALUATOR_SYMBOLS[name] for name in evaluator_names}
    expected_derived = {
        name: expression for name, expression in DERIVED_EXPRESSIONS.items()
        if version >= 2 or name != "h_normalisation"
    }
    order = registry.get("parameter_order")
    if not isinstance(order, list) or not all(
        isinstance(name, str) for name in order
    ):
        raise DYRuntimeParameterError(
            "DY runtime parameter registry has an invalid parameter_order."
        )
    if len(order) != len(set(order)):
        raise DYRuntimeParameterError(
            "DY runtime parameter registry parameter_order contains duplicates."
        )
    if order != list(PARAMETER_ORDER):
        raise DYRuntimeParameterError(
            "DY runtime parameter registry has an unsupported parameter order."
        )
    defaults = registry.get("defaults")
    if not isinstance(defaults, Mapping):
        raise DYRuntimeParameterError(
            "DY runtime parameter registry has no defaults mapping."
        )
    missing = [name for name in order if name not in defaults]
    extra = sorted(set(defaults).difference(order))
    if missing:
        raise DYRuntimeParameterError(
            "Missing DY runtime parameter defaults: " + ", ".join(missing)
        )
    if extra:
        raise DYRuntimeParameterError(
            "Unknown DY runtime parameter defaults: " + ", ".join(extra)
        )
    evaluator_order = registry.get("evaluator_parameter_order")
    evaluator_symbols = registry.get("evaluator_symbols")
    if not isinstance(evaluator_order, list) or not all(
        isinstance(name, str) for name in evaluator_order
    ):
        raise DYRuntimeParameterError(
            "DY runtime parameter registry has an invalid evaluator_parameter_order."
        )
    if evaluator_order != list(evaluator_names):
        raise DYRuntimeParameterError(
            "DY runtime parameter registry has an unsupported evaluator parameter order."
        )
    if not isinstance(evaluator_symbols, Mapping):
        raise DYRuntimeParameterError(
            "DY runtime parameter registry has no evaluator_symbols mapping."
        )
    missing_symbols = [
        name for name in evaluator_order if name not in evaluator_symbols
    ]
    if missing_symbols:
        raise DYRuntimeParameterError(
            "Missing DY evaluator symbols: " + ", ".join(missing_symbols)
        )
    if any(
        not isinstance(evaluator_symbols[name], str)
        for name in evaluator_order
    ):
        raise DYRuntimeParameterError("DY evaluator symbols must be strings.")
    if dict(evaluator_symbols) != expected_symbols:
        raise DYRuntimeParameterError(
            "DY runtime parameter registry has unsupported evaluator symbols."
        )
    derived = registry.get("derived_expressions")
    if not isinstance(derived, Mapping) or any(
        not isinstance(name, str) or not isinstance(expression, str)
        for name, expression in derived.items()
    ):
        raise DYRuntimeParameterError(
            "DY runtime parameter registry has invalid derived_expressions."
        )
    if dict(derived) != expected_derived:
        raise DYRuntimeParameterError(
            "DY runtime parameter registry has unsupported derived expressions."
        )

    detached = deepcopy(dict(registry))
    detached["defaults"] = {
        name: exact_decimal_string(defaults[name], name=name) for name in order
    }
    # This also checks positivity, bounds, and mass classification.
    resolve_runtime_parameter_values(
        detached, {}, decimal_digit_precision=64, _validated=True
    )
    return detached


def _validate_resolved_base_values(
    values: Mapping[str, Decimal], defaults: Mapping[str, Decimal]
) -> None:
    for name in (
        "muv",
        "lambda_sq",
        "mur_sq",
        "alpha_s",
        "alpha_ew_inverse",
        "pi",
    ):
        if name in values and values[name] <= 0:
            raise DYRuntimeParameterError(
                f"DY runtime parameter '{name}' must be positive."
            )
    if "m_top" in values:
        if values["m_top"] < 0:
            raise DYRuntimeParameterError(
                "DY runtime parameter 'm_top' must be non-negative."
            )
        default_is_massless = defaults["m_top"].is_zero()
        override_is_massless = values["m_top"].is_zero()
        if default_is_massless != override_is_massless:
            raise DYRuntimeParameterError(
                "Changing m_top between massive and massless requires bundle "
                "regeneration."
            )
    if "zmin" in values and "zmax" in values:
        if not Decimal(0) <= values["zmin"] < values["zmax"] <= Decimal(1):
            raise DYRuntimeParameterError(
                "DY runtime parameters require 0 <= zmin < zmax <= 1."
            )


def resolve_runtime_parameter_values(
    registry: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
    *,
    decimal_digit_precision: int | None,
    _validated: bool = False,
) -> dict[str, float | Decimal]:
    """Resolve defaults and overrides for native or precision-preserving use.

    ``decimal_digit_precision=None`` selects native binary floats.  Any positive
    precision returns ``Decimal`` values, including derived couplings evaluated
    in a guarded local decimal context.
    """
    checked = (
        dict(registry)
        if _validated
        else validate_runtime_parameter_registry(registry)
    )
    order = list(checked["parameter_order"])
    supplied = {} if overrides is None else dict(overrides)
    invalid_names = [repr(name) for name in supplied if not isinstance(name, str)]
    if invalid_names:
        raise DYRuntimeParameterError(
            "DY runtime parameter names must be strings: "
            + ", ".join(invalid_names)
        )
    unknown = sorted(set(supplied).difference(order))
    if unknown:
        raise DYRuntimeParameterError(
            "Unknown DY runtime parameter override(s): " + ", ".join(unknown)
        )

    defaults = {
        name: Decimal(str(checked["defaults"][name])) for name in order
    }
    resolved = dict(defaults)
    for name, value in supplied.items():
        resolved[name] = Decimal(exact_decimal_string(value, name=name))
    _validate_resolved_base_values(resolved, defaults)

    if decimal_digit_precision is None:
        native = {name: float(value) for name, value in resolved.items()}
        overflowed = [
            name for name, value in native.items() if not math.isfinite(value)
        ]
        if overflowed:
            raise DYRuntimeParameterError(
                "DY runtime parameter(s) exceed the native floating-point "
                "range: " + ", ".join(overflowed)
            )
        underflowed = [
            name
            for name in (
                "m_top",
                "muv",
                "lambda_sq",
                "mur_sq",
                "alpha_s",
                "alpha_ew_inverse",
                "pi",
            )
            if not resolved[name].is_zero() and native[name] == 0.0
        ]
        if underflowed:
            raise DYRuntimeParameterError(
                "DY runtime parameter(s) fall below the native floating-point "
                "range: " + ", ".join(underflowed)
            )
        native["strong_coupling"] = 2.0 * math.sqrt(
            native["alpha_s"] * native["pi"]
        )
        native["electromagnetic_coupling"] = 2.0 * math.sqrt(
            native["pi"] / native["alpha_ew_inverse"]
        )
        if not all(
            math.isfinite(native[name]) and native[name] > 0.0
            for name in ("strong_coupling", "electromagnetic_coupling")
        ):
            raise DYRuntimeParameterError(
                "Derived DY couplings are outside the native floating-point range."
            )
        if checked["version"] >= 2:
            native["h_normalisation"] = h_normalisation(None)
        return native

    if decimal_digit_precision < 1:
        raise DYRuntimeParameterError(
            "DY runtime parameter precision must be positive."
        )
    significant_digits = max(
        (len(value.as_tuple().digits) for value in resolved.values()), default=1
    )
    with localcontext() as context:
        context.prec = max(decimal_digit_precision + 16, significant_digits + 8)
        precise = {name: +value for name, value in resolved.items()}
        precise["strong_coupling"] = +(
            Decimal(2) * (precise["alpha_s"] * precise["pi"]).sqrt()
        )
        precise["electromagnetic_coupling"] = +(
            Decimal(2) * (precise["pi"] / precise["alpha_ew_inverse"]).sqrt()
        )
    if checked["version"] >= 2:
        precise["h_normalisation"] = h_normalisation(decimal_digit_precision)
    return precise


def parse_runtime_parameter_assignments(
    assignments: list[str] | tuple[str, ...] | None,
) -> dict[str, str]:
    """Parse repeatable CLI ``NAME=DECIMAL`` assignments without float casts."""
    result: dict[str, str] = {}
    for assignment in assignments or ():
        name, separator, value = str(assignment).partition("=")
        name = name.strip()
        if not separator or not name:
            raise DYRuntimeParameterError(
                f"Invalid DY runtime parameter assignment {assignment!r}; "
                "expected NAME=DECIMAL."
            )
        if name in result:
            raise DYRuntimeParameterError(
                f"Duplicate DY runtime parameter override '{name}'."
            )
        result[name] = exact_decimal_string(value, name=name)
    return result


def merge_runtime_parameter_overrides(
    integration_overrides: Mapping[str, Any] | None,
    cli_overrides: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Merge integration-card and CLI values, with CLI taking precedence."""
    result: dict[str, str] = {}
    for source in (integration_overrides or {}, cli_overrides or {}):
        invalid_names = [
            repr(name) for name in source if not isinstance(name, str)
        ]
        if invalid_names:
            raise DYRuntimeParameterError(
                "DY runtime parameter names must be strings: "
                + ", ".join(invalid_names)
            )
        unknown = sorted(set(source).difference(PARAMETER_ORDER))
        if unknown:
            raise DYRuntimeParameterError(
                "Unknown DY runtime parameter override(s): " + ", ".join(unknown)
            )
        for name, value in source.items():
            result[name] = exact_decimal_string(value, name=name)
    return result
