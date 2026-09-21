# src/processes/dy/dy_compiled_bundle.py
from __future__ import annotations

import inspect
import json
import math
import os
import pickle
import shutil
import tempfile
import time
from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from functools import lru_cache
from itertools import product
from typing import Any, Callable

import numpy as np
from symbolica import AtomType, E, Evaluator, Expression, Replacement, S

from utils.utils import (
    EVALUATORS_FOLDER,
    ParamBuilder,
    PygloopEvaluator,
    logger,
    pygloopException,
)
from utils.vectors import Vector

pjoin = os.path.join

_SYMBOLICA_EVALUATOR_TAKES_CONSTANTS = (
    "constants" in inspect.signature(Expression.evaluator).parameters
)
_SYMBOLICA_HAS_EVALUATE_WITH_PREC = hasattr(Expression, "evaluate_with_prec")


def _build_symbolica_evaluator(
    expr: Expression,
    params: list[Expression],
    *,
    functions=None,
    **options,
) -> Evaluator:
    """Construct an evaluator across the Symbolica 1.5 and 2.1 APIs."""
    functions = {} if functions is None else functions
    if _SYMBOLICA_EVALUATOR_TAKES_CONSTANTS:
        return expr.evaluator({}, functions, params, **options)
    return expr.evaluator(params, functions=functions, **options)


def _real_symbolica_value(value):
    if isinstance(value, complex):
        if value.imag != 0:
            raise ValueError(f"Expected a real Symbolica value, got {value}.")
        return value.real
    if isinstance(value, (list, tuple)) and len(value) == 2:
        real, imaginary = value
        if imaginary != 0:
            raise ValueError(
                "Expected a real arbitrary-precision Symbolica value, "
                f"got ({real}, {imaginary})."
            )
        return real
    return value


def _evaluate_symbolica_expression(
    expr: Expression,
    constants: dict[Expression, object],
):
    """Evaluate a real expression across the Symbolica 1.5 and 2.1 APIs."""
    if _SYMBOLICA_HAS_EVALUATE_WITH_PREC:
        return _real_symbolica_value(expr.evaluate(constants, {}))
    return _real_symbolica_value(expr.evaluate(constants))


def _evaluate_symbolica_expression_with_prec(
    expr: Expression,
    constants: dict[Expression, object],
    decimal_digit_precision: int,
):
    """Evaluate a real expression at arbitrary precision on both APIs."""
    if _SYMBOLICA_HAS_EVALUATE_WITH_PREC:
        value = expr.evaluate_with_prec(constants, {}, decimal_digit_precision)
    else:
        value = expr.evaluate(constants, decimal_digit_precision)
    return _real_symbolica_value(value)


def _evaluate_symbolica_evaluator_with_prec(
    evaluator: Evaluator,
    inputs: list[Decimal],
    decimal_digit_precision: int,
):
    """Evaluate one real output through Symbolica's optimised evaluator."""
    if decimal_digit_precision < 1:
        raise ValueError("Symbolica evaluator precision must be positive.")

    # Symbolica 2.1 derives an evaluator input's working precision from the
    # number of significant digits carried by its Decimal.  Decimal("2.1"),
    # for example, would otherwise be evaluated with only two significant
    # digits even when evaluate_with_prec(..., 64) is requested.  Scientific
    # formatting pads (or rounds) each finite input to the requested number of
    # significant digits without introducing the binary approximation of a
    # float conversion.
    prepared_inputs = [
        _prepare_symbolica_input_with_prec(value, decimal_digit_precision)
        for value in inputs
    ]
    return _evaluate_prepared_symbolica_evaluator_with_prec(
        evaluator, prepared_inputs, decimal_digit_precision
    )


def _prepare_symbolica_input_with_prec(
    value: Decimal, decimal_digit_precision: int
) -> Decimal:
    """Pad one precise evaluator input to the requested significant digits."""
    return (
        Decimal(format(value, f".{decimal_digit_precision - 1}e"))
        if value.is_finite()
        else value
    )


def _evaluate_prepared_symbolica_evaluator_with_prec(
    evaluator: Evaluator,
    prepared_inputs: list[Decimal],
    decimal_digit_precision: int,
):
    """Evaluate inputs already padded for the requested precision."""
    if decimal_digit_precision <= 0:
        raise ValueError("decimal_digit_precision must be positive.")
    outputs = evaluator.evaluate_with_prec(
        prepared_inputs, decimal_digit_precision
    )
    if len(outputs) != 1:
        raise ValueError(
            f"Expected one Symbolica evaluator output, got {len(outputs)}."
        )
    return _real_symbolica_value(outputs[0])


from processes.dy.dy_graph_utils import (
    _strip_quotes,
)
from processes.dy.dy_stability import SoftEdgeRouting, soft_center_for_rescaling
from processes.dy.dy_runtime_parameters import (
    EVALUATOR_PARAMETER_ORDER,
    EVALUATOR_SYMBOLS,
    PI_DECIMAL,
    RUNTIME_PARAMETER_REGISTRY_KEY,
    build_runtime_parameter_registry,
    exact_decimal_string,
    resolve_runtime_parameter_values,
    validate_runtime_parameter_registry,
)

# MT = 0.69200000000000000000000000000000  # s=500
# MT = 0.46133333333333333333333333333333  # s=750
# MT = 0.346  # s=1000
# MT = 0.173  # s=2000
MT = 173
TTBAR_CM_E_SURFACE_SCHEMA = "p2_eq_minus_p1_v1"
DY_CM_EVALUATOR_SCHEMA = "p2_eq_minus_p1_all_expressions_v1"


class DYNumericalEvaluationError(pygloopException):
    """An unresolved numerical value, eligible for a higher-precision retry."""


# Integral_0^infinity exp(-t^2-t^-2) dt = sqrt(pi)*exp(-2)/2.
# The exact 2*exp(2)/sqrt(pi) is computed at the requested runtime precision.
# Keeping a shared input avoids generation-time float folding and this
# Symbolica release's large-negative-rational serialization defect.
H_FUNCTION_NORMALISATION = E(EVALUATOR_SYMBOLS["h_normalisation"])


def exponential_h_function(t: Expression) -> Expression:
    """The exponential importance map with a precision-aware normalization."""
    return (-(t ** 2) - 1 / (t ** 2)).exp() * H_FUNCTION_NORMALISATION


def _checked_saved_evaluator_bytes(evaluator: Evaluator) -> bytes:
    """Fail generation if persistence changes exact constants or instructions."""
    payload = evaluator.save()
    restored = Evaluator.load(payload)
    if repr(evaluator.get_instructions()) != repr(restored.get_instructions()):
        raise pygloopException(
            "Saved evaluator integrity failure: signed constants or instructions "
            "changed during Symbolica save/load. Do not integrate this bundle."
        )
    return payload


def heaviside_theta(x):
    if x > 0:
        return 1
    else:
        return 0


def _coerce_numeric_param(value):
    if isinstance(value, (int, float)):
        return float(value)

    value_str = str(value).strip()
    try:
        return float(value_str)
    except ValueError:
        return float(Fraction(value_str))


def _is_ttbar_process(process: str) -> bool:
    return process in {"tt~", "ttbar"}


def _format_float_expr(value: float) -> str:
    return f"{value:.17g}"


def _complex_to_symbolica_expr(value: complex) -> Expression:
    value = complex(value)
    real = 0.0 if abs(value.real) < 1.0e-15 else value.real
    imag = 0.0 if abs(value.imag) < 1.0e-15 else value.imag

    if real == 0.0 and imag == 0.0:
        return E("0")

    if imag == 0.0:
        return E(_format_float_expr(real))

    imag_coeff = E(_format_float_expr(abs(imag)))
    imag_expr = imag_coeff * E("1i")
    if imag < 0.0:
        imag_expr = -imag_expr

    if real == 0.0:
        return imag_expr

    return E(_format_float_expr(real)) + imag_expr


@lru_cache(maxsize=1)
def _sm_ttbar_couplings() -> dict[str, Expression]:
    """Return exact UFO coupling relations over common real runtime inputs."""
    strong = E(EVALUATOR_SYMBOLS["strong_coupling"])
    electric = E(EVALUATOR_SYMBOLS["electromagnetic_coupling"])
    imaginary = E("1i")
    return {
        "GC_1": -imaginary * electric / 3,
        "GC_10": -strong,
        "GC_11": imaginary * strong,
        "GC_12": imaginary * strong**2,
    }


def substitute_process_couplings(expr: Expression, process: str, L: int) -> Expression:
    """Set graph-level couplings; public DY weights restore their fixed factor.

    Keeping DY evaluator bundles coupling-stripped preserves the reusable
    graph algebra. ``DY._apply_dy_beam_weights`` applies the process-wide
    ``-e^4 Q_d^2 g_s^2`` factor exactly once at integration time.
    """
    if _is_ttbar_process(process):
        couplings = _sm_ttbar_couplings()
        for coupling_name, coupling_value in couplings.items():
            expr = expr.replace(E(coupling_name), coupling_value)
        if L == 2:
            expr = E("1i") * expr
        return expr

    expr = expr.replace(E("GC_11"), E("1"))
    expr = expr.replace(E("GC_1"), E("1"))
    expr = expr.replace(E("GC_10"), E("1"))
    expr = expr.replace(E("GC_12"), E("1"))
    return expr


class evaluate_integrand:
    def _replace_couplings(self, expr: Expression, include_tr: bool) -> Expression:
        if include_tr:
            expr = expr.replace(E("ca"), E("Nc"))
            expr = expr.replace(E("cf"), E("(Nc^2-1)/(2*Nc)"))
            expr = expr.replace(E("CA"), E("Nc"))
            expr = expr.replace(E("CF"), E("(Nc^2-1)/(2*Nc)"))
            expr = expr.replace(E("TR"), E("1/2"))
            expr = expr.replace(E("Nc"), E("3"))
        return expr

    @staticmethod
    def _routing_expression(edge_attributes: dict, key: str) -> Expression:
        value = edge_attributes.get(key, "0")
        return E(_strip_quotes(str(value)))

    def _build_ttbar_pt_sq_expression(self) -> Expression | None:
        if not _is_ttbar_process(self.process):
            return None

        graph_edges_by_id = {
            edge.get_attributes().get("id"): edge
            for edge in self.routed_integrand.cut_graph.graph.get_edges()
        }
        ttbar_momentum_components = [E("0"), E("0"), E("0")]
        found_top_edge = False

        for cut_edge in self.routed_integrand.cut_graph.final_cut:
            cut_edge_id = cut_edge.get_attributes().get("id")
            graph_edge = graph_edges_by_id.get(cut_edge_id, cut_edge)
            edge_attributes = graph_edge.get_attributes()
            particle = _strip_quotes(str(edge_attributes.get("particle", "")))
            if particle not in {"t", "t~"}:
                continue

            found_top_edge = True
            cut_sign_key = (
                "is_cut_DY" if "is_cut_DY" in edge_attributes else "is_cut"
            )
            cut_sign = self._routing_expression(edge_attributes, cut_sign_key)

            for component_index in range(1, 4):
                momentum_component = E("0")
                for loop_index in range(self.L):
                    routing_key = f"routing_k{loop_index}"
                    momentum_component += (
                        self._routing_expression(edge_attributes, routing_key)
                        * E("t")
                        * E(f"k({loop_index},{component_index})")
                    )
                momentum_component += self._routing_expression(
                    edge_attributes, "routing_p1"
                ) * E(f"p(1,{component_index})")
                momentum_component += self._routing_expression(
                    edge_attributes, "routing_p2"
                ) * E(f"p(2,{component_index})")
                ttbar_momentum_components[component_index - 1] += (
                    cut_sign * momentum_component
                )

        if not found_top_edge:
            return None

        beam_components = [
            E(f"p(1,{component_index})") for component_index in range(1, 4)
        ]
        ttbar_momentum_sq = sum(
            component**2 for component in ttbar_momentum_components
        )
        beam_momentum_sq = sum(component**2 for component in beam_components)
        ttbar_dot_beam = sum(
            ttbar_component * beam_component
            for ttbar_component, beam_component in zip(
                ttbar_momentum_components, beam_components, strict=True
            )
        )
        return ttbar_momentum_sq - ttbar_dot_beam**2 / beam_momentum_sq

    def impose_rest_frame(self, integrand):
        return integrand.replace(E("p(x_,1)"), E("0")).replace(E("p(x_,2)"), E("0"))

    @staticmethod
    def _ttbar_cm_reduce_momentum(
        loop_momentum: Expression,
        p1_coefficient: Expression,
        p2_coefficient: Expression,
    ) -> Expression:
        """Apply the exact ttbar CM identity p2=-p1 before squaring."""
        return loop_momentum + (p1_coefficient - p2_coefficient) * E("p(1)")

    @staticmethod
    def _spatial_components(momentum: Expression) -> tuple[Expression, ...]:
        """Keep a self-dot product as a manifest sum of component squares."""
        return tuple(
            momentum.replace(E("x_(y_)"), E(f"x_(y_,{component})"))
            for component in range(1, 4)
        )

    @staticmethod
    def _ttbar_cm_reduce_components(expression: Expression) -> Expression:
        """Eliminate independent p2 components from a ttbar CM expression."""
        return expression.replace_multiple([
            Replacement(E(f"p(2,{component})"), -E(f"p(1,{component})"))
            for component in range(1, 4)
        ])

    def concretise_scalar_products(self, integrand):

        if self.process == "DY":
            integrand = integrand.replace(E("m(a)^2"), E("4*z*sp3D(p(1),p(1))"))

        if self.process == "tt~":
            integrand = integrand.replace(
                E("m(t)"), E(EVALUATOR_SYMBOLS["m_top"])
            )

        return integrand.replace(
            E("sp3D(w_(x_),z_(y_))"),
            E("w_(x_,1)*z_(y_,1)+w_(x_,2)*z_(y_,2)+w_(x_,3)*z_(y_,3)"),
        )

    @staticmethod
    def drop_exact_zero_sqrts(expr: Expression) -> Expression:
        out = expr
        while True:
            changed = False
            for match in list(out.match(E("x_^(1/2)"))):
                radicand = match[E("x_")]
                try:
                    if radicand.expand().to_canonical_string() == "0":
                        out = out.replace(radicand ** E("1/2"), E("0"))
                        changed = True
                except Exception:
                    continue
            if not changed:
                return out

    def t_parametrise(self, integrand):

        t = S("t")
        for loop_index in range(self.L):
            vector = E(f"k({loop_index})")
            integrand = integrand.replace(vector, t * vector)
            for component in range(1, 4):
                momentum = E(f"k({loop_index},{component})")
                integrand = integrand.replace(momentum, t * momentum)

        if self.process == "DY":
            integrand = integrand.replace(
                E("z"),
                t**2 * E("z"),
            )

        return integrand

    def _evaluator_preparation_replacements(
        self, observable_params: dict
    ) -> list[Replacement]:
        """Build the substitutions used to prepare evaluator expressions.

        ``replace_multiple`` is simultaneous, so replacements whose old
        implementation depended on a later substitution (notably the colour
        constants through ``Nc``) are resolved directly here.
        """
        t = S("t")
        replacements = []
        for loop_index in range(self.L):
            vector = E(f"k({loop_index})")
            replacements.append(Replacement(vector, t * vector))
            for component in range(1, 4):
                momentum = E(f"k({loop_index},{component})")
                replacements.append(Replacement(momentum, t * momentum))

        if self.process == "DY":
            replacements.append(Replacement(E("z"), t**2 * E("z")))

        replacements.extend(
            [
                Replacement(E("ca"), E("3")),
                Replacement(E("CA"), E("3")),
                Replacement(E("Nc"), E("3")),
                Replacement(E("cf"), E("4/3")),
                Replacement(E("CF"), E("4/3")),
                Replacement(E("TR"), E("1/2")),
                Replacement(E("m(t)"), E(EVALUATOR_SYMBOLS["m_top"])),
                Replacement(E("MT"), E(EVALUATOR_SYMBOLS["m_top"])),
                Replacement(
                    E("\U0001d70b"),
                    E(EVALUATOR_SYMBOLS["pi"]),
                ),
            ]
        )
        return replacements

    def _prepare_evaluator_expression(
        self, expr: Expression, observable_params: dict
    ) -> Expression:
        """Concretise scalar products, then prepare an evaluator in one pass."""
        expr = self.concretise_scalar_products(expr)
        expr = expr.replace_multiple(
            self._evaluator_preparation_replacements(observable_params)
        )
        if self.process == "DY":
            expr = self._ttbar_cm_reduce_components(expr)
            expr = self.drop_exact_zero_sqrts(expr)
        return expr

    @staticmethod
    def _is_theta_function(expr: Expression) -> bool:
        return bool(expr.is_type(AtomType.Fn)) and expr.get_name() == E(
            "Θ(x)"
        ).get_name()

    @classmethod
    def _global_theta_extraction(
        cls, expr: Expression, *, extract_arguments: bool
    ) -> tuple[Expression, list[Expression]]:
        """Apply the legacy global theta scan/removal as a safe fallback."""
        matches = list(expr.match(E("Θ(x___)")))
        arguments = (
            [match[E("x___")] for match in matches] if extract_arguments else []
        )
        return expr.replace(E("Θ(x___)"), E("1")), arguments

    @classmethod
    def _extract_top_level_theta_factors(
        cls, expr: Expression, *, extract_arguments: bool = True
    ) -> tuple[Expression, list[Expression], bool]:
        """Remove immediate theta factors without traversing the large core.

        The returned boolean records whether the legacy global fallback was
        required.  Symbolica's canonical multiplication order supplies a
        deterministic order for structurally extracted theta arguments.
        """
        factors = list(expr) if bool(expr.is_type(AtomType.Mul)) else [expr]
        core_factors: list[Expression] = []
        theta_arguments: list[Expression] = []
        incompatible_theta = False

        for factor in factors:
            if not cls._is_theta_function(factor):
                core_factors.append(factor)
                continue

            arguments = list(factor)
            if len(arguments) != 1:
                incompatible_theta = True
                core_factors.append(factor)
                continue
            if extract_arguments:
                theta_arguments.append(arguments[0])

        core = E("1")
        for factor in core_factors:
            core *= factor

        # A theta below an addition, power, or another function is not an
        # immediate multiplicative constraint.  Retain the old global
        # behaviour for such expressions.
        if incompatible_theta or any(core.match(E("Θ(x___)"))):
            fallback_core, fallback_arguments = cls._global_theta_extraction(
                expr, extract_arguments=extract_arguments
            )
            return fallback_core, fallback_arguments, True

        return core, theta_arguments, False

    def set_e_surface(self):
        final_moms = []
        e_surface = E(
            "-(4*(p(1,1)^2+p(1,2)^2+p(1,3)^2))^(1/2)"
        )  # -E("s^(1/2)")  # E("-4*p(1,3)^2")  # check

        for ep in self.routed_integrand.cut_graph.final_cut:
            ep_atts = ep.get_attributes()
            id = ep_atts["id"]

            for e in self.routed_integrand.cut_graph.graph.get_edges():
                e_atts = e.get_attributes()
                if e_atts["id"] == id:
                    k_keys = ["routing_k" + str(i) for i in range(self.L)]
                    loop_coeff = [E(e_atts[rout]) for rout in k_keys]
                    particle_type = _strip_quotes(str(e_atts["particle"]))
                    mass = (
                        E("0")
                        if particle_type in ["d", "d~", "g", "ghG", "ghG~"]
                        else E(f"m({particle_type})")
                    )
                    loop_momentum = sum(
                        loop_coeff[i] * E(f"k({i})") for i in range(self.L)
                    )
                    p1_coefficient = E(e_atts["routing_p1"])
                    p2_coefficient = E(e_atts["routing_p2"])
                    if self.process in {"DY", "tt~"}:
                        final_mom = self._ttbar_cm_reduce_momentum(
                            loop_momentum,
                            p1_coefficient,
                            p2_coefficient,
                        )
                    else:
                        final_mom = (
                            loop_momentum
                            + p1_coefficient * E("p(1)")
                            + p2_coefficient * E("p(2)")
                        )
                    final_moms.append([
                        final_mom,
                        mass,
                    ])
                    if self.process in {"DY", "tt~"}:
                        components = self._spatial_components(final_mom)
                        norm_squared = sum(
                            component**2 for component in components
                        )
                    else:
                        norm_squared = self.sp3D(final_mom, final_mom)
                    e_surface += (norm_squared + mass**2) ** E("1/2")

        e_surface = self.concretise_scalar_products(e_surface)
        if len(self.routed_integrand.replacements) > 0:
            # Do a simultaneous component substitution. Sequential component
            # replacement rewrites inside already-substituted expressions and
            # breaks rotational covariance for the collinear terms.
            patt = self.concretise_scalar_products(
                self.routed_integrand.replacements[0]
            )
            repl = self.concretise_scalar_products(
                self.routed_integrand.replacements[1]
            )

            tmp_keys = [E(f"__tmp_kcomp_{i}") for i in range(1, 4)]
            patt_comps = [
                patt.replace(E("x_(y_)"), E(f"x_(y_,{i})")) for i in range(1, 4)
            ]
            repl_comps = [
                repl.replace(E("x_(y_)"), E(f"x_(y_,{i})")) for i in range(1, 4)
            ]
            if self.process in {"DY", "tt~"}:
                repl_comps = [
                    self._ttbar_cm_reduce_components(component)
                    for component in repl_comps
                ]

            for i in range(3):
                e_surface = e_surface.replace(patt_comps[i], tmp_keys[i])
            for i in range(3):
                e_surface = e_surface.replace(tmp_keys[i], repl_comps[i])

        e_surface = self.concretise_scalar_products(e_surface)
        if self.process in {"DY", "tt~"}:
            e_surface = self._ttbar_cm_reduce_components(e_surface)
        # e_surface = self.impose_rest_frame(e_surface)

        rescaled_e_surface = e_surface
        for i_loop in range(self.L):
            rescaled_e_surface = rescaled_e_surface.replace(
                E(f"k({i_loop},x_)"), E(f"t*k({i_loop},x_)")
            )
        if self.process == "DY":
            rescaled_e_surface = rescaled_e_surface.replace(E("z"), E("t^2*z"))
        # Reversal note: before this cleanup pass, we returned the rescaled
        # surface directly and left exact sqrt(0) factors unsimplified.
        return self.drop_exact_zero_sqrts(rescaled_e_surface)

    def __init__(
        self,
        L,
        process,
        routed_integrand,
        n_hornerscheme_iterations,  #: int | None = None,
        n_cpe_iterations,  #: int | None = None,
        observable_params,
        *,
        use_exponential_importance_map: bool = True,
    ):
        self.L = L
        process_key = str(process).lower()
        if process_key == "dy":
            self.process = "DY"
        elif process_key == "tt~":
            self.process = "tt~"
        else:
            raise ValueError(f"Unsupported DY evaluator process {process!r}.")
        self.routed_integrand = routed_integrand
        self.dy_cm_evaluator_schema = (
            DY_CM_EVALUATOR_SCHEMA if self.process == "DY" else None
        )
        self.use_exponential_importance_map = use_exponential_importance_map

        self.symbols = []
        for i in range(self.L):
            for j in range(1, 4):
                self.symbols.append(E(f"k({i},{j})"))
        for i in range(1, 3):
            for j in range(1, 4):
                self.symbols.append(E(f"p({i},{j})"))

        if process == "DY":
            self.symbols.append(E("z"))

        self.symbols.append(E("t"))

        # Keep one deterministic evaluator ABI for every term in a bundle.
        # Some terms do not depend on every entry, but a shared order makes
        # native, E-surface, theta, and saved fallback evaluators agree exactly.
        for parameter_name in EVALUATOR_PARAMETER_ORDER:
            parameter = E(EVALUATOR_SYMBOLS[parameter_name])
            if parameter not in self.symbols:
                self.symbols.append(parameter)

        self.observable_params = observable_params
        self.runtime_parameter_registry = build_runtime_parameter_registry({
            "m_top": observable_params.get("m_top", MT),
            "muv": observable_params.get("mUV", 1),
            "lambda_sq": observable_params.get("Lambdasq", 1),
            "mur_sq": observable_params.get("mursq", 1),
            "alpha_s": observable_params.get("alpha_s", "0.118"),
            "alpha_ew_inverse": observable_params.get(
                "alpha_ew_inverse", "132.507"
            ),
            "zmin": observable_params.get("zmin", 0),
            "zmax": observable_params.get("zmax", 1),
            "pi": observable_params.get("pi", PI_DECIMAL),
        })
        self.runtime_parameter_defaults = resolve_runtime_parameter_values(
            self.runtime_parameter_registry,
            {},
            decimal_digit_precision=None,
        )

        self.theta_expressions: list[Expression] = []
        self._theta_val: list[Evaluator] | None = None
        self.physical_z_min_theta_index: int | None = None
        self.physical_z_max_theta_index: int | None = None

        supplied_theta_expressions = getattr(
            self.routed_integrand, "theta_expressions", None
        )
        integrand_core, extracted_theta_expressions, _ = (
            self._extract_top_level_theta_factors(
                self.routed_integrand.integrand,
                extract_arguments=supplied_theta_expressions is None,
            )
        )

        # Supplied expressions are authoritative.  The structural pass still
        # removes theta factors from the integrand, but does not duplicate
        # their arguments.
        if supplied_theta_expressions is not None:
            raw_theta_expressions = supplied_theta_expressions
        else:
            raw_theta_expressions = extracted_theta_expressions

        self.theta_expressions.extend(
            self._prepare_evaluator_expression(theta_expr, observable_params)
            for theta_expr in raw_theta_expressions
        )
        self.routed_integrand.integrand = self._prepare_evaluator_expression(
            integrand_core, observable_params
        )

        if len(self.routed_integrand.cut_graph.final_cut) > 1 and self.process == "DY":
            theta_zmin_expr = E("t^2*z") - E(EVALUATOR_SYMBOLS["zmin"])
            self.physical_z_min_theta_index = len(self.theta_expressions)
            self.theta_expressions.append(theta_zmin_expr)

        if len(self.routed_integrand.cut_graph.final_cut) > 1 and self.process == "DY":
            theta_zmax_expr = E(EVALUATOR_SYMBOLS["zmax"]) - E("t^2*z")
            self.physical_z_max_theta_index = len(self.theta_expressions)
            self.theta_expressions.append(theta_zmax_expr)

        self.sp3D = S("sp3D", is_linear=True, is_symmetric=True)

        self.e_surface = self.set_e_surface()
        self.ttbar_pt_sq_expression = self._build_ttbar_pt_sq_expression()

        ht = exponential_h_function(E("t"))
        if self.process == "DY":
            jacobian = E("t") ** 5 / self.e_surface.derivative(E("t"))
        if self.process == "tt~":
            jacobian = E("t") ** (3 * self.L) / self.e_surface.derivative(E("t"))

        # This diagnostic switch controls h(t) only.  Keep the production
        # derivative and Jacobian branches below active in either mode.
        if self.use_exponential_importance_map:
            self.routed_integrand.integrand = self.routed_integrand.integrand * ht

        if self.routed_integrand.t_derivative:
            jacobian1 = 1 / self.e_surface.derivative(E("t")) ** 2 / 4

            term1 = (
                (self.routed_integrand.integrand * E("t") ** (3 * self.L)).derivative(
                    E("t")
                )
            ) * jacobian1

            jacobian2 = (
                -1
                * self.e_surface.derivative(E("t")).derivative(E("t"))
                / self.e_surface.derivative(E("t")) ** 3
                / 4
            )

            term2 = (
                self.routed_integrand.integrand * E("t") ** (3 * self.L)
            ) * jacobian2

            self.routed_integrand.integrand = term1 + term2

        else:
            self.routed_integrand.integrand = self.routed_integrand.integrand * jacobian

        if self.process == "DY":
            self.routed_integrand.integrand = self.drop_exact_zero_sqrts(
                self._ttbar_cm_reduce_components(self.routed_integrand.integrand)
            )
        self.integrand_expression = self.routed_integrand.integrand
        ## ADD THETA OF t^2 z

        # Symbolica 2.1 evaluator construction is substantially more expensive
        # than expression evaluation at a handful of validation points.  Keep
        # the compiled evaluators lazy: production compilation and compiled
        # evaluation still materialise them through the properties below,
        # while arbitrary-precision reference checks avoid paying that cost.
        self._n_hornerscheme_iterations = n_hornerscheme_iterations
        self._n_cpe_iterations = n_cpe_iterations
        self._evaluator: Evaluator | None = None

        # self.is_rescaling_necessary = True
        # if self.e_surface.derivative(E("t")) == E("0"):
        #    self.is_rescaling_necessary = False
        #    self.routed_integrand.integrand = self.routed_integrand.integrand.replace(
        #        E("z"), E("1")
        #    )

    @property
    def theta_val(self) -> list[Evaluator]:
        if self._theta_val is None:
            self._theta_val = [
                _build_symbolica_evaluator(theta_expr, self.symbols)
                for theta_expr in self.theta_expressions
            ]
        return self._theta_val

    @property
    def evaluator(self) -> Evaluator:
        if self._evaluator is None:
            self._evaluator = _build_symbolica_evaluator(
                self.integrand_expression,
                self.symbols,
                iterations=self._n_hornerscheme_iterations,
                cpe_iterations=self._n_cpe_iterations,
            )
        return self._evaluator

    def set_t_value(self, k, p1, p2, z):

        s = 4 * (p1[0] ** 2 + p1[1] ** 2 + p1[2] ** 2)

        e_surface = self.e_surface

        for j in range(self.L):
            for i in range(3):
                e_surface = e_surface.replace(E(f"k({j},{i + 1})"), k[j][i])
        for i in range(3):
            e_surface = e_surface.replace(E(f"p(1,{i + 1})"), p1[i])
        for i in range(3):
            e_surface = e_surface.replace(E(f"p(2,{i + 1})"), p2[i])
        e_surface = e_surface.replace(E("s"), s)
        if self.process == "DY":
            e_surface = e_surface.replace(E("z"), z)
        for parameter_name in EVALUATOR_PARAMETER_ORDER:
            e_surface = e_surface.replace(
                E(EVALUATOR_SYMBOLS[parameter_name]),
                E(repr(float(self.runtime_parameter_defaults[parameter_name]))),
            )

        return e_surface.nsolve(E("t"), 1.0)

    def debug_printout(self, k, p1, p2, z):
        momenta = []

        tstar = self.set_t_value(k, p1, p2, z)

        input_k = {
            E(f"k({i},{j + 1})"): k[i][j] for (i, j) in product(range(self.L), range(3))
        }
        input_p1 = {E(f"p(1,{j + 1})"): p1[j] for j in range(3)}
        input_p2 = {E(f"p(2,{j + 1})"): p2[j] for j in range(3)}
        input = input_k | input_p1 | input_p2
        input[E("z")] = z
        input[E("t")] = tstar
        input.update({
            E(EVALUATOR_SYMBOLS[name]): value
            for name, value in self.runtime_parameter_defaults.items()
            if name in EVALUATOR_PARAMETER_ORDER
        })

        for e in self.routed_integrand.cut_graph.graph.get_edges():
            e_atts = e.get_attributes()

            k_keys = ["routing_k" + str(i) for i in range(self.L)]
            loop_coeff = [E(e_atts[rout]) for rout in k_keys]
            particle_type = _strip_quotes(str(e_atts["particle"]))
            mass_sq = E("0")

            if particle_type == "a":
                mass_sq = E("t") ** 2 * E(f"m({particle_type})") ** 2

            if particle_type == "t":
                mass_sq = E(f"m({particle_type})") ** 2

            mom = (
                sum(loop_coeff[i] * E(f"k({i})") for i in range(self.L))
                + E(e_atts["routing_p1"]) * E("p(1)")
                + E(e_atts["routing_p2"]) * E("p(2)")
            )
            momenta.append([mom, mass_sq, e_atts["id"]])

        energies = {}
        masses = {}
        qmomenta = {}

        eval_emr_int = self.routed_integrand.emr_integrand
        eval_emr_int = eval_emr_int.replace(
            E("sp3(x_,y_)"), self.sp3D(E("x_"), E("y_"))
        )
        eval_emr_int = self.concretise_scalar_products(eval_emr_int)
        eval_emr_int = self._replace_couplings(eval_emr_int, include_tr=False)
        eval_emr_int = eval_emr_int.replace(E("TR"), E("1"))

        for mom, mass_sq, id in momenta:
            rep = self.routed_integrand.replacements
            # non_rep_mom = mom
            if len(rep) > 0:
                patt = rep[0]
                repl = rep[1]
                # mom = mom.replace(patt, repl)

            mom_old = self.concretise_scalar_products(mom)
            mom = self.t_parametrise(mom_old)
            mom3d = [
                mom.replace(E("k(x_)"), E(f"k(x_,{i})")).replace(
                    E("p(x_)"), E(f"p(x_,{i})")
                )
                for i in range(1, 4)
            ]

            for key, val in input.items():
                for i in range(3):
                    mom3d[i] = mom3d[i].replace(key, val)
                    mass_sq = mass_sq.replace(
                        E("m(t)"), E(EVALUATOR_SYMBOLS["m_top"])
                    ).replace(key, val)

            energy_symbol = E(f"En({id})")
            energies[energy_symbol] = (
                mom3d[0] ** 2 + mom3d[1] ** 2 + mom3d[2] ** 2 + mass_sq
            ) ** E("1/2")

            masses[E(f"m({id})^2")] = mass_sq
            qmomenta[E(f"q({id})")] = mom3d
            for i in range(1, 4):
                eval_emr_int = eval_emr_int.replace(E(f"q({id},{i})"), mom3d[i - 1])
            eval_emr_int = eval_emr_int.replace(
                energy_symbol, energies[energy_symbol]
            )
            eval_emr_int = eval_emr_int.replace(
                E("MT"), E(EVALUATOR_SYMBOLS["m_top"])
            )

        ht = exponential_h_function(E("t"))
        jacobian = self.e_surface.derivative(E("t"))

        for i in range(1, 4):
            jacobian = jacobian.replace(E(f"p(1,{i})"), p1[i - 1])
            jacobian = jacobian.replace(E(f"p(2,{i})"), p2[i - 1])

        jacobian = jacobian.replace(E("z"), z)

        # print(self.routed_integrand.cut_graph.graph)
        print("input parameters: ", input)
        print("energies:", energies)
        print("masses: ", masses)
        print("momenta: ", qmomenta)
        # print(self.routed_integrand.integrand)
        emr_int = deepcopy(self.routed_integrand.emr_integrand)
        # print("EMR: ", emr_int)
        # print("e_surface : ", self.e_surface)
        print("h(t): ", ht.replace(E("t"), tstar))
        # print(jacobian)
        # print("delta jacobian : ", jacobian.replace(E("t"), tstar).expand())
        print(
            "Evaluated EMR: ",
            eval_emr_int
            .replace(E("1000000^(1/2)"), E("1000"))
            .replace(E("(1/1000000)^(1/2)"), E("1/1000"))
            .expand(),
        )
        jac_rep = jacobian.replace(E("t"), tstar)
        for key, val in energies.items():
            jac_rep = jac_rep.replace(key, val)
        for j in range(self.L):
            for i in range(1, 4):
                jac_rep = jac_rep.replace(E(f"k({j},{i})"), k[j][i - 1])

        print("evaluated jac: ", jac_rep)

        # print("Evaluated EMR: ", emr_int)

    def param_builder(self, k, p1, p2, z):
        param_list = []
        for i in range(self.L):
            for j in range(3):
                param_list.append(k[i][j])  # noqa:PERF401
        for j in range(3):
            param_list.append(p1[j])  # noqa:PERF401
        for j in range(3):
            param_list.append(p2[j])  # noqa:PERF401
        if self.process == "DY":
            param_list.append(z)

        t_sol = self.set_t_value(k, p1, p2, z)

        param_list.append(t_sol)

        param_list.extend(
            self.runtime_parameter_defaults[name]
            for name in EVALUATOR_PARAMETER_ORDER
        )

        return param_list

    def _evaluate_expression_arb(
        self,
        expr: Expression,
        values: dict[Expression, str],
        decimal_digit_precision: int,
    ) -> Decimal | None:
        try:
            value = _evaluate_symbolica_expression_with_prec(
                expr, values, decimal_digit_precision
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return None

        try:
            decimal_value = DYCompiledBundle._decimal_from_number(value)
        except (InvalidOperation, ValueError, pygloopException):
            return None

        if decimal_value.is_nan() or not decimal_value.is_finite():
            return None
        return decimal_value

    def eval(self, k, p1, p2, z, mode="compiled", decimal_digit_precision=100):
        param_list = self.param_builder(k, p1, p2, z)
        if mode == "arb":
            string_values = {
                symbol: repr(_coerce_numeric_param(value))
                for symbol, value in zip(self.symbols, param_list, strict=True)
            }
            precise_runtime = resolve_runtime_parameter_values(
                self.runtime_parameter_registry, {},
                decimal_digit_precision=decimal_digit_precision,
            )
            string_values.update({
                E(EVALUATOR_SYMBOLS[name]): str(precise_runtime[name])
                for name in EVALUATOR_PARAMETER_ORDER
            })

            for theta_expr in self.theta_expressions:
                th_value = self._evaluate_expression_arb(
                    theta_expr, string_values, decimal_digit_precision
                )
                print("x,1-x: ", th_value)
                if th_value is None or th_value <= 0:
                    return Decimal(0)

            value = self._evaluate_expression_arb(
                self.integrand_expression,
                string_values,
                decimal_digit_precision,
            )
            return value
        if mode != "compiled":
            raise pygloopException(f"Unsupported evaluate_integrand mode '{mode}'.")

        param_list = [_coerce_numeric_param(v) for v in param_list]

        theta = 1
        for th in self.theta_val:
            print("x,1-x: ", th.evaluate(param_list)[0][0])
            theta *= heaviside_theta(th.evaluate(param_list)[0][0])  # th_tol + 1.0e-10)

        # self.debug_printout(k, p1, p2, z)

        if theta == 1:
            return self.evaluator.evaluate(param_list) * theta
        else:
            return 0


@dataclass
class DYCompiledTerm:
    evaluator_name: str
    e_surface: Expression | None
    theta_expressions: list[Expression | None]
    integrand_expression: Expression | None
    t_initial_guess: float
    graph_group_name: str | None = None
    e_surface_evaluator: Evaluator | None = None
    theta_evaluators: list[Evaluator | None] | None = None
    integrand_evaluator: Evaluator | None = None
    integrand_evaluator_parameter_order: list[Expression] | None = None
    ttbar_pt_sq_expression: Expression | None = None
    ttbar_pt_sq_evaluator: Evaluator | None = None
    approximation_type: str | None = None
    source_graph_name: str | None = None
    routed_graph_name: str | None = None
    edge_routings: dict[str, dict[str, Any]] | None = None
    closed_ghost_loop_count: int | None = None
    physical_z_min_theta_index: int | None = None
    physical_z_max_theta_index: int | None = None


@dataclass
class DYCompiledBatchRequest:
    """One row for :meth:`DYCompiledBundle.evaluate_batch`.

    ``root_share_key`` is deliberately explicit.  Equal keys permit E-surface
    roots to be shared between rows with identical kinematics (for example the
    three mUV branches of one common Monte-Carlo sample).  Orbit variants must
    use different keys.
    """

    loop_momenta: Sequence[Vector]
    p1: Vector
    p2: Vector
    z: float
    m_uv: float | Decimal | None = None
    theta_tolerance: float = 0.0
    channel_selector: int | None = None
    ttbar_pt_min: float | None = None
    integrated_uv_ct_filter: str | None = "all"
    physical_z_min: float | None = None
    physical_z_max: float | None = None
    runtime_parameters: Mapping[str, Any] | None = None
    root_share_key: Hashable | None = None
    term_observer: Any = None


@dataclass
class DYCompiledEvaluationDiagnostics:
    """Structured double-precision failures for one bundle evaluation."""

    failed_terms: set[str] = field(default_factory=set)
    failed_surfaces: set[str] = field(default_factory=set)

    @property
    def has_t_solver_failure(self) -> bool:
        return bool(self.failed_surfaces)

    def record_t_solver_failure(
        self,
        term: DYCompiledTerm,
        surface_group: int | None,
    ) -> None:
        graph_name = (
            term.source_graph_name
            or term.routed_graph_name
            or term.graph_group_name
            or "unknown-graph"
        )
        approximation = term.approximation_type or "full"
        self.failed_terms.add(
            f"{graph_name}/{approximation}/{term.evaluator_name}"
        )
        surface_suffix = (
            f"surface-group-{surface_group}"
            if surface_group is not None
            else f"surface-{term.evaluator_name}"
        )
        self.failed_surfaces.add(
            f"{graph_name}/{approximation}/{surface_suffix}"
        )


class DYDoubleRootFailure(pygloopException):
    """A compiled DY evaluation for which a float E-surface root was lost."""

    def __init__(self, diagnostics: DYCompiledEvaluationDiagnostics):
        self.diagnostics = diagnostics
        surfaces = ", ".join(sorted(diagnostics.failed_surfaces))
        super().__init__(
            "Double-precision DY t solver failed for "
            f"{surfaces or 'an unknown E-surface'}."
        )


class DYCompiledBundle:
    METADATA_FILE = "bundle_metadata.json"
    BUNDLE_FORMAT_VERSION = 10
    PHYSICAL_Z_BOUNDARY_ULPS = 32
    DOUBLE_FLOAT_PRECISION = 32
    CORE_METADATA_KEYS = frozenset(
        {
            "bundle_format_version",
            "process",
            "integrand_name",
            "n_loops",
            "fallback_precision",
            "fallback_backend",
            "fallback_backends",
            "fallback_parameter_order",
            RUNTIME_PARAMETER_REGISTRY_KEY,
            "terms",
        }
    )

    def __init__(
        self,
        process: str,
        integrand_name: str,
        n_loops: int,
        terms: list[DYCompiledTerm],
        evaluators: dict[str, PygloopEvaluator],
        *,
        metadata: Mapping[str, Any] | None = None,
    ):
        self.process = process
        self.integrand_name = integrand_name
        self.n_loops = n_loops
        self.terms = terms
        self.evaluators = evaluators
        self._metadata = deepcopy(dict(metadata)) if metadata is not None else {}
        self._ttbar_cm_e_surface = (
            self._metadata.get("ttbar_cm_e_surface_schema")
            == TTBAR_CM_E_SURFACE_SCHEMA
        )
        self._dy_cm_reduced = (
            self._metadata.get("dy_cm_evaluator_schema")
            == DY_CM_EVALUATOR_SCHEMA
        )
        raw_runtime_registry = self._metadata.get(RUNTIME_PARAMETER_REGISTRY_KEY)
        try:
            self._runtime_parameter_registry = (
                validate_runtime_parameter_registry(raw_runtime_registry)
                if raw_runtime_registry is not None
                else None
            )
        except ValueError as exc:
            raise pygloopException(
                f"Malformed DY runtime parameter registry in bundle "
                f"'{integrand_name}': {exc}"
            ) from exc
        self._runtime_parameter_cache: dict[
            tuple[int | None, tuple[tuple[str, str], ...]],
            dict[str, float | Decimal],
        ] = {}
        self.t_symbol = E("t")
        self._t_key = self.t_symbol
        self._z_key = E("z")
        self._zmin_key = E(EVALUATOR_SYMBOLS["zmin"])
        self._zmax_key = E(EVALUATOR_SYMBOLS["zmax"])
        self._muv_key = E("mUV")
        self._p11 = E("p(1,1)")
        self._p12 = E("p(1,2)")
        self._p13 = E("p(1,3)")
        self._p21 = E("p(2,1)")
        self._p22 = E("p(2,2)")
        self._p23 = E("p(2,3)")
        self._k_keys = [
            (E(f"k({i},1)"), E(f"k({i},2)"), E(f"k({i},3)")) for i in range(n_loops)
        ]
        fallback_parameter_order = self._metadata.get("fallback_parameter_order")
        if isinstance(fallback_parameter_order, list) and fallback_parameter_order:
            self._fallback_param_order = [E(str(param)) for param in fallback_parameter_order]
        else:
            self._fallback_param_order = self._fallback_params_for_n_loops(
                n_loops, self._runtime_parameter_registry
            )
        try:
            self._fallback_t_index = self._fallback_param_order.index(self._t_key)
        except ValueError:
            self._fallback_t_index = None

        self._value_key_by_name: dict[str, Expression] = {
            self._normalize_symbol_key(self._p11.to_canonical_string()): self._p11,
            self._normalize_symbol_key(self._p12.to_canonical_string()): self._p12,
            self._normalize_symbol_key(self._p13.to_canonical_string()): self._p13,
            self._normalize_symbol_key(self._p21.to_canonical_string()): self._p21,
            self._normalize_symbol_key(self._p22.to_canonical_string()): self._p22,
            self._normalize_symbol_key(self._p23.to_canonical_string()): self._p23,
            self._normalize_symbol_key(self._z_key.to_canonical_string()): self._z_key,
            self._normalize_symbol_key(
                self._muv_key.to_canonical_string()
            ): self._muv_key,
            self._normalize_symbol_key(self._t_key.to_canonical_string()): self._t_key,
        }
        for ks in self._k_keys:
            for k_expr in ks:
                self._value_key_by_name[
                    self._normalize_symbol_key(k_expr.to_canonical_string())
                ] = k_expr
        if self._runtime_parameter_registry is not None:
            evaluator_symbols = self._runtime_parameter_registry["evaluator_symbols"]
            for parameter_name in self._runtime_parameter_registry[
                "evaluator_parameter_order"
            ]:
                parameter = E(evaluator_symbols[parameter_name])
                self._value_key_by_name[
                    self._normalize_symbol_key(parameter.to_canonical_string())
                ] = parameter

        self._input_plans: dict[str, list[tuple[tuple[Expression], Expression]]] = {}
        self._input_index_plans: dict[str, list[tuple[int, Expression]]] = {}
        for evaluator_name, pe in self.evaluators.items():
            plan: list[tuple[tuple[Expression], Expression]] = []
            index_plan: list[tuple[int, Expression]] = []
            for head in pe.param_builder.order:
                key = self._normalize_symbol_key(head[0].to_canonical_string())
                if key not in self._value_key_by_name:
                    raise pygloopException(
                        f"Missing runtime key mapping for symbol '{head[0].to_canonical_string()}' "
                        f"(normalized '{key}') in compiled evaluator '{pe.name}'."
                    )
                plan.append((head, self._value_key_by_name[key]))
                pos = pe.param_builder.positions[head][0]
                index_plan.append((pos, self._value_key_by_name[key]))
            self._input_plans[evaluator_name] = plan
            self._input_index_plans[evaluator_name] = index_plan

        self._fallback_value_keys = self._value_keys_for_parameter_order(
            self._fallback_param_order
        )
        self._integrand_value_keys: dict[str, list[Expression] | None] = {
            term.evaluator_name: (
                self._value_keys_for_parameter_order(
                    term.integrand_evaluator_parameter_order
                )
                if term.integrand_evaluator_parameter_order is not None
                else None
            )
            for term in self.terms
        }

        self._t_guess_by_term = {
            t.evaluator_name: float(t.t_initial_guess) for t in self.terms
        }
        self._e_surface_group_ids = self._build_e_surface_group_ids(self.terms)
        graph_group_terms: dict[str, list[DYCompiledTerm]] = {}
        for term in self.terms:
            group_name = term.graph_group_name or self._graph_group_name_from_term(term)
            graph_group_terms.setdefault(group_name, []).append(term)
        self._graph_group_names = sorted(
            graph_group_terms.keys(), key=self._graph_group_sort_key
        )
        self._graph_group_terms = {
            group_name: graph_group_terms[group_name]
            for group_name in self._graph_group_names
        }
        self._graph_group_closed_ghost_loop_counts: dict[str, int] = {}
        legacy_groups: list[str] = []
        for group_name, group_terms in self._graph_group_terms.items():
            raw_counts = [term.closed_ghost_loop_count for term in group_terms]
            if all(count is None for count in raw_counts):
                self._graph_group_closed_ghost_loop_counts[group_name] = 0
                legacy_groups.append(group_name)
                continue
            if any(count is None for count in raw_counts):
                raise pygloopException(
                    f"DY graph group {group_name!r} mixes legacy and explicit "
                    "closed-ghost-loop metadata."
                )
            counts = set()
            for count in raw_counts:
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise pygloopException(
                        f"DY graph group {group_name!r} has invalid "
                        f"closed_ghost_loop_count={count!r}."
                    )
                counts.add(count)
            if len(counts) != 1:
                raise pygloopException(
                    f"DY graph group {group_name!r} has conflicting closed-ghost-"
                    f"loop metadata: {sorted(counts)}."
                )
            self._graph_group_closed_ghost_loop_counts[group_name] = counts.pop()
        if legacy_groups:
            logger.warning(
                "DY bundle '%s' has legacy graph groups without closed-ghost-loop "
                "metadata; applying automatic factor +1 to: %s",
                self.integrand_name,
                ", ".join(legacy_groups),
            )
        self._soft_center_term_cache: dict[
            tuple[SoftEdgeRouting, int], DYCompiledTerm
        ] = {}

    def _value_keys_for_parameter_order(
        self, parameter_order: Sequence[Expression]
    ) -> list[Expression]:
        value_keys: list[Expression] = []
        for parameter in parameter_order:
            key = self._normalize_symbol_key(parameter.to_canonical_string())
            value_key = self._value_key_by_name.get(key)
            if value_key is None:
                raise pygloopException(
                    "Missing runtime key mapping for saved evaluator symbol "
                    f"'{parameter.to_canonical_string()}'."
                )
            value_keys.append(value_key)
        return value_keys

    @staticmethod
    def _e_surface_equivalence_key(
        term: DYCompiledTerm,
    ) -> tuple[str, str | bytes] | None:
        """Return a conservative, exact key for reusable E-surface roots."""
        if term.e_surface is not None:
            return ("expression", term.e_surface.to_canonical_string())
        if term.e_surface_evaluator is None:
            return None
        try:
            payload = term.e_surface_evaluator.save()
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            # Serialisation is only needed for this optional optimisation.  A
            # non-serialisable evaluator remains valid and simply gets no key.
            return None
        return ("saved_evaluator", bytes(payload))

    @classmethod
    def _build_e_surface_group_ids(
        cls, terms: list[DYCompiledTerm]
    ) -> dict[str, int]:
        """Group byte-identical E-surfaces without retaining their payloads."""
        groups: dict[tuple[str, str | bytes], int] = {}
        group_ids: dict[str, int] = {}
        for term in terms:
            key = cls._e_surface_equivalence_key(term)
            if key is None:
                continue
            group_ids[term.evaluator_name] = groups.setdefault(key, len(groups))
        return group_ids

    @staticmethod
    def _bundle_dir(process: str, integrand_name: str) -> str:
        return pjoin(EVALUATORS_FOLDER, process, integrand_name)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a detached copy of all metadata loaded for this bundle."""
        return deepcopy(self._metadata)

    @property
    def runtime_parameter_registry(self) -> dict[str, Any] | None:
        """Return the versioned registry, or ``None`` for a legacy bundle."""
        return (
            deepcopy(self._runtime_parameter_registry)
            if self._runtime_parameter_registry is not None
            else None
        )

    @property
    def supports_runtime_parameter_overrides(self) -> bool:
        return self._runtime_parameter_registry is not None

    def resolve_runtime_parameters(
        self,
        overrides: Mapping[str, Any] | None = None,
        *,
        decimal_digit_precision: int | None = None,
    ) -> dict[str, float | Decimal]:
        """Resolve bundle defaults and overrides with a cached numeric backend."""
        supplied = {} if overrides is None else dict(overrides)
        if self._runtime_parameter_registry is None:
            if supplied:
                raise pygloopException(
                    f"DY bundle '{self.integrand_name}' predates runtime-parameter "
                    "metadata. Regenerate it before supplying runtime overrides."
                )
            return {}
        cache_key = (
            decimal_digit_precision,
            tuple(sorted((str(name), str(value)) for name, value in supplied.items())),
        )
        cached = self._runtime_parameter_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        try:
            resolved = resolve_runtime_parameter_values(
                self._runtime_parameter_registry,
                supplied,
                decimal_digit_precision=decimal_digit_precision,
            )
        except ValueError as exc:
            raise pygloopException(str(exc)) from exc
        self._runtime_parameter_cache[cache_key] = dict(resolved)
        return resolved

    @property
    def bundle_metadata(self) -> dict[str, Any]:
        """Alias for :attr:`metadata` with an explicit bundle-scoped name."""
        return self.metadata

    @property
    def provenance_metadata(self) -> dict[str, Any]:
        """Return optional top-level metadata without structural bundle fields."""
        return {
            key: deepcopy(value)
            for key, value in self._metadata.items()
            if key not in self.CORE_METADATA_KEYS
        }

    @classmethod
    def metadata_path(cls, process: str, integrand_name: str) -> str:
        """Return the metadata path for a compiled bundle."""
        return pjoin(cls._bundle_dir(process, integrand_name), cls.METADATA_FILE)

    @classmethod
    def augment_metadata_file(
        cls,
        metadata_path: str | os.PathLike[str],
        updates: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Atomically add or replace optional top-level bundle metadata.

        Structural fields are owned by the bundle writer and cannot be changed
        through this helper.  The replacement file is written in the bundle
        directory so that ``os.replace`` remains atomic on a single filesystem.
        """
        metadata_path = os.path.abspath(os.fspath(metadata_path))
        if not os.path.isfile(metadata_path):
            raise pygloopException(f"Missing bundle metadata: {metadata_path}")
        if not isinstance(updates, Mapping):
            raise pygloopException("DY bundle metadata updates must be a mapping.")

        detached_updates = deepcopy(dict(updates))
        invalid_keys = [key for key in detached_updates if not isinstance(key, str)]
        if invalid_keys:
            raise pygloopException("DY bundle metadata update keys must be strings.")
        reserved_keys = sorted(cls.CORE_METADATA_KEYS.intersection(detached_updates))
        if reserved_keys:
            raise pygloopException(
                "Cannot augment structural DY bundle metadata fields: "
                + ", ".join(reserved_keys)
            )

        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise pygloopException(
                f"Could not read DY bundle metadata '{metadata_path}': {exc}"
            ) from exc
        if not isinstance(metadata, dict):
            raise pygloopException(
                f"DY bundle metadata must contain a JSON object: {metadata_path}"
            )

        metadata.update(detached_updates)
        try:
            serialised = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        except (TypeError, ValueError) as exc:
            raise pygloopException(
                f"DY bundle metadata updates are not valid JSON: {exc}"
            ) from exc

        metadata_dir = os.path.dirname(metadata_path)
        current_mode = os.stat(metadata_path).st_mode & 0o7777
        temporary_path = None
        try:
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".{cls.METADATA_FILE}.",
                suffix=".tmp",
                dir=metadata_dir,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                os.fchmod(handle.fileno(), current_mode)
                handle.write(serialised)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, metadata_path)
            temporary_path = None
        except OSError as exc:
            raise pygloopException(
                f"Could not atomically update DY bundle metadata "
                f"'{metadata_path}': {exc}"
            ) from exc
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass

        return deepcopy(metadata)

    @classmethod
    def augment_metadata(
        cls,
        process: str,
        integrand_name: str,
        updates: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Atomically augment optional metadata for a named compiled bundle."""
        return cls.augment_metadata_file(
            cls.metadata_path(process, integrand_name), updates
        )

    @staticmethod
    def _copy_if_present(src: str, dst: str) -> None:
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

    @classmethod
    def merge_existing_bundles(
        cls,
        process: str,
        integrand_name: str,
        n_loops: int,
        source_integrand_names: list[str],
    ) -> "DYCompiledBundle":
        if not source_integrand_names:
            raise pygloopException("Cannot merge an empty list of DY bundles.")

        final_dir = cls._bundle_dir(process, integrand_name)
        staging_dir = f"{final_dir}.tmp_merge_{os.getpid()}_{time.monotonic_ns()}"
        if os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir)
        os.makedirs(staging_dir, exist_ok=True)

        final_terms = []
        fallback_precision = None
        fallback_backend = None
        fallback_backends = None
        fallback_parameter_order = None
        runtime_parameter_registry = None
        runtime_parameter_registry_seen = False
        shared_optional_metadata = None
        global_term_index = 0
        try:
            for source_integrand_name in source_integrand_names:
                source_dir = cls._bundle_dir(process, source_integrand_name)
                source_metadata_path = pjoin(source_dir, cls.METADATA_FILE)
                if not os.path.isfile(source_metadata_path):
                    raise pygloopException(
                        f"Missing source DY bundle metadata: {source_metadata_path}"
                    )

                with open(source_metadata_path, "r", encoding="utf-8") as f:
                    source_metadata = json.load(f)

                source_optional_metadata = {
                    key: deepcopy(value)
                    for key, value in source_metadata.items()
                    if key not in cls.CORE_METADATA_KEYS
                }
                if shared_optional_metadata is None:
                    shared_optional_metadata = source_optional_metadata
                elif shared_optional_metadata != source_optional_metadata:
                    # Provenance is meaningful only as one coherent record.
                    # Never attach one shard's optional metadata to a merged
                    # bundle when the source records disagree.
                    shared_optional_metadata = {}

                if source_metadata.get("process") != process:
                    raise pygloopException(
                        f"Cannot merge DY bundle '{source_integrand_name}' for "
                        f"process '{source_metadata.get('process')}' into '{process}'."
                    )
                if int(source_metadata.get("n_loops", -1)) != int(n_loops):
                    raise pygloopException(
                        f"Cannot merge DY bundle '{source_integrand_name}' with "
                        f"n_loops={source_metadata.get('n_loops')} into n_loops={n_loops}."
                    )

                source_fallback_precision = int(
                    source_metadata.get("fallback_precision", 80)
                )
                source_fallback_backend = source_metadata.get("fallback_backend", "arb")
                source_fallback_backends = source_metadata.get(
                    "fallback_backends", [source_fallback_backend]
                )
                source_fallback_order = source_metadata.get(
                    "fallback_parameter_order", []
                )
                source_runtime_registry = source_metadata.get(
                    RUNTIME_PARAMETER_REGISTRY_KEY
                )
                if fallback_precision is None:
                    fallback_precision = source_fallback_precision
                    fallback_backend = source_fallback_backend
                    fallback_backends = source_fallback_backends
                    fallback_parameter_order = source_fallback_order
                    runtime_parameter_registry = deepcopy(source_runtime_registry)
                    runtime_parameter_registry_seen = True
                elif (
                    fallback_precision != source_fallback_precision
                    or fallback_backend != source_fallback_backend
                    or fallback_backends != source_fallback_backends
                    or fallback_parameter_order != source_fallback_order
                    or runtime_parameter_registry != source_runtime_registry
                ):
                    raise pygloopException(
                        f"Cannot merge DY bundle '{source_integrand_name}' with "
                        "different fallback metadata."
                    )

                for term in source_metadata["terms"]:
                    evaluator_name = term["evaluator_name"]
                    for suffix in (
                        ".cpp",
                        ".so",
                        "_param_builder.json",
                    ):
                        cls._copy_if_present(
                            pjoin(source_dir, f"{evaluator_name}{suffix}"),
                            pjoin(staging_dir, f"{evaluator_name}{suffix}"),
                        )

                    source_additional_data = pjoin(
                        source_dir, f"{evaluator_name}_additional_data.pkl"
                    )
                    final_additional_data = pjoin(
                        staging_dir, f"{evaluator_name}_additional_data.pkl"
                    )
                    if os.path.exists(source_additional_data):
                        with open(source_additional_data, "rb") as handle:
                            additional_data = pickle.load(handle)
                        if isinstance(additional_data, dict):
                            additional_data = dict(additional_data)
                            additional_data["integrand_name"] = integrand_name
                            additional_data["term_id"] = global_term_index
                        os.makedirs(os.path.dirname(final_additional_data), exist_ok=True)
                        with open(final_additional_data, "wb") as handle:
                            pickle.dump(additional_data, handle)

                    merged_term = dict(term)

                    # Saved Symbolica evaluators support evaluate_with_prec at
                    # arbitrary precision.  Expression bodies used by bundle
                    # formats <= 5 are therefore redundant and can be very
                    # large; never propagate them into a merged bundle.
                    for expression_key in (
                        "e_surface",
                        "theta_expressions",
                        "integrand_expression",
                        "ttbar_pt_sq_expression",
                    ):
                        merged_term.pop(expression_key, None)

                    def copy_saved_evaluator(relpath, kind, theta_index=None):
                        if relpath is None:
                            return None
                        final_relpath = cls._saved_evaluator_relpath(
                            global_term_index, kind, theta_index
                        )
                        cls._copy_if_present(
                            pjoin(source_dir, relpath),
                            pjoin(staging_dir, final_relpath),
                        )
                        return final_relpath

                    merged_term["e_surface_evaluator"] = copy_saved_evaluator(
                        term.get("e_surface_evaluator"), "e_surface"
                    )
                    merged_term["integrand_evaluator"] = copy_saved_evaluator(
                        term.get("integrand_evaluator"), "integrand"
                    )
                    merged_term["ttbar_pt_sq_evaluator"] = copy_saved_evaluator(
                        term.get("ttbar_pt_sq_evaluator"), "ttbar_pt_sq"
                    )
                    merged_term["theta_evaluators"] = [
                        copy_saved_evaluator(relpath, "theta", theta_index)
                        for theta_index, relpath in enumerate(
                            term.get("theta_evaluators", [])
                        )
                    ]
                    final_terms.append(merged_term)
                    global_term_index += 1

            metadata = {
                "bundle_format_version": cls.BUNDLE_FORMAT_VERSION,
                "process": process,
                "integrand_name": integrand_name,
                "n_loops": n_loops,
                "fallback_precision": fallback_precision,
                "fallback_backend": fallback_backend,
                "fallback_backends": fallback_backends,
                "fallback_parameter_order": fallback_parameter_order,
                RUNTIME_PARAMETER_REGISTRY_KEY: runtime_parameter_registry,
                "terms": final_terms,
            }
            if not runtime_parameter_registry_seen or runtime_parameter_registry is None:
                metadata.pop(RUNTIME_PARAMETER_REGISTRY_KEY, None)
            metadata.update(shared_optional_metadata or {})
            with open(pjoin(staging_dir, cls.METADATA_FILE), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            if os.path.isdir(final_dir):
                shutil.rmtree(final_dir)
            os.replace(staging_dir, final_dir)
        except Exception:
            if os.path.isdir(staging_dir):
                shutil.rmtree(staging_dir)
            raise

        return cls.load(process, integrand_name)

    @staticmethod
    def _fallback_params_for_n_loops(
        n_loops: int,
        runtime_parameter_registry: Mapping[str, Any] | None = None,
    ) -> list[Expression]:
        params: list[Expression] = []
        for i in range(n_loops):
            params.extend([E(f"k({i},1)"), E(f"k({i},2)"), E(f"k({i},3)")])
        params.extend([
            E("p(1,1)"),
            E("p(1,2)"),
            E("p(1,3)"),
            E("p(2,1)"),
            E("p(2,2)"),
            E("p(2,3)"),
            E("z"),
            E("mUV"),
            E("t"),
        ])
        if runtime_parameter_registry is not None:
            evaluator_symbols = runtime_parameter_registry["evaluator_symbols"]
            for name in runtime_parameter_registry["evaluator_parameter_order"]:
                parameter = E(evaluator_symbols[name])
                if parameter not in params:
                    params.append(parameter)
        return params

    @staticmethod
    def _saved_evaluator_relpath(term_index: int, kind: str, theta_index: int | None = None) -> str:
        if theta_index is None:
            filename = f"term_{term_index:04d}_{kind}.sev"
        else:
            filename = f"term_{term_index:04d}_{kind}_{theta_index:04d}.sev"
        return pjoin("higher_precision_evaluators", filename)

    @classmethod
    def _write_saved_evaluator(
        cls,
        out_dir: str,
        relpath: str,
        expr: Expression,
        fallback_params: list[Expression],
    ) -> None:
        path = pjoin(out_dir, relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        evaluator = _build_symbolica_evaluator(expr, fallback_params)
        payload = _checked_saved_evaluator_bytes(evaluator)
        with open(path, "wb") as handle:
            handle.write(payload)

    @staticmethod
    def _write_existing_evaluator(
        out_dir: str,
        relpath: str,
        evaluator: Evaluator,
    ) -> None:
        path = pjoin(out_dir, relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = _checked_saved_evaluator_bytes(evaluator)
        with open(path, "wb") as handle:
            handle.write(payload)

    @staticmethod
    def _load_saved_evaluator(out_dir: str, relpath: str | None) -> Evaluator | None:
        if relpath is None:
            return None
        path = pjoin(out_dir, relpath)
        if not os.path.isfile(path):
            raise pygloopException(f"Missing saved DY higher-precision evaluator: {path}")
        with open(path, "rb") as handle:
            return Evaluator.load(handle.read())

    @staticmethod
    def _build_param_builder(symbols: list[Expression]) -> ParamBuilder:
        pb = ParamBuilder()
        pb.add_parameter_list((E("dummy"),), len(symbols))
        pb.order = []
        pb.positions = {}
        for i, s in enumerate(symbols):
            head = (s,)
            pb.order.append(head)
            pb.positions[head] = (i, i + 1)
            pb.np[i] = 0.0
        return pb

    # src/processes/dy/dy_evaluators.py

    @staticmethod
    def _normalize_symbol_key(key: str) -> str:
        # drop namespace prefix if present, keep tail symbol form
        if "::" in key:
            key = key.split("::")[-1]
        return key

    @staticmethod
    def _graph_group_name_from_evaluator_name(evaluator_name: str) -> str | None:
        if not evaluator_name.startswith("graph_"):
            return None
        graph_prefix, separator, _rest = evaluator_name.partition("_cut_")
        if separator == "":
            return None
        return graph_prefix

    @staticmethod
    def _graph_group_sort_key(group_name: str) -> tuple[int, int | str]:
        if group_name.startswith("graph_"):
            suffix = group_name.removeprefix("graph_")
            if suffix.isdigit():
                return (0, int(suffix))
        return (1, group_name)

    def _graph_group_name_from_term(self, term: DYCompiledTerm) -> str:
        evaluator = self.evaluators.get(term.evaluator_name)
        if evaluator is not None:
            group_name = evaluator.additional_data.get("graph_group_name")
            if group_name is not None:
                return str(group_name)
        parsed_group_name = self._graph_group_name_from_evaluator_name(
            term.evaluator_name
        )
        if parsed_group_name is not None:
            return parsed_group_name
        if evaluator is not None:
            source_graph_name = evaluator.additional_data.get("source_graph_name")
            if source_graph_name is not None:
                return str(source_graph_name)
        return term.evaluator_name

    def graph_channel_names(self) -> list[str]:
        return list(self._graph_group_names)

    def graph_channel_count(self) -> int:
        return len(self._graph_group_names)

    def graph_channel_closed_ghost_loop_counts(self) -> list[int]:
        return [
            self._graph_group_closed_ghost_loop_counts[group_name]
            for group_name in self._graph_group_names
        ]

    def graph_channel_automatic_factors(self) -> list[int]:
        return [
            -1 if count % 2 else 1
            for count in self.graph_channel_closed_ghost_loop_counts()
        ]

    def _automatic_graph_factor(self, term: DYCompiledTerm) -> int:
        group_name = term.graph_group_name or self._graph_group_name_from_term(term)
        return -1 if self._graph_group_closed_ghost_loop_counts[group_name] % 2 else 1

    def _apply_automatic_graph_factor(self, term: DYCompiledTerm, value):
        return value * self._automatic_graph_factor(term)

    def _term_source_graph_name(self, term: DYCompiledTerm) -> str | None:
        if term.source_graph_name is not None:
            return term.source_graph_name
        evaluator = self.evaluators.get(term.evaluator_name)
        if evaluator is None:
            return None
        value = evaluator.additional_data.get("source_graph_name")
        return str(value) if value is not None else None

    def graph_channel_source_names(self) -> list[str | None]:
        names: list[str | None] = []
        for group_name in self._graph_group_names:
            source_names = {
                source_name
                for term in self._graph_group_terms[group_name]
                if (source_name := self._term_source_graph_name(term)) is not None
            }
            if len(source_names) > 1:
                raise pygloopException(
                    f"DY graph channel {group_name!r} mixes source graphs "
                    f"{sorted(source_names)}."
                )
            names.append(next(iter(source_names), None))
        return names

    @staticmethod
    def _normalise_soft_mirror_edge_specs(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            raw_specs = [value]
        elif isinstance(value, (list, tuple)):
            raw_specs = list(value)
        else:
            raise pygloopException(
                "DY soft-mirror edges must be a string or a list of strings."
            )
        specs = tuple(str(spec).strip() for spec in raw_specs)
        if any(not spec for spec in specs):
            raise pygloopException("DY soft-mirror edge specifications cannot be empty.")
        return specs

    def resolve_soft_mirror_routing(
        self,
        specifications: Any,
        channel_selector: int | None,
    ) -> SoftEdgeRouting | None:
        """Resolve a GRAPH:EDGE selector and validate routing across all terms."""
        specs = self._normalise_soft_mirror_edge_specs(specifications)
        if not specs:
            return None

        source_names = self.graph_channel_source_names()
        known_graph_identifiers = set(self._graph_group_names)
        known_graph_identifiers.update(name for name in source_names if name is not None)
        graph_to_edge: dict[str, str] = {}
        bare_edge: str | None = None
        for spec in specs:
            graph_name, separator, edge_id = spec.partition(":")
            if separator:
                graph_name = graph_name.strip()
                edge_id = edge_id.strip()
                if not graph_name or not edge_id:
                    raise pygloopException(
                        f"Invalid DY soft-mirror edge specification {spec!r}; "
                        "expected GRAPH:EDGE."
                    )
                if graph_name not in known_graph_identifiers:
                    raise pygloopException(
                        f"Unknown DY soft-mirror graph {graph_name!r}; available "
                        f"graphs are {sorted(known_graph_identifiers)}."
                    )
                if graph_name in graph_to_edge:
                    raise pygloopException(
                        f"Duplicate DY soft-mirror specification for {graph_name!r}."
                    )
                graph_to_edge[graph_name] = edge_id
            else:
                if bare_edge is not None or graph_to_edge:
                    raise pygloopException(
                        "A bare DY soft-mirror edge cannot be combined with other "
                        "edge specifications."
                    )
                bare_edge = graph_name.strip()

        if bare_edge is not None and self.graph_channel_count() != 1:
            raise pygloopException(
                "A bare --dy-soft-mirror-edge is only valid for a single-graph "
                "bundle; use GRAPH:EDGE for multi-graph bundles."
            )
        if channel_selector is None:
            if self.graph_channel_count() != 1:
                raise pygloopException(
                    "Graph-specific DY soft mirroring in a multi-graph bundle "
                    "requires graph multi-channeling."
                )
            channel_selector = 0
        if channel_selector < 0 or channel_selector >= self.graph_channel_count():
            raise pygloopException(
                f"DY graph channel {channel_selector} is out of range for soft mirroring."
            )

        group_name = self._graph_group_names[channel_selector]
        source_name = source_names[channel_selector]
        if bare_edge is not None:
            edge_id = bare_edge
        else:
            matching_edges = {
                edge
                for identifier in (group_name, source_name)
                if identifier is not None and (edge := graph_to_edge.get(identifier))
            }
            if len(matching_edges) > 1:
                raise pygloopException(
                    f"Conflicting DY soft-mirror edges select graph {source_name or group_name}."
                )
            if not matching_edges:
                return None
            edge_id = next(iter(matching_edges))

        routing_label = source_name or group_name
        selected_routing: SoftEdgeRouting | None = None
        for term in self._graph_group_terms[group_name]:
            edge_routings = term.edge_routings
            if edge_routings is None:
                evaluator = self.evaluators.get(term.evaluator_name)
                if evaluator is not None:
                    edge_routings = evaluator.additional_data.get("edge_routings")
            if not isinstance(edge_routings, dict):
                raise pygloopException(
                    f"DY bundle {self.integrand_name!r} lacks routing metadata for "
                    f"term {term.evaluator_name!r}. Regenerate the bundle before "
                    "enabling soft mirroring."
                )
            # Reduced terms, notably integrated UV counterterms, need not retain
            # every edge of the parent graph.  They are still evaluated at the
            # parent graph's mirrored loop point.  Validate every retained copy
            # of the selected edge and require at least one such copy below.
            if edge_id not in edge_routings:
                continue
            try:
                term_routing = SoftEdgeRouting.from_metadata(
                    routing_label,
                    edge_routings[edge_id],
                )
            except ValueError as exc:
                raise pygloopException(
                    f"Invalid DY soft-mirror routing in term "
                    f"{term.evaluator_name!r}: {exc}"
                ) from exc
            if selected_routing is None:
                selected_routing = term_routing
            elif term_routing != selected_routing:
                raise pygloopException(
                    f"DY soft-mirror edge {routing_label}:{edge_id} "
                    "has inconsistent routing across cuts/terms. Regenerate with "
                    "a common LMB before enabling soft mirroring."
                )

        if selected_routing is None:
            raise pygloopException(
                f"DY bundle {self.integrand_name!r} has no routing metadata for "
                f"edge {routing_label}:{edge_id}. Regenerate the bundle "
                "and verify that the selected parent edge survives at least one term."
            )
        return selected_routing

    def _soft_center_target_term(
        self,
        routing: SoftEdgeRouting,
        channel_selector: int | None,
    ) -> DYCompiledTerm:
        """Select the unique soft term whose E-surface defines the centre."""
        if channel_selector is None:
            if self.graph_channel_count() != 1:
                raise pygloopException(
                    "Shifted DY soft centring requires graph multi-channeling."
                )
            channel_selector = 0
        cache_key = (routing, channel_selector)
        cached = self._soft_center_term_cache.get(cache_key)
        if cached is not None:
            return cached

        if channel_selector < 0 or channel_selector >= self.graph_channel_count():
            raise pygloopException(
                f"DY graph channel {channel_selector} is out of range for soft centring."
            )
        group_name = self._graph_group_names[channel_selector]
        candidates: list[DYCompiledTerm] = []
        for term in self._graph_group_terms[group_name]:
            if term.approximation_type != "soft":
                continue
            metadata = (term.edge_routings or {}).get(routing.edge_id)
            if metadata is None:
                continue
            try:
                term_routing = SoftEdgeRouting.from_metadata(
                    routing.source_graph_name, metadata
                )
            except ValueError as exc:
                raise pygloopException(
                    f"Invalid shifted soft routing in term {term.evaluator_name!r}: "
                    f"{exc}"
                ) from exc
            if term_routing == routing:
                candidates.append(term)

        if len(candidates) != 1:
            raise pygloopException(
                f"Shifted soft edge {routing.label} requires exactly one matching "
                f"'soft' term in graph channel {group_name!r}; found "
                f"{len(candidates)}."
            )
        target = candidates[0]
        if target.e_surface is None and target.e_surface_evaluator is None:
            raise pygloopException(
                f"Soft term {target.evaluator_name!r} has no saved E-surface evaluator."
            )
        self._soft_center_term_cache[cache_key] = target
        return target

    def soft_center_for_routing(
        self,
        loop_momenta: list[Vector],
        p1: Vector,
        p2: Vector,
        z: float | Decimal,
        m_uv: float | Decimal | None,
        routing: SoftEdgeRouting,
        channel_selector: int | None,
        *,
        decimal_digit_precision: int | None = None,
        runtime_parameters: Mapping[str, Any] | None = None,
    ) -> Vector:
        """Solve the selected soft E-surface and return its raw pivot centre."""
        if not routing.has_external_offset:
            raise pygloopException(
                f"Soft edge {routing.label} has no shifted centre to solve."
            )
        if len(loop_momenta) != self.n_loops:
            raise pygloopException(
                f"Soft edge {routing.label} expects {self.n_loops} loop momenta, "
                f"got {len(loop_momenta)}."
            )

        target = self._soft_center_target_term(routing, channel_selector)
        pivot_keys = self._k_keys[routing.pivot_loop_index]

        if decimal_digit_precision is None:
            vals, _externals = self._build_runtime_values(
                loop_momenta,
                p1,
                p2,
                float(z),
                float(m_uv) if m_uv is not None else None,
                runtime_parameters,
            )

            def update_float(t_value: float, values: dict[Expression, float]) -> None:
                centre = soft_center_for_rescaling(
                    loop_momenta, p1, p2, routing, t_value
                )
                for key, component in zip(
                    pivot_keys, centre.to_list(), strict=True
                ):
                    values[key] = float(component)

            t_solution = self.solve_t_newton_bisect(
                target.e_surface,
                target.e_surface_evaluator,
                vals,
                self._t_key,
                t0=target.t_initial_guess,
                max_bracket_expands=64,
                eval_map=vals,
                update_values_for_t=update_float,
                minimum_t=1.0e-100,
            )
        else:
            if decimal_digit_precision < 2:
                raise pygloopException(
                    "Shifted soft centring needs at least two decimal digits."
                )
            with localcontext() as context:
                context.prec = decimal_digit_precision + 12
                vals, _externals = self._build_runtime_values_prec(
                    loop_momenta,
                    p1,
                    p2,
                    z,
                    m_uv,
                    runtime_parameters,
                    decimal_digit_precision,
                )

                def update_precise(
                    t_value: Decimal,
                    values: dict[Expression, Decimal],
                ) -> None:
                    centre = soft_center_for_rescaling(
                        loop_momenta, p1, p2, routing, t_value
                    )
                    for key, component in zip(
                        pivot_keys, centre.to_list(), strict=True
                    ):
                        values[key] = self._decimal_from_number(component)

                t_solution = self.solve_t_convex_bisect_prec(
                    target.e_surface,
                    target.e_surface_evaluator,
                    vals,
                    self._t_key,
                    decimal_digit_precision,
                    t0=self._decimal_from_number(target.t_initial_guess),
                    max_expand_rounds=64,
                    eval_map=vals,
                    update_values_for_t=update_precise,
                    minimum_t=Decimal(1).scaleb(
                        -(decimal_digit_precision + 16)
                    ),
                )
                if t_solution is None:
                    raise pygloopException(
                        f"Could not solve the conditional soft centre for "
                        f"{routing.label}."
                    )
                return soft_center_for_rescaling(
                    loop_momenta, p1, p2, routing, t_solution
                )

        if t_solution is None:
            raise pygloopException(
                f"Could not solve the conditional soft centre for {routing.label}."
            )
        return soft_center_for_rescaling(loop_momenta, p1, p2, routing, t_solution)

    @staticmethod
    def _normalise_integrated_uv_ct_filter(value: str | None) -> str:
        if value is None:
            return "all"
        normalised = str(value).replace("-", "_")
        if normalised not in {"all", "only", "exclude"}:
            raise pygloopException(
                "DY integrated UV counterterm filter must be one of "
                "'all', 'only', or 'exclude'."
            )
        return normalised

    def _filter_terms_by_approximation_type(
        self,
        terms: list[DYCompiledTerm],
        integrated_uv_ct_filter: str | None,
    ) -> list[DYCompiledTerm]:
        integrated_uv_ct_filter = self._normalise_integrated_uv_ct_filter(
            integrated_uv_ct_filter
        )
        if integrated_uv_ct_filter == "all":
            return terms

        missing = [
            term.evaluator_name for term in terms if term.approximation_type is None
        ]
        if missing:
            raise pygloopException(
                "DY integrated UV counterterm filtering requires term "
                "approximation metadata. Regenerate the compiled DY bundle. "
                f"Missing metadata for: {', '.join(missing[:8])}"
            )

        if integrated_uv_ct_filter == "only":
            return [term for term in terms if term.approximation_type == "uv_int"]
        return [term for term in terms if term.approximation_type != "uv_int"]

    def terms_for_channel(
        self,
        channel_selector: int | None,
        integrated_uv_ct_filter: str | None = "all",
    ) -> list[DYCompiledTerm]:
        if channel_selector is None:
            terms = self.terms
        else:
            if channel_selector < 0 or channel_selector >= self.graph_channel_count():
                raise pygloopException(
                    f"DY graph channel {channel_selector} out of range for bundle "
                    f"'{self.integrand_name}' with {self.graph_channel_count()} channels."
                )
            terms = self._graph_group_terms[self._graph_group_names[channel_selector]]
        return self._filter_terms_by_approximation_type(
            terms,
            integrated_uv_ct_filter,
        )

    @staticmethod
    def _serialise_routing_value(value: Any, default: str = "0") -> str:
        if value is None:
            return default
        return _strip_quotes(str(value)).strip()

    @classmethod
    def _edge_routings_from_evaluator(
        cls,
        evaluator: Any,
        n_loops: int,
    ) -> dict[str, dict[str, Any]]:
        supplied = getattr(evaluator, "edge_routings", None)
        if supplied is not None:
            return deepcopy(supplied)

        routed_integrand = getattr(evaluator, "routed_integrand", None)
        cut_graph = getattr(routed_integrand, "cut_graph", None)
        graph = getattr(cut_graph, "graph", None)
        if graph is None:
            return {}

        routings: dict[str, dict[str, Any]] = {}
        for edge in graph.get_edges():
            attributes = edge.get_attributes()
            edge_id = cls._serialise_routing_value(attributes.get("id"), "")
            if not edge_id:
                continue
            lmb_id = attributes.get("lmb_id")
            routings[edge_id] = {
                "edge_id": edge_id,
                "particle": cls._serialise_routing_value(
                    attributes.get("particle"), ""
                ),
                "loop_coefficients": [
                    cls._serialise_routing_value(
                        attributes.get(f"routing_k{loop_index}"), "0"
                    )
                    for loop_index in range(n_loops)
                ],
                "p1_coefficient": cls._serialise_routing_value(
                    attributes.get("routing_p1"), "0"
                ),
                "p2_coefficient": cls._serialise_routing_value(
                    attributes.get("routing_p2"), "0"
                ),
                "lmb_id": (
                    cls._serialise_routing_value(lmb_id, "")
                    if lmb_id is not None
                    else None
                ),
            }
        return routings

    @classmethod
    def create_from_evaluators(
        cls,
        process: str,
        integrand_name: str,
        n_loops: int,
        observable: str,
        evaluators: list,
        fallback_precision: int = 80,
        bundle_metadata: Mapping[str, Any] | None = None,
    ) -> DYCompiledBundle:
        if len(evaluators) == 0:
            raise pygloopException(
                "Cannot create DYCompiledBundle from empty evaluator list."
            )

        out_dir = cls._bundle_dir(process, integrand_name)
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        loaded_evaluators: dict[str, PygloopEvaluator] = {}
        terms: list[DYCompiledTerm] = []
        fallback_precision = int(fallback_precision)
        if fallback_precision < 2:
            raise pygloopException("DY fallback precision must be at least two digits.")
        fallback_backend = "saved_evaluator"
        raw_runtime_registry = getattr(
            evaluators[0], "runtime_parameter_registry", None
        )
        if raw_runtime_registry is None:
            raise pygloopException(
                "Generated DY evaluators do not carry runtime-parameter defaults."
            )
        try:
            runtime_parameter_registry = validate_runtime_parameter_registry(
                raw_runtime_registry
            )
        except ValueError as exc:
            raise pygloopException(
                f"Cannot create DY bundle with malformed runtime parameters: {exc}"
            ) from exc
        for evaluator_index, evaluator in enumerate(evaluators[1:], start=1):
            if getattr(evaluator, "runtime_parameter_registry", None) != raw_runtime_registry:
                raise pygloopException(
                    "Cannot create a DY bundle from evaluators with different "
                    f"runtime-parameter registries (term {evaluator_index})."
                )
        fallback_params = cls._fallback_params_for_n_loops(
            n_loops, runtime_parameter_registry
        )

        for i, ev in enumerate(evaluators):
            evaluator_name = getattr(ev, "compiled_name", f"term_{i}_integrand")
            if evaluator_name in loaded_evaluators:
                raise pygloopException(
                    f"Duplicate compiled evaluator name '{evaluator_name}'."
                )
            theta_expressions = list(getattr(ev, "theta_expressions", []))
            physical_z_theta_indices: dict[str, int | None] = {}
            for boundary in ("min", "max"):
                attribute = f"physical_z_{boundary}_theta_index"
                theta_index = getattr(ev, attribute, None)
                if theta_index is not None and (
                    isinstance(theta_index, bool)
                    or not isinstance(theta_index, int)
                    or not 0 <= theta_index < len(theta_expressions)
                ):
                    raise pygloopException(
                        f"Generated DY evaluator '{evaluator_name}' has invalid "
                        f"{attribute}={theta_index!r} for "
                        f"{len(theta_expressions)} theta expressions."
                    )
                physical_z_theta_indices[attribute] = theta_index
            pb = cls._build_param_builder(ev.symbols)
            graph_group_name = cls._graph_group_name_from_evaluator_name(evaluator_name)
            additional_data = {
                "process": process,
                "integrand_name": integrand_name,
                "observable": observable,
                "term_id": i,
            }
            if graph_group_name is not None:
                additional_data["graph_group_name"] = graph_group_name
            for attribute, theta_index in physical_z_theta_indices.items():
                if theta_index is not None:
                    additional_data[attribute] = theta_index
            source_graph_name = getattr(ev, "source_graph_name", None)
            if source_graph_name is not None:
                source_graph_name = str(source_graph_name)
                additional_data["source_graph_name"] = source_graph_name
            routed_graph_name = getattr(ev, "routed_graph_name", None)
            if routed_graph_name is not None:
                routed_graph_name = str(routed_graph_name)
                additional_data["routed_graph_name"] = routed_graph_name
            edge_routings = cls._edge_routings_from_evaluator(ev, n_loops)
            if edge_routings:
                additional_data["edge_routings"] = edge_routings
            approximation_type = getattr(ev, "approximation_type", None)
            if approximation_type is not None:
                approximation_type = str(approximation_type)
                additional_data["approximation_type"] = approximation_type
            closed_ghost_loop_count = getattr(
                ev,
                "closed_ghost_loop_count",
                None,
            )
            if closed_ghost_loop_count is not None:
                if (
                    isinstance(closed_ghost_loop_count, bool)
                    or not isinstance(closed_ghost_loop_count, int)
                    or closed_ghost_loop_count < 0
                ):
                    raise pygloopException(
                        "Generated DY evaluator has invalid "
                        f"closed_ghost_loop_count={closed_ghost_loop_count!r}."
                    )
                additional_data["closed_ghost_loop_count"] = (
                    closed_ghost_loop_count
                )

            pe = PygloopEvaluator(
                evaluator=ev.evaluator,
                param_builder=pb,
                name=evaluator_name,
                output_length=1,
                additional_data=additional_data,
                complexified=False,
            )
            pe.compile(
                out_dir,
                optimization_level=3,  # max in your current setup
                native=True,
                inline_asm="default",
            )
            pe.save(out_dir)

            loaded_evaluators[evaluator_name] = PygloopEvaluator.load(
                out_dir, evaluator_name
            )

            e_surface = ev.e_surface
            integrand_expression = getattr(ev, "integrand_expression", None)
            ttbar_pt_sq_expression = getattr(ev, "ttbar_pt_sq_expression", None)
            e_surface_evaluator_path = None
            theta_evaluator_paths: list[str | None] = []
            integrand_evaluator_path = None
            integrand_evaluator_parameter_order = None
            ttbar_pt_sq_evaluator_path = None
            e_surface_evaluator_path = cls._saved_evaluator_relpath(
                i, "e_surface"
            )
            cls._write_saved_evaluator(
                out_dir, e_surface_evaluator_path, e_surface, fallback_params
            )
            for theta_index, theta_expr in enumerate(theta_expressions):
                theta_path = cls._saved_evaluator_relpath(
                    i, "theta", theta_index
                )
                cls._write_saved_evaluator(
                    out_dir, theta_path, theta_expr, fallback_params
                )
                theta_evaluator_paths.append(theta_path)
            if integrand_expression is not None:
                integrand_evaluator_path = cls._saved_evaluator_relpath(
                    i, "integrand"
                )
                cls._write_existing_evaluator(
                    out_dir, integrand_evaluator_path, ev.evaluator
                )
                integrand_evaluator_parameter_order = list(ev.symbols)
            if ttbar_pt_sq_expression is not None:
                ttbar_pt_sq_evaluator_path = cls._saved_evaluator_relpath(
                    i, "ttbar_pt_sq"
                )
                cls._write_saved_evaluator(
                    out_dir,
                    ttbar_pt_sq_evaluator_path,
                    ttbar_pt_sq_expression,
                    fallback_params,
                )

            terms.append(
                DYCompiledTerm(
                    evaluator_name=evaluator_name,
                    e_surface=e_surface,
                    theta_expressions=theta_expressions,
                    integrand_expression=integrand_expression,
                    ttbar_pt_sq_expression=ttbar_pt_sq_expression,
                    e_surface_evaluator=(
                        cls._load_saved_evaluator(out_dir, e_surface_evaluator_path)
                        if e_surface_evaluator_path is not None
                        else None
                    ),
                    theta_evaluators=[
                        cls._load_saved_evaluator(out_dir, theta_path)
                        for theta_path in theta_evaluator_paths
                    ],
                    integrand_evaluator=(
                        cls._load_saved_evaluator(out_dir, integrand_evaluator_path)
                        if integrand_evaluator_path is not None
                        else None
                    ),
                    integrand_evaluator_parameter_order=(
                        integrand_evaluator_parameter_order
                    ),
                    ttbar_pt_sq_evaluator=(
                        cls._load_saved_evaluator(
                            out_dir, ttbar_pt_sq_evaluator_path
                        )
                        if ttbar_pt_sq_evaluator_path is not None
                        else None
                    ),
                    t_initial_guess=1.0,
                    graph_group_name=graph_group_name,
                    approximation_type=approximation_type,
                    source_graph_name=source_graph_name,
                    routed_graph_name=routed_graph_name,
                    edge_routings=edge_routings,
                    closed_ghost_loop_count=closed_ghost_loop_count,
                    physical_z_min_theta_index=physical_z_theta_indices[
                        "physical_z_min_theta_index"
                    ],
                    physical_z_max_theta_index=physical_z_theta_indices[
                        "physical_z_max_theta_index"
                    ],
                )
            )

        metadata = {
            "bundle_format_version": cls.BUNDLE_FORMAT_VERSION,
            "process": process,
            "integrand_name": integrand_name,
            "n_loops": n_loops,
            "fallback_precision": fallback_precision,
            "fallback_backend": fallback_backend,
            "fallback_backends": ["saved_evaluator"],
            "fallback_parameter_order": [
                param.to_canonical_string() for param in fallback_params
            ],
            RUNTIME_PARAMETER_REGISTRY_KEY: runtime_parameter_registry,
            "terms": [
                {
                    "evaluator_name": t.evaluator_name,
                    "e_surface_evaluator": cls._saved_evaluator_relpath(
                        i, "e_surface"
                    ),
                    "theta_evaluators": [
                        cls._saved_evaluator_relpath(i, "theta", theta_index)
                        for theta_index, _theta in enumerate(t.theta_expressions)
                    ],
                    "integrand_evaluator": (
                        cls._saved_evaluator_relpath(i, "integrand")
                        if t.integrand_expression is not None
                        else None
                    ),
                    "integrand_evaluator_parameter_order": (
                        [
                            param.to_canonical_string()
                            for param in t.integrand_evaluator_parameter_order
                        ]
                        if t.integrand_evaluator_parameter_order is not None
                        else None
                    ),
                    "ttbar_pt_sq_evaluator": (
                        cls._saved_evaluator_relpath(i, "ttbar_pt_sq")
                        if t.ttbar_pt_sq_expression is not None
                        else None
                    ),
                    "t_initial_guess": t.t_initial_guess,
                    "graph_group_name": t.graph_group_name,
                    "approximation_type": t.approximation_type,
                    "source_graph_name": t.source_graph_name,
                    "routed_graph_name": t.routed_graph_name,
                    "edge_routings": t.edge_routings,
                    "closed_ghost_loop_count": t.closed_ghost_loop_count,
                    "physical_z_min_theta_index": (
                        t.physical_z_min_theta_index
                    ),
                    "physical_z_max_theta_index": (
                        t.physical_z_max_theta_index
                    ),
                }
                for i, t in enumerate(terms)
            ],
        }
        if bundle_metadata is not None:
            detached_metadata = deepcopy(dict(bundle_metadata))
            reserved = sorted(
                cls.CORE_METADATA_KEYS.intersection(detached_metadata)
            )
            if reserved:
                raise pygloopException(
                    "Cannot set structural DY bundle metadata fields: "
                    + ", ".join(reserved)
                )
            metadata.update(detached_metadata)
        cm_schemas = {getattr(ev, "dy_cm_evaluator_schema", None) for ev in evaluators}
        if len(cm_schemas) != 1:
            raise pygloopException("Cannot mix rest-frame and unreduced evaluators.")
        cm_schema = cm_schemas.pop()
        if cm_schema is not None:
            metadata["dy_cm_evaluator_schema"] = cm_schema
        with open(pjoin(out_dir, cls.METADATA_FILE), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return cls(
            process,
            integrand_name,
            n_loops,
            terms,
            loaded_evaluators,
            metadata=metadata,
        )

    @classmethod
    def load(cls, process: str, integrand_name: str) -> DYCompiledBundle:
        out_dir = cls._bundle_dir(process, integrand_name)
        metadata_path = pjoin(out_dir, cls.METADATA_FILE)
        if not os.path.isfile(metadata_path):
            raise pygloopException(f"Missing bundle metadata: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        terms: list[DYCompiledTerm] = []
        evaluators: dict[str, PygloopEvaluator] = {}
        bundle_format_version = int(metadata.get("bundle_format_version", 1))
        for t in metadata["terms"]:
            name = t["evaluator_name"]
            evaluators[name] = PygloopEvaluator.load(out_dir, name)
            graph_group_name = t.get("graph_group_name")
            if graph_group_name is None:
                graph_group_name = cls._graph_group_name_from_evaluator_name(name)
            approximation_type = t.get("approximation_type")
            if approximation_type is None:
                approximation_type = evaluators[name].additional_data.get(
                    "approximation_type"
                )
            source_graph_name = t.get("source_graph_name")
            if source_graph_name is None:
                source_graph_name = evaluators[name].additional_data.get(
                    "source_graph_name"
                )
            routed_graph_name = t.get("routed_graph_name")
            if routed_graph_name is None:
                routed_graph_name = evaluators[name].additional_data.get(
                    "routed_graph_name"
                )
            edge_routings = t.get("edge_routings")
            if edge_routings is None:
                edge_routings = evaluators[name].additional_data.get("edge_routings")
            closed_ghost_loop_count = t.get("closed_ghost_loop_count")
            if closed_ghost_loop_count is None:
                closed_ghost_loop_count = evaluators[name].additional_data.get(
                    "closed_ghost_loop_count"
                )
            e_surface_raw = t.get("e_surface")
            theta_raw = t.get("theta_expressions", [])
            theta_evaluator_raw = t.get("theta_evaluators", [])
            physical_z_theta_indices: dict[str, int | None] = {}
            theta_count = max(len(theta_raw), len(theta_evaluator_raw))
            for boundary in ("min", "max"):
                attribute = f"physical_z_{boundary}_theta_index"
                theta_index = t.get(attribute)
                if theta_index is None:
                    theta_index = evaluators[name].additional_data.get(attribute)
                if theta_index is not None and (
                    isinstance(theta_index, bool)
                    or not isinstance(theta_index, int)
                    or not 0 <= theta_index < theta_count
                ):
                    raise pygloopException(
                        f"DY bundle term '{name}' has invalid "
                        f"{attribute}={theta_index!r} for {theta_count} theta "
                        "expressions. Regenerate the bundle."
                    )
                physical_z_theta_indices[attribute] = theta_index
            integrand_raw = t.get("integrand_expression")
            integrand_evaluator_parameter_order_raw = t.get(
                "integrand_evaluator_parameter_order"
            )
            ttbar_pt_sq_raw = t.get("ttbar_pt_sq_expression")
            terms.append(
                DYCompiledTerm(
                    evaluator_name=name,
                    e_surface=E(e_surface_raw) if e_surface_raw is not None else None,
                    theta_expressions=[E(x) for x in theta_raw],
                    integrand_expression=(
                        E(integrand_raw)
                        if bundle_format_version >= 2
                        and integrand_raw is not None
                        else None
                    ),
                    e_surface_evaluator=cls._load_saved_evaluator(
                        out_dir, t.get("e_surface_evaluator")
                    ),
                    theta_evaluators=[
                        cls._load_saved_evaluator(out_dir, relpath)
                        for relpath in theta_evaluator_raw
                    ],
                    integrand_evaluator=cls._load_saved_evaluator(
                        out_dir, t.get("integrand_evaluator")
                    ),
                    integrand_evaluator_parameter_order=(
                        [
                            E(param)
                            for param in integrand_evaluator_parameter_order_raw
                        ]
                        if integrand_evaluator_parameter_order_raw is not None
                        else None
                    ),
                    ttbar_pt_sq_expression=(
                        E(ttbar_pt_sq_raw) if ttbar_pt_sq_raw is not None else None
                    ),
                    ttbar_pt_sq_evaluator=cls._load_saved_evaluator(
                        out_dir, t.get("ttbar_pt_sq_evaluator")
                    ),
                    t_initial_guess=float(t.get("t_initial_guess", 1.0)),
                    graph_group_name=graph_group_name,
                    approximation_type=(
                        str(approximation_type)
                        if approximation_type is not None
                        else None
                    ),
                    source_graph_name=(
                        str(source_graph_name)
                        if source_graph_name is not None
                        else None
                    ),
                    routed_graph_name=(
                        str(routed_graph_name)
                        if routed_graph_name is not None
                        else None
                    ),
                    edge_routings=(
                        deepcopy(edge_routings)
                        if isinstance(edge_routings, dict)
                        else None
                    ),
                    closed_ghost_loop_count=closed_ghost_loop_count,
                    physical_z_min_theta_index=physical_z_theta_indices[
                        "physical_z_min_theta_index"
                    ],
                    physical_z_max_theta_index=physical_z_theta_indices[
                        "physical_z_max_theta_index"
                    ],
                )
            )

        return cls(
            process=metadata["process"],
            integrand_name=metadata["integrand_name"],
            n_loops=int(metadata["n_loops"]),
            terms=terms,
            evaluators=evaluators,
            metadata=metadata,
        )

    @staticmethod
    def _strip_python_namespace(expr: Expression) -> Expression:
        # Avoid constructing namespaced symbols with potentially wrong attributes.
        canon = expr.to_canonical_string().replace("python::{}::", "")
        return E(canon)

    @staticmethod
    def _replace_values(expr: Expression, values: dict[str, float]) -> Expression:
        # out = DYCompiledBundle._strip_python_namespace(expr)
        out = expr
        for k, v in values.items():
            out = out.replace(k, E(f"{v:.16e}"))
        return out

    @staticmethod
    def _decimal_from_number(value: float | Decimal | str | int) -> Decimal:
        if isinstance(value, Decimal):
            return value
        if isinstance(value, int):
            return Decimal(value)
        if isinstance(value, float):
            if not math.isfinite(value):
                raise pygloopException(
                    f"Cannot convert non-finite float '{value}' to Decimal."
                )
            return Decimal.from_float(value)
        return Decimal(str(value))

    @staticmethod
    def _legacy_decimal_from_number(
        value: float | Decimal | str | int,
    ) -> Decimal:
        """Reproduce the pre-stability-pipeline Decimal conversion."""
        if isinstance(value, Decimal):
            return value
        if isinstance(value, int):
            return Decimal(value)
        if isinstance(value, float):
            if not math.isfinite(value):
                raise pygloopException(
                    f"Cannot convert non-finite float '{value}' to Decimal."
                )
            return Decimal(repr(value))
        return Decimal(str(value))

    def _fallback_input_values(
        self, values: dict[Expression, float | Decimal]
    ) -> list[float | Decimal]:
        return [values[param] for param in self._fallback_param_order]

    def _fallback_input_values_for_order(
        self,
        values: dict[Expression, float | Decimal],
        parameter_order: list[Expression],
    ) -> list[float | Decimal]:
        ordered_values = []
        for param in parameter_order:
            key = self._normalize_symbol_key(param.to_canonical_string())
            value_key = self._value_key_by_name.get(key)
            if value_key is None or value_key not in values:
                raise pygloopException(
                    f"Missing runtime value for fallback evaluator symbol "
                    f"'{param.to_canonical_string()}'."
                )
            ordered_values.append(values[value_key])
        return ordered_values

    @staticmethod
    def _single_evaluator_output(value):
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise pygloopException(
                    f"Expected one Symbolica evaluator output, got {len(value)}"
                )
            return DYCompiledBundle._single_evaluator_output(value[0])
        if hasattr(value, "size") and hasattr(value, "item"):
            if value.size != 1:
                raise pygloopException(
                    f"Expected one Symbolica evaluator output, got shape {value.shape}"
                )
            return value.item()
        return value

    def _evaluate_float_expression(
        self,
        expr: Expression | None,
        evaluator: Evaluator | None,
        values: dict[Expression, float],
        *,
        packed_input_values: list[float] | None = None,
    ) -> float:
        if expr is not None:
            return float(_evaluate_symbolica_expression(expr, values))
        if evaluator is None:
            raise pygloopException(
                "No float or DoubleFloat evaluator data is present in this DY bundle. "
                "Regenerate the bundle with --dy-fallback-precision 32."
            )
        value = self._single_evaluator_output(
            evaluator.evaluate(
                self._fallback_input_values(values)
                if packed_input_values is None
                else packed_input_values
            )
        )
        return float(value)

    def _evaluate_expression_with_prec(
        self,
        expr: Expression | None,
        evaluator: Evaluator | None,
        values: dict[Expression, Decimal],
        decimal_digit_precision: int,
        evaluator_parameter_order: list[Expression] | None = None,
        prepared_input_values: list[Decimal] | None = None,
    ) -> Decimal | None:
        if evaluator is not None:
            if prepared_input_values is None:
                input_values = (
                    self._fallback_input_values_for_order(
                        values, evaluator_parameter_order
                    )
                    if evaluator_parameter_order is not None
                    else self._fallback_input_values(values)
                )
                prepared_inputs = [
                    _prepare_symbolica_input_with_prec(
                        self._decimal_from_number(value),
                        decimal_digit_precision,
                    )
                    for value in input_values
                ]
            else:
                prepared_inputs = prepared_input_values
            try:
                value = _evaluate_prepared_symbolica_evaluator_with_prec(
                    evaluator,
                    prepared_inputs,
                    decimal_digit_precision,
                )
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                try:
                    complex_outputs = evaluator.evaluate_complex_with_prec(
                        [(value, Decimal(0)) for value in prepared_inputs],
                        decimal_digit_precision,
                    )
                    if len(complex_outputs) != 1:
                        return None
                    value = complex_outputs[0]
                    if not isinstance(value, (list, tuple)) or len(value) != 2:
                        return None
                    value, imaginary_part = value
                    if DYCompiledBundle._decimal_from_number(imaginary_part) != 0:
                        return None
                except BaseException as complex_exc:
                    if isinstance(complex_exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    return None
            try:
                decimal_value = DYCompiledBundle._decimal_from_number(value)
            except (InvalidOperation, ValueError, pygloopException):
                return None
            if decimal_value.is_nan() or not decimal_value.is_finite():
                return None
            return decimal_value

        if expr is None:
            raise pygloopException(
                "No saved fallback evaluator data is present in this DY bundle. "
                "Regenerate the compiled bundle."
            )
        prepared_values = {
            key: (
                Decimal(format(value, f".{decimal_digit_precision - 1}e"))
                if value.is_finite()
                else value
            )
            for key, value in values.items()
        }
        try:
            value = _evaluate_symbolica_expression_with_prec(
                expr, prepared_values, decimal_digit_precision
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return None

        try:
            decimal_value = DYCompiledBundle._decimal_from_number(value)
        except (InvalidOperation, ValueError, pygloopException):
            return None

        if decimal_value.is_nan() or not decimal_value.is_finite():
            return None
        return decimal_value

    def _evaluate_expression_with_prec_legacy(
        self,
        expr: Expression | None,
        evaluator: Evaluator | None,
        values: dict[Expression, Decimal],
        decimal_digit_precision: int,
        evaluator_parameter_order: list[Expression] | None = None,
    ) -> Decimal | None:
        """Evaluate using the historical fallback-input conversion contract."""
        if evaluator is not None:
            input_values = (
                self._fallback_input_values_for_order(
                    values, evaluator_parameter_order
                )
                if evaluator_parameter_order is not None
                else self._fallback_input_values(values)
            )
            try:
                value = self._single_evaluator_output(
                    evaluator.evaluate_with_prec(
                        input_values,
                        decimal_digit_precision,
                    )
                )
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                try:
                    complex_outputs = evaluator.evaluate_complex_with_prec(
                        [(value, Decimal(0)) for value in input_values],
                        decimal_digit_precision,
                    )
                    if len(complex_outputs) != 1:
                        return None
                    value = complex_outputs[0]
                    if not isinstance(value, (list, tuple)) or len(value) != 2:
                        return None
                    value, imaginary_part = value
                    if self._legacy_decimal_from_number(imaginary_part) != 0:
                        return None
                except BaseException as complex_exc:
                    if isinstance(complex_exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    return None
            try:
                decimal_value = self._legacy_decimal_from_number(value)
            except (InvalidOperation, ValueError, pygloopException):
                return None
            if decimal_value.is_nan() or not decimal_value.is_finite():
                return None
            return decimal_value

        if expr is None:
            raise pygloopException(
                "No saved fallback evaluator data is present in this DY bundle. "
                "Regenerate the compiled bundle."
            )
        string_values = {key: str(value) for key, value in values.items()}
        try:
            value = _evaluate_symbolica_expression_with_prec(
                expr, string_values, decimal_digit_precision
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return None
        try:
            decimal_value = self._legacy_decimal_from_number(value)
        except (InvalidOperation, ValueError, pygloopException):
            return None
        if decimal_value.is_nan() or not decimal_value.is_finite():
            return None
        return decimal_value

    def _ttbar_pt_cut_passes(
        self,
        term: DYCompiledTerm,
        vals: dict[Expression, float],
        ttbar_pt_min: float | None,
    ) -> bool:
        if ttbar_pt_min is None:
            return True
        if term.ttbar_pt_sq_expression is None and term.ttbar_pt_sq_evaluator is None:
            raise pygloopException(
                f"DY bundle '{self.integrand_name}' does not contain ttbar pT cut "
                "metadata. Regenerate the DY bundle before using --dy-ttbar-pt-min."
            )
        pt_sq = self._evaluate_float_expression(
            term.ttbar_pt_sq_expression,
            term.ttbar_pt_sq_evaluator,
            vals,
        )
        return pt_sq >= float(ttbar_pt_min) ** 2

    def _ttbar_pt_cut_passes_with_prec(
        self,
        term: DYCompiledTerm,
        vals: dict[Expression, Decimal],
        ttbar_pt_min: float | None,
        decimal_digit_precision: int,
    ) -> bool:
        if ttbar_pt_min is None:
            return True
        if term.ttbar_pt_sq_expression is None and term.ttbar_pt_sq_evaluator is None:
            raise pygloopException(
                f"DY bundle '{self.integrand_name}' does not contain ttbar pT cut "
                "metadata. Regenerate the DY bundle before using --dy-ttbar-pt-min."
            )
        pt_sq = self._evaluate_expression_with_prec(
            term.ttbar_pt_sq_expression,
            term.ttbar_pt_sq_evaluator,
            vals,
            decimal_digit_precision,
        )
        if pt_sq is None:
            raise pygloopException(
                f"Failed to evaluate ttbar pT cut for DY term '{term.evaluator_name}'."
            )
        pt_min = Decimal(str(ttbar_pt_min))
        return pt_sq >= pt_min * pt_min

    def _ttbar_pt_cut_passes_with_prec_legacy(
        self,
        term: DYCompiledTerm,
        vals: dict[Expression, Decimal],
        ttbar_pt_min: float | None,
        decimal_digit_precision: int,
    ) -> bool:
        if ttbar_pt_min is None:
            return True
        if term.ttbar_pt_sq_expression is None and term.ttbar_pt_sq_evaluator is None:
            raise pygloopException(
                f"DY bundle '{self.integrand_name}' does not contain ttbar pT cut "
                "metadata. Regenerate the DY bundle before using --dy-ttbar-pt-min."
            )
        pt_sq = self._evaluate_expression_with_prec_legacy(
            term.ttbar_pt_sq_expression,
            term.ttbar_pt_sq_evaluator,
            vals,
            decimal_digit_precision,
        )
        if pt_sq is None:
            raise pygloopException(
                f"Failed to evaluate ttbar pT cut for DY term '{term.evaluator_name}'."
            )
        pt_min = Decimal(str(ttbar_pt_min))
        return pt_sq >= pt_min * pt_min

    @classmethod
    def _physical_z_boundary_close(cls, physical_z, boundary) -> bool:
        try:
            physical_z_float = float(physical_z)
            boundary_float = float(boundary)
        except (TypeError, ValueError, OverflowError):
            return False
        if not (
            math.isfinite(physical_z_float) and math.isfinite(boundary_float)
        ):
            return False
        scale = max(abs(physical_z_float), abs(boundary_float), 1.0)
        tolerance = cls.PHYSICAL_Z_BOUNDARY_ULPS * math.ulp(scale)
        return abs(physical_z_float - boundary_float) <= tolerance

    @classmethod
    def _physical_z_cut_passes(
        cls,
        physical_z,
        physical_z_min,
        physical_z_max,
    ) -> bool:
        if (
            physical_z_min is not None
            and physical_z < physical_z_min
            and not cls._physical_z_boundary_close(physical_z, physical_z_min)
        ):
            return False
        if (
            physical_z_max is not None
            and physical_z > physical_z_max
            and not cls._physical_z_boundary_close(physical_z, physical_z_max)
        ):
            return False
        return True

    def _theta_cut_passes(
        self,
        term: DYCompiledTerm,
        theta_index: int,
        theta_value,
        theta_tolerance,
        physical_z,
        values: Mapping[Expression, Any],
    ) -> bool:
        if theta_value is None:
            if self._dy_cm_reduced:
                raise DYNumericalEvaluationError(
                    f"Nonfinite cut constraint {theta_index} for {term.evaluator_name}."
                )
            return False
        if theta_value >= -theta_tolerance:
            return True

        boundary_key: Expression | None = None
        if theta_index == term.physical_z_min_theta_index:
            boundary_key = self._zmin_key
        elif theta_index == term.physical_z_max_theta_index:
            boundary_key = self._zmax_key
        if boundary_key is None:
            return False

        boundary = values.get(boundary_key)
        return boundary is not None and self._physical_z_boundary_close(
            physical_z, boundary
        )

    def supports_arb(self) -> bool:
        return all(self._term_supports_fallback(term) for term in self.terms)

    @staticmethod
    def _term_supports_fallback(term: DYCompiledTerm) -> bool:
        if term.e_surface_evaluator is None and term.e_surface is None:
            return False
        if term.integrand_evaluator is None and term.integrand_expression is None:
            return False

        theta_expressions = list(term.theta_expressions)
        theta_evaluators = list(term.theta_evaluators or [])
        for theta_index in range(max(len(theta_expressions), len(theta_evaluators))):
            expression = (
                theta_expressions[theta_index]
                if theta_index < len(theta_expressions)
                else None
            )
            evaluator = (
                theta_evaluators[theta_index]
                if theta_index < len(theta_evaluators)
                else None
            )
            if expression is None and evaluator is None:
                return False
        return True

    def supports_double_float_fallback(self) -> bool:
        # Kept as a compatibility alias. Saved Symbolica evaluators are not
        # tied to 32 digits; the same sidecars support every fallback precision.
        return self.supports_arb()

    def require_arb_supported(self) -> None:
        if self.supports_arb():
            return
        raise pygloopException(
            f"DY bundle '{self.integrand_name}' does not contain complete saved "
            "fallback evaluators. Regenerate the DY bundle."
        )

    def require_fallback_supported(self, decimal_digit_precision: int) -> None:
        if decimal_digit_precision < 1:
            raise pygloopException(
                "Higher-precision DY evaluation requires a positive precision."
            )
        if self.supports_arb():
            return
        self.require_arb_supported()

    def _runtime_evaluator_values(
        self,
        runtime_parameters: Mapping[str, Any] | None,
        m_uv: float | Decimal | None,
        *,
        decimal_digit_precision: int | None,
    ) -> dict[Expression, float | Decimal]:
        """Map public runtime values onto the symbols in evaluator ABIs."""
        overrides = {} if runtime_parameters is None else dict(runtime_parameters)
        if self._runtime_parameter_registry is None:
            # ``m_uv`` is the historical positional input and remains usable by
            # legacy/manual bundles. New named overrides require registry data.
            if overrides:
                self.resolve_runtime_parameters(overrides)
            value: float | Decimal
            if decimal_digit_precision is None:
                value = float(1.0 if m_uv is None else m_uv)
            else:
                value = self._decimal_from_number(1 if m_uv is None else m_uv)
            return {self._muv_key: value}

        if m_uv is not None:
            if "muv" in overrides:
                try:
                    named_muv = Decimal(
                        exact_decimal_string(overrides["muv"], name="muv")
                    )
                    positional_muv = Decimal(
                        exact_decimal_string(m_uv, name="muv")
                    )
                except ValueError as exc:
                    raise pygloopException(str(exc)) from exc
                if named_muv != positional_muv:
                    raise pygloopException(
                        "Conflicting DY runtime values were supplied for 'muv'."
                    )
            overrides.setdefault("muv", m_uv)
        resolved = self.resolve_runtime_parameters(
            overrides,
            decimal_digit_precision=decimal_digit_precision,
        )
        evaluator_symbols = self._runtime_parameter_registry["evaluator_symbols"]
        return {
            E(evaluator_symbols[name]): resolved[name]
            for name in self._runtime_parameter_registry[
                "evaluator_parameter_order"
            ]
        }

    def _validate_ttbar_cm_momenta(
        self,
        p1: Vector,
        p2: Vector,
        *,
        decimal_digit_precision: int | None = None,
    ) -> None:
        """Guard the CM identity assumed by reduced ttbar E-surfaces."""
        if not (self._ttbar_cm_e_surface or self._dy_cm_reduced):
            return

        if decimal_digit_precision is None:
            p1_values = tuple(float(value) for value in p1.to_list())
            p2_values = tuple(float(value) for value in p2.to_list())
            scale = max(1.0, *(abs(value) for value in (*p1_values, *p2_values)))
            tolerance = 64.0 * math.ulp(scale)
        else:
            p1_values = tuple(
                self._decimal_from_number(value) for value in p1.to_list()
            )
            p2_values = tuple(
                self._decimal_from_number(value) for value in p2.to_list()
            )
            scale = max(
                Decimal(1),
                *(abs(value) for value in (*p1_values, *p2_values)),
            )
            tolerance = scale * Decimal(10) ** (
                -max(decimal_digit_precision - 8, 12)
            )

        residual = max(
            abs(first + second)
            for first, second in zip(p1_values, p2_values, strict=True)
        )
        if residual > tolerance:
            raise pygloopException(
                "The compiled evaluator assumes partonic CM kinematics "
                f"p2=-p1, but the spatial residual is {residual!s} "
                f"(tolerance {tolerance!s})."
            )

    def _build_runtime_values(
        self,
        loop_momenta: list[Vector],
        p1: Vector,
        p2: Vector,
        z: float,
        m_uv: float | None,
        runtime_parameters: Mapping[str, Any] | None = None,
    ) -> tuple[
        dict[Expression, float], tuple[float, float, float, float, float, float]
    ]:
        self._validate_ttbar_cm_momenta(p1, p2)
        vals: dict[Expression, float] = {}
        for i, k in enumerate(loop_momenta):
            kx, ky, kz = k.to_list()
            k1, k2, k3 = self._k_keys[i]
            vals[k1] = float(kx)
            vals[k2] = float(ky)
            vals[k3] = float(kz)

        p1x, p1y, p1z = p1.to_list()
        p2x, p2y, p2z = p2.to_list()
        vals[self._p11] = float(p1x)
        vals[self._p12] = float(p1y)
        vals[self._p13] = float(p1z)
        vals[self._p21] = float(p2x)
        vals[self._p22] = float(p2y)
        vals[self._p23] = float(p2z)
        vals[self._z_key] = float(z)
        vals.update(
            self._runtime_evaluator_values(
                runtime_parameters,
                m_uv,
                decimal_digit_precision=None,
            )
        )

        return vals, (p1x, p1y, p1z, p2x, p2y, p2z)

    def _build_runtime_values_prec(
        self,
        loop_momenta: list[Vector] | tuple[Vector, ...],
        p1: Vector,
        p2: Vector,
        z: float | Decimal,
        m_uv: float | Decimal | None,
        runtime_parameters: Mapping[str, Any] | None = None,
        decimal_digit_precision: int = 80,
    ) -> tuple[
        dict[Expression, Decimal],
        tuple[Decimal, Decimal, Decimal, Decimal, Decimal, Decimal],
    ]:
        """Build fallback inputs without crossing a binary-float boundary."""
        self._validate_ttbar_cm_momenta(
            p1,
            p2,
            decimal_digit_precision=decimal_digit_precision,
        )
        vals: dict[Expression, Decimal] = {}
        for i, k in enumerate(loop_momenta):
            kx, ky, kz = k.to_list()
            k1, k2, k3 = self._k_keys[i]
            vals[k1] = self._decimal_from_number(kx)
            vals[k2] = self._decimal_from_number(ky)
            vals[k3] = self._decimal_from_number(kz)

        p1_values = tuple(self._decimal_from_number(value) for value in p1.to_list())
        p2_values = tuple(self._decimal_from_number(value) for value in p2.to_list())
        p1x, p1y, p1z = p1_values
        p2x, p2y, p2z = p2_values
        vals[self._p11] = p1x
        vals[self._p12] = p1y
        vals[self._p13] = p1z
        vals[self._p21] = p2x
        vals[self._p22] = p2y
        vals[self._p23] = p2z
        vals[self._z_key] = self._decimal_from_number(z)
        vals.update(
            self._runtime_evaluator_values(
                runtime_parameters,
                m_uv,
                decimal_digit_precision=decimal_digit_precision,
            )
        )

        return vals, (p1x, p1y, p1z, p2x, p2y, p2z)

    def _initial_t_guess(
        self,
        term: DYCompiledTerm,
        vals: dict[Expression, float],
        p1x: float,
        p1y: float,
        p1z: float,
    ) -> float:
        valst1 = vals.copy()
        valst1[self._t_key] = 1.0
        p_norm = math.sqrt(p1x**2 + p1y**2 + p1z**2)
        if p_norm == 0.0:
            return 1.0
        try:
            denom = (
                self._evaluate_float_expression(
                    term.e_surface, term.e_surface_evaluator, valst1
                )
                + 2.0 * p_norm
            )
            if denom == 0.0 or not math.isfinite(denom):
                return 1.0
            guess = abs(2.0 * p_norm / denom)
            if not math.isfinite(guess) or guess <= 0.0:
                return 1.0
            return guess
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return 1.0

    def _initial_t_guess_prec(
        self,
        term: DYCompiledTerm,
        vals: dict[Expression, Decimal],
        p1x: Decimal,
        p1y: Decimal,
        p1z: Decimal,
        decimal_digit_precision: int,
    ) -> Decimal:
        valst1 = vals.copy()
        valst1[self._t_key] = Decimal(1)
        p_norm = (p1x * p1x + p1y * p1y + p1z * p1z).sqrt()
        if p_norm.is_zero():
            return Decimal(1)
        try:
            surface = self._evaluate_expression_with_prec(
                term.e_surface,
                term.e_surface_evaluator,
                valst1,
                decimal_digit_precision,
            )
            if surface is None:
                return Decimal(1)
            denominator = surface + Decimal(2) * p_norm
            if denominator.is_zero() or not denominator.is_finite():
                return Decimal(1)
            guess = abs(Decimal(2) * p_norm / denominator)
            if not guess.is_finite() or guess <= 0:
                return Decimal(1)
            return guess
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return Decimal(1)

    #    @staticmethod
    #    def _set_inputs(pe: PygloopEvaluator, values: dict[str, float]) -> None:
    #        for head in pe.param_builder.order:
    #            raw_key = head[0].to_canonical_string()
    #            key = DYCompiledBundle._normalize_symbol_key(raw_key)
    #
    #            if key not in values:
    #                raise pygloopException(
    #                    f"Missing value for symbol '{raw_key}' (normalized '{key}') in compiled evaluator '{pe.name}'."
    #                )
    #            pe.param_builder.set_parameter_values(head, [values[key]])

    @staticmethod
    def _set_inputs(
        pe: PygloopEvaluator,
        values: dict[Expression, float],
        input_plan: list[tuple[tuple[Expression], Expression]],
    ) -> None:
        for head, value_key in input_plan:
            if value_key not in values:
                raise pygloopException(
                    f"Missing value for symbol '{value_key.to_canonical_string()}' in compiled evaluator '{pe.name}'."
                )
            pe.param_builder.set_parameter_values(head, [values[value_key]])

    @staticmethod
    def _set_inputs_fast(
        pe: PygloopEvaluator,
        values: dict[Expression, float],
        input_index_plan: list[tuple[int, Expression]],
    ) -> None:
        arr = pe.param_builder.np
        for idx, value_key in input_index_plan:
            if value_key not in values:
                raise pygloopException(
                    f"Missing value for symbol '{value_key.to_canonical_string()}' in compiled evaluator '{pe.name}'."
                )
            arr[idx] = values[value_key]

    def solve_t_newton_bisect(
        self,
        term_e_surface: Expression | None,
        term_e_surface_evaluator: Evaluator | None,
        vals: dict[Expression, float],
        t_key: Expression,
        t0: float = 1.0,
        tol_f: float = 1e-12,
        tol_x: float = 1e-12,
        max_iter: int = 32,
        max_bracket_expands: int = 64,
        eval_map: dict[Expression, float] | None = None,
        update_values_for_t: (
            Callable[[float, dict[Expression, float]], None] | None
        ) = None,
        minimum_t: float = 0.0,
    ) -> float | None:
        """Solve a positive E-surface root with a validated secant bracket."""

        if eval_map is None:
            eval_map = vals
        minimum_t = float(minimum_t)
        if not math.isfinite(minimum_t) or minimum_t < 0.0:
            raise pygloopException(
                "The minimum E-surface t must be finite and non-negative."
            )

        fallback_t_index = self._fallback_t_index
        reuse_packed_inputs = (
            term_e_surface is None
            and term_e_surface_evaluator is not None
            and eval_map is vals
            and update_values_for_t is None
            and t_key == self._t_key
            and fallback_t_index is not None
        )
        packed_inputs: list[float] | None = None

        def f(t: float) -> float | None:
            nonlocal packed_inputs
            if not math.isfinite(t) or t < minimum_t:
                return None
            eval_map[t_key] = t
            if update_values_for_t is not None:
                update_values_for_t(t, eval_map)
            if reuse_packed_inputs:
                if packed_inputs is None:
                    packed_inputs = self._fallback_input_values(eval_map)
                else:
                    packed_inputs[fallback_t_index] = t
            try:
                y = float(
                    self._evaluate_float_expression(
                        term_e_surface,
                        term_e_surface_evaluator,
                        eval_map,
                        packed_input_values=packed_inputs,
                    )
                )
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                return None
            return y if math.isfinite(y) else None

        x0 = max(minimum_t, float(t0))
        if not math.isfinite(x0):
            x0 = max(minimum_t, 1.0)
        f0 = f(x0)
        if f0 is not None and abs(f0) <= tol_f:
            return x0

        span = max(1.0, abs(x0))
        bracket: tuple[float, float, float, float] | None = None
        residual_scale = max(1.0, abs(f0) if f0 is not None else 1.0)
        for _ in range(max_bracket_expands + 1):
            a = max(minimum_t, x0 - span)
            b = max(a + max(1.0e-14, abs(a) * 1.0e-15), x0 + span)
            fa, fb = f(a), f(b)
            if fa is not None:
                residual_scale = max(residual_scale, abs(fa))
                if abs(fa) <= tol_f:
                    return a
            if fb is not None:
                residual_scale = max(residual_scale, abs(fb))
                if abs(fb) <= tol_f:
                    return b
            if fa is not None and fb is not None and fa * fb <= 0.0:
                bracket = (a, b, fa, fb)
                break
            span *= 2.0
        if bracket is None:
            return None

        a, b, fa, fb = bracket
        best_x, best_f = (a, fa) if abs(fa) <= abs(fb) else (b, fb)
        for _ in range(max(max_iter, 64)):
            if fb != fa:
                x_next = b - fb * (b - a) / (fb - fa)
            else:
                x_next = 0.5 * (a + b)
            guard = 0.05 * (b - a)
            if (
                not math.isfinite(x_next)
                or x_next <= a + guard
                or x_next >= b - guard
            ):
                x_next = 0.5 * (a + b)
            f_next = f(x_next)
            if f_next is None:
                x_next = 0.5 * (a + b)
                f_next = f(x_next)
                if f_next is None:
                    return None

            if abs(f_next) < abs(best_f):
                best_x, best_f = x_next, f_next
            if abs(f_next) <= tol_f:
                return x_next
            if fa * f_next <= 0.0:
                b, fb = x_next, f_next
            else:
                a, fa = x_next, f_next
            if abs(b - a) <= tol_x * max(1.0, abs(best_x)):
                residual_limit = max(tol_f, 1.0e-10 * residual_scale)
                return best_x if abs(best_f) <= residual_limit else None

        residual_limit = max(tol_f, 1.0e-10 * residual_scale)
        return best_x if abs(best_f) <= residual_limit else None

    def solve_t_convex_bisect(
        self,
        term_e_surface: Expression,
        vals: dict[Expression, float],
        t_key: Expression,
        t0: float = 1.0,
        tol_f: float = 1e-16,
        tol_x: float = 1e-16,
        max_iter: int = 80,
        max_expand_rounds: int = 24,
        probes_per_round: int = 17,
        eval_map: dict[Expression, float] | None = None,
    ) -> float | None:
        """
        Robust convex-friendly root finder for term_e_surface(t)=0.
        Strategy:
        1) Discover a valid sign-change bracket by sampling an expanding interval.
        2) Refine with pure bisection.
        Returns one root (prefers bracket closest to t0), or None if no bracket found.
        """
        if eval_map is None:
            eval_map = vals

        def f(t: float) -> float | None:
            if not math.isfinite(t):
                return None
            eval_map[t_key] = t
            try:
                y = _evaluate_symbolica_expression(term_e_surface, eval_map)
                return y if math.isfinite(y) else None
            except Exception:
                return None

        x0 = float(t0)
        if not math.isfinite(x0):
            x0 = 1.0

        y0 = f(x0)
        if y0 is not None and abs(y0) <= tol_f:
            return x0

        # Bracket discovery by sampled expanding intervals around x0.
        # For convex functions there may be 0/1/2 roots; we choose bracket nearest x0.
        span = max(1.0, abs(x0))
        best_bracket: tuple[float, float, float, float] | None = None
        for r in range(max_expand_rounds):
            left = x0 - span
            right = x0 + span
            step = (right - left) / float(probes_per_round - 1)

            prev_x: float | None = None
            prev_y: float | None = None
            for i in range(probes_per_round):
                x = left + step * i
                y = f(x)
                if y is None:
                    continue
                if abs(y) <= tol_f:
                    return x
                if prev_x is not None and prev_y is not None and prev_y * y <= 0.0:
                    # Pick bracket whose midpoint is closest to x0.
                    mid = 0.5 * (prev_x + x)
                    if best_bracket is None or abs(mid - x0) < abs(
                        0.5 * (best_bracket[0] + best_bracket[1]) - x0
                    ):
                        best_bracket = (prev_x, x, prev_y, y)
                prev_x, prev_y = x, y

            if best_bracket is not None:
                break
            span *= 2.0

        if best_bracket is None:
            return None

        a, b, fa, fb = best_bracket
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        # Pure bisection refinement.
        for _ in range(max_iter):
            m = 0.5 * (a + b)
            fm = f(m)
            if fm is None:
                # If midpoint fails, try quarter points before giving up.
                q1 = 0.25 * a + 0.75 * b
                fq1 = f(q1)
                if fq1 is not None:
                    m, fm = q1, fq1
                else:
                    q2 = 0.75 * a + 0.25 * b
                    fq2 = f(q2)
                    if fq2 is None:
                        return None
                    m, fm = q2, fq2

            if abs(fm) <= tol_f or abs(b - a) <= tol_x * max(1.0, abs(m)):
                return m

            if fa * fm <= 0.0:
                b, fb = m, fm
            else:
                a, fa = m, fm

        return 0.5 * (a + b)

    def solve_t_convex_bisect_prec(
        self,
        term_e_surface: Expression | None,
        term_e_surface_evaluator: Evaluator | None,
        vals: dict[Expression, Decimal],
        t_key: Expression,
        decimal_digit_precision: int,
        t0: float | Decimal = 1.0,
        max_iter: int = 80,
        max_expand_rounds: int = 24,
        probes_per_round: int = 17,
        eval_map: dict[Expression, Decimal] | None = None,
        precision_preserving: bool = True,
        update_values_for_t: (
            Callable[[Decimal, dict[Expression, Decimal]], None] | None
        ) = None,
        minimum_t: float | Decimal = Decimal(0),
    ) -> Decimal | None:
        if decimal_digit_precision <= 0:
            raise pygloopException(
                "Arbitrary-precision evaluation requires a positive decimal precision."
            )

        if eval_map is None:
            eval_map = vals

        if precision_preserving:
            # Resolve t to nearly the requested working precision. The old
            # half-precision/32-digit cap made an 80-digit Arb retry depend on a
            # much less accurate root and defeated the purpose of upcasting.
            tol_power = max(decimal_digit_precision - 8, 12)
            effective_max_iter = max(max_iter, 4 * tol_power)
            evaluate_expression = self._evaluate_expression_with_prec
            convert_number = self._decimal_from_number
        else:
            tol_power = min(max(decimal_digit_precision // 2, 12), 32)
            effective_max_iter = max_iter
            evaluate_expression = self._evaluate_expression_with_prec_legacy
            convert_number = self._legacy_decimal_from_number
        tol_f = Decimal(10) ** (-tol_power)
        tol_x = Decimal(10) ** (-tol_power)
        minimum = convert_number(minimum_t)
        if not minimum.is_finite() or minimum < 0:
            raise pygloopException(
                "The minimum precise E-surface t must be finite and non-negative."
            )

        fallback_t_index = self._fallback_t_index
        reuse_prepared_inputs = (
            precision_preserving
            and term_e_surface is None
            and term_e_surface_evaluator is not None
            and eval_map is vals
            and update_values_for_t is None
            and t_key == self._t_key
            and fallback_t_index is not None
        )
        prepared_inputs: list[Decimal] | None = None

        def f(t: Decimal) -> Decimal | None:
            nonlocal prepared_inputs
            if t.is_nan() or t < minimum:
                return None
            eval_map[t_key] = t
            if update_values_for_t is not None:
                update_values_for_t(t, eval_map)
            if reuse_prepared_inputs:
                if prepared_inputs is None:
                    prepared_inputs = [
                        _prepare_symbolica_input_with_prec(
                            self._decimal_from_number(value),
                            decimal_digit_precision,
                        )
                        for value in self._fallback_input_values(eval_map)
                    ]
                else:
                    prepared_inputs[fallback_t_index] = (
                        _prepare_symbolica_input_with_prec(
                            self._decimal_from_number(t),
                            decimal_digit_precision,
                        )
                    )
                return self._evaluate_expression_with_prec(
                    term_e_surface,
                    term_e_surface_evaluator,
                    eval_map,
                    decimal_digit_precision,
                    prepared_input_values=prepared_inputs,
                )
            return evaluate_expression(
                term_e_surface,
                term_e_surface_evaluator,
                eval_map,
                decimal_digit_precision,
            )

        try:
            x0 = convert_number(t0)
        except (InvalidOperation, ValueError, pygloopException):
            x0 = Decimal(1)
        if not x0.is_finite():
            x0 = Decimal(1)
        if x0 < minimum:
            x0 = minimum

        y0 = f(x0)
        if y0 is not None and abs(y0) <= tol_f:
            return x0

        one = Decimal(1)
        best_bracket: tuple[Decimal, Decimal, Decimal, Decimal] | None = None
        span = max(one, abs(x0))
        for _ in range(max_expand_rounds):
            left = max(minimum, x0 - span)
            right = x0 + span
            if right <= left:
                right = left + one
            step = (right - left) / Decimal(probes_per_round - 1)

            prev_x: Decimal | None = None
            prev_y: Decimal | None = None
            for i in range(probes_per_round):
                x = left + step * Decimal(i)
                y = f(x)
                if y is None:
                    continue
                if abs(y) <= tol_f:
                    return x
                if prev_x is not None and prev_y is not None and prev_y * y <= 0:
                    mid = (prev_x + x) / 2
                    if best_bracket is None or abs(mid - x0) < abs(
                        (best_bracket[0] + best_bracket[1]) / 2 - x0
                    ):
                        best_bracket = (prev_x, x, prev_y, y)
                prev_x, prev_y = x, y

            if best_bracket is not None:
                break
            span *= 2

        if best_bracket is None:
            return None

        a, b, fa, fb = best_bracket
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        for _ in range(effective_max_iter):
            m = (a + b) / 2
            fm = f(m)
            if fm is None:
                q1 = (a + 3 * b) / 4
                fq1 = f(q1)
                if fq1 is not None:
                    m, fm = q1, fq1
                else:
                    q2 = (3 * a + b) / 4
                    fq2 = f(q2)
                    if fq2 is None:
                        return None
                    m, fm = q2, fq2

            if abs(fm) <= tol_f or abs(b - a) <= tol_x * max(one, abs(m)):
                return m

            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm

        return (a + b) / 2

    def _evaluate_arb_terms_legacy(
        self,
        loop_momenta: list[Vector] | tuple[Vector, ...],
        p1: Vector,
        p2: Vector,
        z: float | Decimal,
        m_uv: float | Decimal | None,
        runtime_parameters: Mapping[str, Any] | None,
        decimal_digit_precision: int,
        theta_tolerance: float | Decimal,
        channel_selector: int | None,
        ttbar_pt_min: float | None,
        integrated_uv_ct_filter: str | None,
        physical_z_min: float | Decimal | None,
        physical_z_max: float | Decimal | None,
        return_support: bool = False,
    ) -> (
        tuple[Decimal, list[tuple[str, Decimal]]]
        | tuple[
            Decimal,
            list[tuple[str, Decimal]],
            list[tuple[str, bool]],
        ]
    ):
        """Keep reference diagnostics on their historical numerical contract."""
        vals, (p1x, p1y, p1z, _p2x, _p2y, _p2z) = self._build_runtime_values(
            loop_momenta, p1, p2, z, m_uv, runtime_parameters
        )
        # Keep the legacy conversion contract for sampled kinematics, but never
        # round registry parameters through binary64 on a fallback path.
        vals.update(
            self._runtime_evaluator_values(
                runtime_parameters,
                m_uv,
                decimal_digit_precision=decimal_digit_precision,
            )
        )
        dec_vals = {
            key: self._legacy_decimal_from_number(value)
            for key, value in vals.items()
        }
        total = Decimal(0)
        term_values: list[tuple[str, Decimal]] = []
        term_support: list[tuple[str, bool]] = []
        theta_tol = self._legacy_decimal_from_number(theta_tolerance)
        z_min = (
            self._legacy_decimal_from_number(physical_z_min)
            if physical_z_min is not None
            else None
        )
        z_max = (
            self._legacy_decimal_from_number(physical_z_max)
            if physical_z_max is not None
            else None
        )

        for term in self.terms_for_channel(
            channel_selector, integrated_uv_ct_filter
        ):
            my_t0 = self._initial_t_guess(term, vals, p1x, p1y, p1z)
            t_sol = self.solve_t_convex_bisect_prec(
                term.e_surface,
                term.e_surface_evaluator,
                dec_vals,
                self._t_key,
                decimal_digit_precision=decimal_digit_precision,
                t0=my_t0,
                eval_map=dec_vals,
                precision_preserving=False,
            )
            if t_sol is None:
                raise pygloopException(
                    "Failed to solve t in arbitrary precision for DY term "
                    f"'{term.evaluator_name}'."
                )

            dec_vals[self._t_key] = t_sol
            physical_z = t_sol * t_sol * dec_vals[self._z_key]
            if not self._physical_z_cut_passes(
                physical_z, z_min, z_max
            ):
                term_values.append((term.evaluator_name, Decimal(0)))
                term_support.append((term.evaluator_name, False))
                continue
            if not self._ttbar_pt_cut_passes_with_prec_legacy(
                term,
                dec_vals,
                ttbar_pt_min,
                decimal_digit_precision,
            ):
                term_values.append((term.evaluator_name, Decimal(0)))
                term_support.append((term.evaluator_name, False))
                continue

            theta_passes = True
            theta_expressions = list(term.theta_expressions)
            theta_evaluators = list(term.theta_evaluators or [])
            theta_count = max(len(theta_expressions), len(theta_evaluators))
            theta_expressions.extend(
                [None] * (theta_count - len(theta_expressions))
            )
            theta_evaluators.extend(
                [None] * (theta_count - len(theta_evaluators))
            )
            for theta_index, (theta_expression, theta_evaluator) in enumerate(
                zip(theta_expressions, theta_evaluators)
            ):
                theta_value = self._evaluate_expression_with_prec_legacy(
                    theta_expression,
                    theta_evaluator,
                    dec_vals,
                    decimal_digit_precision,
                )
                if not self._theta_cut_passes(
                    term,
                    theta_index,
                    theta_value,
                    theta_tol,
                    physical_z,
                    dec_vals,
                ):
                    theta_passes = False
                    break
            if not theta_passes:
                term_values.append((term.evaluator_name, Decimal(0)))
                term_support.append((term.evaluator_name, False))
                continue

            term_value = self._evaluate_expression_with_prec_legacy(
                term.integrand_expression,
                term.integrand_evaluator,
                dec_vals,
                decimal_digit_precision,
                term.integrand_evaluator_parameter_order,
            )
            if term_value is None:
                raise pygloopException(
                    f"Failed to evaluate DY term '{term.evaluator_name}' "
                    "in arbitrary precision."
                )
            term_value = self._apply_automatic_graph_factor(term, term_value)
            total += term_value
            term_values.append((term.evaluator_name, term_value))
            term_support.append((term.evaluator_name, True))

        if return_support:
            return total, term_values, term_support
        return total, term_values

    def evaluate_arb(
        self,
        loop_momenta: list[Vector] | tuple[Vector, ...],
        p1: Vector,
        p2: Vector,
        z: float | Decimal,
        m_uv: float | Decimal | None = None,
        decimal_digit_precision: int = 80,
        theta_tolerance: float | Decimal = 0.0,
        channel_selector: int | None = None,
        ttbar_pt_min: float | None = None,
        integrated_uv_ct_filter: str | None = "all",
        physical_z_min: float | Decimal | None = None,
        physical_z_max: float | Decimal | None = None,
        runtime_parameters: Mapping[str, Any] | None = None,
    ) -> Decimal:
        total, _term_values = self.evaluate_arb_terms(
            loop_momenta,
            p1,
            p2,
            z,
            m_uv,
            decimal_digit_precision=decimal_digit_precision,
            theta_tolerance=theta_tolerance,
            channel_selector=channel_selector,
            ttbar_pt_min=ttbar_pt_min,
            integrated_uv_ct_filter=integrated_uv_ct_filter,
            physical_z_min=physical_z_min,
            physical_z_max=physical_z_max,
            precision_preserving=True,
            runtime_parameters=runtime_parameters,
        )
        return total

    def evaluate_arb_terms(
        self,
        loop_momenta: list[Vector] | tuple[Vector, ...],
        p1: Vector,
        p2: Vector,
        z: float | Decimal,
        m_uv: float | Decimal | None = None,
        decimal_digit_precision: int = 80,
        theta_tolerance: float | Decimal = 0.0,
        channel_selector: int | None = None,
        ttbar_pt_min: float | None = None,
        integrated_uv_ct_filter: str | None = "all",
        precision_preserving: bool = False,
        physical_z_min: float | Decimal | None = None,
        physical_z_max: float | Decimal | None = None,
        return_support: bool = False,
        runtime_parameters: Mapping[str, Any] | None = None,
        term_observer: Any = None,
    ) -> (
        tuple[Decimal, list[tuple[str, Decimal]]]
        | tuple[
            Decimal,
            list[tuple[str, Decimal]],
            list[tuple[str, bool]],
        ]
    ):
        """Evaluate terms and optionally return their exact cut/theta support.

        The optional support rows are derived from the same per-term precise
        ``t`` solution as the value.  They therefore distinguish a supported
        term whose numerator cancels to zero from a term rejected by a
        physical or theta predicate.
        """
        self.require_fallback_supported(decimal_digit_precision)
        if decimal_digit_precision <= 0:
            raise pygloopException(
                "Higher-precision DY evaluation requires a positive precision."
            )

        if term_observer is not None and not precision_preserving:
            raise ValueError("Cut observers require precision-preserving HP evaluation")
        if not precision_preserving:
            return self._evaluate_arb_terms_legacy(
                loop_momenta,
                p1,
                p2,
                z,
                m_uv,
                runtime_parameters,
                decimal_digit_precision,
                theta_tolerance,
                channel_selector,
                ttbar_pt_min,
                integrated_uv_ct_filter,
                physical_z_min,
                physical_z_max,
                return_support,
            )

        # Decimal arithmetic otherwise inherits the process-global default of 28
        # digits, silently truncating 32/80-digit evaluator outputs while summing
        # terms or solving t. Keep guard digits around the complete fallback.
        with localcontext() as context:
            context.prec = decimal_digit_precision + 12
            dec_vals, (p1x, p1y, p1z, _p2x, _p2y, _p2z) = (
                self._build_runtime_values_prec(
                    loop_momenta,
                    p1,
                    p2,
                    z,
                    m_uv,
                    runtime_parameters,
                    decimal_digit_precision,
                )
            )
            total = Decimal(0)
            term_values: list[tuple[str, Decimal]] = []
            term_support: list[tuple[str, bool]] = []
            theta_tol = self._decimal_from_number(theta_tolerance)
            z_min = (
                self._decimal_from_number(physical_z_min)
                if physical_z_min is not None
                else None
            )
            z_max = (
                self._decimal_from_number(physical_z_max)
                if physical_z_max is not None
                else None
            )

            for term in self.terms_for_channel(
                channel_selector, integrated_uv_ct_filter
            ):
                my_t0 = self._initial_t_guess_prec(
                    term,
                    dec_vals,
                    p1x,
                    p1y,
                    p1z,
                    decimal_digit_precision,
                )
                t_sol = self.solve_t_convex_bisect_prec(
                    term.e_surface,
                    term.e_surface_evaluator,
                    dec_vals,
                    self._t_key,
                    decimal_digit_precision=decimal_digit_precision,
                    t0=my_t0,
                    eval_map=dec_vals,
                )
                if t_sol is None:
                    raise DYNumericalEvaluationError(
                        "Failed to solve t in higher precision for DY term "
                        f"'{term.evaluator_name}'."
                    )

                dec_vals[self._t_key] = t_sol
                physical_z = t_sol * t_sol * dec_vals[self._z_key]
                if not self._physical_z_cut_passes(
                    physical_z, z_min, z_max
                ):
                    term_values.append((term.evaluator_name, Decimal(0)))
                    term_support.append((term.evaluator_name, False))
                    continue
                if not self._ttbar_pt_cut_passes_with_prec(
                    term,
                    dec_vals,
                    ttbar_pt_min,
                    decimal_digit_precision,
                ):
                    term_values.append((term.evaluator_name, Decimal(0)))
                    term_support.append((term.evaluator_name, False))
                    continue

                theta_passes = True
                theta_expressions = list(term.theta_expressions)
                theta_evaluators = list(term.theta_evaluators or [])
                theta_count = max(len(theta_expressions), len(theta_evaluators))
                theta_expressions.extend(
                    [None] * (theta_count - len(theta_expressions))
                )
                theta_evaluators.extend(
                    [None] * (theta_count - len(theta_evaluators))
                )
                for theta_index, (th, th_evaluator) in enumerate(
                    zip(theta_expressions, theta_evaluators)
                ):
                    th_val = self._evaluate_expression_with_prec(
                        th,
                        th_evaluator,
                        dec_vals,
                        decimal_digit_precision,
                    )
                    if not self._theta_cut_passes(
                        term,
                        theta_index,
                        th_val,
                        theta_tol,
                        physical_z,
                        dec_vals,
                    ):
                        theta_passes = False
                        break
                if not theta_passes:
                    term_values.append((term.evaluator_name, Decimal(0)))
                    term_support.append((term.evaluator_name, False))
                    continue

                term_value = self._evaluate_expression_with_prec(
                    term.integrand_expression,
                    term.integrand_evaluator,
                    dec_vals,
                    decimal_digit_precision,
                    term.integrand_evaluator_parameter_order,
                )
                if term_value is None:
                    raise DYNumericalEvaluationError(
                        f"Failed to evaluate DY term '{term.evaluator_name}' "
                        "in higher precision."
                    )
                term_value = self._apply_automatic_graph_factor(term, term_value)
                total += term_value
                if term_observer is not None:
                    term_observer(physical_z, term_value)
                term_values.append((term.evaluator_name, term_value))
                term_support.append((term.evaluator_name, True))

            precise_total = +total
            precise_terms = [
                (term_name, +term_value) for term_name, term_value in term_values
            ]

        if return_support:
            return precise_total, precise_terms, term_support
        return precise_total, precise_terms

    @staticmethod
    def _coerce_saved_real_outputs(
        outputs: Any, expected_rows: int
    ) -> list[float]:
        array = np.asarray(outputs)
        if array.size != expected_rows:
            raise pygloopException(
                "Saved DY evaluator returned "
                f"{array.size} values for {expected_rows} input rows."
            )
        flat = array.reshape(-1)
        if np.iscomplexobj(flat):
            if np.any(flat.imag != 0.0):
                raise pygloopException(
                    "A saved real DY evaluator returned a complex value."
                )
            flat = flat.real
        return [float(value) for value in flat]

    def _evaluate_saved_real_rows(
        self,
        evaluator: Evaluator,
        rows: Sequence[Sequence[float]],
        *,
        errors_as_none: bool = False,
    ) -> list[float | None]:
        if len(rows) == 0:
            return []
        try:
            return self._coerce_saved_real_outputs(
                evaluator.evaluate(np.asarray(rows, dtype=float)), len(rows)
            )
        except BaseException as batch_exc:
            if isinstance(batch_exc, (KeyboardInterrupt, SystemExit)):
                raise

        values: list[float | None] = []
        for row in rows:
            try:
                value = self._single_evaluator_output(evaluator.evaluate(list(row)))
                if isinstance(value, complex):
                    if value.imag != 0.0:
                        raise pygloopException(
                            "A saved real DY evaluator returned a complex value."
                        )
                    value = value.real
                values.append(float(value))
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                if not errors_as_none:
                    raise
                values.append(None)
        return values

    def _evaluate_float_rows(
        self,
        expression: Expression | None,
        evaluator: Evaluator | None,
        values: Sequence[dict[Expression, float]],
        *,
        errors_as_none: bool = False,
    ) -> list[float | None]:
        if expression is not None:
            outputs: list[float | None] = []
            for row in values:
                try:
                    outputs.append(float(_evaluate_symbolica_expression(expression, row)))
                except BaseException as exc:
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    if not errors_as_none:
                        raise
                    outputs.append(None)
            return outputs
        if evaluator is None:
            raise pygloopException(
                "No float or DoubleFloat evaluator data is present in this DY bundle. "
                "Regenerate the bundle with --dy-fallback-precision 32."
            )
        packed = [
            [float(row[value_key]) for value_key in self._fallback_value_keys]
            for row in values
        ]
        return self._evaluate_saved_real_rows(
            evaluator, packed, errors_as_none=errors_as_none
        )

    def _evaluate_surface_rows(
        self,
        term: DYCompiledTerm,
        values: Sequence[dict[Expression, float]],
        ts: Sequence[float],
    ) -> list[float | None]:
        if len(values) != len(ts):
            raise pygloopException("DY E-surface batch inputs no longer align.")
        for row, t_value in zip(values, ts, strict=True):
            row[self._t_key] = t_value
        outputs = self._evaluate_float_rows(
            term.e_surface,
            term.e_surface_evaluator,
            values,
            errors_as_none=True,
        )
        return [
            value if value is not None and math.isfinite(value) else None
            for value in outputs
        ]

    def _initial_t_guesses_batch(
        self,
        term: DYCompiledTerm,
        values: Sequence[dict[Expression, float]],
        p1_values: Sequence[tuple[float, float, float]],
    ) -> list[float]:
        surfaces = self._evaluate_surface_rows(
            term, values, [1.0] * len(values)
        )
        guesses: list[float] = []
        for surface, (p1x, p1y, p1z) in zip(
            surfaces, p1_values, strict=True
        ):
            p_norm = math.sqrt(p1x**2 + p1y**2 + p1z**2)
            if p_norm == 0.0 or surface is None:
                guesses.append(1.0)
                continue
            denominator = surface + 2.0 * p_norm
            if denominator == 0.0 or not math.isfinite(denominator):
                guesses.append(1.0)
                continue
            guess = abs(2.0 * p_norm / denominator)
            guesses.append(
                guess if math.isfinite(guess) and guess > 0.0 else 1.0
            )
        return guesses

    def _solve_t_newton_bisect_batch(
        self,
        term: DYCompiledTerm,
        values: Sequence[dict[Expression, float]],
        t0s: Sequence[float],
        *,
        tol_f: float = 1.0e-12,
        tol_x: float = 1.0e-12,
        max_iter: int = 32,
        max_bracket_expands: int = 64,
    ) -> list[float | None]:
        """Lockstep form of ``solve_t_newton_bisect`` for independent rows."""
        if len(values) != len(t0s):
            raise pygloopException("DY root-solver batch inputs no longer align.")
        count = len(values)
        if count == 0:
            return []

        fallback_t_index = self._fallback_t_index
        reuse_packed_inputs = (
            term.e_surface is None
            and term.e_surface_evaluator is not None
            and fallback_t_index is not None
        )
        packed_inputs: np.ndarray | None = None
        if reuse_packed_inputs:
            packed_inputs = np.asarray(
                [
                    [
                        float(t0)
                        if value_key == self._t_key
                        else float(row[value_key])
                        for value_key in self._fallback_value_keys
                    ]
                    for row, t0 in zip(values, t0s, strict=True)
                ],
                dtype=float,
            )

        def evaluate_surfaces(
            indices: Sequence[int], ts: Sequence[float]
        ) -> list[float | None]:
            if len(indices) != len(ts):
                raise pygloopException("DY E-surface batch inputs no longer align.")
            selected_values = [values[index] for index in indices]
            for row, t_value in zip(selected_values, ts, strict=True):
                row[self._t_key] = t_value
            if packed_inputs is None:
                return self._evaluate_surface_rows(term, selected_values, ts)
            if not indices:
                return []
            if len(indices) == count and all(
                index == position for position, index in enumerate(indices)
            ):
                selected_inputs = packed_inputs
            else:
                selected_inputs = packed_inputs[np.asarray(indices, dtype=np.intp)]
            selected_inputs[:, fallback_t_index] = ts
            assert term.e_surface_evaluator is not None
            outputs = self._evaluate_saved_real_rows(
                term.e_surface_evaluator,
                selected_inputs,
                errors_as_none=True,
            )
            return [
                value if value is not None and math.isfinite(value) else None
                for value in outputs
            ]

        roots: list[float | None] = [None] * count
        x0 = [max(0.0, float(value)) for value in t0s]
        x0 = [value if math.isfinite(value) else 1.0 for value in x0]
        f0 = evaluate_surfaces(range(count), x0)
        unresolved: set[int] = set()
        residual_scale = [
            max(1.0, abs(value) if value is not None else 1.0) for value in f0
        ]
        spans = [max(1.0, abs(value)) for value in x0]
        brackets: list[tuple[float, float, float, float] | None] = [None] * count
        for index, value in enumerate(f0):
            if value is not None and abs(value) <= tol_f:
                roots[index] = x0[index]
            else:
                unresolved.add(index)

        for _ in range(max_bracket_expands + 1):
            if not unresolved:
                break
            indices = sorted(unresolved)
            left = [max(0.0, x0[index] - spans[index]) for index in indices]
            right = [
                max(
                    a + max(1.0e-14, abs(a) * 1.0e-15),
                    x0[index] + spans[index],
                )
                for index, a in zip(indices, left, strict=True)
            ]
            fa_values = evaluate_surfaces(indices, left)
            fb_values = evaluate_surfaces(indices, right)
            for index, a, b, fa, fb in zip(
                indices, left, right, fa_values, fb_values, strict=True
            ):
                if fa is not None:
                    residual_scale[index] = max(residual_scale[index], abs(fa))
                    if abs(fa) <= tol_f:
                        roots[index] = a
                        unresolved.remove(index)
                        continue
                if fb is not None:
                    residual_scale[index] = max(residual_scale[index], abs(fb))
                    if abs(fb) <= tol_f:
                        roots[index] = b
                        unresolved.remove(index)
                        continue
                if fa is not None and fb is not None and fa * fb <= 0.0:
                    brackets[index] = (a, b, fa, fb)
                    unresolved.remove(index)
                else:
                    spans[index] *= 2.0

        active = {index for index, bracket in enumerate(brackets) if bracket is not None}
        best_x = [0.0] * count
        best_f = [math.inf] * count
        for index in active:
            a, b, fa, fb = brackets[index]  # type: ignore[misc]
            if abs(fa) <= abs(fb):
                best_x[index], best_f[index] = a, fa
            else:
                best_x[index], best_f[index] = b, fb

        for _ in range(max(max_iter, 64)):
            if not active:
                break
            indices = sorted(active)
            next_x: list[float] = []
            for index in indices:
                a, b, fa, fb = brackets[index]  # type: ignore[misc]
                candidate = (
                    b - fb * (b - a) / (fb - fa)
                    if fb != fa
                    else 0.5 * (a + b)
                )
                guard = 0.05 * (b - a)
                if (
                    not math.isfinite(candidate)
                    or candidate <= a + guard
                    or candidate >= b - guard
                ):
                    candidate = 0.5 * (a + b)
                next_x.append(candidate)
            next_f = evaluate_surfaces(indices, next_x)

            failed_positions = [
                position for position, value in enumerate(next_f) if value is None
            ]
            if failed_positions:
                midpoint_x = []
                midpoint_indices = []
                for position in failed_positions:
                    index = indices[position]
                    a, b, _fa, _fb = brackets[index]  # type: ignore[misc]
                    midpoint_x.append(0.5 * (a + b))
                    midpoint_indices.append(index)
                midpoint_f = evaluate_surfaces(midpoint_indices, midpoint_x)
                for position, candidate, value in zip(
                    failed_positions, midpoint_x, midpoint_f, strict=True
                ):
                    next_x[position] = candidate
                    next_f[position] = value

            for position, index in enumerate(indices):
                candidate = next_x[position]
                value = next_f[position]
                if value is None:
                    active.remove(index)
                    continue
                a, b, fa, fb = brackets[index]  # type: ignore[misc]
                if abs(value) < abs(best_f[index]):
                    best_x[index], best_f[index] = candidate, value
                if abs(value) <= tol_f:
                    roots[index] = candidate
                    active.remove(index)
                    continue
                if fa * value <= 0.0:
                    b, fb = candidate, value
                else:
                    a, fa = candidate, value
                brackets[index] = (a, b, fa, fb)
                if abs(b - a) <= tol_x * max(1.0, abs(best_x[index])):
                    residual_limit = max(tol_f, 1.0e-10 * residual_scale[index])
                    roots[index] = (
                        best_x[index]
                        if abs(best_f[index]) <= residual_limit
                        else None
                    )
                    active.remove(index)

        for index in active:
            residual_limit = max(tol_f, 1.0e-10 * residual_scale[index])
            roots[index] = (
                best_x[index] if abs(best_f[index]) <= residual_limit else None
            )
        return roots

    def _batch_request_root_signature(
        self,
        request: DYCompiledBatchRequest,
        values: Mapping[Expression, float],
    ) -> tuple[float, ...]:
        kinematics = tuple(
            float(component)
            for vector in (*request.loop_momenta, request.p1, request.p2)
            for component in vector.to_list()
        ) + (float(request.z),)
        # mUV is the one runtime input known not to enter an E-surface.  All
        # other resolved parameters are part of the signature so an explicit
        # share key cannot accidentally reuse a root across, for example,
        # different top masses.
        root_runtime = tuple(
            float(value)
            for key, value in values.items()
            if key not in {self._muv_key, self._t_key}
        )
        return kinematics + root_runtime

    def _compiled_integrand_rows(
        self,
        term: DYCompiledTerm,
        values: Sequence[dict[Expression, float]],
    ) -> list[complex]:
        pe = self.evaluators[term.evaluator_name]
        inputs = np.tile(pe.param_builder.np, (len(values), 1))
        for index, value_key in self._input_index_plans[term.evaluator_name]:
            inputs[:, index] = [row[value_key] for row in values]
        if pe.complexified:
            pairwise = np.zeros((len(values), inputs.shape[1] * 2), dtype=float)
            pairwise[:, 0::2] = inputs.real
            pairwise[:, 1::2] = inputs.imag
            inputs = pairwise
        outputs = pe.get_compiled_evaluator().evaluate(inputs)
        converted = pe._pairwise_to_complex(outputs, pe.output_length)
        converted = np.asarray(converted).reshape(len(values), pe.output_length)
        return [complex(row[0]) for row in converted]

    def _saved_jit_integrand_rows(
        self,
        term: DYCompiledTerm,
        values: Sequence[dict[Expression, float]],
    ) -> list[complex]:
        evaluator = term.integrand_evaluator
        value_keys = self._integrand_value_keys[term.evaluator_name]
        if evaluator is None or value_keys is None:
            return self._compiled_integrand_rows(term, values)
        packed = [
            [float(row[value_key]) for value_key in value_keys] for row in values
        ]
        outputs = self._evaluate_saved_real_rows(evaluator, packed)
        return [complex(float(value), 0.0) for value in outputs]

    def evaluate_batch(
        self,
        requests: Sequence[DYCompiledBatchRequest],
        *,
        integrand_backend: str = "compiled",
    ) -> list[complex]:
        totals, _diagnostics = self.evaluate_batch_with_diagnostics(
            requests, integrand_backend=integrand_backend
        )
        return totals

    def evaluate_batch_with_diagnostics(
        self,
        requests: Sequence[DYCompiledBatchRequest],
        *,
        integrand_backend: str = "compiled",
    ) -> tuple[list[complex], list[DYCompiledEvaluationDiagnostics]]:
        """Evaluate channel-grouped rows with batched roots and integrands.

        The legacy scalar API remains unchanged.  ``saved-jit`` selects the
        bundle's saved real Symbolica evaluators; it intentionally uses the
        standard JIT path and never enables direct translation.
        """
        if integrand_backend not in {"compiled", "saved-jit"}:
            raise pygloopException(
                f"Unsupported DY batch integrand backend '{integrand_backend}'."
            )
        request_rows = list(requests)
        if not request_rows:
            return [], []

        runtime_rows = [
            self._build_runtime_values(
                list(request.loop_momenta),
                request.p1,
                request.p2,
                request.z,
                request.m_uv,
                request.runtime_parameters,
            )
            for request in request_rows
        ]
        values = [row[0] for row in runtime_rows]
        external_values = [row[1] for row in runtime_rows]
        totals = [0.0 + 0.0j for _ in request_rows]
        diagnostics = [
            DYCompiledEvaluationDiagnostics() for _request in request_rows
        ]
        grouped: dict[tuple[int | None, str | None], list[int]] = {}
        for index, request in enumerate(request_rows):
            grouped.setdefault(
                (request.channel_selector, request.integrated_uv_ct_filter), []
            ).append(index)

        for (channel_selector, integrated_filter), indices in grouped.items():
            terms = self.terms_for_channel(channel_selector, integrated_filter)
            surface_roots: dict[int, list[float | None]] = {}
            share_representatives: list[int] = []
            aliases: dict[int, list[int]] = {}
            seen_share_keys: dict[Hashable, int] = {}
            signatures: dict[Hashable, tuple[float, ...]] = {}
            for local_index, request_index in enumerate(indices):
                request = request_rows[request_index]
                share_key = request.root_share_key
                if share_key is None:
                    share_representatives.append(local_index)
                    aliases[local_index] = [local_index]
                    continue
                signature = self._batch_request_root_signature(
                    request, values[request_index]
                )
                if share_key in seen_share_keys:
                    representative = seen_share_keys[share_key]
                    if signatures[share_key] != signature:
                        raise pygloopException(
                            "DY rows with the same root_share_key do not have "
                            "identical kinematics."
                        )
                    aliases[representative].append(local_index)
                else:
                    seen_share_keys[share_key] = local_index
                    signatures[share_key] = signature
                    share_representatives.append(local_index)
                    aliases[local_index] = [local_index]

            representative_values = [
                values[indices[local_index]] for local_index in share_representatives
            ]
            representative_p1 = [
                external_values[indices[local_index]][:3]
                for local_index in share_representatives
            ]

            for term in terms:
                surface_group = self._e_surface_group_ids.get(term.evaluator_name)
                roots = (
                    surface_roots.get(surface_group)
                    if surface_group is not None
                    else None
                )
                if roots is None:
                    guesses = self._initial_t_guesses_batch(
                        term, representative_values, representative_p1
                    )
                    representative_roots = self._solve_t_newton_bisect_batch(
                        term, representative_values, guesses
                    )
                    roots = [None] * len(indices)
                    for representative_position, representative in enumerate(
                        share_representatives
                    ):
                        for local_index in aliases[representative]:
                            roots[local_index] = representative_roots[
                                representative_position
                            ]
                    if surface_group is not None:
                        surface_roots[surface_group] = roots

                active_local: list[int] = []
                for local_index, root in enumerate(roots):
                    if root is None:
                        diagnostics[indices[local_index]].record_t_solver_failure(
                            term, surface_group
                        )
                        continue
                    request_index = indices[local_index]
                    request = request_rows[request_index]
                    row = values[request_index]
                    row[self._t_key] = root
                    if not self._physical_z_cut_passes(
                        root * root * row[self._z_key],
                        request.physical_z_min,
                        request.physical_z_max,
                    ):
                        continue
                    if not self._ttbar_pt_cut_passes(
                        term, row, request.ttbar_pt_min
                    ):
                        continue
                    active_local.append(local_index)

                theta_expressions = list(term.theta_expressions)
                theta_evaluators = list(term.theta_evaluators or [])
                theta_count = max(len(theta_expressions), len(theta_evaluators))
                theta_expressions.extend(
                    [None] * (theta_count - len(theta_expressions))
                )
                theta_evaluators.extend(
                    [None] * (theta_count - len(theta_evaluators))
                )
                for theta_index, (theta_expression, theta_evaluator) in enumerate(
                    zip(theta_expressions, theta_evaluators, strict=True)
                ):
                    if not active_local:
                        break
                    theta_rows = [values[indices[index]] for index in active_local]
                    theta_values = self._evaluate_float_rows(
                        theta_expression, theta_evaluator, theta_rows
                    )
                    active_local = [
                        local_index
                        for local_index, theta_value in zip(
                            active_local, theta_values, strict=True
                        )
                        if self._theta_cut_passes(
                            term,
                            theta_index,
                            theta_value,
                            float(
                                request_rows[indices[local_index]].theta_tolerance
                            ),
                            values[indices[local_index]][self._t_key]
                            * values[indices[local_index]][self._t_key]
                            * values[indices[local_index]][self._z_key],
                            values[indices[local_index]],
                        )
                    ]
                if not active_local:
                    continue

                integrand_rows = [values[indices[index]] for index in active_local]
                term_values = (
                    self._compiled_integrand_rows(term, integrand_rows)
                    if integrand_backend == "compiled"
                    else self._saved_jit_integrand_rows(term, integrand_rows)
                )
                for local_index, term_value in zip(
                    active_local, term_values, strict=True
                ):
                    request_index = indices[local_index]
                    term_value = self._apply_automatic_graph_factor(term, term_value)
                    totals[request_index] += term_value
                    observer = request_rows[request_index].term_observer
                    if observer is not None:
                        row = values[request_index]
                        observer(row[self._t_key]**2 * row[self._z_key], term_value)
        return totals, diagnostics

    def prewarm_batch_backend(
        self,
        requests: Sequence[DYCompiledBatchRequest],
        *,
        integrand_backend: str = "saved-jit",
    ) -> None:
        """Warm a batch backend once inside the current worker process."""
        self.evaluate_batch(requests, integrand_backend=integrand_backend)

    def evaluate(
        self,
        loop_momenta: list[Vector],
        p1: Vector,
        p2: Vector,
        z: float,
        m_uv: float | Decimal | None = None,
        mode: str = "compiled",
        decimal_digit_precision: int | None = None,
        theta_tolerance: float = 0.0,
        channel_selector: int | None = None,
        ttbar_pt_min: float | None = None,
        integrated_uv_ct_filter: str | None = "all",
        physical_z_min: float | None = None,
        physical_z_max: float | None = None,
        runtime_parameters: Mapping[str, Any] | None = None,
        raise_on_t_solver_failure: bool = False,
    ) -> complex:
        if mode == "arb":
            if decimal_digit_precision is None:
                decimal_digit_precision = 80
            return complex(
                float(
                    self.evaluate_arb(
                        loop_momenta,
                        p1,
                        p2,
                        z,
                        m_uv,
                        decimal_digit_precision=decimal_digit_precision,
                        theta_tolerance=theta_tolerance,
                        channel_selector=channel_selector,
                        ttbar_pt_min=ttbar_pt_min,
                        integrated_uv_ct_filter=integrated_uv_ct_filter,
                        physical_z_min=physical_z_min,
                        physical_z_max=physical_z_max,
                        runtime_parameters=runtime_parameters,
                    )
                ),
                0.0,
            )
        if mode != "compiled":
            raise pygloopException(f"Unsupported DY bundle evaluation mode '{mode}'.")

        vals, (p1x, p1y, p1z, _p2x, _p2y, _p2z) = self._build_runtime_values(
            loop_momenta, p1, p2, z, m_uv, runtime_parameters
        )

        total = 0.0 + 0.0j
        theta_tol = float(theta_tolerance)
        # Deliberately call-local: scale branches, rotations, and soft mirrors
        # enter through separate evaluate calls and must never share roots.
        root_cache: dict[tuple[int, float], float | None] = {}

        # Sum over all cut graphs
        for term in self.terms_for_channel(channel_selector, integrated_uv_ct_filter):
            my_t0 = self._initial_t_guess(term, vals, p1x, p1y, p1z)
            surface_group_id = self._e_surface_group_ids.get(term.evaluator_name)
            root_cache_key = (
                (surface_group_id, my_t0)
                if surface_group_id is not None
                else None
            )
            root_was_cached = (
                root_cache_key is not None and root_cache_key in root_cache
            )
            t_sol = root_cache[root_cache_key] if root_was_cached else None
            if not root_was_cached:
                t_sol = self.solve_t_newton_bisect(
                    term.e_surface,
                    term.e_surface_evaluator,
                    vals,
                    self._t_key,
                    t0=my_t0,  # fixed per-term start for benchmark-stable branch
                    eval_map=vals,
                )
                if root_cache_key is not None:
                    root_cache[root_cache_key] = t_sol

            # t_sol = self.solve_t_convex_bisect(
            #     term.e_surface,
            #     vals,
            #     self._t_key,
            #     t0=my_t0,  # fixed per-term start for benchmark-stable branch
            #     eval_map=vals,
            # )

            if t_sol is None:
                if raise_on_t_solver_failure:
                    diagnostics = DYCompiledEvaluationDiagnostics()
                    diagnostics.record_t_solver_failure(term, surface_group_id)
                    raise DYDoubleRootFailure(diagnostics)
                continue

            vals[self._t_key] = t_sol
            physical_z = t_sol * t_sol * vals[self._z_key]
            if not self._physical_z_cut_passes(
                physical_z,
                physical_z_min,
                physical_z_max,
            ):
                continue
            if not self._ttbar_pt_cut_passes(term, vals, ttbar_pt_min):
                continue

            # print("---------------")
            theta = 1
            theta_expressions = list(term.theta_expressions)
            theta_evaluators = list(term.theta_evaluators or [])
            theta_count = max(len(theta_expressions), len(theta_evaluators))
            theta_expressions.extend([None] * (theta_count - len(theta_expressions)))
            theta_evaluators.extend([None] * (theta_count - len(theta_evaluators)))
            for theta_index, (th, th_evaluator) in enumerate(
                zip(theta_expressions, theta_evaluators)
            ):
                th_val = self._evaluate_float_expression(th, th_evaluator, vals)
                # if th_val > 0:
                #    print("-->", 1)
                # else:
                #    print("-->", 0)

                if not self._theta_cut_passes(
                    term,
                    theta_index,
                    th_val,
                    theta_tol,
                    physical_z,
                    vals,
                ):
                    theta = 0
                    break
            if theta == 0:
                continue

            pe = self.evaluators[term.evaluator_name]

            self._set_inputs_fast(
                pe, vals, self._input_index_plans[term.evaluator_name]
            )

            term_value = complex(pe.evaluate(eager=False)[0])
            total += self._apply_automatic_graph_factor(term, term_value)

        return total


class compile_integrands:
    def __init__(
        self,
        L,
        process,
        name,
        observable,
        evaluators,
        fallback_precision: int = 80,
        bundle_metadata: Mapping[str, Any] | None = None,
    ):
        self.L = L
        self.process = process
        self.observable = observable
        self.evaluators = evaluators
        self.name = name
        self.fallback_precision = int(fallback_precision)
        self.bundle_metadata = (
            deepcopy(dict(bundle_metadata))
            if bundle_metadata is not None
            else None
        )

    def save_compiled_integrand(self):

        DYCompiledBundle.create_from_evaluators(
            process=self.process,
            integrand_name=self.name,
            n_loops=self.L,
            observable=self.observable,
            evaluators=self.evaluators,
            fallback_precision=self.fallback_precision,
            bundle_metadata=self.bundle_metadata,
        )
