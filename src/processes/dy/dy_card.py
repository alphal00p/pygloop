"""Strict, versioned settings cards for DY generation and integration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from processes.dy.dy_runtime_parameters import (
    PARAMETER_ORDER as RUNTIME_PARAMETER_ORDER,
    DYRuntimeParameterError,
    exact_decimal_string,
)
from processes.dy.dy_precision import DEFAULT_PRECISION_LADDER, validate_precision_ladder
from processes.dy.dy_top_self_energy import (
    TOP_SELF_ENERGY_RENORMALISATION_MODES,
    projected_os_schema_for_mode,
    resolve_top_self_energy_renormalisation,
)


SCHEMA_VERSION = 1
SUPPORTED_COMMANDS = frozenset({"generate", "integrate"})
SUPPORTED_CHANNELS = frozenset({(0, 0), (0, 1), (1, 0), (1, -1), (-1, 1)})
PHYSICAL_CHANNEL_LABELS = frozenset({"qqbar", "qg", "gg"})


class DYCardError(ValueError):
    """Raised when a DY card is malformed or internally inconsistent."""


@dataclass(frozen=True)
class _Field:
    path: str
    destination: str
    kind: str
    choices: frozenset[Any] | None = None


def _field(
    path: str,
    destination: str,
    kind: str,
    choices: Iterable[Any] | None = None,
) -> _Field:
    return _Field(
        path,
        destination,
        kind,
        frozenset(choices) if choices is not None else None,
    )


_FIELDS = (
    _field("common.process", "process", "str", ["dy"]),
    _field("common.process_name", "dy_process_name", "str"),
    _field("common.n_loops", "n_loops", "int", [1, 2]),
    _field("common.final_state", "dy_final_state", "str_list"),
    _field("common.channel", "dy_channel", "int_list"),
    _field("common.diagrams", "diagrams", "str_list"),
    _field("common.process_basename", "overwrite_process_basename", "str"),
    _field(
        "common.verbosity", "verbosity", "str", ["debug", "info", "critical"]
    ),
    _field("common.skip_ps_validation", "dy_skip_ps_validation", "bool"),
    _field("common.kinematics.m_top", "m_top", "float"),
    _field("common.kinematics.m_higgs", "m_higgs", "float"),
    _field("common.kinematics.pg1", "pg1", "float_list"),
    _field("common.kinematics.pg2", "pg2", "float_list"),
    _field("common.kinematics.ph1", "ph1", "float_list"),
    _field("common.kinematics.ph2", "ph2", "float_list"),
    _field("common.kinematics.ph3", "ph3", "float_list"),
    _field("common.kinematics.helicities", "helicities", "int_list"),
    _field("common.scales.muv", "dy_muv", "float"),
    _field("common.scales.lambda_sq", "dy_lambda_sq", "float"),
    _field("common.scales.mur_sq", "dy_mur_sq", "float"),
    _field(
        "common.initial_state.external_gluon_polarisation",
        "external_gluon_polarisation",
        "bool",
    ),
    _field(
        "common.initial_state.symmetrise_p1_p2", "dy_symmetrise_p1_p2", "bool"
    ),
    _field(
        "common.precision.fallback_precision", "dy_fallback_precision", "int"
    ),
    _field(
        "common.gammaloop.generation_configuration",
        "gammaloop_configuration",
        "path",
    ),
    _field(
        "common.gammaloop.runtime_configuration", "runtime_configuration", "path"
    ),
    _field("common.gammaloop.settings", "gammaloop_settings", "str_list"),
    _field("common.gammaloop.general_settings", "general_settings", "str_list"),
    _field("generate.clean", "clean", "bool"),
    _field(
        "generate.generation_type",
        "generation_type",
        "str",
        ["gammaloop", "spenso", "all"],
    ),
    _field("generate.parallel_graphs", "dy_parallel_graphs", "int"),
    _field(
        "generate.enable_integrated_uv_counterterms",
        "dy_enable_integrated_uv_cts",
        "bool",
    ),
    _field(
        "generate.top_self_energy_os_subtraction",
        "dy_top_self_energy_os_subtraction",
        "bool",
    ),
    _field(
        "generate.top_self_energy_renormalisation",
        "dy_top_self_energy_renormalisation",
        "str",
        TOP_SELF_ENERGY_RENORMALISATION_MODES,
    ),
    _field(
        "generate.check_generation_limits", "dy_check_generation_limits", "bool"
    ),
    _field(
        "generate.threshold_h_function",
        "dy_threshold_h_function",
        "str",
        ["gaussian", "inverse_square_damped"],
    ),
    _field(
        "generate.spenso.evaluator_compiler",
        "integrand_evaluator_compiler",
        "str",
        ["symbolica_only", "symjit"],
    ),
    _field(
        "generate.spenso.full_integrand_strategy",
        "full_spenso_integrand_strategy",
        "str",
        ["merging", "summing", "function_map"],
    ),
    _field(
        "generate.spenso.horner_iterations", "n_iterations_hornerscheme", "int"
    ),
    _field("generate.spenso.cpe_iterations", "n_iterations_cpe", "int"),
    _field(
        "integrate.implementation",
        "integrand_implementation",
        "str",
        ["gammaloop", "zenos", "spenso_parametric", "spenso_summed"],
    ),
    _field(
        "integrate.parameterisation",
        "parameterisation",
        "str",
        ["cartesian", "spherical", "log_spherical"],
    ),
    _field("integrate.multi_channeling", "multi_channeling", "bool"),
    _field(
        "integrate.physical_channel_multi_channeling",
        "physical_channel_multi_channeling",
        "bool",
    ),
    _field(
        "integrate.physical_channel_cards",
        "dy_physical_channel_cards",
        "path_mapping",
    ),
    _field(
        "integrate.physical_channel_backend",
        "dy_physical_channel_backend",
        "str",
        ["compiled", "saved-jit"],
    ),
    _field(
        "integrate.physical_channel_batch_size",
        "dy_physical_channel_batch_size",
        "int",
    ),
    _field(
        "integrate.physical_channel_warmup_iterations",
        "dy_physical_channel_warmup_iterations",
        "int",
    ),
    _field(
        "integrate.physical_channel_min_fraction",
        "dy_physical_channel_min_fraction",
        "float",
    ),
    _field(
        "integrate.physical_channel_target_relative_precision",
        "dy_physical_channel_target_relative_precision",
        "float",
    ),
    _field(
        "integrate.physical_channel_live_diagnostics",
        "dy_physical_channel_live_diagnostics",
        "bool",
    ),
    _field(
        "integrate.integrator",
        "integrator",
        "str",
        ["naive", "symbolica", "vegas", "gammaloop", "dy-paired-sobol"],
    ),
    _field("integrate.paired_sobol.replicas", "dy_paired_replicas", "int"),
    _field("integrate.paired_sobol.learning_points", "dy_paired_learning_points", "int_list"),
    _field("integrate.paired_sobol.production_points", "dy_paired_production_points", "int"),
    _field("integrate.paired_sobol.max_points", "dy_paired_max_points", "int"),
    _field("integrate.paired_sobol.audit_block_size", "dy_paired_audit_block_size", "int"),
    _field("integrate.paired_sobol.target_relative_error", "dy_paired_target_relative_error", "float"),
    _field("integrate.paired_sobol.auxiliary_error_fraction", "dy_paired_auxiliary_error_fraction", "float"),
    _field("integrate.paired_sobol.output_directory", "dy_paired_output_directory", "path"),
    _field("integrate.paired_sobol.z_histogram_edges", "dy_paired_z_histogram_edges", "float_list"),
    _field("integrate.iterations", "n_iterations", "int"),
    _field("integrate.points_per_iteration", "points_per_iteration", "int"),
    _field("integrate.cores", "n_cores", "int"),
    _field("integrate.seed", "seed", "int"),
    _field("integrate.phase", "phase", "str", ["real", "imag"]),
    _field("integrate.restart", "restart", "bool"),
    _field("integrate.workspace_name", "run_workspace_name", "str"),
    _field("integrate.target", "target", "target"),
    _field(
        "integrate.runtime_parameters",
        "dy_runtime_parameters",
        "decimal_mapping",
    ),
    _field(
        "integrate.selection.integrated_uv_counterterms",
        "dy_integrated_uv_ct_filter",
        "str",
        ["all", "only", "exclude"],
    ),
    _field("integrate.selection.graphs", "dy_integration_graphs", "str_list"),
    _field(
        "integrate.selection.graph_weights", "dy_graph_weights", "float_list"
    ),
    _field("integrate.selection.ttbar_pt_min", "dy_ttbar_pt_min", "float"),
    _field(
        "integrate.stability.rotation_check_digits",
        "dy_rotation_check_digits",
        "int",
    ),
    _field(
        "integrate.stability.rotation_check_count",
        "dy_rotation_check_count",
        "int",
    ),
    _field(
        "integrate.stability.rotation_check_eps", "dy_rotation_check_eps", "float"
    ),
    _field(
        "integrate.stability.rotation_check_arb_digits",
        "dy_rotation_check_arb_digits",
        "int",
    ),
    _field(
        "integrate.stability.backend",
        "dy_stability_backend",
        "str",
        ["double-float", "arb"],
    ),
    _field("integrate.stability.precision_ladder", "dy_precision_ladder", "int_list"),
    _field(
        "integrate.stability.arb_digits", "dy_stability_arb_digits", "int"
    ),
    _field(
        "integrate.stability.relative_tolerance", "dy_stability_rtol", "float"
    ),
    _field(
        "integrate.stability.absolute_tolerance", "dy_stability_atol", "float"
    ),
    _field("integrate.stability.theta_tolerance", "dy_theta_tol", "float"),
    _field(
        "integrate.stability.large_weight_precision",
        "dy_large_weight_precision",
        "float",
    ),
    _field(
        "integrate.stability.large_weight_clip", "dy_large_weight_clip", "float"
    ),
    _field(
        "integrate.stability.soft_mirror_edges", "dy_soft_mirror_edge", "str_list"
    ),
    _field(
        "integrate.stability.grouped_soft_pair",
        "dy_grouped_soft_pair",
        "str_list",
    ),
    _field(
        "integrate.stability.grouped_soft_pair_transform",
        "dy_grouped_soft_pair_transform",
        "str",
        ["invert"],
    ),
    _field("integrate.beams.enabled", "dy_integrate_beams", "bool"),
    _field(
        "integrate.beams.parameterisation",
        "dy_beam_parameterisation",
        "str",
        ["x1_x2", "beta_y"],
    ),
    _field("integrate.beams.z_bin", "dy_z_bin", "float_list"),
    _field("integrate.beams.q_bins", "dy_q_bins", "float_list"),
    _field("integrate.beams.q_min", "dy_q_min", "float"),
    _field("integrate.beams.q_max", "dy_q_max", "float"),
    _field(
        "integrate.beams.physical_normalisation",
        "dy_physical_normalisation",
        "bool",
    ),
    _field(
        "integrate.beams.integrated_leptonic_phase_space",
        "dy_integrated_leptonic_phase_space",
        "bool",
    ),
    _field("integrate.beams.pdf.set", "dy_pdf_set", "str"),
    _field("integrate.beams.pdf.member", "dy_pdf_member", "int"),
    _field("integrate.beams.pdf.muf_sq", "dy_muf_sq", "float"),
    _field(
        "integrate.beams.scheme_counterterm.enabled",
        "dy_msbar_scheme_counterterm",
        "bool",
    ),
    _field(
        "integrate.beams.scheme_counterterm.sobol_power",
        "dy_scheme_counterterm_sobol_power",
        "int",
    ),
    _field(
        "integrate.beams.scheme_counterterm.replicas",
        "dy_scheme_counterterm_replicas",
        "int",
    ),
    _field(
        "integrate.beams.scheme_counterterm.factor",
        "dy_scheme_counterterm_factor",
        "float",
    ),
    _field(
        "integrate.beams.scheme_counterterm.born_bundle",
        "dy_scheme_born_bundle",
        "str",
    ),
    _field(
        "integrate.beams.scheme_counterterm.born_bundles",
        "dy_scheme_born_bundles",
        "str_list",
    ),
    _field(
        "integrate.beams.scheme_counterterm.alpha_s",
        "dy_scheme_alpha_s",
        "float",
    ),
    _field(
        "integrate.beams.scheme_counterterm.clip",
        "dy_scheme_counterterm_clip",
        "float",
    ),
    _field("integrate.decoupling.enabled", "dy_decoupling", "bool"),
)

FIELD_BY_PATH = MappingProxyType({field.path: field for field in _FIELDS})
CARD_TO_DESTINATION = MappingProxyType(
    {field.path: field.destination for field in _FIELDS}
)
_INLINE_TABLE_PATHS = frozenset(
    {
        "integrate.target",
        "integrate.runtime_parameters",
        "integrate.physical_channel_cards",
    }
)
_KNOWN_TABLES = frozenset(
    prefix
    for field in _FIELDS
    for prefix in (
        ".".join(field.path.split(".")[:index])
        for index in range(1, len(field.path.split(".")))
    )
)


_BASE_DEFAULTS: dict[str, Any] = {
    "common.process": "dy",
    "common.process_name": "DY",
    "common.n_loops": 1,
    "common.final_state": ["a"],
    "common.channel": [1, -1],
    "common.diagrams": None,
    "common.process_basename": "DY",
    "common.verbosity": "info",
    "common.skip_ps_validation": False,
    "common.kinematics.m_top": 173.0,
    "common.kinematics.m_higgs": 125.0,
    "common.kinematics.pg1": [500.0, 0.0, 0.0, 500.0],
    "common.kinematics.pg2": [500.0, 0.0, 0.0, -500.0],
    "common.kinematics.ph1": [
        438.5555662246945,
        155.3322001835378,
        348.0160396513587,
        -177.3773615718412,
    ],
    "common.kinematics.ph2": [
        356.3696374921922,
        -16.802389008511,
        -318.7291102436005,
        97.48719163688098,
    ],
    "common.kinematics.ph3": [
        205.0747962831133,
        -138.5298111750267,
        -29.28692940775817,
        79.8901699349603,
    ],
    "common.kinematics.helicities": [1, 1, 0, 0, 0],
    "common.initial_state.external_gluon_polarisation": False,
    "common.initial_state.symmetrise_p1_p2": False,
    "common.precision.fallback_precision": 80,
    "common.gammaloop.settings": [],
    "common.gammaloop.general_settings": [],
    "generate.clean": False,
    "generate.generation_type": "all",
    "generate.parallel_graphs": 1,
    "generate.enable_integrated_uv_counterterms": False,
    "generate.top_self_energy_os_subtraction": False,
    "generate.top_self_energy_renormalisation": "no-os",
    "generate.check_generation_limits": False,
    "generate.threshold_h_function": None,
    "generate.spenso.evaluator_compiler": "symbolica_only",
    "generate.spenso.full_integrand_strategy": None,
    "generate.spenso.horner_iterations": 100,
    "generate.spenso.cpe_iterations": None,
    "integrate.implementation": "gammaloop",
    "integrate.parameterisation": "spherical",
    "integrate.multi_channeling": False,
    "integrate.physical_channel_multi_channeling": False,
    "integrate.physical_channel_cards": None,
    "integrate.physical_channel_backend": "saved-jit",
    "integrate.physical_channel_batch_size": 256,
    "integrate.physical_channel_warmup_iterations": 0,
    "integrate.physical_channel_min_fraction": 0.0,
    "integrate.physical_channel_target_relative_precision": None,
    "integrate.physical_channel_live_diagnostics": False,
    "integrate.integrator": "gammaloop",
    "integrate.paired_sobol.replicas": 16,
    "integrate.paired_sobol.learning_points": [1024,2048,4096],
    "integrate.paired_sobol.production_points": 131072,
    "integrate.paired_sobol.max_points": 1048576,
    "integrate.paired_sobol.audit_block_size": 512,
    "integrate.paired_sobol.target_relative_error": 0.01,
    "integrate.paired_sobol.auxiliary_error_fraction": 0.1,
    "integrate.paired_sobol.output_directory": None,
    "integrate.paired_sobol.z_histogram_edges": [],
    "integrate.iterations": 10,
    "integrate.points_per_iteration": 1000,
    "integrate.cores": 1,
    "integrate.seed": 1337,
    "integrate.phase": "real",
    "integrate.restart": False,
    "integrate.workspace_name": None,
    "integrate.target": None,
    "integrate.runtime_parameters": {},
    "integrate.selection.integrated_uv_counterterms": "all",
    "integrate.selection.graphs": None,
    "integrate.selection.graph_weights": None,
    "integrate.selection.ttbar_pt_min": None,
    "integrate.stability.rotation_check_digits": None,
    "integrate.stability.rotation_check_count": 1,
    "integrate.stability.rotation_check_eps": 1.0e-15,
    "integrate.stability.rotation_check_arb_digits": 80,
    "integrate.stability.backend": None,
    "integrate.stability.precision_ladder": None,
    "integrate.stability.arb_digits": None,
    "integrate.stability.relative_tolerance": None,
    "integrate.stability.absolute_tolerance": None,
    "integrate.stability.theta_tolerance": 0.0,
    "integrate.stability.large_weight_precision": None,
    "integrate.stability.large_weight_clip": None,
    "integrate.stability.soft_mirror_edges": None,
    "integrate.stability.grouped_soft_pair": None,
    "integrate.stability.grouped_soft_pair_transform": None,
    "integrate.beams.enabled": False,
    "integrate.beams.parameterisation": "x1_x2",
    "integrate.beams.z_bin": None,
    "integrate.beams.q_bins": None,
    "integrate.beams.q_min": None,
    "integrate.beams.q_max": None,
    "integrate.beams.physical_normalisation": False,
    "integrate.beams.integrated_leptonic_phase_space": False,
    "integrate.beams.pdf.set": None,
    "integrate.beams.pdf.member": 0,
    "integrate.beams.pdf.muf_sq": None,
    "integrate.beams.scheme_counterterm.enabled": False,
    "integrate.beams.scheme_counterterm.sobol_power": 15,
    "integrate.beams.scheme_counterterm.replicas": 8,
    "integrate.beams.scheme_counterterm.factor": None,
    "integrate.beams.scheme_counterterm.born_bundle": None,
    "integrate.beams.scheme_counterterm.born_bundles": None,
    "integrate.beams.scheme_counterterm.alpha_s": 0.118,
    "integrate.beams.scheme_counterterm.clip": None,
    "integrate.decoupling.enabled": False,
}

_OPERATIONAL_GENERATION_PATHS = frozenset(
    {
        "common.verbosity",
        "common.skip_ps_validation",
        "generate.clean",
        "generate.parallel_graphs",
        "generate.check_generation_limits",
    }
)
_RUNTIME_DEFAULT_GENERATION_PATHS = frozenset(
    {
        "common.kinematics.m_top",
        "common.kinematics.m_higgs",
        "common.kinematics.pg1",
        "common.kinematics.pg2",
        "common.kinematics.ph1",
        "common.kinematics.ph2",
        "common.kinematics.ph3",
        "common.scales.muv",
        "common.scales.lambda_sq",
        "common.scales.mur_sq",
    }
)
_CONFIGURATION_PATHS = (
    "common.gammaloop.generation_configuration",
    "common.gammaloop.runtime_configuration",
)
_OPTIONAL_MOMENTA = frozenset(
    {
        "common.kinematics.ph1",
        "common.kinematics.ph2",
        "common.kinematics.ph3",
    }
)
_TOP_SELF_ENERGY_MODE_PATH = "generate.top_self_energy_renormalisation"
_TOP_SELF_ENERGY_LEGACY_PATH = "generate.top_self_energy_os_subtraction"
_PROJECTED_OS_SCHEMA_FINGERPRINT_PATH = (
    "generate.top_self_energy_projected_schema_version"
)


def _clone(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clone(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_value(field: _Field, value: Any) -> Any:
    kind = field.kind
    if kind == "bool":
        if type(value) is not bool:
            raise DYCardError(f"{field.path} must be a boolean")
        result = value
    elif kind == "int":
        if type(value) is not int:
            raise DYCardError(f"{field.path} must be an integer")
        result = value
    elif kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DYCardError(f"{field.path} must be a number")
        result = float(value)
        if not math.isfinite(result):
            raise DYCardError(f"{field.path} must be finite")
    elif kind in {"str", "path"}:
        if not isinstance(value, str):
            raise DYCardError(f"{field.path} must be a string")
        result = value
    elif kind in {"str_list", "int_list", "float_list"}:
        if not isinstance(value, (list, tuple)):
            raise DYCardError(f"{field.path} must be an array")
        result = []
        item_kind = kind.removesuffix("_list")
        for index, item in enumerate(value):
            item_field = _Field(f"{field.path}[{index}]", field.destination, item_kind)
            result.append(_normalise_value(item_field, item))
    elif kind == "target":
        if isinstance(value, complex):
            result = {"real": float(value.real), "imag": float(value.imag)}
        elif isinstance(value, Mapping):
            unknown = set(value).difference({"real", "imag"})
            if unknown:
                raise DYCardError(
                    f"{field.path} has unknown keys: {', '.join(sorted(unknown))}"
                )
            result = {
                "real": _normalise_value(
                    _Field(f"{field.path}.real", field.destination, "float"),
                    value.get("real", 0.0),
                ),
                "imag": _normalise_value(
                    _Field(f"{field.path}.imag", field.destination, "float"),
                    value.get("imag", 0.0),
                ),
            }
        else:
            raise DYCardError(f"{field.path} must be an inline {{real, imag}} table")
    elif kind == "decimal_mapping":
        if not isinstance(value, Mapping):
            raise DYCardError(f"{field.path} must be a TOML table")
        unknown = sorted(set(value).difference(RUNTIME_PARAMETER_ORDER))
        if unknown:
            raise DYCardError(
                f"{field.path} has unknown parameters: {', '.join(unknown)}"
            )
        result = {}
        for name, item in value.items():
            try:
                result[name] = exact_decimal_string(item, name=name)
            except DYRuntimeParameterError as exc:
                raise DYCardError(str(exc)) from exc
    elif kind == "path_mapping":
        if not isinstance(value, Mapping):
            raise DYCardError(f"{field.path} must be a TOML table")
        result = {}
        for raw_label, item in value.items():
            label = str(raw_label).strip().lower()
            if label not in PHYSICAL_CHANNEL_LABELS:
                known = ", ".join(sorted(PHYSICAL_CHANNEL_LABELS))
                raise DYCardError(
                    f"{field.path} has unsupported channel {raw_label!r}; "
                    f"expected one of {known}"
                )
            if label in result:
                raise DYCardError(
                    f"{field.path} contains duplicate channel {label!r}"
                )
            if not isinstance(item, str) or not item.strip():
                raise DYCardError(
                    f"{field.path}.{label} must be a nonempty card path"
                )
            result[label] = item
    else:  # pragma: no cover - internal schema invariant
        raise AssertionError(f"Unknown DY card field kind {kind!r}")

    if field.choices is not None and result not in field.choices:
        choices = ", ".join(repr(choice) for choice in sorted(field.choices))
        raise DYCardError(f"{field.path} must be one of {choices}; got {result!r}")
    return result


def _flatten_card_table(
    table: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in table.items():
        if key == "command":
            location = f"{prefix}.command" if prefix else "command"
            raise DYCardError(f"{location} is forbidden; select a CLI subcommand")
        path = f"{prefix}.{key}" if prefix else key
        if path == "schema_version":
            flattened[path] = value
            continue
        if path in _INLINE_TABLE_PATHS:
            flattened[path] = value
            continue
        if isinstance(value, Mapping):
            if path not in _KNOWN_TABLES:
                raise DYCardError(f"Unknown DY card table [{path}]")
            flattened.update(_flatten_card_table(value, prefix=path))
        else:
            if path not in FIELD_BY_PATH:
                raise DYCardError(f"Unknown DY card key {path}")
            flattened[path] = value
    return flattened


def _dynamic_defaults(values: Mapping[str, Any] | None = None) -> dict[str, Any]:
    defaults = {path: _clone(value) for path, value in _BASE_DEFAULTS.items()}
    if values:
        defaults.update({path: _clone(value) for path, value in values.items()})
    loops = int(defaults.get("common.n_loops", 1))
    if loops == 1:
        defaults.setdefault("common.scales.muv", 1.0)
        defaults.setdefault("common.scales.lambda_sq", 2.0)
        defaults.setdefault("common.scales.mur_sq", 1.0)
    else:
        defaults.setdefault("common.scales.muv", 2000.0)
        defaults.setdefault("common.scales.lambda_sq", 50000.0)
        defaults.setdefault("common.scales.mur_sq", 50000.0)
    repo_root = Path(__file__).resolve().parents[3]
    defaults.setdefault(
        "common.gammaloop.generation_configuration",
        str((repo_root / "configs" / "DY" / "generate.toml").resolve()),
    )
    defaults.setdefault(
        "common.gammaloop.runtime_configuration",
        str((repo_root / "configs" / "DY" / "runtime.toml").resolve()),
    )
    return defaults


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _require_positive(values: Mapping[str, Any], *paths: str) -> None:
    for path in paths:
        value = values.get(path)
        if value is not None and value <= 0:
            raise DYCardError(f"{path} must be positive")


def _minkowski_square(momentum: list[float]) -> float:
    return momentum[0] ** 2 - sum(component**2 for component in momentum[1:])


def _validate_phase_space(values: Mapping[str, Any]) -> None:
    """Mirror the existing DY phase-space sanity checks without constructing DY."""
    incoming = [
        values["common.kinematics.pg1"],
        values["common.kinematics.pg2"],
    ]
    outgoing = [
        values["common.kinematics.ph1"],
        values["common.kinematics.ph2"],
        values["common.kinematics.ph3"],
    ]
    incoming_sum = [sum(momentum[i] for momentum in incoming) for i in range(4)]
    s_hat = _minkowski_square(incoming_sum)
    if s_hat <= 0:
        raise DYCardError("The DY phase-space point must have positive s")
    sqrt_s = math.sqrt(s_hat)
    tolerance = 1.0e-10
    for momentum in incoming:
        if math.sqrt(abs(_minkowski_square(momentum))) / sqrt_s > tolerance:
            raise DYCardError("Incoming DY momenta must be massless")
    expected_mass = values["common.kinematics.m_higgs"]
    for momentum in outgoing:
        mass = math.sqrt(abs(_minkowski_square(momentum)))
        if abs(mass - expected_mass) / sqrt_s > tolerance:
            raise DYCardError("Outgoing DY momenta must be on shell")
    residual = [
        incoming_sum[i] - sum(momentum[i] for momentum in outgoing)
        for i in range(4)
    ]
    if any(abs(component) / sqrt_s > tolerance for component in residual):
        raise DYCardError("The DY phase-space point violates momentum conservation")


def _validate_values(values: dict[str, Any], supplied: set[str]) -> None:
    for path, value in tuple(values.items()):
        field = FIELD_BY_PATH.get(path)
        if field is not None and value is not None:
            values[path] = _normalise_value(field, value)

    mode_path = "generate.top_self_energy_renormalisation"
    legacy_path = "generate.top_self_energy_os_subtraction"
    try:
        top_self_energy_mode = resolve_top_self_energy_renormalisation(
            values.get(mode_path) if mode_path in supplied else None,
            values.get(legacy_path) if legacy_path in supplied else None,
            legacy_is_explicit=legacy_path in supplied,
        )
    except ValueError as exc:
        raise DYCardError(str(exc)) from exc
    values[mode_path] = top_self_energy_mode

    if values["common.process"] != "dy":
        raise DYCardError("common.process must be 'dy' for --dy-card")
    if not _is_nonempty_string(values["common.process_name"]):
        raise DYCardError("common.process_name must be nonempty")
    process_name = values["common.process_name"].lower()
    if process_name not in {"dy", "tt~"}:
        raise DYCardError("common.process_name must be 'dy' or 'tt~'")
    loops = values["common.n_loops"]
    if loops not in {1, 2}:
        raise DYCardError("common.n_loops must be 1 or 2 for DY")
    if (
        values.get("integrate.runtime_parameters")
        and values.get("integrate.implementation") != "zenos"
    ):
        raise DYCardError(
            "integrate.runtime_parameters requires integrate.implementation = 'zenos'"
        )
    if loops == 2 and process_name != "tt~":
        raise DYCardError("Two-loop DY cards currently require process_name = 'tt~'")

    final_state = values["common.final_state"]
    if not final_state or any(not _is_nonempty_string(item) for item in final_state):
        raise DYCardError("common.final_state must contain nonempty particle names")
    channel = values["common.channel"]
    if len(channel) != 2 or tuple(channel) not in SUPPORTED_CHANNELS:
        raise DYCardError(
            "common.channel must be one of [0,0], [0,1], [1,0], [1,-1], or [-1,1]"
        )
    diagrams = values.get("common.diagrams")
    if diagrams is not None:
        if any(not _is_nonempty_string(item) for item in diagrams):
            raise DYCardError("common.diagrams entries must be nonempty strings")
        if len(set(diagrams)) != len(diagrams):
            raise DYCardError("common.diagrams entries must be unique")
    if not _is_nonempty_string(values["common.process_basename"]):
        raise DYCardError("common.process_basename must be nonempty")

    for path in (
        "common.kinematics.pg1",
        "common.kinematics.pg2",
        "common.kinematics.ph1",
        "common.kinematics.ph2",
        "common.kinematics.ph3",
    ):
        if len(values[path]) != 4:
            raise DYCardError(f"{path} must contain exactly four finite entries")
    if len(values["common.kinematics.helicities"]) != 5:
        raise DYCardError("common.kinematics.helicities must contain five integers")
    for path in ("common.kinematics.m_top", "common.kinematics.m_higgs"):
        if values[path] < 0:
            raise DYCardError(f"{path} must be nonnegative")
    _require_positive(
        values,
        "common.scales.muv",
        "common.scales.lambda_sq",
        "common.scales.mur_sq",
    )
    if values["common.precision.fallback_precision"] < 2:
        raise DYCardError("common.precision.fallback_precision must be at least 2")
    for path in _CONFIGURATION_PATHS:
        config_path = Path(values[path])
        if not config_path.is_file():
            raise DYCardError(f"{path} is not a regular file: {config_path}")
    if not values["common.skip_ps_validation"]:
        _validate_phase_space(values)

    if values["generate.parallel_graphs"] < 1:
        raise DYCardError("generate.parallel_graphs must be at least 1")
    if values["generate.spenso.horner_iterations"] < 1:
        raise DYCardError("generate.spenso.horner_iterations must be positive")
    cpe_iterations = values.get("generate.spenso.cpe_iterations")
    if cpe_iterations is not None and cpe_iterations < 1:
        raise DYCardError("generate.spenso.cpe_iterations must be positive")
    supplied_spenso = any(path.startswith("generate.spenso.") for path in supplied)
    if supplied_spenso and values["generate.generation_type"] == "gammaloop":
        raise DYCardError(
            "generate.spenso settings require generation_type = 'spenso' or 'all'"
        )
    if top_self_energy_mode != "no-os":
        setting = "generate.top_self_energy_renormalisation"
        if mode_path not in supplied and legacy_path in supplied:
            setting = legacy_path
        if process_name != "tt~":
            raise DYCardError(
                f"{setting} requires "
                "common.process_name = 'tt~'"
            )
        if loops != 2:
            raise DYCardError(
                f"{setting} requires common.n_loops = 2"
            )
        if not values["generate.enable_integrated_uv_counterterms"]:
            raise DYCardError(
                f"{setting} requires "
                "generate.enable_integrated_uv_counterterms = true"
            )
        if tuple(channel) not in {(1, -1), (-1, 1), (0, 0)}:
            raise DYCardError(
                f"{setting} supports only "
                "common.channel = [1,-1], [-1,1], or [0,0]"
            )
    if (
        loops == 2
        and process_name == "tt~"
        and (
            tuple(channel) == (0, 0)
            or values["integrate.beams.enabled"]
        )
        and not values["common.initial_state.symmetrise_p1_p2"]
    ):
        raise DYCardError(
            "Two-loop ttbar generation and hadronic integration require "
            "common.initial_state.symmetrise_p1_p2 = true for every incoming "
            "channel"
        )

    for path in (
        "integrate.iterations",
        "integrate.points_per_iteration",
        "integrate.cores",
    ):
        if values[path] < 1:
            raise DYCardError(f"{path} must be positive")
    cpu_count = os.cpu_count() or 1
    if values["integrate.cores"] > cpu_count:
        raise DYCardError(
            f"integrate.cores ({values['integrate.cores']}) exceeds available CPUs ({cpu_count})"
        )
    physical_multi_channeling = values[
        "integrate.physical_channel_multi_channeling"
    ]
    physical_channel_cards = values.get("integrate.physical_channel_cards")
    if values["integrate.physical_channel_batch_size"] < 1:
        raise DYCardError(
            "integrate.physical_channel_batch_size must be positive"
        )
    if values["integrate.physical_channel_warmup_iterations"] < 0:
        raise DYCardError(
            "integrate.physical_channel_warmup_iterations must be nonnegative"
        )
    physical_channel_min_fraction = values[
        "integrate.physical_channel_min_fraction"
    ]
    physical_channel_target_relative_precision = values[
        "integrate.physical_channel_target_relative_precision"
    ]
    physical_channel_live_diagnostics = values[
        "integrate.physical_channel_live_diagnostics"
    ]
    if (
        not math.isfinite(physical_channel_min_fraction)
        or physical_channel_min_fraction < 0.0
    ):
        raise DYCardError(
            "integrate.physical_channel_min_fraction must be finite and "
            "nonnegative"
        )
    if (
        physical_channel_target_relative_precision is not None
        and physical_channel_target_relative_precision <= 0.0
    ):
        raise DYCardError(
            "integrate.physical_channel_target_relative_precision must be "
            "positive when supplied"
        )
    if physical_multi_channeling:
        if values["integrate.integrator"] not in {"symbolica", "dy-paired-sobol"}:
            raise DYCardError(
                "integrate.physical_channel_multi_channeling requires "
                "integrate.integrator = 'symbolica' or 'dy-paired-sobol'"
            )
        if values["integrate.implementation"] != "zenos":
            raise DYCardError(
                "integrate.physical_channel_multi_channeling requires "
                "integrate.implementation = 'zenos'"
            )
        if not values["integrate.multi_channeling"]:
            raise DYCardError(
                "integrate.physical_channel_multi_channeling requires "
                "integrate.multi_channeling = true"
            )
        if physical_channel_cards is None or len(physical_channel_cards) < 2:
            raise DYCardError(
                "integrate.physical_channel_multi_channeling requires at least "
                "two entries in integrate.physical_channel_cards"
            )
        if (
            physical_channel_min_fraction * len(physical_channel_cards)
            > 1.0 + 1.0e-15
        ):
            raise DYCardError(
                "integrate.physical_channel_min_fraction cannot exceed "
                "1 / number of physical channels"
            )
        duplicate_cards = len(set(physical_channel_cards.values())) != len(
            physical_channel_cards
        )
        if duplicate_cards:
            raise DYCardError(
                "integrate.physical_channel_cards must reference distinct cards"
            )
        for label, card_path in physical_channel_cards.items():
            if not Path(card_path).is_file():
                raise DYCardError(
                    f"integrate.physical_channel_cards.{label} is not a regular "
                    f"file: {card_path}"
                )
    elif physical_channel_cards is not None:
        raise DYCardError(
            "integrate.physical_channel_cards requires "
            "integrate.physical_channel_multi_channeling = true"
        )
    elif values["integrate.physical_channel_warmup_iterations"]:
        raise DYCardError(
            "integrate.physical_channel_warmup_iterations requires "
            "integrate.physical_channel_multi_channeling = true"
        )
    elif physical_channel_min_fraction:
        raise DYCardError(
            "integrate.physical_channel_min_fraction requires "
            "integrate.physical_channel_multi_channeling = true"
        )
    elif physical_channel_target_relative_precision is not None:
        raise DYCardError(
            "integrate.physical_channel_target_relative_precision requires "
            "integrate.physical_channel_multi_channeling = true"
        )
    elif physical_channel_live_diagnostics:
        raise DYCardError(
            "integrate.physical_channel_live_diagnostics requires "
            "integrate.physical_channel_multi_channeling = true"
        )
    if values["integrate.integrator"] == "dy-paired-sobol":
        from processes.dy.dy_paired_settings import PairedSobolSettings
        try:
            PairedSobolSettings(**{
                name: values["integrate.paired_sobol." + name]
                for name in PairedSobolSettings.__dataclass_fields__
            }).validate()
        except ValueError as exc:
            raise DYCardError(str(exc)) from exc
        if not physical_multi_channeling:
            raise DYCardError("dy-paired-sobol requires physical_channel_multi_channeling")
        if values["integrate.phase"] != "real" or values["integrate.parameterisation"] != "spherical":
            raise DYCardError("dy-paired-sobol requires real phase and spherical parameterisation")
        if values["integrate.beams.q_bins"] or values["integrate.restart"] or values["integrate.physical_channel_warmup_iterations"]:
            raise DYCardError("dy-paired-sobol uses its own learning schedule; Q-bin observers/restart are unsupported")
    elif values["integrate.paired_sobol.z_histogram_edges"]:
        raise DYCardError("z_histogram_edges currently requires integrator = 'dy-paired-sobol'")
    uv_filter = values["integrate.selection.integrated_uv_counterterms"]
    if uv_filter != "all" and values["integrate.implementation"] != "zenos":
        raise DYCardError(
            "integrate.selection.integrated_uv_counterterms only/exclude requires "
            "implementation = 'zenos'"
        )
    if (
        uv_filter == "only"
        and not values["generate.enable_integrated_uv_counterterms"]
    ):
        raise DYCardError(
            "integrate.selection.integrated_uv_counterterms = 'only' requires "
            "generate.enable_integrated_uv_counterterms = true"
        )
    graph_subset = values.get("integrate.selection.graphs")
    if graph_subset is not None:
        if not (
            values["integrate.implementation"] == "zenos"
            and values["integrate.integrator"] == "symbolica"
            and values["integrate.multi_channeling"]
        ):
            raise DYCardError(
                "integrate.selection.graphs requires implementation = 'zenos', "
                "integrator = 'symbolica', and multi_channeling = true"
            )
        if not graph_subset or len(set(graph_subset)) != len(graph_subset):
            raise DYCardError(
                "integrate.selection.graphs must be nonempty and contain unique selectors"
            )
        diagram_set = set(diagrams or [])
        for selector in graph_subset:
            if re.fullmatch(r"(?:graph_)?\d+", selector):
                continue
            if selector not in diagram_set:
                raise DYCardError(
                    f"integrate.selection.graphs contains source graph {selector!r} "
                    "which is absent from common.diagrams"
                )
    graph_weights = values.get("integrate.selection.graph_weights")
    if graph_weights is not None:
        if not (
            values["integrate.implementation"] == "zenos"
            and values["integrate.integrator"] == "symbolica"
            and values["integrate.multi_channeling"]
        ):
            raise DYCardError(
                "integrate.selection.graph_weights requires implementation = "
                "'zenos', integrator = 'symbolica', and multi_channeling = true"
            )
        if not graph_weights:
            raise DYCardError(
                "integrate.selection.graph_weights must contain at least one weight"
            )
    ttbar_pt = values.get("integrate.selection.ttbar_pt_min")
    if ttbar_pt is not None:
        if ttbar_pt < 0:
            raise DYCardError("integrate.selection.ttbar_pt_min must be nonnegative")
        if process_name != "tt~":
            raise DYCardError(
                "integrate.selection.ttbar_pt_min is supported only for ttbar"
            )
        if values["integrate.implementation"] != "zenos":
            raise DYCardError(
                "integrate.selection.ttbar_pt_min requires implementation = 'zenos'"
            )

    rotation_digits = values.get("integrate.stability.rotation_check_digits")
    if rotation_digits is not None and rotation_digits < 1:
        raise DYCardError(
            "integrate.stability.rotation_check_digits must be positive"
        )
    if values["integrate.stability.rotation_check_count"] < 1:
        raise DYCardError(
            "integrate.stability.rotation_check_count must be positive"
        )
    if values["integrate.stability.rotation_check_eps"] < 0:
        raise DYCardError("integrate.stability.rotation_check_eps must be nonnegative")
    if values["integrate.stability.rotation_check_arb_digits"] < 2:
        raise DYCardError(
            "integrate.stability.rotation_check_arb_digits must be at least 2"
        )
    backend = values.get("integrate.stability.backend")
    arb_digits = values.get("integrate.stability.arb_digits")
    if arb_digits is not None and backend != "arb":
        raise DYCardError("integrate.stability.arb_digits requires backend = 'arb'")
    if backend == "double-float":
        resolved_precision = 32
    elif backend == "arb":
        resolved_precision = (
            arb_digits
            if arb_digits is not None
            else values["integrate.stability.rotation_check_arb_digits"]
        )
        if resolved_precision == 32:
            raise DYCardError("backend = 'arb' cannot resolve to 32 digits")
        if resolved_precision < 2:
            raise DYCardError("Arb precision must be at least 2 digits")
    else:
        resolved_precision = values["common.precision.fallback_precision"]
    ladder_path = "integrate.stability.precision_ladder"
    ladder = values.get(ladder_path)
    legacy_paths = {"common.precision.fallback_precision", "integrate.stability.backend",
                    "integrate.stability.arb_digits", "integrate.stability.rotation_check_arb_digits"}
    if ladder is None:
        ladder = [16, resolved_precision] if supplied & legacy_paths else list(DEFAULT_PRECISION_LADDER)
    try:
        ladder = validate_precision_ladder(ladder)
    except ValueError as exc:
        raise DYCardError(str(exc)) from exc
    if ladder_path in supplied and supplied & legacy_paths:
        # Legacy backend/fallback controls describe the first retry only.
        if (ladder[1] if len(ladder) > 1 else 16) != resolved_precision:
            raise DYCardError("integrate.stability.precision_ladder conflicts with legacy fallback/backend settings")
    values[ladder_path] = list(ladder)
    if (
        backend is not None
        and "common.precision.fallback_precision" in supplied
        and values["common.precision.fallback_precision"] != resolved_precision
    ):
        raise DYCardError(
            "common.precision.fallback_precision conflicts with integrate.stability.backend"
        )
    for path in (
        "integrate.stability.relative_tolerance",
        "integrate.stability.absolute_tolerance",
        "integrate.stability.theta_tolerance",
    ):
        value = values.get(path)
        if value is not None and value < 0:
            raise DYCardError(f"{path} must be nonnegative")
    if (
        values.get("integrate.stability.relative_tolerance") == 0
        and values.get("integrate.stability.absolute_tolerance") == 0
    ):
        raise DYCardError("Relative and absolute stability tolerances cannot both be zero")
    _require_positive(
        values,
        "integrate.stability.large_weight_precision",
        "integrate.stability.large_weight_clip",
    )
    large_precision = values.get("integrate.stability.large_weight_precision")
    large_clip = values.get("integrate.stability.large_weight_clip")
    if large_precision is not None and large_clip is not None and large_precision > large_clip:
        raise DYCardError(
            "integrate.stability.large_weight_precision cannot exceed large_weight_clip"
        )
    soft_edges = values.get("integrate.stability.soft_mirror_edges")
    if soft_edges is not None:
        if values["integrate.implementation"] != "zenos":
            raise DYCardError(
                "integrate.stability.soft_mirror_edges requires "
                "implementation = 'zenos'"
            )
        if len(diagrams or []) > 1 and not values["integrate.multi_channeling"]:
            raise DYCardError(
                "Soft mirroring in a multi-graph bundle requires "
                "integrate.multi_channeling = true"
            )
        if not soft_edges or len(set(soft_edges)) != len(soft_edges):
            raise DYCardError(
                "integrate.stability.soft_mirror_edges must be nonempty and unique"
            )
        selected_graphs: set[str] = set()
        has_bare_edge = False
        for selector in soft_edges:
            graph, separator, edge = selector.partition(":")
            if separator:
                if diagrams and graph not in set(diagrams):
                    raise DYCardError(
                        f"Soft-mirror graph {graph!r} is absent from common.diagrams"
                    )
                if graph in selected_graphs:
                    raise DYCardError(
                        f"Duplicate soft-mirror specification for graph {graph!r}"
                    )
                selected_graphs.add(graph)
            else:
                edge = graph
                has_bare_edge = True
                if diagrams is None or len(diagrams) != 1:
                    raise DYCardError(
                        "A bare soft-mirror edge requires exactly one common.diagram"
                    )
            if not edge.isdigit() or int(edge) <= 0:
                raise DYCardError(
                    f"Soft-mirror selector {selector!r} must use a positive edge identifier"
                )
        if has_bare_edge and len(soft_edges) != 1:
            raise DYCardError(
                "A bare soft-mirror edge cannot be combined with other selectors"
            )

    grouped_soft_pair = values.get("integrate.stability.grouped_soft_pair")
    grouped_soft_pair_transform = values.get(
        "integrate.stability.grouped_soft_pair_transform"
    )
    if (grouped_soft_pair is None) != (grouped_soft_pair_transform is None):
        raise DYCardError(
            "integrate.stability.grouped_soft_pair and "
            "grouped_soft_pair_transform must be supplied together"
        )
    if grouped_soft_pair is not None:
        if len(grouped_soft_pair) != 2 or len(set(grouped_soft_pair)) != 2:
            raise DYCardError(
                "integrate.stability.grouped_soft_pair must contain exactly "
                "two distinct graph selectors"
            )
        if values["integrate.implementation"] != "zenos":
            raise DYCardError(
                "integrate.stability.grouped_soft_pair requires "
                "implementation = 'zenos'"
            )
        if not values["integrate.multi_channeling"]:
            raise DYCardError(
                "integrate.stability.grouped_soft_pair requires "
                "integrate.multi_channeling = true"
            )
        if values["integrate.parameterisation"] != "spherical":
            raise DYCardError(
                "integrate.stability.grouped_soft_pair requires the spherical "
                "parameterisation"
            )
        if not values["common.initial_state.symmetrise_p1_p2"]:
            raise DYCardError(
                "integrate.stability.grouped_soft_pair requires p1<->p2 "
                "symmetrisation"
            )
        if soft_edges is None or len(soft_edges) < 2:
            raise DYCardError(
                "integrate.stability.grouped_soft_pair requires soft-mirror "
                "edges for both paired graph channels"
            )

    beams_enabled = values["integrate.beams.enabled"]
    scheme_enabled = values["integrate.beams.scheme_counterterm.enabled"]
    decoupling_enabled = values["integrate.decoupling.enabled"]
    is_one_loop_dy = process_name == "dy" and loops == 1
    is_two_loop_ttbar_qqbar = (
        process_name == "tt~"
        and loops == 2
        and tuple(channel) in {(1, -1), (-1, 1)}
    )
    is_two_loop_ttbar_gg = (
        process_name == "tt~"
        and loops == 2
        and tuple(channel) == (0, 0)
    )
    is_two_loop_ttbar_qg = (
        process_name == "tt~"
        and loops == 2
        and tuple(channel) in {(1, 0), (0, 1)}
    )
    subordinate_defaults = {
        "integrate.beams.parameterisation": "x1_x2",
        "integrate.beams.z_bin": None,
        "integrate.beams.q_bins": None,
        "integrate.beams.q_min": None,
        "integrate.beams.q_max": None,
        "integrate.beams.physical_normalisation": False,
        "integrate.beams.integrated_leptonic_phase_space": False,
        "integrate.beams.pdf.set": None,
        "integrate.beams.pdf.member": 0,
        "integrate.beams.pdf.muf_sq": None,
        "integrate.beams.scheme_counterterm.enabled": False,
        "integrate.beams.scheme_counterterm.sobol_power": 15,
        "integrate.beams.scheme_counterterm.replicas": 8,
        "integrate.beams.scheme_counterterm.factor": None,
        "integrate.beams.scheme_counterterm.born_bundle": None,
        "integrate.beams.scheme_counterterm.born_bundles": None,
        "integrate.beams.scheme_counterterm.alpha_s": 0.118,
        "integrate.beams.scheme_counterterm.clip": None,
    }
    nondefault_beam_fields = [
        path
        for path, default in subordinate_defaults.items()
        if values.get(path) != default
    ]
    partonic_ttbar_scheme = (
        not beams_enabled
        and scheme_enabled
        and (
            is_two_loop_ttbar_qqbar
            or is_two_loop_ttbar_qg
            or is_two_loop_ttbar_gg
        )
    )
    partonic_dy = not beams_enabled and is_one_loop_dy
    partonic_scheme_fields = {
        "integrate.beams.physical_normalisation",
        "integrate.beams.scheme_counterterm.enabled",
        "integrate.beams.scheme_counterterm.sobol_power",
        "integrate.beams.scheme_counterterm.replicas",
        "integrate.beams.scheme_counterterm.factor",
        "integrate.beams.scheme_counterterm.born_bundle",
        "integrate.beams.scheme_counterterm.born_bundles",
        "integrate.beams.scheme_counterterm.alpha_s",
        "integrate.beams.scheme_counterterm.clip",
    }
    partonic_dy_fields = partonic_scheme_fields | {
        "integrate.beams.z_bin",
        "integrate.beams.integrated_leptonic_phase_space",
    }
    unsupported_without_beams = [
        path
        for path in nondefault_beam_fields
        if not (
            (partonic_ttbar_scheme and path in partonic_scheme_fields)
            or (partonic_dy and path in partonic_dy_fields)
        )
    ]
    if not beams_enabled and unsupported_without_beams:
        raise DYCardError(
            f"{unsupported_without_beams[0]} requires integrate.beams.enabled = true"
        )
    if (
        values["integrate.beams.parameterisation"] == "beta_y"
        and str(values["common.process_name"]).lower() != "tt~"
    ):
        raise DYCardError(
            "integrate.beams.parameterisation = 'beta_y' requires "
            "common.process_name = 'tt~'"
        )
    z_bin = values.get("integrate.beams.z_bin")
    if z_bin is not None and (
        len(z_bin) != 2 or not 0 <= z_bin[0] < z_bin[1] <= 1
    ):
        raise DYCardError("integrate.beams.z_bin requires 0 <= min < max <= 1")
    q_bins = values.get("integrate.beams.q_bins")
    if q_bins is not None:
        if len(q_bins) < 2 or any(
            not (math.isfinite(lower) and math.isfinite(upper))
            or lower < 0.0
            or lower >= upper
            for lower, upper in zip(q_bins, q_bins[1:])
        ):
            raise DYCardError(
                "integrate.beams.q_bins requires at least two strictly "
                "increasing nonnegative edges"
            )
        if not physical_multi_channeling:
            raise DYCardError(
                "integrate.beams.q_bins requires "
                "integrate.physical_channel_multi_channeling = true"
            )
        if process_name != "dy":
            raise DYCardError(
                "integrate.beams.q_bins currently supports Drell-Yan only"
            )
        if scheme_enabled and q_bins[0] <= 0.0:
            raise DYCardError(
                "Drell-Yan scheme counterterms require the first q_bins edge "
                "to be positive"
            )
        if len(q_bins) - 1 > values["integrate.physical_channel_batch_size"]:
            raise DYCardError(
                "The number of integrate.beams.q_bins cannot exceed "
                "integrate.physical_channel_batch_size"
            )
    q_min = values.get("integrate.beams.q_min")
    q_max = values.get("integrate.beams.q_max")
    if q_min is not None and q_min < 0:
        raise DYCardError("integrate.beams.q_min must be nonnegative")
    if q_max is not None and q_max <= 0:
        raise DYCardError("integrate.beams.q_max must be positive")
    if q_min is not None and q_max is not None and q_min >= q_max:
        raise DYCardError("integrate.beams requires q_min < q_max")
    pdf_set = values.get("integrate.beams.pdf.set")
    if pdf_set is not None and not _is_nonempty_string(pdf_set):
        raise DYCardError("integrate.beams.pdf.set must be nonempty")
    if values["integrate.beams.pdf.member"] < 0:
        raise DYCardError("integrate.beams.pdf.member must be nonnegative")
    muf_sq = values.get("integrate.beams.pdf.muf_sq")
    if muf_sq is not None and muf_sq <= 0:
        raise DYCardError("integrate.beams.pdf.muf_sq must be positive")
    if pdf_set is not None and muf_sq is None:
        lambda_sq = values["common.scales.lambda_sq"]
        mur_sq = values["common.scales.mur_sq"]
        if lambda_sq <= 0 or mur_sq <= 0 or lambda_sq != mur_sq:
            raise DYCardError(
                "Without beams.pdf.muf_sq, lambda_sq and mur_sq must be equal and positive"
            )
    if values["integrate.beams.physical_normalisation"]:
        if "common.channel" not in supplied:
            raise DYCardError(
                "integrate.beams.physical_normalisation requires an explicit common.channel"
            )
    if values["integrate.beams.integrated_leptonic_phase_space"] and not (
        process_name == "dy" and loops == 1
    ):
        raise DYCardError(
            "integrated leptonic phase space supports one-loop Drell-Yan only"
        )

    sobol_power = values["integrate.beams.scheme_counterterm.sobol_power"]
    replicas = values["integrate.beams.scheme_counterterm.replicas"]
    if not 1 <= sobol_power <= 30:
        raise DYCardError("Scheme-counterterm sobol_power must be between 1 and 30")
    if replicas < 2:
        raise DYCardError("Scheme-counterterm replicas must be at least 2")
    _require_positive(values, "integrate.beams.scheme_counterterm.clip")
    if beams_enabled and values["integrate.implementation"] != "zenos":
        raise DYCardError(
            "integrate.beams.enabled requires integrate.implementation = 'zenos'"
        )
    if (
        partonic_ttbar_scheme
        or (partonic_dy and scheme_enabled)
    ) and values["integrate.implementation"] != "zenos":
        raise DYCardError(
            "Partonic scheme conversion requires "
            "integrate.implementation = 'zenos'"
        )
    if scheme_enabled:
        is_two_loop_ttbar = process_name == "tt~" and loops == 2
        if not (is_one_loop_dy or is_two_loop_ttbar):
            raise DYCardError(
                "Scheme counterterms support one-loop Drell-Yan or two-loop ttbar"
            )
        allowed_channels = {(1, 0), (0, 1), (1, -1), (-1, 1)}
        if is_two_loop_ttbar_gg:
            allowed_channels.add((0, 0))
        if tuple(channel) not in allowed_channels:
            raise DYCardError("Scheme counterterm is unsupported for common.channel")
        if not values["integrate.beams.physical_normalisation"]:
            raise DYCardError(
                "Scheme counterterm requires physical normalisation"
            )
        if is_two_loop_ttbar_qqbar:
            if beams_enabled and pdf_set is None:
                raise DYCardError(
                    "Hadronic two-loop ttbar qqbar scheme conversion requires PDF weighting"
                )
            if not beams_enabled and pdf_set is not None:
                raise DYCardError(
                    "Partonic two-loop ttbar qqbar scheme conversion does not accept PDFs"
                )
        elif is_two_loop_ttbar_gg:
            if beams_enabled and pdf_set is None:
                raise DYCardError(
                    "Hadronic two-loop ttbar gg scheme conversion requires PDF weighting"
                )
            if not beams_enabled and pdf_set is not None:
                raise DYCardError(
                    "Partonic two-loop ttbar gg scheme conversion does not accept PDFs"
                )
        elif is_two_loop_ttbar_qg:
            if beams_enabled and pdf_set is None:
                raise DYCardError(
                    "Hadronic two-loop ttbar qg scheme conversion requires PDF weighting"
                )
            if not beams_enabled and pdf_set is not None:
                raise DYCardError(
                    "Partonic two-loop ttbar qg scheme conversion does not accept PDFs"
                )
        elif is_one_loop_dy:
            if beams_enabled and pdf_set is None:
                raise DYCardError(
                    "Hadronic one-loop DY scheme conversion requires PDF weighting"
                )
            if not beams_enabled and pdf_set is not None:
                raise DYCardError(
                    "Partonic one-loop DY scheme conversion does not accept PDFs"
                )
            if not beams_enabled and tuple(channel) not in {(1, -1), (-1, 1)}:
                raise DYCardError(
                    "Partonic one-loop DY scheme conversion currently supports qqbar only"
                )
            if beams_enabled and (q_min is None or q_min <= 0):
                raise DYCardError(
                    "Hadronic one-loop DY scheme counterterm requires positive q_min"
                )
        if is_two_loop_ttbar:
            _require_positive(
                values,
                "integrate.beams.scheme_counterterm.alpha_s",
            )
    if decoupling_enabled:
        if not scheme_enabled:
            raise DYCardError(
                "integrate.decoupling.enabled requires scheme conversion"
            )
        if not (
            is_two_loop_ttbar_qqbar
            or is_two_loop_ttbar_qg
            or is_two_loop_ttbar_gg
        ):
            raise DYCardError(
                "Decoupling supports two-loop ttbar qqbar, qg, and gg only"
            )


def _nested_settings(
    values: Mapping[str, Any], command: str, supplied: set[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    prefixes = ("common.", f"{command}.")
    for field in _FIELDS:
        if not field.path.startswith(prefixes):
            continue
        if (
            command == "generate"
            and field.path.startswith("generate.spenso.")
            and values.get("generate.generation_type") == "gammaloop"
        ):
            continue
        if (
            command == "integrate"
            and field.path == "common.precision.fallback_precision"
            and values.get("integrate.stability.backend") is not None
            and field.path not in supplied
        ):
            continue
        if field.path in _OPTIONAL_MOMENTA and field.path not in supplied:
            continue
        value = values.get(field.path)
        if command == "integrate" and field.path == "common.precision.fallback_precision":
            ladder = values["integrate.stability.precision_ladder"]
            value = ladder[1] if len(ladder) > 1 else 16
        if value is None:
            continue
        cursor = result
        parts = field.path.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _clone(value)
    return result


def _flat_generation_settings(
    values: Mapping[str, Any],
    supplied: set[str],
    configuration_hashes: Mapping[str, str],
    *,
    compatibility_only: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in _FIELDS:
        path = field.path
        if not (path.startswith("common.") or path.startswith("generate.")):
            continue
        if (
            path.startswith("generate.spenso.")
            and values.get("generate.generation_type") == "gammaloop"
        ):
            continue
        if compatibility_only and path in _OPERATIONAL_GENERATION_PATHS:
            continue
        if compatibility_only and path in _RUNTIME_DEFAULT_GENERATION_PATHS:
            continue
        if compatibility_only and path == _TOP_SELF_ENERGY_LEGACY_PATH:
            # The enum is the canonical compatibility key.  Keeping the
            # legacy boolean here as well would make one physical choice
            # appear twice and would prevent transparent migration of old
            # boolean-only bundle metadata.
            continue
        if compatibility_only and path in _OPTIONAL_MOMENTA and path not in supplied:
            continue
        value = values.get(path)
        if value is None:
            continue
        if path in _CONFIGURATION_PATHS:
            value = f"sha256:{configuration_hashes[path]}"
        result[path] = _clone(value)
    mode = str(values.get(_TOP_SELF_ENERGY_MODE_PATH, "no-os"))
    result[_PROJECTED_OS_SCHEMA_FINGERPRINT_PATH] = (
        projected_os_schema_for_mode(mode)
    )
    return result


def normalise_generation_compatibility_settings(
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonicalise legacy boolean-only generation metadata.

    Historical ``no-os`` and ``os`` bundles remain compatible with the
    corresponding enum request.  No legacy metadata can normalise to the new
    ``projected-os`` mode.
    """

    normalised = _clone(settings)
    mode = normalised.get(_TOP_SELF_ENERGY_MODE_PATH)
    legacy_present = _TOP_SELF_ENERGY_LEGACY_PATH in normalised
    legacy = normalised.pop(_TOP_SELF_ENERGY_LEGACY_PATH, None)
    try:
        resolved = resolve_top_self_energy_renormalisation(
            mode,
            legacy if legacy_present else None,
            legacy_is_explicit=legacy_present,
        )
    except ValueError:
        # Preserve corrupt/conflicting values so the ordinary compatibility
        # diff reports them instead of silently blessing the bundle.
        return normalised
    normalised[_TOP_SELF_ENERGY_MODE_PATH] = resolved
    normalised[_PROJECTED_OS_SCHEMA_FINGERPRINT_PATH] = (
        projected_os_schema_for_mode(resolved)
    )
    return normalised


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, Mapping):
        return "{ " + ", ".join(
            f"{key} = {_toml_value(item)}" for key, item in value.items()
        ) + " }"
    raise TypeError(f"Cannot serialize {type(value).__name__} to TOML")


def _emit_toml_table(
    lines: list[str], table: Mapping[str, Any], prefix: str = ""
) -> None:
    scalars = [(key, value) for key, value in table.items() if not isinstance(value, Mapping)]
    children = [(key, value) for key, value in table.items() if isinstance(value, Mapping)]
    if prefix:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append(f"[{prefix}]")
    for key, value in scalars:
        lines.append(f"{key} = {_toml_value(value)}")
    for key, child in children:
        child_prefix = f"{prefix}.{key}" if prefix else key
        _emit_toml_table(lines, child, child_prefix)


def settings_to_toml(settings: Mapping[str, Any]) -> str:
    lines: list[str] = []
    for key, value in settings.items():
        if not isinstance(value, Mapping):
            lines.append(f"{key} = {_toml_value(value)}")
    for key, value in settings.items():
        if isinstance(value, Mapping):
            _emit_toml_table(lines, value, key)
    return "\n".join(lines).rstrip() + "\n"


def hash_provenance_sources(repo_root: str | os.PathLike[str] | None = None) -> dict[str, str]:
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    paths = [root / "src" / "pygloop.py"]
    paths.extend(sorted((root / "src" / "processes" / "dy").rglob("*.py")))
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in paths
        if path.is_file()
    }


@dataclass(frozen=True)
class EffectiveDYCard:
    command: str
    card_path: str
    source_card_sha256: str
    settings: Mapping[str, Any]
    argparse_values: Mapping[str, Any]
    generation_settings: Mapping[str, Any]
    generation_compatibility_settings: Mapping[str, Any]
    generation_fingerprint: str
    configuration_hashes: Mapping[str, str]
    source_hashes: Mapping[str, str]

    def to_toml(self) -> str:
        settings = self.as_dict()
        # The enum is the canonical serialised interface.  Emitting the
        # deprecated boolean beside projected-os would turn a valid dump into
        # a conflicting card when it is loaded again.
        settings.get("generate", {}).pop(
            "top_self_energy_os_subtraction", None
        )
        return settings_to_toml(settings)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON/TOML-friendly detached effective settings."""
        return _clone(self.settings)

    def with_runtime_parameters(
        self, parameters: Mapping[str, Any]
    ) -> "EffectiveDYCard":
        """Reflect post-merge CLI runtime overrides in effective-card output."""
        if self.command != "integrate":
            raise DYCardError(
                "DY runtime parameters can only be attached to an integration card."
            )
        normalised = _normalise_value(
            FIELD_BY_PATH["integrate.runtime_parameters"], parameters
        )
        settings = self.as_dict()
        settings.setdefault("integrate", {})["runtime_parameters"] = normalised
        argparse_values = dict(self.argparse_values)
        argparse_values["dy_runtime_parameters"] = _clone(normalised)
        return replace(
            self,
            settings=_freeze(settings),
            argparse_values=MappingProxyType(argparse_values),
        )

    def bundle_metadata(self) -> dict[str, Any]:
        mode = self.generation_compatibility_settings[
            _TOP_SELF_ENERGY_MODE_PATH
        ]
        return {
            "dy_generation_settings": _clone(
                self.generation_compatibility_settings
            ),
            "dy_generation_provenance": _clone(self.generation_settings),
            "dy_generation_settings_fingerprint": self.generation_fingerprint,
            "dy_source_card_sha256": self.source_card_sha256,
            "dy_configuration_hashes": dict(self.configuration_hashes),
            "dy_source_hashes": dict(self.source_hashes),
            "dy_top_self_energy_renormalisation": mode,
            "dy_projected_os_construction_schema_version": (
                projected_os_schema_for_mode(mode)
            ),
        }


@dataclass(frozen=True)
class DYCard:
    path: str
    source_card_sha256: str
    values: Mapping[str, Any]
    supplied_paths: frozenset[str]

    def merge(
        self,
        command: str,
        argparse_values: Mapping[str, Any] | object,
        explicit_destinations: Iterable[str] = (),
        generation_defaults: Mapping[str, Any] | None = None,
    ) -> EffectiveDYCard:
        if command not in SUPPORTED_COMMANDS:
            raise DYCardError(
                f"--dy-card supports only generate or integrate, got {command!r}"
            )
        namespace_values = (
            dict(argparse_values)
            if isinstance(argparse_values, Mapping)
            else vars(argparse_values)
        )
        explicit = set(explicit_destinations)
        default_inputs = dict(generation_defaults or {})
        default_inputs.update(
            {path: _clone(value) for path, value in self.values.items()}
        )
        merged = _dynamic_defaults(default_inputs)
        supplied = set(self.supplied_paths)

        for field in _FIELDS:
            if field.destination not in explicit:
                continue
            if not (
                field.path.startswith("common.")
                or field.path.startswith(f"{command}.")
            ):
                continue
            if field.destination not in namespace_values:
                continue
            cli_value = namespace_values[field.destination]
            if field.destination == "generation_type" and isinstance(
                cli_value, (list, tuple)
            ):
                if len(cli_value) != 1:
                    raise DYCardError("generation_type must contain one value")
                cli_value = cli_value[0]
            normalised_cli_value = _normalise_value(field, cli_value)
            if field.kind == "path":
                normalised_cli_value = str(
                    Path(normalised_cli_value).expanduser().resolve()
                )
            elif field.kind == "path_mapping":
                normalised_cli_value = {
                    label: str(Path(path).expanduser().resolve())
                    for label, path in normalised_cli_value.items()
                }
            merged[field.path] = normalised_cli_value
            supplied.add(field.path)

        if "common.scales.muv" not in supplied:
            merged["common.scales.muv"] = 1.0 if merged["common.n_loops"] == 1 else 2000.0
        if "common.scales.lambda_sq" not in supplied:
            merged["common.scales.lambda_sq"] = (
                2.0 if merged["common.n_loops"] == 1 else 50000.0
            )
        if "common.scales.mur_sq" not in supplied:
            merged["common.scales.mur_sq"] = (
                1.0 if merged["common.n_loops"] == 1 else 50000.0
            )

        _validate_values(merged, supplied)
        configuration_hashes = {
            path: _sha256_file(merged[path]) for path in _CONFIGURATION_PATHS
        }
        generation_settings = _flat_generation_settings(
            merged,
            supplied,
            configuration_hashes,
            compatibility_only=False,
        )
        compatibility_settings = _flat_generation_settings(
            merged,
            supplied,
            configuration_hashes,
            compatibility_only=True,
        )
        fingerprint = _sha256_bytes(
            _canonical_json(compatibility_settings).encode("utf-8")
        )

        projected: dict[str, Any] = {}
        for field in _FIELDS:
            if not (
                field.path.startswith("common.")
                or field.path.startswith(f"{command}.")
            ):
                continue
            value = merged.get(field.path)
            if command == "integrate" and field.path == "common.precision.fallback_precision":
                ladder = merged["integrate.stability.precision_ladder"]
                value = ladder[1] if len(ladder) > 1 else 16
            if field.destination == "generation_type" and value is not None:
                value = [value]
            elif field.destination == "target" and value is not None:
                value = complex(value["real"], value["imag"])
            projected[field.destination] = _clone(value)

        settings = _nested_settings(merged, command, supplied)
        return EffectiveDYCard(
            command=command,
            card_path=self.path,
            source_card_sha256=self.source_card_sha256,
            settings=_freeze(settings),
            argparse_values=MappingProxyType(projected),
            generation_settings=_freeze(generation_settings),
            generation_compatibility_settings=_freeze(compatibility_settings),
            generation_fingerprint=fingerprint,
            configuration_hashes=MappingProxyType(configuration_hashes),
            source_hashes=MappingProxyType(hash_provenance_sources()),
        )


def load_dy_card(path: str | os.PathLike[str]) -> DYCard:
    card_path = Path(path).expanduser().resolve()
    try:
        raw_bytes = card_path.read_bytes()
    except OSError as exc:
        raise DYCardError(f"Could not read DY card '{card_path}': {exc}") from exc
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DYCardError(f"DY card is not valid UTF-8: {card_path}") from exc
    try:
        parsed = tomllib.loads(raw_text)
    except tomllib.TOMLDecodeError as exc:
        raise DYCardError(f"Could not parse DY card '{card_path}': {exc}") from exc
    if not isinstance(parsed, dict):  # pragma: no cover - tomllib invariant
        raise DYCardError("A DY card must contain a TOML document")
    flattened = _flatten_card_table(parsed)
    schema_version = flattened.pop("schema_version", None)
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise DYCardError(
            f"schema_version must be the integer {SCHEMA_VERSION}; got {schema_version!r}"
        )

    normalised: dict[str, Any] = {}
    for path_key, value in flattened.items():
        field = FIELD_BY_PATH[path_key]
        normalised_value = _normalise_value(field, value)
        if field.kind == "path":
            candidate = Path(normalised_value).expanduser()
            if not candidate.is_absolute():
                candidate = card_path.parent / candidate
            normalised_value = str(candidate.resolve())
        elif field.kind == "path_mapping":
            resolved_paths = {}
            for label, raw_path in normalised_value.items():
                candidate = Path(raw_path).expanduser()
                if not candidate.is_absolute():
                    candidate = card_path.parent / candidate
                resolved_paths[label] = str(candidate.resolve())
            normalised_value = resolved_paths
        normalised[path_key] = normalised_value

    initial = _dynamic_defaults(normalised)
    _validate_values(initial, set(normalised))
    return DYCard(
        path=str(card_path),
        source_card_sha256=_sha256_bytes(raw_bytes),
        values=_freeze(normalised),
        supplied_paths=frozenset(normalised),
    )


def compatibility_diff(
    generated: Mapping[str, Any], requested: Mapping[str, Any]
) -> list[tuple[str, Any, Any]]:
    """Return stable field-level differences between compatibility projections."""
    differences: list[tuple[str, Any, Any]] = []
    missing = object()
    for path in sorted(set(generated) | set(requested)):
        generated_value = generated.get(path, missing)
        requested_value = requested.get(path, missing)
        if generated_value != requested_value:
            differences.append((
                path,
                "<missing>" if generated_value is missing else generated_value,
                "<missing>" if requested_value is missing else requested_value,
            ))
    return differences


def generation_settings_fingerprint(settings: Mapping[str, Any]) -> str:
    """Calculate the stable fingerprint of a compatibility projection."""
    return _sha256_bytes(_canonical_json(_clone(settings)).encode("utf-8"))


def format_compatibility_diff(
    bundle_name: str,
    differences: Iterable[tuple[str, Any, Any]],
) -> str:
    lines = [f"Cannot integrate bundle {bundle_name}:"]
    lines.extend(
        f"  {path}: generated={generated!r} requested={requested!r}"
        for path, generated, requested in differences
    )
    lines.append("Regenerate the bundle or select a compatible card.")
    return "\n".join(lines)


__all__ = [
    "CARD_TO_DESTINATION",
    "DYCard",
    "DYCardError",
    "EffectiveDYCard",
    "SCHEMA_VERSION",
    "compatibility_diff",
    "format_compatibility_diff",
    "generation_settings_fingerprint",
    "hash_provenance_sources",
    "load_dy_card",
    "normalise_generation_compatibility_settings",
    "settings_to_toml",
]
