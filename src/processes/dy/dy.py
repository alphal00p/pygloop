from __future__ import annotations

import concurrent.futures
import contextlib
import copy
import io
import json
import logging
import math
import multiprocessing
import os
import random
import shutil
import time
import traceback
from collections.abc import Mapping, Sequence
from copy import deepcopy
from decimal import Decimal, localcontext
from itertools import product  # noqa: F401
from pprint import pformat, pprint  # noqa: F401
from typing import Any, Callable

import numpy as np
import progressbar  # pyright: ignore
import vegas  # type: ignore

from gammaloop import (  # isort: skip # type: ignore # noqa: F401
    GammaLoopAPI,
    LogLevel,
    evaluate_graph_overall_factor,
)

try:
    from gammaloop import git_version  # isort: skip # type: ignore # noqa: F401
except ImportError:
    try:
        from gammaloop import __version__ as git_version  # isort: skip # type: ignore # noqa: F401
    except ImportError:
        git_version = "unknown"
from matplotlib.typing import CapStyleType, ColorType  # noqa: F401 # pyright: ignore
from symbolica import E, Expression, NumericalIntegrator, Sample  # pyright: ignore
from symbolica.community.idenso import (  # noqa: F401 # pyright: ignore
    simplify_color,
    simplify_gamma,
    simplify_metrics,
)
from symbolica.community.spenso import *  # noqa: F403 # type: ignore

from processes.dy.dy_classes import (  # noqa: F401
    DYDotGraphs,
    VacuumDotGraph,
    canonicalise_vacuum_graph,
    filter_symmetrised_p1_p2_routed_cuts,
)
from processes.dy.dy_evaluators import (
    DYCompiledBatchRequest,
    DYCompiledBundle,
    DYCompiledEvaluationDiagnostics,
    DYDoubleRootFailure,
    DY_CM_EVALUATOR_SCHEMA,
    DYNumericalEvaluationError,
    TTBAR_CM_E_SURFACE_SCHEMA,
    compile_integrands,
    evaluate_integrand,
)
from processes.dy.dy_graph_utils import _strip_quotes
from processes.dy.dy_ghosts import (
    closed_ghost_loop_count,
    ghost_particle_names_from_model_metadata,
)
from processes.dy.dy_infrared_test import (
    approach_point,
    # evaluate_integrand,
    infrared_test,
    ultraviolet_test,
)
from processes.dy.dy_integrand import (
    EMRIntegrandConstructor,
    LoopIntegrandConstructor,
    resolve_threshold_h_function,
    routed_cut_graph,
)
from processes.dy.dy_pdf import (
    DY_INTEGRATED_LEPTONIC_PHASE_SPACE_FACTOR,
    DYGGAuxiliaryResult,
    DYPDFProvider,
    DYRegularSchemeConvolution,
    DYQQbarAuxiliaryResult,
    DYSchemeCountertermResult,
    QG_SCHEME_COUNTERTERM_FACTOR,
    QQBAR_SCHEME_COUNTERTERM_FACTOR,
    finite_g_to_q_scheme_kernel,
    finite_q_to_g_scheme_kernel,
    integrate_gq_scheme_counterterm,
    integrate_partonic_qqbar_scheme_counterterm,
    integrate_qqbar_scheme_counterterm,
    integrate_regular_born_scheme_counterterm,
    integrate_ttbar_gg_auxiliary,
    integrate_ttbar_qqbar_auxiliary,
    integrate_ttbar_qqbar_scheme_counterterm,
    physical_beam_normalisation_factor,
    resolve_factorisation_scale_sq,
)
from processes.dy.dy_runtime_parameters import (
    DY_COUPLING_NORMALISATION_CONVENTION,
    PI_DECIMAL,
    dy_coupling_normalisation_factor,
    exact_decimal_string,
)
from processes.dy.dy_stability import (
    RotationDescriptor,
    SoftEdgeRouting,
    build_high_precision_sample,
    decimal_from_input,
    decimal_values_agree,
    diagnostic_soft_equator_map,
    diagnostic_soft_radial_map,
    float_values_agree,
    mirror_loop_momenta_for_soft_edge,
    parameterize_ttbar_beam_fractions,
    rotation_descriptors_from_xs,
    rotation_matrix_from_descriptor,
    rotate_vector,
)
from processes.dy.dy_top_self_energy import (
    projected_os_schema_for_mode,
    resolve_top_self_energy_renormalisation,
)
from utils.utils import (
    CONFIGS_FOLDER,  # noqa: F401
    DOTS_FOLDER,  # noqa: F401
    EVALUATORS_FOLDER,  # noqa: F401
    GAMMALOOP_STATES_FOLDER,  # noqa: F401
    INTEGRATION_WORKSPACE_FOLDER,  # noqa: F401
    OUTPUTS_FOLDER,  # noqa: F401
    PYGLOOP_FOLDER,
    Colour,
    IntegrationResult,
    SymbolicaSample,
    chunks,
    expr_to_string,
    logger,
    pygloopException,
    set_gammaloop_level,
    set_tmp_logger_level,  # noqa: F401
    write_text_with_dirs,
)
from utils.vectors import LorentzVector, Vector
from processes.dy.dy_precision import run_precision_ladder, validate_precision_ladder

pjoin = os.path.join

TOLERANCE: float = 1e-10
DY_STABILITY_RESCUE_PRECISIONS: tuple[int, ...] = ()  # no implicit rescue levels
DY_STABILITY_FAILURE_COUNTERS = (
    "stability_hp_disagreement_count",
    "stability_hp_nonfinite_count",
    "stability_hp_error_count",
)
DY_DEFAULT_LAMBDA_MUR_SQ: dict[int, tuple[float, float]] = {
    1: (2.0, 1.0),
    2: (50000.0, 50000.0),
}
DY_GROUPED_SOFT_PAIR_SELECTOR = -2


def _dy_top_self_energy_mode(process: object) -> str:
    """Resolve modern and legacy attributes on real and lightweight DY objects."""

    mode = getattr(process, "dy_top_self_energy_renormalisation", None)
    legacy = getattr(process, "dy_top_self_energy_os_subtraction", False)
    return resolve_top_self_energy_renormalisation(
        mode,
        legacy,
        legacy_is_explicit=mode is None,
    )


def _dy_top_self_energy_legacy_forwarding(process: object) -> bool | None:
    """Forward the old flag except when it would mislabel projected-os."""

    mode = _dy_top_self_energy_mode(process)
    return None if mode == "projected-os" else mode == "os"


def _dy_process_2l_graph_worker(task: dict[str, Any]) -> dict[str, Any]:
    worker_name = task["worker_name"]
    log_path = task["log_path"]
    start = time.monotonic()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    try:
        with open(log_path, "w", encoding="utf-8", buffering=1) as log_handle:
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(
                log_handle
            ):
                worker_cls = type(
                    f"{worker_name}Process",
                    (DY,),
                    {"name": worker_name},
                )
                worker = worker_cls(
                    m_top=task["m_top"],
                    m_higgs=task["m_higgs"],
                    ps_point=task["ps_point"],
                    helicities=task["helicities"],
                    n_loops=2,
                    toml_config_path=task["toml_config_path"],
                    runtime_toml_config_path=task["runtime_toml_config_path"],
                    final_state=task["final_state"],
                    process_name=task["process_name"],
                    dy_channel=task["dy_channel"],
                    skip_ps_validation=True,
                    integrate_beams=task["integrate_beams"],
                    external_gluon_polarisation=task[
                        "external_gluon_polarisation"
                    ],
                    disable_integrated_uv_cts=task["disable_integrated_uv_cts"],
                    dy_top_self_energy_os_subtraction=task.get(
                        "dy_top_self_energy_os_subtraction",
                        None
                        if task.get("dy_top_self_energy_renormalisation")
                        is not None
                        else False,
                    ),
                    dy_top_self_energy_renormalisation=task.get(
                        "dy_top_self_energy_renormalisation"
                    ),
                    dy_check_generation_limits=task["dy_check_generation_limits"],
                    dy_threshold_h_function=task.get("dy_threshold_h_function"),
                    dy_include_disabled_threshold_counterterms=task.get(
                        "dy_include_disabled_threshold_counterterms", False
                    ),
                    dy_fallback_precision=task["dy_fallback_precision"],
                    dy_lambda_sq=task["dy_lambda_sq"],
                    dy_mur_sq=task["dy_mur_sq"],
                    dy_observable_muv=task["dy_observable_muv"],
                    dy_parallel_graphs=1,
                    symmetrise_p1_p2=task.get("symmetrise_p1_p2", False),
                    skip_gl_worker_init=True,
                    load_compiled_bundle=False,
                    clean=True,
                    logger_level=logging.CRITICAL,
                )
                worker.dy_graph_index_offset = int(task["graph_index"])
                worker.dy_emr_state_name = worker_name
                worker._dy_ghost_particle_names = frozenset(
                    task.get("ghost_particle_names", ())
                )
                processed_graphs = worker.process_2L_generated_graphs(
                    DYDotGraphs(dot_str=task["graph_dot"])
                )
                processed_dot = "\n\n".join(g.to_string() for g in processed_graphs)
        return {
            "ok": True,
            "graph_index": int(task["graph_index"]),
            "graph_name": task["graph_name"],
            "worker_name": worker_name,
            "bundle_name": f"{worker_name}_2L_processed",
            "processed_dot": processed_dot,
            "log_path": log_path,
            "elapsed": time.monotonic() - start,
        }
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        with open(log_path, "a", encoding="utf-8") as log_handle:
            log_handle.write("\nDY parallel graph worker failed:\n")
            log_handle.write(traceback.format_exc())
        return {
            "ok": False,
            "graph_index": int(task["graph_index"]),
            "graph_name": task.get("graph_name"),
            "worker_name": worker_name,
            "log_path": log_path,
            "elapsed": time.monotonic() - start,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }

RESCALING: float = 1.0


def _diagnostic_soft_radius_hp_trigger(
    xs: Sequence[float], routing: SoftEdgeRouting | None
) -> bool:
    raw_threshold = os.environ.get("PYGLOOP_DIAGNOSTIC_SOFT_RADIUS_HP")
    if raw_threshold is None or routing is None:
        return False
    threshold = float(raw_threshold)
    radial_index = 3 * routing.pivot_loop_index
    return 0.0 <= float(xs[radial_index]) < threshold


class DY(object):
    name = "DY"

    def __init__(
        self,
        m_top: float,
        m_higgs: float,
        ps_point: list[LorentzVector],
        helicities: list[int] | None = None,
        n_loops: int = 1,
        toml_config_path: str | None = None,
        runtime_toml_config_path: str | None = None,
        final_state: list[str] | None = None,
        process_name: str | None = None,
        diagrams: list[str] | None = None,
        dy_channel: tuple[int, int] | list[int] | None = None,
        skip_ps_validation: bool = False,
        integrate_beams: bool = False,
        dy_z_bin: tuple[float, float] | list[float] | None = None,
        dy_q_min: float | None = None,
        dy_q_max: float | None = None,
        dy_physical_normalisation: bool = False,
        dy_integrated_leptonic_phase_space: bool = False,
        external_gluon_polarisation: bool = False,
        disable_integrated_uv_cts: bool = True,
        dy_top_self_energy_os_subtraction: bool | None = None,
        dy_top_self_energy_renormalisation: str | None = None,
        dy_check_generation_limits: bool = False,
        dy_threshold_h_function: str | None = None,
        dy_include_disabled_threshold_counterterms: bool = False,
        dy_parallel_graphs: int = 1,
        dy_fallback_precision: int | None = None,
        dy_lambda_sq: float | None = None,
        dy_mur_sq: float | None = None,
        dy_pdf_set: str | None = None,
        dy_pdf_member: int = 0,
        dy_muf_sq: float | None = None,
        dy_pdf_luminosity_family: str | None = None,
        dy_msbar_scheme_counterterm: bool = False,
        dy_decoupling: bool = False,
        dy_scheme_counterterm_sobol_power: int = 15,
        dy_scheme_counterterm_replicas: int = 8,
        dy_scheme_counterterm_factor: float | None = None,
        dy_scheme_born_bundle: str | None = None,
        dy_scheme_born_bundles: list[str] | tuple[str, ...] | None = None,
        dy_scheme_alpha_s: float = 0.118,
        dy_scheme_counterterm_clip: float | None = None,
        dy_observable_muv: float | None = None,
        dy_runtime_parameters: Mapping[str, Any] | None = None,
        skip_gl_worker_init: bool = False,
        load_compiled_bundle: bool = True,
        clean=True,
        logger_level: int | None = None,
        symmetrise_p1_p2: bool = False,
        process_basename: str | None = None,
        **opts,
    ):
        start_logger_level = logger.getEffectiveLevel()
        if logger_level is not None:
            logger.setLevel(logger_level)

        if process_basename is None:
            self.name = type(self).name
        else:
            normalised_basename = str(process_basename).strip()
            if not normalised_basename:
                raise ValueError("DY process_basename cannot be empty.")
            if os.path.basename(normalised_basename) != normalised_basename:
                raise ValueError(
                    "DY process_basename must be a basename, not a path."
                )
            self.name = normalised_basename

        self.m_top = m_top
        self.m_higgs = m_higgs
        self.ps_point = ps_point
        if helicities is None:
            helicities = [1, 1, 0, 0, 0]
        self.helicities = helicities
        self.n_loops = n_loops
        self.final_state = (
            copy.deepcopy(final_state) if final_state is not None else ["a"]
        )
        supplied_process_name = process_name if process_name is not None else "DY"
        self.process_name = (
            "DY" if str(supplied_process_name).lower() == "dy" else supplied_process_name
        )
        self.diagrams = copy.deepcopy(diagrams) if diagrams is not None else None
        dy_channel_was_explicit = dy_channel is not None
        self.dy_channel = (
            tuple(int(parton) for parton in dy_channel)
            if dy_channel is not None
            else (1, -1)
        )
        if self.dy_channel not in {(0, 0), (0, 1), (1, 0), (1, -1), (-1, 1)}:
            raise ValueError(
                "Unsupported DY channel "
                f"{self.dy_channel}; supported channels are (0,0), (0,1), "
                "(1,0), (1,-1), and (-1,1)."
            )
        self.dy_msbar_scheme_counterterm = bool(dy_msbar_scheme_counterterm)
        self.dy_decoupling = bool(dy_decoupling)
        self._dy_ttbar_qqbar_scheme_mode = (
            self.process_name.lower() == "tt~"
            and self.n_loops == 2
            and self.dy_channel in {(1, -1), (-1, 1)}
        )
        self._dy_ttbar_gg_scheme_mode = (
            self.process_name.lower() == "tt~"
            and self.n_loops == 2
            and self.dy_channel == (0, 0)
        )
        self._dy_ttbar_qg_scheme_mode = (
            self.process_name.lower() == "tt~"
            and self.n_loops == 2
            and self.dy_channel in {(1, 0), (0, 1)}
        )
        self._dy_ttbar_partonic_scheme_mode = (
            self._dy_ttbar_qqbar_scheme_mode
            or self._dy_ttbar_gg_scheme_mode
            or self._dy_ttbar_qg_scheme_mode
        )
        if self.dy_decoupling and not self.dy_msbar_scheme_counterterm:
            raise pygloopException(
                "The ttbar decoupling contribution requires MSbar scheme conversion."
            )
        if self.dy_decoupling and not self._dy_ttbar_partonic_scheme_mode:
            raise pygloopException(
                "The decoupling contribution supports two-loop ttbar qqbar, "
                "qg, and gg channels only."
            )
        self.integrate_beams = bool(integrate_beams)
        self._dy_partonic_one_loop_mode = (
            not self.integrate_beams
            and self.process_name.lower() == "dy"
            and self.n_loops == 1
        )
        self.dy_z_bin: tuple[float, float] | None = None
        if dy_z_bin is not None:
            if self.process_name.lower() != "dy":
                raise pygloopException("--dy-z-bin requires Drell-Yan.")
            z_min, z_max = (float(value) for value in dy_z_bin)
            if not (
                math.isfinite(z_min)
                and math.isfinite(z_max)
                and 0.0 <= z_min < z_max <= 1.0
            ):
                raise pygloopException(
                    "--dy-z-bin requires 0 <= ZMIN < ZMAX <= 1."
                )
            self.dy_z_bin = (z_min, z_max)

        self.dy_q_min = float(dy_q_min) if dy_q_min is not None else None
        self.dy_q_max = float(dy_q_max) if dy_q_max is not None else None
        if self.dy_q_min is not None or self.dy_q_max is not None:
            if not self.integrate_beams or self.process_name.lower() != "dy":
                raise pygloopException(
                    "--dy-q-min/--dy-q-max require beam-convoluted Drell-Yan."
                )
            if self.dy_q_min is not None and (
                not math.isfinite(self.dy_q_min) or self.dy_q_min < 0.0
            ):
                raise pygloopException("--dy-q-min must be finite and non-negative.")
            if self.dy_q_max is not None and (
                not math.isfinite(self.dy_q_max) or self.dy_q_max <= 0.0
            ):
                raise pygloopException("--dy-q-max must be finite and positive.")
            if (
                self.dy_q_min is not None
                and self.dy_q_max is not None
                and self.dy_q_min >= self.dy_q_max
            ):
                raise pygloopException("DY virtuality bounds require QMIN < QMAX.")
        self.dy_physical_normalisation = bool(dy_physical_normalisation)
        self.dy_physical_normalisation_factor = 1.0
        if self.dy_physical_normalisation:
            if not self.integrate_beams and not (
                self._dy_partonic_one_loop_mode
                or (
                    self.dy_msbar_scheme_counterterm
                    and self._dy_ttbar_partonic_scheme_mode
                )
            ):
                raise pygloopException(
                    "DY physical normalisation requires --dy-integrate-beams, "
                    "except for one-loop partonic DY and two-loop partonic "
                    "ttbar scheme conversion."
                )
            if not dy_channel_was_explicit:
                raise pygloopException(
                    "DY physical normalisation requires an explicit --dy-channel."
                )
            self.dy_physical_normalisation_factor = (
                physical_beam_normalisation_factor(
                    self.dy_channel,
                    self.n_loops,
                )
            )
        self.dy_integrated_leptonic_phase_space = bool(
            dy_integrated_leptonic_phase_space
        )
        self.dy_integrated_leptonic_phase_space_factor = 1.0
        if self.dy_integrated_leptonic_phase_space:
            if self.process_name.lower() != "dy":
                raise pygloopException(
                    "The integrated leptonic phase-space factor requires "
                    "Drell-Yan."
                )
            if self.n_loops != 1:
                raise pygloopException(
                    "The integrated leptonic phase-space factor currently "
                    "supports one-loop Drell-Yan."
                )
            self.dy_integrated_leptonic_phase_space_factor = (
                DY_INTEGRATED_LEPTONIC_PHASE_SPACE_FACTOR
            )
        self.external_gluon_polarisation = bool(external_gluon_polarisation)
        self.dy_fallback_precision = (
            int(dy_fallback_precision) if dy_fallback_precision is not None else 80
        )
        self.enforce_ttbar_beam_threshold = (
            self.integrate_beams and self.process_name.lower() == "tt~"
        )
        self.skip_gl_worker_init = bool(skip_gl_worker_init)
        self.load_compiled_bundle = bool(load_compiled_bundle)
        self.disable_integrated_uv_cts = bool(disable_integrated_uv_cts)
        self.dy_include_disabled_threshold_counterterms = bool(
            dy_include_disabled_threshold_counterterms
        )
        try:
            self.dy_top_self_energy_renormalisation = (
                resolve_top_self_energy_renormalisation(
                    dy_top_self_energy_renormalisation,
                    dy_top_self_energy_os_subtraction,
                )
            )
        except ValueError as error:
            raise pygloopException(str(error)) from error
        # This compatibility attribute deliberately denotes the historical
        # ``os`` choice only.  In particular, it never aliases projected-os.
        self.dy_top_self_energy_os_subtraction = (
            self.dy_top_self_energy_renormalisation == "os"
        )
        if self.dy_top_self_energy_renormalisation != "no-os":
            option = (
                "--dy-top-self-energy-os-subtraction"
                if dy_top_self_energy_renormalisation is None
                else "--dy-top-self-energy-renormalisation"
            )
            if self.process_name.lower() != "tt~":
                raise pygloopException(
                    f"{option} requires "
                    "process_name='tt~'."
                )
            if self.n_loops != 2:
                raise pygloopException(
                    f"{option} requires n_loops=2."
                )
            if self.disable_integrated_uv_cts:
                raise pygloopException(
                    f"{option} requires integrated "
                    "UV counterterms to be enabled."
                )
            if self.dy_channel not in {(1, -1), (-1, 1), (0, 0)}:
                raise pygloopException(
                    f"{option} supports only channels "
                    "(1,-1), (-1,1), and (0,0)."
                )
        self.dy_check_generation_limits = bool(dy_check_generation_limits)
        try:
            self.dy_threshold_h_function = resolve_threshold_h_function(
                self.process_name, dy_threshold_h_function
            )
        except ValueError as error:
            raise pygloopException(str(error)) from error
        self.dy_parallel_graphs = max(1, int(dy_parallel_graphs))
        self.symmetrise_p1_p2 = bool(symmetrise_p1_p2)
        if (
            self.integrate_beams
            and self.n_loops == 2
            and self.process_name.lower() == "tt~"
            and not self.symmetrise_p1_p2
        ):
            raise pygloopException(
                "Two-loop hadronic ttbar integration requires "
                "symmetrise_p1_p2=True for every incoming channel."
            )
        self.dy_graph_index_offset = 0
        self.dy_emr_state_name = None
        self.dy_lambda_sq = float(dy_lambda_sq) if dy_lambda_sq is not None else None
        self.dy_mur_sq = float(dy_mur_sq) if dy_mur_sq is not None else None
        self.dy_pdf_set = (
            str(dy_pdf_set).strip() if dy_pdf_set is not None else None
        )
        self.dy_pdf_member = int(dy_pdf_member)
        self.dy_muf_sq = dy_muf_sq
        self.dy_pdf_luminosity_family = (
            DYPDFProvider.normalise_luminosity_family(
                dy_pdf_luminosity_family
            )
        )
        self._dy_pdf_provider: DYPDFProvider | None = None
        if self.dy_pdf_luminosity_family is not None:
            if self.dy_pdf_set is None or not self.integrate_beams:
                raise pygloopException(
                    "A DY PDF luminosity family requires beam-convoluted PDF "
                    "weighting."
                )
            family_channels = {
                "qqbar": {(1, -1), (-1, 1)},
                "qg": {(1, 0), (0, 1)},
                "gg": {(0, 0)},
            }
            if self.dy_channel not in family_channels[
                self.dy_pdf_luminosity_family
            ]:
                raise pygloopException(
                    f"DY channel {self.dy_channel} is incompatible with PDF "
                    f"luminosity family {self.dy_pdf_luminosity_family!r}."
                )
        if self.dy_pdf_set is not None:
            if not self.dy_pdf_set:
                raise pygloopException("The DY PDF set name cannot be empty.")
            if not self.integrate_beams:
                raise pygloopException(
                    "DY PDF weighting requires --dy-integrate-beams."
                )
            if not dy_channel_was_explicit:
                raise pygloopException(
                    "DY PDF weighting requires an explicit --dy-channel."
            )
            if self.dy_pdf_member < 0:
                raise pygloopException("The DY PDF member must be non-negative.")
            try:
                default_lambda_sq, default_mur_sq = DY_DEFAULT_LAMBDA_MUR_SQ[
                    self.n_loops
                ]
            except KeyError as exc:
                raise pygloopException(
                    "DY PDF weighting currently supports one or two loops."
                ) from exc
            self.dy_muf_sq = resolve_factorisation_scale_sq(
                dy_muf_sq,
                (
                    self.dy_lambda_sq
                    if self.dy_lambda_sq is not None
                    else default_lambda_sq
                ),
                self.dy_mur_sq if self.dy_mur_sq is not None else default_mur_sq,
            )
        self.dy_scheme_counterterm_sobol_power = int(
            dy_scheme_counterterm_sobol_power
        )
        self.dy_scheme_counterterm_replicas = int(
            dy_scheme_counterterm_replicas
        )
        self.dy_scheme_born_bundle = (
            str(dy_scheme_born_bundle).strip()
            if dy_scheme_born_bundle is not None
            else None
        )
        configured_born_bundles = []
        if self.dy_scheme_born_bundle:
            configured_born_bundles.append(self.dy_scheme_born_bundle)
        if dy_scheme_born_bundles is not None:
            configured_born_bundles.extend(
                str(bundle_name).strip()
                for bundle_name in dy_scheme_born_bundles
            )
        if any(not bundle_name for bundle_name in configured_born_bundles):
            raise pygloopException(
                "DY scheme-counterterm Born bundle names cannot be empty."
            )
        self.dy_scheme_born_bundles = tuple(
            dict.fromkeys(configured_born_bundles)
        )
        self.dy_scheme_alpha_s = float(dy_scheme_alpha_s)
        self.dy_scheme_counterterm_clip = (
            float(dy_scheme_counterterm_clip)
            if dy_scheme_counterterm_clip is not None
            else None
        )
        self._dy_scheme_born_compiled_bundle: DYCompiledBundle | None = None
        self._dy_scheme_born_compiled_bundles: dict[
            tuple[int, int],
            tuple[DYCompiledBundle, bool],
        ] = {}
        self.dy_scheme_counterterm_factor_was_explicit = (
            dy_scheme_counterterm_factor is not None
        )
        if dy_scheme_counterterm_factor is None:
            if (
                self.dy_msbar_scheme_counterterm
                and self.process_name.lower() == "dy"
                and self.n_loops == 1
                and self.dy_channel in {(1, -1), (-1, 1)}
            ):
                self.dy_scheme_counterterm_factor = QQBAR_SCHEME_COUNTERTERM_FACTOR
            elif (
                self.dy_msbar_scheme_counterterm
                and self.process_name.lower() == "tt~"
                and self.n_loops == 2
                and self.dy_channel in {(1, 0), (0, 1)}
            ):
                self.dy_scheme_counterterm_factor = QG_SCHEME_COUNTERTERM_FACTOR
            elif (
                self.dy_msbar_scheme_counterterm
                and self._dy_ttbar_qqbar_scheme_mode
            ):
                self.dy_scheme_counterterm_factor = 0.5 * self.dy_scheme_alpha_s
            else:
                self.dy_scheme_counterterm_factor = 1.0
        else:
            self.dy_scheme_counterterm_factor = float(
                dy_scheme_counterterm_factor
            )
        if self.dy_msbar_scheme_counterterm:
            is_one_loop_dy = (
                self.process_name.lower() == "dy"
                and self.n_loops == 1
            )
            is_two_loop_ttbar = (
                self.integrate_beams
                and self.process_name.lower() == "tt~"
                and self.n_loops == 2
            )
            is_partonic_two_loop_ttbar_qqbar = (
                not self.integrate_beams and self._dy_ttbar_qqbar_scheme_mode
            )
            is_partonic_two_loop_ttbar_gg = (
                not self.integrate_beams and self._dy_ttbar_gg_scheme_mode
            )
            is_partonic_two_loop_ttbar_qg = (
                not self.integrate_beams and self._dy_ttbar_qg_scheme_mode
            )
            if not (
                is_one_loop_dy
                or is_two_loop_ttbar
                or is_partonic_two_loop_ttbar_qqbar
                or is_partonic_two_loop_ttbar_gg
                or is_partonic_two_loop_ttbar_qg
            ):
                raise pygloopException(
                    "The DY MSbar scheme counterterm supports beam-convoluted "
                    "one-loop Drell-Yan and two-loop ttbar, plus fixed-partonic "
                    "one-loop Drell-Yan qqbar and two-loop ttbar qqbar, qg, "
                    "and gg."
                )
            supported_channels = {(1, 0), (0, 1), (1, -1), (-1, 1)}
            if self._dy_ttbar_gg_scheme_mode:
                supported_channels.add((0, 0))
            if self.dy_channel not in supported_channels:
                raise pygloopException(
                    "The DY MSbar scheme counterterm requires channel selection "
                    "compatible with the chosen process."
                )
            if self.integrate_beams:
                if self.dy_pdf_set is None or self.dy_muf_sq is None:
                    raise pygloopException(
                        "Hadronic DY MSbar scheme conversion requires PDF weighting."
                    )
            elif self.dy_pdf_set is not None or self.dy_muf_sq is not None:
                raise pygloopException(
                    "Partonic scheme conversion does not accept PDFs."
                )
            if (
                self._dy_partonic_one_loop_mode
                and self.dy_channel not in {(1, -1), (-1, 1)}
            ):
                raise pygloopException(
                    "Partonic one-loop DY scheme conversion currently supports "
                    "qqbar only."
                )
            if not self.dy_physical_normalisation:
                raise pygloopException(
                    "The DY MSbar scheme counterterm requires physical "
                    "normalisation."
                )
            if is_one_loop_dy and self.integrate_beams and (
                self.dy_q_min is None or self.dy_q_min <= 0.0
            ):
                raise pygloopException(
                    "The DY MSbar scheme counterterm requires a positive "
                    "--dy-q-min."
                )
            if not 1 <= self.dy_scheme_counterterm_sobol_power <= 30:
                raise pygloopException(
                    "DY scheme-counterterm Sobol power must be between 1 and 30."
                )
            if self.dy_scheme_counterterm_replicas < 2:
                raise pygloopException(
                    "DY scheme-counterterm integration requires at least two "
                    "replicas."
                )
            if not math.isfinite(self.dy_scheme_counterterm_factor):
                raise pygloopException(
                    "DY scheme-counterterm factor must be finite."
                )
            if (
                self.process_name.lower() == "tt~"
                and self.n_loops == 2
                and (
                    not math.isfinite(self.dy_scheme_alpha_s)
                    or self.dy_scheme_alpha_s <= 0.0
                )
            ):
                raise pygloopException(
                    "The two-loop ttbar scheme conversion requires a "
                    "finite, positive alpha_s."
                )
            if self.dy_scheme_counterterm_clip is not None and (
                not math.isfinite(self.dy_scheme_counterterm_clip)
                or self.dy_scheme_counterterm_clip <= 0.0
            ):
                raise pygloopException(
                    "DY scheme-counterterm clipping threshold must be finite "
                    "and positive."
                )
        self.dy_observable_muv = (
            float(dy_observable_muv) if dy_observable_muv is not None else None
        )
        # Keep only the user-supplied overrides here. Each compiled bundle
        # resolves omitted entries against its own recorded defaults.
        self.dy_runtime_parameters = dict(dy_runtime_parameters or {})
        self.dy_coupling_normalisation_applied = (
            self.process_name.lower() == "dy"
        )
        self.dy_coupling_normalisation_factor = 1.0
        self.dy_coupling_normalisation_convention = (
            DY_COUPLING_NORMALISATION_CONVENTION
            if self.dy_coupling_normalisation_applied
            else "none"
        )

        self.skip_ps_validation = bool(skip_ps_validation)
        if not self.skip_ps_validation:
            self.valide_ps_point()
        self.rotation_unstable_count: int = 0
        self.rotation_unstable_example: list[float] | None = None
        self.rotation_unstable_example_momentum_point: str | None = None
        self.rotation_hp_retry_count: int = 0
        self.rotation_hp_salvaged_count: int = 0
        self.rotation_hp_retry_example: list[float] | None = None
        self.rotation_hp_retry_example_momentum_point: str | None = None
        self.rotation_hp_retry_example_rel: float | None = None
        self.large_weight_hp_retry_count: int = 0
        self.large_weight_hp_salvaged_count: int = 0
        self.large_weight_unstable_count: int = 0
        self.large_weight_zeroed_count: int = 0
        self.large_weight_zeroed_signed_sum: float = 0.0
        self.large_weight_zeroed_abs_sum: float = 0.0
        self.large_weight_retry_example: list[float] | None = None
        self.large_weight_retry_example_momentum_point: str | None = None
        self.large_weight_retry_example_compiled_wgt: float | None = None
        self.large_weight_retry_example_arb_wgt: float | None = None
        self.stability_float_pair_accepted_count: int = 0
        self.stability_float_mismatch_retry_count: int = 0
        self.stability_float_nonfinite_retry_count: int = 0
        self.stability_hp_retry_count: int = 0
        self.stability_hp_accepted_count: int = 0
        self.stability_hp_escalation_count: int = 0
        self.stability_hp_escalation_accepted_count: int = 0
        self.stability_hp_disagreement_count: int = 0
        self.stability_hp_nonfinite_count: int = 0
        self.stability_hp_error_count: int = 0
        self.stability_hp_failure_example: list[float] | None = None
        self.stability_hp_failure_example_momentum_point: str | None = None
        self.stability_hp_failure_reason: str | None = None
        self.t_solver_float_failure_sample_count: int = 0
        self.t_solver_float_failed_term_count: int = 0
        self.t_solver_float_failed_surface_count: int = 0
        self.t_solver_hp_retry_count: int = 0
        self.t_solver_hp_salvaged_count: int = 0
        self.t_solver_hp_unresolved_count: int = 0
        self.t_solver_failure_surfaces: dict[str, int] = {}
        self.t_solver_failure_terms: dict[str, int] = {}
        self.soft_mirror_pair_count: int = 0
        self.soft_mirror_large_trigger_count: int = 0
        self.soft_mirror_hp_orbit_retry_count: int = 0
        self.soft_mirror_hp_orbit_salvaged_count: int = 0
        self.soft_mirror_hp_orbit_failure_count: int = 0
        self.soft_mirror_max_raw_side_wgt: float | None = None
        self.soft_mirror_max_raw_side_wgt_point: list[float] | None = None
        self.soft_mirror_max_raw_side: str | None = None
        self.soft_mirror_max_post_average_wgt: float | None = None
        self.soft_mirror_max_post_average_wgt_point: list[float] | None = None
        self.soft_mirror_min_residual_ratio: float | None = None
        self.soft_mirror_min_residual_ratio_point: list[float] | None = None
        self.soft_mirror_min_residual_ratio_sides: tuple[float, float] | None = None
        self.max_preclip_wgt: float | None = None
        self.max_preclip_wgt_point: list[float] | None = None
        self.max_preclip_wgt_momentum_point: str | None = None
        self.nan_weight_count: int = 0
        self.nan_weight_example: list[float] | None = None
        self.nan_weight_example_momentum_point: str | None = None
        self.nan_weight_rotated_count: int = 0
        self.nan_weight_rotated_example: list[float] | None = None
        self.nan_weight_rotated_example_momentum_point: str | None = None
        self.max_wgt: float | None = None
        self.max_wgt_point: list[float] | None = None
        self.max_wgt_jacobian: float | None = None
        self.max_wgt_momentum_point: str | None = None
        self.max_stable_wgt: float | None = None
        self.max_stable_wgt_point: list[float] | None = None
        self.max_stable_wgt_jacobian: float | None = None
        self.max_stable_wgt_momentum_point: str | None = None

        self.e_cm = math.sqrt(abs((self.ps_point[0] + self.ps_point[1]).squared()))

        if toml_config_path is None:
            toml_config_path = pjoin(CONFIGS_FOLDER, self.name, "generate.toml")
        self.toml_config_path = toml_config_path

        if runtime_toml_config_path is None:
            runtime_toml_config_path = pjoin(CONFIGS_FOLDER, self.name, "runtime.toml")
        self.runtime_toml_config_path = runtime_toml_config_path

        self.gl_worker = None
        self._dy_ghost_particle_names: frozenset[str] | None = None
        self.clean = clean
        if not self.skip_gl_worker_init:
            gl_states_folder = pjoin(GAMMALOOP_STATES_FOLDER, self.name)
            if os.path.exists(gl_states_folder):
                if clean:
                    logger.info(
                        f"Removing existing GammaLoop state in {Colour.GREEN}{gl_states_folder}{Colour.END}"
                    )  # nopep8
                    shutil.rmtree(gl_states_folder)
                else:
                    logger.info(
                        f"Reusing existing GammaLoop state in {Colour.GREEN}{gl_states_folder}{Colour.END}"
                    )  # nopep8

            logger_level = logger.getEffectiveLevel()
            if logger_level <= logging.DEBUG:
                gl_log_level = LogLevel.Debug
            elif logger_level <= logging.INFO:
                gl_log_level = LogLevel.Info
            elif logger_level <= logging.WARNING:
                gl_log_level = LogLevel.Warn
            elif logger_level <= logging.ERROR:
                gl_log_level = LogLevel.Error
            else:
                gl_log_level = LogLevel.Off

            logger.info(
                f"Initializing GammaLoop API (git {Colour.BLUE}{git_version}{Colour.END}) for process {Colour.GREEN}{self.name}{Colour.END}"
            )  # nopep8
            self.gl_worker = GammaLoopAPI(
                pjoin(PYGLOOP_FOLDER, "outputs", "gammaloop_states", self.name),
                # log_file_name=self.name,
                # log_level=gl_log_level,
            )
            self.set_log_level(logger_level)

            logger.info(
                f"Setting gammaloop starting configuration from toml file {Colour.BLUE}{toml_config_path}{Colour.END}."
            )
            self.gl_worker.run(f"set global file {toml_config_path}")  # nopep8
            self.setup_gl_worker()

            amplitudes, cross_sections = self.gl_worker.list_outputs()
            if len(amplitudes) == 0 and len(cross_sections) == 0:
                logger.info("No output yet in the GammaLoop state loaded.")
            if len(amplitudes) > 0:
                logger.info(
                    f"Available amplitudes: {Colour.GREEN}{pformat(amplitudes)}{Colour.END}"
                )
            if len(cross_sections) > 0:
                logger.info(
                    f"Available cross sections: {Colour.GREEN}{pformat(cross_sections)}{Colour.END}"
                )

            logger.info(f"Setting runtime configuration for all outputs from toml file: {Colour.BLUE}{runtime_toml_config_path}{Colour.END}.")  # fmt: off
            for output_name, output_id in amplitudes.items():
                # Currently bugged: not all functionalities available on integrands not yet generated
                if "_generated_graphs" in output_name:
                    continue
                self.gl_worker.run(f"set process -p {output_id} -i {output_name} file {self.runtime_toml_config_path}")  # fmt: off
                self.set_sample_point(
                    self.ps_point, self.helicities, str(output_id), output_name
                )

            self.save_state()
        # Cache some quantities for performance
        self.cache: dict[str, Any] = {}

        self.compiled_bundle: DYCompiledBundle | None = None
        if self.load_compiled_bundle:
            integrand_name = self.get_integrand_name()
            bundle_processes = [self.process_name]
            if self.name not in bundle_processes:
                bundle_processes.append(self.name)
            for bundle_process in bundle_processes:
                bundle_dir = pjoin(EVALUATORS_FOLDER, bundle_process, integrand_name)
                bundle_metadata = pjoin(bundle_dir, DYCompiledBundle.METADATA_FILE)
                if not os.path.exists(bundle_metadata):
                    continue
                try:
                    self.compiled_bundle = DYCompiledBundle.load(
                        bundle_process, integrand_name
                    )
                    logger.info(
                        f"Loaded compiled DY bundle from {Colour.GREEN}{bundle_dir}{Colour.END}"
                    )
                    break
                except Exception as e:
                    logger.warning(
                        f"Failed loading compiled DY bundle from {bundle_dir}: {e}"
                    )

        resolved_runtime_parameters: Mapping[str, float | Decimal] = {}
        if self.compiled_bundle is not None:
            # Bundle defaults are the lowest-precedence runtime source.  Keep
            # thresholds and PDF-scale bookkeeping aligned with the exact same
            # resolved values used by the evaluator, even when an integration
            # card carries different generation defaults.
            resolved_runtime_parameters = (
                self.compiled_bundle.resolve_runtime_parameters(
                    self.dy_runtime_parameters
                )
            )
            if resolved_runtime_parameters:
                self.m_top = float(resolved_runtime_parameters["m_top"])
                self.dy_observable_muv = float(
                    resolved_runtime_parameters["muv"]
                )
                self.dy_lambda_sq = float(
                    resolved_runtime_parameters["lambda_sq"]
                )
                self.dy_mur_sq = float(
                    resolved_runtime_parameters["mur_sq"]
                )
                if self.dy_pdf_set is not None and dy_muf_sq is None:
                    self.dy_muf_sq = resolve_factorisation_scale_sq(
                        None,
                        self.dy_lambda_sq,
                        self.dy_mur_sq,
                    )

        if self.dy_coupling_normalisation_applied:
            alpha_s = resolved_runtime_parameters.get(
                "alpha_s", self.dy_runtime_parameters.get("alpha_s", "0.118")
            )
            alpha_ew_inverse = resolved_runtime_parameters.get(
                "alpha_ew_inverse",
                self.dy_runtime_parameters.get("alpha_ew_inverse", "132.507"),
            )
            pi_value = resolved_runtime_parameters.get(
                "pi", self.dy_runtime_parameters.get("pi", PI_DECIMAL)
            )
            self.dy_coupling_normalisation_factor = float(
                dy_coupling_normalisation_factor(
                    alpha_s,
                    alpha_ew_inverse,
                    pi_value,
                )
            )

        logger.setLevel(start_logger_level)

    def __deepcopy__(self, _memo) -> DY:
        copied_self = DY(
            self.m_top,
            self.m_higgs,
            copy.deepcopy(self.ps_point, _memo),
            copy.deepcopy(self.helicities, _memo),
            self.n_loops,
            self.toml_config_path,
            self.runtime_toml_config_path,
            copy.deepcopy(self.final_state, _memo),
            self.process_name,
            dy_channel=self.dy_channel,
            clean=False,
            logger_level=logging.CRITICAL,
            skip_ps_validation=self.skip_ps_validation,
            integrate_beams=self.integrate_beams,
            dy_z_bin=self.dy_z_bin,
            dy_q_min=self.dy_q_min,
            dy_q_max=self.dy_q_max,
            dy_physical_normalisation=self.dy_physical_normalisation,
            dy_integrated_leptonic_phase_space=(
                self.dy_integrated_leptonic_phase_space
            ),
            disable_integrated_uv_cts=self.disable_integrated_uv_cts,
            dy_top_self_energy_os_subtraction=(
                _dy_top_self_energy_legacy_forwarding(self)
            ),
            dy_top_self_energy_renormalisation=(
                _dy_top_self_energy_mode(self)
            ),
            dy_check_generation_limits=self.dy_check_generation_limits,
            dy_threshold_h_function=self.dy_threshold_h_function,
            dy_include_disabled_threshold_counterterms=(
                self.dy_include_disabled_threshold_counterterms
            ),
            dy_parallel_graphs=self.dy_parallel_graphs,
            symmetrise_p1_p2=self.symmetrise_p1_p2,
            dy_fallback_precision=self.dy_fallback_precision,
            dy_lambda_sq=self.dy_lambda_sq,
            dy_mur_sq=self.dy_mur_sq,
            dy_pdf_set=self.dy_pdf_set,
            dy_pdf_member=self.dy_pdf_member,
            dy_muf_sq=self.dy_muf_sq,
            dy_pdf_luminosity_family=self.dy_pdf_luminosity_family,
            dy_msbar_scheme_counterterm=self.dy_msbar_scheme_counterterm,
            dy_decoupling=self.dy_decoupling,
            dy_scheme_counterterm_sobol_power=(
                self.dy_scheme_counterterm_sobol_power
            ),
            dy_scheme_counterterm_replicas=self.dy_scheme_counterterm_replicas,
            dy_scheme_counterterm_factor=(
                self.dy_scheme_counterterm_factor
                if self.dy_scheme_counterterm_factor_was_explicit
                else None
            ),
            dy_scheme_born_bundle=self.dy_scheme_born_bundle,
            dy_scheme_born_bundles=self.dy_scheme_born_bundles,
            dy_scheme_alpha_s=self.dy_scheme_alpha_s,
            dy_scheme_counterterm_clip=self.dy_scheme_counterterm_clip,
            dy_observable_muv=self.dy_observable_muv,
            dy_runtime_parameters=self.dy_runtime_parameters,
            skip_gl_worker_init=self.skip_gl_worker_init,
            load_compiled_bundle=self.load_compiled_bundle,
            process_basename=self.name,
        )
        return copied_self

    def builder_inputs(self) -> dict[str, Any]:
        return {
            "m_top": self.m_top,
            "m_higgs": self.m_higgs,
            "ps_point": self.ps_point,
            "helicities": self.helicities,
            "n_loops": self.n_loops,
            "toml_config_path": self.toml_config_path,
            "runtime_toml_config_path": self.runtime_toml_config_path,
            "final_state": copy.deepcopy(self.final_state),
            "process_name": self.process_name,
            "diagrams": copy.deepcopy(self.diagrams),
            "dy_channel": self.dy_channel,
            "skip_ps_validation": self.skip_ps_validation,
            "integrate_beams": self.integrate_beams,
            "dy_z_bin": self.dy_z_bin,
            "dy_q_min": self.dy_q_min,
            "dy_q_max": self.dy_q_max,
            "dy_physical_normalisation": self.dy_physical_normalisation,
            "dy_integrated_leptonic_phase_space": (
                self.dy_integrated_leptonic_phase_space
            ),
            "external_gluon_polarisation": self.external_gluon_polarisation,
            "disable_integrated_uv_cts": self.disable_integrated_uv_cts,
            "dy_top_self_energy_os_subtraction": (
                _dy_top_self_energy_legacy_forwarding(self)
            ),
            "dy_top_self_energy_renormalisation": (
                _dy_top_self_energy_mode(self)
            ),
            "dy_check_generation_limits": self.dy_check_generation_limits,
            "dy_threshold_h_function": self.dy_threshold_h_function,
            "dy_include_disabled_threshold_counterterms": (
                self.dy_include_disabled_threshold_counterterms
            ),
            "dy_parallel_graphs": self.dy_parallel_graphs,
            "symmetrise_p1_p2": self.symmetrise_p1_p2,
            "dy_fallback_precision": self.dy_fallback_precision,
            "dy_lambda_sq": self.dy_lambda_sq,
            "dy_mur_sq": self.dy_mur_sq,
            "dy_pdf_set": self.dy_pdf_set,
            "dy_pdf_member": self.dy_pdf_member,
            "dy_muf_sq": self.dy_muf_sq,
            "dy_pdf_luminosity_family": self.dy_pdf_luminosity_family,
            "dy_msbar_scheme_counterterm": self.dy_msbar_scheme_counterterm,
            "dy_decoupling": self.dy_decoupling,
            "dy_scheme_counterterm_sobol_power": (
                self.dy_scheme_counterterm_sobol_power
            ),
            "dy_scheme_counterterm_replicas": (
                self.dy_scheme_counterterm_replicas
            ),
            "dy_scheme_counterterm_factor": self.dy_scheme_counterterm_factor,
            "dy_scheme_born_bundle": self.dy_scheme_born_bundle,
            "dy_scheme_born_bundles": list(self.dy_scheme_born_bundles),
            "dy_scheme_alpha_s": self.dy_scheme_alpha_s,
            "dy_scheme_counterterm_clip": self.dy_scheme_counterterm_clip,
            "dy_observable_muv": self.dy_observable_muv,
            "dy_runtime_parameters": copy.deepcopy(self.dy_runtime_parameters),
            "process_basename": self.name,
        }

    def process_uses_z(self) -> bool:
        return self.process_name.lower() == "dy"

    def dy_observable_params(
        self,
        default_lambda_sq: float,
        default_muv: float,
        default_mur_sq: float,
    ) -> dict[str, float]:
        return {
            # A few lightweight graph-construction tests instantiate DY via
            # ``__new__``. Keep their historical default while normal process
            # instances always provide the configured mass.
            "m_top": getattr(self, "m_top", 173.0),
            "zmin": 0.0,
            "zmax": 1.0,
            "Lambdasq": (
                self.dy_lambda_sq
                if self.dy_lambda_sq is not None
                else default_lambda_sq
            ),
            "mUV": (
                self.dy_observable_muv
                if self.dy_observable_muv is not None
                else default_muv
            ),
            "mursq": self.dy_mur_sq if self.dy_mur_sq is not None else default_mur_sq,
        }

    def sampled_uses_z(self, integrand_implementation: dict[str, Any] | str) -> bool:
        integrand_implementation = self._normalize_integrand_implementation(
            integrand_implementation
        )
        return (
            self.process_uses_z()
            and integrand_implementation.get("integrand_type") == "zenos"
        )

    def sampled_uses_beam_fractions(
        self, integrand_implementation: dict[str, Any] | str
    ) -> bool:
        integrand_implementation = self._normalize_integrand_implementation(
            integrand_implementation
        )
        return (
            self.integrate_beams
            and integrand_implementation.get("integrand_type") == "zenos"
        )

    def integration_dimension(
        self, integrand_implementation: dict[str, Any] | str
    ) -> int:
        return (
            3 * self.n_loops
            + int(self.sampled_uses_z(integrand_implementation))
            + 2 * int(self.sampled_uses_beam_fractions(integrand_implementation))
        )

    @staticmethod
    def _channel_selector_from_multi_channeling(
        multi_channeling: bool | int,
    ) -> int | None:
        if isinstance(multi_channeling, bool):
            return None
        return int(multi_channeling)

    def graph_channel_names(
        self, integrand_implementation: dict[str, Any] | str
    ) -> list[str]:
        integrand_implementation = self._normalize_integrand_implementation(
            integrand_implementation
        )
        if integrand_implementation.get("integrand_type") != "zenos":
            return []
        if self.compiled_bundle is None:
            return []
        return self.compiled_bundle.graph_channel_names()

    def _integration_graph_channel_indices(
        self,
        integrand_implementation: dict[str, Any] | str,
        selectors: list[str] | None,
    ) -> list[int] | None:
        implementation = self._normalize_integrand_implementation(
            integrand_implementation
        )
        if (
            selectors is None
            and implementation.get("dy_grouped_soft_pair") is None
            and implementation.get("dy_grouped_soft_pair_transform") is None
        ):
            return None
        channel_names = self.graph_channel_names(integrand_implementation)
        if len(channel_names) == 0:
            raise pygloopException(
                "DY integration graph selection requires a compiled zenos "
                "bundle with graph-grouped evaluators."
            )

        selected_indices = (
            list(range(len(channel_names)))
            if selectors is None
            else self._resolve_graph_channel_selectors(channel_names, selectors)
        )
        grouped_pair = self._grouped_soft_pair(integrand_implementation)
        if grouped_pair is None:
            return None if selectors is None else selected_indices

        first, second, _transform = grouped_pair
        selected_pair = (first in selected_indices, second in selected_indices)
        if selected_pair[0] != selected_pair[1]:
            raise pygloopException(
                "DY grouped soft-pair integration must select both paired "
                "graphs or neither of them."
            )
        if not selected_pair[0]:
            return selected_indices

        collapsed: list[int] = []
        pair_inserted = False
        for index in selected_indices:
            if index in (first, second):
                if not pair_inserted:
                    collapsed.append(DY_GROUPED_SOFT_PAIR_SELECTOR)
                    pair_inserted = True
                continue
            collapsed.append(index)
        return collapsed

    def _resolve_graph_channel_selectors(
        self,
        channel_names: Sequence[str],
        selectors: Sequence[str],
    ) -> list[int]:
        """Resolve native, numeric, or source graph selectors in card order."""

        aliases = {name: index for index, name in enumerate(channel_names)}
        if self.diagrams:
            if len(self.diagrams) != len(channel_names):
                raise pygloopException(
                    "Cannot map source graph names to compiled DY channels: "
                    f"--diagrams contains {len(self.diagrams)} names but the "
                    f"bundle contains {len(channel_names)} graph channels. "
                    "Use native graph_N channel names instead."
                )
            aliases.update({
                str(graph_name): index
                for index, graph_name in enumerate(self.diagrams)
            })

        selected_indices: list[int] = []
        for raw_selector in selectors:
            selector = str(raw_selector)
            if selector in aliases:
                channel_index = aliases[selector]
            elif selector.isdigit():
                channel_index = int(selector)
                if channel_index >= len(channel_names):
                    raise pygloopException(
                        f"DY integration graph channel {channel_index} out of "
                        f"range for {len(channel_names)} channels."
                    )
            else:
                known_aliases = list(channel_names)
                if self.diagrams and len(self.diagrams) == len(channel_names):
                    known_aliases.extend(str(name) for name in self.diagrams)
                raise pygloopException(
                    f"Unknown DY integration graph '{selector}'. Known graph "
                    f"selectors: {', '.join(known_aliases)}"
                )
            if channel_index not in selected_indices:
                selected_indices.append(channel_index)

        if len(selected_indices) == 0:
            raise pygloopException("DY integration graph selection cannot be empty.")
        return selected_indices

    def _grouped_soft_pair(
        self, integrand_implementation: Mapping[str, Any] | str
    ) -> tuple[int, int, str] | None:
        implementation = self._normalize_integrand_implementation(
            integrand_implementation
        )
        selectors = implementation.get("dy_grouped_soft_pair")
        transform = implementation.get("dy_grouped_soft_pair_transform")
        if selectors is None and transform is None:
            return None
        if selectors is None or transform is None:
            raise pygloopException(
                "DY grouped soft-pair graphs and transform must be configured "
                "together."
            )
        if implementation.get("integrand_type") != "zenos":
            raise pygloopException(
                "DY grouped soft-pair integration requires the zenos integrand."
            )
        if str(transform) != "invert":
            raise pygloopException(
                f"Unsupported DY grouped soft-pair transform {transform!r}."
            )
        channel_names = self.graph_channel_names(implementation)
        pair = self._resolve_graph_channel_selectors(
            channel_names, [str(selector) for selector in selectors]
        )
        if len(pair) != 2:
            raise pygloopException(
                "DY grouped soft-pair integration requires exactly two "
                "distinct graph channels."
            )
        if not getattr(self, "symmetrise_p1_p2", False):
            raise pygloopException(
                "DY grouped soft-pair integration requires p1<->p2 "
                "symmetrisation."
            )
        return pair[0], pair[1], str(transform)

    def _integration_graph_channel_name(
        self,
        integrand_implementation: Mapping[str, Any] | str,
        selector: int,
    ) -> str:
        if selector != DY_GROUPED_SOFT_PAIR_SELECTOR:
            names = self.graph_channel_names(integrand_implementation)
            if selector < 0 or selector >= len(names):
                raise pygloopException(
                    f"DY graph channel {selector} is out of range for "
                    f"{len(names)} channels."
                )
            return names[selector]
        grouped_pair = self._grouped_soft_pair(integrand_implementation)
        if grouped_pair is None:
            raise pygloopException(
                "The grouped soft-pair channel was selected without a card "
                "mapping."
            )
        first, second, transform = grouped_pair
        names = self.graph_channel_names(integrand_implementation)
        return f"{names[first]}+{names[second]}[{transform}]"

    def _grouped_soft_pair_coordinates(
        self,
        coordinates: Sequence[float],
        *,
        transformed_member: bool,
    ) -> list[float]:
        mapped = list(coordinates)
        if not transformed_member:
            return mapped
        if len(mapped) < 3:
            raise pygloopException(
                "DY grouped soft-pair inversion requires three spherical "
                "loop-momentum coordinates."
            )
        mapped[2] = 1.0 - mapped[2]
        mapped[1] = (mapped[1] + 0.5) % 1.0
        return mapped

    @staticmethod
    def _build_symbolica_discrete_integrator(
        n_dim: int,
        n_channels: int,
        *,
        train_on_avg: bool = False,
    ) -> NumericalIntegrator:
        return NumericalIntegrator.discrete(
            [
                NumericalIntegrator.continuous(n_dim)
                for _ in range(n_channels)
            ],
            train_on_avg=train_on_avg,
        )

    @staticmethod
    def _symbolica_sample_weight(sample: Sample) -> float:
        # Symbolica stores cumulative weights for nested integrators.  For a
        # discrete-of-continuous integrator, weights[0] already contains the
        # complete inverse sampling density (including the selected discrete
        # channel and its continuous child).  Later entries are child-layer
        # cumulative weights, not independent factors; multiplying them would
        # double-count the adapted continuous weight.
        if not sample.weights:
            raise pygloopException("Symbolica sample has no integration weight.")
        return float(sample.weights[0])

    def _symbolica_graph_channel_batch_estimates(
        self,
        graph_channel_names: list[str],
        samples: list[Sample],
        sample_values: list[float],
    ) -> list[float]:
        if len(samples) == 0:
            return [0.0 for _graph_channel_name in graph_channel_names]

        channel_totals = [0.0 for _graph_channel_name in graph_channel_names]
        for sample, sample_value in zip(samples, sample_values, strict=True):
            channel_index = int(sample.d[0])
            if channel_index < 0 or channel_index >= len(graph_channel_names):
                raise pygloopException(
                    f"Sample discrete channel {channel_index} out of range for "
                    f"{len(graph_channel_names)} graph channels."
                )
            channel_totals[channel_index] += float(
                sample_value
            ) * self._symbolica_sample_weight(sample)

        normalization = float(len(samples))
        return [channel_total / normalization for channel_total in channel_totals]

    def _symbolica_graph_channel_raw_estimates(
        self,
        integrator: NumericalIntegrator,
        n_dim: int,
        graph_channel_names: list[str],
        samples: list[Sample],
        sample_values: list[float],
        continuous_learning_rate: float,
        discrete_learning_rate: float,
    ) -> list[float]:
        if len(graph_channel_names) == 0:
            return []

        grid_state = integrator.export_grid(False)
        raw_channel_estimates: list[float] = []
        for channel_index, _graph_channel_name in enumerate(graph_channel_names):
            shadow_integrator = self._build_symbolica_discrete_integrator(
                n_dim, len(graph_channel_names)
            )
            shadow_integrator.import_grid(grid_state)
            shadow_values = [
                sample_value if int(sample.d[0]) == channel_index else 0.0
                for sample, sample_value in zip(samples, sample_values, strict=True)
            ]
            shadow_integrator.add_training_samples(samples, shadow_values)
            channel_avg, _channel_err, _channel_chi_sq = shadow_integrator.update(
                continuous_learning_rate=continuous_learning_rate,
                discrete_learning_rate=discrete_learning_rate,
            )
            raw_channel_estimates.append(float(channel_avg))

        return raw_channel_estimates

    def _symbolica_graph_channel_contributions(
        self,
        graph_channel_names: list[str],
        graph_channel_observers: list[NumericalIntegrator],
        samples: list[Sample],
        sample_values: list[float],
        continuous_learning_rate: float,
        discrete_learning_rate: float,
    ) -> list[tuple[str, float, float, int]]:
        n_channels = len(graph_channel_names)
        if n_channels == 0:
            return []
        if len(graph_channel_observers) != n_channels:
            raise pygloopException(
                f"Expected {n_channels} graph-channel observers, got "
                f"{len(graph_channel_observers)}."
            )

        channel_counts = [0 for _graph_channel_name in graph_channel_names]
        for sample, sample_value in zip(samples, sample_values, strict=True):
            channel_index = int(sample.d[0])
            if channel_index < 0 or channel_index >= n_channels:
                raise pygloopException(
                    f"Sample discrete channel {channel_index} out of range for "
                    f"{n_channels} graph channels."
                )
            channel_counts[channel_index] += 1

        channel_contributions: list[tuple[str, float, float, int]] = []
        for channel_index, graph_channel_name in enumerate(graph_channel_names):
            masked_values = [
                sample_value if int(sample.d[0]) == channel_index else 0.0
                for sample, sample_value in zip(samples, sample_values, strict=True)
            ]
            graph_channel_observers[channel_index].add_training_samples(
                samples, masked_values
            )
            channel_avg, channel_err, _channel_chi_sq = graph_channel_observers[
                channel_index
            ].update(
                continuous_learning_rate=continuous_learning_rate,
                discrete_learning_rate=discrete_learning_rate,
            )
            channel_contributions.append((
                graph_channel_name,
                float(channel_avg),
                float(channel_err),
                channel_counts[channel_index],
            ))
        return channel_contributions

    @staticmethod
    def _symbolica_graph_channel_report(
        graph_channel_contributions: list[tuple[str, float, float, int]],
        total_avg: float,
    ) -> str:
        lines = ["| > Graph-channel Symbolica observer estimates:"]
        for (
            graph_channel_name,
            contribution,
            error,
            n_channel_samples,
        ) in graph_channel_contributions:
            lines.append(
                f"| >   {graph_channel_name:<12}: {contribution:.16e} "
                f"+/- {error:.2e} [{n_channel_samples} selected]"
            )
        channel_sum = sum(
            contribution
            for _name, contribution, _error, _count in graph_channel_contributions
        )
        lines.append(f"| >   {'sum':<12}: {channel_sum:.16e}")
        lines.append(f"| >   {'total':<12}: {float(total_avg):.16e}")
        difference = channel_sum - float(total_avg)
        lines.append(f"| >   {'difference':<12}: {difference:.16e}")
        if not math.isclose(difference, 0.0, abs_tol=1e-12, rel_tol=1e-10):
            lines.append("| >   warning     : graph-channel sum differs from total")
        return "\n".join(lines)

    def set_log_level(self, level) -> None:
        if self.gl_worker is None:
            return
        if level <= logging.DEBUG:
            lvl = "debug"
        elif level <= logging.INFO:
            lvl = "info"
        elif level <= logging.WARNING:
            lvl = "warn"
        elif level <= logging.ERROR:
            lvl = "error"
        else:
            lvl = "off"
        self.gl_worker.run(
            f"set global kv global.logfile_directive='gammalooprs={lvl},{lvl}'"
        )
        self.gl_worker.run(
            f"set global kv global.display_directive='gammalooprs={lvl},{lvl}'"
        )

    def set_sample_point(
        self,
        momenta: list[LorentzVector],
        helicities: list[int],
        process_id: str | None,
        integrand_name: str | None,
    ) -> None:
        if process_id is None and integrand_name is None:
            card = "default-runtime"
        else:
            card = f"process -p {process_id} -i {integrand_name}"

        momenta_in = list(momenta)
        # Place dependent last to ensure that the incoming are exactly longitudinal
        # so that polarization vector definitions don't suddenly jump
        momenta_in[-1] = "dependent"  # type: ignore
        # fmt: off
        momenta_str = "[" + ",".join("[" + ",".join(f"{vi:.16e}" for vi in v.to_list()) + "]" if not isinstance(v, str) else f'"{v}"' for v in momenta_in) + "]"
         # fmt: on
        helicities_str = "[" + ",".join(f"{h:+d}" for h in helicities) + "]"

        kinematics_set_command = f'set {card} kv kinematics.externals={{"type":"constant","data":{{"momenta":{momenta_str},"helicities":{helicities_str}}}}}'  # fmt: off
        logger.debug("Setting kinematic point with:\n%s", kinematics_set_command)
        # self.gl_worker.run(kinematics_set_command)

    def set_model(self) -> None:
        self.gl_worker.run("import model sm-default.json")
        # self.gl_worker.run("set model MT={{re:{:.16f},im:0.0}}".format(self.m_top))
        # self.gl_worker.run("set model MH={{re:{:.16f},im:0.0}}".format(self.m_higgs))
        # self.gl_worker.run("set model WT={re:0.0,im:0.0}")
        # self.gl_worker.run("set model WH={re:0.0,im:0.0}")
        # self.gl_worker.run("set model ymt={{re:{:.16f},im:0.0}}".format(self.m_top))

    def setup_gl_worker(self) -> None:
        self.set_model()
        # Set default kinematics
        self.set_sample_point(self.ps_point, self.helicities, None, None)
        # print(dir(self.gl_worker))
        # self.gl_worker.run("save state -o")

    def _resolve_dy_ghost_particle_names(self) -> frozenset[str]:
        cached = getattr(self, "_dy_ghost_particle_names", None)
        if cached is not None:
            return cached
        if self.gl_worker is None:
            raise pygloopException(
                "Generated DY graph terms require GammaLoop ghost-particle metadata."
            )
        names = ghost_particle_names_from_model_metadata(self.gl_worker.get_model())
        self._dy_ghost_particle_names = names
        return names

    def save_state(self) -> None:
        self.gl_worker.run("save state -o")

    def get_color_projector(self) -> Expression:
        return E(
            "spenso::g(spenso::cof(3,gammalooprs::hedge(1)),spenso::dind(spenso::cof(3,gammalooprs::hedge(3))))*spenso::g(spenso::cof(3,gammalooprs::hedge(0)),spenso::dind(spenso::cof(3,gammalooprs::hedge(2))))"
        )

    def get_spin_projector(self) -> Expression:
        return E(
            "spenso::gamma(spenso::bis(4,gammalooprs::hedge(0)),spenso::bis(4,gammalooprs::hedge(2)),spenso::mink(4,mu))*gammalooprs::Q(0,spenso::mink(4,mu))*spenso::gamma(spenso::bis(4,gammalooprs::hedge(3)),spenso::bis(4,gammalooprs::hedge(1)),spenso::mink(4,nu))*gammalooprs::Q(1,spenso::mink(4,nu))"
        )

    #    def process_1L_generated_graphs(self, graphs: DYDotGraphs) -> DYDotGraphs:
    #        processed_graphs = DYDotGraphs()
    #
    #        filtered_graphs = DYDotGraphs()
    #        filtered_graphs.extend(
    #            copy.deepcopy(graphs.filter_particle_definition(["t", "t~"]))
    #        )
    #        filtered_graphs.extend(
    #            copy.deepcopy(graphs.filter_particle_definition(["t", "t"]))
    #        )
    #        filtered_graphs.extend(
    #            copy.deepcopy(graphs.filter_particle_definition(["t~", "t~"]))
    #        )
    #
    #        print("filtered graphs: ", len(graphs.filter_particle_definition(["t", "t~"])))
    #
    #        processor = EMRIntegrandConstructor([], "DY", 1)
    #        loop_processor = LoopIntegrandConstructor([], "DY", 1)
    #
    #        for graph in filtered_graphs:
    #            g = copy.deepcopy(graph)
    #            print("generator graph")
    #            print(g.dot)
    #            vacuum_g = g.get_vacuum_graph()
    #            print("vacuum graph")
    #            print(vacuum_g.dot)
    #            _cuts = vacuum_g.get_cutkosky_cuts()
    #            routed_graphs = vacuum_g.cut_graphs_with_routing_leading_virtuality(
    #                [], ["t", "t~"]
    #            )
    #            routed_graphs.extend(
    #                vacuum_g.cut_graphs_with_routing_leading_virtuality([], ["t", "t"])
    #            )
    #            routed_graphs.extend(
    #                vacuum_g.cut_graphs_with_routing_leading_virtuality([], ["t~", "t~"])
    #            )
    #
    #            for gg in routed_graphs:
    #                # print(gg[3])
    #                processed_graphs.append(gg[3])
    #
    #        print("n routed:", len(processed_graphs))
    #        return processed_graphs

    def process_1L_generated_graphs(self, graphs: DYDotGraphs) -> DYDotGraphs:
        final_state = copy.deepcopy(self.final_state)
        process_name = self.process_name
        n_loops = self.n_loops
        channel = getattr(self, "dy_channel", (1, -1))

        processed_graphs = DYDotGraphs()

        filtered_graphs = DYDotGraphs()
        # filtered_graphs.extend(
        #    copy.deepcopy(graphs.filter_particle_definition(final_state))
        # )
        filtered_graphs.extend(copy.deepcopy(graphs))

        print("############################")
        print("Filtered graphs: ", len(filtered_graphs))
        print("############################")

        processor = EMRIntegrandConstructor(
            [], process_name, n_loops, state_name=self.dy_emr_state_name
        )
        loop_processor = LoopIntegrandConstructor(
            [],
            process_name,
            n_loops,
            channel=channel,
            external_gluon_polarisation=self.external_gluon_polarisation,
            disable_integrated_uv_cts=self.disable_integrated_uv_cts,
            top_self_energy_os_subtraction=(
                _dy_top_self_energy_legacy_forwarding(self)
            ),
            top_self_energy_renormalisation=_dy_top_self_energy_mode(self),
            threshold_h_function=getattr(
                self, "dy_threshold_h_function", None
            ),
            include_disabled_threshold_counterterms=getattr(
                self, "dy_include_disabled_threshold_counterterms", False
            ),
            symmetrise_p1_p2=self.symmetrise_p1_p2,
        )

        all_routed_integrands = []
        all_evaluators = []

        for graph_index, graph in enumerate(filtered_graphs):
            vac_g = canonicalise_vacuum_graph(copy.deepcopy(graph))

            vacuum_g = VacuumDotGraph(copy.deepcopy(vac_g.dot))

            # _cuts = vacuum_g.get_cutkosky_cuts()
            routed_graphs = vacuum_g.cut_graphs_with_routing_leading_virtuality(
                [], final_state
            )
            indexed_routed_graphs = [
                (routed_graph_index, routed_graph)
                for routed_graph_index, routed_graph in enumerate(routed_graphs)
                if _strip_quotes(str(routed_graph[3].get("particle_channel")))
                == str(channel)
            ]
            if self.symmetrise_p1_p2:
                symmetrised_graphs = filter_symmetrised_p1_p2_routed_cuts(
                    [routed_graph for _, routed_graph in indexed_routed_graphs]
                )
                retained_graph_ids = {id(graph) for graph in symmetrised_graphs}
                indexed_routed_graphs = [
                    (routed_graph_index, routed_graph)
                    for routed_graph_index, routed_graph in indexed_routed_graphs
                    if id(routed_graph) in retained_graph_ids
                ]

            print("############################")
            print("Routed graphs: ", len(routed_graphs))
            print("############################")

            routed_integrands = []
            evaluators = []

            for routed_graph_index, gg in indexed_routed_graphs:
                processed_graphs.append(gg[3])
                cut_graph = deepcopy(routed_cut_graph(gg[3], gg[0], gg[1], gg[2]))
                # print(cut_graph.graph.get_name())
                print(cut_graph.graph)
                term_integrands = loop_processor.get_integrand(deepcopy(cut_graph))
                graph_ghost_loop_count = (
                    closed_ghost_loop_count(
                        graph,
                        self._resolve_dy_ghost_particle_names(),
                    )
                    if term_integrands
                    else 0
                )

                if self.dy_check_generation_limits:
                    routed_integrands.extend(deepcopy(term_integrands))

                observable_params = self.dy_observable_params(
                    default_lambda_sq=2,
                    default_muv=1,
                    default_mur_sq=1,
                )

                for term_index, term_integrand in enumerate(term_integrands):
                    evaluator = evaluate_integrand(
                        n_loops,
                        process_name,
                        deepcopy(term_integrand),
                        n_hornerscheme_iterations=1000,
                        n_cpe_iterations=10000,
                        observable_params=observable_params,
                    )
                    evaluator.compiled_name = (
                        f"graph_{graph_index}_cut_{routed_graph_index}"
                        f"_term_{term_index}_integrand"
                    )
                    evaluator.source_graph_name = str(graph.dot.get_name()).strip('"')
                    evaluator.routed_graph_name = str(gg[3].get_name()).strip('"')
                    evaluator.closed_ghost_loop_count = graph_ghost_loop_count
                    evaluator.approximation_type = getattr(
                        term_integrand,
                        "approximation_type",
                        None,
                    )
                    evaluators.append(evaluator)

            if self.dy_check_generation_limits:
                all_routed_integrands.extend(routed_integrands)
            all_evaluators.extend(evaluators)

        if self.dy_check_generation_limits and all_routed_integrands:
            approach_limit = approach_point(
                n_loops, process_name, all_routed_integrands
            )
            print("##################")
            z = 0.6
            # ks = [
            #    math.sqrt(z)
            #    * np.array([1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)])
            # ]
            scale = 1000
            ks = [
                # math.sqrt(z)
                scale * np.array([1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)])
            ]
            vp = 0 * np.array([0, 1, 1])
            p1 = scale * np.array([0, 0, 1])
            p2 = scale * np.array([0, 0, -1])
            approach_limit.approach(ks, p1, p2, z, vp)

        if all_evaluators:
            my_compiler = compile_integrands(
                n_loops,
                process_name,
                self.get_integrand_name(),
                "z",
                all_evaluators,
                fallback_precision=self.dy_fallback_precision,
                bundle_metadata={
                    "dy_top_self_energy_renormalisation": (
                        _dy_top_self_energy_mode(self)
                    ),
                    "dy_projected_os_construction_schema_version": (
                        projected_os_schema_for_mode(
                            _dy_top_self_energy_mode(self)
                        )
                    ),
                    "ttbar_cm_e_surface_schema": (
                        TTBAR_CM_E_SURFACE_SCHEMA
                        if process_name == "tt~"
                        else None
                    ),
                    "dy_cm_evaluator_schema": (
                        DY_CM_EVALUATOR_SCHEMA if process_name == "DY" else None
                    ),
                },
            )
            my_compiler.save_compiled_integrand()

        print("n routed:", len(processed_graphs))
        return processed_graphs

    def _require_ttbar_generation_symmetrisation(self) -> None:
        if (
            self.n_loops == 2
            and self.process_name == "tt~"
            and not self.symmetrise_p1_p2
        ):
            raise pygloopException(
                "Two-loop ttbar graph generation requires "
                "symmetrise_p1_p2=True for every incoming channel."
            )

    def process_2L_generated_graphs(self, graphs: DYDotGraphs) -> DYDotGraphs:
        self._require_ttbar_generation_symmetrisation()
        final_state = copy.deepcopy(self.final_state)
        process_name = self.process_name
        n_loops = self.n_loops

        processed_graphs = DYDotGraphs()

        filtered_graphs = DYDotGraphs()
        # filtered_graphs.extend(
        #    copy.deepcopy(graphs.filter_particle_definition(final_state))
        # )
        filtered_graphs.extend(copy.deepcopy(graphs))

        print("############################")
        print("Filtered graphs: ", len(filtered_graphs))
        print("############################")

        if self.dy_parallel_graphs > 1 and len(filtered_graphs) > 1:
            return self._process_2L_generated_graphs_parallel(filtered_graphs)

        channel = self.dy_channel

        processor = EMRIntegrandConstructor(
            [], process_name, n_loops, state_name=self.dy_emr_state_name
        )
        loop_processor = LoopIntegrandConstructor(
            [],
            process_name,
            n_loops,
            channel=channel,
            external_gluon_polarisation=self.external_gluon_polarisation,
            disable_integrated_uv_cts=self.disable_integrated_uv_cts,
            top_self_energy_os_subtraction=(
                _dy_top_self_energy_legacy_forwarding(self)
            ),
            top_self_energy_renormalisation=_dy_top_self_energy_mode(self),
            threshold_h_function=getattr(
                self, "dy_threshold_h_function", None
            ),
            include_disabled_threshold_counterterms=getattr(
                self, "dy_include_disabled_threshold_counterterms", False
            ),
            emr_state_name=self.dy_emr_state_name,
            symmetrise_p1_p2=self.symmetrise_p1_p2,
        )

        all_routed_integrands = []
        all_evaluators = []

        for local_graph_index, graph in enumerate(filtered_graphs):
            graph_index = self.dy_graph_index_offset + local_graph_index
            vac_g = canonicalise_vacuum_graph(copy.deepcopy(graph))

            vacuum_g = VacuumDotGraph(copy.deepcopy(vac_g.dot))

            # _cuts = vacuum_g.get_cutkosky_cuts()
            routed_graphs = vacuum_g.cut_graphs_with_routing_leading_virtuality(
                [], final_state
            )

            indexed_routed_graphs = [
                (routed_graph_index, routed_graph)
                for routed_graph_index, routed_graph in enumerate(routed_graphs)
                if _strip_quotes(str(routed_graph[3].get("particle_channel")))
                == str(channel)
            ]
            if self.symmetrise_p1_p2:
                symmetrised_graphs = filter_symmetrised_p1_p2_routed_cuts(
                    [routed_graph for _, routed_graph in indexed_routed_graphs]
                )
                retained_graph_ids = {id(graph) for graph in symmetrised_graphs}
                indexed_routed_graphs = [
                    (routed_graph_index, routed_graph)
                    for routed_graph_index, routed_graph in indexed_routed_graphs
                    if id(routed_graph) in retained_graph_ids
                ]

            print("############################")
            print("Routed graphs: ", len(routed_graphs))
            print("############################")

            routed_integrands = []
            evaluators = []

            for routed_graph_index, gg in indexed_routed_graphs:
                # if (len(gg[2][0]) == 1 and len(gg[2][1]) == 1):
                #    continue

                # if len(gg[2][1]) != 2:
                #    continue

                # if len(gg[2][1]) != 1 or len(gg[2][0]) != 1:
                #    continue
                #
                processed_graphs.append(gg[3])
                cut_graph = deepcopy(routed_cut_graph(gg[3], gg[0], gg[1], gg[2]))
                # print(cut_graph.graph.get_name())
                # print(cut_graph.graph)
                term_integrands = loop_processor.get_integrand(deepcopy(cut_graph))
                graph_ghost_loop_count = (
                    closed_ghost_loop_count(
                        graph,
                        self._resolve_dy_ghost_particle_names(),
                    )
                    if term_integrands
                    else 0
                )

                if self.dy_check_generation_limits:
                    routed_integrands.extend(deepcopy(term_integrands))

                observable_params = self.dy_observable_params(
                    default_lambda_sq=50000,
                    default_muv=2000,
                    default_mur_sq=50000,
                )

                print("reached evaluator stage")
                for term_index, term_integrand in enumerate(term_integrands):
                    evaluator = evaluate_integrand(
                        n_loops,
                        process_name,
                        deepcopy(term_integrand),
                        n_hornerscheme_iterations=1,
                        n_cpe_iterations=1,
                        observable_params=observable_params,
                    )
                    evaluator.compiled_name = (
                        f"graph_{graph_index}_cut_{routed_graph_index}"
                        f"_term_{term_index}_integrand"
                    )
                    evaluator.source_graph_name = str(graph.dot.get_name()).strip('"')
                    evaluator.routed_graph_name = str(gg[3].get_name()).strip('"')
                    evaluator.closed_ghost_loop_count = graph_ghost_loop_count
                    evaluator.approximation_type = getattr(
                        term_integrand,
                        "approximation_type",
                        None,
                    )
                    evaluators.append(evaluator)
                print("constructed evaluators")

            if self.dy_check_generation_limits:
                all_routed_integrands.extend(routed_integrands)
            all_evaluators.extend(evaluators)

            print("added up evaluators")

        if self.dy_check_generation_limits and all_routed_integrands:
            print("pre limit taker")
            approach_limit = approach_point(
                n_loops, process_name, all_routed_integrands
            )
            print("constructed limit taker")
            print("##################")
            z = 0.6
            # ks = [
            #    math.sqrt(z)
            #    * np.array([1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)])
            # ]
            scale = 1000
            mt = 173
            ks = [
                # math.sqrt(z)
                scale
                * np.array([
                    1.0 / math.sqrt(2),
                    1.0 / math.sqrt(2),
                    0.0,
                ]),
                scale
                * np.array([1 / math.sqrt(3), -1 / math.sqrt(3), 1 / math.sqrt(3)]),
            ]
            # ks=[[2.0543179648600841e+08, -1.8748053733626541e+08, 1.0307223303487062e+08], [-2.6820491673003684e+01, 1.2449677258220136e+02, 1.1282568232590195e+02]]
            # ks = [
            #    # math.sqrt(z)
            #    scale
            #    * np.array([1 / math.sqrt(3), -1 / math.sqrt(3), 1 / math.sqrt(3)]),
            #    scale
            #    * np.array([
            #        0.0,
            #        0.0,
            #        -1 / math.sqrt(5),
            #    ]),
            # ]
            scale = 1000
            # ks = [
            #    # math.sqrt(z)
            #    0
            #    * scale
            #    * np.array([
            #        1,
            #        1 / math.sqrt(5),
            #        -1 / math.sqrt(5),
            #    ]),
            #    -scale
            #    * np.array([
            #        0.0,
            #        1 / math.sqrt(5),
            #        -1 / math.sqrt(5),
            #    ]),
            # ]
            scale = 1000
            ks = [
                # math.sqrt(z)
                scale
                * np.array([
                    0.0,
                    0.0,
                    -1 / math.sqrt(5),
                ]),
                scale * np.array([1 / math.sqrt(3), -1 / math.sqrt(3), 0]),
            ]
            # ks = [
            #    # math.sqrt(z)
            #    scale * np.array([1 / math.sqrt(3), -1 / math.sqrt(3), 0]),
            #    scale
            #    * np.array([
            #        0.00 / math.sqrt(3),
            #        0.00 / math.sqrt(3),
            #        1 / math.sqrt(3),
            #    ]),
            # ]
            vp = 10 * np.array([1 / 10, 1 / 10, -1 / 2])
            p1 = scale * np.array([0, 0, 1])
            p2 = scale * np.array([0, 0, -1])
            print("just about to approach limit")
            approach_limit.approach(ks, p1, p2, z, vp)
        #
        # uv_test = ultraviolet_test(n_loops, process_name, all_routed_integrands)
        # uv_test.approach_limits(2000)

        if all_evaluators:
            my_compiler = compile_integrands(
                n_loops,
                process_name,
                self.get_integrand_name(),
                "z",
                all_evaluators,
                fallback_precision=self.dy_fallback_precision,
                bundle_metadata={
                    "dy_top_self_energy_renormalisation": (
                        _dy_top_self_energy_mode(self)
                    ),
                    "dy_projected_os_construction_schema_version": (
                        projected_os_schema_for_mode(
                            _dy_top_self_energy_mode(self)
                        )
                    ),
                    "ttbar_cm_e_surface_schema": (
                        TTBAR_CM_E_SURFACE_SCHEMA
                        if process_name == "tt~"
                        else None
                    ),
                    "dy_cm_evaluator_schema": (
                        DY_CM_EVALUATOR_SCHEMA if process_name == "DY" else None
                    ),
                },
            )
            my_compiler.save_compiled_integrand()

        print("############################")
        print("Processed graphs: ", len(routed_graphs))
        print("############################")

        return processed_graphs

    def _process_2L_generated_graphs_parallel(
        self, filtered_graphs: DYDotGraphs
    ) -> DYDotGraphs:
        start = time.monotonic()
        integrand_name = self.get_integrand_name()
        run_id = f"{os.getpid()}_{time.monotonic_ns()}"
        worker_count = min(self.dy_parallel_graphs, len(filtered_graphs))
        log_dir = pjoin(
            OUTPUTS_FOLDER,
            "parallel_graph_logs",
            self.name,
            f"{integrand_name}_{run_id}",
        )

        tasks = []
        ghost_particle_names = sorted(self._resolve_dy_ghost_particle_names())
        for graph_index, graph in enumerate(filtered_graphs):
            graph_name = _strip_quotes(str(graph.dot.get_name()))
            safe_graph_name = "".join(
                c if c.isalnum() or c in ("_", "-") else "_" for c in graph_name
            )
            worker_name = (
                f"{self.name}_par_{run_id}_graph_{graph_index:04d}_{safe_graph_name}"
            )
            tasks.append({
                "graph_index": graph_index,
                "graph_name": graph_name,
                "graph_dot": graph.to_string(),
                "ghost_particle_names": ghost_particle_names,
                "worker_name": worker_name,
                "log_path": pjoin(log_dir, f"{worker_name}.log"),
                "m_top": self.m_top,
                "m_higgs": self.m_higgs,
                "ps_point": self.ps_point,
                "helicities": self.helicities,
                "toml_config_path": self.toml_config_path,
                "runtime_toml_config_path": self.runtime_toml_config_path,
                "final_state": self.final_state,
                "process_name": self.process_name,
                "dy_channel": self.dy_channel,
                "integrate_beams": self.integrate_beams,
                "external_gluon_polarisation": self.external_gluon_polarisation,
                "disable_integrated_uv_cts": self.disable_integrated_uv_cts,
                "dy_top_self_energy_os_subtraction": (
                    _dy_top_self_energy_legacy_forwarding(self)
                ),
                "dy_top_self_energy_renormalisation": (
                    _dy_top_self_energy_mode(self)
                ),
                "dy_check_generation_limits": self.dy_check_generation_limits,
                "dy_threshold_h_function": self.dy_threshold_h_function,
                "dy_include_disabled_threshold_counterterms": (
                    self.dy_include_disabled_threshold_counterterms
                ),
                "symmetrise_p1_p2": self.symmetrise_p1_p2,
                "dy_fallback_precision": self.dy_fallback_precision,
                "dy_lambda_sq": self.dy_lambda_sq,
                "dy_mur_sq": self.dy_mur_sq,
                "dy_observable_muv": self.dy_observable_muv,
            })

        print(
            "DY_PARALLEL_GRAPHS "
            f"workers={worker_count} graphs={len(tasks)} log_dir={log_dir}"
        )

        ctx = multiprocessing.get_context("spawn")
        results = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=ctx,
        ) as executor:
            future_to_task = {
                executor.submit(_dy_process_2l_graph_worker, task): task
                for task in tasks
            }
            for future in concurrent.futures.as_completed(future_to_task):
                result = future.result()
                results.append(result)
                status = "ok" if result.get("ok") else "failed"
                print(
                    "DY_PARALLEL_GRAPH_DONE "
                    f"status={status} graph_index={result.get('graph_index')} "
                    f"graph={result.get('graph_name')} "
                    f"elapsed={float(result.get('elapsed', 0.0)):.3f}s "
                    f"log={result.get('log_path')}"
                )

        failures = [result for result in results if not result.get("ok")]
        if failures:
            details = "\n".join(
                f"graph_index={failure.get('graph_index')} "
                f"graph={failure.get('graph_name')} "
                f"error={failure.get('error')} "
                f"log={failure.get('log_path')}"
                for failure in failures
            )
            raise pygloopException(f"Parallel DY graph generation failed:\n{details}")

        results.sort(key=lambda result: int(result["graph_index"]))
        DYCompiledBundle.merge_existing_bundles(
            process=self.process_name,
            integrand_name=integrand_name,
            n_loops=self.n_loops,
            source_integrand_names=[result["bundle_name"] for result in results],
        )

        processed_graphs = DYDotGraphs()
        for result in results:
            if result["processed_dot"].strip():
                processed_graphs.extend(DYDotGraphs(dot_str=result["processed_dot"]))

        print(
            "DY_PARALLEL_GRAPHS_DONE "
            f"graphs={len(tasks)} workers={worker_count} "
            f"elapsed={time.monotonic() - start:.3f}s"
        )
        return processed_graphs

    @staticmethod
    def initial_state_for_dy_channel(channel: tuple[int, int]) -> str:
        initial_state_by_channel = {
            (0, 0): "g g",
            (0, 1): "g d",
            (1, 0): "d g",
            (1, -1): "d d~",
            (-1, 1): "d~ d",
        }
        try:
            return initial_state_by_channel[tuple(channel)]
        except KeyError as exc:
            raise pygloopException(
                f"Unsupported DY partonic channel {tuple(channel)}."
            ) from exc

    @classmethod
    def initial_state_family_for_dy_channel(cls, channel: tuple[int, int]) -> str:
        if tuple(channel) == (0, 1):
            return cls.initial_state_for_dy_channel((1, 0))
        if tuple(channel) == (-1, 1):
            return cls.initial_state_for_dy_channel((1, -1))
        return cls.initial_state_for_dy_channel(channel)

    def generate_graphs(self) -> None:
        self._require_ttbar_generation_symmetrisation()
        graphs_process_name = self.get_integrand_name(suffix="_generated_graphs")
        integrand_name = self.get_integrand_name()
        amplitudes, _cross_sections = self.gl_worker.list_outputs()
        base_name = self.get_integrand_name(suffix="")
        if graphs_process_name in amplitudes:
            logger.info(
                f"Graphs for amplitude {graphs_process_name} already generated and recycled."
            )
            return
        match self.n_loops:
            case 1:
                logger.info("Generating one-loop graphs ...")
                initial_state = self.initial_state_for_dy_channel(self.dy_channel)
                process_name = self.process_name.lower()
                if process_name == "dy":
                    self.gl_worker.run(
                        f"generate xs {initial_state} > a | d d~ g a QED^2==2 [{{{{1}}}} QCD=1] --only-diagrams --numerator-grouping group_identical_graphs_up_to_scalar_rescaling --symmetrize-left-right-states true --symmetrize-initial-states true -p {base_name} -i {graphs_process_name} --max-multiplicity-for-fast-cut-filter 99"
                    )
                elif process_name == "tt~":
                    numerator_grouping = (
                        " --numerator-grouping group_identical_graphs_up_to_scalar_rescaling"
                        if self.dy_channel in {(0, 0), (1, -1), (-1, 1)}
                        else ""
                    )
                    self.gl_worker.run(
                        f"generate xs {initial_state} > t t~ | d d~ g t t~ [{{{{1}}}} QCD=1] --only-diagrams{numerator_grouping} --symmetrize-left-right-states true --symmetrize-initial-states true -p {base_name} -i {graphs_process_name} --max-multiplicity-for-fast-cut-filter 99"
                    )
                else:
                    raise pygloopException(
                        f"Unsupported one-loop DY process {self.process_name!r}."
                    )

                self.gl_worker.run("save state -o")
                DY_1L_dot_files = self.gl_worker.get_dot_files(
                    process_id=None, integrand_name=graphs_process_name
                )
                write_text_with_dirs(
                    pjoin(DOTS_FOLDER, self.name, f"{graphs_process_name}.dot"),
                    DY_1L_dot_files,
                )
                self.gl_worker.run("save dot")
                self.save_state()
                DY_1L_dot_files_processed = self.process_1L_generated_graphs(
                    DYDotGraphs(dot_str=DY_1L_dot_files)
                )
                print(len(DY_1L_dot_files_processed))
                DY_1L_dot_files_processed.save_to_file(
                    pjoin(DOTS_FOLDER, self.name, f"{integrand_name}.dot")
                )
            case 2:
                logger.info("Generating two-loop graphs ...")
                if self.process_name.lower() == "tt~":
                    select_graphs = (
                        f" --select-graphs {' '.join(self.diagrams)}"
                        if self.diagrams
                        else ""
                    )
                    initial_state = self.initial_state_family_for_dy_channel(
                        self.dy_channel
                    )
                    numerator_grouping = (
                        " --numerator-grouping group_identical_graphs_up_to_scalar_rescaling"
                        if self.dy_channel in {(0, 0), (1, -1), (-1, 1)}
                        else ""
                    )
                    # self.gl_worker.run(  # GL06 GL14  --select-graphs GL00 GL01 GL03 GL04 GL05 GL08 GL12
                    #    f"generate xs d g > t t~ | d d~ g t t~ ghG ghG~ [{{{{2}}}} QCD=1] --only-diagrams --symmetrize-left-right-states true --symmetrize-initial-states true{select_graphs} -p {base_name} -i {graphs_process_name} --max-multiplicity-for-fast-cut-filter 99"
                    # )
                    self.gl_worker.run(  # GL06 GL14  --select-graphs GL14
                        f"generate xs {initial_state} > t t~ | d d~ g t t~ ghG ghG~ [{{{{2}}}} QCD=1] --only-diagrams{numerator_grouping} --symmetrize-left-right-states true --symmetrize-initial-states true{select_graphs} -p {base_name} -i {graphs_process_name} --max-multiplicity-for-fast-cut-filter 99"
                    )
                else:
                    raise ValueError(
                        "t t~ is the only implemented process at two loops"
                    )
                self.gl_worker.run("save state -o")
                DY_2L_dot_files = self.gl_worker.get_dot_files(
                    process_id=None, integrand_name=graphs_process_name
                )
                write_text_with_dirs(
                    pjoin(DOTS_FOLDER, self.name, f"{graphs_process_name}.dot"),
                    DY_2L_dot_files,
                )
                self.gl_worker.run("save dot")
                self.save_state()
                DY_2L_dot_files_processed = self.process_2L_generated_graphs(
                    DYDotGraphs(dot_str=DY_2L_dot_files)
                )
                DY_2L_dot_files_processed.save_to_file(
                    pjoin(DOTS_FOLDER, self.name, f"{integrand_name}.dot")
                )
            case _:
                raise pygloopException(f"Number of loops {self.n_loops} not supported.")

    def generate_spenso_code(self, *args, **opts) -> None:
        evaluator_path = pjoin(
            EVALUATORS_FOLDER, self.name, f"{self.get_integrand_name()}.so"
        )
        if os.path.isfile(evaluator_path):
            if self.clean:
                logger.info(
                    f"Removing existing spenso evaluator {evaluator_path} and re-generating it."
                )
                os.remove(evaluator_path)
            else:
                logger.info(
                    f"Spenso evaluator {evaluator_path} already generated and recycled."
                )
                return
        logger.critical(
            f"Spenso code generation for {self.get_integrand_name()}.so not yet implemented."
        )
        # raise NotImplementedError("Implement spenso code generation.")

    def generate_gammaloop_code(self) -> None:
        logger.info(f"Generating GammaLoop code not applicable for process {self.name}")
        return

    def valide_ps_point(self) -> None:
        # Only perform sanity checks if in the physical region
        s = (self.ps_point[0] + self.ps_point[1]).squared()
        if s < 0:
            raise pygloopException("Only physical ps points are supported currently.")
        sqrt_s = math.sqrt(s)
        p_sum = LorentzVector(0.0, 0.0, 0.0, 0.0)
        for p in self.ps_point[:2]:
            m_g = math.sqrt(abs(p.squared()))
            p_sum += p
            if abs(m_g) / sqrt_s > TOLERANCE:
                raise pygloopException("Incoming gluons must be massless.")
        for p in self.ps_point[2:]:
            m_h = math.sqrt(abs(p.squared()))
            p_sum -= p
            if abs(m_h - self.m_higgs) / sqrt_s > TOLERANCE:
                raise pygloopException("Outgoing Higgs bosons must be on-shell.")

        for p_i in p_sum.to_list():
            if abs(p_i) / sqrt_s > TOLERANCE:
                raise pygloopException(
                    "Provided ps point does not respect momentum conservation."
                )

    def parameterize(
        self, xs: list[float], parameterisation: str, origin: Vector | None = None
    ) -> tuple[Vector, float]:
        diagnostic_jacobian = 1.0
        if parameterisation in {"spherical", "log_spherical"}:
            if origin is not None:
                radial_map = diagnostic_soft_radial_map(xs)
                if radial_map is not None:
                    xs, radial_jacobian = radial_map
                    diagnostic_jacobian *= float(radial_jacobian)
            diagnostic_map = diagnostic_soft_equator_map(xs)
            if diagnostic_map is not None:
                xs, equator_jacobian = diagnostic_map
                diagnostic_jacobian *= equator_jacobian
        match parameterisation:
            case "cartesian":
                return self.cartesian_parameterize(xs, origin)
            case "spherical":
                momentum, jacobian = self.spherical_parameterize(xs, origin)
                return momentum, jacobian * diagnostic_jacobian
            case "log_spherical":
                momentum, jacobian = self.log_spherical_parameterize(xs, origin)
                return momentum, jacobian * diagnostic_jacobian
            case _:
                raise pygloopException(
                    f"Parameterisation {parameterisation} not implemented."
                )

    def cartesian_parameterize(
        self, xs: list[float], origin: Vector | None = None
    ) -> tuple[Vector, float]:
        return self.cartesian_parameterize_v2(xs, origin)

    def cartesian_parameterize_v1(
        self, xs: list[float], origin: Vector | None = None
    ) -> tuple[Vector, float]:
        x, y, z = xs
        scale = self.e_cm * RESCALING
        v = (
            Vector((1 / (1 - x) - 1 / x), (1 / (1 - y) - 1 / y), (1 / (1 - z) - 1 / z))
            * scale
        )
        if origin is not None:
            v = v + origin
        jac = scale * (1 / (1 - x) ** 2 + 1 / x**2)
        jac *= scale * (1 / (1 - y) ** 2 + 1 / y**2)
        jac *= scale * (1 / (1 - z) ** 2 + 1 / z**2)
        return (v, jac)

    def cartesian_parameterize_v2(
        self, xs: list[float], origin: Vector | None = None
    ) -> tuple[Vector, float]:
        x, y, z = xs
        scale = self.e_cm * RESCALING
        v = (
            Vector(
                math.tan((x - 0.5) * math.pi),
                math.tan((y - 0.5) * math.pi),
                math.tan((z - 0.5) * math.pi),
            )
            * scale
        )
        if origin is not None:
            v = v + origin
        jac = scale * math.pi / math.cos((x - 0.5) * math.pi) ** 2
        jac *= scale * math.pi / math.cos((y - 0.5) * math.pi) ** 2
        jac *= scale * math.pi / math.cos((z - 0.5) * math.pi) ** 2
        return (v, jac)

    def cartesian_parameterize_v3(
        self, xs: list[float], origin: Vector | None = None
    ) -> tuple[Vector, float]:
        x, y, z = xs
        scale = self.e_cm * RESCALING
        v = (
            Vector(
                math.log(x) - math.log(1 - x),
                math.log(y) - math.log(1 - y),
                math.log(z) - math.log(1 - z),
            )
            * scale
        )
        if origin is not None:
            v = v + origin
        jac = scale * (1 / x + 1 / (1 - x))
        jac *= scale * (1 / y + 1 / (1 - y))
        jac *= scale * (1 / z + 1 / (1 - z))
        return (v, jac)

    def spherical_parameterize(
        self, xs: list[float], origin: Vector | None = None
    ) -> tuple[Vector, float]:
        rx, thetax, phix = xs
        ecm = self.e_cm
        r = rx / (1 - rx) * ecm
        th = 2 * math.pi * thetax
        ph = math.pi * phix
        v = Vector(
            r * math.cos(th) * math.sin(ph),
            r * math.sin(th) * math.sin(ph),
            r * math.cos(ph),
        )
        if origin is not None:
            v = v + origin
        # k-space Jacobian only; z Jacobian is applied separately in integrand_xspace.
        jac = r**2 * math.sin(ph) * 2 * math.pi**2 * ecm / (1 - rx) ** 2
        return (v, jac)

    def log_spherical_parameterize(
        self, xs: list[float], origin: Vector | None = None
    ) -> tuple[Vector, float]:
        rx, thetax, phix = xs
        ecm = self.e_cm
        rho = math.log(ecm) + math.log(rx) - math.log(1 - rx)
        radius = math.exp(rho)
        th = 2 * math.pi * thetax
        ph = math.pi * phix
        v = Vector(
            radius * math.cos(th) * math.sin(ph),
            radius * math.sin(th) * math.sin(ph),
            radius * math.cos(ph),
        )
        if origin is not None:
            v = v + origin
        # k-space Jacobian only; z Jacobian is applied separately in integrand_xspace.
        # For k = exp(rho) * khat, d^3k = exp(3 rho) sin(phi) d rho d theta d phi.
        jac = radius**3 * math.sin(ph) * 2 * math.pi**2 * (1 / rx + 1 / (1 - rx))
        return (v, jac)

    def sampled_beam_momenta(self, x1: float, x2: float) -> tuple[Vector, Vector]:
        return (
            Vector(0.0, 0.0, self.e_cm * math.sqrt(float(x1 * x2)) / 2),
            Vector(0.0, 0.0, -self.e_cm * math.sqrt(float(x1 * x2)) / 2),
        )
        # return (
        #    Vector(0.0, 0.0, self.e_cm * float(x1)),
        #    Vector(0.0, 0.0, -self.e_cm * float(x2)),
        # )

    def dy_pdf_luminosity(self, x1: float, x2: float) -> float:
        if self.dy_pdf_set is None or self.dy_muf_sq is None:
            raise pygloopException("DY PDF luminosity requested without PDF setup.")
        if self._dy_pdf_provider is None:
            self._dy_pdf_provider = DYPDFProvider(
                self.dy_pdf_set,
                self.dy_pdf_member,
                self.dy_pdf_luminosity_family,
            )
        return self._dy_pdf_provider.luminosity(
            self.dy_channel,
            x1,
            x2,
            float(self.dy_muf_sq),
        )

    def _integrate_dy_msbar_scheme_counterterm(
        self,
        seed: int,
        parameterisation: str = "spherical",
        phase: str = "real",
        workers: int = 1,
    ) -> DYSchemeCountertermResult:
        if self.process_name.lower() == "tt~":
            return self._integrate_ttbar_msbar_scheme_counterterm(
                seed,
                parameterisation,
                phase,
                workers,
            )

        # At fixed partonic energy the z-weighted Born test function is
        # constant, so the complete qqbar distribution can be integrated
        # exactly without introducing PDFs or a Q-min cut.
        physical_normalisation = (
            64.0
            * math.pi
            * self.dy_physical_normalisation_factor
            / self.e_cm**2
        )
        if self.dy_integrated_leptonic_phase_space:
            physical_normalisation *= self.dy_integrated_leptonic_phase_space_factor
        physical_normalisation *= self.dy_coupling_normalisation_factor
        if not self.integrate_beams:
            return integrate_partonic_qqbar_scheme_counterterm(
                channel=self.dy_channel,
                z_bin=self.dy_z_bin,
                physical_normalisation=physical_normalisation,
                replicas=self.dy_scheme_counterterm_replicas,
            )

        if self.dy_pdf_set is None or self.dy_muf_sq is None:
            raise pygloopException(
                "DY MSbar scheme counterterm requested without PDF setup."
            )
        if self._dy_pdf_provider is None:
            self._dy_pdf_provider = DYPDFProvider(
                self.dy_pdf_set,
                self.dy_pdf_member,
                self.dy_pdf_luminosity_family,
            )

        # In the current one-loop DY convention, the integrated Born factor
        # multiplying the finite scheme kernel is (pi/2)*(128/e_cm^2) on top
        # of the selected channel's physical beam normalisation.
        counterterm_integrator = (
            integrate_gq_scheme_counterterm
            if self.dy_channel in {(1, 0), (0, 1)}
            else integrate_qqbar_scheme_counterterm
        )
        return counterterm_integrator(
            provider=self._dy_pdf_provider,
            channel=self.dy_channel,
            muf_sq=float(self.dy_muf_sq),
            e_cm_sq=self.e_cm**2,
            z_bin=self.dy_z_bin,
            q_min=float(self.dy_q_min),
            q_max=self.dy_q_max,
            physical_normalisation=physical_normalisation,
            sobol_power=self.dy_scheme_counterterm_sobol_power,
            replicas=self.dy_scheme_counterterm_replicas,
            seed=seed,
        )

    @staticmethod
    def _scheme_born_channels_from_metadata(
        metadata: dict[str, Any],
    ) -> set[tuple[int, int]]:
        particle_ids = {"g": 0, "d": 1, "d~": -1}

        def coefficient_is(value: Any, target: float) -> bool:
            try:
                return float(str(value)) == target
            except (TypeError, ValueError):
                return False

        channels: set[tuple[int, int]] = set()
        for term in metadata.get("terms", []):
            first_particle = None
            second_particle = None
            for routing in (term.get("edge_routings") or {}).values():
                loop_coefficients = routing.get("loop_coefficients", [])
                if not all(
                    coefficient_is(coefficient, 0.0)
                    for coefficient in loop_coefficients
                ):
                    continue
                p1_coefficient = routing.get("p1_coefficient")
                p2_coefficient = routing.get("p2_coefficient")
                if coefficient_is(p1_coefficient, 1.0) and coefficient_is(
                    p2_coefficient, 0.0
                ):
                    first_particle = routing.get("particle")
                elif coefficient_is(p1_coefficient, 0.0) and coefficient_is(
                    p2_coefficient, 1.0
                ):
                    second_particle = routing.get("particle")
            if (
                first_particle in particle_ids
                and second_particle in particle_ids
            ):
                channels.add(
                    (
                        particle_ids[first_particle],
                        particle_ids[second_particle],
                    )
                )
        return channels

    @staticmethod
    def _scheme_born_channel_match(
        available_channels: set[tuple[int, int]],
        required_channel: tuple[int, int],
    ) -> bool | None:
        if required_channel in available_channels:
            return False
        reversed_channel = (required_channel[1], required_channel[0])
        if reversed_channel in available_channels:
            return True
        return None

    def _load_scheme_born_bundle(self, bundle_name: str) -> DYCompiledBundle:
        bundle = DYCompiledBundle.load(self.process_name, bundle_name)
        if bundle.n_loops != 1:
            raise pygloopException(
                "A scheme-counterterm Born bundle must contain a one-loop "
                f"integrand, got L={bundle.n_loops} for {bundle_name!r}."
            )
        bundle.require_fallback_supported(self.dy_fallback_precision)
        return bundle

    def _discover_scheme_born_bundle_candidates(
        self,
        required_channel: tuple[int, int],
    ) -> list[tuple[str, bool]]:
        process_directory = pjoin(EVALUATORS_FOLDER, self.process_name)
        if not os.path.isdir(process_directory):
            return []
        candidates: list[tuple[str, bool]] = []
        for bundle_name in sorted(os.listdir(process_directory)):
            metadata_path = DYCompiledBundle.metadata_path(
                self.process_name,
                bundle_name,
            )
            if not os.path.isfile(metadata_path):
                continue
            try:
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if int(metadata.get("n_loops", -1)) != 1:
                continue
            available_channels = self._scheme_born_channels_from_metadata(
                metadata
            )
            swap_beams = self._scheme_born_channel_match(
                available_channels,
                required_channel,
            )
            if swap_beams is not None:
                candidates.append((bundle_name, swap_beams))
        return candidates

    def _resolve_scheme_born_bundles(
        self,
        required_channels: tuple[tuple[int, int], ...],
    ) -> dict[tuple[int, int], tuple[DYCompiledBundle, bool]]:
        unresolved = [
            channel
            for channel in required_channels
            if channel not in self._dy_scheme_born_compiled_bundles
        ]
        if not unresolved:
            return {
                channel: self._dy_scheme_born_compiled_bundles[channel]
                for channel in required_channels
            }

        if (
            len(required_channels) == 1
            and self._dy_scheme_born_compiled_bundle is not None
        ):
            self._dy_scheme_born_compiled_bundles[required_channels[0]] = (
                self._dy_scheme_born_compiled_bundle,
                False,
            )
            return {
                required_channels[0]: self._dy_scheme_born_compiled_bundles[
                    required_channels[0]
                ]
            }

        for bundle_name in self.dy_scheme_born_bundles:
            bundle = self._load_scheme_born_bundle(bundle_name)
            available_channels = self._scheme_born_channels_from_metadata(
                bundle.metadata
            )
            matched_channel = None
            matched_swap = None
            for required_channel in unresolved:
                swap_beams = self._scheme_born_channel_match(
                    available_channels,
                    required_channel,
                )
                if swap_beams is None:
                    continue
                if matched_channel is not None:
                    raise pygloopException(
                        f"Born bundle {bundle_name!r} ambiguously matches more "
                        "than one requested initial-state channel."
                    )
                matched_channel = required_channel
                matched_swap = swap_beams
            if matched_channel is None:
                raise pygloopException(
                    f"Born bundle {bundle_name!r} has inferred channels "
                    f"{sorted(available_channels)}, none of which are required "
                    f"for scheme channels {required_channels}."
                )
            self._dy_scheme_born_compiled_bundles[matched_channel] = (
                bundle,
                bool(matched_swap),
            )
            unresolved.remove(matched_channel)

        for required_channel in unresolved:
            candidates = self._discover_scheme_born_bundle_candidates(
                required_channel
            )
            if len(candidates) != 1:
                candidate_names = ", ".join(
                    name for name, _swap_beams in candidates
                ) or "none"
                raise pygloopException(
                    "Could not uniquely resolve the one-loop Born bundle for "
                    f"channel {required_channel}; candidates: {candidate_names}. "
                    "Configure integrate.beams.scheme_counterterm.born_bundles "
                    "explicitly."
                )
            bundle_name, swap_beams = candidates[0]
            self._dy_scheme_born_compiled_bundles[required_channel] = (
                self._load_scheme_born_bundle(bundle_name),
                swap_beams,
            )

        return {
            channel: self._dy_scheme_born_compiled_bundles[channel]
            for channel in required_channels
        }

    def _scheme_born_integrand(
        self,
        bundle: DYCompiledBundle,
        swap_beams: bool,
        parameterisation: str,
        phase: str,
    ) -> Callable[[float, tuple[float, float, float], bool], float]:
        def born_integrand(
            partonic_scale_sq: float,
            loop_coordinates: tuple[float, float, float],
            use_fallback: bool,
        ) -> float:
            loop_momentum, jacobian = self.parameterize(
                list(loop_coordinates),
                parameterisation,
            )
            beam_energy = math.sqrt(partonic_scale_sq) / 2.0
            p1 = Vector(0.0, 0.0, beam_energy)
            p2 = Vector(0.0, 0.0, -beam_energy)
            if swap_beams:
                p1, p2 = p2, p1
            if use_fallback:
                value, _term_values = bundle.evaluate_arb_terms(
                    [loop_momentum],
                    p1,
                    p2,
                    1.0,
                    None,
                    decimal_digit_precision=self.dy_fallback_precision,
                    precision_preserving=True,
                    runtime_parameters=self.dy_runtime_parameters,
                )
            else:
                value = bundle.evaluate(
                    [loop_momentum],
                    p1,
                    p2,
                    1.0,
                    None,
                    runtime_parameters=self.dy_runtime_parameters,
                )
            return self._phase_value(complex(value), phase) * jacobian

        return born_integrand

    @staticmethod
    def _scheme_born_cross_section_factor(bundle: DYCompiledBundle) -> float:
        """Convert a one-loop forward bundle into one physical Born cut.

        The forward representation carries the usual factor two of the
        discontinuity.  If generation retained both routed orientations of a
        cut, their identical contributions must additionally be averaged.
        """

        cut_multiplicities: dict[str, int] = {}
        for term in bundle.terms:
            if term.approximation_type not in {None, "PM"}:
                continue
            group_name = term.graph_group_name or term.source_graph_name
            if group_name is None:
                raise pygloopException(
                    "A scheme-counterterm Born term is missing graph-group "
                    "metadata. Regenerate its compiled bundle."
                )
            cut_multiplicities[group_name] = (
                cut_multiplicities.get(group_name, 0) + 1
            )
        multiplicities = set(cut_multiplicities.values())
        if len(multiplicities) != 1:
            raise pygloopException(
                "A scheme-counterterm Born bundle has inconsistent routed-cut "
                f"multiplicities: {sorted(multiplicities)}."
            )
        if not multiplicities:
            raise pygloopException(
                "A scheme-counterterm Born bundle contains no physical terms."
            )
        return 0.5 / float(multiplicities.pop())

    def _integrate_ttbar_msbar_scheme_counterterm(
        self,
        seed: int,
        parameterisation: str,
        phase: str,
        workers: int = 1,
    ) -> DYSchemeCountertermResult:
        if self.integrate_beams:
            if self.dy_pdf_set is None or self.dy_muf_sq is None:
                raise pygloopException(
                    "Hadronic ttbar MSbar scheme conversion requires PDF setup."
                )
            if self._dy_pdf_provider is None:
                self._dy_pdf_provider = DYPDFProvider(
                    self.dy_pdf_set,
                    self.dy_pdf_member,
                    self.dy_pdf_luminosity_family,
                )
        elif self._dy_pdf_provider is not None or self.dy_muf_sq is not None:
            raise pygloopException(
                "Partonic ttbar MSbar scheme conversion does not accept PDFs."
            )
        if self.dy_channel in {(1, 0), (0, 1)}:
            qqbar_born_channel = (
                (1, -1) if self.dy_channel == (1, 0) else (-1, 1)
            )
            required_channels = (qqbar_born_channel, (0, 0))
            born_bundles = self._resolve_scheme_born_bundles(
                required_channels
            )
            lambda_sq, mu_sq = DY_DEFAULT_LAMBDA_MUR_SQ[self.n_loops]
            if self.dy_lambda_sq is not None:
                lambda_sq = self.dy_lambda_sq
            if self.dy_mur_sq is not None:
                mu_sq = self.dy_mur_sq

            qqbar_bundle, qqbar_swap = born_bundles[qqbar_born_channel]
            gg_bundle, gg_swap = born_bundles[(0, 0)]
            qqbar_born_factor = self._scheme_born_cross_section_factor(
                qqbar_bundle
            )
            gg_born_factor = self._scheme_born_cross_section_factor(gg_bundle)
            convolutions = (
                DYRegularSchemeConvolution(
                    label="D_gq_x_qqbar",
                    born_channel=qqbar_born_channel,
                    physical_normalisation=(
                        qqbar_born_factor
                        * physical_beam_normalisation_factor(
                            qqbar_born_channel,
                            1,
                        )
                    ),
                    kernel=lambda xi: finite_g_to_q_scheme_kernel(
                        xi,
                        lambda_sq,
                        mu_sq,
                        self.dy_scheme_alpha_s,
                    ),
                    born_integrand=self._scheme_born_integrand(
                        qqbar_bundle,
                        qqbar_swap,
                        parameterisation,
                        phase,
                    ),
                ),
                DYRegularSchemeConvolution(
                    label="D_qg_x_gg",
                    born_channel=(0, 0),
                    physical_normalisation=(
                        gg_born_factor
                        * physical_beam_normalisation_factor((0, 0), 1)
                    ),
                    kernel=lambda xi: finite_q_to_g_scheme_kernel(
                        xi,
                        lambda_sq,
                        mu_sq,
                        self.dy_scheme_alpha_s,
                    ),
                    born_integrand=self._scheme_born_integrand(
                        gg_bundle,
                        gg_swap,
                        parameterisation,
                        phase,
                    ),
                ),
            )
            return integrate_regular_born_scheme_counterterm(
                provider=self._dy_pdf_provider,
                channel=self.dy_channel,
                muf_sq=(
                    float(self.dy_muf_sq)
                    if self.dy_muf_sq is not None
                    else None
                ),
                e_cm_sq=self.e_cm**2,
                threshold_sq=4.0 * self.m_top**2,
                convolutions=convolutions,
                sobol_power=self.dy_scheme_counterterm_sobol_power,
                replicas=self.dy_scheme_counterterm_replicas,
                seed=seed,
                clip_threshold=self.dy_scheme_counterterm_clip,
                workers=workers,
                integrate_beams=self.integrate_beams,
            )

        born_bundles = self._resolve_scheme_born_bundles((self.dy_channel,))
        born_bundle, swap_beams = born_bundles[self.dy_channel]
        return integrate_ttbar_qqbar_scheme_counterterm(
            provider=self._dy_pdf_provider,
            channel=self.dy_channel,
            muf_sq=float(self.dy_muf_sq),
            e_cm_sq=self.e_cm**2,
            m_top=self.m_top,
            physical_normalisation=physical_beam_normalisation_factor(
                self.dy_channel,
                1,
            ),
            born_integrand=self._scheme_born_integrand(
                born_bundle,
                swap_beams,
                parameterisation,
                phase,
            ),
            sobol_power=self.dy_scheme_counterterm_sobol_power,
            replicas=self.dy_scheme_counterterm_replicas,
            seed=seed,
            clip_threshold=self.dy_scheme_counterterm_clip,
        )

    def _integrate_ttbar_gg_auxiliary(
        self,
        seed: int,
        parameterisation: str,
        phase: str,
        workers: int = 1,
    ) -> DYGGAuxiliaryResult:
        if not self._dy_ttbar_gg_scheme_mode:
            raise pygloopException(
                "The correlated gg auxiliary path requires two-loop ttbar "
                "with channel (0,0)."
            )

        provider: DYPDFProvider | None = None
        muf_sq: float | None = None
        if self.integrate_beams:
            if self.dy_pdf_set is None or self.dy_muf_sq is None:
                raise pygloopException(
                    "Hadronic ttbar gg scheme conversion requires PDF setup."
                )
            if self._dy_pdf_provider is None:
                self._dy_pdf_provider = DYPDFProvider(
                    self.dy_pdf_set,
                    self.dy_pdf_member,
                    self.dy_pdf_luminosity_family,
                )
            provider = self._dy_pdf_provider
            muf_sq = float(self.dy_muf_sq)

        born_bundle, swap_beams = self._resolve_scheme_born_bundles(((0, 0),))[
            (0, 0)
        ]
        born_integrand = self._scheme_born_integrand(
            born_bundle,
            swap_beams,
            parameterisation,
            phase,
        )
        physical_born_normalisation = (
            self._scheme_born_cross_section_factor(born_bundle)
            * physical_beam_normalisation_factor((0, 0), 1)
        )
        default_lambda_sq, default_mur_sq = DY_DEFAULT_LAMBDA_MUR_SQ[
            self.n_loops
        ]
        lambda_sq = (
            self.dy_lambda_sq
            if self.dy_lambda_sq is not None
            else default_lambda_sq
        )
        mur_sq = (
            self.dy_mur_sq if self.dy_mur_sq is not None else default_mur_sq
        )
        return integrate_ttbar_gg_auxiliary(
            channel=self.dy_channel,
            e_cm_sq=self.e_cm**2,
            m_top=self.m_top,
            physical_normalisation=physical_born_normalisation,
            born_integrand=born_integrand,
            lambda_sq=lambda_sq,
            mur_sq=mur_sq,
            alpha_s=self.dy_scheme_alpha_s,
            sobol_power=self.dy_scheme_counterterm_sobol_power,
            replicas=self.dy_scheme_counterterm_replicas,
            seed=seed,
            clip_threshold=self.dy_scheme_counterterm_clip,
            provider=provider,
            muf_sq=muf_sq,
            integrate_beams=self.integrate_beams,
            workers=workers,
        )

    def _apply_ttbar_gg_scheme(
        self,
        hard_result: IntegrationResult,
        seed: int,
        parameterisation: str,
        phase: str,
        workers: int = 1,
    ) -> IntegrationResult:
        auxiliary = self._integrate_ttbar_gg_auxiliary(
            seed,
            parameterisation,
            phase,
            workers,
        )
        hard_factor = -0.5
        scheme_factor = self.dy_scheme_counterterm_factor
        decoupling_factor = self._ttbar_decoupling_coefficient()

        raw_hard_central = hard_result.central_value
        raw_hard_error = hard_result.error
        applied_hard_central = hard_factor * raw_hard_central
        applied_hard_error = abs(hard_factor) * raw_hard_error
        applied_dgg_minus_lsz = (
            scheme_factor * auxiliary.dgg_minus_lsz_central_value
        )
        applied_dgg_minus_lsz_error = (
            abs(scheme_factor) * auxiliary.dgg_minus_lsz_error
        )
        applied_top_lsz = scheme_factor * auxiliary.top_lsz_central_value
        applied_top_lsz_error = (
            abs(scheme_factor) * auxiliary.top_lsz_error
        )
        applied_scheme_central = (
            scheme_factor * auxiliary.combined_central_value
        )
        applied_scheme_error = abs(scheme_factor) * auxiliary.combined_error
        applied_scheme_replicas = tuple(
            scheme_factor * value for value in auxiliary.combined_replica_values
        )
        applied_born_central = decoupling_factor * auxiliary.born_central_value
        applied_born_error = abs(decoupling_factor) * auxiliary.born_error
        applied_born_replicas = tuple(
            decoupling_factor * value for value in auxiliary.born_replica_values
        )
        applied_auxiliary_replicas = tuple(
            scheme_value + born_value
            for scheme_value, born_value in zip(
                applied_scheme_replicas,
                applied_born_replicas,
                strict=True,
            )
        )
        applied_auxiliary_central, applied_auxiliary_error = (
            self._summarise_replica_values(applied_auxiliary_replicas)
        )

        hard_result.dy_scheme_conversion_enabled = True
        hard_result.dy_scheme_conversion_channel = "gg"
        hard_result.dy_decoupling_enabled = self.dy_decoupling
        hard_result.dy_scheme_alpha_s = self.dy_scheme_alpha_s
        hard_result.dy_hard_factor = hard_factor
        hard_result.dy_hard_unscaled_central_value = raw_hard_central
        hard_result.dy_hard_unscaled_error = raw_hard_error
        hard_result.dy_hard_central_value = applied_hard_central
        hard_result.dy_hard_error = applied_hard_error

        hard_result.dy_scheme_counterterm_factor = scheme_factor
        hard_result.dy_scheme_counterterm_factor_was_explicit = (
            self.dy_scheme_counterterm_factor_was_explicit
        )
        hard_result.dy_scheme_counterterm_unscaled_central_value = (
            auxiliary.combined_central_value
        )
        hard_result.dy_scheme_counterterm_unscaled_error = auxiliary.combined_error
        hard_result.dy_scheme_counterterm_central_value = applied_scheme_central
        hard_result.dy_scheme_counterterm_error = applied_scheme_error
        hard_result.dy_scheme_counterterm_unscaled_replica_values = (
            auxiliary.combined_replica_values
        )
        hard_result.dy_scheme_counterterm_replica_values = (
            applied_scheme_replicas
        )

        hard_result.dy_dgg_minus_lsz_unscaled_central_value = (
            auxiliary.dgg_minus_lsz_central_value
        )
        hard_result.dy_dgg_minus_lsz_unscaled_error = (
            auxiliary.dgg_minus_lsz_error
        )
        hard_result.dy_dgg_minus_lsz_central_value = applied_dgg_minus_lsz
        hard_result.dy_dgg_minus_lsz_error = applied_dgg_minus_lsz_error
        hard_result.dy_top_lsz_unscaled_central_value = (
            auxiliary.top_lsz_central_value
        )
        hard_result.dy_top_lsz_unscaled_error = auxiliary.top_lsz_error
        hard_result.dy_top_lsz_central_value = applied_top_lsz
        hard_result.dy_top_lsz_error = applied_top_lsz_error
        hard_result.dy_scheme_born_central_value = auxiliary.born_central_value
        hard_result.dy_scheme_born_error = auxiliary.born_error
        hard_result.dy_scheme_born_replica_values = auxiliary.born_replica_values

        hard_result.dy_decoupling_coefficient = decoupling_factor
        hard_result.dy_decoupling_born_unscaled_central_value = (
            auxiliary.born_central_value
        )
        hard_result.dy_decoupling_born_unscaled_error = auxiliary.born_error
        hard_result.dy_decoupling_born_central_value = applied_born_central
        hard_result.dy_decoupling_born_error = applied_born_error
        hard_result.dy_decoupling_born_unscaled_replica_values = (
            auxiliary.born_replica_values
        )
        hard_result.dy_decoupling_born_replica_values = applied_born_replicas

        hard_result.dy_auxiliary_central_value = applied_auxiliary_central
        hard_result.dy_auxiliary_error = applied_auxiliary_error
        hard_result.dy_auxiliary_replica_values = applied_auxiliary_replicas
        hard_result.dy_auxiliary_n_samples = auxiliary.n_samples
        hard_result.dy_auxiliary_elapsed_time = auxiliary.elapsed_time
        hard_result.dy_auxiliary_fallback_count = auxiliary.fallback_count
        hard_result.dy_auxiliary_nonfinite_count = auxiliary.nonfinite_count
        hard_result.dy_auxiliary_clipped_count = auxiliary.clipped_count
        hard_result.dy_auxiliary_fallback_fraction = (
            auxiliary.fallback_count / auxiliary.n_samples
            if auxiliary.n_samples
            else 0.0
        )
        hard_result.dy_auxiliary_nonfinite_fraction = (
            auxiliary.nonfinite_count / auxiliary.n_samples
            if auxiliary.n_samples
            else 0.0
        )
        hard_result.dy_auxiliary_clipped_fraction = (
            auxiliary.clipped_count / auxiliary.n_samples
            if auxiliary.n_samples
            else 0.0
        )
        hard_result.dy_scheme_counterterm_n_samples = auxiliary.n_samples
        hard_result.dy_scheme_counterterm_elapsed_time = auxiliary.elapsed_time
        hard_result.dy_scheme_counterterm_fallback_count = auxiliary.fallback_count
        hard_result.dy_scheme_counterterm_nonfinite_count = auxiliary.nonfinite_count
        hard_result.dy_scheme_counterterm_clipped_count = auxiliary.clipped_count
        hard_result.dy_scheme_counterterm_clipped_fraction = (
            hard_result.dy_auxiliary_clipped_fraction
        )
        hard_result.dy_scheme_counterterm_components = {
            component.label: {
                "unscaled_central_value": component.central_value,
                "unscaled_error": component.error,
                "central_value": scheme_factor * component.central_value,
                "error": abs(scheme_factor) * component.error,
                "unscaled_replica_values": component.replica_values,
                "replica_values": tuple(
                    scheme_factor * value for value in component.replica_values
                ),
            }
            for component in auxiliary.components
        }

        hard_result.central_value = (
            applied_hard_central + applied_auxiliary_central
        )
        hard_result.error = math.hypot(
            applied_hard_error,
            applied_auxiliary_error,
        )

        logger.info(
            "ttbar gg scheme/decoupling flags: scheme=%s decoupling=%s; "
            "coefficients hard=%+.16e scheme=%+.16e Born=%+.16e; "
            "physical convention=-1/2*Hraw+(Dgg-LSZ)+top-LSZ+cdec*Born",
            True,
            self.dy_decoupling,
            hard_factor,
            scheme_factor,
            decoupling_factor,
        )
        logger.info(
            "ttbar gg components: hard=%+.16e +/- %.4e; "
            "Dgg-LSZ=%+.16e +/- %.4e; top-LSZ=%+.16e +/- %.4e; "
            "Born=%+.16e +/- %.4e; auxiliary=%+.16e +/- %.4e; "
            "total=%+.16e +/- %.4e",
            applied_hard_central,
            applied_hard_error,
            applied_dgg_minus_lsz,
            applied_dgg_minus_lsz_error,
            applied_top_lsz,
            applied_top_lsz_error,
            applied_born_central,
            applied_born_error,
            applied_auxiliary_central,
            applied_auxiliary_error,
            hard_result.central_value,
            hard_result.error,
        )
        logger.info(
            "ttbar gg auxiliary diagnostics: samples=%d elapsed=%.2fs "
            "fallback/nonfinite/clipped=%d/%d/%d; fractions=%.3e/%.3e/%.3e",
            auxiliary.n_samples,
            auxiliary.elapsed_time,
            auxiliary.fallback_count,
            auxiliary.nonfinite_count,
            auxiliary.clipped_count,
            hard_result.dy_auxiliary_fallback_fraction,
            hard_result.dy_auxiliary_nonfinite_fraction,
            hard_result.dy_auxiliary_clipped_fraction,
        )
        return hard_result

    @staticmethod
    def _summarise_replica_values(
        replica_values: Sequence[float],
    ) -> tuple[float, float]:
        if not replica_values:
            raise pygloopException(
                "A correlated auxiliary estimate requires at least one replica."
            )
        central = math.fsum(replica_values) / len(replica_values)
        if len(replica_values) == 1:
            return central, 0.0
        variance = math.fsum(
            (value - central) ** 2 for value in replica_values
        ) / (len(replica_values) - 1)
        return central, math.sqrt(variance / len(replica_values))

    def _ttbar_decoupling_coefficient(self) -> float:
        if not self.dy_decoupling:
            return 0.0
        # The qg coefficient starts at NLO, so converting alpha_s from the
        # (n_l+1)- to the n_l-flavour scheme first changes it at NNLO.  The
        # NLO decoupling contribution is therefore exactly zero in this channel.
        if self._dy_ttbar_qg_scheme_mode:
            return 0.0
        _lambda_sq, default_mur_sq = DY_DEFAULT_LAMBDA_MUR_SQ[self.n_loops]
        mur_sq = self.dy_mur_sq if self.dy_mur_sq is not None else default_mur_sq
        if not math.isfinite(mur_sq) or mur_sq <= 0.0:
            raise pygloopException(
                "The ttbar decoupling contribution requires finite positive mursq."
            )
        if not math.isfinite(self.m_top) or self.m_top <= 0.0:
            raise pygloopException(
                "The ttbar decoupling contribution requires a finite positive top mass."
            )
        return (
            self.dy_scheme_alpha_s
            / (3.0 * math.pi)
            * math.log(mur_sq / self.m_top**2)
        )

    def _integrate_ttbar_qqbar_auxiliary(
        self,
        seed: int,
        parameterisation: str,
        phase: str,
        workers: int = 1,
    ) -> DYQQbarAuxiliaryResult:
        if not self._dy_ttbar_qqbar_scheme_mode:
            raise pygloopException(
                "The correlated qqbar auxiliary path requires two-loop ttbar qqbar."
            )

        provider: DYPDFProvider | None = None
        muf_sq: float | None = None
        if self.integrate_beams:
            if self.dy_pdf_set is None or self.dy_muf_sq is None:
                raise pygloopException(
                    "Hadronic ttbar qqbar scheme conversion requires PDF setup."
                )
            if self._dy_pdf_provider is None:
                self._dy_pdf_provider = DYPDFProvider(
                    self.dy_pdf_set,
                    self.dy_pdf_member,
                    self.dy_pdf_luminosity_family,
                )
            provider = self._dy_pdf_provider
            muf_sq = float(self.dy_muf_sq)

        # Resolve exactly one generated one-loop Born bundle and reuse the same
        # callable for D_qq and the decoupling Born contribution.
        born_bundle, swap_beams = self._resolve_scheme_born_bundles(
            (self.dy_channel,)
        )[self.dy_channel]
        born_integrand = self._scheme_born_integrand(
            born_bundle,
            swap_beams,
            parameterisation,
            phase,
        )
        raw_two_leg_normalisation = physical_beam_normalisation_factor(
            self.dy_channel,
            1,
        )
        physical_born_normalisation = (
            self._scheme_born_cross_section_factor(born_bundle)
            * raw_two_leg_normalisation
        )
        return integrate_ttbar_qqbar_auxiliary(
            provider=provider,
            channel=self.dy_channel,
            muf_sq=muf_sq,
            e_cm_sq=self.e_cm**2,
            m_top=self.m_top,
            dqq_normalisation=raw_two_leg_normalisation,
            born_normalisation=physical_born_normalisation,
            born_integrand=born_integrand,
            sobol_power=self.dy_scheme_counterterm_sobol_power,
            replicas=self.dy_scheme_counterterm_replicas,
            seed=seed,
            dqq_coefficient=self.dy_scheme_counterterm_factor,
            born_coefficient=self._ttbar_decoupling_coefficient(),
            integrate_beams=self.integrate_beams,
            clip_threshold=self.dy_scheme_counterterm_clip,
            workers=workers,
        )

    def _apply_ttbar_qqbar_scheme_and_decoupling(
        self,
        hard_result: IntegrationResult,
        seed: int,
        parameterisation: str,
        phase: str,
        workers: int = 1,
    ) -> IntegrationResult:
        auxiliary = self._integrate_ttbar_qqbar_auxiliary(
            seed,
            parameterisation,
            phase,
            workers,
        )
        hard_factor = -0.5
        dqq_factor = self.dy_scheme_counterterm_factor
        decoupling_factor = self._ttbar_decoupling_coefficient()

        raw_hard_central = hard_result.central_value
        raw_hard_error = hard_result.error
        applied_hard_central = hard_factor * raw_hard_central
        applied_hard_error = abs(hard_factor) * raw_hard_error
        applied_dqq_central = dqq_factor * auxiliary.dqq_central_value
        applied_dqq_error = abs(dqq_factor) * auxiliary.dqq_error
        applied_born_central = decoupling_factor * auxiliary.born_central_value
        applied_born_error = abs(decoupling_factor) * auxiliary.born_error

        hard_result.dy_scheme_conversion_enabled = True
        hard_result.dy_decoupling_enabled = self.dy_decoupling
        hard_result.dy_scheme_alpha_s = self.dy_scheme_alpha_s
        hard_result.dy_hard_factor = hard_factor
        hard_result.dy_hard_unscaled_central_value = raw_hard_central
        hard_result.dy_hard_unscaled_error = raw_hard_error
        hard_result.dy_hard_central_value = applied_hard_central
        hard_result.dy_hard_error = applied_hard_error

        hard_result.dy_scheme_counterterm_factor = dqq_factor
        hard_result.dy_scheme_counterterm_factor_was_explicit = (
            self.dy_scheme_counterterm_factor_was_explicit
        )
        hard_result.dy_scheme_counterterm_unscaled_central_value = (
            auxiliary.dqq_central_value
        )
        hard_result.dy_scheme_counterterm_unscaled_error = auxiliary.dqq_error
        hard_result.dy_scheme_counterterm_central_value = applied_dqq_central
        hard_result.dy_scheme_counterterm_error = applied_dqq_error
        hard_result.dy_scheme_counterterm_unscaled_replica_values = (
            auxiliary.dqq_replica_values
        )
        hard_result.dy_scheme_counterterm_replica_values = tuple(
            dqq_factor * value for value in auxiliary.dqq_replica_values
        )

        hard_result.dy_decoupling_coefficient = decoupling_factor
        hard_result.dy_decoupling_born_unscaled_central_value = (
            auxiliary.born_central_value
        )
        hard_result.dy_decoupling_born_unscaled_error = auxiliary.born_error
        hard_result.dy_decoupling_born_central_value = applied_born_central
        hard_result.dy_decoupling_born_error = applied_born_error
        hard_result.dy_decoupling_born_unscaled_replica_values = (
            auxiliary.born_replica_values
        )
        hard_result.dy_decoupling_born_replica_values = tuple(
            decoupling_factor * value for value in auxiliary.born_replica_values
        )

        hard_result.dy_auxiliary_central_value = auxiliary.combined_central_value
        hard_result.dy_auxiliary_error = auxiliary.combined_error
        hard_result.dy_auxiliary_replica_values = auxiliary.combined_replica_values
        hard_result.dy_auxiliary_n_samples = auxiliary.n_samples
        hard_result.dy_auxiliary_elapsed_time = auxiliary.elapsed_time
        hard_result.dy_auxiliary_fallback_count = auxiliary.fallback_count
        hard_result.dy_auxiliary_nonfinite_count = auxiliary.nonfinite_count
        hard_result.dy_auxiliary_clipped_count = auxiliary.clipped_count
        hard_result.dy_auxiliary_fallback_fraction = (
            auxiliary.fallback_count / auxiliary.n_samples
            if auxiliary.n_samples
            else 0.0
        )
        hard_result.dy_auxiliary_nonfinite_fraction = (
            auxiliary.nonfinite_count / auxiliary.n_samples
            if auxiliary.n_samples
            else 0.0
        )
        hard_result.dy_auxiliary_clipped_fraction = (
            auxiliary.clipped_count / auxiliary.n_samples
            if auxiliary.n_samples
            else 0.0
        )
        # Keep the established diagnostic names available to downstream tools.
        hard_result.dy_scheme_counterterm_n_samples = auxiliary.n_samples
        hard_result.dy_scheme_counterterm_elapsed_time = auxiliary.elapsed_time
        hard_result.dy_scheme_counterterm_fallback_count = auxiliary.fallback_count
        hard_result.dy_scheme_counterterm_nonfinite_count = auxiliary.nonfinite_count
        hard_result.dy_scheme_counterterm_clipped_count = auxiliary.clipped_count
        hard_result.dy_scheme_counterterm_clipped_fraction = (
            hard_result.dy_auxiliary_clipped_fraction
        )
        hard_result.dy_scheme_counterterm_components = {
            component.label: {
                "unscaled_central_value": component.central_value,
                "unscaled_error": component.error,
                "central_value": dqq_factor * component.central_value,
                "error": abs(dqq_factor) * component.error,
                "unscaled_replica_values": component.replica_values,
                "replica_values": tuple(
                    dqq_factor * value for value in component.replica_values
                ),
            }
            for component in auxiliary.components
        }

        hard_result.central_value = (
            applied_hard_central + auxiliary.combined_central_value
        )
        hard_result.error = math.hypot(applied_hard_error, auxiliary.combined_error)

        logger.info(
            "ttbar qqbar scheme/decoupling flags: scheme=%s decoupling=%s; "
            "coefficients hard=%+.16e Dqq=%+.16e Born=%+.16e",
            True,
            self.dy_decoupling,
            hard_factor,
            dqq_factor,
            decoupling_factor,
        )
        logger.info(
            "ttbar qqbar components: hard=%+.16e +/- %.4e; "
            "Dqq=%+.16e +/- %.4e (raw %+.16e +/- %.4e); "
            "Born=%+.16e +/- %.4e (raw %+.16e +/- %.4e); "
            "auxiliary=%+.16e +/- %.4e; total=%+.16e +/- %.4e",
            applied_hard_central,
            applied_hard_error,
            applied_dqq_central,
            applied_dqq_error,
            auxiliary.dqq_central_value,
            auxiliary.dqq_error,
            applied_born_central,
            applied_born_error,
            auxiliary.born_central_value,
            auxiliary.born_error,
            auxiliary.combined_central_value,
            auxiliary.combined_error,
            hard_result.central_value,
            hard_result.error,
        )
        logger.info(
            "ttbar qqbar auxiliary diagnostics: samples=%d elapsed=%.2fs "
            "fallback/nonfinite/clipped=%d/%d/%d; fractions=%.3e/%.3e/%.3e",
            auxiliary.n_samples,
            auxiliary.elapsed_time,
            auxiliary.fallback_count,
            auxiliary.nonfinite_count,
            auxiliary.clipped_count,
            hard_result.dy_auxiliary_fallback_fraction,
            hard_result.dy_auxiliary_nonfinite_fraction,
            hard_result.dy_auxiliary_clipped_fraction,
        )
        for label, component in hard_result.dy_scheme_counterterm_components.items():
            logger.info(
                "ttbar qqbar Dqq component %s: %+.16e +/- %.4e "
                "(raw %+.16e +/- %.4e)",
                label,
                component["central_value"],
                component["error"],
                component["unscaled_central_value"],
                component["unscaled_error"],
            )
        return hard_result

    def _add_dy_msbar_scheme_counterterm(
        self,
        hard_result: IntegrationResult,
        seed: int,
        parameterisation: str = "spherical",
        phase: str = "real",
        workers: int = 1,
    ) -> IntegrationResult:
        scheme_result = self._integrate_dy_msbar_scheme_counterterm(
            seed,
            parameterisation,
            phase,
            workers,
        )
        scheme_factor = self.dy_scheme_counterterm_factor
        is_one_loop_dy_qqbar = (
            self.process_name.lower() == "dy"
            and self.n_loops == 1
            and self.dy_channel in {(1, -1), (-1, 1)}
        )
        if is_one_loop_dy_qqbar:
            # The generated qqbar forward graphs and qg graphs have opposite
            # relative signs. Convert them to one common DY convention before
            # adding the finite term; both already carry the same coupling
            # normalisation at this point.
            hard_factor = -1.0
        elif (
            self.process_name.lower() == "tt~"
            and self.n_loops == 2
            and self.dy_channel in {(1, 0), (0, 1)}
        ):
            hard_factor = 0.5
        else:
            hard_factor = 1.0
        applied_scheme_factor = scheme_factor
        raw_hard_central_value = hard_result.central_value
        raw_hard_error = hard_result.error
        applied_hard_central_value = hard_factor * raw_hard_central_value
        applied_hard_error = abs(hard_factor) * raw_hard_error
        applied_central_value = (
            applied_scheme_factor * scheme_result.central_value
        )
        applied_error = abs(applied_scheme_factor) * scheme_result.error
        applied_replica_values = tuple(
            applied_scheme_factor * value
            for value in scheme_result.replica_values
        )
        hard_result.dy_hard_factor = hard_factor
        hard_result.dy_hard_unscaled_central_value = raw_hard_central_value
        hard_result.dy_hard_unscaled_error = raw_hard_error
        hard_result.dy_hard_central_value = applied_hard_central_value
        hard_result.dy_hard_error = applied_hard_error
        hard_result.dy_scheme_counterterm_unscaled_central_value = (
            scheme_result.central_value
        )
        hard_result.dy_scheme_counterterm_unscaled_error = scheme_result.error
        hard_result.dy_scheme_counterterm_factor = scheme_factor
        hard_result.dy_scheme_counterterm_applied_factor = applied_scheme_factor
        hard_result.dy_scheme_counterterm_central_value = applied_central_value
        hard_result.dy_scheme_counterterm_error = applied_error
        hard_result.dy_scheme_counterterm_n_samples = scheme_result.n_samples
        hard_result.dy_scheme_counterterm_elapsed_time = scheme_result.elapsed_time
        if scheme_result.method is not None:
            hard_result.dy_scheme_counterterm_method = scheme_result.method
            hard_result.dy_auxiliary_method = scheme_result.method
            hard_result.dy_auxiliary_integration_dimension = scheme_result.integration_dimension
        hard_result.dy_scheme_counterterm_fallback_count = (
            scheme_result.fallback_count
        )
        hard_result.dy_scheme_counterterm_clipped_count = scheme_result.clipped_count
        hard_result.dy_scheme_counterterm_clipped_fraction = (
            scheme_result.clipped_count / scheme_result.n_samples
            if scheme_result.n_samples
            else 0.0
        )
        hard_result.dy_scheme_counterterm_nonfinite_count = (
            scheme_result.nonfinite_count
        )
        hard_result.dy_scheme_counterterm_unscaled_replica_values = (
            scheme_result.replica_values
        )
        hard_result.dy_scheme_counterterm_replica_values = applied_replica_values
        # Expose the same aggregate auxiliary diagnostics as the dedicated
        # ttbar qqbar/gg finalisers.  In the generic path the scheme term is
        # the complete additive auxiliary contribution (and qg decoupling is
        # identically zero at NLO), so these are exact aliases.
        hard_result.dy_auxiliary_central_value = applied_central_value
        hard_result.dy_auxiliary_error = applied_error
        hard_result.dy_auxiliary_replica_values = applied_replica_values
        hard_result.dy_auxiliary_n_samples = scheme_result.n_samples
        hard_result.dy_auxiliary_elapsed_time = scheme_result.elapsed_time
        hard_result.dy_auxiliary_fallback_count = scheme_result.fallback_count
        hard_result.dy_auxiliary_clipped_count = scheme_result.clipped_count
        hard_result.dy_auxiliary_nonfinite_count = scheme_result.nonfinite_count
        hard_result.dy_auxiliary_fallback_fraction = (
            scheme_result.fallback_count / scheme_result.n_samples
            if scheme_result.n_samples
            else 0.0
        )
        hard_result.dy_auxiliary_clipped_fraction = (
            hard_result.dy_scheme_counterterm_clipped_fraction
        )
        hard_result.dy_auxiliary_nonfinite_fraction = (
            scheme_result.nonfinite_count / scheme_result.n_samples
            if scheme_result.n_samples
            else 0.0
        )
        hard_result.dy_scheme_counterterm_components = {
            component.label: {
                "unscaled_central_value": component.central_value,
                "unscaled_error": component.error,
                "central_value": applied_scheme_factor * component.central_value,
                "error": abs(applied_scheme_factor) * component.error,
                "unscaled_replica_values": component.replica_values,
                "replica_values": tuple(
                    applied_scheme_factor * value
                    for value in component.replica_values
                ),
            }
            for component in scheme_result.components
        }
        hard_result.dy_scheme_conversion_enabled = True
        hard_result.dy_scheme_conversion_channel = (
            "qg" if self.dy_channel in {(1, 0), (0, 1)} else "qqbar"
        )
        hard_result.dy_decoupling_enabled = self.dy_decoupling
        is_ttbar_qg = (
            self.process_name.lower() == "tt~"
            and self.n_loops == 2
            and self._dy_ttbar_qg_scheme_mode
        )
        zero_replicas = tuple(0.0 for _value in scheme_result.replica_values)
        hard_result.dy_decoupling_coefficient = 0.0
        hard_result.dy_decoupling_born_unscaled_central_value = 0.0
        hard_result.dy_decoupling_born_unscaled_error = 0.0
        hard_result.dy_decoupling_born_central_value = 0.0
        hard_result.dy_decoupling_born_error = 0.0
        hard_result.dy_decoupling_born_unscaled_replica_values = zero_replicas
        hard_result.dy_decoupling_born_replica_values = zero_replicas
        if is_ttbar_qg:
            hard_result.dy_scheme_conversion_enabled = True
            hard_result.dy_scheme_conversion_channel = "qg"
            hard_result.dy_decoupling_enabled = self.dy_decoupling
            hard_result.dy_decoupling_coefficient = (
                self._ttbar_decoupling_coefficient()
            )
            hard_result.dy_decoupling_born_unscaled_central_value = 0.0
            hard_result.dy_decoupling_born_unscaled_error = 0.0
            hard_result.dy_decoupling_born_central_value = 0.0
            hard_result.dy_decoupling_born_error = 0.0
            hard_result.dy_decoupling_born_unscaled_replica_values = zero_replicas
            hard_result.dy_decoupling_born_replica_values = zero_replicas
        hard_result.central_value = (
            applied_hard_central_value + applied_central_value
        )
        hard_result.error = math.hypot(applied_hard_error, applied_error)
        channel_label = (
            "gq" if self.dy_channel in {(1, 0), (0, 1)} else "qqbar"
        )
        process_label = (
            "ttbar" if self.process_name.lower() == "tt~" else "DY"
        )
        logger.info(
            "%s MSbar %s scheme counterterm: %+.16e +/- %.4e "
            "(raw %+.16e +/- %.4e; factor %+.16e; %d Sobol samples in %.2fs); "
            "fallback/clipped/nonfinite = %d/%d (%.3e)/%d; "
            "combined: %+.16e +/- %.4e",
            process_label,
            channel_label,
            applied_central_value,
            applied_error,
            scheme_result.central_value,
            scheme_result.error,
            scheme_factor,
            scheme_result.n_samples,
            scheme_result.elapsed_time,
            scheme_result.fallback_count,
            scheme_result.clipped_count,
            hard_result.dy_scheme_counterterm_clipped_fraction,
            scheme_result.nonfinite_count,
            hard_result.central_value,
            hard_result.error,
        )
        if is_ttbar_qg:
            logger.info(
                "ttbar qg decoupling: enabled=%s coefficient=%+.16e; "
                "Born contribution=%+.16e +/- %.4e (zero at NLO)",
                self.dy_decoupling,
                hard_result.dy_decoupling_coefficient,
                hard_result.dy_decoupling_born_central_value,
                hard_result.dy_decoupling_born_error,
            )
        for label, component in hard_result.dy_scheme_counterterm_components.items():
            logger.info(
                "%s MSbar %s component %s: %+.16e +/- %.4e",
                process_label,
                channel_label,
                label,
                component["central_value"],
                component["error"],
            )
        return self._attach_dy_coupling_normalisation_metadata(hard_result)

    @staticmethod
    def _apply_dy_pdf_luminosity(
        hard_weight: float,
        luminosity: float,
        xs: list[float],
    ) -> float:
        weighted = hard_weight * luminosity
        if not math.isfinite(weighted):
            raise pygloopException(
                "The PDF-weighted DY sample is non-finite at "
                f"xs={list(xs)}."
            )
        return weighted

    def _apply_dy_beam_weights(
        self,
        hard_weight: float,
        pdf_luminosity: float | None,
        xs: list[float],
    ) -> float:
        weighted = hard_weight
        if pdf_luminosity is not None:
            weighted = self._apply_dy_pdf_luminosity(
                weighted,
                pdf_luminosity,
                xs,
            )
        if self.dy_physical_normalisation:
            weighted *= self.dy_physical_normalisation_factor
            if not math.isfinite(weighted):
                raise pygloopException(
                    "The physically normalised DY beam sample is non-finite at "
                    f"xs={list(xs)}."
                )
        if self.dy_integrated_leptonic_phase_space:
            weighted *= self.dy_integrated_leptonic_phase_space_factor
            if not math.isfinite(weighted):
                raise pygloopException(
                    "The DY sample with integrated leptonic phase space is "
                    f"non-finite at xs={list(xs)}."
                )
        if self.dy_coupling_normalisation_applied:
            weighted *= self.dy_coupling_normalisation_factor
            if not math.isfinite(weighted):
                raise pygloopException(
                    "The coupling-normalised DY sample is non-finite at "
                    f"xs={list(xs)}."
                )
        return weighted

    def _attach_dy_coupling_normalisation_metadata(
        self, result: IntegrationResult
    ) -> IntegrationResult:
        result.dy_coupling_normalisation_applied = (
            self.dy_coupling_normalisation_applied
        )
        result.dy_coupling_normalisation_factor = (
            self.dy_coupling_normalisation_factor
        )
        result.dy_coupling_normalisation_convention = (
            self.dy_coupling_normalisation_convention
        )
        result.dy_coupling_normalisation_includes_unit_conversion = False
        return result

    def _rescale_dy_beam_stability_thresholds(
        self,
        precision_threshold: float | None,
        clip_threshold: float | None,
        pdf_luminosity: float | None,
    ) -> tuple[float | None, float | None]:
        """Express final beam-weight thresholds in pre-weight integrand units."""

        beam_weight_scale = 1.0
        if pdf_luminosity is not None:
            beam_weight_scale *= abs(pdf_luminosity)
        if self.dy_physical_normalisation:
            beam_weight_scale *= abs(self.dy_physical_normalisation_factor)
        if self.dy_integrated_leptonic_phase_space:
            beam_weight_scale *= abs(
                self.dy_integrated_leptonic_phase_space_factor
            )
        if self.dy_coupling_normalisation_applied:
            beam_weight_scale *= abs(self.dy_coupling_normalisation_factor)
        if not math.isfinite(beam_weight_scale):
            raise pygloopException(
                "The DY beam-weight scale for stability thresholds is non-finite."
            )
        if beam_weight_scale == 0.0:
            return None, None
        return (
            precision_threshold / beam_weight_scale
            if precision_threshold is not None
            else None,
            clip_threshold / beam_weight_scale
            if clip_threshold is not None
            else None,
        )

    def ttbar_beam_threshold_passes(
        self,
        x1: float,
        x2: float,
        m_top: float | Decimal | None = None,
    ) -> bool:
        if not self.enforce_ttbar_beam_threshold:
            return True
        mt = float(self.m_top if m_top is None else m_top)
        return float(x1) * float(x2) * (self.e_cm**2) >= 4.0 * (mt**2)

    def _effective_ttbar_beam_mass(
        self,
        integrand_implementation: Mapping[str, Any],
        decimal_digit_precision: int | None = None,
    ) -> float | Decimal:
        # Normal integration setup already resolves the native value once in
        # the constructor. Avoid registry/cache work on every sampled point.
        if decimal_digit_precision is None:
            return float(self.m_top)
        runtime_parameters = integrand_implementation.get(
            "dy_runtime_parameters", self.dy_runtime_parameters
        )
        if self.compiled_bundle is not None:
            resolved = self.compiled_bundle.resolve_runtime_parameters(
                runtime_parameters,
                decimal_digit_precision=decimal_digit_precision,
            )
            if "m_top" in resolved:
                return resolved["m_top"]
        return decimal_from_input(self.m_top)

    def dy_physical_z_interval(
        self, x1: float, x2: float
    ) -> tuple[float, float] | None:
        z_min, z_max = self.dy_z_bin if self.dy_z_bin is not None else (0.0, 1.0)
        shat = float(x1) * float(x2) * (self.e_cm**2)
        if shat <= 0.0:
            return None
        if self.dy_q_min is not None:
            z_min = max(z_min, self.dy_q_min**2 / shat)
        if self.dy_q_max is not None:
            z_max = min(z_max, self.dy_q_max**2 / shat)
        if z_min >= z_max:
            return None
        return z_min, z_max

    def _soft_mirror_routing(
        self,
        integrand_implementation: dict[str, Any],
        channel_selector: int | None,
    ) -> SoftEdgeRouting | None:
        specifications = integrand_implementation.get("dy_soft_mirror_edges")
        if specifications is None:
            return None
        if self.compiled_bundle is None:
            raise pygloopException(
                "DY soft mirroring requires a compiled bundle with edge-routing "
                "metadata. Regenerate and load the zenos bundle."
            )
        normalised = self.compiled_bundle._normalise_soft_mirror_edge_specs(
            specifications
        )
        cache_key = ("dy_soft_mirror_routing", normalised, channel_selector)
        if cache_key not in self.cache:
            self.cache[cache_key] = self.compiled_bundle.resolve_soft_mirror_routing(
                normalised,
                channel_selector,
            )
        routing = self.cache[cache_key]
        return routing if isinstance(routing, SoftEdgeRouting) else None

    def _record_soft_mirror_pair(
        self,
        xs: list[float],
        first: float | Decimal,
        mirrored: float | Decimal,
    ) -> None:
        try:
            first_float = float(first)
        except (OverflowError, ValueError):
            first_float = math.copysign(math.inf, -1.0 if first < 0 else 1.0)
        try:
            mirrored_float = float(mirrored)
        except (OverflowError, ValueError):
            mirrored_float = math.copysign(
                math.inf, -1.0 if mirrored < 0 else 1.0
            )

        for side, value in (("original", first_float), ("mirror", mirrored_float)):
            if self.soft_mirror_max_raw_side_wgt is None or abs(value) > abs(
                self.soft_mirror_max_raw_side_wgt
            ):
                self.soft_mirror_max_raw_side_wgt = value
                self.soft_mirror_max_raw_side_wgt_point = list(xs)
                self.soft_mirror_max_raw_side = side

        average = 0.5 * (first_float + mirrored_float)
        if self.soft_mirror_max_post_average_wgt is None or abs(average) > abs(
            self.soft_mirror_max_post_average_wgt
        ):
            self.soft_mirror_max_post_average_wgt = average
            self.soft_mirror_max_post_average_wgt_point = list(xs)

        side_scale = max(abs(first_float), abs(mirrored_float))
        if side_scale > 0.0 and math.isfinite(side_scale) and math.isfinite(average):
            residual_ratio = abs(average) / side_scale
            if (
                self.soft_mirror_min_residual_ratio is None
                or residual_ratio < self.soft_mirror_min_residual_ratio
            ):
                self.soft_mirror_min_residual_ratio = residual_ratio
                self.soft_mirror_min_residual_ratio_point = list(xs)
                self.soft_mirror_min_residual_ratio_sides = (
                    first_float,
                    mirrored_float,
                )

    @staticmethod
    def _rotation_descriptors_from_xs(
        xs: list[float], count: int
    ) -> tuple[RotationDescriptor, ...]:
        return rotation_descriptors_from_xs(xs, count)

    @staticmethod
    def _rotation_matrix_from_xs(xs: list[float]):
        """Compatibility helper returning the first production f64 rotation."""
        descriptor = rotation_descriptors_from_xs(xs, 1)[0]
        return rotation_matrix_from_descriptor(descriptor)

    @staticmethod
    def _rotation_check_count(integrand_implementation: Mapping[str, Any]) -> int:
        value = integrand_implementation.get("dy_rotation_check_count", 1)
        if value is None:
            value = 1
        try:
            count = int(value)
        except (TypeError, ValueError) as exc:
            raise pygloopException(
                "DY rotation-check count must be a positive integer."
            ) from exc
        if isinstance(value, bool) or count < 1 or str(value).strip() != str(count):
            raise pygloopException(
                "DY rotation-check count must be a positive integer."
            )
        return count

    @staticmethod
    def _rotate_vec(v: Vector, rmat) -> Vector:
        return rotate_vector(v, rmat)

    @staticmethod
    def _validated_optional_threshold(value: Any, option_name: str) -> float | None:
        if value is None:
            return None
        threshold = float(value)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise pygloopException(f"{option_name} must be finite and strictly positive.")
        return threshold

    @classmethod
    def _stability_thresholds(
        cls, integrand_implementation: dict[str, Any]
    ) -> tuple[float | None, float | None]:
        precision_value = integrand_implementation.get("dy_large_weight_precision")
        if precision_value is None:
            precision_value = integrand_implementation.get("dy_large_weight_threshold")
        clip_value = integrand_implementation.get("dy_large_weight_clip")
        if (
            clip_value is None
            and bool(integrand_implementation.get("dy_zero_large_weight_samples", False))
        ):
            clip_value = integrand_implementation.get("dy_large_weight_threshold")

        precision_threshold = cls._validated_optional_threshold(
            precision_value, "DY large-weight precision threshold"
        )
        clip_threshold = cls._validated_optional_threshold(
            clip_value, "DY large-weight clip threshold"
        )
        if (
            precision_threshold is not None
            and clip_threshold is not None
            and precision_threshold > clip_threshold
        ):
            raise pygloopException(
                "DY large-weight precision threshold must not exceed the clip "
                "threshold; clipped points must first pass higher-precision validation."
            )
        return precision_threshold, clip_threshold

    @staticmethod
    def _stability_tolerances(
        integrand_implementation: dict[str, Any], rotation_digits: int
    ) -> tuple[float, float]:
        default_relative = 10.0 ** (-rotation_digits) if rotation_digits > 0 else 1.0e-6
        relative_value = integrand_implementation.get("dy_stability_rtol")
        relative = float(
            default_relative if relative_value is None else relative_value
        )
        absolute_value = integrand_implementation.get("dy_stability_atol")
        if absolute_value is None:
            absolute_value = integrand_implementation.get(
                "dy_rotation_check_eps", 1.0e-15
            )
        absolute = float(absolute_value)
        if not math.isfinite(relative) or relative < 0.0:
            raise pygloopException(
                "DY stability relative tolerance must be finite and non-negative."
            )
        if not math.isfinite(absolute) or absolute < 0.0:
            raise pygloopException(
                "DY stability absolute tolerance must be finite and non-negative."
            )
        if relative == 0.0 and absolute == 0.0:
            raise pygloopException("DY stability tolerances cannot both be zero.")
        return relative, absolute

    @staticmethod
    def _phase_value(value: complex, phase: str) -> float:
        as_complex = complex(value)
        if phase == "real":
            return as_complex.real
        if phase == "imag":
            return as_complex.imag
        raise pygloopException(f"Unsupported integration phase {phase!r}.")

    def _record_hp_failure(
        self,
        xs: list[float],
        momentum_point: str,
        reason: str,
    ) -> None:
        if self.stability_hp_failure_example is None:
            self.stability_hp_failure_example = list(xs)
            self.stability_hp_failure_example_momentum_point = momentum_point
            self.stability_hp_failure_reason = reason

    def _record_t_solver_failure(
        self,
        failed_terms: set[str],
        failed_surfaces: set[str],
    ) -> None:
        """Record one Monte-Carlo sample that lost float E-surface roots."""
        terms = failed_terms or {"unknown-term"}
        surfaces = failed_surfaces or {"unknown-surface"}
        self.t_solver_float_failure_sample_count += 1
        self.t_solver_float_failed_term_count += len(terms)
        self.t_solver_float_failed_surface_count += len(surfaces)
        for term in terms:
            self.t_solver_failure_terms[term] = (
                self.t_solver_failure_terms.get(term, 0) + 1
            )
        for surface in surfaces:
            self.t_solver_failure_surfaces[surface] = (
                self.t_solver_failure_surfaces.get(surface, 0) + 1
            )

    @staticmethod
    def _coherent_high_precision_muv(
        integrand_implementation: Mapping[str, Any],
    ) -> Decimal | None:
        """Upcast positional mUV without conflicting with its named source."""
        if "mUV" not in integrand_implementation:
            return None

        positional_muv = integrand_implementation["mUV"]
        runtime_parameters = integrand_implementation.get(
            "dy_runtime_parameters"
        )
        if (
            not isinstance(runtime_parameters, Mapping)
            or "muv" not in runtime_parameters
        ):
            return decimal_from_input(positional_muv)

        # Native conflict checks use repr(float), which is also how scalar CLI
        # aliases enter the named registry. Compare that public spelling first,
        # then retain the named decimal instead of upcasting the binary float.
        try:
            named_muv = Decimal(
                exact_decimal_string(runtime_parameters["muv"], name="muv")
            )
            positional_decimal = Decimal(
                exact_decimal_string(positional_muv, name="muv")
            )
        except ValueError as exc:
            raise pygloopException(str(exc)) from exc
        if named_muv != positional_decimal:
            raise pygloopException(
                "Conflicting DY runtime values were supplied for 'muv'."
            )
        return named_muv

    def _evaluate_stability_hp_pair(
        self,
        xs: list[float],
        parameterization: str,
        integrand_implementation: dict[str, Any],
        phase: str,
        channel_selector: int | None,
        expects_z: bool,
        expects_beam_fractions: bool,
        rotation_descriptors: Sequence[RotationDescriptor] | Any,
        decimal_digit_precision: int,
        relative_tolerance: float,
        absolute_tolerance: float,
        soft_mirror_routing: SoftEdgeRouting | None = None,
    ) -> tuple[Decimal | None, Decimal | None, str | None, str | None]:
        if self.compiled_bundle is None:
            raise pygloopException(
                "Higher-precision DY validation requires a compiled zenos bundle."
            )
        if decimal_digit_precision < 2:
            raise pygloopException(
                "Higher-precision DY validation requires at least two digits."
            )
        if not rotation_descriptors:
            raise pygloopException(
                "Higher-precision DY validation requires at least one rotation."
            )
        if not isinstance(rotation_descriptors[0], RotationDescriptor):
            # Older replay helpers passed an already-rounded matrix. Re-derive
            # its descriptor from xs rather than promoting that f64 matrix.
            rotation_descriptors = self._rotation_descriptors_from_xs(xs, 1)
        else:
            rotation_descriptors = tuple(rotation_descriptors)
        self.compiled_bundle.require_fallback_supported(decimal_digit_precision)
        try:
            precise_m_uv = self._coherent_high_precision_muv(
                integrand_implementation
            )
            runtime_parameters = integrand_implementation.get(
                "dy_runtime_parameters"
            )
            soft_center_resolver = None
            if (
                soft_mirror_routing is not None
                and soft_mirror_routing.has_external_offset
            ):

                def resolve_soft_center(loop_momenta, p1, p2, z):
                    assert self.compiled_bundle is not None
                    return self.compiled_bundle.soft_center_for_routing(
                        list(loop_momenta),
                        p1,
                        p2,
                        z,
                        precise_m_uv,
                        soft_mirror_routing,
                        channel_selector,
                        decimal_digit_precision=decimal_digit_precision,
                        runtime_parameters=runtime_parameters,
                    )

                soft_center_resolver = resolve_soft_center

            sample = build_high_precision_sample(
                xs,
                n_loops=self.n_loops,
                parameterization=parameterization,
                incoming_momenta=(self.ps_point[0], self.ps_point[1]),
                expects_z=expects_z,
                expects_beam_fractions=expects_beam_fractions,
                rescaling=RESCALING,
                decimal_digit_precision=decimal_digit_precision,
                beam_parameterisation=str(
                    integrand_implementation.get(
                        "dy_beam_parameterisation", "x1_x2"
                    )
                ),
                beam_threshold_mass=(
                    self._effective_ttbar_beam_mass(
                        integrand_implementation,
                        decimal_digit_precision,
                    )
                    if integrand_implementation.get(
                        "dy_beam_parameterisation", "x1_x2"
                    )
                    == "beta_y"
                    else None
                ),
                soft_mirror_routing=soft_mirror_routing,
                soft_center_resolver=soft_center_resolver,
            )
            mirrored_sample = (
                sample.soft_mirrored(soft_mirror_routing)
                if soft_mirror_routing is not None
                else None
            )
            rotations = tuple(
                rotation_matrix_from_descriptor(
                    descriptor,
                    decimal_digit_precision=decimal_digit_precision,
                )
                for descriptor in rotation_descriptors
            )
            rotated_samples = tuple(
                sample.rotated(rotation) for rotation in rotations
            )
            rotated_mirrored_samples = (
                tuple(
                    mirrored_sample.rotated(rotation) for rotation in rotations
                )
                if mirrored_sample is not None
                else ()
            )
            hp_impl = dict(integrand_implementation)
            hp_impl["dy_evaluation_mode"] = "arb"
            hp_impl["dy_fallback_precision"] = decimal_digit_precision
            hp_impl["dy_rotation_check_arb_digits"] = decimal_digit_precision
            hp_impl["z"] = sample.z
            if "mUV" in hp_impl:
                hp_impl["mUV"] = precise_m_uv

            histogram = hp_impl.get("_dy_histogram")
            histogram_orbits = []
            if histogram is not None:
                histogram.reset()
                hp_impl["_dy_histogram_orbits"] = histogram_orbits

            first_total, _first_terms = self._zenos_arb_terms_with_externals(
                list(sample.loop_momenta),
                sample.p1,
                sample.p2,
                hp_impl,
                decimal_digit_precision,
                channel_selector=channel_selector,
            )
            rotated_totals = tuple(
                self._zenos_arb_terms_with_externals(
                    list(rotated_sample.loop_momenta),
                    rotated_sample.p1,
                    rotated_sample.p2,
                    hp_impl,
                    decimal_digit_precision,
                    channel_selector=channel_selector,
                )[0]
                for rotated_sample in rotated_samples
            )
            mirrored_total: Decimal | None = None
            rotated_mirrored_totals: tuple[Decimal, ...] = ()
            if mirrored_sample is not None:
                mirrored_total, _mirrored_terms = self._zenos_arb_terms_with_externals(
                    list(mirrored_sample.loop_momenta),
                    mirrored_sample.p1,
                    mirrored_sample.p2,
                    hp_impl,
                    decimal_digit_precision,
                    channel_selector=channel_selector,
                )
                rotated_mirrored_totals = tuple(
                    self._zenos_arb_terms_with_externals(
                        list(rotated_mirrored_sample.loop_momenta),
                        rotated_mirrored_sample.p1,
                        rotated_mirrored_sample.p2,
                        hp_impl,
                        decimal_digit_precision,
                        channel_selector=channel_selector,
                    )[0]
                    for rotated_mirrored_sample in rotated_mirrored_samples
                )

            with localcontext() as context:
                context.prec = decimal_digit_precision + 12
                if phase == "real":
                    first_weight = first_total * sample.jacobian
                    rotated_weights = tuple(
                        rotated_total * rotated_sample.jacobian
                        for rotated_total, rotated_sample in zip(
                            rotated_totals, rotated_samples, strict=True
                        )
                    )
                    mirrored_weight = (
                        mirrored_total * mirrored_sample.jacobian
                        if mirrored_total is not None and mirrored_sample is not None
                        else None
                    )
                    rotated_mirrored_weights = tuple(
                        rotated_total * rotated_sample.jacobian
                        for rotated_total, rotated_sample in zip(
                            rotated_mirrored_totals,
                            rotated_mirrored_samples,
                            strict=True,
                        )
                    )
                elif phase == "imag":
                    first_weight = Decimal(0)
                    rotated_weights = tuple(
                        Decimal(0) for _ in rotated_samples
                    )
                    mirrored_weight = (
                        Decimal(0) if mirrored_sample is not None else None
                    )
                    rotated_mirrored_weights = tuple(
                        Decimal(0) for _ in rotated_mirrored_samples
                    )
                else:
                    raise pygloopException(f"Unsupported integration phase {phase!r}.")

                all_weights = [first_weight, *rotated_weights]
                if mirrored_weight is not None:
                    all_weights.append(mirrored_weight)
                all_weights.extend(rotated_mirrored_weights)
                packet_orbits = hp_impl.get("_dy_packet_orbits")
                if histogram is not None:
                    samples = [sample, *rotated_samples]
                    if mirrored_sample is not None:
                        samples.extend([mirrored_sample, *rotated_mirrored_samples])
                    weighted_bins = [
                        [value * orbit_sample.jacobian if phase == "real" else Decimal(0)
                         for value in bins]
                        for bins, orbit_sample in zip(histogram_orbits, samples, strict=True)
                    ]
                    all_weights.extend(value for bins in weighted_bins for value in bins)
                    orbit_groups = [(0, range(1, 1+len(rotations)))]
                    if mirrored_sample is not None:
                        start = 1+len(rotations)
                        orbit_groups.append((start, range(start+1, len(weighted_bins))))
                    for original_index, rotated_indices in (() if packet_orbits is not None else orbit_groups):
                        for rotated_index in rotated_indices:
                            for bin_index, (first, rotated) in enumerate(zip(
                                    weighted_bins[original_index], weighted_bins[rotated_index], strict=True)):
                                if not decimal_values_agree(
                                        first, rotated,
                                        relative_tolerance=Decimal(str(relative_tolerance)),
                                        absolute_tolerance=Decimal(str(absolute_tolerance)),
                                        decimal_digit_precision=decimal_digit_precision):
                                    return None, None, "disagreement", f"histogram bin {bin_index} rotation disagrees"
                if any(not weight.is_finite() for weight in all_weights):
                    return (
                        None,
                        None,
                        "nonfinite",
                        "higher-precision stability orbit is non-finite",
                    )
                if packet_orbits is not None:
                    # The sampler assembles all angular partners/graphs before
                    # making the SAME stability decision on their packet sum.
                    values = [first_weight, *rotated_weights]
                    bins = weighted_bins[:1+len(rotations)] if histogram is not None else None
                    if mirrored_weight is not None:
                        values = [(a+b)/2 for a, b in zip(values, [mirrored_weight, *rotated_mirrored_weights], strict=True)]
                        if bins is not None:
                            bins = [[(a+b)/2 for a, b in zip(first, mirror, strict=True)]
                                    for first, mirror in zip(bins, weighted_bins[1+len(rotations):], strict=True)]
                    packet_orbits.append({"values": values, "bins": bins})
                    if histogram is not None:
                        histogram.accept(bins[0])
                    return values[0], first_total, None, None
                for rotation_index, rotated_weight in enumerate(
                    rotated_weights, start=1
                ):
                    if not decimal_values_agree(
                        first_weight,
                        rotated_weight,
                        relative_tolerance=Decimal(str(relative_tolerance)),
                        absolute_tolerance=Decimal(str(absolute_tolerance)),
                        decimal_digit_precision=decimal_digit_precision,
                    ):
                        return (
                            None,
                            None,
                            "disagreement",
                            "higher-precision original/rotated pair disagrees "
                            f"for rotation {rotation_index}",
                        )
                if mirrored_weight is not None:
                    for rotation_index, rotated_mirrored_weight in enumerate(
                        rotated_mirrored_weights, start=1
                    ):
                        if decimal_values_agree(
                            mirrored_weight,
                            rotated_mirrored_weight,
                            relative_tolerance=Decimal(str(relative_tolerance)),
                            absolute_tolerance=Decimal(str(absolute_tolerance)),
                            decimal_digit_precision=decimal_digit_precision,
                        ):
                            continue
                        return (
                            None,
                            None,
                            "disagreement",
                            "higher-precision mirror/rotated-mirror pair "
                            f"disagrees for rotation {rotation_index}",
                        )
                    assert mirrored_total is not None
                    self._record_soft_mirror_pair(xs, first_weight, mirrored_weight)
                    if histogram is not None:
                        histogram.accept((a+b)/2 for a, b in zip(
                            weighted_bins[0], weighted_bins[1+len(rotations)], strict=True))
                    return (
                        +(first_weight + mirrored_weight) / Decimal(2),
                        +(first_total + mirrored_total) / Decimal(2),
                        None,
                        None,
                    )
                if histogram is not None:
                    histogram.accept(weighted_bins[0])
                return +first_weight, +first_total, None, None
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if getattr(self.compiled_bundle, "_dy_cm_reduced", False):
                if isinstance(exc, (DYNumericalEvaluationError, ArithmeticError)):
                    return None, None, "numerical", f"{type(exc).__name__}: {exc}"
                raise pygloopException(
                    f"Unexpected DY HP failure at xs={xs}, graph={channel_selector}: {exc}"
                ) from exc
            return None, None, "error", f"{type(exc).__name__}: {exc}"

    def _evaluate_zenos_stability_sample(
        self,
        xs: list[float],
        parameterization: str,
        integrand_implementation: dict[str, Any],
        phase: str,
        channel_selector: int | None,
        expects_z: bool,
        expects_beam_fractions: bool,
        loop_momenta: list[Vector],
        p1: Vector,
        p2: Vector,
        total_jacobian: float,
        momentum_point: str,
        rotation_digits: int,
        precision_threshold: float | None,
        clip_threshold: float | None,
        soft_mirror_routing: SoftEdgeRouting | None = None,
        soft_center: Vector | None = None,
    ) -> float:
        relative_tolerance, absolute_tolerance = self._stability_tolerances(
            integrand_implementation, rotation_digits
        )
        precision_ladder = self._dy_precision_ladder(integrand_implementation)
        self._count_precision(16, "attempt")
        rotation_count = self._rotation_check_count(integrand_implementation)
        rotation_descriptors = self._rotation_descriptors_from_xs(
            xs, rotation_count
        )
        rotations = tuple(
            rotation_matrix_from_descriptor(descriptor)
            for descriptor in rotation_descriptors
        )
        float_impl = dict(integrand_implementation)
        float_impl["dy_evaluation_mode"] = "compiled"
        failed_root_terms: set[str] = set()
        failed_root_surfaces: set[str] = set()

        def record_root_failure(exc: DYDoubleRootFailure) -> None:
            failed_root_terms.update(exc.diagnostics.failed_terms)
            failed_root_surfaces.update(exc.diagnostics.failed_surfaces)

        if soft_mirror_routing is not None:
            self.soft_mirror_pair_count += 1

        trigger_thresholds = [
            threshold
            for threshold in (precision_threshold, clip_threshold)
            if threshold is not None
        ]

        def exceeds_precision_trigger(value: float) -> bool:
            return any(abs(value) > threshold for threshold in trigger_thresholds)

        first_weight: complex | None = None
        first_final = math.nan
        float_error: str | None = None
        fallback_reason: str | None = None
        try:
            first_weight = self.zenos_integrand_with_externals(
                loop_momenta,
                p1,
                p2,
                float_impl,
                channel_selector=channel_selector,
            )
            first_phase = self._phase_value(first_weight, phase)
            first_final = first_phase * total_jacobian
        except DYDoubleRootFailure as exc:
            record_root_failure(exc)
            fallback_reason = "t_solver_failure"
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            float_error = f"{type(exc).__name__}: {exc}"

        large_trigger = False
        large_trigger_value = first_final
        if fallback_reason is None and (
            float_error is not None or not math.isfinite(first_final)
        ):
            fallback_reason = "float_nonfinite"
        elif fallback_reason is None and exceeds_precision_trigger(first_final):
            large_trigger = True
            # Deliberately do not spend time on a float rotation here.
            fallback_reason = "large_weight"

        mirrored_momenta: tuple[Vector, ...] | None = None
        mirrored_weight: complex | None = None
        mirrored_final = math.nan
        if fallback_reason is None and soft_mirror_routing is not None:
            try:
                mirrored_momenta = mirror_loop_momenta_for_soft_edge(
                    loop_momenta,
                    p1,
                    p2,
                    soft_mirror_routing,
                    soft_center,
                )
                mirrored_weight = self.zenos_integrand_with_externals(
                    list(mirrored_momenta),
                    p1,
                    p2,
                    float_impl,
                    channel_selector=channel_selector,
                )
                mirrored_final = (
                    self._phase_value(mirrored_weight, phase) * total_jacobian
                )
            except DYDoubleRootFailure as exc:
                record_root_failure(exc)
                fallback_reason = "t_solver_failure"
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                float_error = f"{type(exc).__name__}: {exc}"
            if fallback_reason is None and (
                float_error is not None or not math.isfinite(mirrored_final)
            ):
                fallback_reason = "float_nonfinite"
            elif fallback_reason is None and exceeds_precision_trigger(
                mirrored_final
            ):
                large_trigger = True
                large_trigger_value = mirrored_final
                fallback_reason = "large_weight"

        forced_fallback_reason = integrand_implementation.get(
            "_dy_force_stability_hp_reason"
        )
        if str(forced_fallback_reason) == "t_solver_failure":
            failed_root_terms.update(
                str(term)
                for term in integrand_implementation.get(
                    "_dy_force_t_solver_failed_terms", ()
                )
            )
            failed_root_surfaces.update(
                str(surface)
                for surface in integrand_implementation.get(
                    "_dy_force_t_solver_failed_surfaces", ()
                )
            )
        if fallback_reason is None and forced_fallback_reason is not None:
            forced_fallback_reason = str(forced_fallback_reason)
            if forced_fallback_reason not in {
                "float_mismatch",
                "float_nonfinite",
                "large_weight",
                "t_solver_failure",
            }:
                raise pygloopException(
                    "Internal DY forced-HP reason must be 'float_mismatch', "
                    "'float_nonfinite', 'large_weight', or "
                    "'t_solver_failure'."
                )
            fallback_reason = forced_fallback_reason
            if forced_fallback_reason == "large_weight":
                large_trigger = True
                large_trigger_value = first_final

        if fallback_reason is None and _diagnostic_soft_radius_hp_trigger(
            xs, soft_mirror_routing
        ):
            fallback_reason = "float_mismatch"

        if large_trigger and soft_mirror_routing is not None:
            self.soft_mirror_large_trigger_count += 1

        float_relative_difference = math.inf
        if fallback_reason is None:
            for rotation in rotations:
                rotated_final = math.nan
                rotated_momenta = [
                    self._rotate_vec(momentum, rotation)
                    for momentum in loop_momenta
                ]
                rotated_p1 = self._rotate_vec(p1, rotation)
                rotated_p2 = self._rotate_vec(p2, rotation)
                try:
                    rotated_weight = self.zenos_integrand_with_externals(
                        rotated_momenta,
                        rotated_p1,
                        rotated_p2,
                        float_impl,
                        channel_selector=channel_selector,
                    )
                    rotated_final = (
                        self._phase_value(rotated_weight, phase) * total_jacobian
                    )
                except DYDoubleRootFailure as exc:
                    record_root_failure(exc)
                    fallback_reason = "t_solver_failure"
                except BaseException as exc:
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    float_error = f"{type(exc).__name__}: {exc}"

                if fallback_reason is None and not math.isfinite(rotated_final):
                    fallback_reason = "float_nonfinite"
                    self.nan_weight_rotated_count += 1
                    if self.nan_weight_rotated_example is None:
                        self.nan_weight_rotated_example = list(xs)
                        self.nan_weight_rotated_example_momentum_point = momentum_point
                elif fallback_reason is None and not float_values_agree(
                    first_final,
                    rotated_final,
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance,
                ):
                    fallback_reason = "float_mismatch"
                    scale = max(abs(first_final), abs(rotated_final))
                    float_relative_difference = (
                        abs(first_final - rotated_final) / scale
                        if scale > 0.0
                        else math.inf
                    )
                if fallback_reason is not None:
                    break

                if soft_mirror_routing is not None:
                    assert mirrored_momenta is not None
                    rotated_mirrored_final = math.nan
                    rotated_mirrored_momenta = [
                        self._rotate_vec(momentum, rotation)
                        for momentum in mirrored_momenta
                    ]
                    try:
                        rotated_mirrored_weight = (
                            self.zenos_integrand_with_externals(
                                rotated_mirrored_momenta,
                                rotated_p1,
                                rotated_p2,
                                float_impl,
                                channel_selector=channel_selector,
                            )
                        )
                        rotated_mirrored_final = (
                            self._phase_value(rotated_mirrored_weight, phase)
                            * total_jacobian
                        )
                    except DYDoubleRootFailure as exc:
                        record_root_failure(exc)
                        fallback_reason = "t_solver_failure"
                    except BaseException as exc:
                        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                            raise
                        float_error = f"{type(exc).__name__}: {exc}"

                    if fallback_reason is None and not math.isfinite(
                        rotated_mirrored_final
                    ):
                        fallback_reason = "float_nonfinite"
                        self.nan_weight_rotated_count += 1
                        if self.nan_weight_rotated_example is None:
                            self.nan_weight_rotated_example = list(xs)
                            self.nan_weight_rotated_example_momentum_point = (
                                momentum_point
                            )
                    elif fallback_reason is None and not float_values_agree(
                        mirrored_final,
                        rotated_mirrored_final,
                        relative_tolerance=relative_tolerance,
                        absolute_tolerance=absolute_tolerance,
                    ):
                        fallback_reason = "float_mismatch"
                        scale = max(
                            abs(mirrored_final), abs(rotated_mirrored_final)
                        )
                        float_relative_difference = (
                            abs(mirrored_final - rotated_mirrored_final) / scale
                            if scale > 0.0
                            else math.inf
                        )
                    if fallback_reason is not None:
                        break

            if fallback_reason is None:
                self.stability_float_pair_accepted_count += rotation_count * (
                    2 if soft_mirror_routing is not None else 1
                )

        accepted_final: float | Decimal
        accepted_unweighted: float | Decimal
        if fallback_reason is None:
            first_unweighted = (
                self._phase_value(first_weight, phase) if first_weight is not None else 0.0
            )
            if soft_mirror_routing is not None:
                assert mirrored_weight is not None
                accepted_final = 0.5 * (first_final + mirrored_final)
                accepted_unweighted = 0.5 * (
                    first_unweighted + self._phase_value(mirrored_weight, phase)
                )
                self._record_soft_mirror_pair(xs, first_final, mirrored_final)
            else:
                accepted_final = first_final
                accepted_unweighted = first_unweighted
            self._count_precision(16, "accepted")
        else:
            self._count_precision(16, "failed")
            self.stability_hp_retry_count += 1
            if soft_mirror_routing is not None:
                self.soft_mirror_hp_orbit_retry_count += 1
            if fallback_reason == "large_weight":
                self.large_weight_hp_retry_count += 1
                if self.large_weight_retry_example is None:
                    self.large_weight_retry_example = list(xs)
                    self.large_weight_retry_example_momentum_point = momentum_point
                    self.large_weight_retry_example_compiled_wgt = large_trigger_value
            elif fallback_reason == "t_solver_failure":
                self._record_t_solver_failure(
                    failed_root_terms, failed_root_surfaces
                )
                self.t_solver_hp_retry_count += 1
            else:
                self.rotation_hp_retry_count += 1
                if fallback_reason == "float_mismatch":
                    self.stability_float_mismatch_retry_count += 1
                else:
                    self.stability_float_nonfinite_retry_count += 1
                if self.rotation_hp_retry_example is None:
                    self.rotation_hp_retry_example = list(xs)
                    self.rotation_hp_retry_example_momentum_point = momentum_point
                    self.rotation_hp_retry_example_rel = float_relative_difference

            hp_failure_kind, hp_failure_reason = "numerical", "native precision failed; no retry levels"
            def evaluate_retry(precision):
                nonlocal hp_failure_kind, hp_failure_reason
                if precision != precision_ladder[1]:
                    self.stability_hp_escalation_count += 1
                hp_final, hp_unweighted, hp_failure_kind, hp_failure_reason = self._evaluate_stability_hp_pair(
                    xs, parameterization, integrand_implementation, phase,
                    channel_selector, expects_z, expects_beam_fractions,
                    rotation_descriptors, precision, relative_tolerance,
                    absolute_tolerance, soft_mirror_routing)
                if hp_failure_kind is not None or hp_final is None or hp_unweighted is None:
                    raise ArithmeticError(hp_failure_reason or hp_failure_kind or "invalid HP result")
                return hp_final, hp_unweighted

            retry = run_precision_ladder(
                precision_ladder, evaluate_retry, start=1, count=self._count_precision)
            hp_final, hp_unweighted = retry.value if retry.accepted_precision is not None else (None, None)
            if retry.accepted_precision is not None and retry.accepted_precision != precision_ladder[1]:
                self.stability_hp_escalation_accepted_count += 1
            if (
                hp_failure_kind is None
                and hp_final is not None
                and hp_unweighted is not None
            ):
                self.stability_hp_accepted_count += 1
                accepted_final = hp_final
                accepted_unweighted = hp_unweighted if phase == "real" else Decimal(0)
                if soft_mirror_routing is not None:
                    self.soft_mirror_hp_orbit_salvaged_count += 1
                if fallback_reason == "large_weight":
                    self.large_weight_hp_salvaged_count += 1
                    if self.large_weight_retry_example_arb_wgt is None:
                        self.large_weight_retry_example_arb_wgt = float(hp_final)
                elif fallback_reason == "t_solver_failure":
                    self.t_solver_hp_salvaged_count += 1
                else:
                    self.rotation_hp_salvaged_count += 1
            else:
                accepted_final = Decimal(0)
                accepted_unweighted = Decimal(0)
                if soft_mirror_routing is not None:
                    self.soft_mirror_hp_orbit_failure_count += 1
                reason = hp_failure_reason or "unknown higher-precision failure"
                if hp_failure_kind == "disagreement":
                    self.stability_hp_disagreement_count += 1
                elif hp_failure_kind == "nonfinite":
                    self.stability_hp_nonfinite_count += 1
                else:
                    self.stability_hp_error_count += 1
                self._record_hp_failure(xs, momentum_point, reason)
                # Unresolved numerical instability is a zero-weight sample,
                # including for CM-reduced bundles. Unexpected HP exceptions
                # still propagate from _evaluate_stability_hp_pair.
                if fallback_reason == "large_weight":
                    self.large_weight_unstable_count += 1
                elif fallback_reason == "t_solver_failure":
                    self.t_solver_hp_unresolved_count += 1
                else:
                    self.rotation_unstable_count += 1
                    if self.rotation_unstable_example is None:
                        self.rotation_unstable_example = list(xs)
                        self.rotation_unstable_example_momentum_point = momentum_point
                logger.debug(
                    "Rejecting DY sample after higher-precision validation at xs=%s: %s",
                    xs,
                    reason,
                )

        try:
            preclip_weight = float(accepted_final)
        except (OverflowError, ValueError):
            preclip_weight = math.copysign(math.inf, -1.0 if accepted_final < 0 else 1.0)
        if self.max_preclip_wgt is None or abs(preclip_weight) > abs(
            self.max_preclip_wgt
        ):
            self.max_preclip_wgt = preclip_weight
            self.max_preclip_wgt_point = list(xs)
            self.max_preclip_wgt_momentum_point = momentum_point

        if clip_threshold is not None:
            if isinstance(accepted_final, Decimal):
                exceeds_clip = abs(accepted_final) > Decimal(str(clip_threshold))
            else:
                exceeds_clip = math.isfinite(accepted_final) and abs(
                    accepted_final
                ) > clip_threshold
            if exceeds_clip:
                self.large_weight_zeroed_count += 1
                self.large_weight_zeroed_signed_sum += preclip_weight
                self.large_weight_zeroed_abs_sum += abs(preclip_weight)
                accepted_final = Decimal(0) if isinstance(accepted_final, Decimal) else 0.0

        try:
            stable_weight_abs = float(abs(accepted_unweighted))
        except (OverflowError, ValueError):
            stable_weight_abs = math.inf
        if self.max_stable_wgt is None or stable_weight_abs > self.max_stable_wgt:
            self.max_stable_wgt = stable_weight_abs
            self.max_stable_wgt_point = list(xs)
            self.max_stable_wgt_jacobian = total_jacobian
            self.max_stable_wgt_momentum_point = momentum_point

        try:
            final_weight = float(accepted_final)
        except (OverflowError, ValueError):
            final_weight = math.nan
        if not math.isfinite(final_weight):
            self.nan_weight_count += 1
            if self.nan_weight_example is None:
                self.nan_weight_example = list(xs)
                self.nan_weight_example_momentum_point = momentum_point
            final_weight = 0.0

        if self.max_wgt is None or abs(final_weight) > abs(self.max_wgt):
            self.max_wgt = final_weight
            self.max_wgt_point = list(xs)
            self.max_wgt_jacobian = total_jacobian
            self.max_wgt_momentum_point = momentum_point
        return final_weight

    def _integer_counter_snapshot(self) -> dict[str, int]:
        return {
            name: value
            for name, value in vars(self).items()
            if name.endswith("_count") and type(value) is int
        }

    def _restore_integer_counters(self, snapshot: Mapping[str, int]) -> None:
        for name, value in snapshot.items():
            setattr(self, name, int(value))

    def _stability_diagnostic_snapshot(self) -> dict[str, Any]:
        prefixes = (
            "rotation_",
            "large_weight_",
            "stability_",
            "t_solver_",
            "soft_mirror_",
            "nan_weight_",
            "max_",
        )
        return {
            name: deepcopy(value)
            for name, value in vars(self).items()
            if name.startswith(prefixes)
        }

    def _restore_stability_diagnostics(
        self, snapshot: Mapping[str, Any]
    ) -> None:
        for name, value in snapshot.items():
            setattr(self, name, value)

    def _integrand_xspace_batch_scalar_fallback(
        self,
        all_xs: Sequence[list[float]],
        parameterization: str,
        implementations: Sequence[dict[str, Any]],
        phase: str,
        multi_channeling: Sequence[bool | int],
    ) -> tuple[list[float], list[dict[str, int]]]:
        values: list[float] = []
        diagnostics: list[dict[str, int]] = []
        original_muv = self.dy_observable_muv
        try:
            for xs, implementation, channel in zip(
                all_xs, implementations, multi_channeling, strict=True
            ):
                before = self._integer_counter_snapshot()
                if implementation.get("mUV") is not None:
                    self.dy_observable_muv = float(implementation["mUV"])
                values.append(
                    float(
                        self.integrand_xspace(
                            xs,
                            parameterization,
                            implementation,
                            phase,
                            channel,
                        )
                    )
                )
                after = self._integer_counter_snapshot()
                diagnostics.append(
                    {
                        name: after.get(name, 0) - value
                        for name, value in before.items()
                    }
                )
        finally:
            self.dy_observable_muv = original_muv
        return values, diagnostics

    def integrand_xspace_batch(
        self,
        all_xs: Sequence[list[float]],
        parameterization: str,
        integrand_implementations: Sequence[dict[str, Any]],
        phase: str,
        multi_channeling: Sequence[bool | int],
        *,
        integrand_backend: str = "saved-jit",
        kinematic_share_keys: Sequence[Any] | None = None,
    ) -> tuple[list[float], list[dict[str, int]]]:
        """Evaluate a stability orbit in channel-grouped matrix batches.

        This fast path is intentionally narrow: real, compiled ZenoS rows with
        the stability pipeline active. Unsupported rows retain the scalar
        implementation. Kinematic share keys may only join
        mUV/filter variants of the same Monte-Carlo sample; orbit labels are
        appended internally so original, mirror and rotations never mix.
        """
        xs_rows = [list(xs) for xs in all_xs]
        implementations = [
            dict(self._normalize_integrand_implementation(implementation))
            for implementation in integrand_implementations
        ]
        channels = list(multi_channeling)
        if not (
            len(xs_rows) == len(implementations) == len(channels)
        ):
            raise pygloopException("DY x-space batch inputs no longer align.")
        if not xs_rows:
            return [], []
        packet_mode = any(impl.get("_dy_packet_orbits") is not None for impl in implementations)
        if packet_mode and not all(impl.get("_dy_packet_orbits") is not None for impl in implementations):
            raise pygloopException("Packet orbit collection cannot mix ordinary and deferred lanes")
        for implementation in implementations:
            self._dy_precision_ladder(implementation)
        if kinematic_share_keys is None:
            share_keys = list(range(len(xs_rows)))
        else:
            share_keys = list(kinematic_share_keys)
            if len(share_keys) != len(xs_rows):
                raise pygloopException("DY kinematic share keys no longer align.")

        eligible = (
            self.compiled_bundle is not None
            and phase == "real"
            and integrand_backend in {"compiled", "saved-jit"}
            and all(
                implementation.get("integrand_type") == "zenos"
                and (
                    int(implementation.get("dy_rotation_check_digits") or 0) > 0
                    or self._stability_thresholds(implementation) != (None, None)
                )
                for implementation in implementations
            )
        )
        if not eligible:
            if packet_mode:
                raise pygloopException("Native packet orbits require the compiled/saved-jit batch backend")
            return self._integrand_xspace_batch_scalar_fallback(
                xs_rows,
                parameterization,
                implementations,
                phase,
                channels,
            )

        bundle = self.compiled_bundle
        assert bundle is not None
        batch_original_muv = self.dy_observable_muv
        prepared: dict[Any, dict[str, Any]] = {}
        prep_diagnostic_snapshot = self._stability_diagnostic_snapshot()
        original_bundle_evaluate = bundle.evaluate
        original_beam_weights = self._apply_dy_beam_weights
        bundle_had_instance_evaluate = "evaluate" in vars(bundle)
        process_had_instance_beam_weights = "_apply_dy_beam_weights" in vars(self)
        batch_preparation_failed = False

        try:
            for lane, (xs, implementation, channel, share_key) in enumerate(
                zip(
                    xs_rows,
                    implementations,
                    channels,
                    share_keys,
                    strict=True,
                )
            ):
                resolved_channel = self._channel_selector_from_multi_channeling(
                    channel
                )
                rotation_count = self._rotation_check_count(implementation)
                signature = (tuple(xs), resolved_channel, rotation_count)
                if share_key in prepared:
                    if prepared[share_key]["signature"] != signature:
                        raise pygloopException(
                            "DY rows with a common kinematic share key do not "
                            "have identical samples, channels, and rotation counts."
                        )
                    continue

                captured: list[DYCompiledBatchRequest] = []
                prebeam_scale: list[float] = []
                beam_luminosity: list[float | None] = []

                def record_bundle_call(
                    loop_momenta,
                    p1,
                    p2,
                    z,
                    m_uv=None,
                    mode="compiled",
                    decimal_digit_precision=None,
                    theta_tolerance=0.0,
                    channel_selector=None,
                    ttbar_pt_min=None,
                    integrated_uv_ct_filter="all",
                    physical_z_min=None,
                    physical_z_max=None,
                    runtime_parameters=None,
                    raise_on_t_solver_failure=False,
                ):
                    if mode != "compiled":
                        raise pygloopException(
                            "DY batch preparation unexpectedly requested a "
                            "non-compiled evaluator."
                        )
                    captured.append(
                        DYCompiledBatchRequest(
                            tuple(loop_momenta),
                            p1,
                            p2,
                            float(z),
                            m_uv,
                            theta_tolerance=float(theta_tolerance),
                            channel_selector=channel_selector,
                            ttbar_pt_min=ttbar_pt_min,
                            integrated_uv_ct_filter=integrated_uv_ct_filter,
                            physical_z_min=physical_z_min,
                            physical_z_max=physical_z_max,
                            runtime_parameters=runtime_parameters,
                        )
                    )
                    return 1.0 + 0.0j

                def record_beam_weights(weight, pdf_luminosity, sample_xs):
                    prebeam_scale.append(float(weight))
                    beam_luminosity.append(pdf_luminosity)
                    return original_beam_weights(
                        weight, pdf_luminosity, sample_xs
                    )

                prep_implementation = dict(implementation)
                prep_implementation["dy_large_weight_precision"] = None
                prep_implementation["dy_large_weight_threshold"] = None
                prep_implementation["dy_large_weight_clip"] = None
                prep_implementation["dy_zero_large_weight_samples"] = False
                prep_implementation["dy_rotation_check_count"] = rotation_count
                prep_implementation["dy_rotation_check_digits"] = max(
                    1,
                    int(
                        prep_implementation.get("dy_rotation_check_digits") or 0
                    ),
                )
                bundle.evaluate = record_bundle_call  # type: ignore[method-assign]
                self._apply_dy_beam_weights = record_beam_weights  # type: ignore[method-assign]
                if implementation.get("mUV") is not None:
                    self.dy_observable_muv = float(implementation["mUV"])
                final_scale = float(
                    self.integrand_xspace(
                        xs,
                        parameterization,
                        prep_implementation,
                        phase,
                        channel,
                    )
                )
                bundle.evaluate = original_bundle_evaluate  # type: ignore[method-assign]
                self._apply_dy_beam_weights = original_beam_weights  # type: ignore[method-assign]

                soft_routing = self._soft_mirror_routing(
                    implementation, resolved_channel
                )
                expected_calls = (2 + 2 * rotation_count) if soft_routing else (
                    1 + rotation_count
                )
                if captured and len(captured) != expected_calls:
                    batch_preparation_failed = True
                    break
                prepared[share_key] = {
                    "signature": signature,
                    "captured": tuple(captured),
                    "soft": soft_routing is not None,
                    "soft_routing": soft_routing,
                    "rotation_count": rotation_count,
                    "prebeam_scale": (
                        prebeam_scale[-1] if prebeam_scale else final_scale
                    ),
                    "beam_luminosity": (
                        beam_luminosity[-1] if beam_luminosity else None
                    ),
                    "applies_beam_weights": bool(prebeam_scale),
                    "graph_weight": self._dy_graph_channel_weight(
                        implementation, resolved_channel
                    ),
                }
        finally:
            if bundle_had_instance_evaluate:
                bundle.evaluate = original_bundle_evaluate  # type: ignore[method-assign]
            else:
                vars(bundle).pop("evaluate", None)
            if process_had_instance_beam_weights:
                self._apply_dy_beam_weights = original_beam_weights  # type: ignore[method-assign]
            else:
                vars(self).pop("_apply_dy_beam_weights", None)
            self._restore_stability_diagnostics(prep_diagnostic_snapshot)
            self.dy_observable_muv = batch_original_muv

        if batch_preparation_failed:
            if packet_mode:
                raise ArithmeticError("Native packet orbit preparation failed")
            return self._integrand_xspace_batch_scalar_fallback(
                xs_rows,
                parameterization,
                implementations,
                phase,
                channels,
            )

        # Per-request observers have no effect on root sharing or sampling.
        histogram_orbits = {}

        def actual_request(
            captured: DYCompiledBatchRequest,
            implementation: Mapping[str, Any],
            root_key: Any,
        ) -> DYCompiledBatchRequest:
            integrated_filter = implementation.get(
                "dy_integrated_uv_ct_filter",
                captured.integrated_uv_ct_filter,
            )
            if str(integrated_filter).replace("-", "_") == "all":
                integrated_filter = "all"
            histogram = implementation.get("_dy_histogram")
            observer = None
            if histogram is not None:
                observer = histogram.fresh()
                histogram_orbits[(id(implementation), root_key[1])] = observer
            return DYCompiledBatchRequest(
                captured.loop_momenta,
                captured.p1,
                captured.p2,
                captured.z,
                implementation.get("mUV", captured.m_uv),
                theta_tolerance=float(
                    implementation.get("dy_theta_tol") or 0.0
                ),
                channel_selector=captured.channel_selector,
                ttbar_pt_min=(
                    self._validate_ttbar_pt_min(
                        implementation["dy_ttbar_pt_min"]
                    )
                    if implementation.get("dy_ttbar_pt_min") is not None
                    else None
                ),
                integrated_uv_ct_filter=integrated_filter,
                physical_z_min=implementation.get(
                    "dy_physical_z_min", captured.physical_z_min
                ),
                physical_z_max=implementation.get(
                    "dy_physical_z_max", captured.physical_z_max
                ),
                runtime_parameters=implementation.get("dy_runtime_parameters"),
                root_share_key=root_key,
                term_observer=observer.add if observer is not None else None,
            )

        def evaluate_bundle_batch(
            requests: Sequence[DYCompiledBatchRequest],
        ) -> tuple[
            list[complex],
            list[DYCompiledEvaluationDiagnostics | None],
        ]:
            diagnostic_evaluator = getattr(
                bundle, "evaluate_batch_with_diagnostics", None
            )
            if callable(diagnostic_evaluator):
                return diagnostic_evaluator(
                    requests, integrand_backend=integrand_backend
                )
            outputs = bundle.evaluate_batch(
                requests, integrand_backend=integrand_backend
            )
            return outputs, [None] * len(requests)

        first_stage: list[DYCompiledBatchRequest] = []
        first_slots: list[tuple[int, str]] = []
        for lane, (implementation, share_key) in enumerate(
            zip(implementations, share_keys, strict=True)
        ):
            lane_prep = prepared[share_key]
            captured = lane_prep["captured"]
            if not captured:
                continue
            first_stage.append(
                actual_request(
                    captured[0], implementation, (share_key, "original")
                )
            )
            first_slots.append((lane, "original"))
            if lane_prep["soft"]:
                first_stage.append(
                    actual_request(
                        captured[1], implementation, (share_key, "mirror")
                    )
                )
                first_slots.append((lane, "mirror"))

        try:
            first_outputs, first_diagnostics = evaluate_bundle_batch(first_stage)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if packet_mode:
                if isinstance(exc, DYNumericalEvaluationError):
                    raise ArithmeticError(str(exc)) from exc
                raise
            return self._integrand_xspace_batch_scalar_fallback(
                xs_rows,
                parameterization,
                implementations,
                phase,
                channels,
            )
        orbit_values: list[dict[str, complex]] = [dict() for _ in xs_rows]
        fallback_reasons: list[str | None] = [None] * len(xs_rows)
        failed_root_terms: list[set[str]] = [set() for _ in xs_rows]
        failed_root_surfaces: list[set[str]] = [set() for _ in xs_rows]
        for (lane, orbit), value, diagnostic in zip(
            first_slots, first_outputs, first_diagnostics, strict=True
        ):
            orbit_values[lane][orbit] = value
            if diagnostic is not None and diagnostic.has_t_solver_failure:
                fallback_reasons[lane] = "t_solver_failure"
                failed_root_terms[lane].update(diagnostic.failed_terms)
                failed_root_surfaces[lane].update(diagnostic.failed_surfaces)

        for lane, (implementation, share_key) in enumerate(
            zip(implementations, share_keys, strict=True)
        ):
            lane_prep = prepared[share_key]
            if not lane_prep["captured"]:
                continue
            if fallback_reasons[lane] is not None:
                continue
            if _diagnostic_soft_radius_hp_trigger(
                xs_rows[lane], lane_prep["soft_routing"]
            ):
                fallback_reasons[lane] = "float_mismatch"
                continue
            originals = [orbit_values[lane]["original"]]
            if lane_prep["soft"]:
                originals.append(orbit_values[lane]["mirror"])
            scale = float(lane_prep["prebeam_scale"])
            original_finals = [float(value.real) * scale for value in originals]
            if implementation.get("_dy_histogram") is not None:
                for orbit in ("original", "mirror") if lane_prep["soft"] else ("original",):
                    original_finals.extend(
                        float(value.real)*scale for value in
                        histogram_orbits[(id(implementation), orbit)].bins)
            if any(not math.isfinite(value) for value in original_finals):
                fallback_reasons[lane] = "float_nonfinite"
                continue
            if packet_mode:
                continue  # Apply weight triggers to the complete packet.
            precision_threshold, clip_threshold = self._stability_thresholds(
                implementation
            )
            if lane_prep["applies_beam_weights"]:
                precision_threshold, clip_threshold = (
                    self._rescale_dy_beam_stability_thresholds(
                        precision_threshold,
                        clip_threshold,
                        lane_prep["beam_luminosity"],
                    )
                )
            trigger_thresholds = [
                threshold
                for threshold in (precision_threshold, clip_threshold)
                if threshold is not None
            ]
            if any(
                abs(value) > threshold
                for value in original_finals
                for threshold in trigger_thresholds
            ):
                fallback_reasons[lane] = "large_weight"

        second_stage: list[DYCompiledBatchRequest] = []
        second_slots: list[tuple[int, str]] = []
        for lane, (implementation, share_key) in enumerate(
            zip(implementations, share_keys, strict=True)
        ):
            lane_prep = prepared[share_key]
            captured = lane_prep["captured"]
            if not captured or fallback_reasons[lane] is not None:
                continue
            for rotation_index in range(lane_prep["rotation_count"]):
                captured_index = (
                    2 + 2 * rotation_index
                    if lane_prep["soft"]
                    else 1 + rotation_index
                )
                orbit = f"rotated:{rotation_index}"
                second_stage.append(
                    actual_request(
                        captured[captured_index],
                        implementation,
                        (share_key, orbit),
                    )
                )
                second_slots.append((lane, orbit))
                if lane_prep["soft"]:
                    mirrored_orbit = f"rotated-mirror:{rotation_index}"
                    second_stage.append(
                        actual_request(
                            captured[captured_index + 1],
                            implementation,
                            (share_key, mirrored_orbit),
                        )
                    )
                    second_slots.append((lane, mirrored_orbit))
        try:
            second_outputs, second_diagnostics = evaluate_bundle_batch(
                second_stage
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if packet_mode:
                if isinstance(exc, DYNumericalEvaluationError):
                    raise ArithmeticError(str(exc)) from exc
                raise
            return self._integrand_xspace_batch_scalar_fallback(
                xs_rows,
                parameterization,
                implementations,
                phase,
                channels,
            )
        for (lane, orbit), value, diagnostic in zip(
            second_slots, second_outputs, second_diagnostics, strict=True
        ):
            orbit_values[lane][orbit] = value
            if diagnostic is not None and diagnostic.has_t_solver_failure:
                fallback_reasons[lane] = "t_solver_failure"
                failed_root_terms[lane].update(diagnostic.failed_terms)
                failed_root_surfaces[lane].update(diagnostic.failed_surfaces)

        if packet_mode:
            # No per-lane retries or zeros: return all native rotation orbits
            # to the packet owner, or retry the entire packet on root failure.
            for lane, (implementation, share_key) in enumerate(zip(implementations, share_keys, strict=True)):
                prep = prepared[share_key]
                n = prep["rotation_count"]
                histogram = implementation.get("_dy_histogram")
                if fallback_reasons[lane] is not None:
                    raise ArithmeticError(f"native packet lane {lane}: {fallback_reasons[lane]}")
                values = [0.]*(n+1)
                bins = [[0.]*len(histogram.bins) for _ in range(n+1)] if histogram is not None else None
                if prep["captured"]:
                    scale = float(prep["prebeam_scale"])
                    keys = ["original", *[f"rotated:{i}" for i in range(n)]]
                    mirror_keys = ["mirror", *[f"rotated-mirror:{i}" for i in range(n)]]
                    for i, key in enumerate(keys):
                        value = float(orbit_values[lane][key].real)*scale
                        if prep["soft"]:
                            value = .5*(value+float(orbit_values[lane][mirror_keys[i]].real)*scale)
                        values[i] = value
                        if bins is not None:
                            first = histogram_orbits[(id(implementation), key)].bins
                            if prep["soft"]:
                                mirror = histogram_orbits[(id(implementation), mirror_keys[i])].bins
                                first = [(a+b)*.5 for a, b in zip(first, mirror, strict=True)]
                            bins[i] = [float(v.real)*scale for v in first]
                if any(not math.isfinite(v) for v in values + ([] if bins is None else [v for row in bins for v in row])):
                    raise ArithmeticError("Native packet orbit is non-finite")
                implementation["_dy_packet_orbits"].append({"values": values, "bins": bins})
                if histogram is not None:
                    histogram.accept(bins[0])
            return [0.]*len(xs_rows), [{} for _ in xs_rows]

        for lane, (implementation, share_key) in enumerate(
            zip(implementations, share_keys, strict=True)
        ):
            lane_prep = prepared[share_key]
            if not lane_prep["captured"] or fallback_reasons[lane] is not None:
                continue
            relative_tolerance, absolute_tolerance = self._stability_tolerances(
                implementation,
                int(implementation.get("dy_rotation_check_digits") or 0),
            )
            scale = float(lane_prep["prebeam_scale"])
            original = float(orbit_values[lane]["original"].real) * scale
            mirrored = (
                float(orbit_values[lane]["mirror"].real) * scale
                if lane_prep["soft"]
                else math.nan
            )
            for rotation_index in range(lane_prep["rotation_count"]):
                rotated = (
                    float(
                        orbit_values[lane][f"rotated:{rotation_index}"].real
                    )
                    * scale
                )
                if not math.isfinite(rotated):
                    fallback_reasons[lane] = "float_nonfinite"
                elif not float_values_agree(
                    original,
                    rotated,
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance,
                ):
                    fallback_reasons[lane] = "float_mismatch"
                elif lane_prep["soft"]:
                    rotated_mirrored = (
                        float(
                            orbit_values[lane][
                                f"rotated-mirror:{rotation_index}"
                            ].real
                        )
                        * scale
                    )
                    if not math.isfinite(rotated_mirrored):
                        fallback_reasons[lane] = "float_nonfinite"
                    elif not float_values_agree(
                        mirrored,
                        rotated_mirrored,
                        relative_tolerance=relative_tolerance,
                        absolute_tolerance=absolute_tolerance,
                    ):
                        fallback_reasons[lane] = "float_mismatch"
                if fallback_reasons[lane] is not None:
                    break

            if implementation.get("_dy_histogram") is not None and fallback_reasons[lane] is None:
                for original_orbit, rotated_prefix in (("original", "rotated"), ("mirror", "rotated-mirror")):
                    if original_orbit == "mirror" and not lane_prep["soft"]:
                        continue
                    first_bins = histogram_orbits[(id(implementation), original_orbit)].bins
                    for rotation_index in range(lane_prep["rotation_count"]):
                        rotated_bins = histogram_orbits[(id(implementation), f"{rotated_prefix}:{rotation_index}")].bins
                        if any(not float_values_agree(
                                float(first.real)*scale, float(rotated.real)*scale,
                                relative_tolerance=relative_tolerance,
                                absolute_tolerance=absolute_tolerance)
                               for first, rotated in zip(first_bins, rotated_bins, strict=True)):
                            fallback_reasons[lane] = "float_mismatch"
                            break

        values: list[float] = [0.0] * len(xs_rows)
        diagnostics: list[dict[str, int]] = []
        try:
            for lane, (xs, implementation, channel, share_key) in enumerate(
                zip(
                    xs_rows,
                    implementations,
                    channels,
                    share_keys,
                    strict=True,
                )
            ):
                before = self._integer_counter_snapshot()
                lane_prep = prepared[share_key]
                reason = fallback_reasons[lane]
                if implementation.get("mUV") is not None:
                    self.dy_observable_muv = float(implementation["mUV"])
                if not lane_prep["captured"]:
                    values[lane] = 0.0
                    self._count_precision(16, "attempt")
                    self._count_precision(16, "accepted")
                    if implementation.get("_dy_histogram") is not None:
                        histogram = implementation["_dy_histogram"]
                        histogram.accept([0.] * len(histogram.bins))
                elif reason is not None:
                    forced = dict(implementation)
                    forced["_dy_force_stability_hp_reason"] = reason
                    if reason == "t_solver_failure":
                        forced["_dy_force_t_solver_failed_terms"] = tuple(
                            sorted(failed_root_terms[lane])
                        )
                        forced["_dy_force_t_solver_failed_surfaces"] = tuple(
                            sorted(failed_root_surfaces[lane])
                        )
                    values[lane] = float(
                        self.integrand_xspace(
                            xs,
                            parameterization,
                            forced,
                            phase,
                            channel,
                        )
                    )
                else:
                    is_soft = bool(lane_prep["soft"])
                    self._count_precision(16, "attempt")
                    self._count_precision(16, "accepted")
                    accepted_rotation_pairs = lane_prep["rotation_count"]
                    if is_soft:
                        self.soft_mirror_pair_count += 1
                        self.stability_float_pair_accepted_count += (
                            2 * accepted_rotation_pairs
                        )
                        scale = float(lane_prep["prebeam_scale"])
                        accepted_raw = 0.5 * (
                            orbit_values[lane]["original"].real
                            + orbit_values[lane]["mirror"].real
                        )
                        accepted_prebeam = 0.5 * (
                            orbit_values[lane]["original"].real * scale
                            + orbit_values[lane]["mirror"].real * scale
                        )
                        self._record_soft_mirror_pair(
                            xs,
                            orbit_values[lane]["original"].real * scale,
                            orbit_values[lane]["mirror"].real * scale,
                        )
                    else:
                        self.stability_float_pair_accepted_count += (
                            accepted_rotation_pairs
                        )
                        scale = float(lane_prep["prebeam_scale"])
                        accepted_raw = orbit_values[lane]["original"].real
                        accepted_prebeam = accepted_raw * scale
                    accepted_unweighted = (
                        accepted_raw * float(lane_prep["graph_weight"])
                    )
                    momentum_point = "batched xs = [{}]".format(
                        ", ".join(f"{value:+.16e}" for value in xs)
                    )
                    if self.max_preclip_wgt is None or abs(
                        accepted_prebeam
                    ) > abs(self.max_preclip_wgt):
                        self.max_preclip_wgt = float(accepted_prebeam)
                        self.max_preclip_wgt_point = list(xs)
                        self.max_preclip_wgt_momentum_point = momentum_point
                    if self.max_stable_wgt is None or abs(
                        accepted_unweighted
                    ) > self.max_stable_wgt:
                        self.max_stable_wgt = float(abs(accepted_unweighted))
                        self.max_stable_wgt_point = list(xs)
                        self.max_stable_wgt_jacobian = scale
                        self.max_stable_wgt_momentum_point = momentum_point
                    if self.max_wgt is None or abs(accepted_prebeam) > abs(
                        self.max_wgt
                    ):
                        self.max_wgt = float(accepted_prebeam)
                        self.max_wgt_point = list(xs)
                        self.max_wgt_jacobian = scale
                        self.max_wgt_momentum_point = momentum_point
                    values[lane] = float(accepted_prebeam)
                    if implementation.get("_dy_histogram") is not None:
                        histogram = implementation["_dy_histogram"]
                        bins = histogram_orbits[(id(implementation), "original")].bins
                        if is_soft:
                            mirror_bins = histogram_orbits[(id(implementation), "mirror")].bins
                            bins = [(a+b)*.5 for a, b in zip(bins, mirror_bins, strict=True)]
                        histogram.accept(float(value.real)*scale for value in bins)
                    if lane_prep["applies_beam_weights"]:
                        values[lane] = float(
                            self._apply_dy_beam_weights(
                                values[lane],
                                lane_prep["beam_luminosity"],
                                xs,
                            )
                        )
                after = self._integer_counter_snapshot()
                diagnostics.append(
                    {
                        name: after.get(name, 0) - value
                        for name, value in before.items()
                    }
                )
        finally:
            self.dy_observable_muv = batch_original_muv
        return values, diagnostics

    def integrand_xspace(
        self,
        xs: list[float],
        parameterization: str,
        integrand_implementation: dict[str, Any],
        phase: str,
        multi_channeling: bool | int = True,
    ) -> float:
        integrand_implementation = self._normalize_integrand_implementation(
            integrand_implementation
        )
        try:
            # t0 = time.perf_counter()

            impl = dict(integrand_implementation)
            channel_selector = self._channel_selector_from_multi_channeling(
                multi_channeling
            )
            if channel_selector == DY_GROUPED_SOFT_PAIR_SELECTOR:
                if parameterization != "spherical":
                    raise pygloopException(
                        "DY grouped soft-pair inversion requires the spherical "
                        "parameterisation."
                    )
                grouped_pair = self._grouped_soft_pair(impl)
                if grouped_pair is None:
                    raise pygloopException(
                        "The grouped soft-pair channel was selected without a "
                        "card mapping."
                    )
                first, second, _transform = grouped_pair
                second_xs = self._grouped_soft_pair_coordinates(
                    xs, transformed_member=True
                )
                failures_before = tuple(
                    getattr(self, name, 0) for name in DY_STABILITY_FAILURE_COUNTERS
                )
                value = self.integrand_xspace(
                    list(xs), parameterization, impl, phase, first
                ) + self.integrand_xspace(
                    second_xs, parameterization, impl, phase, second
                )
                if failures_before != tuple(
                    getattr(self, name, 0) for name in DY_STABILITY_FAILURE_COUNTERS
                ):
                    # Do not retain the stable member of an unstable sample.
                    return 0.0
                return value
            expects_z = self.sampled_uses_z(impl)
            expects_beam_fractions = self.sampled_uses_beam_fractions(impl)
            beam_parameterisation = str(
                impl.get("dy_beam_parameterisation", "x1_x2")
            )
            if beam_parameterisation not in {"x1_x2", "beta_y"}:
                raise pygloopException(
                    f"Unsupported DY beam parameterisation {beam_parameterisation!r}."
                )
            if beam_parameterisation == "beta_y":
                if not expects_beam_fractions:
                    raise pygloopException(
                        "The beta-Y beam parameterisation requires zenos "
                        "beam-fraction sampling."
                    )
                if self.process_name.lower() != "tt~":
                    raise pygloopException(
                        "The beta-Y beam parameterisation currently supports "
                        "ttbar production only."
                    )
            if self.dy_pdf_set is not None and not expects_beam_fractions:
                raise pygloopException(
                    "DY PDF weighting requires zenos beam-fraction sampling."
                )
            if (
                self.dy_physical_normalisation
                and not expects_beam_fractions
                and not (
                    self._dy_partonic_one_loop_mode
                    or (
                        self.dy_msbar_scheme_counterterm
                        and self._dy_ttbar_partonic_scheme_mode
                    )
                )
            ):
                raise pygloopException(
                    "DY physical normalisation requires zenos beam-fraction sampling."
                )
            n_k_vars = 3 * self.n_loops
            expected_dim = n_k_vars + int(expects_z) + 2 * int(expects_beam_fractions)
            k_rescaling = 1.0
            jac_z = 1
            if expects_z:
                if len(xs) != expected_dim:
                    raise pygloopException(
                        f"Integrand '{impl['integrand_type']}' expects {expected_dim} variables "
                        f"({n_k_vars} loop-momentum variables plus xz), got {len(xs)}."
                    )

                k_xs = xs[:n_k_vars].copy()

                # SMALL VALUE
                a, b = 0.000, 1.0
                k_xs[0] = a + (b - a) * k_xs[0]
                k_rescaling = b - a

                x_z = xs[n_k_vars]

                z_min = float(impl.get("z_min", 0.0))
                z_max = float(impl.get("z_max", 1.0))
                if not (0.0 <= z_min < z_max <= 1.0):
                    raise pygloopException(f"Invalid z range [{z_min}, {z_max}]")

                z_sample = x_z / (1 - x_z)
                jac_z = 1 / (1 - x_z) ** 2

                impl["z"] = z_sample
            else:
                if len(xs) != expected_dim:
                    raise pygloopException(
                        f"Integrand '{impl['integrand_type']}' expects {expected_dim} loop-momentum variables, got {len(xs)}."
                    )
                k_xs = xs
                z_sample = None
                impl["z"] = 1.0

            p1 = self.ps_point[0].spatial()
            p2 = self.ps_point[1].spatial()
            pdf_luminosity: float | None = None
            beam_jacobian = 1.0
            if expects_beam_fractions:
                beam_offset = n_k_vars + int(expects_z)
                beam_threshold_mass = self._effective_ttbar_beam_mass(impl)
                if beam_parameterisation == "beta_y":
                    x1, x2, beam_jacobian = parameterize_ttbar_beam_fractions(
                        xs[beam_offset],
                        xs[beam_offset + 1],
                        m_top=beam_threshold_mass,
                        e_cm=self.e_cm,
                    )
                    x1 = float(x1)
                    x2 = float(x2)
                    beam_jacobian = float(beam_jacobian)
                    if beam_jacobian == 0.0:
                        return 0.0
                else:
                    x1 = xs[beam_offset]
                    x2 = xs[beam_offset + 1]
                if not self.ttbar_beam_threshold_passes(
                    x1, x2, beam_threshold_mass
                ):
                    return 0.0
                if (
                    self.dy_z_bin is not None
                    or self.dy_q_min is not None
                    or self.dy_q_max is not None
                ):
                    physical_z_interval = self.dy_physical_z_interval(x1, x2)
                    if physical_z_interval is None:
                        return 0.0
                    (
                        impl["dy_physical_z_min"],
                        impl["dy_physical_z_max"],
                    ) = physical_z_interval
                p1, p2 = self.sampled_beam_momenta(x1, x2)
                if self.dy_pdf_set is not None:
                    pdf_luminosity = self.dy_pdf_luminosity(x1, x2)
            elif (
                self.dy_z_bin is not None
                or self.dy_q_min is not None
                or self.dy_q_max is not None
            ):
                physical_z_interval = self.dy_physical_z_interval(1.0, 1.0)
                if physical_z_interval is None:
                    return 0.0
                (
                    impl["dy_physical_z_min"],
                    impl["dy_physical_z_max"],
                ) = physical_z_interval

            applies_dy_beam_weights = (
                pdf_luminosity is not None
                or self.dy_physical_normalisation
                or self.dy_integrated_leptonic_phase_space
                or self.dy_coupling_normalisation_applied
            )

            is_zenos = impl.get("integrand_type") == "zenos"
            soft_mirror_requested = impl.get("dy_soft_mirror_edges") is not None
            if soft_mirror_requested and not is_zenos:
                raise pygloopException(
                    "DY soft mirroring is only supported by the zenos integrand."
                )
            soft_mirror_routing = (
                self._soft_mirror_routing(impl, channel_selector)
                if is_zenos and soft_mirror_requested
                else None
            )

            loop_momenta = [Vector(0.0, 0.0, 0.0) for _ in range(self.n_loops)]
            jac_k = 1.0
            shifted_routing = (
                soft_mirror_routing
                if soft_mirror_routing is not None
                and soft_mirror_routing.has_external_offset
                else None
            )
            pivot = shifted_routing.pivot_loop_index if shifted_routing else None
            for i_loop in range(self.n_loops):
                if i_loop == pivot:
                    continue
                k_loop, jac_loop = self.parameterize(
                    k_xs[3 * i_loop : 3 * (i_loop + 1)], parameterization
                )
                loop_momenta[i_loop] = k_loop
                jac_k *= jac_loop

            soft_center: Vector | None = None
            if shifted_routing is not None:
                assert self.compiled_bundle is not None
                soft_center = self.compiled_bundle.soft_center_for_routing(
                    loop_momenta,
                    p1,
                    p2,
                    impl["z"],
                    impl.get("mUV"),
                    shifted_routing,
                    channel_selector,
                    runtime_parameters=impl.get("dy_runtime_parameters"),
                )
                pivot_loop, pivot_jacobian = self.parameterize(
                    k_xs[3 * pivot : 3 * (pivot + 1)],
                    parameterization,
                    origin=soft_center,
                )
                loop_momenta[pivot] = pivot_loop
                jac_k *= pivot_jacobian

            total_jacobian = jac_k * jac_z * k_rescaling * beam_jacobian
            momentum_point = f"k = [{'; '.join('[' + ', '.join(f'{ki:.16e}' for ki in km.to_list()) + ']' for km in loop_momenta)}]"
            if z_sample is not None:
                momentum_point += f", z = {z_sample:.16e}"
            if expects_beam_fractions:
                momentum_point += (
                    f", p1 = [{', '.join(f'{ki:.16e}' for ki in p1.to_list())}]"
                    f", p2 = [{', '.join(f'{ki:.16e}' for ki in p2.to_list())}]"
                )

            # t1 = time.perf_counter()

            # print("-" * 15)
            # print("parametrisation time:", t1 - t0)

            n_digits = impl.get("dy_rotation_check_digits")
            rotation_check_enabled = False
            n_digits_int = 0
            if is_zenos and n_digits is not None:
                n_digits_int = int(n_digits)
                if n_digits_int < 0:
                    raise pygloopException(
                        "DY rotation-check digits must be non-negative."
                    )
                rotation_check_enabled = n_digits_int > 0

            precision_threshold, clip_threshold = self._stability_thresholds(impl)
            if applies_dy_beam_weights:
                precision_threshold, clip_threshold = (
                    self._rescale_dy_beam_stability_thresholds(
                        precision_threshold,
                        clip_threshold,
                        pdf_luminosity,
                    )
                )
            stability_requested = is_zenos and (
                rotation_check_enabled
                or precision_threshold is not None
                or clip_threshold is not None
                or soft_mirror_routing is not None
            )
            if stability_requested:
                stability_weight = self._evaluate_zenos_stability_sample(
                    xs,
                    parameterization,
                    impl,
                    phase,
                    channel_selector,
                    expects_z,
                    expects_beam_fractions,
                    loop_momenta,
                    p1,
                    p2,
                    total_jacobian,
                    momentum_point,
                    n_digits_int,
                    precision_threshold,
                    clip_threshold,
                    soft_mirror_routing,
                    soft_center,
                )
                if applies_dy_beam_weights:
                    return self._apply_dy_beam_weights(
                        stability_weight,
                        pdf_luminosity,
                        xs,
                    )
                return stability_weight

            try:
                if expects_beam_fractions:
                    wgt = self.zenos_integrand_with_externals(
                        loop_momenta,
                        p1,
                        p2,
                        impl,
                        channel_selector=channel_selector,
                    )
                else:
                    wgt = self.integrand(
                        loop_momenta, impl, channel_selector=channel_selector
                    )
            except DYDoubleRootFailure as exc:
                forced_impl = dict(impl)
                forced_impl["_dy_force_stability_hp_reason"] = (
                    "t_solver_failure"
                )
                forced_impl["_dy_force_t_solver_failed_terms"] = tuple(
                    sorted(exc.diagnostics.failed_terms)
                )
                forced_impl["_dy_force_t_solver_failed_surfaces"] = tuple(
                    sorted(exc.diagnostics.failed_surfaces)
                )
                stability_weight = self._evaluate_zenos_stability_sample(
                    xs,
                    parameterization,
                    forced_impl,
                    phase,
                    channel_selector,
                    expects_z,
                    expects_beam_fractions,
                    loop_momenta,
                    p1,
                    p2,
                    total_jacobian,
                    momentum_point,
                    0,
                    None,
                    None,
                )
                if applies_dy_beam_weights:
                    return self._apply_dy_beam_weights(
                        stability_weight,
                        pdf_luminosity,
                        xs,
                    )
                return stability_weight
            wgt = self._sanitize_integrand_weight(
                wgt, xs, momentum_point, rotated=False
            )
            stable_wgt_abs = abs(wgt)
            if self.max_stable_wgt is None or stable_wgt_abs > self.max_stable_wgt:
                self.max_stable_wgt = stable_wgt_abs
                self.max_stable_wgt_point = list(xs)
                self.max_stable_wgt_jacobian = total_jacobian
                self.max_stable_wgt_momentum_point = momentum_point

            final_wgt = self._phase_value(wgt, phase) * total_jacobian
            if self.max_preclip_wgt is None or abs(final_wgt) > abs(
                self.max_preclip_wgt
            ):
                self.max_preclip_wgt = final_wgt
                self.max_preclip_wgt_point = list(xs)
                self.max_preclip_wgt_momentum_point = momentum_point

            if not math.isfinite(final_wgt):
                self.nan_weight_count += 1
                if self.nan_weight_example is None:
                    self.nan_weight_example = list(xs)
                    self.nan_weight_example_momentum_point = momentum_point
                logger.debug(
                    f"Integrand evaluated to non-finite final weight at xs = [{Colour.BLUE}{', '.join(f'{xi:+.16e}' for xi in xs)}{Colour.END}]. Setting it to zero"
                )
                final_wgt = 0.0

            if self.max_wgt is None or abs(final_wgt) > abs(self.max_wgt):
                self.max_wgt = final_wgt
                self.max_wgt_point = list(xs)
                self.max_wgt_jacobian = total_jacobian
                self.max_wgt_momentum_point = momentum_point

            if applies_dy_beam_weights:
                final_wgt = self._apply_dy_beam_weights(
                    final_wgt,
                    pdf_luminosity,
                    xs,
                )

            # print("res")
            # print(xs)
            # print(final_wgt)

        except ZeroDivisionError:
            if getattr(self.compiled_bundle, "_dy_cm_reduced", False):
                raise pygloopException(
                    f"DY division by zero at xs={xs}; no sample accepted."
                ) from None
            logger.debug(
                f"Integrand divided by zero at xs = [{Colour.BLUE}{', '.join(f'{xi:+.16e}' for xi in xs)}{Colour.END}]. Setting it to zero"
            )
            final_wgt = 0.0

        return final_wgt

    def integrand(
        self,
        loop_momenta: list[Vector],
        integrand_implementation: dict[str, Any],
        channel_selector: int | None = None,
    ) -> complex:
        integrand_implementation = self._normalize_integrand_implementation(
            integrand_implementation
        )
        try:
            match integrand_implementation["integrand_type"]:
                case "spenso":
                    return self.spenso_integrand(loop_momenta)
                case "zenos":
                    return self.zenos_integrand(
                        loop_momenta,
                        integrand_implementation,
                        channel_selector=channel_selector,
                    )
                case "gammaloop":
                    return self.gammaloop_integrand(loop_momenta)
                case _:
                    raise pygloopException(
                        f"Integrand implementation {integrand_implementation['integrand_type']} not implemented."
                    )
        except ZeroDivisionError:
            logger.debug(
                f"Integrand divided by zero for ks = [{Colour.BLUE}{
                    ','.join(
                        '[' + ', '.join(f'{ki:+.16e}' for ki in k.to_list()) + ']'
                        for k in loop_momenta
                    )
                }{Colour.END}]. Setting it to zero"
            )
            return 0.0

    def get_integrand_name(self, suffix="_processed"):
        match self.n_loops:
            case 1 | 2:
                return f"{self.name}_{self.n_loops}L{suffix}"
            case _:
                raise pygloopException(f"Number of loops {self.n_loops} not supported.")

    def gammaloop_integrand(self, loop_momenta: list[Vector]) -> complex:
        try:
            process_id = self.cache["process_id"]
        except KeyError:
            amplitudes, _cross_sections = self.gl_worker.list_outputs()
            if self.get_integrand_name() not in amplitudes:
                raise pygloopException(
                    f"Amplitude {self.get_integrand_name()} not found in GammaLoop state. Generate graphs and code first with the generate subcommand."
                )
            process_id = amplitudes[self.get_integrand_name()]
            self.cache["process_id"] = process_id

        res, _jac = self.gl_worker.inspect(
            process_id=process_id,
            integrand_name=self.get_integrand_name(),
            point=[ki for k in loop_momenta for ki in k.to_list()],
            use_f128=False,
            force_radius=False,
            momentum_space=True,
            discrete_dim=[],
        )
        return res

    def spenso_integrand(
        self,
        loop_momentum: list[Vector],
        integrand_implementation: dict[str, Any] | None = None,
    ) -> complex:
        raise ValueError("spenso integrand not implemented")

    def zenos_integrand(
        self,
        loop_momentum: list[Vector],
        integrand_implementation: dict[str, Any] | None = None,
        channel_selector: int | None = None,
    ) -> complex:
        if self.compiled_bundle is None:
            raise pygloopException(
                f"No compiled DY bundle loaded for integrand '{self.get_integrand_name()}'."
            )

        p1 = self.ps_point[0].spatial()
        p2 = self.ps_point[1].spatial()
        return self.zenos_integrand_with_externals(
            loop_momentum,
            p1,
            p2,
            integrand_implementation,
            channel_selector=channel_selector,
        )

    @staticmethod
    def _validate_ttbar_pt_min(value: Any) -> float:
        pt_min = float(value)
        if pt_min < 0.0:
            raise pygloopException("DY ttbar pT lower cut must be non-negative.")
        return pt_min

    def _dy_graph_channel_weight(
        self,
        integrand_implementation: Mapping[str, Any] | None,
        channel_selector: int | None,
    ) -> float:
        weights = (
            integrand_implementation.get("dy_graph_weights")
            if integrand_implementation is not None
            else None
        )
        if weights is None:
            return 1.0
        if self.compiled_bundle is None:
            raise pygloopException("DY graph weights require a compiled bundle.")
        expected = self.compiled_bundle.graph_channel_count()
        if len(weights) != expected:
            raise pygloopException(
                f"Expected {expected} DY graph weights in bundle channel order, "
                f"got {len(weights)}."
            )
        if channel_selector is None:
            raise pygloopException(
                "DY graph weights require graph multi-channeling."
            )
        if channel_selector < 0 or channel_selector >= expected:
            raise pygloopException(
                f"DY graph channel {channel_selector} is out of range for "
                f"{expected} graph weights."
            )
        weight = float(weights[channel_selector])
        if not math.isfinite(weight):
            raise pygloopException("DY graph weights must be finite.")
        return weight

    def zenos_integrand_with_externals(
        self,
        loop_momentum: list[Vector],
        p1: Vector,
        p2: Vector,
        integrand_implementation: dict[str, Any] | None = None,
        channel_selector: int | None = None,
    ) -> complex:
        if self.compiled_bundle is None:
            raise pygloopException(
                f"No compiled DY bundle loaded for integrand '{self.get_integrand_name()}'."
            )

        z: float | Decimal = 1.0
        m_uv: float | Decimal | None = None
        runtime_parameters: Mapping[str, Any] | None = None
        if integrand_implementation is not None:
            m_uv = integrand_implementation.get("mUV")
            runtime_parameters = integrand_implementation.get(
                "dy_runtime_parameters"
            )
            if self.process_uses_z():
                z = integrand_implementation.get("z", z)
        evaluation_mode = "compiled"
        decimal_digit_precision = None
        theta_tolerance = 0.0
        if integrand_implementation is not None:
            evaluation_mode = str(
                integrand_implementation.get("dy_evaluation_mode", evaluation_mode)
            )
            if (
                "dy_fallback_precision" in integrand_implementation
                or "dy_rotation_check_arb_digits" in integrand_implementation
            ):
                decimal_digit_precision = self._dy_fallback_precision(
                    integrand_implementation
                )
            theta_tol = integrand_implementation.get("dy_theta_tol")
            if theta_tol is not None:
                theta_tolerance = float(theta_tol)

        evaluate_kwargs: dict[str, Any] = {
            "mode": evaluation_mode,
            "decimal_digit_precision": decimal_digit_precision,
            "theta_tolerance": theta_tolerance,
            "channel_selector": channel_selector,
        }
        if evaluation_mode == "compiled" and isinstance(
            self.compiled_bundle, DYCompiledBundle
        ):
            evaluate_kwargs["raise_on_t_solver_failure"] = True
        integrated_uv_ct_filter = (
            integrand_implementation.get("dy_integrated_uv_ct_filter")
            if integrand_implementation is not None
            else None
        )
        if (
            integrated_uv_ct_filter is not None
            and str(integrated_uv_ct_filter).replace("-", "_") != "all"
        ):
            evaluate_kwargs["integrated_uv_ct_filter"] = integrated_uv_ct_filter
        if (
            integrand_implementation is not None
            and integrand_implementation.get("dy_ttbar_pt_min") is not None
        ):
            evaluate_kwargs["ttbar_pt_min"] = self._validate_ttbar_pt_min(
                integrand_implementation["dy_ttbar_pt_min"]
            )
        if integrand_implementation is not None:
            if integrand_implementation.get("dy_physical_z_min") is not None:
                evaluate_kwargs["physical_z_min"] = integrand_implementation[
                    "dy_physical_z_min"
                ]
            if integrand_implementation.get("dy_physical_z_max") is not None:
                evaluate_kwargs["physical_z_max"] = integrand_implementation[
                    "dy_physical_z_max"
                ]

        value = self.compiled_bundle.evaluate(
            loop_momentum,
            p1,
            p2,
            z,
            m_uv,
            runtime_parameters=runtime_parameters,
            **evaluate_kwargs,
        )
        return value * self._dy_graph_channel_weight(
            integrand_implementation, channel_selector
        )

    def _zenos_arb_terms_with_externals(
        self,
        loop_momentum: list[Vector],
        p1: Vector,
        p2: Vector,
        integrand_implementation: dict[str, Any] | None,
        decimal_digit_precision: int,
        channel_selector: int | None = None,
    ) -> tuple[Decimal, list[tuple[str, Decimal]]]:
        if self.compiled_bundle is None:
            raise pygloopException(
                f"No compiled DY bundle loaded for integrand '{self.get_integrand_name()}'."
            )

        z: float | Decimal = 1.0
        m_uv: float | Decimal | None = None
        runtime_parameters: Mapping[str, Any] | None = None
        if integrand_implementation is not None:
            m_uv = integrand_implementation.get("mUV")
            runtime_parameters = integrand_implementation.get(
                "dy_runtime_parameters"
            )
            if self.process_uses_z():
                z = integrand_implementation.get("z", z)

        evaluate_kwargs: dict[str, Any] = {
            "decimal_digit_precision": decimal_digit_precision,
            "theta_tolerance": decimal_from_input(
                (integrand_implementation or {}).get("dy_theta_tol", 0.0)
            ),
            "channel_selector": channel_selector,
            "precision_preserving": True,
        }
        integrated_uv_ct_filter = (
            integrand_implementation.get("dy_integrated_uv_ct_filter")
            if integrand_implementation is not None
            else None
        )
        if (
            integrated_uv_ct_filter is not None
            and str(integrated_uv_ct_filter).replace("-", "_") != "all"
        ):
            evaluate_kwargs["integrated_uv_ct_filter"] = integrated_uv_ct_filter
        if (
            integrand_implementation is not None
            and integrand_implementation.get("dy_ttbar_pt_min") is not None
        ):
            evaluate_kwargs["ttbar_pt_min"] = self._validate_ttbar_pt_min(
                integrand_implementation["dy_ttbar_pt_min"]
            )
        if integrand_implementation is not None:
            if integrand_implementation.get("dy_physical_z_min") is not None:
                evaluate_kwargs["physical_z_min"] = decimal_from_input(
                    integrand_implementation["dy_physical_z_min"]
                )
            if integrand_implementation.get("dy_physical_z_max") is not None:
                evaluate_kwargs["physical_z_max"] = decimal_from_input(
                    integrand_implementation["dy_physical_z_max"]
                )

        histogram_orbits = (integrand_implementation or {}).get("_dy_histogram_orbits")
        cut_histogram = None
        if histogram_orbits is not None:
            cut_histogram = integrand_implementation["_dy_histogram"].fresh()
            evaluate_kwargs["term_observer"] = cut_histogram.add
        total, terms = self.compiled_bundle.evaluate_arb_terms(
            loop_momentum,
            p1,
            p2,
            z,
            m_uv,
            runtime_parameters=runtime_parameters,
            **evaluate_kwargs,
        )
        graph_weight = decimal_from_input(
            self._dy_graph_channel_weight(
                integrand_implementation, channel_selector
            )
        )
        if cut_histogram is not None:
            histogram_orbits.append([+(value * graph_weight) for value in cut_histogram.bins])
        return (
            +(total * graph_weight),
            [(name, +(value * graph_weight)) for name, value in terms],
        )

    def _zenos_arb_total_with_externals(
        self,
        loop_momentum: list[Vector],
        p1: Vector,
        p2: Vector,
        integrand_implementation: dict[str, Any] | None,
        decimal_digit_precision: int,
        channel_selector: int | None = None,
    ) -> Decimal:
        total, _terms = self._zenos_arb_terms_with_externals(
            loop_momentum,
            p1,
            p2,
            integrand_implementation,
            decimal_digit_precision,
            channel_selector=channel_selector,
        )
        return total

    def _normalize_integrand_implementation(
        self, integrand_implementation: dict[str, Any] | str
    ) -> dict[str, Any]:
        if isinstance(integrand_implementation, str):
            return {"integrand_type": integrand_implementation}
        return integrand_implementation

    def _dy_fallback_precision(
        self, integrand_implementation: dict[str, Any]
    ) -> int:
        ladder = integrand_implementation.get("dy_precision_ladder")
        if ladder is not None:
            ladder = validate_precision_ladder(ladder)
            return ladder[1] if len(ladder) > 1 else 16
        value = integrand_implementation.get("dy_fallback_precision")
        if value is None:
            value = integrand_implementation.get("dy_rotation_check_arb_digits")
        if value is None:
            value = self.dy_fallback_precision
        return int(value)

    def _dy_precision_ladder(self, implementation):
        levels = implementation.get("dy_precision_ladder")
        if levels is None:
            fallback = self._dy_fallback_precision(implementation)
            levels = (16,) if fallback == 16 else (16, fallback)
        levels = validate_precision_ladder(levels)
        for digits in levels:
            for event in ("attempt", "accepted", "failed"):
                name = f"stability_precision_{digits}_{event}_count"
                if not hasattr(self, name):
                    setattr(self, name, 0)
        return levels

    def _count_precision(self, digits, event):
        name = f"stability_precision_{digits}_{event}_count"
        setattr(self, name, getattr(self, name, 0) + 1)

    @staticmethod
    def _integrand_weight_is_finite(value: complex) -> bool:
        try:
            z = complex(value)
        except (OverflowError, TypeError, ValueError):
            return False
        return (
            math.isfinite(z.real)
            and math.isfinite(z.imag)
            and math.isfinite(math.hypot(z.real, z.imag))
        )


    def _sanitize_integrand_weight(
        self,
        value: complex,
        xs: list[float],
        momentum_point: str,
        *,
        rotated: bool,
    ) -> complex:
        if self._integrand_weight_is_finite(value):
            return complex(value)

        if rotated:
            self.nan_weight_rotated_count += 1
            if self.nan_weight_rotated_example is None:
                self.nan_weight_rotated_example = list(xs)
                self.nan_weight_rotated_example_momentum_point = momentum_point
        else:
            self.nan_weight_count += 1
            if self.nan_weight_example is None:
                self.nan_weight_example = list(xs)
                self.nan_weight_example_momentum_point = momentum_point

        logger.debug(
            f"Integrand evaluated to non-finite {'rotated ' if rotated else ''}weight "
            f"at xs = [{Colour.BLUE}{', '.join(f'{xi:+.16e}' for xi in xs)}{Colour.END}]. "
            "Setting it to zero"
        )
        return 0.0 + 0.0j

    @staticmethod
    def _call_args_use_zenos_integrand(call_args: list[Any]) -> bool:
        if len(call_args) < 2:
            return False
        integrand_implementation = call_args[1]
        if isinstance(integrand_implementation, str):
            return integrand_implementation == "zenos"
        if isinstance(integrand_implementation, dict):
            return integrand_implementation.get("integrand_type") == "zenos"
        return False

    @staticmethod
    def _copy_stability_diagnostics(
        process: DY, result: IntegrationResult
    ) -> None:
        for name, value in vars(process).items():
            if name.startswith("stability_precision_") and name.endswith("_count"):
                setattr(result, name, value)
        for attribute in (
            "large_weight_zeroed_signed_sum",
            "large_weight_zeroed_abs_sum",
            "stability_float_pair_accepted_count",
            "stability_float_mismatch_retry_count",
            "stability_float_nonfinite_retry_count",
            "stability_hp_retry_count",
            "stability_hp_accepted_count",
            "stability_hp_escalation_count",
            "stability_hp_escalation_accepted_count",
            "stability_hp_disagreement_count",
            "stability_hp_nonfinite_count",
            "stability_hp_error_count",
            "stability_hp_failure_example",
            "stability_hp_failure_example_momentum_point",
            "stability_hp_failure_reason",
            "t_solver_float_failure_sample_count",
            "t_solver_float_failed_term_count",
            "t_solver_float_failed_surface_count",
            "t_solver_hp_retry_count",
            "t_solver_hp_salvaged_count",
            "t_solver_hp_unresolved_count",
            "t_solver_failure_surfaces",
            "t_solver_failure_terms",
            "soft_mirror_pair_count",
            "soft_mirror_large_trigger_count",
            "soft_mirror_hp_orbit_retry_count",
            "soft_mirror_hp_orbit_salvaged_count",
            "soft_mirror_hp_orbit_failure_count",
            "soft_mirror_max_raw_side_wgt",
            "soft_mirror_max_raw_side_wgt_point",
            "soft_mirror_max_raw_side",
            "soft_mirror_max_post_average_wgt",
            "soft_mirror_max_post_average_wgt_point",
            "soft_mirror_min_residual_ratio",
            "soft_mirror_min_residual_ratio_point",
            "soft_mirror_min_residual_ratio_sides",
            "max_preclip_wgt",
            "max_preclip_wgt_point",
            "max_preclip_wgt_momentum_point",
        ):
            setattr(result, attribute, getattr(process, attribute))

    def integrate(
        self,
        integrator: str,
        parameterisation: str,
        integrand_implementation: dict[str, Any],
        target: float | complex | None = None,
        toml_config_path: str | None = None,
        **opts,
    ) -> IntegrationResult:
        integrand_implementation = self._normalize_integrand_implementation(
            integrand_implementation
        )
        if (
            self.integrate_beams
            and integrand_implementation.get("integrand_type") != "zenos"
        ):
            raise pygloopException(
                "DY beam integration mode currently only supports the 'zenos' integrand implementation."
            )
        if integrand_implementation.get("integrand_type") == "zenos":
            rotation_digits_value = integrand_implementation.get(
                "dy_rotation_check_digits"
            )
            rotation_digits = (
                int(rotation_digits_value)
                if rotation_digits_value is not None
                else 0
            )
            if rotation_digits < 0:
                raise pygloopException(
                    "DY rotation-check digits must be non-negative."
                )
            self._rotation_check_count(integrand_implementation)
            precision_threshold, clip_threshold = self._stability_thresholds(
                integrand_implementation
            )
            stability_requested = (
                rotation_digits > 0
                or precision_threshold is not None
                or clip_threshold is not None
            )
            if stability_requested:
                self._stability_tolerances(
                    integrand_implementation, rotation_digits
                )
                fallback_precision = self._dy_fallback_precision(
                    integrand_implementation
                )
                if fallback_precision < 2:
                    raise pygloopException(
                        "DY fallback precision must be at least two digits."
                    )
                if self.compiled_bundle is None:
                    raise pygloopException(
                        "DY stability validation requires a compiled zenos bundle."
                    )
                for digits in self._dy_precision_ladder(integrand_implementation)[1:]:
                    self.compiled_bundle.require_fallback_supported(digits)
        match integrator:
            case "naive":
                integration_result = self.naive_integrator(
                    parameterisation,
                    integrand_implementation,
                    target,
                    **opts,
                )
            case "vegas":
                integration_result = DY.vegas_integrator(
                    self,
                    parameterisation,
                    integrand_implementation,
                    target,
                    **opts,
                )
            case "symbolica":
                integration_result = self.symbolica_integrator(
                    parameterisation,
                    integrand_implementation,
                    target,
                    **opts,
                )
            case "gammaloop":
                integration_result = self.gammaloop_integrator(target, **opts)
            case _:
                raise pygloopException(f"Integrator {integrator} not implemented.")
        if self.dy_msbar_scheme_counterterm:
            if self._dy_ttbar_gg_scheme_mode:
                integration_result = self._apply_ttbar_gg_scheme(
                    integration_result,
                    int(opts.get("seed", 1337)),
                    parameterisation,
                    str(opts.get("phase", "real")),
                    int(opts.get("n_cores", 1)),
                )
            elif self._dy_ttbar_qqbar_scheme_mode:
                integration_result = self._apply_ttbar_qqbar_scheme_and_decoupling(
                    integration_result,
                    int(opts.get("seed", 1337)),
                    parameterisation,
                    str(opts.get("phase", "real")),
                    int(opts.get("n_cores", 1)),
                )
            else:
                integration_result = self._add_dy_msbar_scheme_counterterm(
                    integration_result,
                    int(opts.get("seed", 1337)),
                    parameterisation,
                    str(opts.get("phase", "real")),
                    int(opts.get("n_cores", 1)),
                )
        if integrand_implementation.get("integrand_type") == "zenos":
            integration_result.precision_ladder = list(self._dy_precision_ladder(integrand_implementation))
            integration_result.precision_failure_action = "zero_complete_sample"
        return self._attach_dy_coupling_normalisation_metadata(
            integration_result
        )

    def gammaloop_integrator(
        self,
        target: float | complex | None = None,
        **opts,
    ) -> IntegrationResult:
        if opts.get("integrand_implementation", "gammaloop") != "gammaloop":
            raise pygloopException(
                "GammaLoop integrator only supports 'gammaloop' integrand implementation."
            )

        integrand_name = self.get_integrand_name()
        amplitudes, _cross_sections = self.gl_worker.list_outputs()
        if integrand_name not in amplitudes:
            raise pygloopException(
                f"Amplitude {integrand_name} not found in GammaLoop state. Generate graphs and code first with the generate subcommand. Available amplitudes: {list(amplitudes.keys())}"
            )  # nopep8

        integration_options = {
            "n_start": opts.get("points_per_iteration", 100_000),
            "n_increase": 0,
            "n_max": opts.get("points_per_iteration", 100_000)
            * opts.get("n_iterations", 10),
            "integrated_phase": opts.get("phase", "real"),
            "seed": opts.get("seed", 1337),
        }
        self.gl_worker.run(
            f"set process -p {amplitudes[integrand_name]} -i {integrand_name} kv {' '.join('integrator.%s=%s' % (k, str(v)) for k, v in integration_options.items())}"
        )

        workspace_dir = pjoin(INTEGRATION_WORKSPACE_FOLDER, self.name, integrand_name)
        if not os.path.exists(workspace_dir):
            os.makedirs(workspace_dir, exist_ok=True)
        results_path = pjoin(workspace_dir, "result.txt")
        integrate_command = [
            [
                "integrate",
            ],
            ["-p", str(amplitudes[integrand_name])],
            ["-i", integrand_name],
            ["--workspace-path", f"{workspace_dir}"],
            ["--result-path", f"{results_path}"],
        ]
        if target is not None:
            if isinstance(target, complex):
                integrate_command.append([
                    "--target",
                    f"{target.real:.16e}",
                    f"{target.imag:.16e}",
                ])
            elif isinstance(target, float):
                integrate_command.append(["--target", f"{target:.16e}", "0.0"])
        if "n_cores" in opts:
            integrate_command.append(["--n-cores", str(opts["n_cores"])])
        if opts.get("restart", False):
            integrate_command.append(["--restart"])

        integrate_command_str = " ".join(
            " ".join(itg_o for itg_o in itg_opt) for itg_opt in integrate_command
        )
        logger.info(
            f"Running GammaLoop integration with command:\n{Colour.GREEN}{integrate_command_str}{Colour.END}"
        )
        t_start = time.time()
        self.gl_worker.run(integrate_command_str)  # nopep8
        t_elapsed = time.time() - t_start

        res = None
        if os.path.isfile(results_path):
            with open(results_path, "r") as f_res:
                res = json.load(f_res)

        integration_result = IntegrationResult(0.0, 0.0)
        if res is None:
            logger.error(
                f"GammaLoop integration finished but no result file found at '{results_path}'."
            )
        else:
            if opts.get("phase", "real") == "real":
                central, error = res["result"]["re"], res["error"]["re"]
            else:
                central, error = res["result"]["im"], res["error"]["im"]
            integration_result = IntegrationResult(
                central, error, n_samples=res["neval"], elapsed_time=t_elapsed
            )
        return integration_result

    @staticmethod
    def naive_worker(
        builder_inputs: tuple[Any], n_points: int, call_args: list[Any]
    ) -> IntegrationResult:
        process_instance = DY(
            *builder_inputs, clean=False, logger_level=logging.CRITICAL
        )  # type: ignore
        this_result = IntegrationResult(0.0, 0.0)
        t_start = time.time()
        n_dim = process_instance.integration_dimension(call_args[1])
        for _ in range(n_points):
            xs = [random.random() for _ in range(n_dim)]
            weight = process_instance.integrand_xspace(xs, *call_args)
            if this_result.max_wgt is None or abs(weight) > abs(this_result.max_wgt):
                this_result.max_wgt = weight
                this_result.max_wgt_point = xs
            this_result.central_value += weight
            this_result.error += weight**2
            this_result.n_samples += 1
        this_result.elapsed_time += time.time() - t_start
        this_result.max_wgt = process_instance.max_wgt
        this_result.max_wgt_point = process_instance.max_wgt_point
        this_result.max_wgt_jacobian = process_instance.max_wgt_jacobian
        this_result.max_wgt_momentum_point = process_instance.max_wgt_momentum_point
        this_result.unstable_count = process_instance.rotation_unstable_count
        this_result.unstable_retry_count = process_instance.rotation_hp_retry_count
        this_result.unstable_salvaged_count = (
            process_instance.rotation_hp_salvaged_count
        )
        this_result.unstable_retry_example = process_instance.rotation_hp_retry_example
        this_result.unstable_retry_example_momentum_point = (
            process_instance.rotation_hp_retry_example_momentum_point
        )
        this_result.unstable_retry_example_rel = (
            process_instance.rotation_hp_retry_example_rel
        )
        this_result.unstable_example = process_instance.rotation_unstable_example
        this_result.unstable_example_momentum_point = (
            process_instance.rotation_unstable_example_momentum_point
        )
        this_result.large_weight_retry_count = (
            process_instance.large_weight_hp_retry_count
        )
        this_result.large_weight_salvaged_count = (
            process_instance.large_weight_hp_salvaged_count
        )
        this_result.large_weight_unstable_count = (
            process_instance.large_weight_unstable_count
        )
        this_result.large_weight_zeroed_count = (
            process_instance.large_weight_zeroed_count
        )
        this_result.nan_weight_count = process_instance.nan_weight_count
        this_result.nan_weight_rotated_count = process_instance.nan_weight_rotated_count
        this_result.nan_weight_example = process_instance.nan_weight_example
        this_result.nan_weight_example_momentum_point = (
            process_instance.nan_weight_example_momentum_point
        )
        this_result.nan_weight_rotated_example = (
            process_instance.nan_weight_rotated_example
        )
        this_result.nan_weight_rotated_example_momentum_point = (
            process_instance.nan_weight_rotated_example_momentum_point
        )
        this_result.large_weight_retry_example = (
            process_instance.large_weight_retry_example
        )
        this_result.large_weight_retry_example_momentum_point = (
            process_instance.large_weight_retry_example_momentum_point
        )
        this_result.large_weight_retry_example_compiled_wgt = (
            process_instance.large_weight_retry_example_compiled_wgt
        )
        this_result.large_weight_retry_example_arb_wgt = (
            process_instance.large_weight_retry_example_arb_wgt
        )
        this_result.max_stable_wgt = process_instance.max_stable_wgt
        this_result.max_stable_wgt_point = process_instance.max_stable_wgt_point
        this_result.max_stable_wgt_jacobian = process_instance.max_stable_wgt_jacobian
        this_result.max_stable_wgt_momentum_point = (
            process_instance.max_stable_wgt_momentum_point
        )
        copy_stability_diagnostics = getattr(
            DY, "_copy_stability_diagnostics", None
        )
        if copy_stability_diagnostics is not None:
            copy_stability_diagnostics(process_instance, this_result)

        return this_result

    @set_gammaloop_level(logging.ERROR, logging.INFO)
    def naive_integrator(
        self,
        parameterisation: str,
        integrand_implementation: dict[str, Any],
        target,
        **opts,
    ) -> IntegrationResult:
        integration_result = IntegrationResult(0.0, 0.0)

        function_call_args = [
            parameterisation,
            integrand_implementation,
            opts["phase"],
            opts["multi_channeling"],
        ]
        for i_iter in range(opts["n_iterations"]):
            logger.info(
                f"Naive integration: starting iteration {Colour.GREEN}{i_iter + 1}/{
                    opts['n_iterations']
                }{Colour.END} using {Colour.BLUE}{opts['points_per_iteration']}{
                    Colour.END
                } points ..."
            )
            if opts["n_cores"] > 1:
                n_points_per_core = opts["points_per_iteration"] // opts["n_cores"]
                all_args = [
                    (self.builder_inputs(), n_points_per_core, function_call_args),
                ] * (opts["n_cores"] - 1)
                all_args.append((
                    self.builder_inputs(),
                    opts["points_per_iteration"] - sum(a[1] for a in all_args),
                    function_call_args,
                ))
                with multiprocessing.Pool(processes=opts["n_cores"]) as pool:
                    all_results = pool.starmap(DY.naive_worker, all_args)

                # Combine results
                for result in all_results:
                    integration_result.combine_with(result)
            else:
                integration_result.combine_with(
                    DY.naive_worker(
                        self.builder_inputs(),
                        opts["points_per_iteration"],
                        function_call_args,
                    )
                )
            # Normalize a copy for temporary printout
            processed_result = copy.deepcopy(integration_result)
            processed_result.normalize()
            logger.info(
                f"... result after this iteration:\n{processed_result.str_report(target)}"
            )

        # Normalize results
        integration_result.normalize()

        return integration_result

    @staticmethod
    def vegas_worker(
        process_builder_inputs: dict[str, Any],
        id: int,
        all_xs: list[list[float]],
        call_args: list[Any],
        skip_gl_worker_init: bool = False,
    ) -> tuple[int, list[float], IntegrationResult]:
        res = IntegrationResult(0.0, 0.0)
        t_start = time.time()
        all_weights = []
        process = DY(
            **process_builder_inputs,
            clean=False,
            logger_level=logging.CRITICAL,
            skip_gl_worker_init=skip_gl_worker_init,
        )  # type: ignore
        for xs in all_xs:
            weight = process.integrand_xspace(xs, *call_args)
            all_weights.append(weight)
            if res.max_wgt is None or abs(weight) > abs(res.max_wgt):
                res.max_wgt = weight
                res.max_wgt_point = xs
            res.central_value += weight
            res.error += weight**2
            res.n_samples += 1
        res.elapsed_time += time.time() - t_start
        res.max_wgt = process.max_wgt
        res.max_wgt_point = process.max_wgt_point
        res.max_wgt_jacobian = process.max_wgt_jacobian
        res.max_wgt_momentum_point = process.max_wgt_momentum_point
        res.unstable_count = process.rotation_unstable_count
        res.unstable_retry_count = process.rotation_hp_retry_count
        res.unstable_salvaged_count = process.rotation_hp_salvaged_count
        res.unstable_retry_example = process.rotation_hp_retry_example
        res.unstable_retry_example_momentum_point = (
            process.rotation_hp_retry_example_momentum_point
        )
        res.unstable_retry_example_rel = process.rotation_hp_retry_example_rel
        res.unstable_example = process.rotation_unstable_example
        res.unstable_example_momentum_point = (
            process.rotation_unstable_example_momentum_point
        )
        res.large_weight_retry_count = process.large_weight_hp_retry_count
        res.large_weight_salvaged_count = process.large_weight_hp_salvaged_count
        res.large_weight_unstable_count = process.large_weight_unstable_count
        res.large_weight_zeroed_count = process.large_weight_zeroed_count
        res.nan_weight_count = process.nan_weight_count
        res.nan_weight_rotated_count = process.nan_weight_rotated_count
        res.nan_weight_example = process.nan_weight_example
        res.nan_weight_example_momentum_point = (
            process.nan_weight_example_momentum_point
        )
        res.nan_weight_rotated_example = process.nan_weight_rotated_example
        res.nan_weight_rotated_example_momentum_point = (
            process.nan_weight_rotated_example_momentum_point
        )
        res.large_weight_retry_example = process.large_weight_retry_example
        res.large_weight_retry_example_momentum_point = (
            process.large_weight_retry_example_momentum_point
        )
        res.large_weight_retry_example_compiled_wgt = (
            process.large_weight_retry_example_compiled_wgt
        )
        res.large_weight_retry_example_arb_wgt = (
            process.large_weight_retry_example_arb_wgt
        )
        res.max_stable_wgt = process.max_stable_wgt
        res.max_stable_wgt_point = process.max_stable_wgt_point
        res.max_stable_wgt_jacobian = process.max_stable_wgt_jacobian
        res.max_stable_wgt_momentum_point = process.max_stable_wgt_momentum_point
        copy_stability_diagnostics = getattr(
            DY, "_copy_stability_diagnostics", None
        )
        if copy_stability_diagnostics is not None:
            copy_stability_diagnostics(process, res)

        return (id, all_weights, res)

    @staticmethod
    def vegas_functor(
        process: DY, res: IntegrationResult, n_cores: int, call_args: list[Any]
    ) -> Callable[[list[list[float]]], list[float]]:
        @vegas.batchintegrand
        def f(all_xs):
            all_weights = []
            if n_cores > 1:
                skip_gl_worker_init = DY._call_args_use_zenos_integrand(call_args)
                all_args = [
                    (
                        process.builder_inputs(),
                        i_chunk,
                        all_xs_split,
                        call_args,
                        skip_gl_worker_init,
                    )
                    for i_chunk, all_xs_split in enumerate(
                        chunks(all_xs, len(all_xs) // n_cores + 1)
                    )
                ]
                with multiprocessing.Pool(processes=n_cores) as pool:
                    all_results = pool.starmap(DY.vegas_worker, all_args)
                for _id, wgts, this_result in sorted(all_results, key=lambda x: x[0]):
                    all_weights.extend(wgts)
                    res.combine_with(this_result)
                return all_weights
            else:
                _id, wgts, this_result = DY.vegas_worker(
                    process.builder_inputs(), 0, all_xs, call_args
                )
                all_weights.extend(wgts)
                res.combine_with(this_result)
            return all_weights

        return f

    @set_gammaloop_level(logging.ERROR, logging.INFO)
    def vegas_integrator(
        self,
        parameterisation: str,
        integrand_implementation: dict[str, Any],
        _target,
        **opts,
    ) -> IntegrationResult:
        integration_result = IntegrationResult(0.0, 0.0)

        n_dim = self.integration_dimension(integrand_implementation)
        integrator = vegas.Integrator(n_dim * [[0, 1]])  # fmt: off

        local_worker = DY.vegas_functor(
            self,
            integration_result,
            opts["n_cores"],
            [
                parameterisation,
                integrand_implementation,
                opts.get("phase", "real"),
                opts["multi_channeling"],
            ],
        )
        # Adapt grid
        integrator(
            local_worker,
            nitn=opts["n_iterations"],
            neval=opts["points_per_iteration"],
            analyzer=vegas.reporter(),
        )
        # Final result
        result = integrator(
            local_worker,
            nitn=opts["n_iterations"],
            neval=opts["points_per_iteration"],
            analyzer=vegas.reporter(),
        )

        integration_result.central_value = result.mean
        integration_result.error = result.sdev
        return integration_result

    @staticmethod
    def symbolica_worker(
        process_builder_inputs: dict[str, Any],
        id: int,
        multi_channeling: bool,
        all_xs: list[SymbolicaSample],
        call_args: list[Any],
        graph_channel_indices: list[int] | None = None,
        skip_gl_worker_init: bool = False,
    ) -> tuple[int, list[float], IntegrationResult]:
        res = IntegrationResult(0.0, 0.0)
        t_start = time.time()
        all_weights = []
        process = DY(
            **process_builder_inputs,
            clean=False,
            logger_level=logging.CRITICAL,
            skip_gl_worker_init=skip_gl_worker_init,
        )  # type: ignore
        for xs in all_xs:
            if not multi_channeling:
                weight = process.integrand_xspace(xs.c, *( call_args + [False, ]))  # fmt: off
            else:
                local_channel_index = int(xs.d[0])
                channel_index = (
                    graph_channel_indices[local_channel_index]
                    if graph_channel_indices is not None
                    else local_channel_index
                )
                weight = process.integrand_xspace(
                    xs.c, *(call_args + [channel_index])
                )
            all_weights.append(weight)
            if res.max_wgt is None or abs(weight) > abs(res.max_wgt):
                res.max_wgt = weight
                if not multi_channeling:
                    res.max_wgt_point = xs.c
                else:
                    res.max_wgt_point = xs.d + xs.c
            res.central_value += weight
            res.error += weight**2
            res.n_samples += 1
        res.elapsed_time += time.time() - t_start
        res.max_wgt = process.max_wgt
        res.max_wgt_point = process.max_wgt_point
        res.max_wgt_jacobian = process.max_wgt_jacobian
        res.max_wgt_momentum_point = process.max_wgt_momentum_point
        res.unstable_count = process.rotation_unstable_count
        res.unstable_retry_count = process.rotation_hp_retry_count
        res.unstable_salvaged_count = process.rotation_hp_salvaged_count
        res.unstable_retry_example = process.rotation_hp_retry_example
        res.unstable_retry_example_momentum_point = (
            process.rotation_hp_retry_example_momentum_point
        )
        res.unstable_retry_example_rel = process.rotation_hp_retry_example_rel
        res.unstable_example = process.rotation_unstable_example
        res.unstable_example_momentum_point = (
            process.rotation_unstable_example_momentum_point
        )
        res.large_weight_retry_count = process.large_weight_hp_retry_count
        res.large_weight_salvaged_count = process.large_weight_hp_salvaged_count
        res.large_weight_unstable_count = process.large_weight_unstable_count
        res.large_weight_zeroed_count = process.large_weight_zeroed_count
        res.nan_weight_count = process.nan_weight_count
        res.nan_weight_rotated_count = process.nan_weight_rotated_count
        res.nan_weight_example = process.nan_weight_example
        res.nan_weight_example_momentum_point = (
            process.nan_weight_example_momentum_point
        )
        res.nan_weight_rotated_example = process.nan_weight_rotated_example
        res.nan_weight_rotated_example_momentum_point = (
            process.nan_weight_rotated_example_momentum_point
        )
        res.large_weight_retry_example = process.large_weight_retry_example
        res.large_weight_retry_example_momentum_point = (
            process.large_weight_retry_example_momentum_point
        )
        res.large_weight_retry_example_compiled_wgt = (
            process.large_weight_retry_example_compiled_wgt
        )
        res.large_weight_retry_example_arb_wgt = (
            process.large_weight_retry_example_arb_wgt
        )
        res.max_stable_wgt = process.max_stable_wgt
        res.max_stable_wgt_point = process.max_stable_wgt_point
        res.max_stable_wgt_jacobian = process.max_stable_wgt_jacobian
        res.max_stable_wgt_momentum_point = process.max_stable_wgt_momentum_point
        copy_stability_diagnostics = getattr(
            DY, "_copy_stability_diagnostics", None
        )
        if copy_stability_diagnostics is not None:
            copy_stability_diagnostics(process, res)

        return (id, all_weights, res)

    @staticmethod
    def symbolica_integrand_function(
        process: DY,
        res: IntegrationResult,
        n_cores: int,
        multi_channeling: bool,
        call_args: list[Any],
        samples: list[Sample],
        graph_channel_indices: list[int] | None = None,
    ) -> list[float]:
        all_weights = []
        if n_cores > 1:
            skip_gl_worker_init = DY._call_args_use_zenos_integrand(call_args)
            all_args = [
                (
                    process.builder_inputs(),
                    i_chunk,
                    multi_channeling,
                    [SymbolicaSample(s) for s in all_xs_split],
                    call_args,
                    graph_channel_indices,
                    skip_gl_worker_init,
                )
                for i_chunk, all_xs_split in enumerate(
                    chunks(samples, len(samples) // n_cores + 1)
                )
            ]
            with multiprocessing.Pool(processes=n_cores) as pool:
                all_results = pool.starmap(DY.symbolica_worker, all_args)
            for _id, wgts, this_result in sorted(all_results, key=lambda x: x[0]):
                all_weights.extend(wgts)
                res.combine_with(this_result)
            return all_weights
        else:
            _id, wgts, this_result = DY.symbolica_worker(
                process.builder_inputs(),
                0,
                multi_channeling,
                [SymbolicaSample(s) for s in samples],
                call_args,
                graph_channel_indices,
            )
            all_weights.extend(wgts)
            res.combine_with(this_result)
        return all_weights

    @set_gammaloop_level(logging.ERROR, logging.INFO)
    def symbolica_integrator(
        self,
        parameterisation: str,
        integrand_implementation: dict[str, Any],
        target,
        **opts,
    ) -> IntegrationResult:
        integration_result = IntegrationResult(0.0, 0.0)
        continuous_learning_rate = 1.0
        discrete_learning_rate = 1.0

        # continuous_learning_rate = 0.0
        # discrete_learning_rate = 0.0

        n_dim = self.integration_dimension(integrand_implementation)

        graph_weights = integrand_implementation.get("dy_graph_weights")
        if graph_weights is not None:
            if not opts["multi_channeling"]:
                raise pygloopException(
                    "DY graph weights require graph multi-channeling."
                )
            if self.compiled_bundle is None:
                raise pygloopException("DY graph weights require a compiled bundle.")
            expected_graph_weights = self.compiled_bundle.graph_channel_count()
            if len(graph_weights) != expected_graph_weights:
                raise pygloopException(
                    f"Expected {expected_graph_weights} DY graph weights in bundle "
                    f"channel order, got {len(graph_weights)}."
                )

        requested_graphs = opts.get("dy_integration_graphs")
        if requested_graphs is not None and not opts["multi_channeling"]:
            raise pygloopException(
                "--dy-integration-graphs requires --multi_channeling."
            )

        if opts["multi_channeling"]:
            all_graph_channel_names = self.graph_channel_names(
                integrand_implementation
            )
            if len(all_graph_channel_names) == 0:
                raise pygloopException(
                    "DY Symbolica multi-channeling requires a compiled zenos bundle "
                    "with graph-grouped evaluators."
                )
            graph_channel_indices = self._integration_graph_channel_indices(
                integrand_implementation, requested_graphs
            )
            if graph_channel_indices is None:
                graph_channel_indices = list(range(len(all_graph_channel_names)))
            graph_channel_names = [
                self._integration_graph_channel_name(
                    integrand_implementation, index
                )
                for index in graph_channel_indices
            ]
            logger.info(
                "Symbolica discrete graph channels: %s (bundle indices: %s)",
                ", ".join(graph_channel_names),
                ", ".join(str(index) for index in graph_channel_indices),
            )
            assert self.compiled_bundle is not None
            ghost_counts = self.compiled_bundle.graph_channel_closed_ghost_loop_counts()
            automatic_factors = self.compiled_bundle.graph_channel_automatic_factors()
            for graph_name, graph_index in zip(
                graph_channel_names,
                graph_channel_indices,
                strict=True,
            ):
                if graph_index == DY_GROUPED_SOFT_PAIR_SELECTOR:
                    pair = self._grouped_soft_pair(integrand_implementation)
                    assert pair is not None
                    logger.info(
                        "DY grouped soft-pair channel %s: bundle indices %d,%d; "
                        "transform=%s",
                        graph_name,
                        pair[0],
                        pair[1],
                        pair[2],
                    )
                    continue
                explicit_factor = (
                    float(graph_weights[graph_index])
                    if graph_weights is not None
                    else 1.0
                )
                automatic_factor = automatic_factors[graph_index]
                logger.info(
                    "DY graph factor %s: closed_ghost_loops=%d automatic=%+d "
                    "explicit=%+.16e combined=%+.16e",
                    graph_name,
                    ghost_counts[graph_index],
                    automatic_factor,
                    explicit_factor,
                    automatic_factor * explicit_factor,
                )
            integrator = self._build_symbolica_discrete_integrator(
                n_dim, len(graph_channel_names)
            )
            graph_channel_observers = [
                self._build_symbolica_discrete_integrator(
                    n_dim, len(graph_channel_names)
                )
                for _graph_channel_name in graph_channel_names
            ]
        else:
            graph_channel_names = []
            graph_channel_indices = None
            integrator = NumericalIntegrator.continuous(n_dim)
            graph_channel_observers = []

        rng = integrator.rng(seed=opts["seed"], stream_id=0)

        for i_iter in range(opts["n_iterations"]):
            logger.info(
                f"Symbolica integration: starting iteration {Colour.GREEN}{i_iter + 1}/{opts['n_iterations']}{Colour.END} using {Colour.BLUE}{opts['points_per_iteration']}{Colour.END} points ..."
            )  # nopep8
            samples = integrator.sample(opts["points_per_iteration"], rng)
            res = DY.symbolica_integrand_function(
                self,
                integration_result,
                opts["n_cores"],
                opts["multi_channeling"],
                [parameterisation, integrand_implementation, opts.get("phase", "real")],
                samples,
                graph_channel_indices,
            )
            integrator.add_training_samples(samples, res)

            graph_channel_contributions: list[tuple[str, float, float, int]] = []
            avg, err, _chi_sq = integrator.update(
                continuous_learning_rate=continuous_learning_rate,
                discrete_learning_rate=discrete_learning_rate,
            )  # type: ignore
            if opts["multi_channeling"]:
                graph_channel_contributions = (
                    self._symbolica_graph_channel_contributions(
                        graph_channel_names,
                        graph_channel_observers,
                        samples,
                        res,
                        continuous_learning_rate,
                        discrete_learning_rate,
                    )
                )
            integration_result.central_value = avg
            integration_result.error = err
            logger.info(
                f"... result after this iteration:\n{integration_result.str_report(target)}"
            )
            if graph_channel_contributions:
                logger.info(
                    self._symbolica_graph_channel_report(
                        graph_channel_contributions, avg
                    )
                )

        return integration_result

    def benchmark_integrand_evaluation(
        self,
        integrand_implementation: dict[str, Any],
        n_evals: int = 1000,
        parameterisation: str = "cartesian",
        phase: str = "real",
        multi_channeling: bool | int = False,
        seed: int = 1337,
    ) -> dict[str, float]:
        """
        Benchmark integrand evaluation cost over n_evals random x-space points.
        Returns timing summary in seconds / microseconds.
        """
        random.seed(seed)

        n_dim = self.integration_dimension(integrand_implementation)

        t0 = time.perf_counter()
        acc = 0.0
        for _ in range(n_evals):
            xs = [random.random() for _ in range(n_dim)]
            w = self.integrand_xspace(
                xs,
                parameterisation,
                integrand_implementation,
                phase,
                multi_channeling,
            )
            acc += abs(w)  # prevent potential optimization/elision
        elapsed = time.perf_counter() - t0

        return {
            "n_evals": float(n_evals),
            "elapsed_s": elapsed,
            "per_eval_us": 1.0e6 * elapsed / n_evals,
            "evals_per_s": n_evals / elapsed if elapsed > 0 else float("inf"),
            "checksum": acc,
        }

    @set_gammaloop_level(logging.ERROR, logging.INFO)
    def plot(self, **opts):
        import matplotlib.pyplot as plt  # type: ignore # nopep8
        import numpy as np  # pyright: ignore
        from mpl_toolkits.mplot3d import (  # pyright: ignore
            Axes3D,  # type: ignore # noqa: F401 # nopep8 # fmt: off
        )

        fixed_x = None
        for i_x in range(3):
            if i_x not in opts["xs"]:
                fixed_x = i_x
                break
        if fixed_x is None:
            raise pygloopException("At least one x must be fixed (0,1 or 2).")
        n_bins = opts["mesh_size"]
        # Create a grid of x and y values within the range [0., 1.]
        # Apply small offset to avoid divisions by zero
        offset = 1e-6
        x = np.linspace(opts["range"][0] + offset, opts["range"][1] - offset, n_bins)
        y = np.linspace(opts["range"][0] + offset, opts["range"][1] - offset, n_bins)
        X, Y = np.meshgrid(x, y)

        # Calculate the values of f(x, y) for each point in the grid
        Z = np.zeros((n_bins, n_bins))
        # Calculate the values of f(x, y) for each point in the grid using nested loops
        xs = [
            0.0,
        ] * 3
        xs[fixed_x] = opts["fixed_x"]
        nb_cores = max(1, int(opts.get("nb_cores", 1)))
        total = n_bins * n_bins
        logger.info(
            f"Evaluating function on grid for plotting over {nb_cores} cores..."
        )

        def sequential_plotting():
            for idx in progressbar.progressbar(range(total), max_value=total):
                i, j = divmod(idx, n_bins)
                xs[opts["xs"][0]] = X[i, j]
                xs[opts["xs"][1]] = Y[i, j]
                if opts["x_space"]:
                    Z[i, j] = self.integrand_xspace(  # type: ignore
                        xs,  # type: ignore
                        opts["parameterisation"],
                        opts["integrand_implementation"],
                        opts.get("phase", "real"),
                        opts["multi_channeling"],
                    )
                else:
                    wgt = self.integrand(
                        [Vector(xs[0], xs[1], xs[2])],  # pyright: ignore
                        opts["integrand_implementation"],
                    )  # type: ignore
                    match opts.get("phase", None):
                        case "real":
                            Z[i, j] = wgt.real  # type: ignore
                        case "imag":
                            Z[i, j] = wgt.imag  # type: ignore
                        case _:
                            Z[i, j] = abs(wgt)  # type: ignore

        if nb_cores == 1:
            sequential_plotting()
        else:
            config = {
                "fixed_x": fixed_x,
                "fixed_value": opts["fixed_x"],
                "xs0": opts["xs"][0],
                "xs1": opts["xs"][1],
                "x_space": opts["x_space"],
                "parameterisation": opts["parameterisation"],
                "integrand_implementation": opts["integrand_implementation"],
                "phase": opts.get("phase", "real")
                if opts["x_space"]
                else opts.get("phase", None),
                "multi_channeling": opts["multi_channeling"],
            }
            try:
                ctx = multiprocessing.get_context("fork")
                chunk_size = max(1, total // (nb_cores * 4))
                tasks = (
                    (i, j, float(X[i, j]), float(Y[i, j]))
                    for i in range(n_bins)
                    for j in range(n_bins)
                )
                with ctx.Pool(
                    processes=nb_cores,
                    initializer=_plot_worker_init,
                    initargs=(self, config),
                ) as pool:
                    for i, j, val in progressbar.progressbar(  # type: ignore
                        pool.imap_unordered(_plot_worker, tasks, chunksize=chunk_size),
                        max_value=total,
                    ):
                        Z[i, j] = val
            except ValueError:
                logger.warning(
                    "Multiprocessing start method does not support forking; running sequentially."
                )
                sequential_plotting()
        logger.info("Done")

        # Take the logarithm of the function values, handling cases where the value is 0
        with np.errstate(divide="ignore"):
            log_Z = np.log10(np.abs(Z))
            # Replace -inf with 0 for visualization
            log_Z[log_Z == -np.inf] = 0

        if opts["x_space"]:
            xs = ["x0", "x1", "x2"]
        else:
            xs = ["kx", "ky", "kz"]
        xs[fixed_x] = str(opts["fixed_x"])

        if not opts["3D"]:
            # Create the heatmap using matplotlib
            plt.figure(figsize=(8, 6))
            plt.imshow(
                log_Z,
                origin="lower",
                extent=[
                    opts["range"][0],
                    opts["range"][1],
                    opts["range"][0],
                    opts["range"][1],
                ],  # type: ignore
                cmap="viridis",
            )  # type: ignore # nopep8
            plt.colorbar(label=f"log10(I({','.join(xs)}))")
        else:
            # Create a 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            # Plot the surface
            surf = ax.plot_surface(X, Y, Z, cmap="viridis")  # type: ignore # nopep8
            # Add a color bar which maps values to colors
            fig.colorbar(surf, shrink=0.5, aspect=5)
            ax.set_zlabel(f"log10(I({','.join(xs)}))")  # type: ignore # nopep8

        plt.xlabel(f"{xs[opts['xs'][0]]}")
        plt.ylabel(f"{xs[opts['xs'][1]]}")
        plt.title(f"log10(I({','.join(xs)}))")
        plt.show()


def _plot_worker_init(base: "DY", config: dict[str, Any]) -> None:
    proc = multiprocessing.current_process()
    proc._plot_worker = copy.deepcopy(base)  # type: ignore
    proc._plot_config = config  # type: ignore


def _plot_worker(task: tuple[int, int, float, float]) -> tuple[int, int, float]:
    proc = multiprocessing.current_process()
    worker = getattr(proc, "_plot_worker", None)
    config = getattr(proc, "_plot_config", None)
    if worker is None or config is None:
        raise pygloopException("Plot worker is not initialized.")
    i, j, x_val, y_val = task
    xs = [0.0, 0.0, 0.0]
    xs[config["fixed_x"]] = config["fixed_value"]
    xs[config["xs0"]] = x_val
    xs[config["xs1"]] = y_val
    if config["x_space"]:
        val = worker.integrand_xspace(
            xs,
            config["parameterisation"],
            config["integrand_implementation"],
            config["phase"],
            config["multi_channeling"],
        )
    else:
        wgt = worker.integrand(
            [Vector(xs[0], xs[1], xs[2])], config["integrand_implementation"]
        )
        match config["phase"]:
            case "real":
                val = wgt.real
            case "imag":
                val = wgt.imag
            case _:
                val = abs(wgt)
    return i, j, val
