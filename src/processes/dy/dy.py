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
    DYCompiledBundle,
    compile_integrands,
    evaluate_integrand,
)
from processes.dy.dy_graph_utils import _strip_quotes
from processes.dy.dy_infrared_test import (
    approach_point,
    # evaluate_integrand,
    infrared_test,
    ultraviolet_test,
)
from processes.dy.dy_integrand import (
    EMRIntegrandConstructor,
    LoopIntegrandConstructor,
    routed_cut_graph,
)
from processes.dy.dy_stability import (
    build_high_precision_sample,
    decimal_from_input,
    decimal_values_agree,
    exact_rotation_from_xs,
    float_values_agree,
    rotate_vector,
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

pjoin = os.path.join

TOLERANCE: float = 1e-10


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
                    dy_check_generation_limits=task["dy_check_generation_limits"],
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
        external_gluon_polarisation: bool = False,
        disable_integrated_uv_cts: bool = True,
        dy_check_generation_limits: bool = False,
        dy_parallel_graphs: int = 1,
        dy_fallback_precision: int | None = None,
        dy_lambda_sq: float | None = None,
        dy_mur_sq: float | None = None,
        dy_observable_muv: float | None = None,
        skip_gl_worker_init: bool = False,
        load_compiled_bundle: bool = True,
        clean=True,
        logger_level: int | None = None,
        symmetrise_p1_p2: bool = False,
        **opts,
    ):
        start_logger_level = logger.getEffectiveLevel()
        if logger_level is not None:
            logger.setLevel(logger_level)

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
        self.process_name = process_name if process_name is not None else "DY"
        self.diagrams = copy.deepcopy(diagrams) if diagrams is not None else None
        self.dy_channel = (
            tuple(int(parton) for parton in dy_channel)
            if dy_channel is not None
            else (1, -1)
        )
        if self.dy_channel not in {(0, 0), (0, 1), (1, 0), (1, -1), (-1, 1)}:
            raise ValueError(
                "Unsupported DY two-loop channel "
                f"{self.dy_channel}; supported channels are (0,0), (0,1), "
                "(1,0), (1,-1), and (-1,1)."
            )
        self.integrate_beams = bool(integrate_beams)
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
        self.dy_check_generation_limits = bool(dy_check_generation_limits)
        self.dy_parallel_graphs = max(1, int(dy_parallel_graphs))
        self.symmetrise_p1_p2 = bool(symmetrise_p1_p2)
        self.dy_graph_index_offset = 0
        self.dy_emr_state_name = None
        self.dy_lambda_sq = float(dy_lambda_sq) if dy_lambda_sq is not None else None
        self.dy_mur_sq = float(dy_mur_sq) if dy_mur_sq is not None else None
        self.dy_observable_muv = (
            float(dy_observable_muv) if dy_observable_muv is not None else None
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
        self.stability_hp_disagreement_count: int = 0
        self.stability_hp_nonfinite_count: int = 0
        self.stability_hp_error_count: int = 0
        self.stability_hp_failure_example: list[float] | None = None
        self.stability_hp_failure_example_momentum_point: str | None = None
        self.stability_hp_failure_reason: str | None = None
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
            dy_check_generation_limits=self.dy_check_generation_limits,
            dy_parallel_graphs=self.dy_parallel_graphs,
            symmetrise_p1_p2=self.symmetrise_p1_p2,
            dy_fallback_precision=self.dy_fallback_precision,
            dy_lambda_sq=self.dy_lambda_sq,
            dy_mur_sq=self.dy_mur_sq,
            dy_observable_muv=self.dy_observable_muv,
            skip_gl_worker_init=self.skip_gl_worker_init,
            load_compiled_bundle=self.load_compiled_bundle,
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
            "external_gluon_polarisation": self.external_gluon_polarisation,
            "disable_integrated_uv_cts": self.disable_integrated_uv_cts,
            "dy_check_generation_limits": self.dy_check_generation_limits,
            "dy_parallel_graphs": self.dy_parallel_graphs,
            "symmetrise_p1_p2": self.symmetrise_p1_p2,
            "dy_fallback_precision": self.dy_fallback_precision,
            "dy_lambda_sq": self.dy_lambda_sq,
            "dy_mur_sq": self.dy_mur_sq,
            "dy_observable_muv": self.dy_observable_muv,
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
        if selectors is None:
            return None

        channel_names = self.graph_channel_names(integrand_implementation)
        if len(channel_names) == 0:
            raise pygloopException(
                "DY integration graph selection requires a compiled zenos "
                "bundle with graph-grouped evaluators."
            )

        aliases = {name: index for index, name in enumerate(channel_names)}
        if self.diagrams is not None:
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
                if self.diagrams is not None and len(self.diagrams) == len(channel_names):
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

    @staticmethod
    def _build_symbolica_discrete_integrator(
        n_dim: int, n_channels: int
    ) -> NumericalIntegrator:
        return NumericalIntegrator.discrete([
            NumericalIntegrator.continuous(n_dim) for _ in range(n_channels)
        ])

    @staticmethod
    def _symbolica_sample_weight(sample: Sample) -> float:
        total_weight = 1.0
        for sample_weight in sample.weights:
            total_weight *= float(sample_weight)
        return total_weight

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
            external_gluon_polarisation=self.external_gluon_polarisation,
            disable_integrated_uv_cts=self.disable_integrated_uv_cts,
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

            print("############################")
            print("Routed graphs: ", len(routed_graphs))
            print("############################")

            routed_integrands = []
            evaluators = []

            for routed_graph_index, gg in enumerate(routed_graphs):
                processed_graphs.append(gg[3])
                cut_graph = deepcopy(routed_cut_graph(gg[3], gg[0], gg[1], gg[2]))
                # print(cut_graph.graph.get_name())
                print(cut_graph.graph)
                term_integrands = loop_processor.get_integrand(deepcopy(cut_graph))

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
            )
            my_compiler.save_compiled_integrand()

        print("n routed:", len(processed_graphs))
        return processed_graphs

    def _require_gg_generation_symmetrisation(self) -> None:
        if (
            self.n_loops == 2
            and self.process_name == "tt~"
            and self.dy_channel == (0, 0)
            and not self.symmetrise_p1_p2
        ):
            raise pygloopException(
                "Two-loop gg graph generation requires "
                "symmetrise_p1_p2=True."
            )

    def process_2L_generated_graphs(self, graphs: DYDotGraphs) -> DYDotGraphs:
        self._require_gg_generation_symmetrisation()
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
                "dy_check_generation_limits": self.dy_check_generation_limits,
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

    def generate_graphs(self) -> None:
        self._require_gg_generation_symmetrisation()
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

                if self.process_name == "dy":
                    self.gl_worker.run(
                        f"generate xs d d~ > a | d d~ g a QED^2==2 [{{{{1}}}} QCD=1] --only-diagrams --numerator-grouping only_detect_zeroes -p {base_name} -i {graphs_process_name} --max-multiplicity-for-fast-cut-filter 99"
                    )

                    self.gl_worker.run(
                        f"generate xs d g > a | d d~ g a QED^2==2 [{{{{1}}}} QCD=1] --only-diagrams --numerator-grouping only_detect_zeroes -p {base_name} -i {graphs_process_name} --max-multiplicity-for-fast-cut-filter 99"
                    )
                if self.process_name == "tt~":
                    # self.gl_worker.run(
                    #        f"generate xs d d~ > t t~ | d d~ g t t~ [{{{{1}}}} QCD=1] --only-diagrams --numerator-grouping group_identical_graphs_up_to_scalar_rescaling --symmetrize-left-right-states true --symmetrize-initial-states true -p {base_name} -i {graphs_process_name} --max-multiplicity-for-fast-cut-filter 99"
                    # )
                    self.gl_worker.run(
                        f"generate xs g g > t t~ | d d~ g t t~ [{{{{1}}}} QCD=1] --only-diagrams --numerator-grouping group_identical_graphs_up_to_scalar_rescaling --symmetrize-left-right-states true --symmetrize-initial-states true -p {base_name} -i {graphs_process_name} --max-multiplicity-for-fast-cut-filter 99"
                    )
                    # self.gl_worker.run(
                    #    f"generate xs ghG ghG~ > t t~ | d d~ g t t~ ghG ghG~ [{{{{1}}}} QCD=1] --only-diagrams --numerator-grouping group_identical_graphs_up_to_scalar_rescaling --symmetrize-left-right-states true --symmetrize-initial-states true -p {base_name} -i {graphs_process_name} --max-multiplicity-for-fast-cut-filter 99"
                    # )

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
                if self.process_name == "tt~":
                    select_graphs = (
                        f" --select-graphs {' '.join(self.diagrams)}"
                        if self.diagrams
                        else ""
                    )
                    initial_state_by_channel = {
                        (0, 0): "g g",
                        (0, 1): "d g",
                        (1, 0): "d g",
                        (1, -1): "d d~",
                        (-1, 1): "d d~",
                    }
                    numerator_grouping = (
                        " --numerator-grouping group_identical_graphs_up_to_scalar_rescaling"
                        if self.dy_channel in {(0, 0), (1, -1), (-1, 1)}
                        else ""
                    )
                    # self.gl_worker.run(  # GL06 GL14  --select-graphs GL00 GL01 GL03 GL04 GL05 GL08 GL12
                    #    f"generate xs d g > t t~ | d d~ g t t~ ghG ghG~ [{{{{2}}}} QCD=1] --only-diagrams --symmetrize-left-right-states true --symmetrize-initial-states true{select_graphs} -p {base_name} -i {graphs_process_name} --max-multiplicity-for-fast-cut-filter 99"
                    # )
                    self.gl_worker.run(  # GL06 GL14  --select-graphs GL14
                        f"generate xs {initial_state_by_channel[self.dy_channel]} > t t~ | d d~ g t t~ ghG ghG~ [{{{{2}}}} QCD=1] --only-diagrams{numerator_grouping} --symmetrize-left-right-states true --symmetrize-initial-states true{select_graphs} -p {base_name} -i {graphs_process_name} --max-multiplicity-for-fast-cut-filter 99"
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
        match parameterisation:
            case "cartesian":
                return self.cartesian_parameterize(xs, origin)
            case "spherical":
                return self.spherical_parameterize(xs, origin)
            case "log_spherical":
                return self.log_spherical_parameterize(xs, origin)
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

    def ttbar_beam_threshold_passes(self, x1: float, x2: float) -> bool:
        if not self.enforce_ttbar_beam_threshold:
            return True
        mt = 173.0
        return float(x1) * float(x2) * (self.e_cm**2) >= 4.0 * (mt**2)

    @staticmethod
    def _rotation_matrix_from_xs(xs: list[float]):
        return exact_rotation_from_xs(xs)

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

    def _evaluate_stability_hp_pair(
        self,
        xs: list[float],
        parameterization: str,
        integrand_implementation: dict[str, Any],
        phase: str,
        channel_selector: int | None,
        expects_z: bool,
        expects_beam_fractions: bool,
        rotation,
        decimal_digit_precision: int,
        relative_tolerance: float,
        absolute_tolerance: float,
    ) -> tuple[Decimal | None, Decimal | None, str | None, str | None]:
        if self.compiled_bundle is None:
            raise pygloopException(
                "Higher-precision DY validation requires a compiled zenos bundle."
            )
        if decimal_digit_precision < 2:
            raise pygloopException(
                "Higher-precision DY validation requires at least two digits."
            )
        self.compiled_bundle.require_fallback_supported(decimal_digit_precision)
        try:
            sample = build_high_precision_sample(
                xs,
                n_loops=self.n_loops,
                parameterization=parameterization,
                incoming_momenta=(self.ps_point[0], self.ps_point[1]),
                expects_z=expects_z,
                expects_beam_fractions=expects_beam_fractions,
                rescaling=RESCALING,
                decimal_digit_precision=decimal_digit_precision,
            )
            rotated_sample = sample.rotated(rotation)
            hp_impl = dict(integrand_implementation)
            hp_impl["dy_evaluation_mode"] = "arb"
            hp_impl["dy_fallback_precision"] = decimal_digit_precision
            hp_impl["dy_rotation_check_arb_digits"] = decimal_digit_precision
            hp_impl["z"] = sample.z
            hp_impl["mUV"] = decimal_from_input(hp_impl.get("mUV", 1.0))

            first_total, _first_terms = self._zenos_arb_terms_with_externals(
                list(sample.loop_momenta),
                sample.p1,
                sample.p2,
                hp_impl,
                decimal_digit_precision,
                channel_selector=channel_selector,
            )
            rotated_total, _rotated_terms = self._zenos_arb_terms_with_externals(
                list(rotated_sample.loop_momenta),
                rotated_sample.p1,
                rotated_sample.p2,
                hp_impl,
                decimal_digit_precision,
                channel_selector=channel_selector,
            )

            with localcontext() as context:
                context.prec = decimal_digit_precision + 12
                if phase == "real":
                    first_weight = first_total * sample.jacobian
                    rotated_weight = rotated_total * rotated_sample.jacobian
                elif phase == "imag":
                    first_weight = Decimal(0)
                    rotated_weight = Decimal(0)
                else:
                    raise pygloopException(f"Unsupported integration phase {phase!r}.")

                if not first_weight.is_finite() or not rotated_weight.is_finite():
                    return (
                        None,
                        None,
                        "nonfinite",
                        "higher-precision pair is non-finite",
                    )
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
                        "higher-precision original/rotated pair disagrees",
                    )
                return +first_weight, +first_total, None, None
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
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
    ) -> float:
        relative_tolerance, absolute_tolerance = self._stability_tolerances(
            integrand_implementation, rotation_digits
        )
        rotation = self._rotation_matrix_from_xs(xs)
        float_impl = dict(integrand_implementation)
        float_impl["dy_evaluation_mode"] = "compiled"

        first_weight: complex | None = None
        first_final = math.nan
        float_error: str | None = None
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
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            float_error = f"{type(exc).__name__}: {exc}"

        fallback_reason: str | None = None
        large_trigger = False
        if float_error is not None or not math.isfinite(first_final):
            fallback_reason = "float_nonfinite"
        else:
            trigger_thresholds = [
                threshold
                for threshold in (precision_threshold, clip_threshold)
                if threshold is not None
            ]
            large_trigger = any(
                abs(first_final) > threshold for threshold in trigger_thresholds
            )
            if large_trigger:
                # Deliberately do not spend time on a float rotation here.
                fallback_reason = "large_weight"

        rotated_final = math.nan
        float_relative_difference = math.inf
        if fallback_reason is None:
            rotated_momenta = [
                self._rotate_vec(momentum, rotation) for momentum in loop_momenta
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
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                float_error = f"{type(exc).__name__}: {exc}"

            if not math.isfinite(rotated_final):
                fallback_reason = "float_nonfinite"
                self.nan_weight_rotated_count += 1
                if self.nan_weight_rotated_example is None:
                    self.nan_weight_rotated_example = list(xs)
                    self.nan_weight_rotated_example_momentum_point = momentum_point
            elif not float_values_agree(
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
            else:
                self.stability_float_pair_accepted_count += 1

        accepted_final: float | Decimal
        accepted_unweighted: float | Decimal
        if fallback_reason is None:
            accepted_final = first_final
            accepted_unweighted = (
                self._phase_value(first_weight, phase)
                if first_weight is not None
                else 0.0
            )
        else:
            self.stability_hp_retry_count += 1
            if fallback_reason == "large_weight":
                self.large_weight_hp_retry_count += 1
                if self.large_weight_retry_example is None:
                    self.large_weight_retry_example = list(xs)
                    self.large_weight_retry_example_momentum_point = momentum_point
                    self.large_weight_retry_example_compiled_wgt = first_final
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

            precision = self._dy_fallback_precision(integrand_implementation)
            (
                hp_final,
                hp_unweighted,
                hp_failure_kind,
                hp_failure_reason,
            ) = self._evaluate_stability_hp_pair(
                xs,
                parameterization,
                integrand_implementation,
                phase,
                channel_selector,
                expects_z,
                expects_beam_fractions,
                rotation,
                precision,
                relative_tolerance,
                absolute_tolerance,
            )
            if (
                hp_failure_kind is None
                and hp_final is not None
                and hp_unweighted is not None
            ):
                self.stability_hp_accepted_count += 1
                accepted_final = hp_final
                accepted_unweighted = hp_unweighted if phase == "real" else Decimal(0)
                if fallback_reason == "large_weight":
                    self.large_weight_hp_salvaged_count += 1
                    if self.large_weight_retry_example_arb_wgt is None:
                        self.large_weight_retry_example_arb_wgt = float(hp_final)
                else:
                    self.rotation_hp_salvaged_count += 1
            else:
                accepted_final = Decimal(0)
                accepted_unweighted = Decimal(0)
                reason = hp_failure_reason or "unknown higher-precision failure"
                if hp_failure_kind == "disagreement":
                    self.stability_hp_disagreement_count += 1
                elif hp_failure_kind == "nonfinite":
                    self.stability_hp_nonfinite_count += 1
                else:
                    self.stability_hp_error_count += 1
                self._record_hp_failure(xs, momentum_point, reason)
                if fallback_reason == "large_weight":
                    self.large_weight_unstable_count += 1
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
            expects_z = self.sampled_uses_z(impl)
            expects_beam_fractions = self.sampled_uses_beam_fractions(impl)
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
            if expects_beam_fractions:
                beam_offset = n_k_vars + int(expects_z)
                x1 = xs[beam_offset]
                x2 = xs[beam_offset + 1]
                if not self.ttbar_beam_threshold_passes(x1, x2):
                    return 0.0
                p1, p2 = self.sampled_beam_momenta(x1, x2)

            loop_momenta = []
            jac_k = 1.0
            for i_loop in range(self.n_loops):
                k_loop, jac_loop = self.parameterize(
                    k_xs[3 * i_loop : 3 * (i_loop + 1)], parameterization
                )
                loop_momenta.append(k_loop)
                jac_k *= jac_loop

            total_jacobian = jac_k * jac_z * k_rescaling
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

            is_zenos = impl.get("integrand_type") == "zenos"
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
            stability_requested = is_zenos and (
                rotation_check_enabled
                or precision_threshold is not None
                or clip_threshold is not None
            )
            if stability_requested:
                return self._evaluate_zenos_stability_sample(
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
                )

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

            # print("res")
            # print(xs)
            # print(final_wgt)

        except ZeroDivisionError:
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
        m_uv: float | Decimal = 1.0
        if integrand_implementation is not None:
            m_uv = integrand_implementation.get("mUV", m_uv)
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

        return self.compiled_bundle.evaluate(
            loop_momentum,
            p1,
            p2,
            z,
            m_uv,
            **evaluate_kwargs,
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
        m_uv: float | Decimal = 1.0
        if integrand_implementation is not None:
            m_uv = integrand_implementation.get("mUV", m_uv)
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

        return self.compiled_bundle.evaluate_arb_terms(
            loop_momentum,
            p1,
            p2,
            z,
            m_uv,
            **evaluate_kwargs,
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
        value = integrand_implementation.get("dy_fallback_precision")
        if value is None:
            value = integrand_implementation.get("dy_rotation_check_arb_digits")
        if value is None:
            value = self.dy_fallback_precision
        return int(value)

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
        for attribute in (
            "large_weight_zeroed_signed_sum",
            "large_weight_zeroed_abs_sum",
            "stability_float_pair_accepted_count",
            "stability_float_mismatch_retry_count",
            "stability_float_nonfinite_retry_count",
            "stability_hp_retry_count",
            "stability_hp_accepted_count",
            "stability_hp_disagreement_count",
            "stability_hp_nonfinite_count",
            "stability_hp_error_count",
            "stability_hp_failure_example",
            "stability_hp_failure_example_momentum_point",
            "stability_hp_failure_reason",
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
                self.compiled_bundle.require_fallback_supported(fallback_precision)
        match integrator:
            case "naive":
                return self.naive_integrator(
                    parameterisation,
                    integrand_implementation,
                    target,
                    **opts,
                )
            case "vegas":
                return DY.vegas_integrator(
                    self,
                    parameterisation,
                    integrand_implementation,
                    target,
                    **opts,
                )
            case "symbolica":
                return self.symbolica_integrator(
                    parameterisation,
                    integrand_implementation,
                    target,
                    **opts,
                )
            case "gammaloop":
                return self.gammaloop_integrator(target, **opts)
            case _:
                raise pygloopException(f"Integrator {integrator} not implemented.")

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
                all_graph_channel_names[index] for index in graph_channel_indices
            ]
            logger.info(
                "Symbolica discrete graph channels: %s (bundle indices: %s)",
                ", ".join(graph_channel_names),
                ", ".join(str(index) for index in graph_channel_indices),
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
