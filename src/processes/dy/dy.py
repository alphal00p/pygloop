from __future__ import annotations

import copy
import json
import logging
import math
import multiprocessing
import os
import random
import shutil
import time
from copy import deepcopy
from decimal import Decimal
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
        skip_ps_validation: bool = False,
        integrate_beams: bool = False,
        disable_integrated_uv_cts: bool = True,
        skip_gl_worker_init: bool = False,
        load_compiled_bundle: bool = True,
        clean=True,
        logger_level: int | None = None,
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
        self.integrate_beams = bool(integrate_beams)
        self.enforce_ttbar_beam_threshold = (
            self.integrate_beams and self.process_name.lower() == "tt~"
        )
        self.skip_gl_worker_init = bool(skip_gl_worker_init)
        self.load_compiled_bundle = bool(load_compiled_bundle)
        self.disable_integrated_uv_cts = bool(disable_integrated_uv_cts)

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
        self.large_weight_retry_example: list[float] | None = None
        self.large_weight_retry_example_momentum_point: str | None = None
        self.large_weight_retry_example_compiled_wgt: float | None = None
        self.large_weight_retry_example_arb_wgt: float | None = None
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
            clean=False,
            logger_level=logging.CRITICAL,
            skip_ps_validation=self.skip_ps_validation,
            integrate_beams=self.integrate_beams,
            skip_gl_worker_init=self.skip_gl_worker_init,
            load_compiled_bundle=self.load_compiled_bundle,
        )
        return copied_self

    def builder_inputs(self) -> tuple:
        return (
            self.m_top,
            self.m_higgs,
            self.ps_point,
            self.helicities,
            self.n_loops,
            self.toml_config_path,
            self.runtime_toml_config_path,
            copy.deepcopy(self.final_state),
            self.process_name,
            self.skip_ps_validation,
            self.integrate_beams,
        )

    def process_uses_z(self) -> bool:
        return self.process_name.lower() == "dy"

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

        processor = EMRIntegrandConstructor([], process_name, n_loops)
        loop_processor = LoopIntegrandConstructor(
            [],
            process_name,
            n_loops,
            disable_integrated_uv_cts=self.disable_integrated_uv_cts,
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

                routed_integrands.extend(deepcopy(term_integrands))

                observable_params = {
                    "zmin": 0.0,
                    "zmax": 1.00000,
                    "Lambdasq": 2,
                    "mUV": 1,
                    "mursq": 1,
                }

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
                    evaluators.append(evaluator)

            all_routed_integrands.extend(routed_integrands)
            all_evaluators.extend(evaluators)

        if all_routed_integrands:
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
            )
            my_compiler.save_compiled_integrand()

        print("n routed:", len(processed_graphs))
        return processed_graphs

    def process_2L_generated_graphs(self, graphs: DYDotGraphs) -> DYDotGraphs:
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

        channel = (1, 0)  # (1, -1)
        channel = (1, -1)

        processor = EMRIntegrandConstructor([], process_name, n_loops)
        loop_processor = LoopIntegrandConstructor(
            [],
            process_name,
            n_loops,
            channel=channel,
            disable_integrated_uv_cts=self.disable_integrated_uv_cts,
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
                # if (len(gg[2][0]) == 1 and len(gg[2][1]) == 1):
                #    continue

                # if len(gg[2][1]) != 2:
                #    continue

                # if len(gg[2][1]) != 1 or len(gg[2][0]) != 1:
                #    continue
                #
                particle_channel = _strip_quotes(str(gg[3].get("particle_channel")))
                if particle_channel != str(channel):
                    continue

                processed_graphs.append(gg[3])
                cut_graph = deepcopy(routed_cut_graph(gg[3], gg[0], gg[1], gg[2]))
                # print(cut_graph.graph.get_name())
                # print(cut_graph.graph)
                term_integrands = loop_processor.get_integrand(deepcopy(cut_graph))

                routed_integrands.extend(deepcopy(term_integrands))

                observable_params = {
                    "zmin": 0.0,
                    "zmax": 1.00000,
                    "Lambdasq": 100000,
                    "mUV": 1000,
                    "mursq": 1,
                }

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
                    evaluators.append(evaluator)
                print("constructed evaluators")

            all_routed_integrands.extend(routed_integrands)
            all_evaluators.extend(evaluators)

            print("added up evaluators")

        if all_routed_integrands:
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
                * 0 * np.array([
                    1.0/math.sqrt(2),
                    1.0/math.sqrt(2),
                    0.0,
                ]),
                scale*np.array([1 / math.sqrt(3), -1 / math.sqrt(3), 1 / math.sqrt(3)]),
            ]
            #ks=[[2.0543179648600841e+08, -1.8748053733626541e+08, 1.0307223303487062e+08], [-2.6820491673003684e+01, 1.2449677258220136e+02, 1.1282568232590195e+02]]
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
            # scale = 1000
            # ks = [
            #    # math.sqrt(z)
            #    scale
            #    * np.array([
            #        0.0,
            #        0.0,
            #        -1 / math.sqrt(5),
            #    ]),
            #    scale * np.array([1 / math.sqrt(3), -1 / math.sqrt(3), 0]),
            # ]
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
            vp =1*np.array([1/10, 1/10, 1 / 2])
            p1 = scale * np.array([0, 0, 1])
            p2 = scale * np.array([0, 0, -1])
            print("just about to approach limit")
            approach_limit.approach(ks, p1, p2, z, vp)
        #
            #uv_test = ultraviolet_test(n_loops, process_name, all_routed_integrands)
            #uv_test.approach_limits(2000)

        if all_evaluators:
            my_compiler = compile_integrands(
                n_loops,
                process_name,
                self.get_integrand_name(),
                "z",
                all_evaluators,
            )
            my_compiler.save_compiled_integrand()

        print("############################")
        print("Processed graphs: ", len(routed_graphs))
        print("############################")

        return processed_graphs

    def generate_graphs(self) -> None:
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

                # --select-graphs GL02
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
                    # self.gl_worker.run(  # GL06 GL14  --select-graphs GL00 GL01 GL03 GL04 GL05 GL08 GL12
                    #    f"generate xs d g > t t~ | d d~ g t t~ ghG ghG~ [{{{{2}}}} QCD=1] --only-diagrams --numerator-grouping group_identical_graphs_up_to_scalar_rescaling --symmetrize-left-right-states true --symmetrize-initial-states true --select-graphs GL00 GL01 GL03 GL04 GL05 GL08 GL12   -p {base_name} -i {graphs_process_name} --max-multiplicity-for-fast-cut-filter 99"
                    # )
                    self.gl_worker.run(  # GL06 GL14  --select-graphs GL14
                        f"generate xs d d~ > t t~ | d d~ g t t~ ghG ghG~ [{{{{2}}}} QCD=1] --only-diagrams --numerator-grouping group_identical_graphs_up_to_scalar_rescaling --symmetrize-left-right-states true --symmetrize-initial-states true --select-graphs GL09 -p {base_name} -i {graphs_process_name} --max-multiplicity-for-fast-cut-filter 99"
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
    def _rotation_matrix_from_xs(xs: list[float]) -> tuple[tuple[float, ...], ...]:
        # Deterministic SO(3) rotation from sample coordinates.
        x0 = xs[0] if len(xs) > 0 else 0.123456789
        x1 = xs[1] if len(xs) > 1 else 0.234567891
        x2 = xs[2] if len(xs) > 2 else 0.345678912
        x3 = xs[3] if len(xs) > 3 else 0.456789123

        ux = math.sin(2.0 * math.pi * x0)
        uy = math.cos(2.0 * math.pi * x1)
        uz = math.sin(2.0 * math.pi * x2 + 0.5)
        norm = math.sqrt(ux * ux + uy * uy + uz * uz)
        if norm <= 1e-16:
            return (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        ux /= norm
        uy /= norm
        uz /= norm

        theta = 2.0 * math.pi * x3
        c = math.cos(theta)
        s = math.sin(theta)
        one_c = 1.0 - c

        return (
            (
                c + ux * ux * one_c,
                ux * uy * one_c - uz * s,
                ux * uz * one_c + uy * s,
            ),
            (
                uy * ux * one_c + uz * s,
                c + uy * uy * one_c,
                uy * uz * one_c - ux * s,
            ),
            (
                uz * ux * one_c - uy * s,
                uz * uy * one_c + ux * s,
                c + uz * uz * one_c,
            ),
        )

    @staticmethod
    def _rotate_vec(v: Vector, rmat: tuple[tuple[float, ...], ...]) -> Vector:
        x, y, z = v.to_list()
        return Vector(
            rmat[0][0] * x + rmat[0][1] * y + rmat[0][2] * z,
            rmat[1][0] * x + rmat[1][1] * y + rmat[1][2] * z,
            rmat[2][0] * x + rmat[2][1] * y + rmat[2][2] * z,
        )

    # dy.py: replace integrand_xspace(...) with this version

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
            wgt_in_arb = str(impl.get("dy_evaluation_mode", "compiled")) == "arb"
            is_zenos = impl.get("integrand_type") == "zenos"

            n_digits = impl.get("dy_rotation_check_digits")
            if is_zenos and n_digits is not None:
                n_digits_int = int(n_digits)
                if n_digits_int > 0:
                    eps = float(impl.get("dy_rotation_check_eps", 1e-15))
                    rmat = self._rotation_matrix_from_xs(xs)
                    rk = [self._rotate_vec(k_loop, rmat) for k_loop in loop_momenta]
                    rp1 = self._rotate_vec(p1, rmat)
                    rp2 = self._rotate_vec(p2, rmat)
                    wgt_rot = self.zenos_integrand_with_externals(
                        rk, rp1, rp2, impl, channel_selector=channel_selector
                    )
                    rel = abs(wgt - wgt_rot) / (abs(wgt) + abs(wgt_rot) + abs(eps))
                    if rel > 10.0 ** (-n_digits_int):
                        self.rotation_hp_retry_count += 1
                        if self.rotation_hp_retry_example is None:
                            self.rotation_hp_retry_example = list(xs)
                            self.rotation_hp_retry_example_momentum_point = (
                                momentum_point
                            )
                            self.rotation_hp_retry_example_rel = rel
                        arb_digits = max(
                            int(impl.get("dy_rotation_check_arb_digits", 80)),
                            n_digits_int + 20,
                        )
                        arb_impl = dict(impl)
                        arb_impl["dy_evaluation_mode"] = "arb"
                        arb_impl["dy_rotation_check_arb_digits"] = arb_digits
                        assert self.compiled_bundle is not None
                        self.compiled_bundle.require_arb_supported()
                        try:
                            wgt_arb = self.zenos_integrand_with_externals(
                                loop_momenta,
                                p1,
                                p2,
                                arb_impl,
                                channel_selector=channel_selector,
                            )
                            if bool(impl.get("dy_accept_all_arb_retries", False)):
                                self.rotation_hp_salvaged_count += 1
                                wgt = wgt_arb
                                wgt_in_arb = True
                            else:
                                wgt_rot_arb = self.zenos_integrand_with_externals(
                                    rk,
                                    rp1,
                                    rp2,
                                    arb_impl,
                                    channel_selector=channel_selector,
                                )
                                rel_arb = abs(wgt_arb - wgt_rot_arb) / (
                                    abs(wgt_arb) + abs(wgt_rot_arb) + abs(eps)
                                )
                                if rel_arb <= 10.0 ** (-n_digits_int):
                                    self.rotation_hp_salvaged_count += 1
                                    wgt = wgt_arb
                                    wgt_in_arb = True
                                else:
                                    self.rotation_unstable_count += 1
                                    if self.rotation_unstable_example is None:
                                        self.rotation_unstable_example = list(xs)
                                        self.rotation_unstable_example_momentum_point = momentum_point
                                    wgt = 0.0 + 0.0j
                        except Exception:
                            self.rotation_unstable_count += 1
                            if self.rotation_unstable_example is None:
                                self.rotation_unstable_example = list(xs)
                                self.rotation_unstable_example_momentum_point = (
                                    momentum_point
                                )
                            wgt = 0.0 + 0.0j

            large_weight_threshold = impl.get("dy_large_weight_threshold")
            if (
                is_zenos
                and not wgt_in_arb
                and large_weight_threshold is not None
                and float(large_weight_threshold) > 0.0
            ):
                phase_wgt = wgt.real if phase == "real" else wgt.imag
                tentative_final_wgt = phase_wgt * total_jacobian
                if math.isfinite(tentative_final_wgt) and abs(
                    tentative_final_wgt
                ) > float(large_weight_threshold):
                    self.large_weight_hp_retry_count += 1
                    if self.large_weight_retry_example is None:
                        self.large_weight_retry_example = list(xs)
                        self.large_weight_retry_example_momentum_point = momentum_point
                        self.large_weight_retry_example_compiled_wgt = (
                            tentative_final_wgt
                        )

                    arb_impl = dict(impl)
                    arb_impl["dy_evaluation_mode"] = "arb"
                    arb_impl["dy_rotation_check_arb_digits"] = int(
                        impl.get("dy_rotation_check_arb_digits", 80)
                    )
                    assert self.compiled_bundle is not None
                    self.compiled_bundle.require_arb_supported()
                    try:
                        arb_digits = int(arb_impl["dy_rotation_check_arb_digits"])
                        arb_total, arb_terms = self.compiled_bundle.evaluate_arb_terms(
                            loop_momenta,
                            p1,
                            p2,
                            float(arb_impl.get("z", 1.0)),
                            float(arb_impl.get("mUV", 1.0)),
                            decimal_digit_precision=arb_digits,
                            theta_tolerance=float(arb_impl.get("dy_theta_tol", 0.0)),
                            channel_selector=channel_selector,
                        )
                        wgt_arb = complex(float(arb_total), 0.0)
                        phase_wgt_arb = (
                            wgt_arb.real if phase == "real" else wgt_arb.imag
                        )
                        final_wgt_arb = phase_wgt_arb * total_jacobian
                        cancellation_veto_rel = Decimal(
                            str(
                                impl.get(
                                    "dy_large_weight_cancellation_veto_rel",
                                    1.0e-12,
                                )
                            )
                        )
                        max_term = max(
                            (abs(term_value) for _name, term_value in arb_terms),
                            default=Decimal(0),
                        )
                        cancellation_veto = (
                            cancellation_veto_rel > 0
                            and max_term > 0
                            and abs(arb_total) / max_term < cancellation_veto_rel
                        )
                        if math.isfinite(final_wgt_arb) and not cancellation_veto:
                            self.large_weight_hp_salvaged_count += 1
                            if self.large_weight_retry_example_arb_wgt is None:
                                self.large_weight_retry_example_arb_wgt = final_wgt_arb
                            wgt = wgt_arb
                        else:
                            self.large_weight_unstable_count += 1
                            if (
                                math.isfinite(final_wgt_arb)
                                and self.large_weight_retry_example_arb_wgt is None
                            ):
                                self.large_weight_retry_example_arb_wgt = final_wgt_arb
                            wgt = 0.0 + 0.0j
                    except Exception:
                        self.large_weight_unstable_count += 1
                        wgt = 0.0 + 0.0j

            stable_wgt_abs = abs(wgt)
            if self.max_stable_wgt is None or stable_wgt_abs > self.max_stable_wgt:
                self.max_stable_wgt = stable_wgt_abs
                self.max_stable_wgt_point = list(xs)
                self.max_stable_wgt_jacobian = total_jacobian
                self.max_stable_wgt_momentum_point = momentum_point

            # t2 = time.perf_counter()

            # print("overall integrand time:", t2 - t1)

            wgt = wgt.real if phase == "real" else wgt.imag
            final_wgt = wgt * total_jacobian

            if self.max_wgt is None or abs(final_wgt) > abs(self.max_wgt):
                self.max_wgt = final_wgt
                self.max_wgt_point = list(xs)
                self.max_wgt_jacobian = total_jacobian
                self.max_wgt_momentum_point = momentum_point

            # print("res")
            # print(xs)
            # print(final_wgt)

            if math.isnan(final_wgt):
                logger.debug(
                    f"Integrand evaluated to NaN at xs = [{Colour.BLUE}{', '.join(f'{xi:+.16e}' for xi in xs)}{Colour.END}]. Setting it to zero"
                )
                final_wgt = 0.0

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

        z = 1.0
        m_uv = 1.0
        if integrand_implementation is not None:
            m_uv = float(integrand_implementation.get("mUV", m_uv))
            if self.process_uses_z():
                z = float(integrand_implementation.get("z", z))
        evaluation_mode = "compiled"
        decimal_digit_precision = None
        theta_tolerance = 0.0
        if integrand_implementation is not None:
            evaluation_mode = str(
                integrand_implementation.get("dy_evaluation_mode", evaluation_mode)
            )
            arb_digits = integrand_implementation.get("dy_rotation_check_arb_digits")
            if arb_digits is not None:
                decimal_digit_precision = int(arb_digits)
            theta_tol = integrand_implementation.get("dy_theta_tol")
            if theta_tol is not None:
                theta_tolerance = float(theta_tol)

        return self.compiled_bundle.evaluate(
            loop_momentum,
            p1,
            p2,
            z,
            m_uv,
            mode=evaluation_mode,
            decimal_digit_precision=decimal_digit_precision,
            theta_tolerance=theta_tolerance,
            channel_selector=channel_selector,
        )

    def _normalize_integrand_implementation(
        self, integrand_implementation: dict[str, Any] | str
    ) -> dict[str, Any]:
        if isinstance(integrand_implementation, str):
            return {"integrand_type": integrand_implementation}
        return integrand_implementation

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
        process_builder_inputs: tuple[Any],
        id: int,
        all_xs: list[list[float]],
        call_args: list[Any],
        skip_gl_worker_init: bool = False,
    ) -> tuple[int, list[float], IntegrationResult]:
        res = IntegrationResult(0.0, 0.0)
        t_start = time.time()
        all_weights = []
        process = DY(
            *process_builder_inputs,
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
        process_builder_inputs: tuple[Any],
        id: int,
        multi_channeling: bool,
        all_xs: list[SymbolicaSample],
        call_args: list[Any],
        skip_gl_worker_init: bool = False,
    ) -> tuple[int, list[float], IntegrationResult]:
        res = IntegrationResult(0.0, 0.0)
        t_start = time.time()
        all_weights = []
        process = DY(
            *process_builder_inputs,
            clean=False,
            logger_level=logging.CRITICAL,
            skip_gl_worker_init=skip_gl_worker_init,
        )  # type: ignore
        for xs in all_xs:
            if not multi_channeling:
                weight = process.integrand_xspace(xs.c, *( call_args + [False, ]))  # fmt: off
            else:
                weight = process.integrand_xspace(xs.c, *(call_args + [xs.d[0]]))
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

        return (id, all_weights, res)

    @staticmethod
    def symbolica_integrand_function(
        process: DY,
        res: IntegrationResult,
        n_cores: int,
        multi_channeling: bool,
        call_args: list[Any],
        samples: list[Sample],
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

        if opts["multi_channeling"]:
            graph_channel_names = self.graph_channel_names(integrand_implementation)
            if len(graph_channel_names) == 0:
                raise pygloopException(
                    "DY Symbolica multi-channeling requires a compiled zenos bundle "
                    "with graph-grouped evaluators."
                )
            logger.info(
                "Symbolica discrete graph channels: %s",
                ", ".join(graph_channel_names),
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
