from __future__ import annotations

import copy
import json
import logging
import math
import os
import shutil
import time
from pprint import pformat, pprint  # noqa: F401
from typing import Any

import progressbar

from gammaloop import GammaLoopAPI, LogLevel, evaluate_graph_overall_factor, git_version  # isort: skip # type: ignore # noqa: F401

from matplotlib.typing import CapStyleType, ColorType  # noqa: F401
from symbolica import (
    E,
    Evaluator,
    Expression,
    Replacement,
    S,
)

from symbolica.community.idenso import simplify_gamma, simplify_metrics, simplify_color, cook_indices  # isort: skip # noqa: F401
from symbolica.community.spenso import TensorLibrary, TensorNetwork
from ufo_model_loader.commands import Model

from utils.cff import CFFStructure, CFFTerm
from utils.naive_integrator import naive_integrator as run_naive_integrator
from utils.plotting import plot_integrand
from utils.polarizations import vxxxxx
from utils.symbolica_integrator import symbolica_integrator as run_symbolica_integrator
from utils.utils import (
    CONFIGS_FOLDER,  # noqa: F401
    DOTS_FOLDER,  # noqa: F401
    EVALUATORS_FOLDER,  # noqa: F401
    GAMMALOOP_STATES_FOLDER,  # noqa: F401
    INTEGRATION_WORKSPACE_FOLDER,  # noqa: F401
    OUTPUTS_FOLDER,  # noqa: F401
    PYGLOOP_FOLDER,
    RESOURCES_FOLDER,  # noqa: F401
    Colour,
    DotGraph,
    DotGraphs,
    IntegrationResult,
    ParamBuilder,
    PygloopEvaluator,
    expr_to_string,
    logger,
    pygloopException,
    set_gammaloop_level,
    set_tmp_logger_level,  # noqa: F401
    write_text_with_dirs,
)
from utils.vectors import LorentzVector, Vector
from utils.vegas_integrator import vegas_integrator as run_vegas_integrator

pjoin = os.path.join

TOLERANCE: float = 1e-10

RESCALING: float = 10.0


class GGHHH(object):
    name = "GGHHH"
    name_for_resources = "GGHHH"

    DEFAULT_COMPILATION_OPTIONS = {
        "inline_asm": "default",  # "none"
        "optimization_level": 3,
        "native": True,
    }
    VERBOSE_FULL_EVALUATOR = False
    COMPLEXIFY_EVALUATOR = False
    FREEZE_INPUT_PHASES = True
    ENABLE_CFF_TERM = True
    ENABLE_NUMERATOR_TERM = True

    DEBUG_FULL_EVALUATOR_PATH = None  # "/Users/vjhirsch/Documents/Work/pygloop/TMP/cff_evaluator_inputs.py"

    SB = {
        "etaSelector": S("pygloop::ηs"),
        "etaSigma": S("pygloop::ση"),
        "parametricEtaSigma": S("pygloop::param_ση"),
        "energySign": S("pygloop::σE"),
        "parametricEnergySign": S("pygloop::param_σE"),
        "Qspatial": S("pygloop::Qs"),
        "Kspatial": S("pygloop::Ks"),
        "Q": S("pygloop::Q"),
        "o_id": S("pygloop::o_id"),
        "f_id": S("pygloop::f_id"),
        "vector_pol": S("gammalooprs::ϵ"),
        "externalP": S("gammalooprs::P"),
        "ICFF": S("pygloop::ICFF"),
    }

    SPENSO_EVALUATOR_NAMES = ["parametric_integrand_evaluator", "full_integrand_evaluator", "input_parameters_evaluator"]

    def __init__(
        self,
        m_top: float,
        m_higgs: float,
        ps_point: list[LorentzVector],
        helicities: list[int] | None = None,
        n_loops: int = 1,
        toml_config_path: str | None = None,
        runtime_toml_config_path: str | None = None,
        clean=True,
        logger_level: int | None = None,
        gammaloop_settings: list[str] | None = None,
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

        self.valide_ps_point()

        self.e_cm = math.sqrt(abs((self.ps_point[0] + self.ps_point[1]).squared()))

        gl_states_folder = pjoin(GAMMALOOP_STATES_FOLDER, self.name)
        self.clean = clean
        if os.path.exists(gl_states_folder):
            if clean:
                logger.info(f"Removing existing GammaLoop state in {Colour.GREEN}{gl_states_folder}{Colour.END}")  # fmt: off
                shutil.rmtree(gl_states_folder)
            else:
                logger.info(f"Reusing existing GammaLoop state in {Colour.GREEN}{gl_states_folder}{Colour.END}")  # fmt: off

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
        )  # fmt: off
        self.gl_worker = GammaLoopAPI(
            pjoin(GAMMALOOP_STATES_FOLDER, self.name),
            log_file_name=self.name,
            log_level=gl_log_level,
        )
        self.set_log_level(logger_level)

        if toml_config_path is None:
            toml_config_path = pjoin(CONFIGS_FOLDER, self.name_for_resources, "generate.toml")

        self.toml_config_path = toml_config_path
        logger.info(f"Setting gammaloop starting configuration from toml file {Colour.BLUE}{toml_config_path}{Colour.END}.")
        self.gl_worker.run(f"set global file {toml_config_path}")  # fmt: off
        self.setup_gl_worker()

        amplitudes, cross_sections = self.gl_worker.list_outputs()
        if len(amplitudes) == 0 and len(cross_sections) == 0:
            logger.info("No output yet in the GammaLoop state loaded.")
        if len(amplitudes) > 0:
            logger.info(f"Available amplitudes: {Colour.GREEN}{pformat(amplitudes)}{Colour.END}")
        if len(cross_sections) > 0:
            logger.info(f"Available cross sections: {Colour.GREEN}{pformat(cross_sections)}{Colour.END}")

        if runtime_toml_config_path is None:
            runtime_toml_config_path = pjoin(CONFIGS_FOLDER, self.name_for_resources, "runtime.toml")
        self.runtime_toml_config_path = runtime_toml_config_path

        logger.info(f"Setting runtime configuration for all outputs from toml file: {Colour.BLUE}{runtime_toml_config_path}{Colour.END}.")  # fmt: off
        for output_name, output_id in amplitudes.items():
            # Currently bugged: not all functionalities available on integrands not yet generated
            if not os.path.isdir(
                pjoin(GAMMALOOP_STATES_FOLDER, self.name, "processes", "amplitudes", self.get_integrand_name(suffix=""), output_name, "integrand")
            ):
                continue
            self.gl_worker.run(f"set process -p {output_id} -i {output_name} file {self.runtime_toml_config_path}")  # fmt: off
            self.set_sample_point(self.ps_point, self.helicities, str(output_id), output_name)

        if gammaloop_settings is not None:
            for setting_command in gammaloop_settings:
                logger.info(f"Applying custom GammaLoop setting: {Colour.GREEN}{setting_command}{Colour.END}.")  # fmt: off
                self.gl_worker.run(setting_command)

        self.save_state()
        # Cache some quantities for performance
        self.cache: dict[str, Any] = {}

        self.spenso_evaluators: dict[str, dict[str, PygloopEvaluator] | None] = {}

        # Loading spenso integrand evaluators if there are any already generated
        for integrand_name in [ self.get_integrand_name(), ]:  # fmt: off
            integrand_evaluator_directory = pjoin(EVALUATORS_FOLDER, self.name, integrand_name)
            if self.clean and os.path.exists(integrand_evaluator_directory):
                logger.info(f"Removing existing evaluators in {Colour.GREEN}{integrand_evaluator_directory}{Colour.END}")  # fmt: off
                shutil.rmtree(integrand_evaluator_directory)
                self.spenso_evaluators[integrand_name] = None
                continue

            evaluators = {}
            at_least_one_evaluator_found = False
            integrand_param_builders = []
            parameters_param_builders = []
            for evaluator_name in GGHHH.SPENSO_EVALUATOR_NAMES:
                if os.path.exists(pjoin(integrand_evaluator_directory, f"{evaluator_name}.so")):
                    at_least_one_evaluator_found = True
                    evaluators[evaluator_name] = PygloopEvaluator.load(pjoin(EVALUATORS_FOLDER, self.name, integrand_name), evaluator_name)
                    if evaluator_name == "input_parameters_evaluator":
                        parameters_param_builders.append(evaluators[evaluator_name].param_builder)
                    else:
                        integrand_param_builders.append(evaluators[evaluator_name].param_builder)
                else:
                    evaluators[evaluator_name] = None
            if at_least_one_evaluator_found:
                self.spenso_evaluators[integrand_name] = evaluators
                self.initialize_param_builders(integrand_param_builders, parameters_param_builders[0])
            else:
                self.spenso_evaluators[integrand_name] = None

        logger.setLevel(start_logger_level)

    def __deepcopy__(self, _memo) -> GGHHH:
        copied_self = GGHHH(
            self.m_top,
            self.m_higgs,
            copy.deepcopy(self.ps_point, _memo),
            copy.deepcopy(self.helicities, _memo),
            self.n_loops,
            self.toml_config_path,
            self.runtime_toml_config_path,
            clean=False,
            logger_level=logging.CRITICAL,
        )
        return copied_self

    def get_model(self):
        try:
            return Model.from_json(self.gl_worker.get_model())
        except Exception as e:
            raise pygloopException(f"Could not get model from GammaLoop worker. Error: {e}") from e

    def builder_inputs(self) -> tuple:
        return (self.m_top, self.m_higgs, self.ps_point, self.helicities, self.n_loops, self.toml_config_path, self.runtime_toml_config_path)

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
        self.gl_worker.run(f"set global kv global.logfile_directive='gammalooprs={lvl},{lvl}'")
        self.gl_worker.run(f"set global kv global.display_directive='gammalooprs={lvl},{lvl}'")

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
        self.gl_worker.run(kinematics_set_command)

    def set_model(self) -> None:
        self.gl_worker.run("import model sm-default.json")
        self.gl_worker.run("set model MT={{re:{:.16f},im:0.0}}".format(self.m_top))
        self.gl_worker.run("set model MH={{re:{:.16f},im:0.0}}".format(self.m_higgs))
        self.gl_worker.run("set model WT={re:0.0,im:0.0}")
        self.gl_worker.run("set model WH={re:0.0,im:0.0}")
        self.gl_worker.run("set model ymt={{re:{:.16f},im:0.0}}".format(self.m_top))

    def setup_gl_worker(self) -> None:
        self.set_model()
        # Set default kinematics
        self.set_sample_point(self.ps_point, self.helicities, None, None)
        # print(dir(self.gl_worker))
        # self.gl_worker.run("save state -o")

    def save_state(self) -> None:
        self.gl_worker.run("save state -o")

    def get_color_projector(self) -> Expression:
        return E("(1/8)*spenso::g(spenso::coad(8,gammalooprs::hedge(0)),spenso::coad(8,gammalooprs::hedge(1)))")

    def process_1L_generated_graphs(self, graphs: DotGraphs) -> DotGraphs:
        processed_graphs = DotGraphs()
        for g_input in graphs:
            g: DotGraph = copy.deepcopy(g_input)
            attrs = g.get_attributes()
            attrs["num"] = f'"{expr_to_string(g.get_numerator())}"'
            attrs["projector"] = f'"{expr_to_string(g.get_projector() * self.get_color_projector())}"'
            # VHHACK: for testing purposes only
            # attrs["num"] = '"1"'
            # attrs["projector"] = '"1"'
            g.set_local_numerators_to_one()
            processed_graphs.append(g)

        return processed_graphs

    def process_3L_generated_graphs(self, graphs: DotGraphs) -> DotGraphs:
        processed_graphs = DotGraphs()
        for g_input in graphs:
            g = copy.deepcopy(g_input)
            attrs = g.get_attributes()
            attrs["num"] = f'"{expr_to_string(g.get_numerator())}"'
            attrs["projector"] = f'"{expr_to_string(g.get_projector() * self.get_color_projector())}"'
            g.set_local_numerators_to_one()
            processed_graphs.append(g)

        return processed_graphs

    def process_2L_generated_graphs(self, graphs: DotGraphs) -> DotGraphs:
        processed_graphs = DotGraphs()
        for g_input in graphs:
            g = copy.deepcopy(g_input)
            attrs = g.get_attributes()
            attrs["num"] = f'"{expr_to_string(g.get_numerator())}"'
            attrs["projector"] = f'"{expr_to_string(g.get_projector() * self.get_color_projector())}"'
            g.set_local_numerators_to_one()
            processed_graphs.append(g)

        return processed_graphs

    def generate_graphs(self) -> None:
        graphs_process_name = self.get_integrand_name(suffix="_generated_graphs")
        integrand_name = self.get_integrand_name()
        amplitudes, _cross_sections = self.gl_worker.list_outputs()
        base_name = self.get_integrand_name(suffix="")
        if graphs_process_name in amplitudes:
            logger.info(f"Graphs for amplitude {graphs_process_name} already generated and recycled.")
            return
        match self.n_loops:
            case 1:
                logger.info("Generating one-loop graphs ...")
                self.gl_worker.run(
                    f"generate amp g g > h h h / u d c s b QED==3 [{{1}}] --only-diagrams --numerator-grouping only_detect_zeroes --select-graphs GL15 --loop-momentum-bases GL15=8 -p {base_name} -i {graphs_process_name}"
                )
                self.gl_worker.run("save state -o")
                GGHHH_1L_dot_files = self.gl_worker.get_dot_files(process_id=None, integrand_name=graphs_process_name)
                write_text_with_dirs(
                    pjoin(DOTS_FOLDER, self.name, f"{graphs_process_name}.dot"),
                    GGHHH_1L_dot_files,
                )
                self.gl_worker.run("save dot")
                GGHHH_1L_dot_files_processed = self.process_1L_generated_graphs(DotGraphs(dot_str=GGHHH_1L_dot_files))
                GGHHH_1L_dot_files_processed.save_to_file(pjoin(DOTS_FOLDER, self.name, f"{integrand_name}.dot"))
            case 2:
                logger.info("Generating two-loop graphs ...")
                self.gl_worker.run(
                    f"generate amp g g > h h h | g h t t~ QED==3 [{{2}}] --only-diagrams --numerator-grouping only_detect_zeroes --veto-vertex-interactions V_6 V_9 V_36 V_37 --number-of-fermion-loops 1 1 --select-graphs GL303 -p {base_name} -i {graphs_process_name}"
                )
                self.gl_worker.run("save state -o")
                GGHHH_2L_dot_files = self.gl_worker.get_dot_files(process_id=None, integrand_name=graphs_process_name)
                write_text_with_dirs(
                    pjoin(DOTS_FOLDER, self.name, f"{graphs_process_name}.dot"),
                    GGHHH_2L_dot_files,
                )
                self.gl_worker.run("save dot")
                GGHHH_2L_dot_files_processed = self.process_2L_generated_graphs(DotGraphs(dot_str=GGHHH_2L_dot_files))
                GGHHH_2L_dot_files_processed.save_to_file(pjoin(DOTS_FOLDER, self.name, f"{integrand_name}.dot"))
            case 3:
                logger.info("Importing three-loop graphs ...")
                three_loop_input_dot_graphs = pjoin(RESOURCES_FOLDER, self.name_for_resources, "3L_graphs.dot")
                self.gl_worker.run(f"import graphs {three_loop_input_dot_graphs} -p {base_name} -i {graphs_process_name}")
                self.gl_worker.run("save state -o")
                GGHHH_3L_dot_files = self.gl_worker.get_dot_files(process_id=None, integrand_name=graphs_process_name)
                write_text_with_dirs(
                    pjoin(DOTS_FOLDER, self.name, f"{graphs_process_name}.dot"),
                    GGHHH_3L_dot_files,
                )
                self.gl_worker.run("save dot")
                GGHHH_3L_dot_files_processed = self.process_3L_generated_graphs(DotGraphs(dot_str=GGHHH_3L_dot_files))
                GGHHH_3L_dot_files_processed.save_to_file(pjoin(DOTS_FOLDER, self.name, f"{integrand_name}.dot"))
            case _:
                raise pygloopException(f"Number of loops {self.n_loops} not supported.")

        amplitudes, _cross_sections = self.gl_worker.list_outputs()
        if graphs_process_name not in amplitudes:
            raise pygloopException(f"Amplitude with named integrand {graphs_process_name} not found in GammaLoop state. Generate graphs first.")

        if not os.path.isfile(pjoin(DOTS_FOLDER, self.name, f"{integrand_name}.dot")):
            raise pygloopException(
                f"Processed dot file not found at {pjoin(DOTS_FOLDER, self.name, f'{integrand_name}.dot')}. Generate graphs first."
            )

        self.gl_worker.run(
            f"import graphs {pjoin(DOTS_FOLDER, self.name, f'{integrand_name}.dot')} -p {amplitudes[graphs_process_name]} -i {integrand_name}"
        )

        self.save_state()

    def get_model_parameters(self) -> list[tuple[Expression, complex]]:
        model = self.get_model()
        res = []
        for param in model.parameters:
            if param.value is None:
                raise pygloopException(f"Model parameter '{param.name}' has no value.")
            res.append((E(f"UFO::{param.name}"), param.value))
        return res

    def get_model_couplings(self) -> list[tuple[Expression, complex]]:
        model = self.get_model()
        res = []
        for coupl in model.couplings:
            if coupl.value is None:
                raise pygloopException(f"Model coupling '{coupl.name}' has no value.")
            res.append((E(f"UFO::{coupl.name}"), coupl.value))
        return res

    def get_constants_for_evaluator(self) -> dict[Expression, Expression]:
        # TODO
        return {
            E("spenso::TR"): E("1/2"),
        }

    def get_float_constants_for_evaluator(self) -> dict[Expression, complex]:
        return {Expression.PI: complex(math.pi, 0.0)}

    def initialize_param_builders(
        self,
        integrand_param_builders: list[ParamBuilder],
        input_params_param_builder: ParamBuilder,
    ):
        # Update model parameters
        model_inputs = self.get_model_parameters() + self.get_model_couplings()
        for param, value in model_inputs:
            for itg_pb in integrand_param_builders:
                itg_pb.set_parameter((param,), value)
            input_params_param_builder.set_parameter((param,), value)

        # Set external momenta
        for i_p, p in enumerate(self.ps_point):
            input_params_param_builder.set_parameter_values((self.SB["externalP"], E(str(i_p))), [complex(p_i) for p_i in p.to_list()])

        # Add the polarization vectors
        for e_i in [0, 1]:
            pol_vector = vxxxxx(self.ps_point[e_i].to_list(), 0.0, self.helicities[e_i], -1)[2::]
            for itg_pb in integrand_param_builders:
                itg_pb.set_parameter_values((self.SB["vector_pol"], E(str(e_i))), [complex(pol_i) for pol_i in pol_vector])

    def set_from_sample(
        self,
        integrand_evaluator: PygloopEvaluator,
        input_params_evaluator: PygloopEvaluator | None = None,
        ks: list[Vector] | None = None,
        cff_term: CFFTerm | None = None,
        family_id: int | None = None,
    ) -> None:
        integrand_param_builder = integrand_evaluator.param_builder

        # Set loop momenta and emr momenta
        if ks is not None:
            assert input_params_evaluator is not None
            for i_k, k in enumerate(ks):
                input_params_evaluator.param_builder.set_parameter_values((self.SB["Kspatial"], E(str(i_k))), [complex(k_i) for k_i in k.to_list()])

            # Evaluate derived inputs
            derived_inputs = input_params_evaluator.evaluate()
            integrand_param_builder.set_parameter_values_within_range(0, len(derived_inputs), derived_inputs)

        # Add the E-surfaces selectors
        if cff_term is not None and family_id is not None:
            integrand_param_builder.set_parameter_values((self.SB["etaSigma"], self.SB["o_id"], self.SB["f_id"]), cff_term.masks[family_id])

        # Add the energy signs
        if cff_term is not None:
            integrand_param_builder.set_parameter_values((self.SB["energySign"], self.SB["o_id"]), cff_term.orientation_signs)

    def build_parametric_integrand_evaluator(self, graph: DotGraph, integrand: Expression, cff_structure: CFFStructure) -> PygloopEvaluator:
        internal_edges = graph.get_internal_edges()
        max_internal_edge_id = 0 if len(internal_edges) == 0 else max(int(e.get("id")) for e in internal_edges)

        param_builder = ParamBuilder()

        # START of "computable parameters" listing
        # Add on-shell energies and external momenta energies
        param_builder.add_parameter_list((CFFStructure.SB["E"],), max_internal_edge_id + 1)
        # Add the spatial part of momenta
        for e_i in range(max_internal_edge_id + 1):
            param_builder.add_parameter_list((self.SB["Qspatial"], E(str(e_i))), 3)
        # END of "computable parameters" listing

        # Add the polarization vectors
        for e_i in [0, 1]:
            param_builder.add_parameter_list((self.SB["vector_pol"], E(str(e_i))), 4)
            param_builder.force_parameters_to_complex([(self.SB["vector_pol"], E(str(e_i)))])
        # Add the E-surfaces selectors
        param_builder.add_parameter_list((self.SB["etaSigma"], self.SB["o_id"], self.SB["f_id"]), len(cff_structure.e_surfaces))
        # Add the energy signs
        param_builder.add_parameter_list((self.SB["energySign"], self.SB["o_id"]), max_internal_edge_id + 1)

        model_inputs = self.get_model_parameters() + self.get_model_couplings()
        for param, value in model_inputs:
            param_builder.add_parameter((param,))
            param_builder.set_parameter((param,), value)
        for param, value in self.get_float_constants_for_evaluator().items():
            param_builder.add_parameter((param,))
            param_builder.set_parameter((param,), value)

        if GGHHH.FREEZE_INPUT_PHASES:
            param_builder.freeze_all_current_parameters_phase()
        # fmt: off
        evaluator = integrand.evaluator(
            constants = self.get_constants_for_evaluator(),
            functions = {}, # type: ignore
            params = param_builder.get_parameters(),
            iterations = 100,
            n_cores = 8,
            verbose = False,
            external_functions = None,
            conditionals = [self.SB["etaSelector"],]
        )

        pygloop_evaluator = PygloopEvaluator(evaluator, param_builder, "parametric_integrand_evaluator", additional_data={'cff_structure': copy.deepcopy(cff_structure)})
        if GGHHH.FREEZE_INPUT_PHASES and not GGHHH.COMPLEXIFY_EVALUATOR:
            pygloop_evaluator.freeze_input_phases()
        if GGHHH.COMPLEXIFY_EVALUATOR:
            pygloop_evaluator.complexify()
        return pygloop_evaluator

    def build_full_integrand_evaluator(
        self,
        integrand_expression: Expression,
        cff_structure: CFFStructure,
        integrand_param_builder: ParamBuilder,
        strategy: str = "merging",
        n_hornerscheme_iterations: int = 100,
        n_cpe_iterations: int | None = None,
    ) -> PygloopEvaluator:
        if strategy not in ["merging", "summing", "function_map"]:
            raise pygloopException(f"Evaluation combination strategy '{strategy}' not supported.")
        total_start = time.time()

        selector_execution = [
            Replacement(self.SB["etaSelector"](E("1"), E("true_"), E("false_")), E("true_")),
            Replacement(self.SB["etaSelector"](E("0"), E("true_"), E("false_")), E("false_")),
        ]
        input_params = integrand_param_builder.get_parameters()
        constants = self.get_constants_for_evaluator()

        total_n_cff_terms = len(cff_structure.expressions)
        total_evaluators_to_build = sum(len(term.families) for term in cff_structure.expressions)
        total_evaluator_time = 0.0
        total_merge_time = 0.0
        current_eval_index = 0
        evaluator_calls = 0
        merge_calls = 0

        progress_format_str = (
            " cff %(cff_term)d/%(total_cff_terms)d | family %(family_idx)d/%(family_total)d | "
            "eval %(eval_idx)d/%(eval_total)d | merge_t %(merge_time).3fs (avg %(merge_avg).2fs) | "
            "eval_t %(eval_time).3fs (avg %(eval_avg).2fs) "
            if strategy == "merging"
            else " cff %(cff_term)d/%(total_cff_terms)d | family %(family_idx)d/%(family_total)d | eval %(eval_idx)d/%(eval_total)d | merge_t %(merge_time).3fs | eval_t %(eval_time).3fs "
        )
        progress_format_mapping: dict[str, Any] = {
            "cff_term": 0,
            "total_cff_terms": max(total_n_cff_terms, 1),
            "family_idx": 0,
            "family_total": 0,
            "eval_idx": 0,
            "eval_total": max(total_evaluators_to_build, 1),
            "merge_time": 0.0,
            "eval_time": 0.0,
        }
        if strategy == "merging":
            progress_format_mapping["merge_avg"] = 0.0
            progress_format_mapping["eval_avg"] = 0.0
        progress_format = progressbar.FormatCustomText(progress_format_str, progress_format_mapping)
        progress_bar: progressbar.ProgressBar | None = None
        if total_evaluators_to_build > 0:
            progress_bar = progressbar.ProgressBar(
                max_value=total_evaluators_to_build,
                widgets=[progressbar.Percentage(), progressbar.Bar(), progress_format, progressbar.AdaptiveETA()],
            )
            progress_bar.start()

        full_expression: Expression = E("0")
        full_evaluator: Evaluator | None = None
        n_terms = 0
        n_edge_orientations = len(cff_structure.expressions[0].orientation)
        n_e_surfaces = len(cff_structure.expressions[0].families[0])
        parametric_integrand_arguments = []
        for o_i in range(n_edge_orientations):
            parametric_integrand_arguments.append(self.SB["parametricEnergySign"](E(str(o_i))))
        for eta_i in range(n_e_surfaces):
            parametric_integrand_arguments.append(self.SB["parametricEtaSigma"](E(str(eta_i))))
        parametric_integrand_function_signature = self.SB["ICFF"](*parametric_integrand_arguments)
        # fmt: off
        parametric_integrand_function_definition = integrand_expression.replace(
            self.SB["energySign"](self.SB["o_id"], E("x_")),
            self.SB["parametricEnergySign"](E("x_"))
        ).replace(
            self.SB["etaSigma"](self.SB["o_id"], self.SB["f_id"], E("y_")),
            self.SB["parametricEtaSigma"](E("y_"))
        )
        # fmt: on
        for cff_i, cff_term in enumerate(cff_structure.expressions):
            orientation_substitutions = []
            for o_i, o in enumerate(cff_term.orientation):
                if strategy == "function_map":
                    orientation_substitutions.append(
                        Replacement(self.SB["parametricEnergySign"](E(str(o_i))), E("-1") if o.is_reversed() else E("1"))
                    )
                else:
                    orientation_substitutions.append(
                        Replacement(self.SB["energySign"](self.SB["o_id"], E(str(o_i))), E("-1") if o.is_reversed() else E("1"))
                    )
            if strategy == "function_map":
                parametric_expression = parametric_integrand_function_signature
            else:
                parametric_expression = integrand_expression
            orientation_substituted_integrand = parametric_expression.replace_multiple(orientation_substitutions)
            family_total = len(cff_term.families)
            for family_i, f in enumerate(cff_term.families):
                if progress_bar is not None:
                    merge_avg = (total_merge_time / merge_calls) if merge_calls > 0 else 0.0
                    eval_avg = (total_evaluator_time / evaluator_calls) if evaluator_calls > 0 else 0.0
                    progress_format.update_mapping(
                        **{
                            "cff_term": cff_i + 1,
                            "total_cff_terms": total_n_cff_terms,
                            "family_idx": family_i + 1,
                            "family_total": family_total,
                            "eval_idx": current_eval_index + 1,
                            "eval_total": total_evaluators_to_build,
                            "merge_time": total_merge_time,
                            "eval_time": total_evaluator_time,
                            **({"merge_avg": merge_avg, "eval_avg": eval_avg} if strategy == "merging" else {}),
                        }
                    )
                e_surface_selector_substitutions = []
                for eta_i, is_present in enumerate(f):
                    if strategy == "function_map":
                        e_surface_selector_substitutions.append(
                            Replacement(self.SB["parametricEtaSigma"](E(str(eta_i))), E("1") if is_present else E("0"))
                        )
                    else:
                        e_surface_selector_substitutions.append(
                            Replacement(self.SB["etaSigma"](self.SB["o_id"], self.SB["f_id"], E(str(eta_i))), E("1") if is_present else E("0"))
                        )
                concretized_integrand = orientation_substituted_integrand.replace_multiple(e_surface_selector_substitutions)
                if strategy != "function_map":
                    concretized_integrand = concretized_integrand.replace_multiple(selector_execution)
                n_terms += 1
                if strategy in ["summing", "function_map"]:
                    full_expression += concretized_integrand
                elif strategy == "merging":
                    evaluator_start = time.time()
                    concretized_evaluator = concretized_integrand.evaluator(
                        constants=constants,
                        functions={},  # type: ignore
                        params=input_params,
                        iterations=n_hornerscheme_iterations,
                        n_cores=8,
                        verbose=False,
                        external_functions=None,
                        conditionals=None,
                        cpe_iterations=n_cpe_iterations,
                    )
                    if GGHHH.FREEZE_INPUT_PHASES and not GGHHH.COMPLEXIFY_EVALUATOR:
                        concretized_evaluator.set_subcomponents(integrand_param_builder.get_components_phase())
                    if GGHHH.COMPLEXIFY_EVALUATOR:
                        concretized_evaluator.complexify(
                            real_components=integrand_param_builder.get_real_components(),
                        )
                    total_evaluator_time += time.time() - evaluator_start
                    evaluator_calls += 1
                    if full_evaluator is None:
                        full_evaluator = concretized_evaluator
                    else:
                        merge_start = time.time()
                        full_evaluator.merge(concretized_evaluator, cpe_iterations=n_cpe_iterations)
                        total_merge_time += time.time() - merge_start
                        merge_calls += 1
                current_eval_index += 1
                if progress_bar is not None:
                    merge_avg = (total_merge_time / merge_calls) if merge_calls > 0 else 0.0
                    eval_avg = (total_evaluator_time / evaluator_calls) if evaluator_calls > 0 else 0.0
                    progress_format.update_mapping(
                        **{
                            "cff_term": cff_i + 1,
                            "total_cff_terms": total_n_cff_terms,
                            "family_idx": family_i + 1,
                            "family_total": family_total,
                            "eval_idx": current_eval_index,
                            "eval_total": total_evaluators_to_build,
                            "merge_time": total_merge_time,
                            "eval_time": total_evaluator_time,
                            **({"merge_avg": merge_avg, "eval_avg": eval_avg} if strategy == "merging" else {}),
                        }
                    )
                    progress_bar.update(current_eval_index)

        if progress_bar is not None:
            progress_bar.finish()

        if strategy in ["summing", "function_map"]:
            logger.info(f"Generating full evaluator with symbolica using strategy '{strategy}' ...")
            if strategy == "function_map":
                # Sadly, functions currently requires arguments to be variable names directly. So we need to map them to variables
                function_args_replacement = []
                for o_i in range(n_edge_orientations):
                    function_args_replacement.append(
                        Replacement(self.SB["parametricEnergySign"](E(str(o_i))), E(f"{self.SB['parametricEnergySign'].get_name()}_{o_i}"))
                    )
                for eta_i in range(n_e_surfaces):
                    function_args_replacement.append(
                        Replacement(self.SB["parametricEtaSigma"](E(str(eta_i))), E(f"{self.SB['parametricEtaSigma'].get_name()}_{eta_i}"))
                    )
                parametric_integrand_arguments = [arg.replace_multiple(function_args_replacement) for arg in parametric_integrand_arguments]
                parametric_integrand_function_definition = parametric_integrand_function_definition.replace_multiple(function_args_replacement)
                functions: dict[tuple[Expression, str, tuple[Expression]], Expression] = {
                    (self.SB["ICFF"], "ICFF", tuple(parametric_integrand_arguments)): parametric_integrand_function_definition,
                }
                conditionals = [self.SB["etaSelector"]]
            else:
                functions = {}
                conditionals = None

            if GGHHH.DEBUG_FULL_EVALUATOR_PATH is not None:
                logger.critical("Debugging full evaluator generation inputs written to %s", GGHHH.DEBUG_FULL_EVALUATOR_PATH)
                with open(GGHHH.DEBUG_FULL_EVALUATOR_PATH, "w") as debug_file:
                    debug_file.write("# Auto-generated file for debugging full evaluator generation\n")
                    debug_file.write(f"expression=E('{full_expression.to_canonical_string()}')\n")
                    debug_file.write(
                        f"constants={{ {', '.join(f"E('{k.to_canonical_string()}'): E('{v.to_canonical_string()}')" for k, v in constants.items())} }}\n"
                    )
                    functions_formatted = []
                    for k, v in functions.items():
                        args_str = ", ".join(f'E("{arg.to_canonical_string()}")' for arg in k[2])
                        functions_formatted.append(f'(E("{k[0].to_canonical_string()}"), "{k[1]}", ({args_str})): E("{v.to_canonical_string()}")')
                    debug_file.write(f"functions={{ {', '.join(functions_formatted)} }}\n")
                    debug_file.write(f"input_params=[{', '.join(f'E("{param.to_canonical_string()}")' for param in input_params)}]\n")
                    if conditionals is not None:
                        debug_file.write(f"conditionals=[{', '.join(f'E("{cond.to_canonical_string()}")' for cond in conditionals)}]\n")
                    else:
                        debug_file.write("conditionals=None\n")
                    debug_file.write(f"real_components={integrand_param_builder.get_real_components()}\n")
                    debug_file.write(f"components_phase={integrand_param_builder.get_components_phase()}\n")

            evaluator_start = time.time()
            full_evaluator = full_expression.evaluator(
                constants=constants,
                functions=functions,  # type: ignore
                params=input_params,
                iterations=n_hornerscheme_iterations,
                n_cores=8,
                external_functions=None,
                conditionals=conditionals,
                verbose=GGHHH.VERBOSE_FULL_EVALUATOR,
                cpe_iterations=n_cpe_iterations,
            )
            if GGHHH.FREEZE_INPUT_PHASES and not GGHHH.COMPLEXIFY_EVALUATOR:
                full_evaluator.set_subcomponents(integrand_param_builder.get_components_phase())
            if GGHHH.COMPLEXIFY_EVALUATOR:
                full_evaluator.complexify(
                    real_components=integrand_param_builder.get_real_components(),
                )
            total_evaluator_time += time.time() - evaluator_start

        assert full_evaluator is not None

        if strategy == "merging":
            output_length = n_terms
        else:
            output_length = 1

        total_time = time.time() - total_start
        evaluator_pct = (total_evaluator_time / total_time * 100.0) if total_time > 0 else 0.0
        merge_pct = (total_merge_time / total_time * 100.0) if total_time > 0 else 0.0
        logger.info(
            "Evaluator timings: total %s%.3fs%s, evaluator calls %s%.3fs%s (%s%.1f%%%s), merges %s%.3fs%s (%s%.1f%%%s) (strategy=%s)",
            Colour.BLUE,
            total_time,
            Colour.END,
            Colour.BLUE,
            total_evaluator_time,
            Colour.END,
            Colour.GREEN,
            evaluator_pct,
            Colour.END,
            Colour.BLUE,
            total_merge_time,
            Colour.END,
            Colour.GREEN,
            merge_pct,
            Colour.END,
            strategy,
        )
        return PygloopEvaluator(
            full_evaluator,
            copy.deepcopy(integrand_param_builder),
            "full_integrand_evaluator",
            output_length=output_length,
            additional_data={"aggregation_strategy": strategy},
            complexified=GGHHH.COMPLEXIFY_EVALUATOR,
        )

    def build_parameter_evaluators(
        self,
        graph: DotGraph,
        n_hornerscheme_iterations: int = 100,
        _n_cpe_iterations: int | None = None,
    ) -> PygloopEvaluator:
        internal_edges = graph.get_internal_edges()
        external_edges = graph.get_external_edges()
        max_internal_edge_id = 0 if len(internal_edges) == 0 else max(int(e.get("id")) for e in internal_edges)
        max_external_edge_id = 0 if len(external_edges) == 0 else max(int(e.get("id")) for e in external_edges)  # noqa: F841

        lmb_projections = [Replacement(source, target) for source, target in graph.get_emr_replacements(head=self.SB["Q"].get_name())]
        computable_parameters = []
        model = self.get_model()
        edges = graph.dot.get_edges()

        # Add evaluation rules for the energies
        for e_i in range(max_internal_edge_id + 1):
            if e_i <= max_external_edge_id:
                external_energy = self.SB["Q"](E(f"{e_i}"), E("0"))
                external_energy = external_energy.replace_multiple(lmb_projections, repeat=True)
                computable_parameters.append(external_energy)
            else:
                particle = model.get_particle(edges[e_i].get("particle").replace('"', ""))
                on_shell_energy = E("0")
                for i in range(1, 4):
                    on_shell_energy += self.SB["Q"](E(f"{e_i}"), E(str(i))) * self.SB["Q"](E(f"{e_i}"), E(str(i)))
                if particle.mass.name.upper() != "ZERO":
                    on_shell_energy += S(f"UFO::{particle.mass.name}") ** 2
                on_shell_energy = on_shell_energy ** E("1/2")
                on_shell_energy = on_shell_energy.replace_multiple(lmb_projections, repeat=True)
                # Map to Kspatial
                on_shell_energy = on_shell_energy.replace(E("gammalooprs::K(x_,y_)"), self.SB["Kspatial"](E("x_"), E("y_-1")))
                computable_parameters.append(on_shell_energy)

        # Add evaluation rules for the spatial EMR momenta
        for e_i in range(max_internal_edge_id + 1):
            for i in range(1, 4):
                spatial_momentum = self.SB["Q"](E(f"{e_i}"), E(str(i)))
                spatial_momentum = spatial_momentum.replace_multiple(lmb_projections, repeat=True)
                # Map to Kspatial
                spatial_momentum = spatial_momentum.replace(E("gammalooprs::K(x_,y_)"), self.SB["Kspatial"](E("x_"), E("y_-1")))
                computable_parameters.append(spatial_momentum)

        # Now the param builder for these input parameters calculations
        param_builder = ParamBuilder()

        # External 4-momenta
        for e_i in range(max_external_edge_id + 1):
            param_builder.add_parameter_list((self.SB["externalP"], E(str(e_i))), 4)

        # Internal loop 3-momenta
        for i_lmb in range(self.n_loops):
            param_builder.add_parameter_list((self.SB["Kspatial"], E(str(i_lmb))), 3)

        # And finally the model parameters as they are needed for the particle masses
        model_inputs = self.get_model_parameters() + self.get_model_couplings()
        for param, value in model_inputs:
            param_builder.add_parameter((param,))
            param_builder.set_parameter((param,), value)

        if GGHHH.FREEZE_INPUT_PHASES:
            param_builder.freeze_all_current_parameters_phase()

        # We can now build the evaluator
        evaluator = Expression.evaluator_multiple(
            computable_parameters,
            constants={},
            functions={},  # type: ignore
            params=param_builder.get_parameters(),
            iterations=n_hornerscheme_iterations,
            n_cores=8,
            verbose=False,
            external_functions=None,  # type: ignore
        )
        pygloop_evaluator = PygloopEvaluator(evaluator, param_builder, "input_parameters_evaluator", output_length=len(computable_parameters))
        if GGHHH.FREEZE_INPUT_PHASES and not GGHHH.COMPLEXIFY_EVALUATOR:
            pygloop_evaluator.freeze_input_phases()
        if GGHHH.COMPLEXIFY_EVALUATOR:
            pygloop_evaluator.complexify()
        return pygloop_evaluator

    def generate_spenso_code(
        self,
        *args,
        integrand_evaluator_compiler: str = "symbolica_only",
        full_spenso_integrand_strategy: str | None = None,
        evaluators_compilation_options: dict[str, Any] | None = None,
        n_hornerscheme_iterations: int = 100,
        n_cpe_iterations: int | None = None,
        **opts,
    ) -> None:
        integrand_name = self.get_integrand_name()

        if self.spenso_evaluators[integrand_name] is not None:
            logger.info(f"Reusing existing Spenso evaluators for integrand {Colour.GREEN}{integrand_name}{Colour.END}.")  # fmt: off
            return

        if integrand_evaluator_compiler not in ["symbolica_only", "symjit"]:
            raise pygloopException(
                f"Integrand evaluator compiler '{integrand_evaluator_compiler}' not recognized. Input should be one of: 'symbolica' or 'symjit'."
            )
        if full_spenso_integrand_strategy not in [None, "merging", "summing", "function_map"]:
            raise pygloopException(
                f"Full spenso integrand strategy '{full_spenso_integrand_strategy}' not recognized. Input should be one of: None, 'merging', 'summing' or 'function_map'."
            )

        spenso_evaluator_compilation_options = GGHHH.DEFAULT_COMPILATION_OPTIONS.copy()
        spenso_evaluator_compilation_options.update(evaluators_compilation_options or {})

        amplitudes, _cross_sections = self.gl_worker.list_outputs()
        if integrand_name not in amplitudes:
            raise pygloopException(
                f"Amplitude {self.get_integrand_name()} not found in GammaLoop state. Generate graphs and code first with the generate subcommand."
            )
        process_id = amplitudes[integrand_name]

        graph_name = None
        match self.n_loops:
            case 1:
                graph_name = "GL15"
            case 2:
                graph_name = "GL303"
            case 3:
                graph_name = "pentaboxbox"
            case _:
                raise pygloopException(f"Number of loops {self.n_loops} not supported.")

        gghhh_dot_graphs = DotGraphs(dot_str=self.gl_worker.get_dot_files(process_id=process_id, integrand_name=integrand_name))

        gghhh_graph = gghhh_dot_graphs.get_graph(graph_name)

        t_spenso_generation_start = time.time()

        cff_structure = self.gl_worker.generate_cff_as_json_string(
            dot_string=gghhh_graph.to_string(),
            subgraph_nodes=[],
            reverse_dangling=[],
            orientation_pattern=None,
        )

        numerator = simplify_color(gghhh_graph.get_numerator() * gghhh_graph.get_projector())
        # The SM UFO stupidly writes ProjP + ProjM instead of the identity for the yukawa interaction, simply this away...
        numerator = numerator.replace(E("spenso::projm(x_,y_)+spenso::projp(x_,y_)"), E("spenso::g(x_,y_)"), repeat=True)

        hep_lib = TensorLibrary.hep_lib_atom()  # type: ignore
        tn = TensorNetwork(cook_indices(numerator), hep_lib)

        tn.execute(hep_lib)
        numerator_expr = tn.result_scalar()

        # Unwrap spenso::cind(x)
        numerator_expr = numerator_expr.replace(E("spenso::cind(x_)"), E("x_"), repeat=True)

        # Now substitute with on-shell energies, and rename spatial components
        numerator_expr = numerator_expr.replace(
            E("gammalooprs::Q(x_,0)"), self.SB["energySign"](self.SB["o_id"], E("x_")) * CFFStructure.SB["E"](S("x_")), repeat=True
        )
        numerator_expr = numerator_expr.replace(E("gammalooprs::Q(x_,i_)"), self.SB["Qspatial"](S("x_"), S("i_") - 1), repeat=True)

        # print(numerator_expr.to_canonical_string())
        # stop

        try:
            cff_structure = json.loads(cff_structure)
        except json.JSONDecodeError as e:
            raise pygloopException(f"Error decoding CFF structure JSON: {e}") from e
        cff_structure = CFFStructure(cff_structure)

        # stop

        # print(cff_structure.__str__(show_families=True))
        # Build a CFF term evaluators, with each esurface wrapped around a selector
        cff_term = E("1")
        for e_surf in cff_structure.e_surfaces:
            cff_term *= self.SB["etaSelector"](self.SB["etaSigma"](self.SB["o_id"], self.SB["f_id"], e_surf.id), e_surf.expression, 1)

        internal_edges = gghhh_graph.get_internal_edges()
        external_edges = gghhh_graph.get_external_edges()
        max_internal_edge_id = 0 if len(internal_edges) == 0 else max(int(e.get("id")) for e in internal_edges)
        max_external_edge_id = 0 if len(external_edges) == 0 else max(int(e.get("id")) for e in external_edges)  # noqa: F841

        # Add the CFF normalization factor
        for e_i in range(max_internal_edge_id + 1):
            if e_i <= max_external_edge_id:
                continue
            cff_term *= 2 * CFFStructure.SB["E"](E(str(e_i)))
        # overall normalization
        cff_term *= -(((-2 * Expression.PI) ** 3) ** self.n_loops)

        if not GGHHH.ENABLE_CFF_TERM:
            cff_term = E("1")
        if not GGHHH.ENABLE_NUMERATOR_TERM:
            numerator_expr = E("1")

        # print(numerator_expr.to_canonical_string())
        integrand_expression = numerator_expr / cff_term

        # Build the integrand evaluator
        parametric_evaluator = self.build_parametric_integrand_evaluator(gghhh_graph, integrand_expression, cff_structure)

        # Build the aggregated summed integrand evaluator
        if full_spenso_integrand_strategy is not None:
            full_evaluator = self.build_full_integrand_evaluator(
                integrand_expression,
                cff_structure,
                parametric_evaluator.param_builder,
                strategy=full_spenso_integrand_strategy,
                n_hornerscheme_iterations=n_hornerscheme_iterations,
                n_cpe_iterations=n_cpe_iterations,
            )
        else:
            full_evaluator = None

        # Now build the input parameter evaluator
        # This is technically not necessary, I was just curious to see how performance looks like when not using function maps for the on-shell energies and emr decomposition
        params_evaluator = self.build_parameter_evaluators(
            gghhh_graph, n_hornerscheme_iterations=n_hornerscheme_iterations, _n_cpe_iterations=n_cpe_iterations
        )
        self.initialize_param_builders(
            [parametric_evaluator.param_builder] + [full_evaluator.param_builder,] if full_evaluator is not None else [],
            params_evaluator.param_builder,
        )  # fmt: off

        evaluator_directory = pjoin(EVALUATORS_FOLDER, self.name, self.get_integrand_name())
        os.makedirs(evaluator_directory, exist_ok=True)

        logger.info(f"Compiling evaluators in {Colour.GREEN}{evaluator_directory}{Colour.END}...")
        t_compile_start = time.time()
        parametric_evaluator.compile(
            evaluator_directory, integrand_evaluator_compiler=integrand_evaluator_compiler, **spenso_evaluator_compilation_options
        )
        parametric_evaluator.save(evaluator_directory)
        params_evaluator.compile(
            evaluator_directory, integrand_evaluator_compiler=integrand_evaluator_compiler, **spenso_evaluator_compilation_options
        )
        params_evaluator.save(evaluator_directory)
        if full_evaluator is not None:
            full_evaluator.compile(
                evaluator_directory, integrand_evaluator_compiler=integrand_evaluator_compiler, **spenso_evaluator_compilation_options
            )
            full_evaluator.save(evaluator_directory)
        size_on_disk = self.get_size_on_disk(
            {"integrand_type": "spenso_parametric", "evaluator_compilation": "symbolica"}
            if full_spenso_integrand_strategy is None
            else {"integrand_type": "spenso_summed", "evaluator_compilation": "symbolica"}
        )
        if size_on_disk is None:
            size_on_disk_str = f"{Colour.RED}N/A MB{Colour.END}"
        else:
            size_on_disk_str = f"{Colour.GREEN}{size_on_disk / 1000_000.0:.2f} MB{Colour.END}"
        logger.info(
            f"Compiled evaluators in {Colour.GREEN}{evaluator_directory}{Colour.END} in {Colour.BLUE}{(time.time() - t_compile_start):.2f} seconds{Colour.END}. [ {size_on_disk_str} on disk ] "
        )
        logger.info(f"Saved spenso integrand evaluators to {Colour.GREEN}{evaluator_directory}{Colour.END}")
        self.spenso_evaluators[integrand_name] = {  # type: ignore
            "parametric_integrand_evaluator": parametric_evaluator,
            "full_integrand_evaluator": full_evaluator,
            "input_parameters_evaluator": params_evaluator,
        }

        logger.info(
            f"Spenso code generation for process {Colour.BLUE}{self.name} ({self.n_loops} loops){Colour.END} took {Colour.GREEN}{(time.time() - t_spenso_generation_start):.2f} seconds{Colour.END}."
        )

        if GGHHH.DEBUG_FULL_EVALUATOR_PATH is not None:
            all_evaluators = self.spenso_evaluators[integrand_name]
            logger.critical("Debugging full evaluator call inputs written to %s", GGHHH.DEBUG_FULL_EVALUATOR_PATH)
            with open(GGHHH.DEBUG_FULL_EVALUATOR_PATH, "a") as debug_file:
                self.initialize_param_builders(
                    [ all_evaluators["parametric_integrand_evaluator"].param_builder ] + ([ all_evaluators["full_integrand_evaluator"].param_builder, ] if all_evaluators["full_integrand_evaluator"] is not None else [ ]), # type: ignore
                    all_evaluators["input_parameters_evaluator"].param_builder # type: ignore
                )  # fmt: off
                self.set_from_sample(
                    all_evaluators["full_integrand_evaluator"], # type: ignore
                    all_evaluators["input_parameters_evaluator"], # type: ignore
                    ks = [ Vector(100.0, 200.0, 300.0), ],
                )  # fmt: off
                debug_file.write(
                    f"call_inputs=[{','.join('%.15e' % input for input in all_evaluators['full_integrand_evaluator'].param_builder.get_values(True))}]\n"  # type: ignore
                )

        # ####################
        # TEST INTEGRAND START
        # ####################

        # # Try and reload the full integrand evaluator as opposed to taking current live version
        # # self.spenso_evaluators[integrand_name]["full_integrand_evaluator"] = PygloopEvaluator.load(evaluator_directory, "full_integrand_evaluator")  # type: ignore

        # all_evaluators = self.spenso_evaluators[integrand_name]
        # assert all_evaluators is not None
        # self.initialize_param_builders(
        #     [ all_evaluators["parametric_integrand_evaluator"].param_builder ] + ([ all_evaluators["full_integrand_evaluator"].param_builder, ] if all_evaluators["full_integrand_evaluator"] is not None else [ ]),
        #     all_evaluators["input_parameters_evaluator"].param_builder
        # )  # fmt: off

        # self.set_from_sample(
        #     all_evaluators["parametric_integrand_evaluator"], # type: ignore
        #     all_evaluators["input_parameters_evaluator"], # type: ignore
        #     ks = [ Vector(100.0, 200.0, 300.0), ],
        # )  # fmt: off
        # total_res_eager = complex(0.0, 0.0)
        # total_res_compiled = complex(0.0, 0.0)
        # for cff_term in all_evaluators["parametric_integrand_evaluator"].additional_data["cff_structure"].expressions:  # type: ignore
        #     for f_i in range(len(cff_term.families)):
        #         self.set_from_sample(
        #             all_evaluators["parametric_integrand_evaluator"], # type: ignore
        #             cff_term = cff_term,
        #             family_id = f_i,
        #         )  # fmt: off
        #         total_res_eager += all_evaluators["parametric_integrand_evaluator"].evaluate(eager=True)  # type: ignore
        #         total_res_compiled += all_evaluators["parametric_integrand_evaluator"].evaluate(eager=False)  # type: ignore
        # print("Parametric integrand test evaluation result (eager):", total_res_eager)
        # print("Parametric integrand test evaluation result (compiled):", total_res_compiled)

        # self.set_from_sample(
        #     all_evaluators["full_integrand_evaluator"], # type: ignore
        #     all_evaluators["input_parameters_evaluator"], # type: ignore
        #     ks = [ Vector(100.0, 200.0, 300.0), ],
        # )  # fmt: off

        # full_result = all_evaluators["full_integrand_evaluator"].evaluate(eager=True)  # type: ignore
        # if all_evaluators["full_integrand_evaluator"].additional_data["aggregation_strategy"] == "merging":  # type: ignore
        #     full_result = full_result.sum()
        # else:
        #     full_result = full_result[0]
        # print("Full integrand test evaluation result (eager):", full_result)
        # full_result = all_evaluators["full_integrand_evaluator"].evaluate(eager=False)  # type: ignore
        # if all_evaluators["full_integrand_evaluator"].additional_data["aggregation_strategy"] == "merging":  # type: ignore
        #     full_result = full_result.sum()
        # else:
        #     full_result = full_result[0]
        # print("Full integrand test evaluation result (compiled):", full_result)

        # ##################
        # TEST INTEGRAND END
        # ##################

    def generate_gammaloop_code(self) -> None:
        amplitudes, _cross_sections = self.gl_worker.list_outputs()
        integrand_name = self.get_integrand_name()
        process_graphs_name = self.get_integrand_name(suffix="_generated_graphs")
        if process_graphs_name not in amplitudes or integrand_name not in amplitudes:
            raise pygloopException(f"Amplitude with named integrand {process_graphs_name} not found in GammaLoop state. Generate graphs first.")
        if not os.path.isfile(pjoin(DOTS_FOLDER, self.name, f"{integrand_name}.dot")):
            raise pygloopException(
                f"Processed dot file not found at {pjoin(DOTS_FOLDER, self.name, f'{integrand_name}.dot')}. Generate graphs first."
            )
        if os.path.isfile(pjoin(PYGLOOP_FOLDER, "outputs", "gammaloop_states", self.name, f"{integrand_name}_is_generated")):
            logger.info(f"Amplitude {integrand_name} already generated and recycled.")
            return

        t_gammaloop_generation_start = time.time()
        self.gl_worker.run(f"generate existing -p {amplitudes[process_graphs_name]} -i {integrand_name}")
        logger.info(
            f"GammaLoop code generation for process {Colour.BLUE}{self.name} ({self.n_loops} loops){Colour.END} took {Colour.GREEN}{(time.time() - t_gammaloop_generation_start):.2f} seconds{Colour.END}."
        )
        self.save_state()
        with open(pjoin(PYGLOOP_FOLDER, "outputs", "gammaloop_states", self.name, f"{integrand_name}_is_generated.txt"), "w") as f:
            f.write("generated")

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
                raise pygloopException("Provided ps point does not respect momentum conservation.")

    def parameterize(self, xs: list[float], parameterisation: str, origin: Vector | None = None) -> tuple[Vector, float]:
        match parameterisation:
            case "cartesian":
                return self.cartesian_parameterize(xs, origin)
            case "spherical":
                return self.spherical_parameterize(xs, origin)
            case _:
                raise pygloopException(f"Parameterisation {parameterisation} not implemented.")

    def cartesian_parameterize(self, xs: list[float], origin: Vector | None = None) -> tuple[Vector, float]:
        return self.cartesian_parameterize_v2(xs, origin)

    def cartesian_parameterize_v1(self, xs: list[float], origin: Vector | None = None) -> tuple[Vector, float]:
        x, y, z = xs
        scale = self.e_cm * RESCALING
        v = Vector((1 / (1 - x) - 1 / x), (1 / (1 - y) - 1 / y), (1 / (1 - z) - 1 / z)) * scale
        if origin is not None:
            v = v + origin
        jac = scale * (1 / (1 - x) ** 2 + 1 / x**2)
        jac *= scale * (1 / (1 - y) ** 2 + 1 / y**2)
        jac *= scale * (1 / (1 - z) ** 2 + 1 / z**2)
        return (v, jac)

    def cartesian_parameterize_v2(self, xs: list[float], origin: Vector | None = None) -> tuple[Vector, float]:
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

    def cartesian_parameterize_v3(self, xs: list[float], origin: Vector | None = None) -> tuple[Vector, float]:
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

    def spherical_parameterize(self, xs: list[float], origin: Vector | None = None) -> tuple[Vector, float]:
        rx, costhetax, phix = xs
        scale = self.e_cm * RESCALING
        r = rx / (1 - rx) * scale
        costheta = (0.5 - costhetax) * 2
        sintheta = math.sqrt(1 - costheta**2)
        phi = phix * 2 * math.pi
        v = Vector(r * sintheta * math.cos(phi), r * sintheta * math.sin(phi), r * costheta)
        if origin is not None:
            v = v + origin
        jac = 2 * (2 * math.pi) * (r**2 * scale / (1 - rx) ** 2)
        return (v, jac)

    def integrand_xspace(
        self,
        xs: list[float],
        parameterization: str,
        integrand_implementation: dict[str, Any],
        phase: str,
        multi_channeling: bool | int = True,
    ) -> float:
        try:
            if multi_channeling is False:
                n_loops = self.n_loops
                expected_len = 3 * n_loops
                if len(xs) != expected_len:
                    raise pygloopException(f"Expected {expected_len} integration variables for {n_loops} loop momenta, received {len(xs)}.")
                loop_momenta: list[Vector] = []
                total_jac = 1.0
                for i_loop in range(n_loops):
                    k, jac = self.parameterize(xs[3 * i_loop : 3 * (i_loop + 1)], parameterization)
                    loop_momenta.append(k)
                    total_jac *= jac
                wgt = self.integrand(loop_momenta, integrand_implementation)
                if phase == "real":
                    wgt = wgt.real
                else:
                    wgt = wgt.imag
                final_wgt = wgt * total_jac
            else:
                if self.n_loops != 1:
                    raise pygloopException("Multi-channeling only implemented for one-loop processes.")
                final_wgt = 0.0
                multi_channeling_power = 3
                q_offsets = [
                    Vector(0.0, 0.0, 0.0),
                    self.ps_point[1].spatial(),
                    (self.ps_point[1] - self.ps_point[2]).spatial(),
                    (self.ps_point[1] - self.ps_point[2] - self.ps_point[3]).spatial(),  # fmt: off
                    (self.ps_point[1] - self.ps_point[2] - self.ps_point[3] - self.ps_point[4]).spatial(),  # fmt: off
                ]
                for i_channel in range(5):
                    if multi_channeling is True or multi_channeling == i_channel:
                        k, jac = self.parameterize(xs, parameterization, q_offsets[i_channel] * -1)
                        inv_oses = [
                            1.0 / math.sqrt((k + q_offsets[i_prop]).squared() + self.m_top**2)
                            for i_prop in range(5)  # fmt: off
                        ]
                        wgt = self.integrand([k], integrand_implementation)
                        if phase == "real":
                            wgt = wgt.real
                        else:
                            wgt = wgt.imag
                        final_wgt += jac * inv_oses[i_channel] ** multi_channeling_power * wgt / sum(t**multi_channeling_power for t in inv_oses)

            if math.isnan(final_wgt):
                logger.debug(
                    f"Integrand evaluated to NaN at xs = [{Colour.BLUE}{', '.join(f'{xi:+.16e}' for xi in xs)}{Colour.END}]. Setting it to zero"
                )  # fmt: off
                final_wgt = 0.0
        except ZeroDivisionError:
            logger.debug(
                f"Integrand divided by zero at xs = [{Colour.BLUE}{', '.join(f'{xi:+.16e}' for xi in xs)}{Colour.END}]. Setting it to zero"
            )  # fmt: off
            final_wgt = 0.0

        return final_wgt

    def integrand(self, loop_momenta: list[Vector], integrand_implementation: dict[str, Any]) -> complex:
        try:
            match integrand_implementation["integrand_type"]:
                case "spenso_parametric":
                    return self.spenso_integrand(loop_momenta, integrand_implementation, parametric=True)
                case "spenso_summed":
                    return self.spenso_integrand(loop_momenta, integrand_implementation, parametric=False)
                case "gammaloop":
                    return self.gammaloop_integrand(loop_momenta)
                case _:
                    raise pygloopException(f"Integrand implementation {integrand_implementation['integrand_type']} not implemented.")
        except ZeroDivisionError:
            logger.debug(
                f"Integrand divided by zero for ks = [{Colour.BLUE}{
                    ','.join('[' + ', '.join(f'{ki:+.16e}' for ki in k.to_list()) + ']' for k in loop_momenta)
                }{Colour.END}]. Setting it to zero"
            )
            return 0.0

    def get_integrand_name(self, suffix="_processed"):
        match self.n_loops:
            case 1 | 2 | 3:
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
        self, loop_momenta: list[Vector], integrand_implementation: dict[str, Any], parametric=False, prefer_eager_mode=False
    ) -> complex:
        try:
            integrand_name = self.cache["integrand_name"]
        except KeyError:
            integrand_name = self.get_integrand_name()
            self.cache["integrand_name"] = integrand_name

        if self.spenso_evaluators[integrand_name] is None:
            raise pygloopException(f"Spenso evaluators for integrand {integrand_name} not generated yet. Run generate_spenso_code first.")

        parameters_evaluator = self.spenso_evaluators[integrand_name]["input_parameters_evaluator"]  # type: ignore
        if not parametric:
            full_evaluator = self.spenso_evaluators[integrand_name]["full_integrand_evaluator"]  # type: ignore
            if full_evaluator is None:
                raise pygloopException("Full spenso integrand evaluator not built.")
            self.set_from_sample(full_evaluator, parameters_evaluator, ks=loop_momenta)  # fmt: off
            full_evaluator_result = full_evaluator.evaluate(
                eager=prefer_eager_mode, prefer_symjit=(integrand_implementation["evaluator_compiler"] == "symjit")
            )
            if full_evaluator.additional_data["aggregation_strategy"] == "merging":
                full_evaluator_result = full_evaluator_result.sum()
            else:
                full_evaluator_result = full_evaluator_result[0]
            return full_evaluator_result

        else:
            parametric_evaluator = self.spenso_evaluators[integrand_name]["parametric_integrand_evaluator"]  # type: ignore
            cff_structure = parametric_evaluator.additional_data["cff_structure"]

            final_result = complex(0.0, 0.0)

            self.set_from_sample(parametric_evaluator, parameters_evaluator, ks=loop_momenta)  # fmt: off
            for cff_term in cff_structure.expressions:
                for f_i in range(len(cff_term.families)):
                    self.set_from_sample(parametric_evaluator, cff_term=cff_term, family_id=f_i)
                    final_result += parametric_evaluator.evaluate(
                        eager=prefer_eager_mode, prefer_symjit=(integrand_implementation["evaluator_compiler"] == "symjit")
                    )[0]

            return final_result

    def integrate(
        self,
        integrator: str,
        parameterisation: str,
        integrand_implementation: dict[str, Any],
        target: float | complex | None = None,
        toml_config_path: str | None = None,
        **opts,
    ) -> IntegrationResult:
        match integrator:
            case "naive":
                return self.naive_integrator(
                    parameterisation,
                    integrand_implementation,
                    target,
                    **opts,
                )
            case "vegas":
                return self.vegas_integrator(
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
            raise pygloopException("GammaLoop integrator only supports 'gammaloop' integrand implementation.")

        integrand_name = self.get_integrand_name()
        amplitudes, _cross_sections = self.gl_worker.list_outputs()
        if integrand_name not in amplitudes:
            raise pygloopException(
                f"Amplitude {integrand_name} not found in GammaLoop state. Generate graphs and code first with the generate subcommand. Available amplitudes: {list(amplitudes.keys())}"
            )  # fmt: off

        integration_options = {
            "n_start": opts.get("points_per_iteration", 100_000),
            "n_increase": 0,
            "n_max": opts.get("points_per_iteration", 100_000) * opts.get("n_iterations", 10),
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
                integrate_command.append(["--target", f"{target.real:.16e}", f"{target.imag:.16e}"])
            elif isinstance(target, float):
                integrate_command.append(["--target", f"{target:.16e}", "0.0"])
        if "n_cores" in opts:
            integrate_command.append(["--n-cores", str(opts["n_cores"])])
        if opts.get("restart", False):
            integrate_command.append(["--restart"])

        integrate_command_str = " ".join(" ".join(itg_o for itg_o in itg_opt) for itg_opt in integrate_command)
        logger.info(f"Running GammaLoop integration with command:\n{Colour.GREEN}{integrate_command_str}{Colour.END}")
        t_start = time.time()
        self.gl_worker.run(integrate_command_str)  # fmt: off
        t_elapsed = time.time() - t_start

        res = None
        if os.path.isfile(results_path):
            with open(results_path, "r") as f_res:
                res = json.load(f_res)

        integration_result = IntegrationResult(0.0, 0.0)
        if res is None:
            logger.error(f"GammaLoop integration finished but no result file found at '{results_path}'.")
        else:
            if opts.get("phase", "real") == "real":
                central, error = res["result"]["re"], res["error"]["re"]
            else:
                central, error = res["result"]["im"], res["error"]["im"]
            integration_result = IntegrationResult(central, error, n_samples=res["neval"], elapsed_time=t_elapsed)
        return integration_result

    def get_size_on_disk(self, integrand_implementation: dict[str, Any]) -> int | None:
        """Returns the size on disk in bytes of the integrand implementation data."""
        integrand_name = self.get_integrand_name()
        match integrand_implementation["integrand_type"]:
            case "spenso_parametric":
                evaluator_directory = pjoin(EVALUATORS_FOLDER, self.name, integrand_name)
                size = 0
                if integrand_implementation.get("evaluator_compilation", "symbolica") == "symjit":
                    files_list = [
                        "input_parameters_evaluator.so",
                        "parametric_integrand_evaluator.sjb",
                        "parametric_integrand_evaluator_additional_data.pkl",
                    ]
                else:
                    files_list = [
                        "input_parameters_evaluator.so",
                        "parametric_integrand_evaluator.so",
                        "parametric_integrand_evaluator_additional_data.pkl",
                    ]
                for file_name in files_list:
                    file_path = pjoin(evaluator_directory, file_name)
                    if os.path.exists(file_path):
                        size += os.path.getsize(file_path)
                    else:
                        return None
                return size

            case "spenso_summed":
                evaluator_directory = pjoin(EVALUATORS_FOLDER, self.name, integrand_name)
                size = 0
                if integrand_implementation.get("evaluator_compilation", "symbolica") == "symjit":
                    files_list = ["input_parameters_evaluator.so", "full_integrand_evaluator.sjb"]
                else:
                    files_list = ["input_parameters_evaluator.so", "full_integrand_evaluator.so"]
                for file_name in files_list:
                    file_path = pjoin(evaluator_directory, file_name)
                    if os.path.exists(file_path):
                        size += os.path.getsize(file_path)
                    else:
                        return None
                return size

            case "gammaloop":
                binary_dump_path = pjoin(
                    GAMMALOOP_STATES_FOLDER,
                    self.name,
                    "processes",
                    "amplitudes",
                    f"{self.name}_{self.n_loops}L",
                    integrand_name,
                    "integrand",
                    "integrand.bin",
                )
                if os.path.exists(binary_dump_path):
                    return os.path.getsize(binary_dump_path)
                # if self.i
                # total_size = 0
                # for dirpath, dirnames, filenames in os.walk(pjoin(PYGLOOP_FOLDER, "outputs", "gammaloop_states")):
                #     for f in filenames:
                #         fp = os.path.join(dirpath, f)
                #         total_size += os.path.getsize(fp)
                # return total_size
                return 0
            case _:
                raise pygloopException(f"Integrand implementation {integrand_implementation['integrand_type']} not implemented.")

    @set_gammaloop_level(logging.ERROR, logging.INFO)
    def naive_integrator(
        self,
        parameterisation: str,
        integrand_implementation: dict[str, Any],
        target,
        **opts,
    ) -> IntegrationResult:
        return run_naive_integrator(self, parameterisation, integrand_implementation, target, **opts)

    @set_gammaloop_level(logging.ERROR, logging.INFO)
    def vegas_integrator(
        self,
        parameterisation: str,
        integrand_implementation: dict[str, Any],
        _target,
        **opts,
    ) -> IntegrationResult:
        return run_vegas_integrator(self, parameterisation, integrand_implementation, _target, **opts)

    @set_gammaloop_level(logging.ERROR, logging.INFO)
    def symbolica_integrator(
        self,
        parameterisation: str,
        integrand_implementation: dict[str, Any],
        target,
        **opts,
    ) -> IntegrationResult:
        return run_symbolica_integrator(self, parameterisation, integrand_implementation, target, **opts)

    @set_gammaloop_level(logging.ERROR, logging.INFO)
    def plot(self, **opts):
        return plot_integrand(self, **opts)
