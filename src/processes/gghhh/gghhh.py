# Targets for the pentagon one-loop integral with cyclic ordering of the external legs
# IMPORTANT: the results below comes from the raw amplitude AMP(1,5) of MADLOOP *divided by -2* to account for Tr(t^a t^b) = 1/2 delta^{ab}.
# Hel 1 is : --000
# Hel 2 is : -+000
# Hel 3 is : +-000
# Hel 4 is : ++000
#
# This amplitude graph never has any pole.

# 0.5000000000000000D+03   0.0000000000000000D+00   0.0000000000000000D+00   0.5000000000000000D+03
# 0.5000000000000000D+03   0.0000000000000000D+00   0.0000000000000000D+00  -0.5000000000000000D+03
# 0.4385555662246945D+03   0.1553322001835378D+03   0.3480160396513587D+03  -0.1773773615718412D+03
# 0.3563696374921922D+03  -0.1680238900851100D+02  -0.3187291102436005D+03   0.9748719163688098D+02
# 0.2050747962831133D+03  -0.1385298111750267D+03  -0.2928692940775817D+02   0.7989016993496030D+02

# # Physical top mass (MT=ymt=173.0)
#  >>> IHEL =            1
#  AMPL(1,5)=  (-3.14551938508347352E-006,4.46854051027942967E-006)
#  >>> IHEL =            2
#  AMPL(1,5)=   (9.55408514080194571E-009,2.78298167757932416E-006)
#  >>> IHEL =            3
#  AMPL(1,5)= (-3.19380629855654766E-006,-9.01168425551671615E-006)
#  >>> IHEL =            4
#  AMPL(1,5)=  (1.89203604685291554E-006,-5.46603099163412881E-006)

# # Unphysical top mass (MT=ymt=1000.0)
#  >>> IHEL =            1
#  AMPL(1,5)=   (6.56089133881205492E-004,4.17078968906596113E-006)
#  >>> IHEL =            2
#  AMPL(1,5)= (-2.89630814594050972E-006,-8.87804989993630875E-006)
#  >>> IHEL =            3
#  AMPL(1,5)=  (-2.89630814768248677E-006,8.87804990034476668E-006)
#  >>> IHEL =            4
#  AMPL(1,5)=  (6.56089133881216768E-004,-4.17078968913725420E-006)

# 0.5000000000000000D+03   0.0000000000000000D+00   0.0000000000000000D+00   0.5000000000000000D+03
# 0.5000000000000000D+03   0.0000000000000000D+00   0.0000000000000000D+00  -0.5000000000000000D+03
# 0.4622059639026168D+03   0.1678033838387855D+03   0.2872919263250002D+03  -0.2954906538418281D+03
# 0.1553689858956567D+03  -0.6346487586464051D+02  -0.3905281750811410D+02   0.5442066477367285D+02
# 0.3824250502017265D+03  -0.1043385079741450D+03  -0.2482391088168861D+03   0.2410699890681553D+03

# # Physical top mass (MT=ymt=173.0)

#  >>> IHEL =            1
#  AMPL(1,5)=   (1.79229062116069573E-005,5.23819986497045118E-007)
#  >>> IHEL =            2
#  AMPL(1,5)=   (1.35530147927836316E-005,2.77672899246084792E-006)
#  >>> IHEL =            3
#  AMPL(1,5)=  (5.03924664020387931E-007,-1.45537484583090386E-005)
#  >>> IHEL =            4
#  AMPL(1,5)=   (1.80944073681505885E-005,3.67594748147673556E-006)

# # Unphysical top mass (MT=ymt=1000.0)

#  >>> IHEL =            1
#  AMPL(1,5)=  (6.09909283686016468E-004,-1.09117176750885081E-006)
#  >>> IHEL =            2
#  AMPL(1,5)= (-7.88985817775678482E-006,-1.08700048931499477E-005)
#  >>> IHEL =            3
#  AMPL(1,5)=  (-7.88985817821675929E-006,1.08700048932430332E-005)
#  >>> IHEL =            4
#  AMPL(1,5)=   (6.09909283686009963E-004,1.09117176736822661E-006)

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

from utils.naive_integrator import naive_integrator as run_naive_integrator
from utils.plotting import plot_integrand
from utils.polarizations import ixxxxx
from utils.symbolica_integrator import symbolica_integrator as run_symbolica_integrator
from utils.utils import (
    CONFIGS_FOLDER,  # noqa: F401
    DOTS_FOLDER,  # noqa: F401
    EVALUATORS_FOLDER,  # noqa: F401
    GAMMALOOP_STATES_FOLDER,  # noqa: F401
    INTEGRATION_WORKSPACE_FOLDER,  # noqa: F401
    OUTPUTS_FOLDER,  # noqa: F401
    PYGLOOP_FOLDER,
    CFFStructure,
    CFFTerm,
    Colour,
    DotGraph,
    DotGraphs,
    IntegrationResult,
    ParamBuilder,
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

    SB = {
        "etaSelector": S("pygloop::ηs"),
        "etaSigma": S("pygloop::ση"),
        "energySign": S("pygloop::σE"),
        "Qspatial": S("pygloop::Qs"),
        "Kspatial": S("pygloop::Ks"),
        "Q": S("pygloop::Q"),
        "o_id": S("pygloop::o_id"),
        "f_id": S("pygloop::f_id"),
        "vector_pol": S("gammalooprs::ϵ"),
        "externalP": S("gammalooprs::P"),
    }

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
            pjoin(PYGLOOP_FOLDER, "outputs", "gammaloop_states", self.name),
            log_file_name=self.name,
            log_level=gl_log_level,
        )
        self.set_log_level(logger_level)

        if toml_config_path is None:
            toml_config_path = pjoin(CONFIGS_FOLDER, self.name, "generate.toml")

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
            runtime_toml_config_path = pjoin(CONFIGS_FOLDER, self.name, "runtime.toml")
        self.runtime_toml_config_path = runtime_toml_config_path

        logger.info(f"Setting runtime configuration for all outputs from toml file: {Colour.BLUE}{runtime_toml_config_path}{Colour.END}.")  # fmt: off
        for output_name, output_id in amplitudes.items():
            # Currently bugged: not all functionalities available on integrands not yet generated
            if "_generated_graphs" in output_name:
                continue
            self.gl_worker.run(f"set process -p {output_id} -i {output_name} file {self.runtime_toml_config_path}")  # fmt: off
            self.set_sample_point(self.ps_point, self.helicities, str(output_id), output_name)

        self.save_state()
        # Cache some quantities for performance
        self.cache: dict[str, Any] = {}

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
        return {E("spenso::TR"): E("1/2")}

    def initialize_param_builders(
        self,
        integrand_param_builder: ParamBuilder,
        input_params_param_builder: ParamBuilder,
    ):
        # Update model parameters
        model_inputs = self.get_model_parameters() + self.get_model_couplings()
        for param, value in model_inputs:
            integrand_param_builder.set_parameter((param,), value)
            input_params_param_builder.set_parameter((param,), value)

        # Set external momenta
        for i_p, p in enumerate(self.ps_point):
            input_params_param_builder.set_parameter_values((self.SB["externalP"], E(str(i_p))), [complex(p_i) for p_i in p.to_list()])

        # Add the polarization vectors
        for e_i in [0, 1]:
            pol_vector = ixxxxx(self.ps_point[e_i].to_list(), 0.0, self.helicities[e_i], 1)[2::]
            integrand_param_builder.set_parameter_values((self.SB["vector_pol"], E(str(e_i))), [complex(pol_i) for pol_i in pol_vector])

    def set_from_sample(
        self,
        ks: list[Vector] | None,
        cff_term: CFFTerm | None,
        family_id: int | None,
        integrand_param_builder: ParamBuilder,
        input_params_evaluator: Evaluator,
        input_params_param_builder: ParamBuilder,
    ) -> None:
        # Set loop momenta and emr momenta
        if ks is not None:
            for i_k, k in enumerate(ks):
                input_params_param_builder.set_parameter_values((self.SB["Kspatial"], E(str(i_k))), [complex(k_i) for k_i in k.to_list()])

            # Evaluate derived inputs
            derived_inputs = input_params_evaluator.evaluate_complex(input_params_param_builder.get_values()[None, :])[0]
            integrand_param_builder.set_parameter_values_within_range(0, len(derived_inputs), derived_inputs)

        # Add the E-surfaces selectors
        if cff_term is not None and family_id is not None:
            integrand_param_builder.set_parameter_values((self.SB["etaSigma"], self.SB["o_id"], self.SB["f_id"]), cff_term.masks[family_id])

        # Add the energy signs
        if cff_term is not None:
            integrand_param_builder.set_parameter_values((self.SB["energySign"], self.SB["o_id"]), cff_term.orientation_signs)

    def build_integrand_evaluator(self, graph: DotGraph, integrand: Expression, cff_structure: CFFStructure) -> tuple[Evaluator, ParamBuilder]:
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

        # Add the E-surfaces selectors
        param_builder.add_parameter_list((self.SB["etaSigma"], self.SB["o_id"], self.SB["f_id"]), len(cff_structure.e_surfaces))
        # Add the energy signs
        param_builder.add_parameter_list((self.SB["energySign"], self.SB["o_id"]), max_internal_edge_id + 1)

        model_inputs = self.get_model_parameters() + self.get_model_couplings()
        for param, value in model_inputs:
            param_builder.add_parameter((param,))
            param_builder.set_parameter((param,), value)

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

        return (evaluator, param_builder)

    def build_full_integrand_evaluator(
        self, integrand_expression: Expression, cff_structure: CFFStructure, integrand_param_builder: ParamBuilder
    ) -> Evaluator:
        selector_execution = [
            Replacement(self.SB["etaSelector"](E("1"), E("true_"), E("false_")), E("true_")),
            Replacement(self.SB["etaSelector"](E("0"), E("true_"), E("false_")), E("false_")),
        ]
        input_params = integrand_param_builder.get_parameters()
        constants = self.get_constants_for_evaluator()
        full_evaluator = None
        for cff_term in cff_structure.expressions:
            orientation_substitutions = []
            for o_i, o in enumerate(cff_term.orientation):
                orientation_substitutions.append(
                    Replacement(self.SB["energySign"](self.SB["o_id"], E(str(o_i))), E("-1") if o.is_reversed() else E("1"))
                )
            orientation_substituted_integrand = integrand_expression.replace_multiple(orientation_substitutions)
            for _f_i, f in enumerate(cff_term.families):
                e_surface_selector_substitutions = []
                for eta_i, is_present in enumerate(f):
                    e_surface_selector_substitutions.append(
                        Replacement(self.SB["etaSigma"](self.SB["o_id"], self.SB["f_id"], E(str(eta_i))), E("1") if is_present else E("0"))
                    )
                concretized_integrand = orientation_substituted_integrand.replace_multiple(e_surface_selector_substitutions)
                concretized_integrand = concretized_integrand.replace_multiple(selector_execution)
                concretized_evaluator = concretized_integrand.evaluator(
                    constants=constants,
                    functions={},  # type: ignore
                    params=input_params,
                    iterations=100,
                    n_cores=8,
                    verbose=False,
                    external_functions=None,
                    conditionals=None,
                )
                if full_evaluator is None:
                    full_evaluator = concretized_evaluator
                else:
                    full_evaluator.merge(concretized_evaluator)

        assert full_evaluator is not None
        return full_evaluator

    def build_parameter_evaluators(self, graph: DotGraph) -> tuple[Evaluator, ParamBuilder]:
        internal_edges = graph.get_internal_edges()
        external_edges = graph.get_external_edges()
        max_internal_edge_id = 0 if len(internal_edges) == 0 else max(int(e.get("id")) for e in internal_edges)
        max_external_edge_id = 0 if len(external_edges) == 0 else max(int(e.get("id")) for e in external_edges)  # noqa: F841

        lmb_projections = [Replacement(source, target) for source, target in graph.get_emr_replacements(head=self.SB["Q"].get_name())]
        computable_parameters = []
        model = self.get_model()
        edges = graph.dot.get_edges()

        # Add evaluation rules for the energies
        for e_i in range(max_internal_edge_id):
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

        # We can now build the evaluator
        # TODO @Ben: add support for conditionals
        evaluator = Expression.evaluator_multiple(
            computable_parameters,
            constants={},
            functions={},  # type: ignore
            params=param_builder.get_parameters(),
            iterations=100,
            n_cores=8,
            verbose=False,
            external_functions=None,  # type: ignore
        )
        return (evaluator, param_builder)

    def generate_spenso_code(self) -> None:
        evaluator_path = pjoin(EVALUATORS_FOLDER, self.name, f"{self.get_integrand_name()}.so")
        if os.path.isfile(evaluator_path):
            if self.clean:
                logger.info(f"Removing existing spenso evaluator {evaluator_path} and re-generating it.")
                os.remove(evaluator_path)
            else:
                logger.info(f"Spenso evaluator {evaluator_path} already generated and recycled.")
                return

        integrand_name = self.get_integrand_name()
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
            case _:
                raise pygloopException(f"Number of loops {self.n_loops} not supported.")

        gghhh_dot_graphs = DotGraphs(dot_str=self.gl_worker.get_dot_files(process_id=process_id, integrand_name=integrand_name))

        gghhh_graph = gghhh_dot_graphs.get_graph(graph_name)

        cff_structure = self.gl_worker.generate_cff_as_json_string(
            dot_string=gghhh_graph.to_string(),
            subgraph_nodes=[],
            reverse_dangling=[],
            orientation_pattern=None,
        )

        numerator = simplify_color(gghhh_graph.get_numerator() * gghhh_graph.get_projector())
        # The SM UFO stupidly writes ProjP + ProjM instead of the identity for the yukawa interaction, simply this away...
        numerator = numerator.replace(E("spenso::projm(x_,y_)+spenso::projp(x_,y_)"), E("spenso::g(x_,y_)"), repeat=True)

        hep_lib = TensorLibrary.hep_lib()
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
        # fmt: off
        #         tn_A = TensorNetwork(
        #             E("""

        # (
        #       UFO::MT*spenso::g(spenso::bis(4,gammalooprs::hedge(10)),spenso::bis(4,gammalooprs::hedge(9)))
        #     +   gammalooprs::Q(7,spenso::mink(4,gammalooprs::edge(7,1)))
        #       * spenso::gamma(spenso::bis(4,gammalooprs::hedge(10)),spenso::bis(4,gammalooprs::hedge(9)),spenso::mink(4,gammalooprs::edge(7,1)))
        # )*(
        #       UFO::MT*spenso::g(spenso::bis(4,gammalooprs::hedge(11)),spenso::bis(4,gammalooprs::hedge(12)))
        #     +   gammalooprs::Q(8,spenso::mink(4,gammalooprs::edge(8,1)))
        #       * spenso::gamma(spenso::bis(4,gammalooprs::hedge(12)),spenso::bis(4,gammalooprs::hedge(11)),spenso::mink(4,gammalooprs::edge(8,1)))
        # )*(
        #       UFO::MT*spenso::g(spenso::bis(4,gammalooprs::hedge(13)),spenso::bis(4,gammalooprs::hedge(14)))
        #     +   gammalooprs::Q(9,spenso::mink(4,gammalooprs::edge(9,1)))
        #       * spenso::gamma(spenso::bis(4,gammalooprs::hedge(14)),spenso::bis(4,gammalooprs::hedge(13)),spenso::mink(4,gammalooprs::edge(9,1)))
        # )*(
        #       UFO::MT*spenso::g(spenso::bis(4,gammalooprs::hedge(5)),spenso::bis(4,gammalooprs::hedge(6)))
        #     +   gammalooprs::Q(5,spenso::mink(4,gammalooprs::edge(5,1)))
        #       * spenso::gamma(spenso::bis(4,gammalooprs::hedge(6)),spenso::bis(4,gammalooprs::hedge(5)),spenso::mink(4,gammalooprs::edge(5,1)))
        # )*(
        #       UFO::MT*spenso::g(spenso::bis(4,gammalooprs::hedge(7)),spenso::bis(4,gammalooprs::hedge(8)))
        #     +   gammalooprs::Q(6,spenso::mink(4,gammalooprs::edge(6,1)))
        #       * spenso::gamma(spenso::bis(4,gammalooprs::hedge(8)),spenso::bis(4,gammalooprs::hedge(7)),spenso::mink(4,gammalooprs::edge(6,1)))
        # )*(
        #     spenso::projm(spenso::bis(4,gammalooprs::hedge(13)),spenso::bis(4,gammalooprs::hedge(10)))+spenso::projp(spenso::bis(4,gammalooprs::hedge(13)),spenso::bis(4,gammalooprs::hedge(10)))
        # )*(
        #     spenso::projm(spenso::bis(4,gammalooprs::hedge(5)),spenso::bis(4,gammalooprs::hedge(14)))+spenso::projp(spenso::bis(4,gammalooprs::hedge(5)),spenso::bis(4,gammalooprs::hedge(14)))
        # )*(
        #     spenso::projm(spenso::bis(4,gammalooprs::hedge(7)),spenso::bis(4,gammalooprs::hedge(6)))+spenso::projp(spenso::bis(4,gammalooprs::hedge(7)),spenso::bis(4,gammalooprs::hedge(6)))
        # )*1𝑖*UFO::GC_11^2*UFO::GC_94^3*spenso::TR
        # *gammalooprs::ϵ(0,spenso::mink(4,gammalooprs::hedge(0)))
        # *gammalooprs::ϵ(1,spenso::mink(4,gammalooprs::hedge(1)))
        # *spenso::gamma(spenso::bis(4,gammalooprs::hedge(11)),spenso::bis(4,gammalooprs::hedge(8)),spenso::mink(4,gammalooprs::hedge(0)))
        # *spenso::gamma(spenso::bis(4,gammalooprs::hedge(9)),spenso::bis(4,gammalooprs::hedge(12)),spenso::mink(4,gammalooprs::hedge(1)))
        #             """),
        #             hep_lib,
        #         )

        try:
            cff_structure = json.loads(cff_structure)
        except json.JSONDecodeError as e:
            raise pygloopException(f"Error decoding CFF structure JSON: {e}") from e
        cff_structure = CFFStructure(cff_structure)
        # print(cff_structure.__str__(show_families=True))
        # stop

        # print(cff_structure.__str__(show_families=True))
        # Build a CFF term evaluators, with each esurface wrapped around a selector
        cff_term = E("1")
        for e_surf in cff_structure.e_surfaces:
            cff_term *= self.SB["etaSelector"](self.SB["etaSigma"](self.SB["o_id"], self.SB["f_id"], e_surf.id), e_surf.expression, E("1"))

        internal_edges = gghhh_graph.get_internal_edges()
        external_edges = gghhh_graph.get_external_edges()
        max_internal_edge_id = 0 if len(internal_edges) == 0 else max(int(e.get("id")) for e in internal_edges)
        max_external_edge_id = 0 if len(external_edges) == 0 else max(int(e.get("id")) for e in external_edges)  # noqa: F841

        # Add the CFF normalization factor
        for e_i in range(max_internal_edge_id+1):
            if e_i > max_external_edge_id:
                continue
            cff_term *= E("2")*CFFStructure.SB["E"](E(str(e_i)))

        integrand_expression = numerator_expr / cff_term

        # Build the integrand evaluator
        parametric_integrand_evaluator, integrand_param_builder = self.build_integrand_evaluator(gghhh_graph, integrand_expression, cff_structure)

        # Build the aggregated summed integrand evaluator
        full_integrand_evaluator = self.build_full_integrand_evaluator(integrand_expression,cff_structure,integrand_param_builder)

        # Now build the input parameter evaluator
        params_evaluator, params_param_builder = self.build_parameter_evaluators(gghhh_graph)

        self.initialize_param_builders(integrand_param_builder,params_param_builder)
        # Now build the param_builder for this
        # fmt: off
        self.set_from_sample(
                ks=[Vector(100.,200.,300.),],
                cff_term = cff_structure.expressions[0],
                family_id = 0,
                integrand_param_builder = integrand_param_builder,
                input_params_evaluator = params_evaluator,
                input_params_param_builder = params_param_builder
        )

        test_eval_result = parametric_integrand_evaluator.evaluate_complex(integrand_param_builder.get_values()[None, :])[0].sum()
        print("Evaluation of integrand from CFF term #0:",test_eval_result)

        self.set_from_sample(
                ks=[Vector(100.,200.,300.),],
                cff_term = None,
                family_id = None,
                integrand_param_builder = integrand_param_builder,
                input_params_evaluator = params_evaluator,
                input_params_param_builder = params_param_builder
        )

        test_eval_result = full_integrand_evaluator.evaluate_complex(integrand_param_builder.get_values()[None, :])[0].sum()
        print("Evaluation of complete integrand using the aggregated evaluator term #0:", test_eval_result)

        # fmt: on

        # raise NotImplementedError("Implement spenso code generation.")

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

        self.gl_worker.run(f"generate existing -p {amplitudes[process_graphs_name]} -i {integrand_name}")

        with open(os.path.isfile(pjoin(PYGLOOP_FOLDER, "outputs", "gammaloop_states", self.name, f"{integrand_name}_is_generated")), "w") as f:
            f.write("generated")

        self.save_state()

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
        integrand_implementation: str,
        phase: str,
        multi_channeling: bool | int = True,
    ) -> float:
        try:
            if multi_channeling is False:
                k, jac = self.parameterize(xs, parameterization)
                wgt = self.integrand([k], integrand_implementation)
                if phase == "real":
                    wgt = wgt.real
                else:
                    wgt = wgt.imag
                final_wgt = wgt * jac
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

    def integrand(self, loop_momenta: list[Vector], integrand_implementation: str) -> complex:
        try:
            match integrand_implementation:
                case "spenso":
                    return self.spenso_integrand(loop_momenta)
                case "gammaloop":
                    return self.gammaloop_integrand(loop_momenta)
                case _:
                    raise pygloopException(f"Integrand implementation {integrand_implementation} not implemented.")
        except ZeroDivisionError:
            logger.debug(
                f"Integrand divided by zero for ks = [{Colour.BLUE}{
                    ','.join('[' + ', '.join(f'{ki:+.16e}' for ki in k.to_list()) + ']' for k in loop_momenta)
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

    def spenso_integrand(self, loop_momentum: list[Vector]) -> complex:
        raise NotImplementedError("Implement spenso integrand.")

    def integrate(
        self,
        integrator: str,
        parameterisation: str,
        integrand_implementation: str,
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

    @set_gammaloop_level(logging.ERROR, logging.INFO)
    def naive_integrator(
        self,
        parameterisation: str,
        integrand_implementation: str,
        target,
        **opts,
    ) -> IntegrationResult:
        return run_naive_integrator(self, parameterisation, integrand_implementation, target, **opts)

    @set_gammaloop_level(logging.ERROR, logging.INFO)
    def vegas_integrator(
        self,
        parameterisation: str,
        integrand_implementation: str,
        _target,
        **opts,
    ) -> IntegrationResult:
        return run_vegas_integrator(self, parameterisation, integrand_implementation, _target, **opts)

    @set_gammaloop_level(logging.ERROR, logging.INFO)
    def symbolica_integrator(
        self,
        parameterisation: str,
        integrand_implementation: str,
        target,
        **opts,
    ) -> IntegrationResult:
        return run_symbolica_integrator(self, parameterisation, integrand_implementation, target, **opts)

    @set_gammaloop_level(logging.ERROR, logging.INFO)
    def plot(self, **opts):
        return plot_integrand(self, **opts)
