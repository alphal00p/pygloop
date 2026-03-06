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
from itertools import product  # noqa: F401
from pprint import pformat, pprint  # noqa: F401
from typing import Any, Callable

import numpy as np
import progressbar  # pyright: ignore
import vegas  # type: ignore

# fmt: off
from gammaloop import ( GammaLoopAPI, LogLevel, evaluate_graph_overall_factor, git_version )  # isort: skip # type: ignore # noqa: F401
# fmt: on
from matplotlib.typing import CapStyleType, ColorType  # noqa: F401 # pyright: ignore
from symbolica import E, Expression, NumericalIntegrator, Sample  # pyright: ignore
from symbolica.community.idenso import (  # noqa: F401 # pyright: ignore
    simplify_color,
    simplify_gamma,
    simplify_metrics,
)
from symbolica.community.spenso import *  # noqa: F403 # type: ignore

from processes.dy.dy_classes import DYDotGraphs, VacuumDotGraph  # noqa: F401
from processes.dy.dy_evaluators import (
    DYCompiledBundle,
    compile_integrands,
    evaluate_integrand,
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
        skip_ps_validation: bool = False,
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

        self.skip_ps_validation = bool(skip_ps_validation)
        if not self.skip_ps_validation:
            self.valide_ps_point()

        self.e_cm = math.sqrt(abs((self.ps_point[0] + self.ps_point[1]).squared()))

        gl_states_folder = pjoin(GAMMALOOP_STATES_FOLDER, self.name)
        self.clean = clean
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
            log_file_name=self.name,
            log_level=gl_log_level,
        )
        self.set_log_level(logger_level)

        if toml_config_path is None:
            toml_config_path = pjoin(CONFIGS_FOLDER, self.name, "generate.toml")

        self.toml_config_path = toml_config_path
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

        if runtime_toml_config_path is None:
            runtime_toml_config_path = pjoin(CONFIGS_FOLDER, self.name, "runtime.toml")
        self.runtime_toml_config_path = runtime_toml_config_path

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
        integrand_name = self.get_integrand_name()
        bundle_dir = pjoin(EVALUATORS_FOLDER, self.name, integrand_name)
        bundle_metadata = pjoin(bundle_dir, DYCompiledBundle.METADATA_FILE)
        if os.path.exists(bundle_metadata):
            try:
                self.compiled_bundle = DYCompiledBundle.load(self.name, integrand_name)
                logger.info(
                    f"Loaded compiled DY bundle from {Colour.GREEN}{bundle_dir}{Colour.END}"
                )
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
            clean=False,
            logger_level=logging.CRITICAL,
            skip_ps_validation=self.skip_ps_validation,
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
            self.skip_ps_validation,
        )

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
        return E(
            "spenso::g(spenso::cof(3,gammalooprs::hedge(1)),spenso::dind(spenso::cof(3,gammalooprs::hedge(3))))*spenso::g(spenso::cof(3,gammalooprs::hedge(0)),spenso::dind(spenso::cof(3,gammalooprs::hedge(2))))"
        )

    def get_spin_projector(self) -> Expression:
        return E(
            "spenso::gamma(spenso::bis(4,gammalooprs::hedge(0)),spenso::bis(4,gammalooprs::hedge(2)),spenso::mink(4,mu))*gammalooprs::Q(0,spenso::mink(4,mu))*spenso::gamma(spenso::bis(4,gammalooprs::hedge(3)),spenso::bis(4,gammalooprs::hedge(1)),spenso::mink(4,nu))*gammalooprs::Q(1,spenso::mink(4,nu))"
        )

    def process_1L_generated_graphs(self, graphs: DYDotGraphs) -> DYDotGraphs:
        processed_graphs = DYDotGraphs()

        filtered_graphs = DYDotGraphs()
        filtered_graphs.extend(copy.deepcopy(graphs.filter_particle_definition(["a"])))

        processor = EMRIntegrandConstructor([], "DY", 1)
        loop_processor = LoopIntegrandConstructor([], "DY", 1)

        for graph in filtered_graphs:
            g = copy.deepcopy(graph)
            print("generator graph")
            print(g.dot)
            vacuum_g = g.get_vacuum_graph()
            print("vacuum graph")
            print(vacuum_g.dot)
            _cuts = vacuum_g.get_cutkosky_cuts()
            routed_graphs = vacuum_g.cut_graphs_with_routing_leading_virtuality(
                [], ["a"]
            )
            routed_integrands = []
            evaluators = []
            for gg in routed_graphs:
                # print(gg[3])
                processed_graphs.append(gg[3])
                cut_graph = deepcopy(routed_cut_graph(gg[3], gg[0], gg[1], gg[2]))
                if len(gg[2][1]) == 1 and len(gg[2][0]) == 1:
                    print(cut_graph.graph.get_name())
                    print(cut_graph.graph)
                    term_integrands = loop_processor.get_integrand(deepcopy(cut_graph))

                    # Keep full routed list for approach/IR/UV tests
                    routed_integrands.extend(deepcopy(term_integrands))

                    observable_params = {"zmin": 0.0, "zmax": 1.0}

                    # Build one evaluator per routed term
                    evaluators.extend(
                        evaluate_integrand(
                            1,
                            "DY",
                            deepcopy(term_integrand),
                            n_hornerscheme_iterations=1000,  # increase
                            n_cpe_iterations=10000,
                            observable_params=observable_params,
                        )
                        for term_integrand in term_integrands
                    )

            approach_limit = approach_point(1, "DY", routed_integrands)
            print("##################")
            z = 0.6
            ks = [math.sqrt(z) * np.array([0, 1 / math.sqrt(2), 1 / math.sqrt(2)])]
            ks = [
                math.sqrt(z)
                * np.array([1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)])
            ]
            vp = np.array([0.3, -0.2, 0.11])
            p1 = np.array([0, 0, 1])
            p2 = np.array([0, 0, -1])
            #
            approach_limit.approach(ks, p1, p2, z, vp)

            # ir_test = infrared_test(1, "DY", routed_integrands)
            # print("##################")
            # ir_test.approach_limits(1)

            # uv_test = ultraviolet_test(1, "DY", routed_integrands)
            # print("##################")
            # uv_test.approach_limits(1)

            my_compiler = compile_integrands(
                1, "DY", self.get_integrand_name(), "z", evaluators
            )
            my_compiler.save_compiled_integrand()

        print("n routed:", len(processed_graphs))
        return processed_graphs

    def process_2L_generated_graphs(self, graphs: DYDotGraphs) -> DYDotGraphs:
        processed_graphs = DYDotGraphs()
        for g_input in graphs:
            g = copy.deepcopy(g_input)
            attrs = g.get_attributes()
            attrs["num"] = f'"{expr_to_string(g.get_numerator())}"'
            attrs["projector"] = (
                f'"{expr_to_string(g.get_projector() * self.get_color_projector())}"'
            )

            g.set_local_numerators_to_one()
            processed_graphs.append(g)

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
                # self.gl_worker.run(  # GL09
                #    f"generate amp d d~ > d d~ | d d~ g a QED==2 [{{1}}] --only-diagrams --numerator-grouping only_detect_zeroes --select-graphs GL02 -p {base_name} -i {graphs_process_name}"
                # )

                self.gl_worker.run(  ## GL04
                    f"generate amp d g > d g | d d~ g a QED==2 [{{1}}] --only-diagrams --numerator-grouping only_detect_zeroes --select-graphs GL11 -p {base_name} -i {graphs_process_name}"  #
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
                self.gl_worker.run(
                    f"generate amp g g > h h h | g h t t~ QED==3 [{{2}}] --only-diagrams --numerator-grouping only_detect_zeroes --veto-vertex-interactions V_6 V_9 V_36 V_37 --number-of-fermion-loops 1 1 --select-graphs GL303 -p {base_name} -i {graphs_process_name}"
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
            t0 = time.perf_counter()

            impl = dict(integrand_implementation)
            expects_z = impl.get("integrand_type") == "zenos"
            jac_z = 1
            if expects_z:
                if len(xs) != 4:
                    raise pygloopException(
                        f"Integrand '{impl['integrand_type']}' expects 4 variables [xk0,xk1,xk2,xz], got {len(xs)}."
                    )
                k_xs = xs[:3]
                x_z = xs[3]

                z_min = float(impl.get("z_min", 0.0))
                z_max = float(impl.get("z_max", 1.0))
                if not (0.0 <= z_min < z_max <= 1.0):
                    raise pygloopException(f"Invalid z range [{z_min}, {z_max}]")

                z_sample = x_z / (1 - x_z)
                jac_z = 1 / (1 - x_z) ** 2

                impl["z"] = z_sample
            else:
                if len(xs) != 3:
                    raise pygloopException(
                        f"Integrand '{impl['integrand_type']}' expects 3 variables [xk0,xk1,xk2], got {len(xs)}."
                    )
                k_xs = xs
                z_sample = None

            k, jac_k = self.parameterize(k_xs, parameterization)

            t1 = time.perf_counter()

            print("-" * 15)
            print("parametrisation time:", t1 - t0)

            wgt = self.integrand([k], impl)

            t2 = time.perf_counter()

            print("overall integrand time:", t2 - t1)

            wgt = wgt.real if phase == "real" else wgt.imag
            final_wgt = wgt * jac_k * jac_z

            print("res")
            print(xs)
            print(final_wgt)

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
        self, loop_momenta: list[Vector], integrand_implementation: dict[str, Any]
    ) -> complex:
        integrand_implementation = self._normalize_integrand_implementation(
            integrand_implementation
        )
        try:
            match integrand_implementation["integrand_type"]:
                case "spenso":
                    return self.spenso_integrand(loop_momenta)
                case "zenos":
                    return self.zenos_integrand(loop_momenta, integrand_implementation)
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
    ) -> complex:
        if self.compiled_bundle is None:
            raise pygloopException(
                f"No compiled DY bundle loaded for integrand '{self.get_integrand_name()}'."
            )

        p1 = self.ps_point[0].spatial()
        p2 = self.ps_point[1].spatial()
        z = 1.0
        m_uv = 1.0
        if integrand_implementation is not None:
            z = float(integrand_implementation.get("z", z))
            m_uv = float(integrand_implementation.get("mUV", m_uv))

        return self.compiled_bundle.evaluate(loop_momentum, p1, p2, z, m_uv)

    def _normalize_integrand_implementation(
        self, integrand_implementation: dict[str, Any] | str
    ) -> dict[str, Any]:
        if isinstance(integrand_implementation, str):
            return {"integrand_type": integrand_implementation}
        return integrand_implementation

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
        n_dim = 4 if call_args[1].get("integrand_type") == "zenos" else 3
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
    ) -> tuple[int, list[float], IntegrationResult]:
        res = IntegrationResult(0.0, 0.0)
        t_start = time.time()
        all_weights = []
        process = DY(
            *process_builder_inputs, clean=False, logger_level=logging.CRITICAL
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

        return (id, all_weights, res)

    @staticmethod
    def vegas_functor(
        process: DY, res: IntegrationResult, n_cores: int, call_args: list[Any]
    ) -> Callable[[list[list[float]]], list[float]]:
        @vegas.batchintegrand
        def f(all_xs):
            all_weights = []
            if n_cores > 1:
                all_args = [
                    (process.builder_inputs(), i_chunk, all_xs_split, call_args)
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

        n_dim = 4 if integrand_implementation.get("integrand_type") == "zenos" else 3
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
    ) -> tuple[int, list[float], IntegrationResult]:
        res = IntegrationResult(0.0, 0.0)
        t_start = time.time()
        all_weights = []
        process = DY(
            *process_builder_inputs, clean=False, logger_level=logging.CRITICAL
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
            all_args = [
                (
                    process.builder_inputs(),
                    i_chunk,
                    multi_channeling,
                    [SymbolicaSample(s) for s in all_xs_split],
                    call_args,
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

        n_dim = 4 if integrand_implementation.get("integrand_type") == "zenos" else 3

        if opts["multi_channeling"]:
            integrator = NumericalIntegrator.discrete([
                NumericalIntegrator.continuous(n_dim),
                NumericalIntegrator.continuous(n_dim),
                NumericalIntegrator.continuous(n_dim),
                NumericalIntegrator.continuous(n_dim),
                NumericalIntegrator.continuous(n_dim),
            ])
        else:
            integrator = NumericalIntegrator.continuous(n_dim)

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

            # Learning rate is 1.5
            avg, err, _chi_sq = integrator.update(
                continuous_learning_rate=1.5, discrete_learning_rate=1.5
            )  # type: ignore
            integration_result.central_value = avg
            integration_result.error = err
            logger.info(
                f"... result after this iteration:\n{integration_result.str_report(target)}"
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

        # zenos uses 4 integration vars (x0,x1,x2,xz), others use 3
        n_dim = 4 if integrand_implementation.get("integrand_type") == "zenos" else 3

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
