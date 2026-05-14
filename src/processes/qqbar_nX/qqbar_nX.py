from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import shutil
import subprocess
import time
import tomllib
from decimal import Decimal, localcontext
from pprint import pformat
from typing import Any

from gammaloop import DotExportSettings, GammaLoopAPI  # type: ignore

from processes.qqbar_nX.qqbar_nX_counterterms import (
    EXACT_XI_AUXILIARY_MODE,
    build_isr_counterterm_graphs,
    identify_light_line_structure,
    minimise_edge_attributes_for_import,
)
from processes.qqbar_nX.qqbar_nX_graphs import (
    SelectionReport,
    TopologySelectorConfig,
    dot_graphs_to_string,
    graph_external_edges,
    graph_internal_edges,
    graph_name,
    parse_dot_graphs,
    select_top_pentagon_isr_graphs,
    strip_quotes,
)
from utils.utils import (
    DOTS_FOLDER,
    GAMMALOOP_STATES_FOLDER,
    Colour,
    IntegrationResult,
    logger,
    pygloopException,
    write_text_with_dirs,
)
from utils.vectors import LorentzVector, Vector

pjoin = os.path.join


PROCESS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = pjoin(PROCESS_DIR, "config.toml")
INSPECT_RESULT_RE = re.compile(
    r"integrand result\s*│\s*"
    r"(?P<re>[+-]?(?:\d+(?:\.\d*)?|\.\d+)e[+-]?\d+)\s+"
    r"(?P<im>[+-]?(?:\d+(?:\.\d*)?|\.\d+)e[+-]?\d+)i",
    re.IGNORECASE,
)
ACCEPTED_EVENTS_RE = re.compile(r"accepted events\s*│\s*(?P<count>\d+)")
LMB_TERM_RE = re.compile(
    r"(?P<coef>[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)\*)?)"
    r"(?P<kind>[KP])\((?P<index>\d+),"
)
SINGLE_EXTERNAL_RE = re.compile(r"^P\((?P<index>\d+),a___\)$")


def _load_toml(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise pygloopException(f"qqbar_nX config file not found: {path}")
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _is_set(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return False
    return True


def _override(cli_value: Any, config_value: Any) -> Any:
    return config_value if _is_set(config_value) else cli_value


def _as_list(value: Any, *, name: str) -> list[Any]:
    if isinstance(value, list):
        return value
    raise pygloopException(f"qqbar_nX config entry '{name}' must be a list.")


def _optional_bool(value: Any, *, name: str) -> bool | None:
    if not _is_set(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    raise pygloopException(
        f"qqbar_nX config entry '{name}' must be true, false, or \"none\"."
    )


def _format_cli_float(value: float) -> str:
    """GammaLoop's CLI treats negative scientific notation after -x as an option."""
    text = f"{float(value):.16f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def _momenta_toml_array(momenta: list[list[float]], *, dependent_last: bool) -> str:
    entries = [
        "[{:.16e}, {:.16e}, {:.16e}, {:.16e}]".format(*momentum)
        for momentum in (momenta[:-1] if dependent_last else momenta)
    ]
    if dependent_last:
        entries.append('"dependent"')
    return ",\n    ".join(entries)


def _helicity_label(helicities: list[int]) -> str:
    if helicities[:2] == [1, -1]:
        return "pm"
    if helicities[:2] == [1, 1]:
        return "pp"
    return "_".join("p" if h > 0 else "m" if h < 0 else "0" for h in helicities)


def _helicity_toml_fragment(helicities: list[int]) -> str:
    return (
        "[kinematics.externals.data]\n"
        f"helicities = [{', '.join(str(h) for h in helicities)}]\n"
    )


def _lambda_label(lambda_value: float) -> str:
    return f"{lambda_value:.0e}".replace("+", "").replace("-", "m")


def _complex_from_api_value(value: Any) -> complex:
    if isinstance(value, complex):
        return value
    if hasattr(value, "re") and hasattr(value, "im"):
        return complex(float(value.re), float(value.im))
    return complex(value)


def _minkowski_square_numeric(momentum: list[float]) -> float:
    return (
        momentum[0] * momentum[0]
        - momentum[1] * momentum[1]
        - momentum[2] * momentum[2]
        - momentum[3] * momentum[3]
    )


def _parse_lmb_representation(lmb_rep: str) -> dict[str, dict[int, float]]:
    cleaned = (
        strip_quotes(lmb_rep)
        .replace(" ", "")
        .replace("+-", "-")
        .replace("-+", "-")
        .replace("++", "+")
    )
    coefficients: dict[str, dict[int, float]] = {"K": {}, "P": {}}
    for match in LMB_TERM_RE.finditer(cleaned):
        coef_text = match.group("coef")
        if coef_text in {"", "+"}:
            coefficient = 1.0
        elif coef_text == "-":
            coefficient = -1.0
        else:
            coefficient = float(coef_text.rstrip("*"))
        kind = match.group("kind")
        index = int(match.group("index"))
        coefficients[kind][index] = coefficients[kind].get(index, 0.0) + coefficient
    return coefficients


def _add_scaled(target: list[float], coefficient: float, vector: list[float]) -> None:
    for index in range(3):
        target[index] += coefficient * vector[index]


def _sub_scaled(target: list[float], coefficient: float, vector: list[float]) -> None:
    for index in range(3):
        target[index] -= coefficient * vector[index]


def _toml_command_string(command: str) -> str:
    if "\n" not in command:
        return json.dumps(command)
    escaped = command.replace("\\", "\\\\").replace('"""', '\\"""')
    return f'"""{escaped}"""'


class qqbar_nX(object):
    """Process driver for q qbar -> n colorless bosons via a massive top loop."""

    name = "qqbar_nX"
    name_for_resources = "qqbar_nX"

    def __init__(
        self,
        m_top: float,
        m_higgs: float,
        ps_point: list[LorentzVector],
        helicities: list[int] | None = None,
        n_loops: int = 2,
        toml_config_path: str | None = None,
        runtime_toml_config_path: str | None = None,
        clean: bool = True,
        logger_level: int | None = None,
        gammaloop_settings: list[str] | None = None,
        qqbar_nx_config_path: str | None = None,
        final_state: list[str] | None = None,
        skip_gl_worker_init: bool = False,
        **_opts: Any,
    ):
        start_logger_level = logger.getEffectiveLevel()
        if logger_level is not None:
            logger.setLevel(logger_level)

        self.config_path = qqbar_nx_config_path or DEFAULT_CONFIG_PATH
        self.config = _load_toml(self.config_path)
        cli_overrides = self.config.get("cli_overrides", {})

        self.m_top = float(_override(m_top, cli_overrides.get("m_top")))
        self.m_higgs = float(_override(m_higgs, cli_overrides.get("m_higgs")))
        self.n_loops = int(_override(n_loops, cli_overrides.get("n_loops")))
        self.clean = bool(_override(clean, cli_overrides.get("clean")))
        self.toml_config_path = _override(
            toml_config_path, cli_overrides.get("toml_config_path")
        )
        self.runtime_toml_config_path = _override(
            runtime_toml_config_path, cli_overrides.get("runtime_toml_config_path")
        )
        self.gammaloop_settings = _override(
            gammaloop_settings, cli_overrides.get("gammaloop_settings")
        )

        process_config = self.config.get("process", {})
        self.initial_state = tuple(process_config.get("initial_state", ["d", "d~"]))
        configured_final_state = _override(
            final_state, process_config.get("final_state", ["h", "h", "h"])
        )
        if configured_final_state is None:
            configured_final_state = ["h", "h", "h"]
        self.final_state = tuple(configured_final_state)
        self.allowed_particles = tuple(
            process_config.get(
                "allowed_particles",
                ["d", "d~", "g", "h", "t", "t~"],
            )
        )
        self.model = process_config.get("model", "sm-default.json")
        self.qed_order = int(process_config.get("qed_order", 3))
        self.qcd_coupling_filter = _override(
            None, process_config.get("qcd_coupling_filter")
        )

        self.ps_point = ps_point
        self.helicities = list(
            _override(helicities or [1, -1, 0, 0, 0], cli_overrides.get("helicities"))
        )

        generation_config = self.config.get("generation", {})
        self.numerator_grouping = generation_config.get(
            "numerator_grouping", "no_grouping"
        )
        self.graph_prefix = generation_config.get("graph_prefix", "GL")
        self.max_multiplicity_for_fast_cut_filter = int(
            generation_config.get("max_multiplicity_for_fast_cut_filter", 99)
        )
        self.global_prefactor_projector = _override(
            None, generation_config.get("global_prefactor_projector")
        )
        self.veto_vertex_interactions = tuple(
            generation_config.get(
                "veto_vertex_interactions", ["V_6", "V_9", "V_36", "V_37"]
            )
        )
        self.number_of_fermion_loops = tuple(
            generation_config.get("number_of_fermion_loops", [1, 1])
        )
        self.generate_existing_after_selection = bool(
            generation_config.get("generate_existing_after_selection", False)
        )
        symmetrize_final_states = _override(
            None, generation_config.get("symmetrize_final_states")
        )
        self.symmetrize_final_states = _optional_bool(
            symmetrize_final_states, name="generation.symmetrize_final_states"
        )

        counterterm_config = self.config.get("counterterms", {})
        self.build_isr_counterterms = bool(counterterm_config.get("build_isr", True))
        self.import_subtracted_dot = bool(
            counterterm_config.get("import_subtracted_dot", True)
        )
        self.subtracted_suffix = counterterm_config.get(
            "subtracted_suffix", "_top_pentagon_isr_subtracted"
        )
        self.counterterm_projector_mode = counterterm_config.get(
            "projector_mode", "leading"
        )
        self.counterterm_denominator_strategy = counterterm_config.get(
            "denominator_strategy", "contract"
        )
        self.counterterm_auxiliary_denominator_mode = counterterm_config.get(
            "auxiliary_denominator_mode", "global"
        )
        self.counterterm_global_phase = str(counterterm_config.get("global_phase", "1"))
        self.counterterm_normalization_factor = str(
            counterterm_config.get("normalization_factor", "1")
        )
        self.counterterm_uv_inert_dod = int(counterterm_config.get("uv_inert_dod", -100))
        self.counterterm_use_parametric_xi = bool(
            counterterm_config.get("use_parametric_xi", False)
        )
        self.xi_parameter_names = tuple(
            counterterm_config.get(
                "xi_parameter_names", ["xi0", "xi1", "xi2", "xi3"]
            )
        )
        self.xi_default_values = tuple(
            float(value)
            for value in counterterm_config.get(
                "xi_default_values", [1000.0, 0.0, 0.0, 100.0]
            )
        )
        if len(self.xi_parameter_names) != 4:
            raise pygloopException("counterterms.xi_parameter_names must contain 4 names.")
        if len(self.xi_default_values) != 4:
            raise pygloopException("counterterms.xi_default_values must contain 4 numbers.")

        standalone_config = self.config.get("standalone_run_card", {})
        self.write_standalone_run_card = bool(
            standalone_config.get("write_run_card", True)
        )
        self.low_stat_n_start = int(standalone_config.get("low_stat_n_start", 16))
        self.low_stat_n_max = int(standalone_config.get("low_stat_n_max", 64))
        self.low_stat_n_cores = int(standalone_config.get("low_stat_n_cores", 1))
        self.low_stat_batch_size = int(standalone_config.get("low_stat_batch_size", 16))
        self.evaluator_summed_function_map = bool(
            standalone_config.get("evaluator_summed_function_map", False)
        )
        self.enable_threshold_subtraction = bool(
            standalone_config.get("enable_threshold_subtraction", False)
        )
        self.check_esurface_at_generation = bool(
            standalone_config.get("check_esurface_at_generation", False)
        )
        self.assume_positive_external_energies = bool(
            standalone_config.get("assume_positive_external_energies", True)
        )

        standalone_test_config = self.config.get("standalone_tests", {})
        configured_gammaloop_cli = standalone_test_config.get(
            "gammaloop_cli_path", "gammaloop"
        )
        self.gammaloop_cli_path = (
            configured_gammaloop_cli if _is_set(configured_gammaloop_cli) else "gammaloop"
        )
        self.standalone_test_timeout = int(
            standalone_test_config.get("command_timeout_seconds", 600)
        )
        self.run_low_stat_standalone_test = bool(
            standalone_test_config.get("run_low_stat_integration", True)
        )

        test_config = self.config.get("tests", {})
        self.smoke_test_original_graphs = int(
            test_config.get("smoke_test_original_graphs", 1)
        )
        self.smoke_test_suffix = test_config.get(
            "smoke_test_suffix", "_top_pentagon_isr_smoke"
        )
        self.collinear_fraction_x = float(
            test_config.get("collinear_fraction_x", 0.3)
        )
        self.collinear_precision = str(
            test_config.get("collinear_precision", "Double")
        )
        allowed_collinear_precisions = {"Double", "Quad", "ArbPrec"}
        if self.collinear_precision not in allowed_collinear_precisions:
            raise pygloopException(
                "tests.collinear_precision must be one of "
                f"{sorted(allowed_collinear_precisions)}."
            )
        self.collinear_lambdas = tuple(
            float(value)
            for value in test_config.get(
                "collinear_lambdas",
                [1.0, 0.1, 0.01, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7],
            )
        )
        if not self.collinear_lambdas:
            raise pygloopException("tests.collinear_lambdas must not be empty.")
        self.collinear_arb_display_digits = int(
            test_config.get("collinear_arb_display_digits", 50)
        )
        self.four_d_counterterm_global_phase = str(
            test_config.get("four_d_global_phase", "1")
        )
        self.cff_meta_graph_name = str(test_config.get("cff_meta_graph_name", "GL05"))
        raw_cff_orientation = test_config.get("cff_meta_orientation_id", 0)
        self.cff_meta_orientation_id = (
            None
            if str(raw_cff_orientation).lower() == "none"
            else int(raw_cff_orientation)
        )

        selection_config = self.config.get("selection", {})
        vertex_ids = selection_config.get("vertex_ids", {})
        self.selector_config = TopologySelectorConfig(
            initial_state=tuple(self.initial_state),  # type: ignore[arg-type]
            final_state=tuple(self.final_state),
            light_quark_gluon_vertex_id=vertex_ids.get(
                "light_quark_gluon", "V_74"
            ),
            top_gluon_vertex_id=vertex_ids.get("top_gluon", "V_137"),
            top_higgs_vertex_id=vertex_ids.get("top_higgs", "V_141"),
        )

        self.skip_gl_worker_init = bool(skip_gl_worker_init)
        self.gl_worker: GammaLoopAPI | None = None
        self.cache: dict[str, Any] = {}
        self.last_selection_report: SelectionReport | None = None

        if not self.skip_gl_worker_init:
            self._initialise_gl_worker()

        logger.setLevel(start_logger_level)

    def __deepcopy__(self, memo: dict[int, Any]) -> "qqbar_nX":
        return qqbar_nX(
            self.m_top,
            self.m_higgs,
            copy.deepcopy(self.ps_point, memo),
            copy.deepcopy(self.helicities, memo),
            self.n_loops,
            self.toml_config_path,
            self.runtime_toml_config_path,
            clean=False,
            logger_level=logging.CRITICAL,
            gammaloop_settings=copy.deepcopy(self.gammaloop_settings, memo),
            qqbar_nx_config_path=self.config_path,
            final_state=list(self.final_state),
        )

    def builder_inputs(self) -> tuple:
        return (
            self.m_top,
            self.m_higgs,
            self.ps_point,
            self.helicities,
            self.n_loops,
            self.toml_config_path,
            self.runtime_toml_config_path,
            self.config_path,
            list(self.final_state),
        )

    def set_log_level(self, level: int | None) -> None:
        if level is not None:
            logger.setLevel(level)

    @property
    def state_folder(self) -> str:
        return pjoin(GAMMALOOP_STATES_FOLDER, self.name)

    @property
    def dot_folder(self) -> str:
        return pjoin(DOTS_FOLDER, self.name)

    def get_integrand_name(self, suffix: str = "_selected_grouped") -> str:
        initial = "_".join(p.replace("~", "bar") for p in self.initial_state)
        final = "_".join(self.final_state)
        return f"{self.name}_{initial}_{final}_{self.n_loops}L{suffix}"

    def get_subtracted_integrand_name(self) -> str:
        return self.get_integrand_name(suffix=self.subtracted_suffix)

    def uses_fake_xi_externals(self) -> bool:
        return self.counterterm_auxiliary_denominator_mode == EXACT_XI_AUXILIARY_MODE

    def get_subtracted_process_name(self) -> str:
        base_name = self.get_integrand_name(suffix="")
        if self.uses_fake_xi_externals():
            return f"{base_name}_xi_ext"
        return base_name

    def _xi_external_momentum(self) -> list[float]:
        if self.counterterm_use_parametric_xi:
            return [float(value) for value in self.xi_default_values]
        if len(self.ps_point) < 2:
            raise pygloopException("Need two incoming momenta to build xi=p1+p2.")
        return [
            float(self.ps_point[0].t + self.ps_point[1].t),
            float(self.ps_point[0].x + self.ps_point[1].x),
            float(self.ps_point[0].y + self.ps_point[1].y),
            float(self.ps_point[0].z + self.ps_point[1].z),
        ]

    def _fake_xi_pair_momenta(
        self, fake_xi_momentum: Any | None = None
    ) -> dict[str, list[float]]:
        default_xi = self._xi_external_momentum()
        if fake_xi_momentum is None:
            return {"p1": list(default_xi), "p2": list(default_xi)}
        if isinstance(fake_xi_momentum, dict):
            p1_value = (
                fake_xi_momentum.get("p1")
                or fake_xi_momentum.get(2)
                or fake_xi_momentum.get("2")
                or fake_xi_momentum.get(4)
                or fake_xi_momentum.get("4")
                or default_xi
            )
            p2_value = (
                fake_xi_momentum.get("p2")
                or fake_xi_momentum.get(3)
                or fake_xi_momentum.get("3")
                or fake_xi_momentum.get(5)
                or fake_xi_momentum.get("5")
                or default_xi
            )
            return {"p1": list(p1_value), "p2": list(p2_value)}
        shared = list(fake_xi_momentum)
        return {"p1": shared, "p2": list(shared)}

    def _physical_momenta_4d(self) -> list[list[float]]:
        return [
            [
                float(momentum.t),
                float(momentum.x),
                float(momentum.y),
                float(momentum.z),
            ]
            for momentum in self.ps_point
        ]

    def _current_external_momenta_4d(
        self, *, fake_xi_momentum: list[float] | None = None
    ) -> list[list[float]]:
        physical = self._physical_momenta_4d()
        if not self.uses_fake_xi_externals():
            return physical
        if len(physical) != 5:
            raise pygloopException(
                "qqbar_nX fake-xi topology expects five physical external momenta."
            )
        fake_xi = self._fake_xi_pair_momenta(fake_xi_momentum)
        return [
            physical[0],
            physical[1],
            fake_xi["p1"],
            fake_xi["p2"],
            fake_xi["p1"],
            fake_xi["p2"],
            physical[2],
            physical[3],
            physical[4],
        ]

    def _fake_xi_edge_momentum_map(
        self, *, fake_xi_momentum: list[float] | None = None
    ) -> dict[int, list[float]]:
        physical = self._physical_momenta_4d()
        if len(physical) != 5:
            raise pygloopException(
                "qqbar_nX fake-xi topology expects five physical external momenta."
            )
        fake_xi = self._fake_xi_pair_momenta(fake_xi_momentum)
        return {
            0: physical[0],
            1: physical[1],
            2: fake_xi["p1"],
            3: fake_xi["p2"],
            4: fake_xi["p1"],
            5: fake_xi["p2"],
            6: physical[2],
            7: physical[3],
            8: physical[4],
        }

    def _external_momenta_for_graph_4d(
        self, graph: Any, *, fake_xi_momentum: list[float] | None = None
    ) -> list[list[float]]:
        momenta = [
            list(momentum)
            for momentum in self._current_external_momenta_4d(
                fake_xi_momentum=fake_xi_momentum
            )
        ]
        if not self.uses_fake_xi_externals():
            return momenta

        edge_momenta = self._fake_xi_edge_momentum_map(
            fake_xi_momentum=fake_xi_momentum
        )
        for edge in graph_external_edges(graph):
            raw_edge_id = edge.get_attributes().get("id")
            if raw_edge_id is None:
                continue
            edge_id = int(strip_quotes(raw_edge_id))
            if edge_id not in edge_momenta:
                continue
            lmb_rep = strip_quotes(edge.get_attributes().get("lmb_rep", ""))
            match = SINGLE_EXTERNAL_RE.fullmatch(lmb_rep.replace(" ", ""))
            if match is None:
                continue
            p_index = int(match.group("index"))
            if p_index >= len(momenta):
                momenta.extend(
                    [[0.0, 0.0, 0.0, 0.0] for _ in range(p_index + 1 - len(momenta))]
                )
            momenta[p_index] = list(edge_momenta[edge_id])
        return momenta

    def _external_spatial_vectors_for_graph(
        self,
        graph: Any | None = None,
        *,
        fake_xi_momentum: list[float] | None = None,
    ) -> dict[int, list[float]]:
        external_momenta = (
            self._current_external_momenta_4d(fake_xi_momentum=fake_xi_momentum)
            if graph is None
            else self._external_momenta_for_graph_4d(
                graph, fake_xi_momentum=fake_xi_momentum
            )
        )
        return {
            index: [momentum[1], momentum[2], momentum[3]]
            for index, momentum in enumerate(external_momenta)
        }

    def _current_external_spatial_vectors(self) -> dict[int, list[float]]:
        return self._external_spatial_vectors_for_graph()

    def _helicities_for_current_externals(
        self, helicities: list[int] | None = None
    ) -> list[int]:
        values = list(self.helicities if helicities is None else helicities)
        if not self.uses_fake_xi_externals():
            return values
        if len(values) == 9:
            return values
        if len(values) != 5:
            raise pygloopException(
                "qqbar_nX fake-xi topology expects helicities for either five "
                "physical externals or all nine DOT externals."
            )
        return [values[0], values[1], 0, 0, 0, 0, values[2], values[3], values[4]]

    def _initialise_gl_worker(self) -> None:
        if os.path.exists(self.state_folder):
            if self.clean:
                logger.info(
                    f"Removing existing GammaLoop state in {Colour.GREEN}{self.state_folder}{Colour.END}"
                )
                shutil.rmtree(self.state_folder)
            else:
                logger.info(
                    f"Reusing existing GammaLoop state in {Colour.GREEN}{self.state_folder}{Colour.END}"
                )

        logger.info(
            f"Initializing GammaLoop API for process {Colour.GREEN}{self.name}{Colour.END}"
        )
        self.gl_worker = GammaLoopAPI(self.state_folder, clean_state=False)
        self.set_model()
        self.set_default_runtime_settings()
        if self.gammaloop_settings is not None:
            for setting_command in self.gammaloop_settings:
                logger.info(
                    f"Applying custom GammaLoop setting: {Colour.GREEN}{setting_command}{Colour.END}."
                )
                self.require_gl_worker().run(setting_command)
        self.save_state()

    def require_gl_worker(self) -> GammaLoopAPI:
        if self.gl_worker is None:
            self._initialise_gl_worker()
        assert self.gl_worker is not None
        return self.gl_worker

    def set_model(self) -> None:
        worker = self.require_gl_worker()
        worker.run(f"import model {self.model}")
        worker.run("set model MT={:.16f}".format(self.m_top))
        worker.run("set model MH={:.16f}".format(self.m_higgs))
        worker.run("set model WT=0.0")
        worker.run("set model WH=0.0")
        worker.run("set model ymt={:.16f}".format(self.m_top))

    def set_default_runtime_settings(self) -> None:
        if self.toml_config_path is not None:
            self.require_gl_worker().run(f"set global file {self.toml_config_path}")

    def save_state(self) -> None:
        self.require_gl_worker().run("save state -o")

    def build_feyngen_command(self) -> str:
        if self.n_loops != 2:
            raise pygloopException("qqbar_nX currently implements only two-loop graph generation.")
        if tuple(self.initial_state) != ("d", "d~"):
            raise pygloopException("qqbar_nX currently supports only d d~ initial state.")
        if tuple(self.final_state) != ("h", "h", "h"):
            raise pygloopException("qqbar_nX currently supports only h h h final state.")

        lhs = " ".join(self.initial_state)
        rhs = " ".join(self.final_state)
        allowed = " ".join(self.allowed_particles)
        base_name = self.get_integrand_name(suffix="")
        raw_name = self.get_integrand_name(suffix="_raw_generated_graphs")
        loop_order = f"[{{{self.n_loops}}}]"
        coupling_filter = f"QED=={self.qed_order}"
        if self.qcd_coupling_filter:
            coupling_filter = f"{coupling_filter} {self.qcd_coupling_filter}"
        veto = ""
        if self.veto_vertex_interactions:
            veto = "--veto-vertex-interactions " + " ".join(
                self.veto_vertex_interactions
            )
        fermion_loop_bounds = " ".join(str(v) for v in self.number_of_fermion_loops)
        sym_final = ""
        if self.symmetrize_final_states is not None:
            sym_final = (
                f"--symmetrize-final-states {str(bool(self.symmetrize_final_states)).lower()}"
            )
        prefactor_projector = ""
        if self.global_prefactor_projector:
            prefactor_projector = (
                f"--global-prefactor-projector '{self.global_prefactor_projector}'"
            )

        return " ".join(
            part
            for part in [
                f"generate amp {lhs} > {rhs} | {allowed} {coupling_filter} {loop_order}",
                "--only-diagrams",
                prefactor_projector,
                f"--numerator-grouping {self.numerator_grouping}",
                veto,
                f"--number-of-fermion-loops {fermion_loop_bounds}",
                f"--graph-prefix {self.graph_prefix}",
                f"--max-multiplicity-for-fast-cut-filter {self.max_multiplicity_for_fast_cut_filter}",
                sym_final,
                f"-p {base_name}",
                f"-i {raw_name}",
            ]
            if part
        )

    def generate_graphs(self, *, import_selected: bool = True) -> None:
        worker = self.require_gl_worker()
        raw_name = self.get_integrand_name(suffix="_raw_generated_graphs")
        selected_name = self.get_integrand_name()
        amplitudes, _cross_sections = worker.list_outputs()

        if raw_name not in amplitudes:
            command = self.build_feyngen_command()
            logger.info(
                f"Generating raw qqbar_nX graphs with GammaLoop command:\n{Colour.BLUE}{command}{Colour.END}"
            )
            started = time.time()
            worker.run(command)
            logger.info(
                f"Raw graph generation completed in {Colour.GREEN}{time.time() - started:.2f}s{Colour.END}."
            )
            self.save_state()
        else:
            logger.info(f"Raw graphs for amplitude {raw_name} already exist and are recycled.")

        dot_settings = DotExportSettings()
        dot_settings.include_autogenerated_fields = True
        raw_dot = worker.get_dot_files(
            process_id=None,
            integrand_name=raw_name,
            settings=dot_settings,
        )
        raw_dot_path = pjoin(self.dot_folder, f"{raw_name}.dot")
        write_text_with_dirs(raw_dot_path, raw_dot)

        report = self.select_and_group_graphs(raw_dot)
        self.last_selection_report = report
        if not report.accepted:
            manifest_path = pjoin(self.dot_folder, f"{selected_name}.manifest.json")
            write_text_with_dirs(
                manifest_path,
                json.dumps(report.manifest(), indent=2, sort_keys=True),
            )
            raise pygloopException(
                f"No qqbar_nX top-pentagon ISR graphs selected. See {manifest_path}."
            )

        selected_dot = dot_graphs_to_string([item.graph for item in report.accepted])
        selected_dot_path = pjoin(self.dot_folder, f"{selected_name}.dot")
        manifest_path = pjoin(self.dot_folder, f"{selected_name}.manifest.json")
        write_text_with_dirs(selected_dot_path, selected_dot)
        write_text_with_dirs(
            manifest_path,
            json.dumps(
                {
                    **report.manifest(),
                    "raw_dot_path": raw_dot_path,
                    "selected_dot_path": selected_dot_path,
                    "config_path": self.config_path,
                    "feyngen_command": self.build_feyngen_command(),
                },
                indent=2,
                sort_keys=True,
            ),
        )

        logger.info(
            "qqbar_nX selected %s/%s graphs into %s canonical topology groups.",
            len(report.accepted),
            len(report.accepted) + len(report.rejected),
            len(report.groups),
        )
        logger.info(
            f"Selection manifest written to {Colour.GREEN}{manifest_path}{Colour.END}."
        )

        if import_selected:
            amplitudes, _cross_sections = worker.list_outputs()
            if selected_name not in amplitudes:
                worker.run(
                    f"import graphs {selected_dot_path} -p {amplitudes[raw_name]} -i {selected_name}"
                )
                self.save_state()
            else:
                logger.info(
                    f"Selected graph amplitude {selected_name} already exists and is recycled."
                )

    def select_and_group_graphs(self, raw_dot: str) -> SelectionReport:
        graphs = parse_dot_graphs(raw_dot)
        report = select_top_pentagon_isr_graphs(graphs, self.selector_config)
        logger.debug("qqbar_nX selection report:\n%s", pformat(report.manifest()))
        return report

    def test_graph_selection(self) -> dict[str, Any]:
        self.generate_graphs()
        report = self.last_selection_report
        if report is None:
            selected_name = self.get_integrand_name()
            manifest_path = pjoin(self.dot_folder, f"{selected_name}.manifest.json")
            with open(manifest_path, encoding="utf-8") as handle:
                return json.load(handle)
        return report.manifest()

    def _standalone_momenta_toml(self) -> str:
        if self.uses_fake_xi_externals():
            return _momenta_toml_array(
                self._current_external_momenta_4d(), dependent_last=True
            )

        momenta: list[str] = []
        for momentum in self.ps_point[:-1]:
            momenta.append(
                "[{:.16e}, {:.16e}, {:.16e}, {:.16e}]".format(
                    momentum.t, momentum.x, momentum.y, momentum.z
                )
            )
        momenta.append('"dependent"')
        return ",\n    ".join(momenta)

    def _standalone_additional_param_values_toml(self) -> str:
        if not self.counterterm_use_parametric_xi:
            return ""
        values = ", ".join(f"{value:.16e}" for value in self.xi_default_values)
        return f"additional_param_values = [{values}]\n"

    def _commands_toml_array(self, commands: list[str]) -> str:
        return ",\n    ".join(_toml_command_string(command) for command in commands)

    def get_standalone_state_folder(self, integrand_name: str) -> str:
        return pjoin(GAMMALOOP_STATES_FOLDER, f"{self.name}_standalone_{integrand_name}")

    def _integration_workspace(self, integrand_name: str, helicities: list[int]) -> str:
        return pjoin(
            self.dot_folder,
            f"{integrand_name}_integration_workspace_{_helicity_label(helicities)}",
        )

    def _set_process_helicities_command(
        self, process_name: str, integrand_name: str, helicities: list[int]
    ) -> str:
        return (
            f"set process -p {process_name} -i {integrand_name} string "
            f"'\n{_helicity_toml_fragment(helicities)}'"
        )

    def _first_original_graph_from_dot(self, dot_path: str) -> Any:
        with open(dot_path, encoding="utf-8") as handle:
            graphs = parse_dot_graphs(handle.read())
        for graph in graphs:
            if not graph_name(graph).endswith("_ct"):
                return graph
        raise pygloopException(f"No original graph found in {dot_path}.")

    def _original_graphs_by_group_from_dot(self, dot_path: str) -> list[tuple[int, Any]]:
        with open(dot_path, encoding="utf-8") as handle:
            graphs = parse_dot_graphs(handle.read())
        by_group: dict[int, Any] = {}
        for graph in graphs:
            if graph_name(graph).endswith("_ct"):
                continue
            raw_group_id = graph.get_attributes().get("group_id")
            if raw_group_id is None:
                continue
            group_id = int(strip_quotes(raw_group_id))
            by_group.setdefault(group_id, graph)
        if not by_group:
            raise pygloopException(f"No grouped original graphs found in {dot_path}.")
        return sorted(by_group.items())

    def _graph_members_by_group_from_dot(self, dot_path: str) -> dict[int, list[Any]]:
        with open(dot_path, encoding="utf-8") as handle:
            graphs = parse_dot_graphs(handle.read())
        by_group: dict[int, list[Any]] = {}
        for graph in graphs:
            raw_group_id = graph.get_attributes().get("group_id")
            if raw_group_id is None:
                continue
            group_id = int(strip_quotes(raw_group_id))
            by_group.setdefault(group_id, []).append(graph)
        if not by_group:
            raise pygloopException(f"No grouped graphs found in {dot_path}.")
        return dict(sorted(by_group.items()))

    def _graph_members_by_group_from_routed_dot(
        self, *, reference_dot_path: str, routed_dot_path: str
    ) -> dict[int, list[Any]]:
        """Reuse grouping metadata from our DOT and routed momenta from GammaLoop."""
        with open(routed_dot_path, encoding="utf-8") as handle:
            routed_graphs = parse_dot_graphs(handle.read())
        routed_by_name = {graph_name(graph): graph for graph in routed_graphs}
        grouped_reference = self._graph_members_by_group_from_dot(reference_dot_path)
        grouped_routed: dict[int, list[Any]] = {}
        for group_id, reference_graphs in grouped_reference.items():
            routed_members: list[Any] = []
            for reference_graph in reference_graphs:
                name = graph_name(reference_graph)
                routed_graph = routed_by_name.get(name)
                if routed_graph is None:
                    raise pygloopException(
                        f"Routed DOT {routed_dot_path} does not contain graph {name}."
                    )
                routed_members.append(routed_graph)
            grouped_routed[group_id] = routed_members
        return grouped_routed

    def _center_of_mass_energy(self) -> float:
        if len(self.ps_point) < 2:
            raise pygloopException("Need at least two incoming momenta for the collinear test.")
        return float(self.ps_point[0].t + self.ps_point[1].t)

    def _collinear_xs_from_graph_fractional(
        self,
        graph: Any,
        *,
        beam: str,
        x_fraction: float,
        lambda_value: float,
        routed_graph: Any | None = None,
        fake_xi_momentum: list[float] | None = None,
    ) -> list[float]:
        if not 0.0 < x_fraction < 1.0:
            raise pygloopException("tests.collinear_fraction_x must lie in (0, 1).")

        e_cm = self._center_of_mass_energy()
        k_perp_norm = lambda_value * e_cm
        beam_index = 0 if beam == "p1" else 1 if beam == "p2" else None
        if beam_index is None:
            raise pygloopException(f"Unknown collinear beam '{beam}'.")
        beam_momentum = self.ps_point[beam_index]
        beam_z_norm = abs(float(beam_momentum.z))
        # The paper's subtraction variables are written in terms of the
        # pinned light-line momentum k1, not the emitted gluon momentum.
        # Approach the two beam-collinear regions through the spatial
        # GammaLoop loop-momentum routing of that light edge.
        # In the all-incoming convention used by the DOT, the p2-collinear
        # light-line routing approaches k1 -> -x p2, hence the positive z
        # component for p2=(E,0,0,-E).
        z_direction = 1.0
        z_component = z_direction * x_fraction * beam_z_norm

        structure = identify_light_line_structure(graph)
        edge = structure.light_edge
        if routed_graph is not None:
            edge = self._edge_by_id(routed_graph, structure.light_edge_id)

        raw_lmb_id = edge.get_attributes().get("lmb_id")
        if routed_graph is None and raw_lmb_id is not None:
            lmb_id = int(strip_quotes(raw_lmb_id))
            spectator_loop_3d = [250.0, 125.0, -375.0]
            loop_vectors = {
                0: list(spectator_loop_3d),
                1: list(spectator_loop_3d),
            }
            loop_vectors[lmb_id] = [k_perp_norm, 0.0, z_component]
            return loop_vectors[0] + loop_vectors[1]

        lmb_rep = edge.get_attributes().get("lmb_rep")
        if lmb_rep is None:
            raise pygloopException(
                f"Cannot build collinear test point for {graph_name(graph)}: "
                "the pinned light-line edge has no lmb_rep."
            )

        coefficients = _parse_lmb_representation(lmb_rep)
        external_vectors = self._external_spatial_vectors_for_graph(
            routed_graph if routed_graph is not None else graph,
            fake_xi_momentum=fake_xi_momentum,
        )
        spectator_loop_3d = [250.0, 125.0, -375.0]
        fixed_loop_vectors = {1: list(spectator_loop_3d)}
        solve_loop_index = 0
        solve_coefficient = coefficients["K"].get(solve_loop_index, 0.0)
        if solve_coefficient == 0.0:
            solve_loop_index = 1
            solve_coefficient = coefficients["K"].get(solve_loop_index, 0.0)
            fixed_loop_vectors = {0: list(spectator_loop_3d)}
        if solve_coefficient == 0.0:
            raise pygloopException(
                f"Cannot solve collinear point for {graph_name(graph)}: "
                f"no loop momentum appears in {strip_quotes(lmb_rep)}."
            )

        rhs = [k_perp_norm, 0.0, z_component]
        for p_index, coefficient in coefficients["P"].items():
            if p_index not in external_vectors:
                raise pygloopException(
                    f"lmb_rep references unavailable external P({p_index}) "
                    f"for {graph_name(graph)}."
                )
            _sub_scaled(rhs, coefficient, external_vectors[p_index])

        for k_index, coefficient in coefficients["K"].items():
            if k_index == solve_loop_index:
                continue
            if k_index not in fixed_loop_vectors:
                fixed_loop_vectors[k_index] = list(spectator_loop_3d)
            _sub_scaled(rhs, coefficient, fixed_loop_vectors[k_index])

        solved_loop = [component / solve_coefficient for component in rhs]
        loop_vectors = dict(fixed_loop_vectors)
        loop_vectors[solve_loop_index] = solved_loop
        if 0 not in loop_vectors or 1 not in loop_vectors:
            raise pygloopException(
                f"Could not construct both loop momenta for {graph_name(graph)}."
            )
        return loop_vectors[0] + loop_vectors[1]

    def _edge_by_id(self, graph: Any, edge_id: int) -> Any:
        for edge in graph.get_edges():
            raw_edge_id = edge.get_attributes().get("id")
            if raw_edge_id is not None and int(strip_quotes(raw_edge_id)) == edge_id:
                return edge
        raise pygloopException(
            f"Could not find routed edge {edge_id} in {graph_name(graph)}."
        )

    def _exact_xi_auxiliary_edge(self, graph: Any) -> Any | None:
        if not self.uses_fake_xi_externals():
            return None
        name = graph_name(graph)
        if not (name.endswith("_isr_p1_ct") or name.endswith("_isr_p2_ct")):
            return None
        mass_edges = [
            edge
            for edge in graph_internal_edges(graph)
            if edge.get_attributes().get("mass") is not None
        ]
        if len(mass_edges) != 1:
            return None
        return mass_edges[0]

    def _exact_xi_fake_momentum_from_auxiliary_signature(
        self, graph: Any, *, shift_sign: float = -1.0
    ) -> list[float] | None:
        details = self._exact_xi_auxiliary_signature_details(
            graph, shift_sign=shift_sign
        )
        if details is None:
            return None
        return details["required_fake_momentum"]

    def _exact_xi_active_fake_edge_ids(self, graph: Any) -> set[int]:
        name = graph_name(graph)
        if name.endswith("_isr_p1_ct"):
            return {2, 4}
        if name.endswith("_isr_p2_ct"):
            return {3, 5}
        return {2, 3, 4, 5}

    def _external_single_p_index_by_edge_id(self, graph: Any) -> dict[int, int]:
        mapping: dict[int, int] = {}
        for edge in graph_external_edges(graph):
            raw_edge_id = edge.get_attributes().get("id")
            if raw_edge_id is None:
                continue
            lmb_rep = strip_quotes(edge.get_attributes().get("lmb_rep", ""))
            match = SINGLE_EXTERNAL_RE.fullmatch(lmb_rep.replace(" ", ""))
            if match is None:
                continue
            mapping[int(strip_quotes(raw_edge_id))] = int(match.group("index"))
        return mapping

    def _exact_xi_auxiliary_signature_details(
        self, graph: Any, *, shift_sign: float = -1.0
    ) -> dict[str, Any] | None:
        auxiliary_edge = self._exact_xi_auxiliary_edge(graph)
        if auxiliary_edge is None:
            return None
        raw_lmb_rep = auxiliary_edge.get_attributes().get("lmb_rep")
        if raw_lmb_rep is None:
            return None

        coefficients = _parse_lmb_representation(strip_quotes(raw_lmb_rep))
        loop_coefficients = {
            index: value
            for index, value in coefficients["K"].items()
            if abs(value) > 1.0e-12
        }
        if set(loop_coefficients) != {0}:
            return None
        loop_sign = loop_coefficients[0]
        p_index_by_edge_id = self._external_single_p_index_by_edge_id(graph)
        active_fake_edge_ids = self._exact_xi_active_fake_edge_ids(graph)
        active_fake_p_indices = {
            p_index_by_edge_id[edge_id]
            for edge_id in active_fake_edge_ids
            if edge_id in p_index_by_edge_id
        }
        if not active_fake_p_indices:
            return None
        fake_coefficient = sum(
            coefficients["P"].get(index, 0.0) for index in active_fake_p_indices
        )
        if abs(fake_coefficient) < 1.0e-12:
            return None

        graph_external_momenta = self._external_momenta_for_graph_4d(graph)
        external_momenta = {
            index: momentum for index, momentum in enumerate(graph_external_momenta)
        }
        external_shift = [0.0, 0.0, 0.0, 0.0]
        for p_index, coefficient in coefficients["P"].items():
            if p_index in active_fake_p_indices or abs(coefficient) < 1.0e-12:
                continue
            if p_index not in external_momenta:
                return None
            for component in range(4):
                external_shift[component] += coefficient * external_momenta[p_index][
                    component
                ]

        xi = self._xi_external_momentum()
        desired_shift = [shift_sign * loop_sign * component for component in xi]
        required_fake_momentum = [
            (desired_shift[component] - external_shift[component]) / fake_coefficient
            for component in range(4)
        ]
        solved_shift = list(external_shift)
        for p_index, coefficient in coefficients["P"].items():
            if p_index not in active_fake_p_indices or abs(coefficient) < 1.0e-12:
                continue
            for component in range(4):
                solved_shift[component] += coefficient * required_fake_momentum[component]
        residual = [
            solved_shift[component] - desired_shift[component]
            for component in range(4)
        ]
        return {
            "auxiliary_edge_id": int(strip_quotes(auxiliary_edge.get("id"))),
            "lmb_rep": strip_quotes(raw_lmb_rep),
            "loop_sign": loop_sign,
            "fake_coefficient": fake_coefficient,
            "external_shift": external_shift,
            "target_non_loop_shift": desired_shift,
            "solved_non_loop_shift": solved_shift,
            "residual_to_k0_minus_xi": residual,
            "max_abs_residual": max(abs(component) for component in residual),
            "active_fake_edge_ids": sorted(active_fake_edge_ids),
            "active_fake_p_indices": sorted(active_fake_p_indices),
            "required_fake_momentum": required_fake_momentum,
            "required_fake_momentum_square": _minkowski_square_numeric(required_fake_momentum),
            "xi": xi,
            "xi_square": _minkowski_square_numeric(xi),
        }

    def _exact_xi_fake_momenta_for_group(
        self, graphs: list[Any], *, shift_sign: float = -1.0
    ) -> dict[str, list[float]] | None:
        if not self.uses_fake_xi_externals():
            return None
        result: dict[str, list[float]] = {}
        for beam, suffix in (("p1", "_isr_p1_ct"), ("p2", "_isr_p2_ct")):
            for graph in graphs:
                if not graph_name(graph).endswith(suffix):
                    continue
                fake_momentum = self._exact_xi_fake_momentum_from_auxiliary_signature(
                    graph, shift_sign=shift_sign
                )
                if fake_momentum is not None:
                    result[beam] = fake_momentum
                break
        if not result:
            return None
        default_xi = self._xi_external_momentum()
        result.setdefault("p1", list(default_xi))
        result.setdefault("p2", list(default_xi))
        return result

    def _exact_xi_fake_momentum_for_group_beam(
        self, graphs: list[Any], *, beam: str, shift_sign: float = -1.0
    ) -> list[float] | None:
        if not self.uses_fake_xi_externals():
            return None
        suffix = "_isr_p1_ct" if beam == "p1" else "_isr_p2_ct"
        for graph in graphs:
            if not graph_name(graph).endswith(suffix):
                continue
            fake_momentum = self._exact_xi_fake_momentum_from_auxiliary_signature(
                graph, shift_sign=shift_sign
            )
            if fake_momentum is not None:
                return fake_momentum
        return None

    def _is_exact_xi_auxiliary_signature(self, lmb_rep: str) -> bool:
        coefficients = _parse_lmb_representation(lmb_rep)
        nonzero_loop = {
            index: value
            for index, value in coefficients["K"].items()
            if abs(value) > 1.0e-12
        }
        nonzero_external = {
            index: value
            for index, value in coefficients["P"].items()
            if abs(value) > 1.0e-12
        }
        fake_pairs = ({2, 4}, {3, 5})
        if set(nonzero_loop) != {0}:
            return False
        for fake_external_ids in fake_pairs:
            if not set(nonzero_external).issubset(fake_external_ids):
                continue
            fake_coefficient = sum(
                nonzero_external.get(index, 0.0) for index in fake_external_ids
            )
            return abs(nonzero_loop[0]) == 1.0 and nonzero_loop[0] == fake_coefficient
        return False

    def _lmb_rep_loop_coefficients(self, lmb_rep: str) -> dict[int, float]:
        coefficients = _parse_lmb_representation(lmb_rep)
        return {
            index: value
            for index, value in coefficients["K"].items()
            if abs(value) > 1.0e-12
        }

    def _lmb_rep_external_shift_4d(
        self, graph: Any, lmb_rep: str
    ) -> list[float] | None:
        coefficients = _parse_lmb_representation(lmb_rep)
        external_momenta = self._external_momenta_for_graph_4d(graph)
        shift = [0.0, 0.0, 0.0, 0.0]
        for p_index, coefficient in coefficients["P"].items():
            if abs(coefficient) < 1.0e-12:
                continue
            if p_index >= len(external_momenta):
                return None
            for component in range(4):
                shift[component] += coefficient * external_momenta[p_index][component]
        return shift

    def _verify_exact_xi_routing(
        self, grouped_routed_graphs: dict[int, list[Any]]
    ) -> list[dict[str, Any]]:
        if not self.uses_fake_xi_externals():
            return []

        report: list[dict[str, Any]] = []
        for group_id, graphs in grouped_routed_graphs.items():
            for graph in graphs:
                name = graph_name(graph)
                if not (name.endswith("_isr_p1_ct") or name.endswith("_isr_p2_ct")):
                    continue
                mass_edges = [
                    edge
                    for edge in graph_internal_edges(graph)
                    if edge.get_attributes().get("mass") is not None
                ]
                if len(mass_edges) != 1:
                    report.append(
                        {
                            "group_id": group_id,
                            "graph": name,
                            "auxiliary_edge_id": None,
                            "lmb_rep": None,
                            "matches_k1_minus_xi": False,
                            "reason": (
                                "expected exactly one massive fake-xi auxiliary "
                                f"edge, found {len(mass_edges)}"
                            ),
                        }
                    )
                    continue
                auxiliary_edge = mass_edges[0]
                raw_lmb_rep = auxiliary_edge.get_attributes().get("lmb_rep", "")
                lmb_rep = strip_quotes(raw_lmb_rep)
                original_name = (
                    name.removesuffix("_isr_p1_ct")
                    if name.endswith("_isr_p1_ct")
                    else name.removesuffix("_isr_p2_ct")
                )
                auxiliary_edge_id = int(strip_quotes(auxiliary_edge.get("id")))
                reference_lmb_rep = None
                loop_residual: dict[int, float] | None = None
                shift_residual = None
                direct_matches = False
                reason = None
                reference_edges = [
                    edge
                    for edge in graph_internal_edges(graph)
                    if strip_quotes(edge.get_attributes().get("lmb_id", "")) == "0"
                ]
                if len(reference_edges) != 1:
                    reason = (
                        "expected exactly one CT light edge with lmb_id=0, "
                        f"found {len(reference_edges)}"
                    )
                else:
                    reference_edge = reference_edges[0]
                    reference_lmb_rep = strip_quotes(reference_edge.get_attributes().get("lmb_rep", ""))
                    ct_loop = self._lmb_rep_loop_coefficients(lmb_rep)
                    reference_loop = self._lmb_rep_loop_coefficients(reference_lmb_rep)
                    all_loop_ids = set(ct_loop) | set(reference_loop)
                    loop_residual = {
                        index: ct_loop.get(index, 0.0) - reference_loop.get(index, 0.0)
                        for index in sorted(all_loop_ids)
                    }
                    ct_shift = self._lmb_rep_external_shift_4d(graph, lmb_rep)
                    reference_shift = self._lmb_rep_external_shift_4d(
                        graph, reference_lmb_rep
                    )
                    if ct_shift is None or reference_shift is None:
                        reason = "could not evaluate external shift from lmb_rep"
                    else:
                        xi = self._xi_external_momentum()
                        shift_difference = [
                            ct_shift[component] - reference_shift[component]
                            for component in range(4)
                        ]
                        target_shift = [-component for component in xi]
                        shift_residual = [
                            shift_difference[component] - target_shift[component]
                            for component in range(4)
                        ]
                        direct_matches = (
                            max((abs(value) for value in loop_residual.values()), default=0.0)
                            < 1.0e-9
                            and max(abs(value) for value in shift_residual) < 1.0e-8
                        )
                        if not direct_matches:
                            reason = (
                                "shifted edge does not equal the pinned light-edge "
                                "signature minus xi"
                            )
                entry = {
                    "group_id": group_id,
                    "graph": name,
                    "auxiliary_edge_id": auxiliary_edge_id,
                    "lmb_rep": lmb_rep,
                    "original_graph": original_name,
                    "reference_light_edge_lmb_rep": reference_lmb_rep,
                    "direct_signature_with_helpers_set_to_xi": direct_matches,
                    "loop_residual_to_light_edge": loop_residual,
                    "residual_to_light_edge_minus_xi": shift_residual,
                    "active_fake_edge_ids": sorted(
                        self._exact_xi_active_fake_edge_ids(graph)
                    ),
                    "reason": reason,
                }
                report.append(entry)

        failing = [
            entry
            for entry in report
            if not entry["direct_signature_with_helpers_set_to_xi"]
        ]
        if failing:
            logger.warning(
                "Exact-xi CT routing does not directly reproduce the paper "
                "auxiliary denominator with all helper pairs set to xi. "
                "Mismatches:\n%s",
                pformat(failing),
            )
        return report

    def _format_xs(self, xs: list[float]) -> list[str]:
        return [_format_cli_float(value) for value in xs]

    def _inspect_command(
        self,
        state_folder: str,
        process_name: str,
        integrand_name: str,
        *,
        xs: list[float],
        graph_id: int | None = None,
        discrete_dim: tuple[int, ...] | list[int] | None = None,
        use_arb_prec: bool = False,
        no_save: bool = True,
    ) -> list[str]:
        command = [self.gammaloop_cli_path]
        if no_save:
            command.append("-n")
        command.extend(
            [
                "-s",
                state_folder,
                "inspect",
                "-p",
                process_name,
                "-i",
                integrand_name,
            ]
        )
        if graph_id is not None:
            command.extend(["--graph-id", str(graph_id)])
        if discrete_dim is not None:
            command.extend(["-d", *(str(value) for value in discrete_dim)])
        if use_arb_prec:
            command.append("-f")
        command.extend(["-m", "-x", *self._format_xs(xs)])
        return command

    def _parse_inspect_log(self, log_path: str) -> dict[str, Any]:
        with open(log_path, encoding="utf-8") as handle:
            log_data = handle.read()
        result_match = INSPECT_RESULT_RE.search(log_data)
        if result_match is None:
            raise pygloopException(f"Could not parse GammaLoop inspect result in {log_path}.")
        accepted_match = ACCEPTED_EVENTS_RE.search(log_data)
        re_text = result_match.group("re")
        im_text = result_match.group("im")
        return {
            "log_path": log_path,
            "result": {
                "re": float(re_text),
                "im": float(im_text),
                "re_decimal": re_text,
                "im_decimal": im_text,
            },
            "accepted_events": int(accepted_match.group("count"))
            if accepted_match is not None
            else None,
            "nan": "│ nan                       │ true" in log_data,
            "stable": "Stable(" in log_data,
        }

    def _inspect_result_is_exact_zero(self, inspect_result: dict[str, Any]) -> bool:
        result = inspect_result["result"]
        return (
            Decimal(result["re_decimal"]) == Decimal(0)
            and Decimal(result["im_decimal"]) == Decimal(0)
        )

    def _complex_abs_decimal(self, inspect_result: dict[str, Any]) -> Decimal:
        result = inspect_result["result"]
        return max(abs(Decimal(result["re_decimal"])), abs(Decimal(result["im_decimal"])))

    def _complex_results_close(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
        *,
        abs_tol: Decimal = Decimal("1e-25"),
        rel_tol: Decimal = Decimal("1e-8"),
    ) -> bool:
        left_result = left["result"]
        right_result = right["result"]
        diffs = [
            abs(Decimal(left_result[key]) - Decimal(right_result[key]))
            for key in ("re_decimal", "im_decimal")
        ]
        scale = max(
            abs_tol,
            self._complex_abs_decimal(left),
            self._complex_abs_decimal(right),
        )
        return all(diff <= max(abs_tol, rel_tol * scale) for diff in diffs)

    def write_gammaloop_run_card(
        self, dot_path: str, *, integrand_name: str | None = None
    ) -> str:
        subtracted_name = integrand_name or self.get_subtracted_integrand_name()
        process_name = self.get_subtracted_process_name()
        run_card_path = pjoin(self.dot_folder, f"{subtracted_name}.toml")
        state_folder = self.get_standalone_state_folder(subtracted_name)
        momenta = self._standalone_momenta_toml()
        helicities = ", ".join(str(h) for h in self._helicities_for_current_externals())
        additional_param_values = self._standalone_additional_param_values_toml()
        pm_helicities = self._helicities_for_current_externals([1, -1, 0, 0, 0])
        pp_helicities = self._helicities_for_current_externals([1, 1, 0, 0, 0])
        template_process_name = "{process_name}"
        template_integrand_name = "{integrand_name}"
        template_dot_path = "{dot_path}"
        default_define_args = (
            f"-D process_name={process_name} "
            f"-D integrand_name={subtracted_name} "
            f"-D dot_path={os.path.abspath(dot_path)} "
            f"-D m_top={self.m_top:.16f} "
            f"-D m_higgs={self.m_higgs:.16f} "
            f"-D ymt={self.m_top:.16f} "
            f"-D enable_thresholds={str(self.enable_threshold_subtraction).lower()} "
            f"-D check_esurface_at_generation={str(self.check_esurface_at_generation).lower()} "
            f"-D assume_positive_external_energies={str(self.assume_positive_external_energies).lower()} "
            f"-D disable_threshold_subtraction={str((not self.enable_threshold_subtraction)).lower()}"
        )
        template_define_args = (
            "-D process_name={process_name} "
            "-D integrand_name={integrand_name} "
            "-D dot_path={dot_path} "
            "-D m_top={m_top} "
            "-D m_higgs={m_higgs} "
            "-D ymt={ymt} "
            "-D enable_thresholds={enable_thresholds} "
            "-D check_esurface_at_generation={check_esurface_at_generation} "
            "-D assume_positive_external_energies={assume_positive_external_energies} "
            "-D disable_threshold_subtraction={disable_threshold_subtraction}"
        )
        grouped_originals = self._original_graphs_by_group_from_dot(dot_path)
        grouped_members = self._graph_members_by_group_from_dot(dot_path)
        inspect_sampling_fragment = (
            "[sampling]\n"
            'graphs = "monte_carlo"\n'
            'orientations = "summed"\n'
            "lmb_multichanneling = false\n"
            'lmb_channels = "summed"\n'
            'coordinate_system = "spherical"\n'
            'mapping = "linear"\n'
            "b = 1.0\n"
        )
        integration_sampling_fragment = (
            "[sampling]\n"
            'graphs = "monte_carlo"\n'
            'orientations = "summed"\n'
            "lmb_multichanneling = true\n"
            'lmb_channels = "monte_carlo"\n'
            'coordinate_system = "spherical"\n'
            'mapping = "linear"\n'
            "b = 1.0\n"
        )
        uv_disabled_commands = [
            "set global kv global.generation.uv.subtract_uv=false",
            "set global kv global.generation.uv.generate_integrated=false",
            "set global kv global.generation.uv.local_uv_cts_from_expanded_4d_integrands=false",
        ]
        threshold_commands = [
            (
                "set global kv "
                "global.generation.threshold_subtraction.enable_thresholds="
                "{enable_thresholds}"
            ),
            (
                "set global kv "
                "global.generation.threshold_subtraction.check_esurface_at_generation="
                "{check_esurface_at_generation}"
            ),
            (
                "set global kv "
                "global.generation.threshold_subtraction.assume_positive_external_energies="
                "{assume_positive_external_energies}"
            ),
        ]
        runtime_subtraction_command = (
            f"set process -p {template_process_name} -i {template_integrand_name} kv "
            "subtraction.disable_threshold_subtraction={disable_threshold_subtraction}"
        )

        load_commands = [
            *uv_disabled_commands,
            *threshold_commands,
            f"import model {self.model}",
            "set model MT={m_top}",
            "set model MH={m_higgs}",
            "set model WT=0.0",
            "set model WH=0.0",
            "set model ymt={ymt}",
            f"remove processes -p {template_process_name}",
            (
                f"import graphs {template_dot_path} -p {template_process_name} "
                f"-i {template_integrand_name}"
            ),
        ]
        generate_commands = [
            f"run _load_subtracted_dot {template_define_args}",
            *uv_disabled_commands,
            *threshold_commands,
            f"generate existing -p {template_process_name} -i {template_integrand_name}",
        ]
        generate_ltd_commands = [
            f"run _load_subtracted_dot {template_define_args}",
            *uv_disabled_commands,
            *threshold_commands,
            "set global kv global.3d_representation=LTD",
            "set global kv global.generation.explicit_orientation_sum_only=true",
            f"generate existing -p {template_process_name} -i {template_integrand_name}",
        ]

        def set_process_sampling_command(fragment: str) -> str:
            return (
                f"set process -p {template_process_name} -i {template_integrand_name} string "
                f"'\n{fragment}'"
            )

        def inspect_group_commands(
            *,
            beam: str,
            group_id: int,
            graph: Any,
            use_arb_prec: bool,
        ) -> list[str]:
            momenta_fragment = self._external_momenta_toml_fragment()
            commands = [
                f"run _generate_subtracted_integrand {template_define_args}",
                f"set process -p {template_process_name} -i {template_integrand_name} defaults",
                runtime_subtraction_command,
                self._set_process_helicities_command(
                    template_process_name, template_integrand_name, pm_helicities
                ),
                set_process_sampling_command(momenta_fragment),
                set_process_sampling_command(inspect_sampling_fragment),
            ]
            precision_flag = "-f " if use_arb_prec else ""
            for lambda_value in self.collinear_lambdas:
                xs = self._format_xs(
                    self._collinear_xs_from_graph_fractional(
                        graph,
                        beam=beam,
                        x_fraction=self.collinear_fraction_x,
                        lambda_value=lambda_value,
                    )
                )
                commands.append(
                    f"inspect -p {template_process_name} -i {template_integrand_name} "
                    f"--discrete-dim {group_id} {precision_flag}-m -x {' '.join(xs)}"
                )
            return commands

        def command_block_toml(name: str, commands: list[str]) -> str:
            return (
                f'[[command_blocks]]\nname = "{name}"\ncommands = [\n'
                f"    {self._commands_toml_array(commands)}\n]\n"
            )

        def default_run_command(block_name: str, **extra_defines: str) -> str:
            define_args = default_define_args
            for key, value in extra_defines.items():
                define_args += f" -D {key}={value}"
            return f"run {block_name} {define_args}"

        def context_define_lines(*, threshold: bool, workspace_path: str) -> list[str]:
            m_top_value = 173.0 if threshold else 1000.0
            return [
                f"-D process_name={process_name}",
                f"-D integrand_name={subtracted_name}",
                f"-D dot_path={os.path.abspath(dot_path)}",
                f"-D m_top={m_top_value:.16f}",
                f"-D m_higgs={self.m_higgs:.16f}",
                f"-D ymt={m_top_value:.16f}",
                f"-D enable_thresholds={str(threshold).lower()}",
                "-D check_esurface_at_generation=false",
                "-D assume_positive_external_energies=false",
                f"-D disable_threshold_subtraction={str((not threshold)).lower()}",
                f"-D workspace_path={workspace_path}",
            ]

        def demo_run_command(*, threshold: bool, workspace_path: str) -> str:
            blocks = (
                "_generate_subtracted_integrand",
                "_inspect_collinear_p1_group0_arb",
                "_inspect_collinear_p2_group0_arb",
                "_low_stat_integrate_pm",
            )
            return (
                f"run {' '.join(blocks)}\n"
                + "\n".join(
                    context_define_lines(
                        threshold=threshold,
                        workspace_path=workspace_path,
                    )
                )
            )

        inspect_group_blocks: list[str] = []
        for beam in ("p1", "p2"):
            for group_id, graph in grouped_originals:
                for use_arb_prec in (False, True):
                    suffix = "_arb" if use_arb_prec else ""
                    block_name = f"inspect_collinear_{beam}_group{group_id}{suffix}"
                    template_block_name = f"_{block_name}"
                    inspect_group_blocks.append(
                        command_block_toml(
                            template_block_name,
                            inspect_group_commands(
                                beam=beam,
                                group_id=group_id,
                                graph=graph,
                                use_arb_prec=use_arb_prec,
                            ),
                        )
                    )
                    inspect_group_blocks.append(
                        command_block_toml(
                            block_name,
                            [default_run_command(template_block_name)],
                        )
                    )

        inspect_p1_commands = [
            f"run inspect_collinear_p1_group{group_id}"
            for group_id, _graph in grouped_originals
        ]
        inspect_p1_arb_commands = [
            f"run inspect_collinear_p1_group{group_id}_arb"
            for group_id, _graph in grouped_originals
        ]
        inspect_p2_commands = [
            f"run inspect_collinear_p2_group{group_id}"
            for group_id, _graph in grouped_originals
        ]
        inspect_p2_arb_commands = [
            f"run inspect_collinear_p2_group{group_id}_arb"
            for group_id, _graph in grouped_originals
        ]

        def integrate_commands_for(helicities_for_block: list[int]) -> list[str]:
            return [
                f"run _generate_subtracted_integrand {template_define_args}",
                f"set process -p {template_process_name} -i {template_integrand_name} defaults",
                runtime_subtraction_command,
                self._set_process_helicities_command(
                    template_process_name, template_integrand_name, helicities_for_block
                ),
                (
                    f"set process -p {template_process_name} -i {template_integrand_name} kv "
                    f"integrator.n_start={self.low_stat_n_start} "
                    "integrator.n_increase=0 "
                    f"integrator.n_max={self.low_stat_n_max} "
                    "integrator.seed=1337 "
                    'integrator.integrated_phase="real"'
                ),
                (
                    f"set process -p {template_process_name} -i {template_integrand_name} string "
                    f"'\n{integration_sampling_fragment}'"
                ),
                (
                    f"integrate -p {template_process_name} -i {template_integrand_name} "
                    f"--n-cores {self.low_stat_n_cores} "
                    f"--batch-size {self.low_stat_batch_size} "
                    "--workspace-path {workspace_path} --restart"
                ),
            ]

        integrate_pm_commands = integrate_commands_for(pm_helicities)
        integrate_pp_commands = integrate_commands_for(pp_helicities)
        workspace_pm = os.path.abspath(
            self._integration_workspace(subtracted_name, pm_helicities)
        )
        workspace_pp = os.path.abspath(
            self._integration_workspace(subtracted_name, pp_helicities)
        )
        demo_workspace_pm = os.path.abspath(
            pjoin(self.dot_folder, f"{subtracted_name}_demo_no_threshold_pm")
        )
        demo_threshold_workspace_pm = os.path.abspath(
            pjoin(self.dot_folder, f"{subtracted_name}_demo_threshold_pm")
        )
        demo_commands = [
            demo_run_command(threshold=False, workspace_path=demo_workspace_pm),
        ]
        demo_with_thresholds_commands = [
            demo_run_command(
                threshold=True,
                workspace_path=demo_threshold_workspace_pm,
            ),
        ]

        run_card = f"""# Intentionally empty: loading this card should only apply settings and
# register command blocks. Use `run <block-name>` explicitly to load,
# generate, inspect, or integrate.
commands = []

[cli_settings]
[cli_settings.state]
folder = {json.dumps(os.path.abspath(state_folder))}
name = {json.dumps(subtracted_name)}

[[command_blocks]]
name = "_load_subtracted_dot"
commands = [
    {self._commands_toml_array(load_commands)}
]

[[command_blocks]]
name = "load_subtracted_dot"
commands = [
    {self._commands_toml_array([default_run_command("_load_subtracted_dot")])}
]

[[command_blocks]]
name = "_generate_subtracted_integrand"
commands = [
    {self._commands_toml_array(generate_commands)}
]

[[command_blocks]]
name = "generate_subtracted_integrand"
commands = [
    {self._commands_toml_array([default_run_command("_generate_subtracted_integrand")])}
]

[[command_blocks]]
name = "_generate_subtracted_integrand_ltd"
commands = [
    {self._commands_toml_array(generate_ltd_commands)}
]

[[command_blocks]]
name = "generate_subtracted_integrand_ltd"
commands = [
    {self._commands_toml_array([default_run_command("_generate_subtracted_integrand_ltd")])}
]

[[command_blocks]]
name = "inspect_collinear_p1"
commands = [
    {self._commands_toml_array(inspect_p1_commands)}
]

[[command_blocks]]
name = "inspect_collinear_p1_arb"
commands = [
    {self._commands_toml_array(inspect_p1_arb_commands)}
]

[[command_blocks]]
name = "inspect_collinear_p2"
commands = [
    {self._commands_toml_array(inspect_p2_commands)}
]

[[command_blocks]]
name = "inspect_collinear_p2_arb"
commands = [
    {self._commands_toml_array(inspect_p2_arb_commands)}
]

{''.join(inspect_group_blocks)}
[[command_blocks]]
name = "_low_stat_integrate_pm"
commands = [
    {self._commands_toml_array(integrate_pm_commands)}
]

[[command_blocks]]
name = "_low_stat_integrate_pp"
commands = [
    {self._commands_toml_array(integrate_pp_commands)}
]

[[command_blocks]]
name = "low_stat_integrate"
commands = [
    {self._commands_toml_array([default_run_command("_low_stat_integrate_pm", workspace_path=workspace_pm)])}
]

[[command_blocks]]
name = "low_stat_integrate_pm"
commands = [
    {self._commands_toml_array([default_run_command("_low_stat_integrate_pm", workspace_path=workspace_pm)])}
]

[[command_blocks]]
name = "low_stat_integrate_pp"
commands = [
    {self._commands_toml_array([default_run_command("_low_stat_integrate_pp", workspace_path=workspace_pp)])}
]

[[command_blocks]]
name = "demo"
commands = [
    {self._commands_toml_array(demo_commands)}
]

[[command_blocks]]
name = "demo_with_thresholds"
commands = [
    {self._commands_toml_array(demo_with_thresholds_commands)}
]

[default_runtime_settings.kinematics]
e_cm = 1000.0

[default_runtime_settings.kinematics.externals]
type = "constant"

[default_runtime_settings.kinematics.externals.data]
momenta = [
    {momenta}
]
helicities = [{helicities}]

[default_runtime_settings.general]
evaluator_method = "SingleParametric"
enable_cache = false
debug_cache = false
generate_events = true
store_additional_weights_in_event = true
{additional_param_values}disable_flux_factor = false
integral_unit = "none"
mu_r = 91.188
m_uv = 91.188

[default_runtime_settings.sampling]
graphs = "monte_carlo"
orientations = "summed"
lmb_multichanneling = false
lmb_channels = "summed"
coordinate_system = "spherical"
mapping = "linear"
b = 1.0

[default_runtime_settings.subtraction]
disable_threshold_subtraction = {str((not self.enable_threshold_subtraction)).lower()}

[cli_settings.global.generation.threshold_subtraction]
enable_thresholds = {str(self.enable_threshold_subtraction).lower()}
check_esurface_at_generation = {str(self.check_esurface_at_generation).lower()}
assume_positive_external_energies = {str(self.assume_positive_external_energies).lower()}

[cli_settings.global.generation.uv]
subtract_uv = false
generate_integrated = false
local_uv_cts_from_expanded_4d_integrands = false

[cli_settings.global.generation.evaluator]
iterative_orientation_optimization = false
store_atom = false
compile = true
summed = false
summed_function_map = {str(self.evaluator_summed_function_map).lower()}

[cli_settings.global.n_cores]
feyngen = 1
generate = {self.low_stat_n_cores}
compile = {self.low_stat_n_cores}
integrate = {self.low_stat_n_cores}
"""
        write_text_with_dirs(run_card_path, run_card)
        return run_card_path

    def build_ir_subtracted_graphs(
        self,
        *,
        max_original_graphs: int | None = None,
        suffix: str | None = None,
        import_graphs: bool | None = None,
        allow_cached_selected_dot: bool = False,
    ) -> dict[str, Any]:
        if not self.build_isr_counterterms:
            raise pygloopException("qqbar_nX ISR counterterms are disabled in config.")

        selected_name = self.get_integrand_name()
        selected_dot_path = pjoin(self.dot_folder, f"{selected_name}.dot")
        should_import_graphs = (
            self.import_subtracted_dot if import_graphs is None else import_graphs
        )
        if (
            allow_cached_selected_dot
            and not self.clean
            and os.path.isfile(selected_dot_path)
        ):
            logger.info(
                "Recycling selected qqbar_nX DOT for fast IR-CT rebuild: %s.",
                selected_dot_path,
            )
        else:
            self.generate_graphs(import_selected=should_import_graphs)

        subtracted_name = self.get_integrand_name(suffix=suffix or self.subtracted_suffix)
        with open(selected_dot_path, encoding="utf-8") as handle:
            selected_dot = handle.read()

        graphs = parse_dot_graphs(selected_dot)
        if max_original_graphs is not None:
            graphs = graphs[:max_original_graphs]
        subtracted_graphs, counterterm_report = build_isr_counterterm_graphs(
            graphs,
            projector_mode=self.counterterm_projector_mode,
            denominator_strategy=self.counterterm_denominator_strategy,
            auxiliary_denominator_mode=self.counterterm_auxiliary_denominator_mode,
            global_phase=self.counterterm_global_phase,
            normalization_factor=self.counterterm_normalization_factor,
            uv_inert_dod=self.counterterm_uv_inert_dod,
            use_parametric_xi=self.counterterm_use_parametric_xi,
            xi_parameter_names=self.xi_parameter_names,  # type: ignore[arg-type]
            xi_default_values=self.xi_default_values,  # type: ignore[arg-type]
        )
        minimise_edge_attributes_for_import(subtracted_graphs)
        subtracted_dot = dot_graphs_to_string(subtracted_graphs)
        subtracted_dot_path = pjoin(self.dot_folder, f"{subtracted_name}.dot")
        manifest_path = pjoin(self.dot_folder, f"{subtracted_name}.manifest.json")
        write_text_with_dirs(subtracted_dot_path, subtracted_dot)
        run_card_path = (
            self.write_gammaloop_run_card(
                subtracted_dot_path, integrand_name=subtracted_name
            )
            if self.write_standalone_run_card
            else None
        )
        manifest = {
            **counterterm_report.manifest(),
            "selected_dot_path": selected_dot_path,
            "subtracted_dot_path": subtracted_dot_path,
            "standalone_run_card_path": run_card_path,
            "standalone_state_folder": self.get_standalone_state_folder(subtracted_name)
            if self.write_standalone_run_card
            else None,
            "description": (
                "Selected d d~ -> h h h top-pentagon graphs plus local ISR-collinear "
                "Delta_1 and Delta_2 counterterm graphs. Counterterms share the original "
                "canonical topology group_id; original graph masters are preserved."
            ),
        }
        write_text_with_dirs(
            manifest_path, json.dumps(manifest, indent=2, sort_keys=True)
        )

        if should_import_graphs:
            worker = self.require_gl_worker()
            amplitudes, _cross_sections = worker.list_outputs()
            if subtracted_name not in amplitudes:
                raw_name = self.get_integrand_name(suffix="_raw_generated_graphs")
                process_ref = (
                    self.get_subtracted_process_name()
                    if self.uses_fake_xi_externals()
                    else amplitudes.get(raw_name, self.get_integrand_name(suffix=""))
                )
                worker.run(
                    f"import graphs {subtracted_dot_path} -p {process_ref} -i {subtracted_name}"
                )
                self.save_state()
            else:
                logger.info(
                    f"IR-subtracted amplitude {subtracted_name} already exists and is recycled."
                )

        logger.info(
            "qqbar_nX wrote %s original graphs and %s ISR counterterm graphs to %s.",
            counterterm_report.manifest()["counts"]["original_graphs"],
            counterterm_report.manifest()["counts"]["counterterm_graphs"],
            subtracted_dot_path,
        )
        return manifest

    def build_ir_smoke_test_graphs(self) -> dict[str, Any]:
        return self.build_ir_subtracted_graphs(
            max_original_graphs=self.smoke_test_original_graphs,
            suffix=self.smoke_test_suffix,
        )

    def _run_standalone_command(
        self,
        *,
        label: str,
        command: list[str],
        log_prefix: str,
    ) -> dict[str, Any]:
        started = time.time()
        try:
            completed = subprocess.run(
                command,
                cwd=os.getcwd(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.standalone_test_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            log_path = pjoin(self.dot_folder, f"{log_prefix}_{label}.log")
            write_text_with_dirs(log_path, exc.stdout or "")
            raise pygloopException(
                f"GammaLoop standalone smoke command '{label}' timed out after "
                f"{self.standalone_test_timeout}s. See {log_path}."
            ) from exc

        log_path = pjoin(self.dot_folder, f"{log_prefix}_{label}.log")
        write_text_with_dirs(log_path, completed.stdout)
        runtime = time.time() - started
        result = {
            "label": label,
            "command": command,
            "returncode": completed.returncode,
            "runtime_s": runtime,
            "log_path": log_path,
        }
        gamma_error = (
            " ERROR " in completed.stdout
            or "Failed to create evaluator" in completed.stdout
            or "Failed to create summed function map" in completed.stdout
            or "TOML parse error" in completed.stdout
            or "Trying to parse runtime settings TOML string" in completed.stdout
        )
        if completed.returncode != 0 or gamma_error:
            raise pygloopException(
                f"GammaLoop standalone smoke command '{label}' failed with exit code "
                f"{completed.returncode}. See {log_path}."
            )
        return result

    def _set_standalone_helicities(
        self,
        *,
        state_folder: str,
        process_name: str,
        integrand_name: str,
        helicities: list[int],
        log_prefix: str,
    ) -> dict[str, Any]:
        label = f"set_helicities_{_helicity_label(helicities)}"
        return self._run_standalone_command(
            label=label,
            log_prefix=log_prefix,
            command=[
                self.gammaloop_cli_path,
                "-o",
                "-s",
                state_folder,
                "set",
                "process",
                "-p",
                process_name,
                "-i",
                integrand_name,
                "string",
                _helicity_toml_fragment(helicities),
            ],
        )

    def _run_inspect_and_parse(
        self,
        *,
        label: str,
        command: list[str],
        log_prefix: str,
        expected_accepted_events: int | None = None,
    ) -> dict[str, Any]:
        command_result = self._run_standalone_command(
            label=label, command=command, log_prefix=log_prefix
        )
        parsed = self._parse_inspect_log(command_result["log_path"])
        if parsed["nan"]:
            raise pygloopException(f"GammaLoop inspect returned nan. See {parsed['log_path']}.")
        if (
            expected_accepted_events is not None
            and parsed["accepted_events"] != expected_accepted_events
        ):
            raise pygloopException(
                "GammaLoop inspect accepted-event count mismatch for "
                f"{label}: expected {expected_accepted_events}, got "
                f"{parsed['accepted_events']}. See {parsed['log_path']}."
            )
        parsed["command"] = command_result
        parsed["exact_zero"] = self._inspect_result_is_exact_zero(parsed)
        return parsed

    def _check_low_stat_integration_result(
        self, integration_result_path: str
    ) -> dict[str, Any]:
        with open(integration_result_path, encoding="utf-8") as handle:
            integration_result = json.load(handle)
        slot = integration_result["slots"][0]
        integral = slot["integral"]
        statistics = slot["integration_statistics"]
        values = [
            integral["result"]["re"],
            integral["result"]["im"],
            integral["error"]["re"],
            integral["error"]["im"],
        ]
        finite_values = all(math.isfinite(float(value)) for value in values)
        exact_zero_result_and_error = all(
            Decimal(str(value)) == Decimal(0) for value in values
        )
        check = {
            "integration_result_path": integration_result_path,
            "neval": integral["neval"],
            "result": integral["result"],
            "error": integral["error"],
            "nan_or_unstable_percentage": statistics["nan_or_unstable_percentage"],
            "f64_percentage": statistics.get("f64_percentage"),
            "f128_percentage": statistics.get("f128_percentage"),
            "arb_percentage": statistics.get("arb_percentage"),
            "finite_values": finite_values,
            "exact_zero_result_and_error": exact_zero_result_and_error,
            "passed": (
                integral["neval"] == self.low_stat_n_max
                and finite_values
                and (
                    statistics["nan_or_unstable_percentage"] == 0.0
                    or exact_zero_result_and_error
                )
            ),
        }
        if not check["passed"]:
            raise pygloopException(
                "GammaLoop low-stat integration smoke check failed. "
                f"See {integration_result_path}."
            )
        return check

    def _run_low_stat_integrate_for_helicity(
        self,
        *,
        state_folder: str,
        integrand_name: str,
        helicities: list[int],
        log_prefix: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        label = f"low_stat_integrate_{_helicity_label(helicities)}"
        command_result = self._run_standalone_command(
            label=label,
            command=[
                self.gammaloop_cli_path,
                "-o",
                "-s",
                state_folder,
                "run",
                label,
            ],
            log_prefix=log_prefix,
        )
        integration_result_path = pjoin(
            self._integration_workspace(integrand_name, helicities),
            "integration_result.json",
        )
        return command_result, self._check_low_stat_integration_result(
            integration_result_path
        )

    def _collinear_precision_settings_fragment(self) -> tuple[str, bool, int]:
        precision_map = {
            "Double": ("Double", False, 17),
            "Quad": ("Quad", False, 34),
            "ArbPrec": ("Arb", True, self.collinear_arb_display_digits),
        }
        gamma_precision, use_arb_prec, display_digits = precision_map[
            self.collinear_precision
        ]
        fragment = (
            "[stability]\n"
            "rotation_axis = [{ type = \"none\" }]\n"
            "check_on_norm = true\n"
            "escalate_if_exact_zero = false\n"
            "\n"
            "[[stability.levels]]\n"
            f'precision = "{gamma_precision}"\n'
            "required_precision_for_re = 1e-10\n"
            "required_precision_for_im = 1e-10\n"
            "escalate_for_large_weight_threshold = -1.0\n"
            "\n"
            "[stability.recording]\n"
            "record_all_stability_levels = true\n"
        )
        return fragment, use_arb_prec, display_digits

    def _collinear_test_sampling_fragment(self) -> str:
        return (
            "[sampling]\n"
            'graphs = "monte_carlo"\n'
            'orientations = "summed"\n'
            "lmb_multichanneling = false\n"
            'lmb_channels = "summed"\n'
            'coordinate_system = "spherical"\n'
            'mapping = "linear"\n'
            "b = 1.0\n"
        )

    def _set_collinear_test_runtime(
        self, api: GammaLoopAPI, *, process_name: str, integrand_name: str
    ) -> tuple[bool, int]:
        stability_fragment, use_arb_prec, display_digits = (
            self._collinear_precision_settings_fragment()
        )
        runtime_fragment = (
            _helicity_toml_fragment(self._helicities_for_current_externals())
            + self._collinear_test_sampling_fragment()
            + stability_fragment
        )
        api.run(f"set process -p {process_name} -i {integrand_name} defaults")
        api.run(
            f"set process -p {process_name} -i {integrand_name} string "
            f"'\n{runtime_fragment}'"
        )
        return use_arb_prec, display_digits

    def _external_momenta_toml_fragment(
        self,
        *,
        fake_xi_momentum: dict[str, list[float]] | list[float] | None = None,
    ) -> str:
        if self.uses_fake_xi_externals():
            momenta = self._current_external_momenta_4d(
                fake_xi_momentum=fake_xi_momentum
            )
        else:
            momenta = self._current_external_momenta_4d()
        return (
            "[kinematics.externals.data]\n"
            "momenta = [\n"
            f"    {_momenta_toml_array(momenta, dependent_last=True)}\n"
            "]\n"
        )

    def _set_default_runtime_external_momenta(
        self,
        api: GammaLoopAPI,
        *,
        fake_xi_momentum: dict[str, list[float]] | list[float] | None = None,
    ) -> None:
        fragment = self._external_momenta_toml_fragment(
            fake_xi_momentum=fake_xi_momentum
        )
        api.run(f"set default-runtime string '\n{fragment}'")

    def _set_process_external_momenta(
        self,
        api: GammaLoopAPI,
        *,
        process_name: str,
        integrand_name: str,
        fake_xi_momentum: dict[str, list[float]] | list[float] | None = None,
    ) -> None:
        fragment = self._external_momenta_toml_fragment(
            fake_xi_momentum=fake_xi_momentum
        )
        api.run(
            f"set process -p {process_name} -i {integrand_name} string "
            f"'\n{fragment}'"
        )

    def _graph_id_name_map_from_integrand_info(
        self, api: GammaLoopAPI, *, integrand_name: str
    ) -> dict[int, str]:
        info = api.get_integrand_info(process_id=0, integrand_name=integrand_name)
        graph_id_to_name: dict[int, str] = {}
        for group_info in info.graph_groups:
            for graph_info in group_info.graphs:
                graph_id_to_name[int(graph_info.graph_id)] = str(graph_info.name)
        return graph_id_to_name

    def _member_values_from_event_groups(
        self,
        sample: Any,
        *,
        graph_id_to_name: dict[int, str],
        ordered_graph_names: list[str],
    ) -> tuple[list[tuple[str, complex]], dict[str, dict[str, Any]]]:
        values_by_name: dict[str, complex] = {}
        event_metadata: dict[str, dict[str, Any]] = {}

        for event_group in sample.event_groups:
            for event in event_group.events:
                cut_info = event.cut_info
                graph_id = int(cut_info.graph_id)
                name = graph_id_to_name.get(graph_id, f"graph#{graph_id}")
                value = _complex_from_api_value(event.weight)
                values_by_name[name] = values_by_name.get(name, 0.0j) + value

                additional_weights = {
                    str(weight.key): _complex_from_api_value(weight.value)
                    for weight in event.additional_weights
                }
                slot = event_metadata.setdefault(
                    name,
                    {
                        "graph_id": graph_id,
                        "graph_group_id": int(cut_info.graph_group_id),
                        "cut_ids": [],
                        "orientation_ids": [],
                        "lmb_channel_ids": [],
                        "additional_weights": {},
                    },
                )
                slot["cut_ids"].append(int(cut_info.cut_id))
                slot["orientation_ids"].append(cut_info.orientation_id)
                slot["lmb_channel_ids"].append(cut_info.lmb_channel_id)
                for key, weight_value in additional_weights.items():
                    current = slot["additional_weights"].get(key, 0.0j)
                    slot["additional_weights"][key] = current + weight_value

        ordered = [
            (name, values_by_name[name])
            for name in ordered_graph_names
            if name in values_by_name
        ]
        for name, value in values_by_name.items():
            if name not in ordered_graph_names:
                ordered.append((name, value))

        for slot in event_metadata.values():
            slot["additional_weights"] = {
                key: {"re": value.real, "im": value.imag}
                for key, value in slot["additional_weights"].items()
            }

        return ordered, event_metadata

    def _ensure_api_test_state(
        self,
        *,
        manifest: dict[str, Any],
        integrand_name: str,
        generate_command_block: str = "generate_subtracted_integrand",
    ) -> GammaLoopAPI:
        state_folder = manifest.get("standalone_state_folder")
        run_card_path = manifest.get("standalone_run_card_path")
        if not state_folder or not run_card_path:
            raise pygloopException(
                "qqbar_nX test_process requires the standalone run card/state paths."
            )

        if self.clean and os.path.isdir(state_folder):
            shutil.rmtree(state_folder)

        state_has_command_blocks = False
        if os.path.isdir(state_folder):
            try:
                api = GammaLoopAPI(state_folder, clean_state=False)
                state_has_command_blocks = bool(api.get_active_command_blocks())
            except Exception:
                state_has_command_blocks = False

        generated_integrand_exists = False
        state_process_folder = pjoin(state_folder, "processes") if state_folder else ""
        if os.path.isdir(state_process_folder):
            for root, _dirs, files in os.walk(state_process_folder):
                if os.path.basename(root) == integrand_name and "amp.bin" in files:
                    generated_integrand_exists = True
                    break

        prepared_api: GammaLoopAPI | None = None
        if not state_has_command_blocks:
            if os.path.isdir(state_folder):
                shutil.rmtree(state_folder)
            logger.info(
                "Preparing qqbar_nX test state through the Python API boot "
                "commands so the generated state matches the API used for "
                "evaluate_samples."
            )
            api = GammaLoopAPI(
                state_folder,
                boot_commands_path=run_card_path,
                clean_state=True,
            )
            api.run(f"run {generate_command_block}")
            api.run("save state -o true")
            prepared_api = api
            generated_integrand_exists = True

        if prepared_api is not None:
            return prepared_api

        api = GammaLoopAPI(state_folder, clean_state=False)
        if not generated_integrand_exists:
            api.run(f"run {generate_command_block}")
            api.run("save state -o true")
        return api

    def _format_complex_for_collinear_table(
        self, value: complex, *, requested_digits: int
    ) -> str:
        # The current GammaLoop Python API returns Python complex values even for
        # Quad/ArbPrec evaluation, so the reliable payload exposed to pygloop is
        # still the f64 conversion of the requested precision result.
        digits = min(requested_digits, 17)
        return f"{value.real:+.{digits}e} {value.imag:+.{digits}e}i"

    def _ratio_colour(self, ratio: float) -> Colour:
        if ratio < 1.0e-6:
            return Colour.GREEN
        if ratio < 1.0e-3:
            return Colour.YELLOW
        return Colour.RED

    def _render_collinear_cancellation_table(
        self,
        *,
        beam: str,
        rows: list[dict[str, Any]],
        display_digits: int,
        backend_label: str | None = None,
    ) -> str:
        from prettytable import PrettyTable

        table = PrettyTable()
        table.field_names = [
            "lambda",
            "group_id",
            "graph members",
            "group CFF sum",
            "abs(sum)/sum(abs)",
        ]
        table.align = "l"
        table.align["abs(sum)/sum(abs)"] = "r"
        table.align["lambda"] = "r"
        table.align["group_id"] = "r"
        for row in rows:
            members = []
            for name, value in row["members"]:
                name_colour = Colour.RED if name.endswith("_ct") else Colour.BLUE
                members.append(
                    f"{name_colour}{name}{Colour.END}: "
                    f"{Colour.CYAN}{self._format_complex_for_collinear_table(value, requested_digits=display_digits)}{Colour.END}"
                )
            ratio = row["ratio"]
            ratio_colour = self._ratio_colour(ratio)
            table.add_row(
                [
                    _lambda_label(row["lambda"]),
                    str(row["group_id"]),
                    "\n".join(members),
                    f"{Colour.GREEN}{self._format_complex_for_collinear_table(row['sum'], requested_digits=display_digits)}{Colour.END}",
                    f"{ratio_colour}{ratio:.3e}{Colour.END}",
                ]
            )
        title = (
            f"qqbar_nX collinear approach {beam}: x={self.collinear_fraction_x}, "
            f"|k_perp|=lambda*sqrt(s), precision={self.collinear_precision}, "
            f"helicities={self._helicities_for_current_externals()}"
        )
        if backend_label:
            title += f", backend={backend_label}"
        return f"{title}\n{table}"

    def test_process(self, mode: str = "gammaloop", **_opts: Any) -> dict[str, Any]:
        normalized_mode = mode.lower()
        if normalized_mode in {"4d", "four-dimensional", "four_dimensional"}:
            return self.test_process_4d()
        if normalized_mode in {"gammaloop", "3d", "cff"}:
            return self.test_process_gammaloop()
        if normalized_mode in {"ltd", "gammaloop-ltd", "3d-ltd"}:
            return self.test_process_gammaloop(representation="LTD")
        if normalized_mode in {"pygloop-cff", "cff-meta"}:
            return self.test_process_pygloop_cff()
        raise pygloopException(
            "qqbar_nX test_process mode must be one of 'gammaloop', 'LTD', "
            "'pygloop-cff' or '4D'."
        )

    def _external_momenta_4d(self) -> list[list[float]]:
        return self._current_external_momenta_4d()

    def _sm_4d_model_values(self) -> dict[str, complex]:
        a_ewm1 = 132.50698
        g_f = 0.0000116639
        a_s = 0.118
        m_z = 91.188
        a_ew = 1.0 / a_ewm1
        m_w = math.sqrt(
            m_z**2 / 2.0
            + math.sqrt(m_z**4 / 4.0 - (a_ew * math.pi * m_z**2) / (g_f * math.sqrt(2.0)))
        )
        ee = 2.0 * math.sqrt(a_ew) * math.sqrt(math.pi)
        sw2 = 1.0 - m_w**2 / m_z**2
        sw = math.sqrt(sw2)
        vev = (2.0 * m_w * sw) / ee
        strong_coupling = 2.0 * math.sqrt(a_s) * math.sqrt(math.pi)
        return {
            "UFO::GC_11": 1j * strong_coupling,
            "UFO::GC_94": -1j * self.m_top / vev,
            "UFO::MT": complex(self.m_top),
        }

    def _collinear_loop_momenta_4d(
        self,
        graph: Any,
        *,
        beam: str,
        x_fraction: float,
        lambda_value: float,
        routed_graph: Any,
        fake_xi_momentum: list[float] | None = None,
    ) -> list[list[float]]:
        if not 0.0 < x_fraction < 1.0:
            raise pygloopException("tests.collinear_fraction_x must lie in (0, 1).")

        e_cm = self._center_of_mass_energy()
        k_perp_norm = lambda_value * e_cm
        if beam == "p1":
            beam_energy = abs(float(self.ps_point[0].t))
            target = [
                x_fraction * beam_energy,
                k_perp_norm,
                0.0,
                x_fraction * beam_energy,
            ]
        elif beam == "p2":
            beam_energy = abs(float(self.ps_point[1].t))
            target = [
                -x_fraction * beam_energy,
                k_perp_norm,
                0.0,
                x_fraction * beam_energy,
            ]
        else:
            raise pygloopException(f"Unknown collinear beam '{beam}'.")

        structure = identify_light_line_structure(graph)
        edge = self._edge_by_id(routed_graph, structure.light_edge_id)
        lmb_rep = edge.get_attributes().get("lmb_rep")
        if lmb_rep is None:
            raise pygloopException(
                f"Cannot build 4D collinear point for {graph_name(graph)}: "
                "the light-line edge has no lmb_rep."
            )

        coefficients = _parse_lmb_representation(lmb_rep)
        external_vectors = {
            index: momentum
            for index, momentum in enumerate(
                self._external_momenta_for_graph_4d(
                    routed_graph, fake_xi_momentum=fake_xi_momentum
                )
            )
        }
        spectator_loop_4d = [1200.0, 250.0, 125.0, -375.0]
        fixed_loop_vectors = {1: list(spectator_loop_4d)}
        solve_loop_index = 0
        solve_coefficient = coefficients["K"].get(solve_loop_index, 0.0)
        if solve_coefficient == 0.0:
            solve_loop_index = 1
            solve_coefficient = coefficients["K"].get(solve_loop_index, 0.0)
            fixed_loop_vectors = {0: list(spectator_loop_4d)}
        if solve_coefficient == 0.0:
            raise pygloopException(
                f"Cannot solve 4D collinear point for {graph_name(graph)}: "
                f"no loop momentum appears in {strip_quotes(lmb_rep)}."
            )

        rhs = list(target)
        for p_index, coefficient in coefficients["P"].items():
            if p_index not in external_vectors:
                raise pygloopException(
                    f"lmb_rep references unavailable external P({p_index}) "
                    f"for {graph_name(graph)}."
                )
            for component in range(4):
                rhs[component] -= coefficient * external_vectors[p_index][component]

        for k_index, coefficient in coefficients["K"].items():
            if k_index == solve_loop_index:
                continue
            fixed_loop_vectors.setdefault(k_index, list(spectator_loop_4d))
            for component in range(4):
                rhs[component] -= coefficient * fixed_loop_vectors[k_index][component]

        loop_vectors = dict(fixed_loop_vectors)
        loop_vectors[solve_loop_index] = [
            component / solve_coefficient for component in rhs
        ]
        if 0 not in loop_vectors or 1 not in loop_vectors:
            raise pygloopException(
                f"Could not construct both 4D loop momenta for {graph_name(graph)}."
            )
        return [loop_vectors[0], loop_vectors[1]]

    def _routed_dot_for_4d_test(
        self, *, manifest: dict[str, Any], integrand_name: str
    ) -> str:
        dot_path = manifest["subtracted_dot_path"]
        routed_dot_path = pjoin(self.dot_folder, f"{integrand_name}.routed.dot")
        if (
            os.path.isfile(routed_dot_path)
            and os.path.getmtime(routed_dot_path) >= os.path.getmtime(dot_path)
        ):
            logger.info("Recycling routed DOT for 4D test: %s.", routed_dot_path)
            return routed_dot_path

        state_folder = manifest.get("standalone_state_folder")
        run_card_path = manifest.get("standalone_run_card_path")
        if not state_folder or not run_card_path:
            raise pygloopException(
                "qqbar_nX 4D test requires the standalone run card/state paths."
            )
        if self.clean and os.path.isdir(state_folder):
            shutil.rmtree(state_folder)
        api = GammaLoopAPI(
            state_folder,
            boot_commands_path=run_card_path,
            clean_state=not os.path.isdir(state_folder),
        )
        api.run("run load_subtracted_dot")
        dot_export_settings = DotExportSettings()
        dot_export_settings.include_autogenerated_fields = True
        routed_dot = api.get_dot_files(
            integrand_name=integrand_name,
            settings=dot_export_settings,
        )
        write_text_with_dirs(routed_dot_path, routed_dot)
        return routed_dot_path

    def test_process_gammaloop(self, *, representation: str = "CFF") -> dict[str, Any]:
        representation = representation.upper()
        if representation not in {"CFF", "LTD"}:
            raise pygloopException(
                "qqbar_nX GammaLoop collinear test representation must be CFF or LTD."
            )
        manifest = self.build_ir_subtracted_graphs()
        subtracted_name = self.get_subtracted_integrand_name()
        process_name = self.get_subtracted_process_name()
        dot_path = manifest["subtracted_dot_path"]
        dot_export_settings = DotExportSettings()
        dot_export_settings.include_autogenerated_fields = True

        api = self._ensure_api_test_state(
            manifest=manifest,
            integrand_name=subtracted_name,
            generate_command_block=(
                "generate_subtracted_integrand_ltd"
                if representation == "LTD"
                else "generate_subtracted_integrand"
            ),
        )
        routed_dot = api.get_dot_files(
            integrand_name=subtracted_name,
            settings=dot_export_settings,
        )
        routed_dot_path = pjoin(self.dot_folder, f"{subtracted_name}.routed.dot")
        write_text_with_dirs(routed_dot_path, routed_dot)

        import numpy as np

        grouped_reference_graphs = self._graph_members_by_group_from_dot(dot_path)
        grouped_graphs = self._graph_members_by_group_from_routed_dot(
            reference_dot_path=dot_path, routed_dot_path=routed_dot_path
        )
        exact_xi_routing = self._verify_exact_xi_routing(grouped_graphs)
        use_arb_prec, display_digits = self._set_collinear_test_runtime(
            api, process_name=process_name, integrand_name=subtracted_name
        )
        graph_id_to_name = self._graph_id_name_map_from_integrand_info(
            api, integrand_name=subtracted_name
        )

        tables: dict[str, str] = {}
        numeric_results: dict[str, Any] = {}

        for beam in ("p1", "p2"):
            beam_rows: list[dict[str, Any]] = []
            numeric_results[beam] = {}
            for group_id, graphs in grouped_graphs.items():
                reference_graphs = grouped_reference_graphs[group_id]
                master_graph = next(
                    (
                        graph
                        for graph in reference_graphs
                        if not graph_name(graph).endswith("_ct")
                    ),
                    reference_graphs[0],
                )
                routed_master_graph = next(
                    (graph for graph in graphs if not graph_name(graph).endswith("_ct")),
                    graphs[0],
                )
                graph_names = [graph_name(graph) for graph in graphs]
                numeric_results[beam][str(group_id)] = {}
                for lambda_value in self.collinear_lambdas:
                    xs = self._collinear_xs_from_graph_fractional(
                        master_graph,
                        beam=beam,
                        x_fraction=self.collinear_fraction_x,
                        lambda_value=lambda_value,
                        routed_graph=routed_master_graph,
                    )
                    points = np.array([xs], dtype=float)
                    group_batch = api.evaluate_samples(
                        points,
                        integrand_name=subtracted_name,
                        use_arb_prec=use_arb_prec,
                        minimal_output=True,
                        return_events=True,
                        momentum_space=True,
                        discrete_dims=np.array([[group_id]], dtype=np.uintp),
                    )
                    sample = group_batch.samples[0]
                    member_values, event_metadata = self._member_values_from_event_groups(
                        sample,
                        graph_id_to_name=graph_id_to_name,
                        ordered_graph_names=graph_names,
                    )
                    group_sum = sum(value for _name, value in member_values)
                    members_source = "event_groups"
                    sum_abs = sum(abs(value) for _name, value in member_values)
                    ratio = abs(group_sum) / sum_abs if sum_abs != 0.0 else 0.0
                    beam_rows.append(
                        {
                            "lambda": lambda_value,
                            "group_id": group_id,
                            "members": [(name, value) for name, value in member_values],
                            "sum": group_sum,
                            "sum_abs": sum_abs,
                            "ratio": ratio,
                        }
                    )
                    numeric_results[beam][str(group_id)][
                        _lambda_label(lambda_value)
                    ] = {
                        "xs": xs,
                        "fake_xi_momentum": None,
                        "group_evaluation_mode": "discrete_group",
                        "members_source": members_source,
                        "members": {
                            name: {"re": value.real, "im": value.imag}
                            for name, value in member_values
                        },
                        "event_metadata": event_metadata,
                        "sum": {"re": group_sum.real, "im": group_sum.imag},
                        "sum_abs": sum_abs,
                        "abs_sum_over_sum_abs": ratio,
                    }
            tables[beam] = self._render_collinear_cancellation_table(
                beam=beam, rows=beam_rows, display_digits=display_digits
            )

        rendered_report = "\n\n".join(tables[beam] for beam in ("p1", "p2"))
        report_suffix = ".collinear_test_ltd" if representation == "LTD" else ".collinear_test"
        report_path = pjoin(self.dot_folder, f"{subtracted_name}{report_suffix}.txt")
        json_path = pjoin(self.dot_folder, f"{subtracted_name}{report_suffix}.json")
        write_text_with_dirs(report_path, rendered_report)
        write_text_with_dirs(
            json_path,
            json.dumps(
                {
                    "precision": self.collinear_precision,
                    "use_arb_prec": use_arb_prec,
                    "display_digits": display_digits,
                    "display_note": (
                        "GammaLoop evaluates at the requested precision, but the "
                        "current Python API returns Python complex values to pygloop."
                    ),
                    "representation": representation,
                    "collinear_fraction_x": self.collinear_fraction_x,
                    "collinear_lambdas": list(self.collinear_lambdas),
                    "dot_path": dot_path,
                    "routed_dot_path": routed_dot_path,
                    "standalone_state_folder": manifest.get("standalone_state_folder"),
                    "exact_xi_routing": exact_xi_routing,
                    "results": numeric_results,
                },
                indent=2,
                sort_keys=True,
            ),
        )
        logger.info("\n%s", rendered_report)
        return {
            "report_path": report_path,
            "json_path": json_path,
            "dot_path": dot_path,
            "routed_dot_path": routed_dot_path,
            "standalone_state_folder": manifest.get("standalone_state_folder"),
            "precision": self.collinear_precision,
            "representation": representation,
            "collinear_fraction_x": self.collinear_fraction_x,
            "exact_xi_routing": exact_xi_routing,
            "tables": tables,
            "results": numeric_results,
        }

    def test_process_4d(self) -> dict[str, Any]:
        from processes.qqbar_nX.qqbar_nX_4d import build_4d_graph_evaluator

        configured_counterterm_global_phase = self.counterterm_global_phase
        self.counterterm_global_phase = self.four_d_counterterm_global_phase
        try:
            manifest = self.build_ir_subtracted_graphs(
                import_graphs=False,
                allow_cached_selected_dot=True,
            )
        finally:
            self.counterterm_global_phase = configured_counterterm_global_phase
        subtracted_name = self.get_subtracted_integrand_name()
        dot_path = manifest["subtracted_dot_path"]
        routed_dot_path = self._routed_dot_for_4d_test(
            manifest=manifest,
            integrand_name=subtracted_name,
        )

        grouped_reference_graphs = self._graph_members_by_group_from_dot(dot_path)
        grouped_graphs = self._graph_members_by_group_from_routed_dot(
            reference_dot_path=dot_path, routed_dot_path=routed_dot_path
        )
        exact_xi_routing = self._verify_exact_xi_routing(grouped_graphs)
        external_count = len(self._external_momenta_4d())
        model_values = self._sm_4d_model_values()
        graph_evaluators: dict[str, Any] = {}
        for graphs in grouped_graphs.values():
            for graph in graphs:
                name = graph_name(graph)
                if name in graph_evaluators:
                    continue
                logger.info("Building 4D Symbolica evaluator for %s.", name)
                graph_evaluators[name] = build_4d_graph_evaluator(
                    graph,
                    external_count=external_count,
                    model_values=model_values,
                    xi_parameter_names=self.xi_parameter_names,  # type: ignore[arg-type]
                    xi_default_values=self.xi_default_values,  # type: ignore[arg-type]
                )

        display_digits = 17
        tables: dict[str, str] = {}
        numeric_results: dict[str, Any] = {}

        for beam in ("p1", "p2"):
            beam_rows: list[dict[str, Any]] = []
            numeric_results[beam] = {}
            for group_id, graphs in grouped_graphs.items():
                reference_graphs = grouped_reference_graphs[group_id]
                master_graph = next(
                    (
                        graph
                        for graph in reference_graphs
                        if not graph_name(graph).endswith("_ct")
                    ),
                    reference_graphs[0],
                )
                routed_master_graph = next(
                    (graph for graph in graphs if not graph_name(graph).endswith("_ct")),
                    graphs[0],
                )
                graph_names = [graph_name(graph) for graph in graphs]
                external_momenta_by_name = {
                    graph_name(graph): self._external_momenta_for_graph_4d(graph)
                    for graph in graphs
                }
                numeric_results[beam][str(group_id)] = {}
                for lambda_value in self.collinear_lambdas:
                    loop_momenta = self._collinear_loop_momenta_4d(
                        master_graph,
                        beam=beam,
                        x_fraction=self.collinear_fraction_x,
                        lambda_value=lambda_value,
                        routed_graph=routed_master_graph,
                    )
                    member_values: list[tuple[str, complex]] = []
                    for name in graph_names:
                        evaluator = graph_evaluators[name]
                        evaluator.set_kinematics(
                            external_momenta=external_momenta_by_name[name],
                            loop_momenta=loop_momenta,
                            helicities=self._helicities_for_current_externals(),
                        )
                        member_values.append((name, evaluator.evaluate()))
                    values = [value for _name, value in member_values]
                    group_sum = sum(values, 0j)
                    sum_abs = sum(abs(value) for value in values)
                    ratio = abs(group_sum) / sum_abs if sum_abs != 0.0 else 0.0
                    beam_rows.append(
                        {
                            "lambda": lambda_value,
                            "group_id": group_id,
                            "members": member_values,
                            "sum": group_sum,
                            "sum_abs": sum_abs,
                            "ratio": ratio,
                        }
                    )
                    numeric_results[beam][str(group_id)][
                        _lambda_label(lambda_value)
                    ] = {
                        "loop_momenta_4d": loop_momenta,
                        "fake_xi_momentum": None,
                        "members": {
                            name: {"re": value.real, "im": value.imag}
                            for name, value in member_values
                        },
                        "sum": {"re": group_sum.real, "im": group_sum.imag},
                        "sum_abs": sum_abs,
                        "abs_sum_over_sum_abs": ratio,
                    }
            tables[beam] = self._render_collinear_cancellation_table(
                beam=beam,
                rows=beam_rows,
                display_digits=display_digits,
                backend_label="4D direct f64",
            )

        rendered_report = "\n\n".join(tables[beam] for beam in ("p1", "p2"))
        report_path = pjoin(self.dot_folder, f"{subtracted_name}.collinear_test_4d.txt")
        json_path = pjoin(self.dot_folder, f"{subtracted_name}.collinear_test_4d.json")
        write_text_with_dirs(report_path, rendered_report)
        write_text_with_dirs(
            json_path,
            json.dumps(
                {
                    "backend": "4D direct f64",
                    "display_digits": display_digits,
                    "collinear_fraction_x": self.collinear_fraction_x,
                    "collinear_lambdas": list(self.collinear_lambdas),
                    "dot_path": dot_path,
                    "routed_dot_path": routed_dot_path,
                    "model_values": {
                        key: {"re": value.real, "im": value.imag}
                        for key, value in model_values.items()
                    },
                    "exact_xi_routing": exact_xi_routing,
                    "results": numeric_results,
                },
                indent=2,
                sort_keys=True,
            ),
        )
        logger.info("\n%s", rendered_report)
        return {
            "report_path": report_path,
            "json_path": json_path,
            "dot_path": dot_path,
            "routed_dot_path": routed_dot_path,
            "backend": "4D direct f64",
            "collinear_fraction_x": self.collinear_fraction_x,
            "exact_xi_routing": exact_xi_routing,
            "tables": tables,
            "results": numeric_results,
        }

    def _ensure_cli_generated_state_for_cff_meta(
        self, *, manifest: dict[str, Any], integrand_name: str
    ) -> None:
        state_folder = manifest.get("standalone_state_folder")
        run_card_path = manifest.get("standalone_run_card_path")
        if not state_folder or not run_card_path:
            raise pygloopException(
                "qqbar_nX pygloop-cff test requires the standalone run card/state paths."
            )

        needs_rebuild = self.clean or not os.path.isdir(state_folder)
        if not needs_rebuild:
            probe = subprocess.run(
                [
                    self.gammaloop_cli_path,
                    "-s",
                    state_folder,
                    "display",
                    "processes",
                ],
                cwd=os.getcwd(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            needs_rebuild = (
                probe.returncode != 0
                or "Invalid export format" in probe.stdout
                or integrand_name not in probe.stdout
            )

        if not needs_rebuild:
            return

        log_prefix = f"{integrand_name}_cff_meta"
        self._run_standalone_command(
            label="load_run_card",
            log_prefix=log_prefix,
            command=[
                self.gammaloop_cli_path,
                "--clean-state",
                run_card_path,
                "quit",
                "-o",
                "true",
            ],
        )
        self._run_standalone_command(
            label="generate_subtracted_integrand",
            log_prefix=log_prefix,
            command=[
                self.gammaloop_cli_path,
                "-o",
                "-s",
                state_folder,
                "run",
                "generate_subtracted_integrand",
            ],
        )

    def test_process_pygloop_cff(self) -> dict[str, Any]:
        from processes.qqbar_nX.qqbar_nX_cff import (
            build_cff_graph_evaluator,
            load_or_build_cff_json,
        )

        manifest = self.build_ir_subtracted_graphs(
            import_graphs=False,
            allow_cached_selected_dot=True,
        )
        subtracted_name = self.get_subtracted_integrand_name()
        process_name = self.get_subtracted_process_name()
        state_folder = manifest["standalone_state_folder"]
        dot_path = manifest["subtracted_dot_path"]
        routed_dot_path = pjoin(self.dot_folder, f"{subtracted_name}.routed.dot")
        if not os.path.isfile(routed_dot_path):
            raise pygloopException(
                "pygloop-cff needs the routed DOT with GammaLoop's actual LMB "
                f"assignments. Missing {routed_dot_path}; run the 4D or "
                "GammaLoop collinear test once to create it."
            )

        self._ensure_cli_generated_state_for_cff_meta(
            manifest=manifest, integrand_name=subtracted_name
        )

        routed_graphs = parse_dot_graphs(open(routed_dot_path, encoding="utf-8").read())
        graph = next(
            (item for item in routed_graphs if graph_name(item) == self.cff_meta_graph_name),
            None,
        )
        if graph is None:
            raise pygloopException(
                f"Could not find graph {self.cff_meta_graph_name} in {routed_dot_path}."
            )

        cff_json_path = pjoin(
            self.dot_folder,
            "cff_debug",
            f"{self.cff_meta_graph_name}_cff.json",
        )
        cff_data = load_or_build_cff_json(
            api=None,
            gammaloop_cli_path=self.gammaloop_cli_path,
            state_folder=state_folder,
            process_name=process_name,
            integrand_name=subtracted_name,
            graph_name_value=self.cff_meta_graph_name,
            json_path=cff_json_path,
            representation="cff",
        )
        graph_id = int(cff_data.get("graph_id", 0))
        orientation_id = self.cff_meta_orientation_id

        edge_momentum_map = (
            self._fake_xi_edge_momentum_map()
            if self.uses_fake_xi_externals()
            else {
                index: momentum
                for index, momentum in enumerate(self._physical_momenta_4d())
            }
        )
        cff_external_edge_ids: list[int] = []
        for ext_name in cff_data.get("graph", {}).get("ext_names", []):
            if not str(ext_name).startswith("e"):
                raise pygloopException(f"Unexpected CFF external name {ext_name!r}.")
            edge_id = int(str(ext_name)[1:])
            if edge_id not in edge_momentum_map:
                raise pygloopException(
                    f"CFF external {ext_name!r} has no DOT edge momentum."
                )
            cff_external_edge_ids.append(edge_id)
        cff_external_momenta: list[list[float]] = [
            [0.0, 0.0, 0.0, 0.0]
            for _ in range((max(cff_external_edge_ids) + 1) if cff_external_edge_ids else 0)
        ]
        for edge_id in cff_external_edge_ids:
            cff_external_momenta[edge_id] = edge_momentum_map[edge_id]

        logger.info(
            "Building qqbar_nX custom CFF evaluator for graph %s, orientation %s.",
            self.cff_meta_graph_name,
            "all" if orientation_id is None else orientation_id,
        )
        evaluator = build_cff_graph_evaluator(
            graph,
            cff_data,
            external_count=len(cff_external_momenta),
            model_values=self._sm_4d_model_values(),
            orientation_id=orientation_id,
        )

        # Use the same simple momentum-space sample as the GammaLoop inspect
        # diagnostic. It is deliberately not summed over groups, so this first
        # parity target isolates one graph/orientation.
        xs = [
            1.0e-4,
            0.0,
            self.collinear_fraction_x * abs(float(self.ps_point[0].t)),
            0.25,
            0.125,
            -0.375,
        ]
        loop_spatial_momenta = [xs[:3], xs[3:]]
        evaluator.set_kinematics(
            external_momenta=cff_external_momenta,
            loop_spatial_momenta=loop_spatial_momenta,
            helicities=self._helicities_for_current_externals(),
            decimal_digit_precision=self.collinear_arb_display_digits,
        )
        custom_value = evaluator.evaluate()
        custom_decimal_re, custom_decimal_im = evaluator.evaluate_with_prec(
            self.collinear_arb_display_digits
        )

        precision_fragment, use_arb_prec, _display_digits = (
            self._collinear_precision_settings_fragment()
        )
        runtime_fragment = (
            _helicity_toml_fragment(self._helicities_for_current_externals())
            + precision_fragment
            + self._collinear_test_sampling_fragment()
        )
        self._run_standalone_command(
            label="cff_meta_runtime",
            command=[
                self.gammaloop_cli_path,
                "-o",
                "-s",
                state_folder,
                "set",
                "process",
                "-p",
                process_name,
                "-i",
                subtracted_name,
                "string",
                f"\n{runtime_fragment}",
            ],
            log_prefix=f"{subtracted_name}_cff_meta",
        )

        inspect_command = [
            self.gammaloop_cli_path,
            "-s",
            state_folder,
            "-o",
            "inspect",
            "-p",
            f"name:{process_name}",
            "-i",
            subtracted_name,
            "--graph-id",
            str(graph_id),
        ]
        if orientation_id is not None:
            inspect_command.extend(["--orientation-id", str(orientation_id)])
        if use_arb_prec:
            inspect_command.append("-f")
        inspect_command.extend(["-m", "-x", *[repr(value) for value in xs]])
        reference = self._run_inspect_and_parse(
            label=f"cff_meta_inspect_{self.cff_meta_graph_name}",
            command=inspect_command,
            log_prefix=f"{subtracted_name}_cff_meta",
            expected_accepted_events=1 if orientation_id is not None else None,
        )
        reference_value = complex(
            reference["result"]["re"],
            reference["result"]["im"],
        )
        reference_decimal_re = Decimal(reference["result"]["re_decimal"])
        reference_decimal_im = Decimal(reference["result"]["im_decimal"])
        difference = custom_value - reference_value
        decimal_difference_re = custom_decimal_re - reference_decimal_re
        decimal_difference_im = custom_decimal_im - reference_decimal_im
        with localcontext() as context:
            context.prec = max(self.collinear_arb_display_digits + 10, 50)
            decimal_difference_abs = (
                decimal_difference_re * decimal_difference_re
                + decimal_difference_im * decimal_difference_im
            ).sqrt()
            reference_decimal_abs = (
                reference_decimal_re * reference_decimal_re
                + reference_decimal_im * reference_decimal_im
            ).sqrt()
            decimal_relative_difference = (
                decimal_difference_abs / reference_decimal_abs
                if reference_decimal_abs != 0
                else Decimal("Infinity")
            )
        relative_difference = (
            abs(difference) / abs(reference_value) if reference_value != 0.0 else math.inf
        )

        report = {
            "graph_name": self.cff_meta_graph_name,
            "graph_id": graph_id,
            "orientation_id": orientation_id,
            "xs": xs,
            "cff_json_path": cff_json_path,
            "dot_path": dot_path,
            "routed_dot_path": routed_dot_path,
            "custom_value": {"re": custom_value.real, "im": custom_value.imag},
            "custom_value_decimal": {
                "re": str(custom_decimal_re),
                "im": str(custom_decimal_im),
            },
            "gammaloop_inspect_value": {
                "re": reference_value.real,
                "im": reference_value.imag,
            },
            "gammaloop_inspect_value_decimal": {
                "re": str(reference_decimal_re),
                "im": str(reference_decimal_im),
            },
            "cff_external_edge_ids": cff_external_edge_ids,
            "lmb_external_edge_map": evaluator.lmb_external_edge_map,
            "difference": {"re": difference.real, "im": difference.imag},
            "difference_decimal": {
                "re": str(decimal_difference_re),
                "im": str(decimal_difference_im),
            },
            "relative_difference": relative_difference,
            "relative_difference_decimal": str(decimal_relative_difference),
            "custom_decimal_precision": self.collinear_arb_display_digits,
            "gammaloop_arb_prec": use_arb_prec,
            "inspect_log_path": reference["log_path"],
        }
        report_path = pjoin(
            self.dot_folder,
            f"{subtracted_name}.cff_meta_parity.json",
        )
        write_text_with_dirs(report_path, json.dumps(report, indent=2, sort_keys=True))
        logger.info(
            "qqbar_nX custom CFF parity %s orientation %s: custom=%s %si, "
            "GammaLoop=%s %si, rel=%s",
            self.cff_meta_graph_name,
            "all" if orientation_id is None else orientation_id,
            f"{custom_decimal_re:+.17E}",
            f"{custom_decimal_im:+.17E}",
            f"{reference_decimal_re:+.17E}",
            f"{reference_decimal_im:+.17E}",
            f"{decimal_relative_difference:.3E}",
        )
        return report

    def run_standalone_collinear_tests(self) -> dict[str, Any]:
        manifest = self.build_ir_smoke_test_graphs()
        subtracted_name = self.get_integrand_name(suffix=self.smoke_test_suffix)
        process_name = self.get_subtracted_process_name()
        run_card_path = manifest["standalone_run_card_path"]
        state_folder = manifest["standalone_state_folder"]
        dot_path = manifest["subtracted_dot_path"]
        smoke_group_id, graph = self._original_graphs_by_group_from_dot(dot_path)[0]
        log_prefix = subtracted_name
        pm_helicities = self._helicities_for_current_externals([1, -1, 0, 0, 0])
        pp_helicities = self._helicities_for_current_externals([1, 1, 0, 0, 0])
        command_results: list[dict[str, Any]] = []

        def run_command(label: str, command: list[str]) -> dict[str, Any]:
            result = self._run_standalone_command(
                label=label, command=command, log_prefix=log_prefix
            )
            command_results.append(result)
            return result

        run_command(
            "load",
            [
                self.gammaloop_cli_path,
                "-o",
                "-s",
                state_folder,
                "--clean-state",
                run_card_path,
                "run",
                "load_subtracted_dot",
            ],
        )
        run_command(
            "generate",
            [
                self.gammaloop_cli_path,
                "-o",
                "-s",
                state_folder,
                "run",
                "generate_subtracted_integrand",
            ],
        )

        generic_xs = [0.11, 0.23, -0.31, 0.47, -0.59, 0.67]
        command_results.append(
            self._set_standalone_helicities(
                state_folder=state_folder,
                process_name=process_name,
                integrand_name=subtracted_name,
                helicities=pm_helicities,
                log_prefix=log_prefix,
            )
        )
        generic_pm = self._run_inspect_and_parse(
            label="inspect_generic_graph0_pm_f64",
            log_prefix=log_prefix,
            command=self._inspect_command(
                state_folder,
                process_name,
                subtracted_name,
                xs=generic_xs,
                graph_id=0,
            ),
            expected_accepted_events=1,
        )
        command_results.append(generic_pm["command"])

        command_results.append(
            self._set_standalone_helicities(
                state_folder=state_folder,
                process_name=process_name,
                integrand_name=subtracted_name,
                helicities=pp_helicities,
                log_prefix=log_prefix,
            )
        )
        generic_pp = self._run_inspect_and_parse(
            label="inspect_generic_graph0_pp_f64",
            log_prefix=log_prefix,
            command=self._inspect_command(
                state_folder,
                process_name,
                subtracted_name,
                xs=generic_xs,
                graph_id=0,
            ),
            expected_accepted_events=1,
        )
        command_results.append(generic_pp["command"])

        both_generic_helicities_zero = bool(
            generic_pm["exact_zero"] and generic_pp["exact_zero"]
        )
        if both_generic_helicities_zero:
            raise pygloopException(
                "Both +- and ++ graph-0 generic helicity probes are exactly zero; "
                "the qqbar_nX projector or helicity setup is likely wrong."
            )

        command_results.append(
            self._set_standalone_helicities(
                state_folder=state_folder,
                process_name=process_name,
                integrand_name=subtracted_name,
                helicities=pm_helicities,
                log_prefix=log_prefix,
            )
        )
        collinear_results: dict[str, Any] = {}
        for beam in ("p1", "p2"):
            beam_results: dict[str, Any] = {}
            for target_label, graph_id, discrete_dim, accepted_events in (
                ("graph0", 0, None, 1),
                ("group", None, (smoke_group_id,), 3),
            ):
                target_results: dict[str, Any] = {"f64": {}, "arb": {}}
                for lambda_value in (1.0e-2, 1.0e-4, 1.0e-6):
                    xs = self._collinear_xs_from_graph_fractional(
                        graph,
                        beam=beam,
                        x_fraction=self.collinear_fraction_x,
                        lambda_value=lambda_value,
                    )
                    lambda_label = _lambda_label(lambda_value)
                    label = (
                        f"collinear_{beam}_{target_label}_pm_f64_{lambda_label}"
                    )
                    parsed = self._run_inspect_and_parse(
                        label=label,
                        log_prefix=log_prefix,
                        command=self._inspect_command(
                            state_folder,
                            process_name,
                            subtracted_name,
                            xs=xs,
                            graph_id=graph_id,
                            discrete_dim=discrete_dim,
                        ),
                        expected_accepted_events=accepted_events,
                    )
                    command_results.append(parsed["command"])
                    parsed["xs"] = xs
                    parsed["precision"] = "f64"
                    parsed["lambda"] = lambda_value
                    target_results["f64"][lambda_label] = parsed

                for lambda_value in (1.0e-4, 1.0e-7):
                    xs = self._collinear_xs_from_graph_fractional(
                        graph,
                        beam=beam,
                        x_fraction=self.collinear_fraction_x,
                        lambda_value=lambda_value,
                    )
                    lambda_label = _lambda_label(lambda_value)
                    label = (
                        f"collinear_{beam}_{target_label}_pm_arb_{lambda_label}"
                    )
                    parsed = self._run_inspect_and_parse(
                        label=label,
                        log_prefix=log_prefix,
                        command=self._inspect_command(
                            state_folder,
                            process_name,
                            subtracted_name,
                            xs=xs,
                            graph_id=graph_id,
                            discrete_dim=discrete_dim,
                            use_arb_prec=True,
                        ),
                        expected_accepted_events=accepted_events,
                    )
                    command_results.append(parsed["command"])
                    parsed["xs"] = xs
                    parsed["precision"] = "arb"
                    parsed["lambda"] = lambda_value
                    target_results["arb"][lambda_label] = parsed
                beam_results[target_label] = target_results
            collinear_results[beam] = beam_results

        collinear_checks: dict[str, Any] = {}
        for beam, beam_results in collinear_results.items():
            collinear_checks[beam] = {}
            for target_label, target_results in beam_results.items():
                f64 = target_results["f64"]
                arb = target_results["arb"]
                arb_agrees_at_1e4 = self._complex_results_close(
                    f64["1em04"], arb["1em04"]
                )
                f64_stable_1e4_to_1e6 = self._complex_results_close(
                    f64["1em04"],
                    f64["1em06"],
                    abs_tol=Decimal("1e-24"),
                    rel_tol=Decimal("1e-3"),
                )
                collinear_checks[beam][target_label] = {
                    "arb_agrees_with_f64_at_1e-4": arb_agrees_at_1e4,
                    "f64_stable_from_1e-4_to_1e-6": f64_stable_1e4_to_1e6,
                    "accepted_events": {
                        "f64_1e-6": f64["1em06"]["accepted_events"],
                        "arb_1e-7": arb["1em07"]["accepted_events"],
                    },
                    "passed": bool(arb_agrees_at_1e4 and f64_stable_1e4_to_1e6),
                }
                if not collinear_checks[beam][target_label]["passed"]:
                    raise pygloopException(
                        "GammaLoop collinear inspect stability check failed for "
                        f"{beam}/{target_label}."
                    )

        low_stat_checks: dict[str, Any] = {}
        if self.run_low_stat_standalone_test:
            for helicities in (pm_helicities, pp_helicities):
                command_result, check = self._run_low_stat_integrate_for_helicity(
                    state_folder=state_folder,
                    integrand_name=subtracted_name,
                    helicities=helicities,
                    log_prefix=log_prefix,
                )
                command_results.append(command_result)
                low_stat_checks[_helicity_label(helicities)] = check
            if (
                low_stat_checks["pm"]["exact_zero_result_and_error"]
                and low_stat_checks["pp"]["exact_zero_result_and_error"]
            ):
                raise pygloopException(
                    "Both +- and ++ low-stat integration results are exactly zero."
                )

        checks: dict[str, Any] = {
            "helicity_support": {
                "pm": generic_pm,
                "pp": generic_pp,
                "not_both_exact_zero": not both_generic_helicities_zero,
                "passed": not both_generic_helicities_zero,
            },
            "collinear": collinear_checks,
            "low_stat_integrate": low_stat_checks,
        }
        standalone_test = {
            "gammaloop_cli_path": self.gammaloop_cli_path,
            "state_folder": state_folder,
            "commands": command_results,
            "checks": checks,
            "collinear_results": collinear_results,
            "notes": (
                "The p1/p2 scan solves the selected graph's internal light-quark "
                "lmb_rep so that the k1 spatial momentum approaches the requested "
                "beam direction. The -f GammaLoop inspect path is recorded as arb/f128."
            ),
        }
        manifest["standalone_collinear_test"] = standalone_test
        manifest["standalone_smoke_test"] = standalone_test
        manifest_path = pjoin(self.dot_folder, f"{subtracted_name}.manifest.json")
        write_text_with_dirs(
            manifest_path, json.dumps(manifest, indent=2, sort_keys=True)
        )
        return manifest

    def run_standalone_smoke_tests(self) -> dict[str, Any]:
        return self.run_standalone_collinear_tests()

    def test_ir_counterterm_import(self) -> dict[str, Any]:
        manifest = self.build_ir_subtracted_graphs()
        subtracted_name = self.get_subtracted_integrand_name()
        dot_echo = self.require_gl_worker().get_dot_files(integrand_name=subtracted_name)
        imported_graph_count = len(parse_dot_graphs(dot_echo))
        manifest["import_test"] = {
            "integrand_name": subtracted_name,
            "imported_graph_count": imported_graph_count,
        }
        return manifest

    def generate_gammaloop_code(self) -> None:
        manifest = self.build_ir_subtracted_graphs()
        if not self.generate_existing_after_selection:
            logger.warning(
                "qqbar_nX GammaLoop code generation is disabled in config. "
                "The IR-subtracted DOT and standalone run card were still written to %s.",
                manifest["subtracted_dot_path"],
            )
            return

        worker = self.require_gl_worker()
        amplitudes, _cross_sections = worker.list_outputs()
        subtracted_name = self.get_subtracted_integrand_name()
        if subtracted_name not in amplitudes:
            raise pygloopException("Build qqbar_nX IR-subtracted graphs before generating code.")
        worker.run("set global kv global.generation.uv.subtract_uv=false")
        worker.run("set global kv global.generation.uv.generate_integrated=false")
        worker.run(
            "set global kv "
            "global.generation.uv.local_uv_cts_from_expanded_4d_integrands=false"
        )
        worker.run(f"generate existing -p {amplitudes[subtracted_name]} -i {subtracted_name}")
        self.save_state()

    def generate_spenso_code(self, *args: Any, **opts: Any) -> None:
        logger.warning("qqbar_nX Spenso code generation is not implemented yet.")

    def parameterize(
        self, xs: list[float], parameterisation: str, origin: Vector | None = None
    ) -> tuple[Vector, float]:
        raise pygloopException("qqbar_nX parameterization is not implemented yet.")

    def integrand_xspace(
        self,
        xs: list[float],
        parameterization: str,
        integrand_implementation: dict[str, Any],
        phase: str | bool | None = None,
        multi_channeling: bool | int = True,
    ) -> float:
        raise pygloopException("qqbar_nX x-space integrand is not implemented yet.")

    def integrand(
        self,
        loop_momenta: list[Vector],
        integrand_implementation: dict[str, Any],
    ) -> complex:
        raise pygloopException(
            "qqbar_nX integrand evaluation awaits the IR counterterm construction stage."
        )

    def integrate(
        self,
        integrator: str,
        parameterisation: str,
        integrand_implementation: dict[str, Any],
        target: float | complex | None = None,
        toml_config_path: str | None = None,
        **opts: Any,
    ) -> IntegrationResult:
        raise pygloopException(
            "qqbar_nX integration awaits the IR counterterm construction stage."
        )

    def plot(self, **opts: Any) -> None:
        raise pygloopException("qqbar_nX plotting is not implemented yet.")


QQbarNX = qqbar_nX
