import logging
import math
import os
from enum import StrEnum
from functools import wraps
from pprint import pprint  # noqa: F401
from typing import Any, Iterator

import numpy
import pydot
from numpy.typing import NDArray
from pydot import Edge, Node  # noqa: F401
from symbolica import E, Evaluator, Expression, Replacement, S, Sample

from utils.vectors import LorentzVector, Vector  # noqa: F401

PYGLOOP_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PYGLOOP_FOLDER, "src")

OUTPUTS_FOLDER = os.path.join(PYGLOOP_FOLDER, "outputs")
DOTS_FOLDER = os.path.join(OUTPUTS_FOLDER, "dot_files")
INTEGRATION_WORKSPACE_FOLDER = os.path.join(OUTPUTS_FOLDER, "integration_workspaces")
EVALUATORS_FOLDER = os.path.join(OUTPUTS_FOLDER, "evaluators")
GAMMALOOP_STATES_FOLDER = os.path.join(OUTPUTS_FOLDER, "gammaloop_states")
CONFIGS_FOLDER = os.path.join(PYGLOOP_FOLDER, "configs")

np_cmplx_one = numpy.complex128(1.0, 0.0)
np_cmplx_zero = numpy.complex128(0.0, 0.0)


def setup_logging():
    logging.basicConfig(
        format=f"{Colour.GREEN}%(levelname)s{Colour.END} {Colour.BLUE}%(funcName)s l.%(lineno)d{Colour.END} {Colour.CYAN}t=%(asctime)s.%(msecs)03d{Colour.END} > %(message)s",  # fmt: off
        datefmt="%Y-%m-%d,%H:%M:%S",
    )


logger = logging.getLogger("pygloop")


class pygloopException(Exception):
    pass


def set_gammaloop_level(enter_level: int, exit_level: int):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self.set_log_level(enter_level)
            try:
                return func(self, *args, **kwargs)
            finally:
                self.set_log_level(exit_level)

        return wrapper

    return decorator


def set_tmp_logger_level(level: int):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            previous_level = logger.level
            logger.setLevel(level)
            try:
                return func(*args, **kwargs)
            finally:
                logger.setLevel(previous_level)

        return wrapper

    return decorator


class Colour(StrEnum):
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


GAMMA_EPS_EXPANSIONS = [
    Replacement(E("dim"), E("4 - 2*ε")),
    Replacement(E("𝚪(1-ε)"), E("1 + γₑ*ε + (1/12)*( 6*γₑ^2 + 𝜋^2)*ε^2 + ε^3*O(Gamma,eps^3)")),
    Replacement(E("𝚪(1+ε)"), E("1 - γₑ*ε + (1/12)*( 6*γₑ^2 + 𝜋^2)*ε^2 + ε^3*O(Gamma,eps^3)")),
    Replacement(E("𝚪(1-b_*ε)"), E("1 + γₑ*b_*ε + (1/12)*( 6*γₑ^2 + 𝜋^2)*b_^2*ε^2 + ε^3*O(Gamma,eps^3)")),
    Replacement(E("𝚪(1+b_*ε)"), E("1 - γₑ*b_*ε + (1/12)*( 6*γₑ^2 + 𝜋^2)*b_^2*ε^2 + ε^3*O(Gamma,eps^3)")),
    Replacement(E("𝚪(ε)"), E("1/ε - γₑ + (1/12)*( 6*γₑ^2 + 𝜋^2)*ε + ε^2*O(Gamma,eps^2)")),
    Replacement(E("𝚪(b_*ε)"), E("1/(b_*ε) - γₑ + (1/12)*( 6*γₑ^2 + 𝜋^2)*b_*ε + ε^2*O(Gamma,eps^2)")),
    Replacement(E("𝚪(2-ε)"), E("1 + (γₑ-1)*ε + (1/12)*( -12*γₑ + 6 * γₑ^2 + 𝜋^2)*ε^2 + ε^3*O(Gamma,eps^3)")),
    Replacement(E("𝚪(2+ε)"), E("1 + (1-γₑ)*ε + (1/12)*( -12*γₑ + 6 * γₑ^2 + 𝜋^2)*ε^2 + ε^3*O(Gamma,eps^3)")),
    Replacement(E("𝚪(2-b_*ε)"), E("1 + (γₑ-1)*b_*ε + (1/12)*( -12*γₑ + 6 * γₑ^2 + 𝜋^2)*b_^2*ε^2 + ε^3*O(Gamma,eps^3)")),
    Replacement(E("𝚪(2+b_*ε)"), E("1 + (1-γₑ)*b_*ε + (1/12)*( -12*γₑ + 6 * γₑ^2 + 𝜋^2)*b_^2*ε^2 + ε^3*O(Gamma,eps^3)")),
]


def eps_expansion_finite(expr: Expression, coeff_index: int = -1) -> Expression:
    expansion = expr.replace_multiple(GAMMA_EPS_EXPANSIONS).series(E("ε"), 0, 0, depth_is_absolute=True).to_expression().coefficient_list(E("ε"))
    if coeff_index is None:
        return expansion
    else:
        return expansion[coeff_index][-1]


def expr_to_string(expr: Expression) -> str:
    """Convert a symbolica expression to string."""
    # return expr.to_canonical_string()
    return expr.format_plain()


# Work around expressions given as strings containing the wrapping quotes
def Es(expr: str) -> Expression:
    return E(expr.replace('"', ""), default_namespace="gammalooprs")


def chunks(a_list: list[Any], n: int) -> Iterator[list[Any]]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(a_list), n):
        yield a_list[i : i + n]


class ParamBuilder(list):
    def __init__(self, cache: Any = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positions: dict[tuple[Expression, ...], tuple[int, int]] = {}
        self.order: list[tuple[Expression, ...]] = []
        self.np = numpy.zeros(0, complex)
        if cache is None:
            self.cache = {}
        else:
            self.cache = cache

    def add_parameter_list(self, head: tuple[Expression, ...], length: int):
        if head in self.positions:
            raise pygloopException(f"Parameter {head} already exists")

        self.positions[head] = (len(self.np), len(self.np) + length)
        self.order.append(head)
        self.np = numpy.resize(self.np, len(self.np) + length)

    def add_parameter(self, param: tuple[Expression, ...]):
        return self.add_parameter_list(param, 1)

    def set_parameter_values(self, head: tuple[Expression, ...], values: list[complex] | NDArray[Any]):
        if head not in self.positions:
            raise pygloopException(f"Could not find parameter {head} in param builder.")

        min, max = self.positions[head]
        if (max - min) != len(values):
            raise pygloopException(f"Length of parameters {head} declared as {max - min}, but {len(values)} values are provided.")
        self.np[min:max] = values

    def set_parameter_values_within_range(self, min: int, max: int, values: list[complex] | NDArray[Any]):
        if (max - min) != len(values):
            raise pygloopException(f"Range declared of ({min},{max}) of different length that the number of values ({len(values)}) provided.")
        self.np[min:max] = values

    def set_parameter(self, param: tuple[Expression, ...], value: complex):
        return self.set_parameter_values(param, [value,])  # fmt: off

    def get_parameters(self):
        params = []
        for p in self.order:
            min, max = self.positions[p]
            if max - min == 1:
                if len(p[1:]) == 0:
                    params.append(p[0])
                else:
                    params.append(p[0](*p[1:]))
            else:
                params.extend(p[0](*p[1:], i) for i in range(max - min))

        return params

    def get_values(self) -> NDArray[numpy.complex128]:
        return self.np


class PygloopEvaluator(object):
    def __init__(self, evaluator: Evaluator, param_builder: ParamBuilder):
        self.evaluator = evaluator
        self.param_builder = param_builder

    def __call__(self) -> NDArray[numpy.complex128]:
        return self.evaluator.evaluate_complex(self.param_builder.get_values()[None, :])[0]


class SymbolicaSample(object):
    def __init__(self, sample: Sample):
        self.c: list[float] = sample.c
        self.d: list[int] = sample.d


class IntegrationResult(object):
    def __init__(
        self,
        central_value: float,
        error: float,
        n_samples: int = 0,
        elapsed_time: float = 0.0,
        max_wgt: float | None = None,
        max_wgt_point: list[float] | None = None,
    ):
        self.n_samples = n_samples
        self.central_value = central_value
        self.error = error
        self.max_wgt = max_wgt
        self.max_wgt_point = max_wgt_point
        self.elapsed_time = elapsed_time

    def combine_with(self, other):
        """Combine self statistics with all those of another IntegrationResult object."""
        self.n_samples += other.n_samples
        self.elapsed_time += other.elapsed_time
        self.central_value += other.central_value
        self.error += other.error
        if other.max_wgt is not None:
            if self.max_wgt is None or abs(self.max_wgt) > abs(other.max_wgt):
                self.max_wgt = other.max_wgt
                self.max_wgt_point = other.max_wgt_point

    def normalize(self):
        """Normalize the statistics."""
        self.central_value /= self.n_samples
        self.error = math.sqrt(abs(self.error / self.n_samples - self.central_value**2) / self.n_samples)

    def str_report(self, target: float | None = None) -> str:
        if self.central_value == 0.0 or self.n_samples == 0:
            return "No integration result available yet"

        # First printout sample and timing statitics
        report = [
            f"Integration result after {Colour.GREEN}{self.n_samples}{Colour.END} evaluations in {Colour.GREEN}{self.elapsed_time:.2f} CPU-s{Colour.END}"  # fmt: off
        ]
        if self.elapsed_time > 0.0:
            report[-1] += f" {Colour.BLUE}({1.0e6 * self.elapsed_time / self.n_samples:.1f} µs / eval){Colour.END}"

        # Also indicate max weight encountered if provided
        if self.max_wgt is not None and self.max_wgt_point is not None:
            report.append(f"Max weight encountered = {self.max_wgt:.5e} at xs = [{' '.join(f'{x:.16e}' for x in self.max_wgt_point)}]")  # fmt: off

        # Finally return information about current best estimate of the central value
        report.append(f"{Colour.GREEN}Central value{Colour.END} : {self.central_value:<+25.16e} +/- {self.error:<12.2e}")  # fmt: off

        err_perc = abs(self.error / self.central_value) * 100
        if err_perc < 1.0:
            report[-1] += f" ({Colour.GREEN}{err_perc:.3f}%{Colour.END})"
        else:
            report[-1] += f" ({Colour.RED}{err_perc:.3f}%{Colour.END})"

        # Also indicate distance to target if specified
        if target is not None and target != 0.0:
            report.append(f"    vs target : {target:<+25.16e} Δ = {self.central_value - target:<+12.2e}")  # fmt: off
            diff_perc = (self.central_value - target) / target * 100
            if abs(diff_perc) < 1.0:
                report[-1] += f" ({Colour.GREEN}{diff_perc:.3f}%{Colour.END}"
            else:
                report[-1] += f" ({Colour.RED}{diff_perc:.3f}%{Colour.END}"
            if abs(diff_perc / err_perc) < 3.0:
                report[-1] += f" {Colour.GREEN} = {abs(diff_perc / err_perc):.2f}σ{Colour.END})"  # fmt: off
            else:
                report[-1] += f" {Colour.RED} = {abs(diff_perc / err_perc):.2f}σ{Colour.END})"  # fmt: off

        # Join all lines and return
        return "\n".join(f"| > {line}" for line in report)


def write_text_with_dirs(path: str, content: str, mode: str = "w", encoding: str = "utf-8") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode, encoding=encoding) as handle:
        handle.write(content)


class DotGraph(object):
    def __init__(self, dot_graph: pydot.Dot):
        self.dot = dot_graph

    def get_attributes(self) -> dict:
        return self.dot.get_attributes()

    def get_numerator(self, include_overall_factor=False) -> Expression:
        num = E("1")
        for node in self.dot.get_nodes():
            if node.get_name() not in ["edge", "node"]:
                n_num = node.get("num")
                if n_num:
                    num *= Es(n_num)
        for edge in self.dot.get_edges():
            e_num = edge.get("num")
            if e_num:
                num *= Es(e_num)

        g_attrs = self.dot.get_attributes()
        if "num" in g_attrs:
            num *= Es(g_attrs["num"])
        if include_overall_factor and "overall_factor_evaluated" in g_attrs:
            num *= Es(g_attrs["overall_factor_evaluated"])

        return num

    def set_local_numerators_to_one(self):
        for node in self.dot.get_nodes():
            if node.get_name() not in ["edge", "node"]:
                node.set("num", "1")
        for edge in self.dot.get_edges():
            edge.set("num", "1")
        self.dot.set_edge_defaults(num='"1"')
        self.dot.set_node_defaults(num='"1"')

    def get_internal_edges(self) -> list[Edge]:
        internal_nodes = [
            n.get_name() for n in self.dot.get_nodes() if not any(marker in n.get_name() for marker in ["graph", "ext", "edge", "node"])
        ]
        external_edges = []
        for edge in self.dot.get_edges():
            source = edge.get_source().split(":")[0]  # type: ignore
            destination = edge.get_destination().split(":")[0]  # type: ignore
            if source in internal_nodes and destination in internal_nodes:
                external_edges.append(edge)

        return external_edges

    def get_external_edges(self) -> list[Edge]:
        internal_nodes = [
            n.get_name() for n in self.dot.get_nodes() if not any(marker in n.get_name() for marker in ["graph", "ext", "edge", "node"])
        ]
        external_edges = []
        for edge in self.dot.get_edges():
            source = edge.get_source().split(":")[0]  # type: ignore
            destination = edge.get_destination().split(":")[0]  # type: ignore
            if not (source in internal_nodes and destination in internal_nodes):
                external_edges.append(edge)

        return external_edges

    def get_propagator_denominators(self) -> Expression:
        den = E("1")
        for edge in self.get_internal_edges():
            attrs = edge.get_attributes()
            a_den = E(f"gammalooprs::Q({edge.get('id')},spenso::cind(0))^2")
            a_den -= E(f"gammalooprs::Q({edge.get('id')},spenso::cind(1))^2")
            a_den -= E(f"gammalooprs::Q({edge.get('id')},spenso::cind(2))^2")
            a_den -= E(f"gammalooprs::Q({edge.get('id')},spenso::cind(3))^2")
            if "mass" in attrs:
                a_den -= Es(f"{attrs.get('mass')}^2")
            den *= a_den
        return den

    def get_projector(self) -> Expression:
        g_attrs = self.dot.get_attributes()
        projector = None
        if "projector" in g_attrs:
            projector = Es(g_attrs["projector"])
        else:
            projector = Es(self.dot.get_graph_defaults()[0]["projector"])

        # TMPVH temporary fix to current issue in gammaloop when building external proectors
        # projector = projector.replace(E("gammalooprs::u(2,x__)"),E("gammalooprs::vbar(2,x__)"),repeat=True)

        return projector

    def get_emr_replacements(self, head="gammalooprs::Q") -> list[tuple[Expression, Expression]]:
        replacements = []
        for edge in self.dot.get_edges():
            replacements.append((E(f"{head}({edge.get('id')},gammalooprs::a___)"), Es(edge.get("lmb_rep"))))
        return replacements

    def to_string(self) -> str:
        return self.dot.to_string()


class DotGraphs(list):
    def __init__(self, dot_str: str | None = None, dot_path: str | None = None):
        if dot_str is None and dot_path is None:
            return
        if dot_path is not None and dot_str is not None:
            raise pygloopException("Only one of dot_str or dot_path should be provided.")

        if dot_path:
            dot_graphs = pydot.graph_from_dot_file(dot_path)
            if dot_graphs is None:
                raise ValueError(f"No graphs found in DOT file: {dot_path}")
            self.extend([DotGraph(g) for g in dot_graphs])
        elif dot_str:
            dot_graphs = pydot.graph_from_dot_data(dot_str)
            if dot_graphs is None:
                raise ValueError("No graphs found in DOT data string.")
            self.extend([DotGraph(g) for g in dot_graphs])

    def get_graph_names(self) -> list[str]:
        return [g.dot.get_name() for g in self]

    def __str__(self) -> str:
        return "\n\n".join([g.to_string() for g in self])

    def get_graph(self, graph_name) -> DotGraph:
        for g in self:
            if g.dot.get_name() == graph_name:
                return g
        raise KeyError(f"Graph with name {graph_name} not found.")

    def save_to_file(self, file_path: str):
        write_text_with_dirs(file_path, "\n\n".join([g.to_string() for g in self]))
