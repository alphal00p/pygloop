import logging
import math
import os
from enum import StrEnum
from functools import wraps
from typing import Any, Iterator
from itertools import combinations
from collections import deque

import pydot
import re
from pprint import pprint
from typing import Optional, Tuple, List
from symbolica import E, Expression, Replacement, Sample

PYGLOOP_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PYGLOOP_FOLDER, "src")

OUTPUTS_FOLDER = os.path.join(PYGLOOP_FOLDER, "outputs")
DOTS_FOLDER = os.path.join(OUTPUTS_FOLDER, "dot_files")
INTEGRATION_WORKSPACE_FOLDER = os.path.join(OUTPUTS_FOLDER, "integration_workspaces")
EVALUATORS_FOLDER = os.path.join(OUTPUTS_FOLDER, "evaluators")
GAMMALOOP_STATES_FOLDER = os.path.join(OUTPUTS_FOLDER, "gammaloop_states")
CONFIGS_FOLDER = os.path.join(PYGLOOP_FOLDER, "configs")


def setup_logging():
    logging.basicConfig(
        format=f"{Colour.GREEN}%(levelname)s{Colour.END} {Colour.BLUE}%(funcName)s l.%(lineno)d{Colour.END} {Colour.CYAN}t=%(asctime)s.%(msecs)03d{Colour.END} > %(message)s",  # nopep8
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
            f"Integration result after {Colour.GREEN}{self.n_samples}{Colour.END} evaluations in {Colour.GREEN}{self.elapsed_time:.2f} CPU-s{
                Colour.END
            }"
        ]
        if self.elapsed_time > 0.0:
            report[-1] += f" {Colour.BLUE}({1.0e6 * self.elapsed_time / self.n_samples:.1f} µs / eval){Colour.END}"

        # Also indicate max weight encountered if provided
        if self.max_wgt is not None and self.max_wgt_point is not None:
            report.append(f"Max weight encountered = {self.max_wgt:.5e} at xs = [{' '.join(f'{x:.16e}' for x in self.max_wgt_point)}]")  # nopep8

        # Finally return information about current best estimate of the central value
        report.append(f"{Colour.GREEN}Central value{Colour.END} : {self.central_value:<+25.16e} +/- {self.error:<12.2e}")  # nopep8

        err_perc = abs(self.error / self.central_value) * 100
        if err_perc < 1.0:
            report[-1] += f" ({Colour.GREEN}{err_perc:.3f}%{Colour.END})"
        else:
            report[-1] += f" ({Colour.RED}{err_perc:.3f}%{Colour.END})"

        # Also indicate distance to target if specified
        if target is not None and target != 0.0:
            report.append(f"    vs target : {target:<+25.16e} Δ = {self.central_value - target:<+12.2e}")  # nopep8
            diff_perc = (self.central_value - target) / target * 100
            if abs(diff_perc) < 1.0:
                report[-1] += f" ({Colour.GREEN}{diff_perc:.3f}%{Colour.END}"
            else:
                report[-1] += f" ({Colour.RED}{diff_perc:.3f}%{Colour.END}"
            if abs(diff_perc / err_perc) < 3.0:
                report[-1] += f" {Colour.GREEN} = {abs(diff_perc / err_perc):.2f}σ{Colour.END})"  # nopep8
            else:
                report[-1] += f" {Colour.RED} = {abs(diff_perc / err_perc):.2f}σ{Colour.END})"  # nopep8

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

    def get_propagator_denominators(self) -> Expression:
        internal_nodes = [
            n.get_name() for n in self.dot.get_nodes() if not any(marker in n.get_name() for marker in ["graph", "ext", "edge", "node"])
        ]
        den = E("1")
        for edge in self.dot.get_edges():
            source = edge.get_source().split(":")[0]  # type: ignore
            destination = edge.get_destination().split(":")[0]  # type: ignore
            attrs = edge.get_attributes()
            if source in internal_nodes and destination in internal_nodes:
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

    def get_emr_replacements(self) -> list[Expression]:
        replacements = []
        for edge in self.dot.get_edges():
            replacements.append((E(f"gammalooprs::Q({edge.get('id')},gammalooprs::a___)"), Es(edge.get("lmb_rep"))))
        return replacements

    def to_string(self) -> str:
        return self.dot.to_string()

    @staticmethod
    def _strip_quotes(s: str) -> str:
        s = s.strip()
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            return s[1:-1]
        return s

    def _is_ext(self, name: str) -> bool:
        return bool(re.fullmatch(r"ext\d+", DotGraph._strip_quotes(name)))

    def _parse_port_endpoint(self, endpoint: str) -> Optional[Tuple[str, int]]:
        """
        Parse 'a:b' (possibly quoted) -> ('a', b). Returns None if not matching.
        """
        ep = DotGraph._strip_quotes(endpoint)
        m = re.fullmatch(r"([^:]+):(\d+)", ep)
        if not m:
            return None
        return m.group(1), int(m.group(2))

    def _edge_particle(self,e: pydot.Edge) -> str:
        attrs = e.get_attributes() or {}
        p = attrs.get("particle", "")
        return DotGraph._strip_quotes(str(p))

    def is_incoming_half_edge(self,e) -> bool:
        """True iff edge is of form ext_i -> a:b."""
        src = e.get_source()
        dst = e.get_destination()
        return self._is_ext(src) and (self._parse_port_endpoint(dst) is not None)

    def is_outgoing_half_edge(self,e) -> bool:
        """True iff edge is of form a:b -> ext_i."""
        src = e.get_source()
        dst = e.get_destination()
        return self._is_ext(dst) and (self._parse_port_endpoint(src) is not None)

    def get_part(self,e):
        attrs=e.get_attributes()
        return attrs["particle"]

    def copy_edge(self, e: pydot.Edge) -> pydot.Edge:
        attrs = dict(e.get_attributes() or {})
        return pydot.Edge(e.get_source(), e.get_destination(), **attrs)

    def _non_ext_endpoint(self, e: pydot.Edge) -> str:
        """Return the endpoint (source or destination) that is NOT ext*."""
        src = e.get_source()
        dst = e.get_destination()

        src_is_ext = self._is_ext(src)
        dst_is_ext = self._is_ext(dst)

        if src_is_ext and not dst_is_ext:
            return dst
        if dst_is_ext and not src_is_ext:
            return src

        raise ValueError(
            f"Expected exactly one ext* endpoint, got: {src} -> {dst}"
        )

    def remove_edge_attr(self, e: pydot.Edge, key: str) -> None:
        if e is None:
            raise ValueError("remove_edge_attr got None edge")

        attrs = e.get_attributes() or {}
        attrs.pop(key, None)   # in-place usually works in pydot
        return e

    def edge_fusion(self, e1: pydot.Edge, e2: pydot.Edge) -> pydot.Edge:
        """
        Return a new edge connecting the non-ext endpoint of e1 to the non-ext
        endpoint of e2, copying ALL attributes from e1.
        """
        u = self._non_ext_endpoint(e1)
        v = self._non_ext_endpoint(e2)

        # Copy attributes from e1
        attrs = dict(e1.get_attributes() or {})
        attrs["is_cut"] = "1"

        e_new=self.remove_edge_attr(pydot.Edge(u, v, **attrs),"lmb_rep")

        # Create the fused edge
        return e_new

    def _parse_port(self,endpoint: str) -> Optional[int]:
        """Return b from 'a:b' (possibly quoted), else None."""
        ep = DotGraph._strip_quotes(endpoint)
        m = re.fullmatch(r"[^:]+:(\d+)", ep)
        return None if not m else int(m.group(1))

    def sort_half_edges_by_port(self, edges: List[pydot.Edge], reverse: bool = False) -> List[pydot.Edge]:
        """
        Return a NEW list sorted by the port index b of the non-ext endpoint 'a:b'.
        Ties are broken deterministically by endpoint string and then edge id (if present).
        """
        def edge_id_int(e: pydot.Edge) -> int:
            attrs = e.get_attributes() or {}
            try:
                return int(self._strip_quotes(str(attrs.get("id", ""))))
            except Exception:
                return 10**18

        def key(e: pydot.Edge):
            end = self._non_ext_endpoint(e)
            b = self._parse_port(end)
            if b is None:
                raise ValueError(f"Non-ext endpoint is not of form a:b: {end}")
            return (b, self._strip_quotes(end), edge_id_int(e))

        return sorted(edges, key=key, reverse=reverse)

    def get_vacuum_graph(self):
        incoming_edges=[]
        outgoing_edges=[]
        paired_up=[]
        new_edges=[]

        vacuum_graph = pydot.Dot(graph_type="digraph")

        for e in self.dot.get_edges():
            if self.is_incoming_half_edge(e):
                incoming_edges.append(e)
            elif self.is_outgoing_half_edge(e):
                outgoing_edges.append(e)
            else:
                ep=self.copy_edge(e)
                ep.set("is_cut", "0")
                vacuum_graph.add_edge(self.remove_edge_attr(ep,"lmb_rep"))

        if len(incoming_edges)!=len(outgoing_edges):
            raise pygloopException("Vacuum graph is not balanced.")

        incoming_edges = self.sort_half_edges_by_port(incoming_edges)
        outgoing_edges = self.sort_half_edges_by_port(outgoing_edges)

        for e in incoming_edges:
            for ep in outgoing_edges:
                if self.get_part(e)==self.get_part(ep) and ep not in paired_up:
                    vacuum_graph.add_edge(self.remove_edge_attr(self.edge_fusion(e,ep),"lmb_rep"))
                    paired_up.append(ep)

        #TODO: RETURN VACUUM GRAPH
        for e in vacuum_graph.get_edges():
            pprint(str(e))

    def _base_node(self, endpoint: str) -> str:
        ep = DotGraph._strip_quotes(endpoint)
        parsed = self._parse_port_endpoint(ep)
        return parsed[0] if parsed else ep

    def _all_node_names(self):
        seen = set()
        out = []
        for e in self.dot.get_edges():
            for raw in (e.get_source(), e.get_destination()):
                name = self._base_node(raw)
                if name not in seen:
                    seen.add(name)
                    out.append(name)
        return out

    def is_connected(self, node_subset_input) -> bool:
        """
        True iff the induced subgraph on node_subset_input (base node names) is connected.
        Uses edges from self.dot.get_edges() and treats them as undirected for connectivity.
        """
        subset = set(node_subset_input.copy())
        if not subset:
            return True
        if len(subset) == 1:
            return True

        start = next(iter(subset))
        visited = {start}
        q = deque([start])

        while q:
            x = q.popleft()

            # scan edges to find neighbors of x inside subset
            for e in self.dot.get_edges():
                u = self._base_node(e.get_source())
                v = self._base_node(e.get_destination())

                # only consider induced edges (both endpoints in subset)
                if u not in subset or v not in subset:
                    continue

                if u == x and v not in visited:
                    visited.add(v)
                    q.append(v)
                elif v == x and u not in visited:
                    visited.add(u)
                    q.append(u)

            if len(visited) == len(subset):
                return True

        return len(visited) == len(subset)

    def contains_ext(self, node_subset, left_externals, right_externals):

        component_edges=[]

        for e in self.dot.get_edges():
            if self._base_node(e.get_source()) in node_subset and self._base_node(e.get_destination()) in node_subset:
                component_edges.append(e)

        stack=left_externals.copy()

        for e in component_edges:
            if e in left_externals:
                stack.remove(e)
            if e in right_externals:
                return False

        if len(stack)==0:
            return True
        return False

    def boundary_edges(self, S):

        out = []
        for e in self.dot.get_edges():
            u = self._base_node(e.get_source())
            v = self._base_node(e.get_destination())
            if (u in S and v not in S) or (u not in S and v in S):
                out.append(e)
        return out

    def get_incoming_edges(self):
        incoming_edges=[]
        for e in self.dot.get_edges():
            if self.is_incoming_half_edge(e):
                incoming_edges.append(e)
        return incoming_edges

    def get_outgoing_edges(self):
        outgoing_edges=[]
        for e in self.dot.get_edges():
            if self.is_outgoing_half_edge(e):
                outgoing_edges.append(e)
        return outgoing_edges

    def enumerate_cutkosky_cuts(self, left_externals, right_externals):

        nodes = set(self._all_node_names())
        all_cuts = [set(s) for r in range(min(len(left_externals),len(right_externals)),len(nodes)+1-min(len(left_externals),len(right_externals))) for s in combinations(nodes, r)]

        good_sets = [
            S for S in all_cuts
            if self.is_connected(S)
                and self.is_connected(nodes - S)
                and self.contains_ext(S, left_externals, right_externals)
            ]

        return good_sets




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

    def save_to_file(self, file_path: str):
        write_text_with_dirs(file_path, "\n\n".join([g.to_string() for g in self]))

    def filter_particle_definition(self, particles):
        newgraphs=[]

        for graph in self:
            incoming_edges=graph.get_incoming_edges()
            outgoing_edges=graph.get_outgoing_edges()
            cutkosky_cuts=graph.enumerate_cutkosky_cuts(incoming_edges,outgoing_edges)

            for c in cutkosky_cuts:
                stack=particles.copy()
                for e in c:
                    if e.get_attributes().get("particle", "") in stack:
                        stack.remove(e.get_attributes().get("particle", ""))
                if len(stack)==0:
                    newpgraphs.append(graph)

        self=newgraphs
