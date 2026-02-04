import os
from itertools import combinations, product
from types import ModuleType
from typing import Iterator, List, Optional, Set, Tuple

try:
    import sympy as sp  # pyright: ignore
except ImportError:
    sp: Optional[ModuleType] = None
    print(
        "Failed to import sympy for DY process (you won't be able to run it then). But honestly that's ok... almighty Symbolica should be used instead."
    )
import copy

import pydot
from symbolica import E, Expression # pyright: ignore

from processes.dy.dy_graph_utils import (
    _all_node_names,
    _base_node,
    _edge_is_cut_value,
    _edge_particle,
    _is_ext,
    _node_key,
    _parse_port,
    _parse_port_endpoint,
    _strip_quotes,
    all_pairs,
    boundary_edges,
    edge_id_int,
    get_directed_cycles,
    is_connected,
    remove_edge_attr,
    sort_half_edges_by_port,
)
from utils.utils import DotGraph, DotGraphs, expr_to_string, pygloopException  # noqa: F401


def Es(expr: str) -> Expression:
    return E(expr.replace('"', ""), default_namespace="gammalooprs")


def write_text_with_dirs(
    path: str, content: str, mode: str = "w", encoding: str = "utf-8"
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode, encoding=encoding) as handle:
        handle.write(content)


class VacuumDotGraph(object):
    def __init__(self, dot_graph: pydot.Dot):  # , num):
        self.dot = dot_graph
        # self.num = num

    # Given a cut (as a set of boundary edges) and a simple directed cycle, computes how
    # many times the cycle crosses the cut
    def compute_directed_winding_from_cut(
        self, directed_cycle: Set[pydot.Edge], cut: List[pydot.Edge]
    ) -> int:
        edges = list(directed_cycle)
        if not edges:
            return 0

        cut_counts = {}
        cut_is_cut = {}
        for ce in cut:
            cid = ce.get_attributes().get("id")
            cut_counts[cid] = cut_counts.get(cid, 0) + 1
            if cid in cut_is_cut:
                if cut_is_cut[cid] != _edge_is_cut_value(ce):
                    raise ValueError(f"Inconsistent is_cut for edge id {cid}")
            else:
                cut_is_cut[cid] = _edge_is_cut_value(ce)
        winding = 0
        for e in edges:
            attrs = e.get_attributes() or {}
            if "dir_in_cycle" not in attrs:
                raise ValueError("Directed cycle edge missing dir_in_cycle")
            direction = int(_strip_quotes(str(attrs.get("dir_in_cycle"))))
            eid = attrs.get("id")
            if eid in cut_counts:
                winding += cut_counts[eid] * cut_is_cut[eid] * direction
        return winding

    # Finds those cycles that have non-zero winding number
    def get_nonzero_winding_cycles(self) -> List[Set[pydot.Edge]]:
        cycles = get_directed_cycles(self.dot)
        cut = [
            e for e in self.dot.get_edges() if e.get_attributes().get("is_cut", 0) != 0
        ]
        return [
            cycle
            for cycle in cycles
            if self.compute_directed_winding_from_cut(cycle, cut) != 0
        ]

    # Finds the set of edges that cut all cycles with non-zero winding at least once, and that are minimal
    # with respect to this property
    def get_minimal_cuts(self) -> List[Set[pydot.Edge]]:
        cycles = self.get_nonzero_winding_cycles()

        if not cycles:
            return [set()]

        cycle_edges = [list(cycle) for cycle in cycles]
        solutions = []
        current = set()
        stack: list[
            tuple[str, int, Optional[Iterator[pydot.Edge]], Optional[pydot.Edge]]
        ] = [("call", 0, None, None)]
        while stack:
            action, idx, it, added = stack.pop()
            if action == "cleanup":
                current.remove(added)
                continue

            if any(sol.issubset(current) for sol in solutions):
                continue
            if idx == len(cycle_edges):
                for sol in list(solutions):
                    if current.issubset(sol):
                        solutions.remove(sol)
                solutions.append(set(current))
                continue

            cycle = cycle_edges[idx]
            if any(e in current for e in cycle):
                stack.append(("call", idx + 1, None, None))
                continue

            if it is None:
                it = iter(cycle)
            for e in it:
                stack.append(("call", idx, it, None))
                current.add(e)
                stack.append(("cleanup", 0, None, e))
                stack.append(("call", idx + 1, None, None))
                break
        return solutions

    # Given the minimal cuts, pads them so that they can have repeated entries, and
    # in particular have edges cut twice, but NOT three or more times (domain specific)
    def get_cutkosky_cuts(self) -> List[List[pydot.Edge]]:
        cycles = get_directed_cycles(self.dot)

        # The filtering logic by target could be certainly substituted by a filtering logic by winding
        # (filter by target < filter by winding). This should be thus merged with the part that finds
        # cut signs in set_cut_labels (since it uses winding to determine the signs). However, I am too lazy.
        def compute_targets(cut, cycles):
            cycle_targets = []
            for cycle in cycles:
                N = 0
                for e in cycle:
                    attrs = e.get_attributes() or {}
                    if "dir_in_cycle" not in attrs:
                        raise ValueError("Directed cycle edge missing dir_in_cycle")
                    for ep in cut:
                        if e.get_attributes()["id"] == ep.get_attributes()["id"]:
                            N += 1
                cycle_targets.append(N % 2)
            return cycle_targets

        cutes = [
            e
            for e in self.dot.get_edges()
            if abs(float(e.get_attributes().get("is_cut", 0))) == 1
        ]
        targets = compute_targets(cutes, cycles)

        old_cutkosky_cuts = self.get_minimal_cuts()
        candidates = [list(c) for c in old_cutkosky_cuts]
        out = []

        for cut in old_cutkosky_cuts:
            new_cut = list(cut)
            for e in cut:
                candidates.append(new_cut + [e])

        for cut in candidates:
            candidate_target = compute_targets(cut, cycles)
            if candidate_target == targets:
                out.append(cut)

        return out

    # Given a process definition with a number of initial and final state massive final_particles
    # returns only the pairs of cutkosky cuts consistent with this definition
    def get_cutkosky_cuts_IF(
        self, initial_massive: List[str], final_massive: List[str]
    ) -> Tuple[List[List[pydot.Edge]], List[List[pydot.Edge]]]:
        cuts = self.get_cutkosky_cuts()

        initial_cuts = []
        final_cuts = []

        for c in cuts:
            massive_in_cut = []

            for e in c:
                particle = _strip_quotes(str(e.get_attributes().get("particle", "")))
                if particle not in ["d", "d~", "g", ""]:
                    massive_in_cut.append(particle)

            if massive_in_cut == initial_massive:
                initial_cuts.append(c)
            if massive_in_cut == final_massive:
                final_cuts.append(c)

        return initial_cuts, final_cuts

    # Checks whether a signed cut reproduces the original winding number for a specific cycle
    def cycle_flow(self, oriented_cycle, signed_cut, graph) -> bool:
        original_cut = [
            e for e in graph.get_edges() if e.get_attributes().get("is_cut", "0") != "0"
        ]
        original_winding = self.compute_directed_winding_from_cut(
            oriented_cycle, original_cut
        )
        new_winding = self.compute_directed_winding_from_cut(oriented_cycle, signed_cut)
        return new_winding == original_winding

    # Checks whether two cuts split the graph in two connected components
    def cut_splits_into_two_components(
        self, initial_cut: List[pydot.Edge], final_cut: List[pydot.Edge]
    ) -> bool:
        removed = set(initial_cut) | set(final_cut)

        nodes = []

        for e in self.dot.get_edges():
            nodes.append(_node_key(e.get_source()))
            nodes.append(_node_key(e.get_destination()))

        nodes = sorted(set(nodes))
        if not nodes:
            return False

        adj = {n: set() for n in nodes}
        for e in self.dot.get_edges():
            if e in removed:
                continue
            u = _node_key(e.get_source())
            v = _node_key(e.get_destination())
            if u == v:
                continue
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        seen = set()
        components = 0
        component_nodes = []
        for n in nodes:
            if n in seen:
                continue
            components += 1
            stack = [n]
            seen.add(n)
            comp = {n}
            while stack:
                cur = stack.pop()
                for nxt in adj.get(cur, ()):
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
                        comp.add(nxt)
            component_nodes.append(comp)
            if components > 2:
                return False

        result = components == 2
        return result

    # Given two cuts, a graph and a set of oriented cycles, establishes what sign must be assigned
    # to the edges in the cut so that the winding computed with each of these cuts matches the original once
    # for any cycle
    def set_cut_labels(
        self,
        initial_cut: List[pydot.Edge],
        final_cut: List[pydot.Edge],
        graph: pydot.Dot,
        cycles: List[Set[pydot.Edge]],
    ) -> tuple[bool, pydot.Dot]:
        new_graph = copy.deepcopy(graph)
        initial_res = []
        final_res = []

        for initial_signs in product([1, -1], repeat=len(initial_cut)):
            check = True
            new_cut = copy.deepcopy(initial_cut)
            for i in range(0, len(initial_cut)):
                new_cut[i].get_attributes()["is_cut"] = initial_signs[i]
            for cycle in cycles:
                if not (self.cycle_flow(cycle, new_cut, graph)):
                    check = False

            if check:
                initial_res = new_cut

        for final_signs in product([1, -1], repeat=len(final_cut)):
            check = True
            new_cut = copy.deepcopy(final_cut)
            for i in range(0, len(final_cut)):
                new_cut[i].get_attributes()["is_cut"] = final_signs[i]
            for cycle in cycles:
                if not (self.cycle_flow(cycle, new_cut, graph)):
                    check = False

            if check:
                final_res = new_cut

        id_to_cut = {}
        for ep in initial_res:
            id_to_cut[ep.get_attributes().get("id")] = ep.get_attributes().get("is_cut")

        for ep in final_res:
            if id_to_cut.get(ep.get_attributes().get("id"), 1000) != 1000 and id_to_cut[
                ep.get_attributes().get("id")
            ] != ep.get_attributes().get("is_cut"):
                print("doooozy: non timeable pair of cuts")
                return False, pydot.Dot(graph_type="digraph")
            id_to_cut[ep.get_attributes().get("id")] = ep.get_attributes().get("is_cut")

        for e in new_graph.get_edges():
            eid = e.get_attributes().get("id")
            e.get_attributes()["is_cut"] = id_to_cut.get(eid, 0)

        return True, new_graph

    # Given a graph, two cuts (initial and final) and a partition, it constructs all routings consistent
    # with the simplest clustering criterion
    def route_cut_graph(
        self,
        graph: pydot.Dot,
        partition: List[List[str]],
    ):

        if len(partition) != 2:
            raise ValueError(
                "partition must contain exactly two subsets of initial_cut indexes"
            )

        edges = graph.get_edges()
        if not edges:
            return graph

        nodes = []
        for e in edges:
            nodes.append(_node_key(e.get_source()))
            nodes.append(_node_key(e.get_destination()))
        nodes = sorted(set(nodes))
        if not nodes:
            return graph
        root = nodes[0]

        rows = []
        rhs = []

        for v in nodes:
            if v == root:
                continue
            row = [0] * len(edges)
            for i, e in enumerate(edges):
                src = _node_key(e.get_source())
                dst = _node_key(e.get_destination())
                if src == v:
                    row[i] = 1
                elif dst == v:
                    row[i] = -1
            rows.append(row)
            rhs.append(0)

        def _add_partition_row(part):
            row = [0] * len(edges)
            counts = {}
            for pe in part:
                counts[pe] = counts.get(pe, 0) + 1
            for i, e in enumerate(edges):
                if e in counts:
                    row[i] = counts[e] * float(e.get_attributes().get("is_cut", 0))
            rows.append(row)
            rhs.append(0)

        part1, part2 = partition
        _add_partition_row(part1)
        _add_partition_row(part2)

        if sp is None:
            raise RuntimeError("sympy not available")

        if rows:
            A = sp.Matrix(rows)
            b = sp.Matrix(rhs)
        else:
            A = sp.Matrix.zeros(0, len(edges))
            b = sp.Matrix([])

        # Particular solutions for p1=1,p2=0 and p1=0,p2=1
        if A.rows:
            b1 = sp.Matrix(b)
            b2 = sp.Matrix(b)
            b1[-2] = 1
            b1[-1] = 0
            b2[-2] = 0
            b2[-1] = 1
            sol1, params1 = A.gauss_jordan_solve(b1)
            sol2, params2 = A.gauss_jordan_solve(b2)
            if params1:
                sol1 = sol1.subs({p: 0 for p in params1})
            if params2:
                sol2 = sol2.subs({p: 0 for p in params2})
            u_p1 = sol1
            u_p2 = sol2
        else:
            u_p1 = sp.Matrix([0] * len(edges))
            u_p2 = sp.Matrix([0] * len(edges))

        # Nullspace basis for free components
        if A.rows:
            nullspace = A.nullspace()
        else:
            nullspace = [sp.eye(len(edges))[:, i] for i in range(len(edges))]

        for i, e in enumerate(edges):
            e.set("routing_p1", str(sp.nsimplify(u_p1[i])))
            e.set("routing_p2", str(sp.nsimplify(u_p2[i])))
            for k, vec in enumerate(nullspace):
                e.set(f"routing_k{k}", str(sp.nsimplify(vec[i])))

        return graph

    # Checks that a certain routed graph has been routed correctly.
    def check_routing(
        self, graph: pydot.Dot, partition: List[List[pydot.Edge]]
    ) -> bool:
        edges = graph.get_edges()
        nodes = []
        for e in edges:
            nodes.append(_node_key(e.get_source()))
            nodes.append(_node_key(e.get_destination()))
        nodes = sorted(set(nodes))

        if sp is None:
            raise RuntimeError("sympy not available")

        for v in nodes:
            bdry = boundary_edges(graph, {v})
            sum_mom_k = 0
            sum_mom_p1 = 0
            sum_mom_p2 = 0
            for e in bdry:
                for ep in edges:
                    if ep.get_attributes()["id"] == e.get_attributes()["id"]:
                        sigma = 1 if _node_key(ep.get_source()) == v else -1
                        sum_mom_k += (
                            sp.Rational(ep.get_attributes()["routing_k0"]) * sigma
                        )
                        sum_mom_p1 += (
                            sp.Rational(ep.get_attributes()["routing_p1"]) * sigma
                        )
                        sum_mom_p2 += (
                            sp.Rational(ep.get_attributes()["routing_p2"]) * sigma
                        )
            if sum_mom_k != 0 or sum_mom_p1 != 0 or sum_mom_p2 != 0:
                print(
                    f"Error at node {v}: sum_mom_k={sum_mom_k}, sum_mom_p1={sum_mom_p1}, sum_mom_p2={sum_mom_p2}"
                )
                return False

        for i in range(0, 1):
            sum_k = 0
            sum_p1 = 0
            sum_p2 = 0
            for e in partition[i]:
                for ep in edges:
                    if ep.get_attributes()["id"] == e.get_attributes()["id"]:
                        sum_k += sp.Rational(
                            ep.get_attributes()["routing_k0"]
                        ) * sp.Rational(ep.get_attributes()["is_cut"])
                        sum_p1 += sp.Rational(
                            ep.get_attributes()["routing_p1"]
                        ) * sp.Rational(ep.get_attributes()["is_cut"])
                        sum_p2 += sp.Rational(
                            ep.get_attributes()["routing_p2"]
                        ) * sp.Rational(ep.get_attributes()["is_cut"])
            if (
                sum_k != 0
                or sum_p1 != (1 if i == 0 else 0)
                or sum_p2 != (1 if i == 1 else 0)
            ):
                print(
                    f"Error at partition {i}: sum_k={sum_k}, sum_p1={sum_p1}, sum_p2={sum_p2}"
                )
                return False

        return True

    # Eliminates all zero-measured cuts (domain specific: no massive particles in the initial state)
    def phase_space_check(
        self, initial_cut: List[pydot.Edge], final_cut: List[pydot.Edge]
    ) -> bool:
        if len(set(initial_cut) - set(final_cut)) == 1:
            return False
        return True

    # Computes all possible cut routed diagrams given a process definition
    def cut_graphs_with_routing(
        self,
        initial_massive: List[str],
        final_massive: List[str],
    ) -> List[
        Tuple[List[pydot.Edge], List[pydot.Edge], List[List[pydot.Edge]], pydot.Dot]
    ]:
        initial_cuts, final_cuts = self.get_cutkosky_cuts_IF(
            initial_massive, final_massive
        )
        routed_cut_graphs = []

        cycles = get_directed_cycles(self.dot)
        for initial_cut in initial_cuts:
            for final_cut in final_cuts:
                connected_components = self.cut_splits_into_two_components(
                    initial_cut, final_cut
                )
                labelled_graph = self.set_cut_labels(
                    initial_cut, final_cut, copy.deepcopy(self.dot), cycles
                )
                if (
                    self.phase_space_check(initial_cut, final_cut)
                    and connected_components
                    and labelled_graph[0]
                ):
                    graph = labelled_graph[1]

                    all_pair_list = all_pairs(initial_cut)
                    for V1, V2 in all_pair_list:
                        idV1 = [e.get_attributes()["id"] for e in V1]
                        idV2 = [e.get_attributes()["id"] for e in V2]

                        new_graph = copy.deepcopy(graph)
                        new_graph.set_name(
                            f"{self.dot.get_name()}_partition_{idV1}_{idV2}"
                        )
                        new_graph.set("partition", f"{[idV1, idV2]}")
                        # new_graph.set("num", str(self.num))
                        graph = self.route_cut_graph(new_graph, [V1, V2])

                        routed_cut_graphs.append([
                            initial_cut,
                            final_cut,
                            [V1, V2],
                            graph,
                        ])
                        if not self.check_routing(graph, [V1, V2]):
                            print("ERROR: Routing is wrongly assigned")
                            raise Exception

        return routed_cut_graphs

    # Returns only the routed interference diagrams that contribute with non-factorisable pieces at leading virtuality
    def cut_graphs_with_routing_leading_virtuality(
        self, initial_massive: List[str], final_massive: List[str]
    ) -> List[
        Tuple[List[pydot.Edge], List[pydot.Edge], List[List[pydot.Edge]], pydot.Dot]
    ]:
        cut_graphs_with_routing = self.cut_graphs_with_routing(
            initial_massive, final_massive
        )
        cut_graphs_with_routing_LV = []

        for graph in cut_graphs_with_routing:
            count_p1 = False
            count_p2 = False

            if sp is None:
                raise RuntimeError("sympy not available")

            for e in graph[3].get_edges():
                if (
                    sp.Rational(e.get_attributes()["routing_p1"]) != 0
                    and sp.Rational(e.get_attributes()["routing_k0"]) == 0
                    and sp.Rational(e.get_attributes()["routing_p2"]) == 0
                    and _strip_quotes(e.get_attributes().get("particle", "")) != "a"
                ):
                    count_p1 = True
                elif (
                    sp.Rational(e.get_attributes()["routing_p2"]) != 0
                    and sp.Rational(e.get_attributes()["routing_k0"]) == 0
                    and sp.Rational(e.get_attributes()["routing_p1"]) == 0
                    and _strip_quotes(e.get_attributes().get("particle", "")) != "a"
                ):
                    count_p2 = True
            if count_p1 and count_p2:
                cut_graphs_with_routing_LV.append(graph)

        return cut_graphs_with_routing_LV

    def to_string(self) -> str:
        return self.dot.to_string()


class DYDotGraph(DotGraph):
    def __init__(self, dot_graph: pydot.Dot):
        self.dot = dot_graph

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

    # True iff edge is of form ext_i -> a:b.
    def is_incoming_half_edge(self, e) -> bool:
        src = e.get_source()
        dst = e.get_destination()
        return _is_ext(src) and (_parse_port_endpoint(dst) is not None)

    # True iff edge is of form a:b -> ext_i.
    def is_outgoing_half_edge(self, e) -> bool:

        src = e.get_source()
        dst = e.get_destination()
        return _is_ext(dst) and (_parse_port_endpoint(src) is not None)

    # Copies an edge
    def copy_edge(self, e: pydot.Edge) -> pydot.Edge:
        attrs = dict(e.get_attributes() or {})
        return pydot.Edge(e.get_source(), e.get_destination(), **attrs)

    # Return the endpoint (source or destination) that is NOT ext*.
    def _non_ext_endpoint(self, e: pydot.Edge) -> str:
        src = e.get_source()
        dst = e.get_destination()

        src_is_ext = _is_ext(src)
        dst_is_ext = _is_ext(dst)

        if src_is_ext and not dst_is_ext:
            return dst
        if dst_is_ext and not src_is_ext:
            return src

        raise ValueError(f"Expected exactly one ext* endpoint, got: {src} -> {dst}")

    # Return a new edge connecting the non-ext endpoint of e1 to the non-ext
    # endpoint of e2, copying ALL attributes from e1.
    def edge_fusion(self, e1: pydot.Edge, e2: pydot.Edge) -> pydot.Edge:
        u = self._non_ext_endpoint(e1)
        v = self._non_ext_endpoint(e2)

        attrs = dict(e1.get_attributes() or {})
        attrs["is_cut"] = "1"

        attrs_e1 = e1.get_attributes()

        print(e1)

        ee = remove_edge_attr(pydot.Edge(u, v, **attrs), "lmb_rep")

        if _edge_particle(ee) == "d":
            ee.set(
                "num",
                f"Q({edge_id_int(ee)},mink(4,mu))*spenso::gamma(spenso::bis(4,hedge({_parse_port(ee.get_source())})),spenso::bis(4,hedge({_parse_port(ee.get_destination())})),spenso::mink(4,mu))*spenso::g(spenso::dind(spenso::cof(3,hedge({_parse_port(ee.get_source())}))),spenso::cof(3,hedge({_parse_port(ee.get_destination())})))",
            )
        if _edge_particle(ee) == "d~":
            ee.set(
                "num",
                f"Q({edge_id_int(ee)},mink(4,mu))*spenso::gamma(spenso::bis(4,hedge({_parse_port(ee.get_destination())})),spenso::bis(4,hedge({_parse_port(ee.get_source())})),spenso::mink(4,mu))*spenso::g(spenso::dind(spenso::cof(3,hedge({_parse_port(ee.get_destination())}))),spenso::cof(3,hedge({_parse_port(ee.get_source())})))",
            )
        if _edge_particle(ee) == "g":
            ee.set(
                "num",
                f"-spenso::g(spenso::mink(4,hedge({_parse_port(ee.get_source())})),spenso::mink(4,hedge({_parse_port(ee.get_destination())})))*spenso::g(spenso::coad(8,hedge({_parse_port(ee.get_destination())})),spenso::coad(8,hedge({_parse_port(ee.get_source())})))",
            )
        return ee

    # Glues the end of the FS diagram (really an amplitude diagram) into a vacuum diagram
    def get_vacuum_graph(self):
        incoming_edges = []
        outgoing_edges = []
        paired_up = []

        vacuum_graph = pydot.Dot(graph_type="digraph")
        name = self.dot.get_name() or ""  # or some default
        vacuum_graph.set_name(name)

        # Preserve all non-ext nodes from the original graph.
        for node in self.dot.get_nodes():
            node_name = _strip_quotes(node.get_name())
            if _is_ext(node_name):
                continue
            vacuum_graph.add_node(node)

        for e in self.dot.get_edges():
            if self.is_incoming_half_edge(e):
                incoming_edges.append(e)
            elif self.is_outgoing_half_edge(e):
                outgoing_edges.append(e)
            else:
                ep = self.copy_edge(e)
                ep.set("is_cut", "0")
                remove_edge_attr(ep, "lmb_rep")
                # remove_edge_attr(ep, "num")
                vacuum_graph.add_edge(ep)

        if len(incoming_edges) != len(outgoing_edges):
            raise pygloopException("Vacuum graph is not balanced.")

        incoming_edges = sort_half_edges_by_port(incoming_edges, self._non_ext_endpoint)
        outgoing_edges = sort_half_edges_by_port(outgoing_edges, self._non_ext_endpoint)

        for e in incoming_edges:
            for ep in outgoing_edges:
                if _edge_particle(e) == _edge_particle(ep) and ep not in paired_up:
                    vacuum_graph.add_edge(
                        remove_edge_attr(self.edge_fusion(e, ep), "lmb_rep")
                    )
                    paired_up.append(ep)

        # NEW STUFF: RELABEL SO THAT IDs ARE CONSECUTIVE
        substitutions=[]
        for e, i in zip(
            vacuum_graph.get_edges(), range(0, len(vacuum_graph.get_edges()))
        ):
            print(e.get('id'))
            print(i)
            print("----")
            pattern=E(f"Q({e.get('id')},y___)", default_namespace="gammalooprs")
            substitution=E(f"Q({i},y___)", default_namespace="gammalooprs")
            substitutions.append((i,pattern,substitution))
            e.get_attributes()["id"] = i

        for e in vacuum_graph.get_edges():
            for (id, pat, rep) in substitutions:
                if e.get("id")==id:
                    e.get_attributes()["num"]=expr_to_string(Es(e.get("num")).replace(pat, rep))

        for v in vacuum_graph.get_nodes():
            for (id, pat, rep) in substitutions:
                v.get_attributes()["num"]=expr_to_string(Es(v.get("num")).replace(pat, rep))

        return VacuumDotGraph(
            vacuum_graph  # , self.get_numerator(include_overall_factor=True)
        )

    def contains_ext(self, node_subset, left_externals, right_externals):
        component_edges = []

        for e in self.dot.get_edges():
            if (
                _base_node(e.get_source()) in node_subset
                and _base_node(e.get_destination()) in node_subset
            ):
                component_edges.append(e)

        stack = left_externals.copy()

        for e in component_edges:
            if e in left_externals:
                stack.remove(e)
            if e in right_externals:
                return False

        if len(stack) == 0:
            return True
        return False

    def get_incoming_edges(self):
        incoming_edges = []
        for e in self.dot.get_edges():
            if self.is_incoming_half_edge(e):
                incoming_edges.append(e)
        return incoming_edges

    def get_outgoing_edges(self):
        outgoing_edges = []
        for e in self.dot.get_edges():
            if self.is_outgoing_half_edge(e):
                outgoing_edges.append(e)
        return outgoing_edges

    def enumerate_cutkosky_cuts(self, left_externals, right_externals):
        nodes = set(_all_node_names(self.dot))
        all_cuts = [
            set(s)
            for r in range(
                min(len(left_externals), len(right_externals)),
                len(nodes) + 1 - min(len(left_externals), len(right_externals)),
            )
            for s in combinations(nodes, r)
        ]

        good_sets = [
            S
            for S in all_cuts
            if is_connected(self.dot, S)
            and is_connected(self.dot, nodes - S)
            and self.contains_ext(S, left_externals, right_externals)
        ]

        return good_sets

    def to_string(self) -> str:
        return self.dot.to_string()


class DYDotGraphs(DotGraphs):
    def __init__(self, dot_str: str | None = None, dot_path: str | None = None):
        if dot_str is None and dot_path is None:
            return
        if dot_path is not None and dot_str is not None:
            raise pygloopException(
                "Only one of dot_str or dot_path should be provided."
            )

        if dot_path:
            dot_graphs = pydot.graph_from_dot_file(dot_path)
            if dot_graphs is None:
                raise ValueError(f"No graphs found in DOT file: {dot_path}")
            self.extend([DYDotGraph(g) for g in dot_graphs])
        elif dot_str:
            dot_graphs = pydot.graph_from_dot_data(dot_str)
            if dot_graphs is None:
                raise ValueError("No graphs found in DOT data string.")
            self.extend([DYDotGraph(g) for g in dot_graphs])

    def get_graph(self, graph_name) -> DYDotGraph:
        for g in self:
            if g.dot.get_name() == graph_name:
                return g
        raise KeyError(f"Graph with name {graph_name} not found.")

    @staticmethod
    def _strip_quotes(s: str) -> str:
        s = s.strip()
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            return s[1:-1]
        return s

    def filter_particle_definition(self, final_particles):
        new_graphs = []

        for graph in self:
            incoming_edges = graph.get_incoming_edges()
            outgoing_edges = graph.get_outgoing_edges()
            cutkosky_cuts = graph.enumerate_cutkosky_cuts(
                incoming_edges, outgoing_edges
            )

            for S in cutkosky_cuts:
                c = boundary_edges(graph.dot, S)
                massive_in_cut = []
                for e in c:
                    particle = _strip_quotes(
                        str(e.get_attributes().get("particle", ""))
                    )
                    if particle not in ["d", "d~", "g"]:
                        massive_in_cut.append(particle)

                if massive_in_cut == final_particles:
                    new_graphs.append(graph)

        return new_graphs

    def save_to_file(self, file_path: str):
        write_text_with_dirs(file_path, "\n\n".join([g.to_string() for g in self]))
