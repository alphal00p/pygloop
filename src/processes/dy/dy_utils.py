import re
from collections import deque
from itertools import combinations
from pprint import pprint
from typing import List, Optional, Tuple

import pydot
from pydot import Edge, Node  # noqa: F401

from utils.utils import DotGraph, DotGraphs, logger, pygloopException  # noqa: F401
from utils.vectors import LorentzVector, Vector  # noqa: F401
from symbolica import E, Expression, NumericalIntegrator, Sample


def Es(expr: str) -> Expression:
    return E(expr.replace('"', ""), default_namespace="gammalooprs")

class VacuumDotGraph(object):
    def __init__(self, dot_graph: pydot.Dot, num):
        self.dot = dot_graph
        self.num = num

    @staticmethod
    def _strip_quotes(s: str) -> str:
        s = s.strip()
        return s[1:-1] if len(s) >= 2 and s[0] == '"' and s[-1] == '"' else s

    def _node_key(self, endpoint: str, collapse_ports: bool = True) -> str:
        ep = self._strip_quotes(endpoint)
        if collapse_ports and ":" in ep:
            return ep.split(":", 1)[0]
        return ep

    def _base_node(self, endpoint: str) -> str:
        ep = DYDotGraph._strip_quotes(endpoint)
        parsed = self._parse_port_endpoint(ep)
        return parsed[0] if parsed else ep

    def _edge_is_cut_value(self, e: pydot.Edge) -> int:
        attrs = e.get_attributes() or {}
        val = attrs.get("is_cut", "0")
        return int(self._strip_quotes(str(val)))

    def _parse_port_endpoint(self, endpoint: str) -> Optional[Tuple[str, int]]:
        """
        Parse 'a:b' (possibly quoted) -> ('a', b). Returns None if not matching.
        """
        ep = DYDotGraph._strip_quotes(endpoint)
        m = re.fullmatch(r"([^:]+):(\d+)", ep)
        if not m:
            return None
        return m.group(1), int(m.group(2))

    def boundary_edges(self, S):

        out = []
        for e in self.dot.get_edges():
            u = self._base_node(e.get_source())
            v = self._base_node(e.get_destination())
            if (u in S and v not in S) or (u not in S and v in S):
                out.append(e)
        return out

    def get_spanning_tree(self):
        edges = self.dot.get_edges()
        if not edges:
            return []

        root = self._node_key(edges[0].get_source())
        S = {root}
        T = []

        changed = True
        while changed:
            changed = False
            for e in self.boundary_edges(S):
                u = self._node_key(e.get_source())
                v = self._node_key(e.get_destination())

                if u in S and v not in S:
                    S.add(v); T.append(e); changed = True
                    break
                if v in S and u not in S:
                    S.add(u); T.append(e); changed = True
                    break

        return T

    def get_cycle_basis(self):
        T=set(self.get_spanning_tree())
        L=set(self.dot.get_edges())-set(T)

        cycles=[]

        for e in L:
            cycle=T.union({e})
            check=True
            while check:
                counter=0
                for e in cycle:
                    if len(set(self.boundary_edges({self._node_key(e.get_destination())})).intersection(cycle))==1 or len(set(self.boundary_edges({self._node_key(e.get_source())})).intersection(cycle))==1:
                        cycle=cycle - {e}
                        break
                    else:
                        counter+=1

                if counter==len(cycle):
                    check=False
            cycles.append(cycle)

        return cycles

    #def compute_winding(self,cycle):

    #    v_start=cycle[0].get_source()

    #    while v_end!=v_start:












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


    @staticmethod
    def _strip_quotes(s: str) -> str:
        s = s.strip()
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            return s[1:-1]
        return s

    def _is_ext(self, name: str) -> bool:
        return bool(re.fullmatch(r"ext\d+", DYDotGraph._strip_quotes(name)))

    def _parse_port_endpoint(self, endpoint: str) -> Optional[Tuple[str, int]]:
        """
        Parse 'a:b' (possibly quoted) -> ('a', b). Returns None if not matching.
        """
        ep = DYDotGraph._strip_quotes(endpoint)
        m = re.fullmatch(r"([^:]+):(\d+)", ep)
        if not m:
            return None
        return m.group(1), int(m.group(2))

    def _edge_particle(self, e: pydot.Edge) -> str:
        attrs = e.get_attributes() or {}
        p = attrs.get("particle", "")
        return DYDotGraph._strip_quotes(str(p))

    def is_incoming_half_edge(self, e) -> bool:
        """True iff edge is of form ext_i -> a:b."""
        src = e.get_source()
        dst = e.get_destination()
        return self._is_ext(src) and (self._parse_port_endpoint(dst) is not None)

    def is_outgoing_half_edge(self, e) -> bool:
        """True iff edge is of form a:b -> ext_i."""
        src = e.get_source()
        dst = e.get_destination()
        return self._is_ext(dst) and (self._parse_port_endpoint(src) is not None)

    def get_part(self, e):
        attrs = e.get_attributes()
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

        raise ValueError(f"Expected exactly one ext* endpoint, got: {src} -> {dst}")

    def remove_edge_attr(self, e: pydot.Edge, key: str) -> None:
        if e is None:
            raise ValueError("remove_edge_attr got None edge")

        attrs = e.get_attributes() or {}
        attrs.pop(key, None)  # in-place usually works in pydot
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

        e_new = self.remove_edge_attr(pydot.Edge(u, v, **attrs), "lmb_rep")

        # Create the fused edge
        return e_new

    def _parse_port(self, endpoint: str) -> Optional[int]:
        """Return b from 'a:b' (possibly quoted), else None."""
        ep = DYDotGraph._strip_quotes(endpoint)
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
        incoming_edges = []
        outgoing_edges = []
        paired_up = []
        new_edges = []

        vacuum_graph = pydot.Dot(graph_type="digraph")

        for e in self.dot.get_edges():
            if self.is_incoming_half_edge(e):
                incoming_edges.append(e)
            elif self.is_outgoing_half_edge(e):
                outgoing_edges.append(e)
            else:
                ep = self.copy_edge(e)
                ep.set("is_cut", "0")
                vacuum_graph.add_edge(self.remove_edge_attr(ep, "lmb_rep"))

        if len(incoming_edges) != len(outgoing_edges):
            raise pygloopException("Vacuum graph is not balanced.")

        incoming_edges = self.sort_half_edges_by_port(incoming_edges)
        outgoing_edges = self.sort_half_edges_by_port(outgoing_edges)

        for e in incoming_edges:
            for ep in outgoing_edges:
                if self.get_part(e) == self.get_part(ep) and ep not in paired_up:
                    vacuum_graph.add_edge(self.remove_edge_attr(self.edge_fusion(e, ep), "lmb_rep"))
                    paired_up.append(ep)

        return VacuumDotGraph(vacuum_graph,self.get_numerator(include_overall_factor=True))

    def _base_node(self, endpoint: str) -> str:
        ep = DYDotGraph._strip_quotes(endpoint)
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
        component_edges = []

        for e in self.dot.get_edges():
            if self._base_node(e.get_source()) in node_subset and self._base_node(e.get_destination()) in node_subset:
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

    def boundary_edges(self, S):
        out = []
        for e in self.dot.get_edges():
            u = self._base_node(e.get_source())
            v = self._base_node(e.get_destination())
            if (u in S and v not in S) or (u not in S and v in S):
                out.append(e)
        return out

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
        nodes = set(self._all_node_names())
        all_cuts = [
            set(s)
            for r in range(min(len(left_externals), len(right_externals)), len(nodes) + 1 - min(len(left_externals), len(right_externals)))
            for s in combinations(nodes, r)
        ]

        good_sets = [
            S for S in all_cuts if self.is_connected(S) and self.is_connected(nodes - S) and self.contains_ext(S, left_externals, right_externals)
        ]

        return good_sets


class DYDotGraphs(DotGraphs):
    def __init__(self, dot_str: str | None = None, dot_path: str | None = None):
        if dot_str is None and dot_path is None:
            return
        if dot_path is not None and dot_str is not None:
            raise pygloopException("Only one of dot_str or dot_path should be provided.")

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

    def filter_particle_definition(self, particles):
        newgraphs = []

        for graph in self:
            incoming_edges = graph.get_incoming_edges()
            outgoing_edges = graph.get_outgoing_edges()
            cutkosky_cuts = graph.enumerate_cutkosky_cuts(incoming_edges, outgoing_edges)

            for c in cutkosky_cuts:
                stack = particles.copy()
                for e in c:
                    if e.get_attributes().get("particle", "") in stack:
                        stack.remove(e.get_attributes().get("particle", ""))
                if len(stack) == 0:
                    newgraphs.append(graph)

        self = newgraphs
