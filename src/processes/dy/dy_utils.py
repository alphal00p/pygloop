import re
from collections import deque
from itertools import combinations, product, permutations
from pprint import pprint
from typing import List, Optional, Tuple
import os
import itertools
try:
    import sympy as sp
except ImportError:
    print(
        "Failed to import sympy for DY process (you won't be able to run it then). But honestly that's ok... almighty Symbolica should be used instead."
    )
import copy
import pydot
from pydot import Edge, Node  # noqa: F401
from utils.utils import DotGraph, DotGraphs, logger, pygloopException  # noqa: F401
from utils.vectors import LorentzVector, Vector  # noqa: F401
from symbolica import E, Expression, NumericalIntegrator, Sample


def Es(expr: str) -> Expression:
    return E(expr.replace('"', ""), default_namespace="gammalooprs")

def write_text_with_dirs(path: str, content: str, mode: str = "w", encoding: str = "utf-8") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode, encoding=encoding) as handle:
        handle.write(content)

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
                    S.add(v)
                    T.append(e)
                    changed = True
                    break
                if v in S and u not in S:
                    S.add(u)
                    T.append(e)
                    changed = True
                    break

        return T

    def get_cycle_basis(self):
        T = set(self.get_spanning_tree())
        L = set(self.dot.get_edges()) - set(T)

        cycles = []

        for e in L:
            cycle = T.union({e})
            check = True
            while check:
                counter = 0
                for e in cycle:
                    if (
                        len(set(self.boundary_edges({self._node_key(e.get_destination())})).intersection(cycle)) == 1
                        or len(set(self.boundary_edges({self._node_key(e.get_source())})).intersection(cycle)) == 1
                    ):
                        cycle = cycle - {e}
                        break
                    else:
                        counter += 1

                if counter == len(cycle):
                    check = False
            cycles.append(cycle)

        return cycles

    def get_simple_cycles(self):
        edges = list(copy.deepcopy(self.dot.get_edges()))
        if not edges:
            return []

        adj = {}
        for e in edges:
            u = self._node_key(e.get_source())
            v = self._node_key(e.get_destination())
            adj.setdefault(u, []).append(e)
            adj.setdefault(v, []).append(e)

        nodes = sorted(adj.keys())
        seen = set()
        cycles = []

        def other_node(edge: pydot.Edge, node: str) -> Optional[str]:
            u = self._node_key(edge.get_source())
            v = self._node_key(edge.get_destination())
            if u == node:
                return v
            if v == node:
                return u
            return None

        def dfs(start, current, parent_edge, path_nodes, path_edges):
            for edge in adj[current]:
                if edge is parent_edge:
                    continue
                nxt = other_node(edge, current)
                if nxt is None:
                    continue
                if nxt == start:
                    if len(path_nodes) >= 2:
                        key = frozenset(id(e) for e in path_edges + [edge])
                        if key not in seen:
                            seen.add(key)
                            cycles.append(set(path_edges + [edge]))
                    continue
                if nxt in path_nodes:
                    continue
                if nxt < start:
                    continue
                dfs(start, nxt, edge, path_nodes + [nxt], path_edges + [edge])

        for start in nodes:
            dfs(start, start, None, [start], [])

        return cycles

    def _get_directed_cycle(self, cycle):
        edges = list(cycle)
        if not edges:
            return copy.deepcopy(cycle)

        adj = {}
        for e in edges:
            u = self._node_key(e.get_source())
            v = self._node_key(e.get_destination())
            adj.setdefault(u, []).append(e)
            adj.setdefault(v, []).append(e)

        def other_node(edge: pydot.Edge, node: str) -> Optional[str]:
            u = self._node_key(edge.get_source())
            v = self._node_key(edge.get_destination())
            if u == node:
                return v
            if v == node:
                return u
            return None

        start = min(adj.keys())
        first_edge = min(adj[start], key=lambda e: other_node(e, start))

        visited = set()
        current = start
        prev_edge = None
        ordered = []

        while True:
            if prev_edge is None:
                edge = first_edge
            else:
                candidates = [e for e in adj[current] if e is not prev_edge]
                if not candidates:
                    raise ValueError("Cycle is not well-formed")
                edge = min(candidates, key=lambda e: other_node(e, current))

            if edge in visited:
                break
            visited.add(edge)

            next_node = other_node(edge, current)
            if next_node is None:
                raise ValueError("Cycle is not well-formed")

            src = self._node_key(edge.get_source())
            dst = self._node_key(edge.get_destination())
            direction = 1 if (src == current and dst == next_node) else -1
            edge.set("dir_in_cycle", direction)
            ordered.append((edge, direction))

            prev_edge = edge
            current = next_node
            if current == start:
                break
            if len(visited) > len(edges):
                raise ValueError("Cycle is not well-formed")

        if len(visited) != len(edges):
            raise ValueError("Cycle is not well-formed")

        # Orient the cycle so a reference is_cut==1 edge has dir_in_cycle == 1.
        flip = False
        for (e, d) in ordered:
            if self._edge_is_cut_value(e) == 1 and d == 1:
                flip = True
        if flip:
            for (e, d) in ordered:
                e.set("dir_in_cycle", -d)

        return copy.deepcopy(cycle)

    def get_directed_cycles(self, cycles=None):
        if cycles is None:
            cycles = self.get_simple_cycles()
        directed_cycles = []
        for cycle in cycles:
            directed_cycles.append(self._get_directed_cycle(cycle))
        return directed_cycles

    def compute_directed_winding_from_cut(self, directed_cycle, cut):
        edges = list(directed_cycle)
        if not edges:
            return 0

        cut_by_id = {ce.get_attributes().get("id"): ce for ce in cut}
        winding = 0
        for e in edges:
            attrs = e.get_attributes() or {}
            if "dir_in_cycle" not in attrs:
                raise ValueError("Directed cycle edge missing dir_in_cycle")
            direction = int(self._strip_quotes(str(attrs.get("dir_in_cycle"))))
            eid = attrs.get("id")
            cut_edge = cut_by_id.get(eid)
            winding += self._edge_is_cut_value(cut_edge) * direction if cut_edge else 0
        return winding


    def get_nonzero_winding_cycles(self):
        cycles = self.get_directed_cycles()
        cut=[e for e in self.dot.get_edges() if e.get_attributes().get("is_cut",0)!=0]
        return [cycle for cycle in cycles if self.compute_directed_winding_from_cut(cycle,cut) != 0]

    def get_cutkosky_cut_1(self):
        cycles = self.get_nonzero_winding_cycles()

        # A CUTKOSKY CUT MUST BE A LIST OF EDGES, POTENTIALLY REPEATED (HAVE TO SET MAX NUMBER OF REPETITIONS)
        # IT HAS THE PROPERTY THAT IT CUTS THROUGH THE CYCLES IN EXACTLY THE SAME WAY AS THE ORIGINAL CUT

        if not cycles:
            return [set()]

        cycle_edges = [list(cycle) for cycle in cycles]
        solutions = []

        def is_superset_of_solution(candidate):
            return any(sol.issubset(candidate) for sol in solutions)

        def add_solution(candidate):
            for sol in list(solutions):
                if candidate.issubset(sol):
                    solutions.remove(sol)
            solutions.append(candidate)

        def backtrack(idx, current):
            if is_superset_of_solution(current):
                return
            if idx == len(cycle_edges):
                add_solution(set(current))
                return

            cycle = cycle_edges[idx]
            if any(e in current for e in cycle):
                backtrack(idx + 1, current)
                return

            for e in cycle:
                current.add(e)
                backtrack(idx + 1, current)
                current.remove(e)

        backtrack(0, set())
        return solutions

    def cycle_flow(self, oriented_cycle, signed_cut, graph):
        original_cut=[e for e in graph.get_edges() if e.get_attributes().get("is_cut","0")!="0"]
        original_winding=self.compute_directed_winding_from_cut(oriented_cycle,original_cut)
        new_winding=self.compute_directed_winding_from_cut(oriented_cycle,signed_cut)
#        print("original cut")
#        for e in original_cut:
#            print(e)
#        print("signed cut")
#        for e in signed_cut:
#                print(e)
#        print("original_winding: ", original_winding)
#        print("new_winding: ", new_winding)
        return new_winding==original_winding



    def get_cutkosky_cuts(self):
        cycles = self.get_directed_cycles()

        def compute_targets(cut,cycles):
            cycle_targets = []
            for cycle in cycles:
                N = 0
                for e in cycle: ###CHANGE TO e in cut
                    attrs = e.get_attributes() or {}
                    if "dir_in_cycle" not in attrs:
                        raise ValueError("Directed cycle edge missing dir_in_cycle")
                    direction = int(self._strip_quotes(str(attrs.get("dir_in_cycle"))))
                    for ep in cut:
                        if e.get_attributes()["id"]==ep.get_attributes()["id"]:
                            N += 1 #direction
                cycle_targets.append(N % 2)
            return cycle_targets


        cutes=[e for e in self.dot.get_edges() if abs(float(e.get_attributes().get("is_cut", 0)))==1]
        targets=compute_targets(cutes,cycles)

        old_cutkosky_cuts = self.get_cutkosky_cut_1()
        candidates = [list(c) for c in old_cutkosky_cuts]
        out = []

        for cut in old_cutkosky_cuts:
            new_cut=list(cut)
            for e in cut:
                candidates.append(new_cut+[e])


        for cut in candidates:
            candidate_target=compute_targets(cut,cycles)
            if candidate_target==targets:
                out.append(cut)


        return out






    def get_cutkosky_cuts_IF(self, initial_massive, final_massive):
        cuts = self.get_cutkosky_cuts()

        initial_cuts = []
        final_cuts = []

        for c in cuts:
            massive_in_cut = []

            for e in c:
                particle = self._strip_quotes(str(e.get_attributes().get("particle", "")))
                if particle not in ["d", "d~", "g"]:
                    massive_in_cut.append(particle)

            if massive_in_cut == initial_massive:
                initial_cuts.append(c)
            if massive_in_cut == final_massive:
                final_cuts.append(c)

        return initial_cuts, final_cuts

    def cut_splits_into_two_components(self, initial_cut, final_cut, return_components: bool = True):
        removed = set(initial_cut) | set(final_cut)

        nodes = []

        for e in self.dot.get_edges():
            nodes.append(self._node_key(e.get_source()))
            nodes.append(self._node_key(e.get_destination()))

        nodes = sorted(set(nodes))
        if not nodes:
            return False

        adj = {n: set() for n in nodes}
        for e in self.dot.get_edges():
            if e in removed:
                continue
            u = self._node_key(e.get_source())
            v = self._node_key(e.get_destination())
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
            if not return_components and components > 2:
                return False

        result = components == 2
        if return_components:
            return result, component_nodes
        return result

    def set_cut_labels_2(self, initial_cut, final_cut, graph, cycles):
        new_graph = copy.deepcopy(graph)
        initial_cut_ids=[e.get_attributes()["id"] for e in initial_cut]
        final_cut_ids=[e.get_attributes()["id"] for e in final_cut]
        initial_res=[]
        final_res=[]

        #print("cut1")
        #for e in initial_cut:
        #    print(e)
        #print("cut2")
        #for e in final_cut:
            #    print(e)

        for initial_signs in product([1, -1], repeat=len(initial_cut)):
            check=True
            new_cut=copy.deepcopy(initial_cut)
            for i in range(0,len(initial_cut)):
                new_cut[i].get_attributes()["is_cut"]=initial_signs[i]
            for cycle in cycles:
                if not (self.cycle_flow(cycle, new_cut, graph)):
                    check=False

            if check:
                initial_res=new_cut



        for final_signs in product([1, -1], repeat=len(final_cut)):
            check=True
            new_cut=copy.deepcopy(final_cut)
            for i in range(0,len(final_cut)):
                new_cut[i].get_attributes()["is_cut"]=final_signs[i]
            for cycle in cycles:
                if not (self.cycle_flow(cycle, new_cut, graph)):
                    check=False

            if check:
                final_res=new_cut

        #print("final cut1")
        #for e in initial_res:
        #    print(e)
        #print("final cut2")
        #for e in final_res:
            #    print(e)

        id_to_cut = {}
        for ep in initial_res:
            id_to_cut[ep.get_attributes().get("id")] = ep.get_attributes().get("is_cut")
        for ep in final_res:
            id_to_cut[ep.get_attributes().get("id")] = ep.get_attributes().get("is_cut")

        for e in new_graph.get_edges():
            eid = e.get_attributes().get("id")
            e.get_attributes()["is_cut"] = id_to_cut.get(eid, 0)



        return new_graph





    def set_cut_labels(self, initial_cut, final_cut, connected_components):
        graph = copy.deepcopy(self.dot)

        initial_cut_ids = [e.get_attributes()["id"] for e in initial_cut]
        final_cut_ids = [e.get_attributes()["id"] for e in final_cut]

        new_cycles = self.get_directed_cycles()

        for e in graph.get_edges():
            if e.get_attributes()["id"] not in initial_cut_ids and e.get_attributes()["id"] not in final_cut_ids:
                e.get_attributes()["is_cut"] = "0"
                continue

            target_ids = initial_cut_ids if e.get_attributes()["id"] in initial_cut_ids else final_cut_ids
            e_id = e.get_attributes()["id"]

            found = False
            for cycle in new_cycles:

                count = sum(1 for ep in set(cycle) if ep.get_attributes()["id"] in target_ids)

                if count == 1:
                    for ep in cycle:
                        if ep.get_attributes()["id"] == e_id:

                            e.get_attributes()["is_cut"] = str(int(ep.get_attributes()["dir_in_cycle"]))

                            found = True
                            break
                if found:
                    break

            if not found:
                raise RuntimeError("Expected a cycle crossing cut once that contains edge")

        return graph




    def route_cut_graph(self, graph, initial_cut, final_cut, partition=None, p1=1, p2=1, root=None):
        ### Assume partition is a list containing two sets of indexes mapping to elements of initial_cut
        if partition is None:
            return graph
        if len(partition) != 2:
            raise ValueError("partition must contain exactly two subsets of initial_cut indexes")

        edges = graph.get_edges()
        if not edges:
            return graph

        nodes = []
        for e in edges:
            nodes.append(self._node_key(e.get_source()))
            nodes.append(self._node_key(e.get_destination()))
        nodes = sorted(set(nodes))
        if not nodes:
            return graph
        if root is None:
            root = nodes[0]

        rows = []
        rhs = []

        for v in nodes:
            if v == root:
                continue
            row = [0] * len(edges)
            for i, e in enumerate(edges):
                src = self._node_key(e.get_source())
                dst = self._node_key(e.get_destination())
                if src == v:
                    row[i] = 1
                elif dst == v:
                    row[i] = -1
            rows.append(row)
            rhs.append(0)

        def _add_partition_row(part):
            row = [0] * len(edges)
            for i, e in enumerate(edges):
                if e in part:
                    row[i] = float(e.get_attributes().get("is_cut", 0))
                else:
                    row[i] = 0
            rows.append(row)
            rhs.append(0)

        part1, part2 = partition
        _add_partition_row(part1)
        _add_partition_row(part2)

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

    def check_routing(self, graph, partition):
        edges = graph.get_edges()
        nodes = []
        for e in edges:
            nodes.append(self._node_key(e.get_source()))
            nodes.append(self._node_key(e.get_destination()))
        nodes = sorted(set(nodes))

        for v in nodes:
            bdry = self.boundary_edges({v})
            sum_mom_k = 0
            sum_mom_p1 = 0
            sum_mom_p2 = 0
            for e in bdry:
                for ep in edges:
                    if ep.get_attributes()["id"] == e.get_attributes()["id"]:
                        sigma = 1 if self._node_key(ep.get_source()) == v else -1
                        sum_mom_k += sp.Rational(ep.get_attributes()["routing_k0"]) * sigma
                        sum_mom_p1 += sp.Rational(ep.get_attributes()["routing_p1"]) * sigma
                        sum_mom_p2 += sp.Rational(ep.get_attributes()["routing_p2"]) * sigma
            if sum_mom_k != 0 or sum_mom_p1 != 0 or sum_mom_p2 != 0:
                print(f"Error at node {v}: sum_mom_k={sum_mom_k}, sum_mom_p1={sum_mom_p1}, sum_mom_p2={sum_mom_p2}")
                return False

        for i in range(0, 1):
            sum_k = 0
            sum_p1 = 0
            sum_p2 = 0
            for e in partition[i]:
                for ep in edges:
                    if ep.get_attributes()["id"] == e.get_attributes()["id"]:
                        sum_k += sp.Rational(ep.get_attributes()["routing_k0"]) * sp.Rational(ep.get_attributes()["is_cut"])
                        sum_p1 += sp.Rational(ep.get_attributes()["routing_p1"]) * sp.Rational(ep.get_attributes()["is_cut"])
                        sum_p2 += sp.Rational(ep.get_attributes()["routing_p2"]) * sp.Rational(ep.get_attributes()["is_cut"])
            if sum_k != 0 or sum_p1 != (1 if i == 0 else 0) or sum_p2 != (1 if i == 1 else 0):
                print(f"Error at partition {i}: sum_k={sum_k}, sum_p1={sum_p1}, sum_p2={sum_p2}")
                return False

        return True

    def cut_graphs_with_routing(self, initial_massive, final_massive):
        initial_cuts, final_cuts = self.get_cutkosky_cuts_IF(initial_massive, final_massive)
        routed_cut_graphs = []

        def all_pairs(V):
            V = list(V)
            for labels in product("ABC", repeat=len(V)):
                if "A" not in labels or "B" not in labels:
                    continue
                V1 = [v for v, lab in zip(V, labels) if lab in ("A", "C")]
                V2 = [v for v, lab in zip(V, labels) if lab in ("B", "C")]
                yield V1, V2


        print("LABELLED GRAPHS")

        cycles=self.get_directed_cycles()
        for initial_cut in initial_cuts:
            for final_cut in final_cuts:
                connected_components = self.cut_splits_into_two_components(initial_cut, final_cut, True)
                if connected_components[0]:
                    graph = self.set_cut_labels_2(initial_cut, final_cut, copy.deepcopy(self.dot), cycles)#self.set_cut_labels(initial_cut, final_cut, connected_components)(initial_cut, final_cut, copy.deepcopy(self.dot), cycles)

                    print("GRAPH-----")
                    for edge in graph.get_edges():
                        print(edge)

                    all_pair_list = all_pairs(initial_cut)
                    for V1, V2 in all_pair_list:

                        idV1 = [e.get_attributes()["id"] for e in V1]
                        idV2 = [e.get_attributes()["id"] for e in V2]

                        new_graph = copy.deepcopy(graph)
                        new_graph.set_name(f"{self.dot.get_name()}_partition_{idV1}_{idV2}")
                        new_graph.set("num",str(self.num))
                        graph = self.route_cut_graph(new_graph, initial_cut, final_cut, [V1, V2])
                        routed_cut_graphs.append([initial_cut, final_cut, [V1, V2], graph])
                        if not self.check_routing(graph, [V1, V2]):
                            print("ERROR: Routing is wrongly assigned")
                            #raise Exception

        return routed_cut_graphs


    def cut_graphs_with_routing_leading_virtuality(self, initial_massive, final_massive):
        cut_graphs_with_routing = self.cut_graphs_with_routing(initial_massive, final_massive)
        cut_graphs_with_routing_LV = []

        #print("---------in LV--------")
        for graph in cut_graphs_with_routing:
            count_p1=False
            count_p2=False


            #print(f"------graph: {graph[3].get_name()}------")
            #print([e.get_attributes()["id"] for e in graph[0]])
            for e in graph[3].get_edges():
            #    print(e)

                if sp.Rational(e.get_attributes()["routing_p1"])!=0 and sp.Rational(e.get_attributes()["routing_k0"])==0 and sp.Rational(e.get_attributes()["routing_p2"])==0 and self._strip_quotes(e.get_attributes().get("particle", "")) != "a":
                    #print("------")
                    #print(e.get_attributes()["particle"])
                    #print(self._strip_quotes(e.get_attributes().get("particle", "")) != "a")
                    count_p1=True
                elif sp.Rational(e.get_attributes()["routing_p2"])!=0 and sp.Rational(e.get_attributes()["routing_k0"])==0 and sp.Rational(e.get_attributes()["routing_p1"])==0 and self._strip_quotes(e.get_attributes().get("particle", "")) != "a":
                    count_p2=True
            if count_p1 and count_p2:
                #print("PASSED")
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

        ee = self.remove_edge_attr(pydot.Edge(u, v, **attrs), "lmb_rep")

        e_new = self.remove_edge_attr(ee, "num")

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

        print("-----Original graph-----")
        for e in self.dot.get_edges():
            print(e)

        vacuum_graph = pydot.Dot(graph_type="digraph")
        vacuum_graph.set_name(self.dot.get_name())

        for e in self.dot.get_edges():
            if self.is_incoming_half_edge(e):
                incoming_edges.append(e)
            elif self.is_outgoing_half_edge(e):
                outgoing_edges.append(e)
            else:
                ep = self.copy_edge(e)
                ep.set("is_cut", "0")
                self.remove_edge_attr(ep, "lmb_rep")
                self.remove_edge_attr(ep, "num")
                vacuum_graph.add_edge(ep)

        if len(incoming_edges) != len(outgoing_edges):
            raise pygloopException("Vacuum graph is not balanced.")

        incoming_edges = self.sort_half_edges_by_port(incoming_edges)
        outgoing_edges = self.sort_half_edges_by_port(outgoing_edges)

        for e in incoming_edges:
            for ep in outgoing_edges:
                if self.get_part(e) == self.get_part(ep) and ep not in paired_up:
                    vacuum_graph.add_edge(self.remove_edge_attr(self.edge_fusion(e, ep), "lmb_rep"))
                    paired_up.append(ep)

        print("-----Vacuum graph-----")
        for e in vacuum_graph.get_edges():
            print(e)

        print("--------")

        return VacuumDotGraph(vacuum_graph, self.get_numerator(include_overall_factor=True))

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

    def to_string(self) -> str:
        return self.dot.to_string()


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
            cutkosky_cuts = graph.enumerate_cutkosky_cuts(incoming_edges, outgoing_edges)

            # Add a new line here to filter out graphs with no cutkosky cuts
            #
            #print("new graph----")

            for S in cutkosky_cuts:
                #print("cutkosky cut")
                c= graph.boundary_edges(S)
                massive_in_cut=[]
                for e in c:
                    particle = self._strip_quotes(str(e.get_attributes().get("particle", "")))
                    if particle not in ["d", "d~", "g"]:
                        massive_in_cut.append(particle)


                #print(massive_in_cut)



                if massive_in_cut == final_particles:
                    #print("True")
                    new_graphs.append(graph)

        return new_graphs

    def save_to_file(self, file_path: str):
        write_text_with_dirs(file_path, "\n\n".join([g.to_string() for g in self]))
