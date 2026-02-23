from __future__ import annotations

import copy
import re
from collections import deque
from typing import TYPE_CHECKING, List, Optional, Tuple

import pydot

from symbolica import E, Evaluator, Expression, Replacement, S  # isort: skip # noqa: F401 # type: ignore
import itertools

from symbolica.community.idenso import (  # type: ignore
    cook_indices,  # noqa: F401
    simplify_color,  # noqa: F401
    simplify_gamma,  # noqa: F401
    simplify_metrics,  # noqa: F401
    to_dots,  # noqa: F401
)  # isort: skip # noqa: F401
from symbolica.community.spenso import (  # noqa: F401 # type: ignore
    TensorLibrary,
    TensorNetwork,
)
from ufo_model_loader.commands import Model  # noqa: F401 # type: ignore

# ====
# IMPORTS ACROSS PROCESSES *NOT* PERMITTED. THEY MUST REMAIN INDEPENDENT!
# REFACTOR NEEDED
# ====
# from processes.dy.dy_graph_utils import (
#     _is_ext_edge,
#     _node_key,
#     boundary_edges,
#     is_connected,
# )
from utils.utils import DotGraph, DotGraphs, expr_to_string

if TYPE_CHECKING:
    from processes.scalar_gravity.scalar_gravity import ScalarGravity


def Es(expr: str) -> Expression:
    return E(expr.replace('"', ""), default_namespace="gammalooprs")


# Remove surrounding double quotes if present.
def _strip_quotes(s: str) -> str:

    if isinstance(s, str):
        s = s.strip()
    else:
        raise TypeError(f"Expected str, got {type(s)}")

    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


# True if the vertex name matches "ext" followed by digits (ignoring quotes).
def _is_ext(name: str) -> bool:
    return bool(re.fullmatch(r"ext\d+", _strip_quotes(name)))


# True if an edge has a vertex that matches "ext" followed by digits (ignoring quotes).
def _is_ext_edge(e) -> bool:
    return _is_ext(e.get_source()) or _is_ext(e.get_destination())


def _node_key(endpoint: str, collapse_ports: bool = True) -> str:
    ep = _strip_quotes(endpoint)
    if collapse_ports and ":" in ep:
        return ep.split(":", 1)[0]
    return ep


# pydot vertices are in the form "v:port", where port is an int; returns (v,port)
def _parse_port_endpoint(endpoint: str) -> Optional[Tuple[str, int]]:
    ep = _strip_quotes(endpoint)
    m = re.fullmatch(r"([^:]+):(\d+)", ep)
    if not m:
        return None
    return m.group(1), int(m.group(2))


# Checks if the subgraph induced by a cut node_subset_input of the graph is connected
def is_connected(graph, node_subset_input) -> bool:

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
        for e in graph.get_edges():
            u = _base_node(e.get_source())
            v = _base_node(e.get_destination())

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


def _base_node(endpoint: str) -> str:
    ep = _strip_quotes(endpoint)
    parsed = _parse_port_endpoint(ep)
    return parsed[0] if parsed else ep


def _ext_idx(node: str) -> int | None:
    m = re.fullmatch(r"ext(\d+)", node)
    return int(m.group(1)) if m else None


def boundary_edges(graph: pydot.Dot, S: set[str]) -> List[pydot.Edge]:
    out = []
    edges = graph.get_edges()
    nodes = set()
    for e in edges:
        nodes.add(_base_node(e.get_source()))
        nodes.add(_base_node(e.get_destination()))
    if not set(S).issubset(nodes):
        raise ValueError("cut contains vertices not present in graph")
    for e in edges:
        u = _base_node(e.get_source())
        v = _base_node(e.get_destination())
        if (u in S and v not in S) or (u not in S and v in S):
            out.append(e)
    return out


class ClassicalLimitProcessor(object):
    def __init__(self, process: ScalarGravity):
        self.process = process

    def get_color_projector(self) -> Expression:
        return E("1")

    def ordered_path_vertices(self, line):
        # Build undirected adjacency from edges in `line`
        adj: dict[str, list[str]] = {}
        for e in line:
            u = _base_node(e.get_source())
            v = _base_node(e.get_destination())
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)

        # Path endpoints are degree-1 vertices
        ends = [v for v, nbrs in adj.items() if len(nbrs) == 1]
        print(ends)

        if len(ends) != 2:
            raise ValueError(f"Expected open path with 2 endpoints, got {len(ends)}")

        # Choose start endpoint: ext with smallest index
        ext_ends: list[tuple[str, int]] = []
        for v in ends:
            idx = _ext_idx(v)
            if idx is not None:
                ext_ends.append((v, idx))

        if not ext_ends:
            raise ValueError("No ext* endpoint found on path")

        start = min(ext_ends, key=lambda t: t[1])[0]

        # Walk path start -> end
        order = [start]
        prev = None
        cur = start
        while True:
            nxt = [x for x in adj[cur] if x != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            order.append(cur)

        return order

    def massive_edge_order(self, graph: DotGraph, edge, particle_type):

        line = []

        for e in graph.dot.get_edges():
            e_atts = e.get_attributes()
            particle = _strip_quotes(str(e_atts["particle"])).strip()
            if particle == particle_type:
                line.append(e)

        vertex_order = self.ordered_path_vertices(line)

        e_source = _node_key(edge.get_source())
        e_destination = _node_key(edge.get_destination())

        if vertex_order.index(e_source) < vertex_order.index(e_destination):
            return True
        else:
            return False

    def classical_limit_in_numerator(self, graph: DotGraph) -> None:

        for v in graph.dot.get_nodes():
            int_id = v.get_attributes().get("int_id", "").strip().strip('"')
            v_name = v.get_name()
            v_boundary = boundary_edges(graph.dot, {v_name})

            v_replacements = []
            for e in v_boundary:
                e_atts = e.get_attributes()
                particle = _strip_quotes(str(e_atts["particle"])).strip()

                if particle.startswith("scalar"):
                    orientation = self.massive_edge_order(graph, e, particle)
                    n = int(particle.rsplit("_", 1)[1]) - 1
                    if orientation:
                        v_replacements.append([
                            E(f"Q({e_atts['id']},spenso::mink(4,y_))"),
                            E(f"Q({n},spenso::mink(4,y_))"),
                        ])
                    else:
                        v_replacements.append([
                            E(f"Q({e_atts['id']},spenso::mink(4,y_))"),
                            -E(f"Q({n},spenso::mink(4,y_))"),
                        ])

            # check signs orientation
            if int_id.startswith("V_S1S1") or int_id.startswith("V_S2S2"):
                num = E(v.get_attributes()["num"].strip().strip('"'))

                for rep in v_replacements:
                    num = num.replace(rep[0], rep[1])

                replaced_num = expr_to_string(num)
                v.get_attributes()["num"] = replaced_num

    def adjust_projectors(self, g: DotGraph) -> None:
        attrs = g.get_attributes()
        attrs["projector"] = (
            f'"{expr_to_string(g.get_projector() * self.get_color_projector())}"'
        )
        return

    def set_group_id(self, g: DotGraph, group_id: int, is_master: bool = False) -> None:
        attrs = g.get_attributes()
        attrs["group_id"] = f'"{group_id}"'
        if is_master:
            attrs["group_master"] = '"true"'
        else:
            attrs["group_master"] = '"false"'

    def generate_UV_CTs(self, g: DotGraph, group_id: int) -> list[DotGraph]:
        match g.dot.get_name():
            case "v_diagram":
                fake_uv: DotGraph = copy.deepcopy(g)
                fake_uv.dot.set_name(f"{g.dot.get_name()}_UV_1")
                self.set_group_id(fake_uv, group_id, is_master=False)
                return [fake_uv]
            case _:
                # TODO
                return []

    # Will take each N-graviton vertex, and express all momenta in its Feynman rule in terms of N-1 momenta.
    # Will make sure that these N-1 momenta do not overlap across vertices.
    # if we represented powers of momenta as dots on edges, the resulting expression for the numerator would
    # have at most two dots per edge.

    def arrange_power_energies(
        self, g: DotGraph
    ) -> List[List[Tuple[Tuple[int], Expression]]]:
        graviton_edges = set([
            e
            for e in g.dot.get_edges()
            if e.get_attributes()["particle"].strip('"') == "graviton"
        ])

        selected_edges = []

        # To express all momenta in a graviton vertex in terms of N-1 momenta, determine substitution rules.
        # Given a vertex Q(1)*Q(2)+Q(1)*Q(1)+Q(1)*Q(3), will express one of the momenta in terms of the other,
        # e.g. Q(3)->Q(2)-Q(1) according to momentum conservation. After the substitution, the vertex will
        # only depend on Q(1) and Q(2). It will also make sure that no other vertex, after respective substitution,
        # contains Q(2) or Q(1)

        # Determine substitution rules

        for v in g.dot.get_nodes():
            int_id = v.get_attributes().get("int_id", "").strip().strip('"')
            name_id = v.get_name()
            if (
                (not int_id.startswith("V_S1S1"))
                and (not int_id.startswith("V_S2S2"))
                and (not name_id.startswith("ext"))
            ):
                v_id = v.get_name()
                bdry = list(boundary_edges(g.dot, {v_id}))

                if len(set(bdry).intersection(graviton_edges)) == len(bdry):
                    selected_edges.append((v, bdry[-1], bdry[:-1]))
                    graviton_edges = graviton_edges - set(bdry[:-1])
                else:
                    bdry_minus_graviton_edges = list(set(bdry) - graviton_edges)
                    selected_edges.append(
                        (
                            v,
                            bdry_minus_graviton_edges[0],
                            list(set(bdry).intersection(graviton_edges)),
                        ),
                    )
                    graviton_edges = graviton_edges - set(bdry)

        num_split = []

        # Enact the replacement rules determined above and group vertex numerator terms by edge ids.

        for v, e_pat, es_sub in selected_edges:
            # Enact the replacement rules determined above.

            num = E(v.get_attributes().get("num").strip('"'))
            if not num:
                continue

            edge_id_pattern = E(f"Q({e_pat.get_attributes()['id']},y___)")
            edge_id_replace = (
                -1 if _node_key(e_pat.get_source()) == v.get_name() else 1
            ) * sum(
                (1 if _node_key(e.get_source()) == v.get_name() else -1)
                * E(f"Q({e.get_attributes()['id']},y___)")
                for e in es_sub
            )
            num = num.replace(edge_id_pattern, edge_id_replace)
            index_set = set([id[S("x_")] for id in num.match(E("Q(x_,y___)"))])

            # Group the terms in each vertex by edge_index: in other words, for a numerator Q(8)*Q(8)+Q(8)*Q(7)+Q(7)*Q(7),
            # write it as [[(8,8),Q(8)*Q(8)],[(8,7),Q(8)*Q(7)],[(7,7),Q(7)*Q(7)]] (of course terms are many more dure to index)
            # combinatorics

            subsets = list(
                itertools.combinations_with_replacement(
                    sorted(index_set), len(index_set)
                )
            )  # type: ignore
            numerator_terms = [[sub, E("0")] for sub in subsets]
            for term in num.expand():
                ids = term.match(E("Q(x_,y___)"))
                term_ids = []
                for mat in ids:
                    term_ids.append(mat[S("x_")])
                for i, sub in zip(range(0, len(subsets)), subsets):
                    if tuple(term_ids) == sub:
                        numerator_terms[i][1] += term

            num_split.append(numerator_terms)

        return num_split

    # Determines connected subsets of S in g containing S1 but not S2

    def subsets_containing_S1_not_S2(self, g, S, S1, S2):
        if S1 & S2:
            return []  # impossible
        rest = list(S - S1 - S2)
        out = []
        for r in range(len(rest) + 1):
            for combo in itertools.combinations(rest, r):
                subset = S1 | set(combo)
                if is_connected(g, subset):
                    out.append(subset)
        return out

    # This function is called iteratively in the next function. It takes previous sets of cuts,
    # possible_cuts=[[S1,S2,...],[S3,S5,...],[S1,S4,...]], and aims at adding to each of the sets
    # of cuts another "compatible" cut if there exists one. "Compatibility" is defined as follows:
    # a) the cut's boundary must contain an edge which is associated with two powers of momentum in the
    # numerator, b) the cut's boundary must not contain any other edge that is associated with one or two
    # powers in the numerator, c) if we plan to add the cut to [S1,S2,...], then its boundary must not
    # contain any edge in the boundary of S1,S2,... (after follow up substitution rules, these edges
    # will also have one power of momentum in the numerator.

    def iterate_remove_square(
        self, g, possible_cuts, reduced_graph_edges_deg2, reduced_graph_edges_deg1
    ):
        chosen_e = reduced_graph_edges_deg2[-1]

        nodes = set([
            v.get_name()
            for v in g.dot.get_nodes()
            if not v.get_name().startswith("ext")
        ])
        v_s = _node_key(chosen_e.get_source())
        v_d = _node_key(chosen_e.get_destination())
        cuts_1 = self.subsets_containing_S1_not_S2(g.dot, nodes, {v_s}, {v_d})
        # cuts_2 = self.subsets_containing_S1_not_S2(g.dot, nodes, {v_d}, {v_s}, root)
        total_cuts = cuts_1  # + cuts_2

        new_possible_cuts = []

        if len(possible_cuts) == 0:
            for cut in total_cuts:
                bry_cut = set(boundary_edges(g.dot, cut)) - {chosen_e}
                if (
                    len(
                        bry_cut.intersection(
                            set(reduced_graph_edges_deg2).union(
                                set(reduced_graph_edges_deg1)
                            )
                        )
                    )
                    == 0
                ):
                    new_possible_cuts.append([[chosen_e, cut]])
        else:
            for previous_cuts in possible_cuts:
                for cut in total_cuts:
                    bry_cut = set(boundary_edges(g.dot, cut)) - {chosen_e}
                    if (
                        len(
                            bry_cut.intersection(
                                set(reduced_graph_edges_deg2).union(
                                    set(reduced_graph_edges_deg1)
                                )
                            )
                        )
                        == 0
                    ):
                        es_previous_cuts = set([
                            e
                            for previous_cut in previous_cuts
                            for e in [
                                ep
                                for ep in boundary_edges(g.dot, previous_cut[1])
                                if not _is_ext_edge(ep)
                            ]
                        ])
                        if len(bry_cut.intersection(es_previous_cuts)) == 0:
                            copy_cut = copy.deepcopy(previous_cuts)
                            copy_cut.append([chosen_e, cut])
                            new_possible_cuts.append(copy_cut)

        reduced_graph_edges_deg2 = reduced_graph_edges_deg2[:-1]

        return new_possible_cuts, reduced_graph_edges_deg2, reduced_graph_edges_deg1

    # Uses the previous function to construct a series of replacements that allow to redistribute edge momenta
    # that appear with power two to other edges, using momentum conservation.

    def get_squared_replacements(self, g: DotGraph, graph_weights):
        flattened_weights = [x for ws in graph_weights for x in ws]
        reduced_graph_edges_deg2 = [
            e
            for e in g.dot.get_edges()
            if flattened_weights.count(int(e.get_attributes()["id"])) > 1
        ]
        reduced_graph_edges_deg1 = [
            e
            for e in g.dot.get_edges()
            if flattened_weights.count(int(e.get_attributes()["id"])) == 1
        ]

        reduced_graph_vertices = set([
            _node_key(e.get_source()) for e in reduced_graph_edges_deg2
        ]).union([_node_key(e.get_destination()) for e in reduced_graph_edges_deg2])
        reduced_loops = len(reduced_graph_edges_deg2) - len(reduced_graph_vertices) + 1

        if reduced_loops <= 0:
            possible_cuts = []
            possible_cuts, reduced_graph_edges_deg2, reduced_graph_edges_deg1 = (
                self.iterate_remove_square(
                    g, possible_cuts, reduced_graph_edges_deg2, reduced_graph_edges_deg1
                )
            )
            while len(reduced_graph_edges_deg2) > 0:
                possible_cuts, reduced_graph_edges_deg2, reduced_graph_edges_deg1 = (
                    self.iterate_remove_square(
                        g,
                        possible_cuts,
                        reduced_graph_edges_deg2,
                        reduced_graph_edges_deg1,
                    )
                )
            return possible_cuts
        elif len(reduced_graph_edges_deg2) == 0:
            return []
        else:
            raise ValueError(
                "Not possible to arrange squared energies... check your graph."
            )

    # The result of self.arrange_power_energies(g) gives a set of summands of the numerator in the following form:
    # [[[(6,6),(terms of vertex1 that are in the form Q(6)*Q(6))],[(2,5),(terms of vertex2 that are in the form Q(2)*Q(5))]],... ]
    # It then uses get_squared replacements to replace extra powers of momentum in each factor so that each momentum appears once,
    # so that the output becomes, assuming momentum conservation gives Q(6)=Q(3)+Q(4),
    # [[[(6,6),(terms of vertex1 that are in the form Q(6)*(Q(3)+Q(4)))],[(2,5),(terms of vertex2 that are in the form Q(2)*Q(5))]],... ]

    def delocalize_numerators(self, g: DotGraph):
        num_split = self.arrange_power_energies(g)

        new_prods = []

        # prod is [[(6,6),(terms of vertex1 that are in the form Q(6)*Q(6))],[(2,5),(terms of vertex2 that are in the form Q(2)*Q(5))]]
        # in the example above

        for prod in itertools.product(*num_split):
            replacements = self.get_squared_replacements(g, [p[0] for p in prod])

            patterns_symb = []
            replacements_symb = []
            new_prod = []

            if len(replacements) > 0:
                # Write down symbolica replacements based on the output of self.get_squared_replacements(g, [p[0] for p in prod])

                for rep in replacements[0]:
                    e_pat = rep[0]
                    es_sub = [e for e in set(boundary_edges(g.dot, rep[1])) - {e_pat}]
                    patterns_symb.append(E(f"Q({e_pat.get_attributes()['id']},y___)"))
                    replacements_symb.append(
                        (-1 if _node_key(e_pat.get_source()) in rep[1] else 1)
                        * sum(
                            (1 if _node_key(e.get_source()) in rep[1] else -1)
                            * E(f"Q({e.get_attributes()['id']},y___)")
                            for e in es_sub
                        )
                    )

                # Make the replacement making sure that if the expression is Q(8)*Q(8), only one instance of Q(8)
                # gets substituted, i.e. Q(8)*Q(8)->(Q(2)-Q(3))*Q(8)

                for pupu in prod:
                    newp = E("0")
                    for monomial in pupu[1].expand():
                        new_monomial = monomial.replace(
                            E("Q(x_,y___)*Q(z_,w___)*rest___"), E("rest___")
                        )
                        q_factors = monomial.replace(
                            E("Q(x_,y___)*Q(z_,w___)*rest___"),
                            E("Q(x_,y___)*Q(z_,w___)"),
                        )
                        q_factor1 = monomial.replace(
                            E("Q(x_,y___)*Q(z_,w___)*rest___"), E("Q(x_,y___)")
                        )
                        q_factor2 = q_factors / q_factor1
                        for rep, sub in zip(patterns_symb, replacements_symb):
                            q_factor1 = q_factor1.replace(rep, sub)
                        newp += new_monomial * q_factor1 * q_factor2
                    new_pupu = [copy.deepcopy(pupu[0]), copy.deepcopy(newp)]
                    new_prod.append(new_pupu)

            else:
                new_prod = copy.deepcopy(prod)

            new_prods.append(new_prod)

        return new_prods

    def re_multiply_numerator(self, graph, graviton_numerator):

        sum = E("0")

        # Unfold numerator for graviton vertices

        for num in graviton_numerator:
            summand = E("1")
            for factor in num:
                summand *= factor
            sum += summand

        # Multiply in numerator for other vertices

        for v in graph.dot.get_nodes():
            int_id = v.get_attributes().get("int_id", "").strip().strip('"')
            if int_id.startswith("V_S1S1") or int_id.startswith("V_S2S2"):
                sum *= Es(v.get_attributes()["num"])
                v.get_attributes()["num"] = "1"

        # Multiply edge numerator

        for e in graph.dot.get_edges():
            sum *= Es(e.get_attributes()["num"])
            e.get_attributes()["num"] = "1"

        return sum

    def process_graphs(self, graphs: DotGraphs) -> DotGraphs:
        processed_graphs = DotGraphs()

        for group_id, g_input in enumerate(graphs):
            g: DotGraph = copy.deepcopy(g_input)

            # Add the main graph
            self.set_group_id(g, group_id, is_master=True)
            self.classical_limit_in_numerator(g)
            self.adjust_projectors(g)
            # self.delocalize_numerators(g)
            print("hiiiiiiiiiiiiiiiii")
            # num_split = self.arrange_power_energies(g)

            reso = self.delocalize_numerators(g)
            # print(reso[8][1][1])
            # g.dot.set("num_matrices", reso)

            numerator = self.re_multiply_numerator(g, reso)

            numerator = E(
                expr_to_string(to_dots(simplify_metrics(simplify_gamma(numerator))))
            ).replace(Es("UFO::dim"), E("4"), repeat=True)

            print("hereeee")
            print(expr_to_string(numerator))

            g.get_attributes()["num"] = expr_to_string(numerator)

            #            for r in reso:
            #                print("PRODUCTTTT")
            #                for p in r:
            #                    for mon in p[1].expand():
            #                        print("---------")
            #                        print(p[0])
            #                        print(mon)

            processed_graphs.append(g)

            # As an example, add a fake UV equal to the original graph
            # processed_graphs.extend(self.generate_UV_CTs(g, group_id))

        return processed_graphs

    def remove_raised_power(self, graph: DotGraph):
        return graph
