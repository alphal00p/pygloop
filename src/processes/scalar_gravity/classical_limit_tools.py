from __future__ import annotations

import copy
from typing import TYPE_CHECKING, List, Tuple

from symbolica import E, Evaluator, Expression, Replacement, S  # isort: skip # noqa: F401 # type: ignore
import itertools

from symbolica.community.idenso import (  # type: ignore
    cook_indices,  # noqa: F401
    simplify_color,  # noqa: F401
    simplify_gamma,  # noqa: F401
    simplify_metrics,  # noqa: F401
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


class ClassicalLimitProcessor(object):
    def __init__(self, process: ScalarGravity):
        self.process = process

    def get_color_projector(self) -> Expression:
        return E("1")

    def classical_limit_in_numerator(self, graph: DotGraph) -> None:
        for v in graph.dot.get_nodes():
            int_id = v.get_attributes().get("int_id", "").strip().strip('"')

            if int_id.startswith("V_S1S1"):
                num = v.get_attributes()["num"].strip().strip('"')
                replaced_num = expr_to_string(
                    E(num).replace(
                        E("Q(x_,spenso::mink(4,y_))"),
                        E("Q(0,spenso::mink(4,y_))"),
                    )
                )
                v.get_attributes()["num"] = replaced_num
            if int_id.startswith("V_S2S2"):
                num = v.get_attributes()["num"].strip().strip('"')
                replaced_num = expr_to_string(
                    E(num).replace(
                        E("Q(x_,spenso::mink(4,y_))"),
                        E("Q(1,spenso::mink(4,y_))"),
                    )
                )
                v.get_attributes()["num"] = replaced_num

    def adjust_projectors(self, g: DotGraph) -> None:
        attrs = g.get_attributes()
        attrs["projector"] = f'"{expr_to_string(g.get_projector() * self.get_color_projector())}"'
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

    def arrange_power_energies(self, g: DotGraph) -> List[List[Tuple[Tuple[int], Expression]]]:
        graviton_edges = set([e for e in g.dot.get_edges() if e.get_attributes()["particle"].strip('"') == "graviton"])

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
            if (not int_id.startswith("V_S1S1")) and (not int_id.startswith("V_S2S2")) and (not name_id.startswith("ext")):
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
            edge_id_replace = (-1 if _node_key(e_pat.get_source()) == v.get_name() else 1) * sum(
                (1 if _node_key(e.get_source()) == v.get_name() else -1) * E(f"Q({e.get_attributes()['id']},y___)") for e in es_sub
            )
            num = num.replace(edge_id_pattern, edge_id_replace)
            index_set = set([id[S("x_")] for id in num.match(E("Q(x_,y___)"))])

            # Group the terms in each vertex by edge_index: in other words, for a numerator Q(8)*Q(8)+Q(8)*Q(7)+Q(7)*Q(7),
            # write it as [[(8,8),Q(8)*Q(8)],[(8,7),Q(8)*Q(7)],[(7,7),Q(7)*Q(7)]] (of course terms are many more dure to index)
            # combinatorics

            subsets = list(itertools.combinations_with_replacement(sorted(index_set), len(index_set)))  # type: ignore
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

    def iterate_remove_square(self, g, possible_cuts, reduced_graph_edges_deg2, reduced_graph_edges_deg1):
        chosen_e = reduced_graph_edges_deg2[-1]

        nodes = set([v.get_name() for v in g.dot.get_nodes() if not v.get_name().startswith("ext")])
        v_s = _node_key(chosen_e.get_source())
        v_d = _node_key(chosen_e.get_destination())
        cuts_1 = self.subsets_containing_S1_not_S2(g.dot, nodes, {v_s}, {v_d})
        # cuts_2 = self.subsets_containing_S1_not_S2(g.dot, nodes, {v_d}, {v_s}, root)
        total_cuts = cuts_1  # + cuts_2

        new_possible_cuts = []

        if len(possible_cuts) == 0:
            for cut in total_cuts:
                bry_cut = set(boundary_edges(g.dot, cut)) - {chosen_e}
                if len(bry_cut.intersection(set(reduced_graph_edges_deg2).union(set(reduced_graph_edges_deg1)))) == 0:
                    new_possible_cuts.append([[chosen_e, cut]])
        else:
            for previous_cuts in possible_cuts:
                for cut in total_cuts:
                    bry_cut = set(boundary_edges(g.dot, cut)) - {chosen_e}
                    if len(bry_cut.intersection(set(reduced_graph_edges_deg2).union(set(reduced_graph_edges_deg1)))) == 0:
                        es_previous_cuts = set(
                            [
                                e
                                for previous_cut in previous_cuts
                                for e in [ep for ep in boundary_edges(g.dot, previous_cut[1]) if not _is_ext_edge(ep)]
                            ]
                        )
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
        reduced_graph_edges_deg2 = [e for e in g.dot.get_edges() if flattened_weights.count(int(e.get_attributes()["id"])) > 1]
        reduced_graph_edges_deg1 = [e for e in g.dot.get_edges() if flattened_weights.count(int(e.get_attributes()["id"])) == 1]

        reduced_graph_vertices = set([_node_key(e.get_source()) for e in reduced_graph_edges_deg2]).union(
            [_node_key(e.get_destination()) for e in reduced_graph_edges_deg2]
        )
        reduced_loops = len(reduced_graph_edges_deg2) - len(reduced_graph_vertices) + 1

        if reduced_loops <= 0:
            possible_cuts = []
            possible_cuts, reduced_graph_edges_deg2, reduced_graph_edges_deg1 = self.iterate_remove_square(
                g, possible_cuts, reduced_graph_edges_deg2, reduced_graph_edges_deg1
            )
            while len(reduced_graph_edges_deg2) > 0:
                possible_cuts, reduced_graph_edges_deg2, reduced_graph_edges_deg1 = self.iterate_remove_square(
                    g,
                    possible_cuts,
                    reduced_graph_edges_deg2,
                    reduced_graph_edges_deg1,
                )
            return possible_cuts
        elif len(reduced_graph_edges_deg2) == 0:
            return []
        else:
            raise ValueError("Not possible to arrange squared energies... check your graph.")

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
                        * sum((1 if _node_key(e.get_source()) in rep[1] else -1) * E(f"Q({e.get_attributes()['id']},y___)") for e in es_sub)
                    )

                # Make the replacement making sure that if the expression is Q(8)*Q(8), only one instance of Q(8)
                # gets substituted, i.e. Q(8)*Q(8)->(Q(2)-Q(3))*Q(8)

                for pupu in prod:
                    newp = E("0")
                    for monomial in pupu[1].expand():
                        new_monomial = monomial.replace(E("Q(x_,y___)*Q(z_,w___)*rest___"), E("rest___"))
                        q_factors = monomial.replace(
                            E("Q(x_,y___)*Q(z_,w___)*rest___"),
                            E("Q(x_,y___)*Q(z_,w___)"),
                        )
                        q_factor1 = monomial.replace(E("Q(x_,y___)*Q(z_,w___)*rest___"), E("Q(x_,y___)"))
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

            g.dot.set("num_matrices", reso)
            #            for r in reso:
            #                print("PRODUCTTTT")
            #                for p in r:
            #                    for mon in p[1].expand():
            #                        print("---------")
            #                        print(p[0])
            #                        print(mon)

            processed_graphs.append(g)

            # As an example, add a fake UV equal to the original graph
            processed_graphs.extend(self.generate_UV_CTs(g, group_id))

        return processed_graphs

    def remove_raised_power(self, graph: DotGraph):
        return graph
