from __future__ import annotations

import copy
from functools import reduce
from logging import raiseExceptions
from typing import TYPE_CHECKING

from processes.dy.dy_graph_utils import boundary_edges

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

from processes.dy.dy_graph_utils import (
    _is_ext,
    _is_ext_edge,
    _node_key,
    boundary_edges,
    is_connected,
)
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

    def arrange_power_energies(self, g: DotGraph):

        graviton_edges = set([
            e
            for e in g.dot.get_edges()
            if e.get_attributes()["particle"].strip('"') == "graviton"
        ])

        selected_edges = []
        # to express all momenta in a graviton vertex in terms of N-1 momenta, determine substitution rules
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
                    selected_edges.append(
                        v,
                        list(set(bdry) - graviton_edges)[0],
                        list(set(bdry).intersection(graviton_edges)),
                    )
                    graviton_edges = graviton_edges - set(bdry)

        # Now enact the replacement rules determined above; also split vertices by index presence
        #
        num_split = []
        for v, e_pat, es_sub in selected_edges:
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
            subsets = list(
                itertools.combinations_with_replacement(
                    sorted(index_set), len(index_set)
                )
            )

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

    def delocalize_numerators(self, g: DotGraph):

        num_split = self.arrange_power_energies(g)

        print(num_split)

        numerator_products = []

        new_prods = []

        for prod in itertools.product(*num_split):
            replacements = self.get_squared_replacements(g, [p[0] for p in prod])

            patterns_symb = []
            replacements_symb = []
            new_prod = []

            if len(replacements) > 0:
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

                newp = E("0")
                for pupu in prod:
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

        # attrs = g.get_attributes()
        # attrs["num"] = f'"{expr_to_string(g.get_numerator())}"'
        # g.set_local_numerators_to_one()

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
            processed_graphs.append(g)

            # As an example, add a fake UV equal to the original graph
            processed_graphs.extend(self.generate_UV_CTs(g, group_id))

        return processed_graphs

    def remove_raised_power(self, graph: DotGraph):
        return graph
