import json
import os
from types import ModuleType
from typing import Iterator, List, Optional, Set, Tuple

import pydot
from gammaloop import (  # iso\rt: skip # type: ignore # noqa: F401
    GammaLoopAPI,
    LogLevel,
    evaluate_graph_overall_factor,
    git_version,
)
from symbolica import E, Expression # pyright: ignore
from symbolica.community.idenso import simplify_color, simplify_gamma, simplify_metrics # pyright: ignore

from processes.dy.dy_graph_utils import (
    _is_ext,
    _node_key,
    boundary_edges,
    get_LR_components,
    get_spanning_tree,
)
from utils.cff import CFFStructure
from utils.utils import PYGLOOP_FOLDER

pjoin = os.path.join

gl_log_level = LogLevel.Off


def Es(expr: str) -> Expression:
    return E(expr.replace('"', ""), default_namespace="gammalooprs")


class routed_cut_graph(object):
    def __init__(self, graph, initial_cut, final_cut, partition):
        self.graph = graph
        self.initial_cut = initial_cut
        self.final_cut = final_cut
        self.partition = partition

    def get_n_loops(self) -> int:
        return len(
            set(self.graph.get_edges())
            - set(get_spanning_tree(self.graph))
            - set(self.initial_cut)
            - set(self.final_cut)
        )


class integrand_info(object):
    def __init__(
        self,
        num,
        graph: routed_cut_graph,
        cff_L,
        cff_R,
        s_bridge_sub_L,
        s_bridge_sub_R,
        has_s_bridge_L,
        has_s_bridge_R,
    ):
        self.num=num
        self.graph = graph
        self.cff_L = cff_L
        self.cff_R = cff_R
        self.s_bridge_sub_L = s_bridge_sub_L
        self.s_bridge_sub_R = s_bridge_sub_R
        self.has_s_bridge_L = has_s_bridge_L
        self.has_s_bridge_R = has_s_bridge_R


class IntegrandConstructor(object):
    def __init__(self, params, name):
        self.params = params
        self.name = name
        self.gl_worker = GammaLoopAPI(
            pjoin(PYGLOOP_FOLDER, "outputs", "gammaloop_states", self.name),
            log_file_name=self.name,
            log_level=gl_log_level,
        )
        # GAMMALOOP_STATE_FOLDER
        self.gl_worker.run("import model sm-default.json")

    def get_numerator(self, graph):
        num = E("1")
        for node in graph.get_nodes():
            if node.get_name() not in ["edge", "node"]:
                n_num = node.get("num")
                if n_num:
                    num *= Es(n_num)
        for edge in graph.get_edges():
            e_num = edge.get("num")
            if e_num:
                num *= Es(e_num)
        res=E(str(simplify_metrics(simplify_gamma(simplify_color(num))).expand()))
        out=res.replace(E("Q(y_,mink(4,x_))")*E("Q(z_,mink(4,x_))"),E("sp(y_,z_)"), repeat=True)

        return out

    def get_CFF(self, graph, subgraph_as_nodes, reversed_edge_flows_ids):

        cff_structure = self.gl_worker.generate_cff_as_json_string(
            dot_string=graph.to_string(),
            subgraph_nodes=subgraph_as_nodes,
            reverse_dangling=reversed_edge_flows_ids,
            orientation_pattern=None,
        )

        try:
            cff_structure = json.loads(cff_structure)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding CFF structure JSON: {e}") from e

        cff_structure = CFFStructure(cff_structure)

        return cff_structure

    def construct_00_cuts(self, info: integrand_info):
        energies=E("1")
        edge_ids=[e.get("id") for e in info.graph.graph.get_edges()]
        numerator=info.num.replace(E("sp(x_,y_)"),E("sigma(x_)*sigma(y_)*E(x_)*E(y_)-sp3D(q(x_),q(y_))"))
        if info.has_s_bridge_L:
            edge_ids.remove(info.s_bridge_sub_L["id_s"])
            energies*=1/(E(f"E({info.s_bridge_sub_L["id_p1"]})")+E(f"E({info.s_bridge_sub_L["id_p2"]})"))**2
            numerator=numerator.replace(E(f"E({info.s_bridge_sub_L["id_s"]})"),E(f"E({info.s_bridge_sub_L["id_p1"]})")+E(f"E({info.s_bridge_sub_L["id_p2"]})"))
            numerator=numerator.replace(E(f"sigma({info.s_bridge_sub_L["id_s"]})"),E("1"))
        if info.has_s_bridge_R:
            edge_ids.remove(info.s_bridge_sub_R["id_s"])
            energies*=1/(E(f"E({info.s_bridge_sub_R["id_p1"]})")+E(f"E({info.s_bridge_sub_R["id_p2"]})"))**2
            numerator=numerator.replace(E(f"E({info.s_bridge_sub_R["id_s"]})"),E(f"E({info.s_bridge_sub_R["id_p1"]})")+E(f"E({info.s_bridge_sub_R["id_p2"]})"))
            numerator=numerator.replace(E(f"sigma({info.s_bridge_sub_R["id_s"]})"),E("1"))
        for id in edge_ids:
            energies*=1/E(f"E({id})")

        esurfaces=E("1")
        if info.cff_L is not None:
            a=1

        if info.cff_R is not None:
            a=1

        return energies*numerator*esurfaces


    def construct_01_cuts(self, info: integrand_info):
        print("here")
        a = 1

    def construct_10_cuts(self, info: integrand_info):
        a = 1

    def construct_11_cuts(self, info: integrand_info):
        a = 1

    def eliminate_s_channel_bridges(self, graph: pydot.Dot, comp):
        new_comp = set()
        s_bridge_sub = {"id_s": 0, "id_p1": 0, "is_p2": 0}
        has_s_bridge = False
        for v in comp:
            bdry = boundary_edges(graph, {v})
            check = 0
            for e in bdry:
                if (
                    e.get_attributes()["routing_p1"] == "1"
                    and e.get_attributes()["routing_k0"] == "0"
                    and e.get_attributes()["routing_p2"] == "0"
                ):
                    check += 1
                    s_bridge_sub["id_p1"] = e.get_attributes()["id"]
                elif (
                    e.get_attributes()["routing_p2"] == "1"
                    and e.get_attributes()["routing_k0"] == "0"
                    and e.get_attributes()["routing_p1"] == "0"
                ):
                    check += 1
                    s_bridge_sub["id_p2"] = e.get_attributes()["id"]
                else:
                    s_bridge_sub["id_s"] = e.get_attributes()["id"]
            if check != 2:
                new_comp.add(v)
            else:
                has_s_bridge = True

        return has_s_bridge, s_bridge_sub, new_comp

    def get_integrand(self, cut_graph: routed_cut_graph):

        comps = get_LR_components(
            cut_graph.graph, cut_graph.initial_cut, cut_graph.final_cut
        )

        num = self.get_numerator(cut_graph.graph)

        has_s_bridge_L, s_bridge_sub_L, new_comp_L = self.eliminate_s_channel_bridges(
            cut_graph.graph, comps[0]
        )
        has_s_bridge_R, s_bridge_sub_R, new_comp_R = self.eliminate_s_channel_bridges(
            cut_graph.graph, comps[1]
        )

        dangling = [[], []]
        for i in [0, 1]:
            for e in boundary_edges(cut_graph.graph, set(comps[i])):
                e_cut_attributes = e.get_attributes()
                if e_cut_attributes.get("is_cut") is None:
                    raise ValueError("No is_cut attribute in cut edge... impossible!")
                elif e_cut_attributes.get("is_cut") == -1:
                    dangling[i].append(int(e_cut_attributes.get("id")))
                e_cut_attributes["is_cut_DY"] = e_cut_attributes.get("is_cut")

        for i in [0, 1]:
            for e in cut_graph.graph.get_edges():
                e_cut_attributes = e.get_attributes()
                if e_cut_attributes.get("is_cut", None) is not None:
                    e_cut_attributes.pop("is_cut")



        #cff_structure_L = self.get_CFF(cut_graph.graph, list(comps[0]), dangling[0])
        #print(cff_structure_L)

        comps = [new_comp_L, new_comp_R]

        if len(comps[0]) > 1:
            cff_structure_L = self.get_CFF(cut_graph.graph, list(comps[0]), dangling[0])
        else:
            cff_structure_L = None
        if len(comps[1]) > 1:
            cff_structure_R = self.get_CFF(cut_graph.graph, list(comps[1]), dangling[1])
        else:
            cff_structure_R = None

        # add cut info: which of the four cases (00, 01, 10, 11) this cut diagram will fall into
        partition = json.loads(cut_graph.graph.get_attributes()["partition"])

        graph_integrand_info = integrand_info(
            num,
            cut_graph,
            cff_structure_L,
            cff_structure_R,
            s_bridge_sub_L,
            s_bridge_sub_R,
            has_s_bridge_L,
            has_s_bridge_R,
        )

        if len(partition[0]) == 1 and len(partition[1]) == 1:
            print(self.construct_00_cuts(graph_integrand_info))
            return self.construct_00_cuts(graph_integrand_info)
        elif len(partition[0]) == 1 and len(partition[1]) > 1:
            return self.construct_01_cuts(graph_integrand_info)
        elif len(partition[0]) > 1 and len(partition[1]) == 1:
            return self.construct_10_cuts(graph_integrand_info)
        else:
            return self.construct_11_cuts(graph_integrand_info)
