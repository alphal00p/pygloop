import json
import os
from copy import deepcopy
from fractions import Fraction

import pydot
from gammaloop import (  # iso\rt: skip # type: ignore # noqa: F401
    GammaLoopAPI,
    LogLevel,
    evaluate_graph_overall_factor,
    git_version,
)
from symbolica import E, Expression  # pyright: ignore
from symbolica.community.idenso import (  # pyright: ignore
    simplify_color,
    simplify_gamma,
    simplify_metrics,
)

from processes.dy.dy_graph_utils import (
    _node_key,
    _strip_quotes,
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
        self.num = num
        self.graph = graph
        self.cff_L = cff_L
        self.cff_R = cff_R
        self.s_bridge_sub_L = s_bridge_sub_L
        self.s_bridge_sub_R = s_bridge_sub_R
        self.has_s_bridge_L = has_s_bridge_L
        self.has_s_bridge_R = has_s_bridge_R


class IntegrandConstructor(object):
    def __init__(self, params, name, L):
        self.L = L
        self.params = params
        self.name = name
        self.gl_worker = GammaLoopAPI(
            pjoin(PYGLOOP_FOLDER, "outputs", "gammaloop_states", self.name),
            log_file_name=self.name,
            log_level=gl_log_level,
        )
        # GAMMALOOP_STATE_FOLDER
        self.gl_worker.run("import model sm-default.json")

    def get_numerator(self, graph) -> Expression:
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
        res = E(str(simplify_metrics(simplify_gamma(simplify_color(num))).expand()))
        out = res.replace(
            E("Q(y_,mink(4,x_))") * E("Q(z_,mink(4,x_))"), E("sp(y_,z_)"), repeat=True
        )

        return out

    def get_CFF(
        self, graph, subgraph_as_nodes, reversed_edge_flows_ids
    ) -> CFFStructure:

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

    def construct_cuts(self, info: integrand_info) -> Expression:
        energies = E("1")
        edge_ids = [e.get("id") for e in info.graph.graph.get_edges()]
        numerator = info.num.replace(
            E("sp(x_,y_)"), E("sigma(x_)*sigma(y_)*E(x_)*E(y_)-sp3(q(x_),q(y_))")
        )
        if info.has_s_bridge_L:
            edge_ids.remove(info.s_bridge_sub_L["id_s"][0])
            energies *= (
                1
                / (
                    E(f"E({info.s_bridge_sub_L['id_p1']})")
                    + E(f"E({info.s_bridge_sub_L['id_p2']})")
                )
                ** 2
            )
            numerator = numerator.replace(
                E(f"E({info.s_bridge_sub_L['id_s'][0]})"),
                info.s_bridge_sub_L["id_s"][1]
                * (
                    E(f"E({info.s_bridge_sub_L['id_p1']})")
                    + E(f"E({info.s_bridge_sub_L['id_p2']})")
                ),
            )
            numerator = numerator.replace(
                E(f"sigma({info.s_bridge_sub_L['id_s'][0]})"), E("1")
            )
        if info.has_s_bridge_R:
            edge_ids.remove(info.s_bridge_sub_R["id_s"][0])
            energies *= (
                1
                / (
                    E(f"E({info.s_bridge_sub_R['id_p1']})")
                    + E(f"E({info.s_bridge_sub_R['id_p2']})")
                )
                ** 2
            )
            numerator = numerator.replace(
                E(f"E({info.s_bridge_sub_R['id_s'][0]})"),
                info.s_bridge_sub_R["id_s"][1]
                * (
                    E(f"E({info.s_bridge_sub_R['id_p1']})")
                    + E(f"E({info.s_bridge_sub_R['id_p2']})")
                ),
            )
            numerator = numerator.replace(
                E(f"sigma({info.s_bridge_sub_R['id_s'][0]})"), E("1")
            )
        for id in edge_ids:
            energies *= 1 / E(f"2*E({id})")

        total1 = E("0")

        if info.cff_L is not None:
            for cffterm in info.cff_L.expressions:
                cff_term = numerator * cffterm.expression
                for o, i in zip(
                    cffterm.orientation, range(0, len(cffterm.orientation))
                ):
                    if o.is_reversed():
                        cff_term = cff_term.replace(E(f"sigma({i})"), E("-1"))
                    if o.is_default():
                        cff_term = cff_term.replace(E(f"sigma({i})"), E("1"))
                total1 += cff_term

            for hetas in info.cff_L.h_surfaces:
                total1 = total1.replace(E(f"pygloop::γ({hetas.id})"), hetas.expression)

            for etas in info.cff_L.e_surfaces:
                total1 = total1.replace(E(f"pygloop::η({etas.id})"), etas.expression)
                # total1=total1.substitute()
        else:
            total1 += numerator

        total2 = E("0")
        if info.cff_R is not None:
            for cffterm in info.cff_R.expressions:
                cff_term = total1 * cffterm.expression
                for o, i in zip(
                    cffterm.orientation, range(0, len(cffterm.orientation))
                ):
                    if o.is_reversed():
                        cff_term = cff_term.replace(E(f"sigma({i})"), E("-1"))
                    if o.is_default():
                        cff_term = cff_term.replace(E(f"sigma({i})"), E("1"))
                total2 += cff_term

            for hetas in info.cff_R.h_surfaces:
                total2 = total2.replace(E(f"pygloop::γ({hetas.id})"), hetas.expression)

            for etas in info.cff_R.e_surfaces:
                total2 = total2.replace(E(f"pygloop::η({etas.id})"), etas.expression)

        else:
            total2 = total1

        return energies * total2

    def eliminate_s_channel_bridges(self, graph: pydot.Dot, comp):
        new_comp = comp
        s_bridge_sub = {"id_s": (0, 0), "id_p1": 0, "id_p2": 0}
        has_s_bridge = False
        for e in graph.get_edges():
            e_atts = e.get_attributes()
            if (
                (
                    e.get_attributes()["routing_p2"] != "0"
                    and e.get_attributes()["routing_k0"] == "0"
                    and e.get_attributes()["routing_p1"] != "0"
                )
                and _node_key(e.get_source()) in comp
                and _node_key(e.get_destination()) in comp
            ):
                has_s_bridge = True
                bdry_src = list(
                    set(boundary_edges(graph, {_node_key(e.get_source())})) - {e}
                )
                bdry_dest = list(
                    set(boundary_edges(graph, {_node_key(e.get_destination())})) - {e}
                )
                if all(ep.get_attributes().get("is_cut", 0) != 0 for ep in bdry_src):
                    new_comp = comp - {_node_key(e.get_source())}
                    sign = 1
                    if (
                        _node_key(bdry_src[0].get_source()) == _node_key(e.get_source())
                        and bdry_src[0].get_attributes()["is_cut"] == 1
                    ) or (
                        _node_key(bdry_src[0].get_destination())
                        == _node_key(e.get_source())
                        and bdry_src[0].get_attributes()["is_cut"] == -1
                    ):
                        sign = -1

                    s_bridge_sub["id_s"] = (e_atts["id"], sign)
                    if len(bdry_src) == 2:
                        s_bridge_sub["id_p1"] = bdry_src[0].get_attributes()["id"]
                        s_bridge_sub["id_p2"] = bdry_src[1].get_attributes()["id"]
                    else:
                        raise ValueError("problem with s-channel identification")
                elif all(ep.get_attributes().get("is_cut", 0) != 0 for ep in bdry_dest):
                    new_comp = comp - {_node_key(e.get_destination())}
                    sign = 1
                    if (
                        _node_key(bdry_dest[0].get_destination())
                        == _node_key(e.get_destination())
                        and bdry_src[0].get_attributes()["is_cut"] == 1
                    ) or (
                        _node_key(bdry_dest[0].get_source())
                        == _node_key(e.get_destination())
                        and bdry_src[0].get_attributes()["is_cut"] == -1
                    ):
                        sign = -1

                    s_bridge_sub["id_s"] = (e_atts["id"], sign)
                    if len(bdry_dest) == 2:
                        s_bridge_sub["id_p1"] = bdry_dest[0].get_attributes()["id"]
                        s_bridge_sub["id_p2"] = bdry_dest[1].get_attributes()["id"]
                    else:
                        raise ValueError("problem with s-channel identification")
                else:
                    raise ValueError("problem with s-channel identification")

        return has_s_bridge, s_bridge_sub, new_comp

    def linearise_scalar_products(self, integrand):
        integrand = integrand.replace(
            E("sp3(x__,z_+w__)"),
            E("sp3(x__, z_)") + E("sp3(x__,w__)"),
            repeat=True,
        )
        integrand = integrand.replace(
            E("sp3(x__,-z_+w__)"),
            -E("sp3(x__, z_)") + E("sp3(x__,w__)"),
            repeat=True,
        )
        integrand = integrand.replace(
            E("sp3(x_+y__,z__)"),
            E("sp3(x_, z__)") + E("sp3(y__,z__)"),
            repeat=True,
        )
        integrand = integrand.replace(
            E("sp3(-x_+y__,z__)"),
            -E("sp3(x_, z__)") + E("sp3(y__,z__)"),
            repeat=True,
        )
        integrand = integrand.replace(E("sp3(-x_,z_)"), E("-sp3(x_,z_)"))
        integrand = integrand.replace(E("sp3(x_,-z_)"), E("-sp3(x_,z_)"))
        return integrand

    def concretise_scalar_products(self, integrand):

        if self.name == "DY":
            integrand = integrand.replace(E("m(a)^2"), E("4*z*sp3(p(1),p(1))"))

        integrand = integrand.replace(
            E("sp3(k(x_),k(y_))"), E("k(x_,1)*k(y_,1)+k(x_,2)*k(y_,2)+k(x_,3)*k(y_,3)")
        )
        integrand = integrand.replace(
            E("sp3(k(x_),p(y_))"), E("k(x_,1)*p(y_,1)+k(x_,2)*p(y_,2)+k(x_,3)*p(y_,3)")
        )
        integrand = integrand.replace(
            E("sp3(p(x_),k(y_))"), E("p(x_,1)*k(y_,1)+p(x_,2)*k(y_,2)+p(x_,3)*k(y_,3)")
        )
        integrand = integrand.replace(
            E("sp3(p(x_),p(y_))"), E("p(x_,1)*p(y_,1)+p(x_,2)*p(y_,2)+p(x_,3)*p(y_,3)")
        )
        return integrand

    def t_parametrise(self, cut_integrand, cut_graph, process="DY"):

        if process == "DY":
            if len(cut_graph.final_cut) > 1:
                cut_integrand = cut_integrand.replace(
                    E("sp3(k(0),x___)"), E("t*sp3(k(0),x___)")
                )
                cut_integrand = cut_integrand.replace(
                    E("sp3(x___,k(0))"), E("t*sp3(x___,k(0))")
                )
                print(cut_integrand)
                t = E("4*sp3(p(1),p(1))-m(a)^2") / (
                    2 * E("sp3(k(0),k(0))^(1/2)") * 2 * E("sp3(p(1),p(1))^(1/2)")
                )
                cut_integrand = cut_integrand.replace(E("t"), t)
                return cut_integrand
            else:
                return cut_integrand

    def integrand_approximator(self, cut_graph: routed_cut_graph, integrand):
        partition = json.loads(cut_graph.graph.get_attributes()["partition"])

        approximated_integrand = integrand

        for e in cut_graph.graph.get_edges():
            e_atts = e.get_attributes()
            eid_raw = e_atts["id"]
            eid = _strip_quotes(eid_raw) if isinstance(eid_raw, str) else eid_raw
            approximated_integrand = approximated_integrand.replace(
                E(f"pygloop::E({eid})"), E(f"E({eid})")
            )

        if len(partition[0]) == 1 and len(partition[1]) == 1:
            for e in cut_graph.graph.get_edges():
                e_atts = e.get_attributes()
                eid_raw = e_atts["id"]
                eid = _strip_quotes(eid_raw) if isinstance(eid_raw, str) else eid_raw
                target = E(f"E({eid})")
                particle = _strip_quotes(str(e_atts["particle"]))
                if particle not in ["d", "d~", "g"]:
                    replacement = E(f"(sp3(q({eid}),q({eid}))+m({particle})^2)^(1/2)")
                else:
                    replacement = E(f"(sp3(q({eid}),q({eid})))^(1/2)")

                approximated_integrand = approximated_integrand.replace(
                    target, replacement
                )
            for e in cut_graph.graph.get_edges():
                e_atts = e.get_attributes()
                routing_items = E("0")
                for i in range(self.L + 1):
                    key = f"routing_k{i}"
                    if key in e_atts:
                        routing_items += E(f"{e_atts[key]}*k[{i}]")
                for i in range(0, 2):
                    key = f"routing_p{i + 1}"
                    if key in e_atts:
                        routing_items += E(f"{e_atts[key]}*p[{i + 1}]")
                approximated_integrand = approximated_integrand.replace(
                    E(f"q({e_atts['id']})"), routing_items
                )

            approximated_integrand = self.linearise_scalar_products(
                approximated_integrand
            )

            approximated_integrand = self.t_parametrise(
                approximated_integrand, cut_graph
            )

            approximated_integrand = self.concretise_scalar_products(
                approximated_integrand
            )

        ### FOLLOWING GEARED AS A REPLACEMENT IN k(0), MIGHT NOT GENERALISE
        if len(partition[0]) > 1 and len(partition[1]) == 1:
            for e in cut_graph.graph.get_edges():
                e_atts = e.get_attributes()
                e_id = e_atts["id"]
                particle = _strip_quotes(str(e_atts["particle"]))

                # FIX DIRECTIONS OF COLLINEAR LIMIT
                if e_id == partition[0][0]:
                    approximated_integrand = approximated_integrand.replace(
                        E(f"E({e_id})"),
                        E(
                            "x*sp3(p(1),p(1))^(1/2)+lam*sp3(p(1),p(1))^(-1/2)*x^(-1)*sp3(kperp(0),kperp(0))"
                        ),
                    )
                elif e_id == partition[0][1]:
                    approximated_integrand = approximated_integrand.replace(
                        E(f"E({e_id})"),
                        E(
                            "(1-x)*sp3(p(1),p(1))^(1/2)+lam*sp3(p(1),p(1))^(-1/2)*(1-x)^(-1)*sp3(kperp(0),kperp(0))"
                        ),
                    )
                elif e_atts["routing_k0"] == "0" and e_atts["routing_p2"] == "0":
                    approximated_integrand = approximated_integrand.replace(
                        E(f"E({e_id})"),
                        E("sp3(p(1),p(1))^(1/2)+lam*p1sq/sp3(p(1),p(1))^(1/2)"),
                    )
                else:
                    if particle not in ["d", "d~", "g"]:
                        replacement = E(
                            f"(sp3(q({e_id}),q({e_id}))+m({particle})^2)^(1/2)"
                        )
                    else:
                        replacement = E(f"(sp3(q({e_id}),q({e_id})))^(1/2)")
                    approximated_integrand = approximated_integrand.replace(
                        E(f"E({e_id})"), replacement
                    )

            approximated_integrand = approximated_integrand.series(
                E("p1sq"), 0, 0
            ).to_expression()

            collinear_momentum = E("0")
            for e in cut_graph.graph.get_edges():
                e_atts = e.get_attributes()
                routing_items = E("0")
                e_id = e_atts["id"]
                for i in range(self.L + 1):
                    key = f"routing_k{i}"
                    if key in e_atts:
                        routing_items += E(f"{e_atts[key]}*k({i})")
                for i in range(0, 2):
                    key = f"routing_p{i + 1}"
                    if key in e_atts:
                        routing_items += E(f"{e_atts[key]}*p({i + 1})")
                approximated_integrand = approximated_integrand.replace(
                    E(f"q({e_atts['id']})"), routing_items
                )
                if e_id == partition[0][0]:
                    collinear_momentum = [routing_items, e_atts["is_cut_DY"]]

            approximated_integrand = self.concretise_scalar_products(
                approximated_integrand
            )

            coeff = collinear_momentum[0].replace(E("b___+k(0)"), E("1"))
            coeff = coeff.replace(E("b___-k(0)"), E("-1"))
            repl = -(collinear_momentum[0] - coeff * E("k(0)")) + collinear_momentum[
                1
            ] * E("x*p(1)")

            for j in range(1, 4):
                repl_i = repl.replace(E("k(x_)"), E(f"k(x_,{j})"))
                repl_i = repl_i.replace(E("p(1)"), E(f"p(1,{j})"))
                repl_i = repl_i.replace(E("p(2)"), E(f"p(2,{j})"))
                approximated_integrand = approximated_integrand.replace(
                    E(f"k(0,{j})"), repl_i
                )

            approximated_integrand = (
                approximated_integrand
                .series(E("lam"), 0, -1)
                .to_expression()
                .replace(E("lam"), E("1"))
            )

            ## replacements only valid when hadrons are aligned with z axis
            repl_x = collinear_momentum[1] * E("(p(1,3)*k(0,3))/(p(1,3)*p(1,3))")
            repl_kperp = E("k(0,1)^2+k(0,2)^2")

            approximated_integrand = approximated_integrand.replace(E("x"), repl_x)
            approximated_integrand = approximated_integrand.replace(
                E("sp3(kperp(0),kperp(0))"), repl_kperp
            )

        ## choose rest frame
        approximated_integrand = approximated_integrand.replace(E("p(x_,2)"), E("0"))
        approximated_integrand = approximated_integrand.replace(E("p(x_,1)"), E("0"))

        return approximated_integrand

    def _routing_sign_match(self, e: pydot.Edge, ep: pydot.Edge):
        a = e.get_attributes()
        b = ep.get_attributes()

        # collect all routing keys (p1, p2 and any k*)
        keys = [k for k in set(a.keys()) | set(b.keys()) if k.startswith("routing_")]

        # if any key missing, treat as 0
        def val(attrs, k):
            return E(attrs.get(k, "0"))

        same = all(val(a, k) == val(b, k) for k in keys)
        opp = all(val(a, k) == -val(b, k) for k in keys)
        return same or opp

    def change_routing(self, graph, lmb_choice):
        edges = list(graph.get_edges())
        if len(edges) == 0:
            return graph

        # Collect routing key sets from graph.
        routing_keys = set()
        for e in edges:
            routing_keys.update(
                k for k in e.get_attributes().keys() if k.startswith("routing_")
            )
        k_keys = sorted(
            [k for k in routing_keys if k.startswith("routing_k")],
            key=lambda x: int(x.replace("routing_k", "")),
        )
        p_keys = sorted([k for k in routing_keys if k.startswith("routing_p")])
        n_loops = len(k_keys)

        print(k_keys)
        print(lmb_choice)

        if n_loops == 0:
            return graph

        if len(lmb_choice) != n_loops:
            raise ValueError(
                f"Invalid lmb_choice length: expected {n_loops}, got {len(lmb_choice)}"
            )

        def _to_frac(v):
            if isinstance(v, Fraction):
                return v
            return Fraction(str(v))

        def _frac_to_str(v: Fraction | int) -> str:
            v = Fraction(v)
            return str(v.numerator) if v.denominator == 1 else str(v)

        def _mat_inv(m):
            n = len(m)
            aug = [
                [Fraction(m[i][j]) for j in range(n)]
                + [Fraction(1 if i == j else 0) for j in range(n)]
                for i in range(n)
            ]
            for col in range(n):
                piv = None
                for row in range(col, n):
                    if aug[row][col] != 0:
                        piv = row
                        break
                if piv is None:
                    raise ValueError(
                        "Invalid lmb_choice: loop routing matrix is singular"
                    )
                if piv != col:
                    aug[col], aug[piv] = aug[piv], aug[col]

                pivot = aug[col][col]
                aug[col] = [x / pivot for x in aug[col]]

                for row in range(n):
                    if row == col:
                        continue
                    fac = aug[row][col]
                    if fac != 0:
                        aug[row] = [
                            aug[row][j] - fac * aug[col][j] for j in range(2 * n)
                        ]

            return [row[n:] for row in aug]

        def _mat_mul(a, b):
            return [
                [
                    sum(a[i][k] * b[k][j] for k in range(len(a[0])))
                    for j in range(len(b[0]))
                ]
                for i in range(len(a))
            ]

        def _norm_id(v):
            if isinstance(v, str):
                return _strip_quotes(v)
            return str(v)

        # Map ids to edge objects (ids can be quoted or unquoted in attrs).
        by_id = {}
        for e in edges:
            eid = _norm_id(e.get_attributes().get("id", ""))
            by_id[eid] = e

        lmb_edges = []
        for sel in lmb_choice:
            sid = _norm_id(sel)
            if sid not in by_id:
                raise ValueError(f"Edge id {sel} in lmb_choice not found in graph")
            lmb_edges.append(by_id[sid])

        # Build C and P from selected lambda edges:
        # l = C k + P p  =>  k = C^{-1} l - C^{-1} P p
        C = []
        P = []
        for e in lmb_edges:
            atts = e.get_attributes()
            C.append([_to_frac(atts.get(k, "0")) for k in k_keys])
            P.append([_to_frac(atts.get(p, "0")) for p in p_keys])

        C_inv = _mat_inv(C)

        # For every edge r = c k + p  =>  r = (c C^{-1}) l + (p - c C^{-1}P) p
        for e in edges:
            atts = e.get_attributes()
            c = [_to_frac(atts.get(k, "0")) for k in k_keys]
            p = [_to_frac(atts.get(pk, "0")) for pk in p_keys]

            new_k = [
                sum(c[a] * C_inv[a][j] for a in range(n_loops)) for j in range(n_loops)
            ]
            if len(p_keys) > 0:
                cCinvP = [
                    sum(new_k[a] * P[a][j] for a in range(n_loops))
                    for j in range(len(p_keys))
                ]
                new_p = [p[j] - cCinvP[j] for j in range(len(p_keys))]
            else:
                new_p = []

            for j, key in enumerate(k_keys):
                e.set(key, _frac_to_str(new_k[j]))
            for j, key in enumerate(p_keys):
                e.set(key, _frac_to_str(new_p[j]))

        return graph

    def get_integrand(self, cut_graph: routed_cut_graph):

        comps = get_LR_components(
            cut_graph.graph, cut_graph.initial_cut, cut_graph.final_cut
        )

        if len(cut_graph.initial_cut) > 1:
            lmb_choice = []
            for e in cut_graph.final_cut:
                # print(e.get_attributes()["particle"])
                e_atts = e.get_attributes()
                if _strip_quotes(str(e_atts["particle"])) == "a":
                    lmb_choice.append(e_atts["id"])
            # print(lmb_choice)
            cut_graph.graph = self.change_routing(cut_graph.graph, lmb_choice)

        else:
            raise ValueError("Implement me")

        num = self.get_numerator(cut_graph.graph)

        has_s_bridge_L, s_bridge_sub_L, new_comp_L = self.eliminate_s_channel_bridges(
            cut_graph.graph, comps[0]
        )
        has_s_bridge_R, s_bridge_sub_R, new_comp_R = self.eliminate_s_channel_bridges(
            cut_graph.graph, comps[1]
        )

        dangling = [[], []]

        cut_ids = [
            e.get_attributes()["id"]
            for e in cut_graph.initial_cut + cut_graph.final_cut
        ]

        # PROBLEM WITH DANGLING EDGES WHEN A SUBGRAPH'S BOUNDARY CONTAINS THE SAME EDGE TWICE
        for i in [0, 1]:
            for e in cut_graph.graph.get_edges():
                if e.get_attributes()["id"] in cut_ids:
                    e_cut_attributes = e.get_attributes()
                    src_sign = 1 if _strip_quotes(e.get_source()) in comps[i] else -1
                    if e_cut_attributes.get("is_cut") is None:
                        raise ValueError(
                            "No is_cut attribute in cut edge... impossible!"
                        )
                    elif (
                        src_sign * e_cut_attributes.get("is_cut") == -1
                    ):  ## FUNKY DANGLING LOGIC, CHECK CONVENTION
                        dangling[i].append(int(e_cut_attributes.get("id")))
                    e_cut_attributes["is_cut_DY"] = e_cut_attributes.get("is_cut")

        for i in [0, 1]:
            for e in cut_graph.graph.get_edges():
                e_cut_attributes = e.get_attributes()
                if e_cut_attributes.get("is_cut", None) is not None:
                    e_cut_attributes.pop("is_cut")

        raised_cut = []
        for e, i in zip(
            cut_graph.graph.get_edges(), range(len(cut_graph.graph.get_edges()))
        ):
            e_atts = e.get_attributes()
            for ep, j in zip(
                cut_graph.graph.get_edges(), range(len(cut_graph.graph.get_edges()))
            ):
                ep_atts = ep.get_attributes()
                if j > i and self._routing_sign_match(e, ep):
                    if (
                        e_atts.get("is_cut_DY", None) is not None
                        or ep_atts.get("is_cut_DY", None) is not None
                    ):
                        raised_cut.append([e_atts["id"], ep_atts["id"]])

        comps = [new_comp_L, new_comp_R]

        ## DEALING WITH SPECTATORS FOR CFF

        input_graph = deepcopy(cut_graph.graph)

        spectator_ids = [
            e.get_attributes()["id"]
            for e in set(cut_graph.initial_cut).intersection(set(cut_graph.final_cut))
        ]

        repl = []
        tot_e = len(input_graph.get_edges())
        for e in input_graph.get_edges():
            e_atts = e.get_attributes()
            e_atts.pop("sink", None)
            e_atts.pop("source", None)
            e_id = e_atts["id"]
            e_source = e.get_source()
            e_dest = e.get_destination()
            if e_id in spectator_ids:
                i = spectator_ids.index(e_id)
                edge_to_add_1 = pydot.Edge(f"ext{2 * i}", e_dest, **e_atts)
                edge_to_add_2 = pydot.Edge(e_source, f"ext{2 * i + 1}", **e_atts)
                edge_to_add_2.set("num", "1")
                edge_to_add_2.set("id", f"{tot_e + i}")
                repl.append([E(f"E({tot_e + i})"), E(f"E({e_id})")])
                input_graph.del_edge(e_source, e_dest, e_id)
                input_graph.add_edge(edge_to_add_1)
                input_graph.add_edge(edge_to_add_2)
                input_graph.add_node(pydot.Node(f"ext{2 * i}", style="invis"))
                input_graph.add_node(pydot.Node(f"ext{2 * i + 1}", style="invis"))
                if e_id in dangling[0]:
                    dangling[0].append(tot_e + i)
                if e_id in dangling[1]:
                    dangling[1].append(tot_e + i)

        if len(comps[0]) > 1:
            # cff_structure_L = self.get_CFF(cut_graph.graph, list(comps[0]), dangling[0])
            cff_structure_L = self.get_CFF(input_graph, list(comps[0]), dangling[0])
        else:
            cff_structure_L = None
        if len(comps[1]) > 1:
            # cff_structure_R = self.get_CFF(cut_graph.graph, list(comps[1]), dangling[1])
            cff_structure_R = self.get_CFF(input_graph, list(comps[1]), dangling[1])
        else:
            cff_structure_R = None

        # add cut info: which of the four cases (00, 01, 10, 11) this cut diagram will fall into
        # partition = json.loads(cut_graph.graph.get_attributes()["partition"])

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

        cut_integrand = self.construct_cuts(graph_integrand_info)

        for r in repl:
            cut_integrand = E(str(cut_integrand)).replace(r[0], r[1])

        if len(raised_cut) > 0:
            cut_integrand = cut_integrand * (
                E(f"E({raised_cut[0][0]})") - E(f"E({raised_cut[0][1]})")
            )

        for e in cut_graph.graph.get_edges():
            cut_val = e.get_attributes().get("is_cut_DY", None)
            cut_id = e.get_attributes().get("id", None)
            if cut_val is not None:
                cut_integrand = cut_integrand.replace(
                    E(f"sigma({cut_id})"), E(f"{cut_val}")
                )

        cut_integrand = self.integrand_approximator(cut_graph, cut_integrand)

        return cut_integrand


class evaluate_integrand(object):
    def __init__(self, L, cut_integrand, process):
        self.L = L
        self.process = process
        self.symbols = []
        for i in range(self.L):
            for j in range(1, 4):
                self.symbols.append(E(f"k({i},{j})"))
        for i in range(1, 3):
            for j in range(1, 4):
                self.symbols.append(E(f"p({i},{j})"))

        if process == "DY":
            self.symbols.append(E("z"))

        self.cut_integrand = cut_integrand

        self.cut_integrand = self.cut_integrand.replace(E("TR"), E("1/2"))
        self.cut_integrand = self.cut_integrand.replace(E("GC_11"), E("1"))
        self.cut_integrand = self.cut_integrand.replace(E("GC_1"), E("1"))

        self.evaluator = self.cut_integrand.evaluator({}, {}, self.symbols)

    def param_builder(self, k, p1, p2, z):
        param_list = []
        for i in range(self.L):
            for j in range(0, 3):
                param_list.append(k[i][j])
        for j in range(0, 3):
            param_list.append(p1[j])
        for j in range(0, 3):
            param_list.append(p2[j])
        if self.process == "DY":
            param_list.append(z)

        return param_list

    def eval(self, k, p1, p2, z):
        param_list = self.param_builder(k, p1, p2, z)
        return self.evaluator.evaluate(param_list)
