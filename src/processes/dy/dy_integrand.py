import json
import os
import re
from copy import deepcopy
from fractions import Fraction

import pydot
from gammaloop import (  # iso\rt: skip # type: ignore # noqa: F401
    GammaLoopAPI,
    LogLevel,
    evaluate_graph_overall_factor,
    git_version,
)
from symbolica import E, Expression, S  # pyright: ignore
from symbolica.community.idenso import (  # pyright: ignore
    simplify_color,
    simplify_gamma,
    simplify_metrics,
)

from processes.dy.dy_graph_utils import (
    _node_key,
    _strip_quotes,
    get_LR_components,
)
from utils.cff import CFFStructure
from utils.utils import PYGLOOP_FOLDER

pjoin = os.path.join

gl_log_level = LogLevel.Off

debug = True


def Es(expr: str) -> Expression:
    return E(expr.replace('"', ""), default_namespace="gammalooprs")


# Little struct that makes it more manageable to deal with cut graphs


class routed_cut_graph(object):
    def __init__(self, graph, initial_cut, final_cut, partition):
        self.graph = graph
        self.initial_cut = initial_cut
        self.final_cut = final_cut
        self.partition = partition


# We will extract amplitude graphs from cut graphs. Amplitude graphs are graphs together with a list
# of replacements that allow to map edge ids in the amplitude graph back to those of the cut graph


class amplitude_graph(object):
    def __init__(self, graph, replacements):
        self.graph = graph
        self.replacements = replacements


class RoutedIntegrand(object):
    def __init__(self, integrand, cut_graph, replacements):
        self.integrand = integrand
        self.cut_graph = cut_graph
        self.replacements = replacements


# This class is responsible for generating the CFF representation of the cut graph


class EMRIntegrandConstructor(object):
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

    # Get the numerator of the graph

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

    # Get cff of a graph; the dependence on subgraph_as_nodes and reversed_edge_flows_ids
    # is explicit but is not used for the rest of the code. All graphs are amplitude graphs now.

    def get_CFF(
        self, graph, subgraph_as_nodes, reversed_edge_flows_ids
    ) -> CFFStructure:
        graph_for_cff = deepcopy(graph)
        self.canonicalize_ports_for_cff(graph_for_cff)

        cff_structure = self.gl_worker.generate_cff_as_json_string(
            dot_string=graph_for_cff.to_string(),
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

    # The following function canonicalises the nodes and the ports of a pydot graph so that they go from 0,...,N and 0,...,M.
    # This normalisation is needed for linnet. One could also remove the ports.

    def canonicalize_ports_for_cff(self, graph):
        port_re = re.compile(r"^([^:]+):(\d+)$")

        def parse_endpoint(endpoint):
            ep = _strip_quotes(str(endpoint))
            m = port_re.fullmatch(ep)
            if not m:
                return ep, None
            return m.group(1), int(m.group(2))

        # Build a deterministic node relabeling map: old label -> "0", "1", ..., "N".
        node_labels = []
        for n in graph.get_nodes():
            name = _strip_quotes(str(n.get_name()))
            if name in ["node", "edge", "graph"]:
                continue
            node_labels.append(name)

        for e in graph.get_edges():
            src_node, _ = parse_endpoint(e.get_source())
            dst_node, _ = parse_endpoint(e.get_destination())
            node_labels.append(src_node)
            node_labels.append(dst_node)

        unique_labels = list(dict.fromkeys(node_labels))
        node_map = {old: str(i) for i, old in enumerate(unique_labels)}

        remapped_nodes = []
        for n in graph.get_nodes():
            name = _strip_quotes(str(n.get_name()))
            if name in ["node", "edge", "graph"]:
                continue
            attrs = deepcopy(n.get_attributes())
            remapped_nodes.append(pydot.Node(node_map[name], **attrs))

        remapped_edges = []
        next_port = 0
        for e in graph.get_edges():
            attrs = deepcopy(e.get_attributes())

            src_node, src_port = parse_endpoint(e.get_source())
            src_node = node_map[src_node]
            if src_port is None:
                new_src = src_node
            else:
                new_src = f"{src_node}:{next_port}"
                next_port += 1

            dst_node, dst_port = parse_endpoint(e.get_destination())
            dst_node = node_map[dst_node]
            if dst_port is None:
                new_dst = dst_node
            else:
                new_dst = f"{dst_node}:{next_port}"
                next_port += 1

            remapped_edges.append(pydot.Edge(new_src, new_dst, **attrs))

        graph.obj_dict["nodes"] = {}
        graph.obj_dict["edges"] = {}
        for n in remapped_nodes:
            graph.add_node(n)
        for e in remapped_edges:
            graph.add_edge(e)

    # The following function can be used to remove unwanted attributes from the graph.

    def normalise_graph(self, graph):

        for e in graph.get_edges():
            e_atts = e.get_attributes()
            if e_atts["is_cut"] != 0:
                e_atts["is_cut_DY"] = e_atts["is_cut"]
            e_atts.pop("is_cut", None)
            e_atts.pop("source", None)
            e_atts.pop("num", None)
            e_atts.pop("sink", None)
            e_atts.pop("is_dummy", None)
            e_atts.pop("dir_in_cycle", None)

    # The following function takes a cut graph and gives out the two amplitude graphs which, glued together
    # give back the original cut graph. In order to do so it has to check if edges of the cut_graph are contained
    # in one or the other graph, if they are "externals", or if they are spectators.
    # Edge ids are normalised to that they go from 0,...,M

    def get_LR_graphs(self, cut_graph):
        comps = get_LR_components(
            cut_graph.graph, cut_graph.initial_cut, cut_graph.final_cut
        )

        graph_L = deepcopy(cut_graph.graph)
        graph_R = deepcopy(cut_graph.graph)
        new_graphs = [graph_L, graph_R]

        highest_ext = 0
        for v in cut_graph.graph.get_nodes():
            name = _strip_quotes(v.get_name())
            if name.startswith("ext"):
                suffix = name[3:]
                if suffix.isdigit():
                    highest_ext = max(highest_ext, int(suffix))

        # TODO: should check indexing logic, it seems a bit contrived.

        tot_e = 0
        replacements = [[], []]

        for i in [0, 1]:
            counter = 1
            for e in cut_graph.graph.get_edges():
                e_atts = e.get_attributes()
                src = e.get_source()
                dest = e.get_destination()
                src_key = _node_key(e.get_source())
                dest_key = _node_key(e.get_destination())
                if src_key not in comps[i] and dest_key not in comps[i]:
                    new_graphs[i].del_edge(src, dest, e_atts["id"])
                elif src_key in comps[i] and dest_key not in comps[i]:
                    new_graphs[i].del_edge(src, dest, e_atts["id"])
                    new_atts = deepcopy(e_atts)
                    new_atts["id"] = tot_e + counter - 1
                    new_graphs[i].add_edge(
                        pydot.Edge(src, f"ext{highest_ext + counter}", **new_atts)
                    )
                    new_graphs[i].add_node(
                        pydot.Node(f"ext{highest_ext + counter}", style="invis")
                    )
                    replacements[i].append([
                        tot_e + counter - 1,
                        e_atts["id"],
                    ])

                    counter += 1
                elif dest_key in comps[i] and src_key not in comps[i]:
                    new_graphs[i].del_edge(src, dest, e_atts["id"])
                    new_atts = deepcopy(e_atts)
                    new_atts["id"] = tot_e + counter - 1
                    new_graphs[i].add_edge(
                        pydot.Edge(f"ext{highest_ext + counter}", dest, **new_atts)
                    )
                    new_graphs[i].add_node(
                        pydot.Node(f"ext{highest_ext + counter}", style="invis")
                    )
                    replacements[i].append([
                        tot_e + counter - 1,
                        e_atts["id"],
                    ])
                    counter += 1
                elif (
                    dest_key in comps[i]
                    and src_key in comps[i]
                    and e_atts.get("is_cut_DY", None) is not None
                    and not (dest_key.startswith("ext") or src_key.startswith("ext"))
                ):
                    new_graphs[i].del_edge(src, dest, e_atts["id"])
                    new_atts1 = deepcopy(e_atts)
                    new_atts1["id"] = tot_e + counter - 1
                    new_graphs[i].add_edge(
                        pydot.Edge(f"ext{highest_ext + counter}", dest, **new_atts1)
                    )
                    new_graphs[i].add_node(
                        pydot.Node(f"ext{highest_ext + counter}", style="invis")
                    )
                    replacements[i].append([
                        tot_e + counter - 1,
                        e_atts["id"],
                    ])
                    counter += 1
                    new_atts2 = deepcopy(e_atts)
                    new_atts2["id"] = tot_e + counter - 1
                    new_graphs[i].add_edge(
                        pydot.Edge(src, f"ext{highest_ext + counter}", **new_atts2)
                    )
                    new_graphs[i].add_node(
                        pydot.Node(f"ext{highest_ext + counter}", style="invis")
                    )
                    replacements[i].append([
                        tot_e + counter - 1,
                        e_atts["id"],
                    ])
                    counter += 1
                else:
                    new_graphs[i].del_edge(src, dest, e_atts["id"])
                    new_atts1 = deepcopy(e_atts)
                    new_atts1["id"] = tot_e + counter - 1
                    new_graphs[i].add_edge(pydot.Edge(src, dest, **new_atts1))
                    replacements[i].append([tot_e + counter - 1, e_atts["id"]])
                    counter += 1

            edge_nodes = set()
            for e in new_graphs[i].get_edges():
                edge_nodes.add(_node_key(e.get_source()))
                edge_nodes.add(_node_key(e.get_destination()))
            for v in list(new_graphs[i].get_nodes()):
                if _node_key(v.get_name()) not in edge_nodes:
                    new_graphs[i].del_node(v)

        return amplitude_graph(new_graphs[0], replacements[0]), amplitude_graph(
            new_graphs[1], replacements[1]
        )

    # This function takes an amplitude graph and makes the composition of its replacements with
    # those of old replacements. In other words, indexes are propagated through the two replacements
    # and the final result is a single replacement expressing this chain of replacements

    def update_substitutions(self, graph: amplitude_graph, old_replacements):
        old_map = {src: dst for src, dst in old_replacements}
        composed = []
        for src, mid in graph.replacements:
            composed.append([src, old_map.get(mid, mid)])
        graph.replacements = composed

    # This function finds s_channel propagators and iteratively divides the amplitude graph into
    # amplitude subgraphs by deleting the s_channel propagators (which must be bridges). This is
    # needed because in the rest-frame the cff representation of massless s-channel propagators
    # is ill-defined. The amplitude subgraphs obtained by this procedure will later be individually fed to
    # the cff generator and the s-channel propagators will be added back by and in non-partial fractioned
    # form.

    def split_s_channels(self, graph: amplitude_graph):

        # Find s-channel propagators

        s_channel_edges = []
        for e in graph.graph.get_edges():
            e_atts = e.get_attributes()
            loop_keys = [f"routing_k{i}" for i in range(0, self.L)]
            s_channel = (
                all(e_atts[key] == "0" for key in loop_keys)
                and e_atts["routing_p1"] != "0"
                and e_atts["routing_p2"] != "0"
            )
            if s_channel:
                s_channel_edges.append(e)

        # Goes through one s-channel propagator at a time and divides the in two more subgraph by cutting it.

        s_channel_edges_copy = deepcopy(s_channel_edges)
        s_split_graphs = [graph]
        while len(s_channel_edges) > 0:
            chosen_s_edge = s_channel_edges.pop()
            chosen_src = chosen_s_edge.get_source()
            chosen_dest = chosen_s_edge.get_destination()
            check = False
            for g in s_split_graphs:
                for e in g.graph.get_edges():
                    src = e.get_source()
                    dest = e.get_destination()
                    if src == chosen_src and dest == chosen_dest:
                        s_cut_graph = routed_cut_graph(g.graph, [e], [], [])
                        graph_L, graph_R = self.get_LR_graphs(s_cut_graph)
                        s_split_graphs.remove(g)
                        self.update_substitutions(graph_L, g.replacements)
                        self.update_substitutions(graph_R, g.replacements)
                        s_split_graphs.append(graph_L)
                        s_split_graphs.append(graph_R)
                        check = True
                        break
                if check:
                    break

        return s_split_graphs, s_channel_edges_copy

    # This function takes a bunch of graphs, contained in s_split_graphs, and constructs the
    # cff for the product of these graphs assuming the numerator is num. It multiplies this by
    # the inverse energies for all edges in the initial and final state cuts (members of cut_graph)
    # and by the s-channel propagators (in the rest freme) contained in s_channel_edges

    def get_cff(self, cut_graph, s_split_graphs, s_channel_edges, num):

        numerator = num.replace(
            E("sp(x_,y_)"), E("sigma(x_)*sigma(y_)*E(x_)*E(y_)-sp3(q(x_),q(y_))")
        )

        cut_g_edges = sorted(
            cut_graph.graph.get_edges(), key=lambda e: int(e.get_attributes()["id"])
        )

        # We only derive the CFF of graphs that have more than one node. It's trivial otherwise
        # and the CFF generator crashes for these graphs for some reason.

        split_graphs_gt2_non_ext = []
        for g in s_split_graphs:
            non_ext_count = 0
            for v in g.graph.get_nodes():
                name = _strip_quotes(v.get_name())
                is_ext = name.startswith("ext") and name[3:].isdigit()
                if not is_ext:
                    non_ext_count += 1
            if non_ext_count > 1:
                split_graphs_gt2_non_ext.append(g)

        # Constructs the product of CFF representations obtained from all the subgraphs obtained
        # by deleting s-channel edges and cut edges. Also reverses sign of external edges that
        # are cut and have negative energy flow.

        previous_cff = numerator

        for g in split_graphs_gt2_non_ext:
            cff_g = self.get_CFF(g.graph, [], [])
            new_cff = E("0")
            g_rep = g.replacements

            for cffterm in cff_g.expressions:
                cff_term = previous_cff * cffterm.expression
                edges_to_reverse = []
                for o, i in zip(
                    cffterm.orientation, range(0, len(cffterm.orientation))
                ):
                    id_in_original_graph = g_rep[i][1]
                    this_e_atts = cut_g_edges[id_in_original_graph].get_attributes()
                    if o.is_reversed():
                        cff_term = cff_term.replace(
                            E(f"sigma({id_in_original_graph})"), E("-1")
                        )
                    if o.is_default():
                        cff_term = cff_term.replace(
                            E(f"sigma({id_in_original_graph})"), E("1")
                        )
                    if this_e_atts.get("is_cut_DY", 0) == -1:
                        edges_to_reverse.append(id_in_original_graph)
                for etas in cff_g.e_surfaces:
                    eta = etas.expression
                    for rep in g.replacements:
                        eta = eta.replace(E(f"pygloop::E({rep[0]})"), E(f"E({rep[1]})"))
                    cff_term = cff_term.replace(E(f"pygloop::η({etas.id})"), eta)

                for id in set(edges_to_reverse):
                    cff_term = cff_term.replace(E(f"E({id})"), -E(f"E({id})"))

                new_cff += cff_term
            previous_cff = new_cff

        # Multiplies by non-partial-fractioned s-channel propagators and sets the s-channel particle's
        # energy in terms of other cut particles by energy conservation. The logic is weak for many s-channel
        # propagators since you can have a subgraph sandwiched between two s-channel propagators (TODO: FIX).
        # Also sets the sign of the propagator sigma(i) by one (i.e. according to original orientation)

        print("prior to setting cut sign")
        print(previous_cff)

        popping_edges = deepcopy(s_channel_edges)

        while len(popping_edges) > 0:
            current_s_edge = popping_edges.pop()
            s_edge_atts = current_s_edge.get_attributes()
            check = False
            for g in s_split_graphs:
                g_rep = g.replacements
                for e in g.graph.get_edges():
                    e_atts = e.get_attributes()
                    e_src = e.get_source()
                    if s_edge_atts["name"] == e_atts["name"]:
                        denom = E("0")
                        for ep in g.graph.get_edges():
                            ep_src = ep.get_source()
                            ep_dest = ep.get_destination()
                            ep_atts = ep.get_attributes()
                            if ep_src.startswith("ext") and ep != e:
                                denom += ep_atts["is_cut_DY"] * E(
                                    f"E({g.replacements[ep_atts['id']][1]})"
                                )
                            if ep_dest.startswith("ext") and ep != e:
                                denom -= ep_atts["is_cut_DY"] * E(
                                    f"E({g.replacements[ep_atts['id']][1]})"
                                )
                        previous_cff = previous_cff / denom**2
                        sign = 1 if e_src.startswith("ext") else -1
                        previous_cff = previous_cff.replace(
                            E(f"E({g.replacements[e_atts['id']][1]})"), -sign * denom
                        )
                        previous_cff = previous_cff.replace(
                            E(f"sigma({g.replacements[e_atts['id']][1]})"), E("1")
                        )
                        check = True
                        break
                if check:
                    break

        # Multiplies by inverse cut energies and substitutes their orientation acoording to
        # the cut.

        energies = E("1")
        for e in cut_graph.graph.get_edges():
            e_atts = e.get_attributes()
            cut_val = e_atts.get("is_cut_DY", None)

            if e not in s_channel_edges:
                energies *= 1 / E(f"2*E({e_atts['id']})")

            if cut_val is not None:
                previous_cff = previous_cff.replace(
                    E(f"sigma({e_atts['id']})"),
                    cut_val,
                )

        return previous_cff * energies

    # Use all the previous functions to get the cff for the cut graph

    def get_integrand(self, cut_graph: routed_cut_graph):

        # Derives numerator, eliminates useless labels, get left and right graphs and further
        # splits them if they have s-channel propagators.

        num = self.get_numerator(cut_graph.graph)

        self.normalise_graph(cut_graph.graph)

        graph_L, graph_R = self.get_LR_graphs(cut_graph)

        s_split_graphs_L, s_channel_edges_L = self.split_s_channels(graph_L)
        s_split_graphs_R, s_channel_edges_R = self.split_s_channels(graph_R)

        if debug:
            print(num)
            print("L graph")
            print(graph_L.graph)
            print(graph_L.replacements)
            print("R graph")
            print(graph_R.graph)
            print(graph_R.replacements)

            print("s-channel splitting for L graph")
            print("length: ", len(s_split_graphs_L))
            for g in s_split_graphs_L:
                print(g.graph)
                print(g.replacements)

            print("s-channel splitting for R graph")
            print("length: ", len(s_split_graphs_R))
            for g in s_split_graphs_R:
                print(g.graph)
                print(g.replacements)

            print(
                self.get_cff(
                    cut_graph,
                    s_split_graphs_L + s_split_graphs_R,
                    s_channel_edges_L + s_channel_edges_R,
                    num,
                )
            )

        ## DEBUG: set numerator to 1
        # num = E("1")

        return self.get_cff(
            cut_graph,
            s_split_graphs_L + s_split_graphs_R,
            s_channel_edges_L + s_channel_edges_R,
            num,
        )


class LoopIntegrandConstructor(object):
    def __init__(self, params, name, L):
        self.L = L
        self.params = params
        self.name = name
        self.emr_processor = EMRIntegrandConstructor(params, name, L)
        self.sp3D = S("sp3D", is_linear=True, is_symmetric=True)

    def replace_energies(self, integrand, cut_graph):

        for e in cut_graph.graph.get_edges():
            e_atts = e.get_attributes()
            eid_raw = e_atts["id"]
            eid = _strip_quotes(eid_raw) if isinstance(eid_raw, str) else eid_raw
            target = E(f"E({eid})")
            particle = _strip_quotes(str(e_atts["particle"]))
            if particle not in ["d", "d~", "g"]:
                replacement = (
                    self.sp3D(E(f"q({eid})"), E(f"q({eid})")) + E(f"m({particle})") ** 2
                ) ** E("1/2")
            else:
                replacement = (self.sp3D(E(f"q({eid})"), E(f"q({eid})"))) ** E("1/2")

            integrand = integrand.replace(target, replacement)

        return integrand

    def route_integrand(self, integrand, cut_graph):

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
            integrand = integrand.replace(E(f"q({e_atts['id']})"), routing_items)

        integrand = integrand.replace(
            E("sp3(x___,y___)"), self.sp3D(S("x___"), S("y___"))
        )
        return integrand

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

    def approximator(self, integrand, cut_graph):

        partition = cut_graph.partition

        if len(partition[0]) == 1 and len(partition[1]) == 1:
            integrand = self.replace_energies(integrand, cut_graph)
            integrand = self.route_integrand(integrand, cut_graph)
            # integrand = self.t_parametrise(integrand, cut_graph)
            return RoutedIntegrand(integrand, cut_graph, [])

        elif len(partition[0]) > 1 and len(partition[1]) == 1:
            x = S("x", is_scalar=True, is_positive=True)
            lam = S("λ", is_scalar=True)
            momentum = E("0")

            coll_moms = []
            for ep in partition[0]:
                ep_atts = ep.get_attributes()
                print("partition edge")
                print(ep)
                id = ep_atts["id"]
                coll_moms.append(id)
                for e in cut_graph.graph.get_edges():
                    e_atts = e.get_attributes()
                    if e_atts["id"] == id:
                        print("found matching")
                        print(e)
                        k_keys = ["routing_k" + str(i) for i in range(0, self.L)]
                        loop_coeff = [E(e_atts[rout]) for rout in k_keys]
                        k_id = next(
                            (i, c) for i, c in enumerate(loop_coeff) if str(c) != "0"
                        )
                        momentum = [
                            (
                                sum(
                                    loop_coeff[i] * E(f"k({i})")
                                    for i in range(0, self.L)
                                )
                                + E(e_atts["routing_p1"]) * E("p(1)")
                                + E(e_atts["routing_p2"]) * E("p(2)")
                            ),
                            e_atts["is_cut_DY"],
                        ]
                        print(momentum)

            kperp_sq = E(f"sp3D(k_perp({k_id[0]}),k_perp({k_id[0]}))")
            coll_en = x * E("sp3D(p(1),p(1))^(1/2)") + lam**2 * kperp_sq / (
                2 * x * E("sp3D(p(1),p(1))^(1/2)")
            )
            a_coll_en = (1 - x) * E("sp3D(p(1),p(1))^(1/2)") + lam**2 * kperp_sq / (
                2 * (1 - x) * E("sp3D(p(1),p(1))^(1/2)")
            )

            # In order to make the collinear replacement always work, replace a final-state energy
            # by energy conservation.

            rep = E("0")

            f_cut_set = set(cut_graph.final_cut)
            i_cut_set = set(cut_graph.initial_cut)
            cut_union = f_cut_set.union(i_cut_set)
            cut_intersection = f_cut_set.intersection(i_cut_set)

            first_f = True
            patt = None
            for e in cut_union - cut_intersection:
                print(e)
                e_atts = e.get_attributes()
                if first_f and e in f_cut_set:
                    patt = E(f"E({e_atts['id']})")
                    first_f = False
                elif e in f_cut_set:
                    rep -= E(f"E({e_atts['id']})")
                else:
                    rep += E(f"E({e_atts['id']})")
            if patt is None:
                raise ValueError("Could not identify a final-cut energy to replace.")

            integrand = integrand.replace(patt, rep)

            integrand = integrand.replace(E(f"E({coll_moms[0]})"), coll_en)
            integrand = integrand.replace(E(f"E({coll_moms[1]})"), a_coll_en)

            integrand = self.replace_energies(integrand, cut_graph)
            integrand = self.route_integrand(integrand, cut_graph)

            # Let s*q(i) be the vector that should become collinear to p(1). s encodes the cut orientation. We
            # write s*q(i)=s*(q(i)-a*k(j))+a*s*k(j)= x*p(1)+lam*k_perp(j) and solve in k(j), giving
            # k(j)=a*s*x*p(1)+a*s*lam*k_perp(j)-a*(q(i)-a*k(j)). Now s=momentum[1] and a=k_id[1] and j=k_id[0].

            repl = k_id[1] * (
                momentum[1] * x * E("p(1)")
                - (momentum[0] - k_id[1] * E(f"k({k_id[0]})"))
                + lam * E(f"k_perp({k_id[0]})")
            )
            integrand = integrand.replace(E(f"k({k_id[0]})"), repl)
            integrand = integrand.replace(
                self.sp3D(E(f"k_perp({k_id[0]})"), E("p(x_)")), E("0")
            )

            # Only consider leading-virtuality contribution

            integrand = integrand.series(lam, 0, -1).to_expression().replace(lam, 1)

            # Invert back the collinear parametrisation. Since s*q(i)= x*p(1)+lam*k_perp(j), we have
            # x=s*q(i).p(1)/p(1).p(1)

            repl_x = (
                momentum[1]
                * self.sp3D(momentum[0], E("p(1)"))
                / self.sp3D(E("p(1)"), E("p(1)"))
            )

            repl_kperp = E(f"k({k_id[0]},1)^2+k({k_id[0]},2)^2")

            integrand = integrand.replace(E("x"), repl_x)
            integrand = integrand.replace(
                self.sp3D(E("k_perp(0)"), E("k_perp(0)")), repl_kperp
            )

            return RoutedIntegrand(
                integrand,
                cut_graph,
                [
                    E(f"k({k_id[0]})"),
                    repl.replace(x, repl_x).series(lam, 0, 0).to_expression(),
                ],
            )

        else:
            raise ValueError("reached not implemented part")

    def get_integrand(self, cut_graph):
        emr_integrand = self.emr_processor.get_integrand(cut_graph)

        if len(cut_graph.final_cut) > 1 and self.name == "DY":
            lmb_choice = []
            for e in cut_graph.final_cut:
                e_atts = e.get_attributes()
                if _strip_quotes(str(e_atts["particle"])) == "a":
                    lmb_choice.append(e_atts["id"])
            cut_graph.graph = self.change_routing(cut_graph.graph, lmb_choice)

        loop_integrand = self.approximator(emr_integrand, cut_graph)

        return loop_integrand


class evaluate_integrand(object):
    def impose_rest_frame(self, integrand):
        return integrand.replace(E("p(x_,1)"), E("0")).replace(E("p(x_,2)"), E("0"))

    def concretise_scalar_products(self, integrand):

        if self.process == "DY":
            integrand = integrand.replace(E("m(a)^2"), E("4*z*sp3D(p(1),p(1))"))

        integrand = integrand.replace(
            E("sp3D(w_(x_),z_(y_))"),
            E("w_(x_,1)*z_(y_,1)+w_(x_,2)*z_(y_,2)+w_(x_,3)*z_(y_,3)"),
        )

        return integrand

    def t_parametrise(self, integrand):

        t = S("t")
        integrand = integrand.replace(
            E("k(x___,y___)"),
            t * E("k(x___,y___)"),
        )
        print(integrand)

        return integrand

    def __init__(self, L, process, routed_integrand):
        self.L = L
        self.process = process
        self.routed_integrand = routed_integrand

        self.symbols = []
        for i in range(self.L):
            for j in range(1, 4):
                self.symbols.append(E(f"k({i},{j})"))
        for i in range(1, 3):
            for j in range(1, 4):
                self.symbols.append(E(f"p({i},{j})"))

        if process == "DY":
            self.symbols.append(E("z"))

        self.symbols.append(E("t"))

        self.routed_integrand.integrand = self.concretise_scalar_products(
            self.routed_integrand.integrand
        )
        self.routed_integrand.integrand = self.impose_rest_frame(
            self.routed_integrand.integrand
        )
        self.routed_integrand.integrand = self.t_parametrise(
            self.routed_integrand.integrand
        )

        self.routed_integrand.integrand = self.routed_integrand.integrand.replace(
            E("TR"), E("1/2")
        )
        self.routed_integrand.integrand = self.routed_integrand.integrand.replace(
            E("GC_11"), E("1")
        )
        self.routed_integrand.integrand = self.routed_integrand.integrand.replace(
            E("GC_1"), E("1")
        )
        self.evaluator = self.routed_integrand.integrand.evaluator({}, {}, self.symbols)

        self.sp3D = S("sp3D", is_linear=True, is_symmetric=True)

    def set_t_value(self, k, p1, p2, z):

        if self.process == "DY":
            if len(self.routed_integrand.cut_graph.final_cut) > 1:
                final_moms = []
                e_surface = E("-s^(1/2)")

                for ep in self.routed_integrand.cut_graph.final_cut:
                    ep_atts = ep.get_attributes()
                    id = ep_atts["id"]
                    for e in self.routed_integrand.cut_graph.graph.get_edges():
                        e_atts = e.get_attributes()
                        if e_atts["id"] == id:
                            k_keys = ["routing_k" + str(i) for i in range(0, self.L)]
                            loop_coeff = [E(e_atts[rout]) for rout in k_keys]
                            mass = (
                                E("0")
                                if _strip_quotes(e_atts["particle"]) != "a"
                                else E("m(a)")
                            )
                            final_mom = (
                                sum(
                                    loop_coeff[i] * E(f"k({i})")
                                    for i in range(0, self.L)
                                )
                                + E(e_atts["routing_p1"]) * E("p(1)")
                                + E(e_atts["routing_p2"]) * E("p(2)")
                            )
                            final_moms.append([
                                final_mom,
                                mass,
                            ])
                            e_surface += (
                                self.sp3D(final_mom, final_mom) + mass**2
                            ) ** E("1/2")

                print(e_surface)
                e_surface = self.concretise_scalar_products(e_surface)
                print("before substitutions")
                print(e_surface)
                if len(self.routed_integrand.replacements) > 0:
                    patts = [
                        self.concretise_scalar_products(
                            self.routed_integrand.replacements[0]
                        ).replace(E("x_(y_)"), E(f"x_(y_,{i})"))
                        for i in range(1, 4)
                    ]
                    repls = [
                        self.concretise_scalar_products(
                            self.routed_integrand.replacements[1]
                        ).replace(E("x_(y_)"), E(f"x_(y_,{i})"))
                        for i in range(1, 4)
                    ]
                    for i in range(0, 3):
                        e_surface = e_surface.replace(patts[i], repls[i])

                    e_surface = self.concretise_scalar_products(e_surface)
                    print("here")
                    print(e_surface)
                    e_surface = self.concretise_scalar_products(e_surface)
                    print(e_surface)
                    e_surface = self.impose_rest_frame(e_surface)
                    print("here")
                    print(e_surface)

                    e_surface = e_surface.replace(E("k(0,x_)"), E("t*k(0,x_)"))
                    s = 4 * p1[2] ** 2
                    input_vals = {
                        E("k(0,1)"): k[0][0],
                        E("k(0,2)"): k[0][1],
                        E("k(0,3)"): k[0][2],
                        E("p(1,3)"): p1[2],
                        E("p(2,3)"): p2[2],
                        E("s"): s,
                        E("z"): z,
                    }
                    print(input_vals)
                    print(e_surface)
                    for j in range(0, self.L):
                        for i in range(0, 3):
                            e_surface = e_surface.replace(E(f"k({j},{i + 1})"), k[j][i])
                    for i in range(0, 3):
                        e_surface = e_surface.replace(E(f"p(1,{i + 1})"), p1[i])
                    for i in range(0, 3):
                        e_surface = e_surface.replace(E(f"p(2,{i + 1})"), p2[i])
                    e_surface = e_surface.replace(E("s"), s)
                    e_surface = e_surface.replace(E("z"), z)

                    print("hereee")
                    print(e_surface)
                    t_sol = e_surface.nsolve(E("t"), 1.0)
                    return t_sol
                else:
                    return 1

            return 1

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

        t_sol = self.set_t_value(k, p1, p2, z)

        param_list.append(t_sol)

        return param_list

    def eval(self, k, p1, p2, z):
        self.set_t_value(k, p1, p2, z)
        param_list = self.param_builder(k, p1, p2, z)
        print(param_list)
        return self.evaluator.evaluate(param_list)
