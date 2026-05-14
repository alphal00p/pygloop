import json
import os
import re
from collections import Counter
from copy import deepcopy
from itertools import product

import numpy as np
import pydot

from gammaloop import (  # isort: skip # type: ignore # noqa: F401
    GammaLoopAPI,
    LogLevel,
    evaluate_graph_overall_factor,
)
from sympy.assumptions.assume import false

try:
    from gammaloop import git_version  # isort: skip # type: ignore # noqa: F401
except ImportError:
    try:
        from gammaloop import __version__ as git_version  # isort: skip # type: ignore # noqa: F401
    except ImportError:
        git_version = "unknown"
from numpy.ma.core import make_mask_descr
from symbolica import AtomType, E, Expression, Replacement, S  # pyright: ignore
from symbolica.community.idenso import (  # pyright: ignore
    simplify_color,
    simplify_gamma,
    simplify_metrics,
)

from processes.dy.dy_evaluators import substitute_process_couplings
from processes.dy.dy_graph_utils import (
    _base_node,
    _node_key,
    _parse_port,
    _strip_quotes,
    boundary_edges,
    change_routing,
    get_LR_components,
    get_simple_cycles,
)
from processes.dy.dy_integrated_uv_ct import (
    construct_integrated_counter_term as construct_integrated_uv_counter_term,
)
from utils.cff import CFFStructure
from utils.utils import (
    EVALUATORS_FOLDER,
    PYGLOOP_FOLDER,
    ParamBuilder,
    PygloopEvaluator,
    expr_to_string,
    logger,
)

pjoin = os.path.join

gl_log_level = LogLevel.Off

debug = False


def Es(expr: str) -> Expression:
    return E(expr.replace('"', ""), default_namespace="gammalooprs")


def heaviside_theta(x):
    if x > 0:
        return 1
    else:
        return 0


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


def _strip_namespaces_structurally(expr: Expression) -> Expression:
    args__ = S("args__")
    atom_repls = []
    seen = set()

    for sym in expr.get_all_symbols():
        full = sym.get_name()
        if "::" not in full:
            continue
        if full.startswith("symbolica::"):
            continue
        if full in seen:
            continue
        seen.add(full)

        short = full.rsplit("::", 1)[-1]
        old = S(full)
        new = S(short)

        # Rename function heads, e.g. spenso::mink(...) -> mink(...)
        expr = expr.replace(
            old(args__),
            new(args__),
            allow_new_wildcards_on_rhs=True,
        )

        # Rename bare symbols/constants, e.g. UFO::MT -> MT
        atom_repls.append(Replacement(old, new))

    if atom_repls:
        expr = expr.replace_multiple(atom_repls)

    return expr


def _raw_graph_numerator(graph) -> Expression:
    numerator = E("1")
    for node in graph.get_nodes():
        if node.get_name() not in ["edge", "node"]:
            node_numerator = node.get("num")
            if node_numerator:
                numerator *= Es(node_numerator)
    for edge in graph.get_edges():
        edge_numerator = edge.get("num")
        if edge_numerator:
            numerator *= Es(edge_numerator)
    return numerator


def _rewrite_repeated_non_cut_edge_momentum_powers(
    numerator: Expression, graph, choice_offset: int = 0
) -> Expression:
    edge_by_id = {
        _strip_quotes(str(e.get_attributes()["id"])): e for e in graph.get_edges()
    }
    non_cut_ids = {
        eid
        for eid, e in edge_by_id.items()
        if all(
            _strip_quotes(str(e.get_attributes().get(key, "0"))) in ["0", "0.0"]
            for key in ["is_cut", "is_cut_DY"]
        )
    }
    incident = {}
    for edge in graph.get_edges():
        incident.setdefault(_base_node(edge.get_source()), []).append(edge)
        incident.setdefault(_base_node(edge.get_destination()), []).append(edge)

    def edge_id(edge) -> str:
        return _strip_quotes(str(edge.get_attributes()["id"]))

    def q(edge: str, slot: Expression) -> Expression:
        return Es(f"Q({edge},{slot.to_canonical_string()})")

    def rules_for(edge: str):
        out = []
        target = edge_by_id[edge]
        for node in [
            _base_node(target.get_source()),
            _base_node(target.get_destination()),
        ]:
            if node.startswith("ext"):
                continue
            others = [e for e in incident[node] if edge_id(e) != edge]
            if not others:
                continue

            def rhs(slot, node=node, others=others, target=target):
                repl = E("0")
                for other in others:
                    sign = 1 if _base_node(other.get_source()) == node else -1
                    repl += sign * q(edge_id(other), slot)
                if _base_node(target.get_source()) == node:
                    repl = -repl
                return repl

            out.append(rhs)
        if not out:
            raise ValueError(f"No momentum-conservation rule for edge {edge}.")
        shift = choice_offset % len(out)
        return out[shift:] + out[:shift]

    def terms(expr):
        expr = expr.expand()
        return list(expr) if bool(expr.is_type(AtomType.Add)) else [expr]

    def factors(term):
        return list(term) if bool(term.is_type(AtomType.Mul)) else [term]

    def match_q(expr):
        for pattern, edge_key, slot_key in [
            (Es("Q(edge_,slot_)"), S("gammalooprs::edge_"), S("gammalooprs::slot_")),
            (E("Q(edge_,slot_)"), S("edge_"), S("slot_")),
        ]:
            match = next(iter(expr.match(pattern)), None)
            if match is not None:
                return match[edge_key].to_canonical_string(), match[slot_key]
        return None

    def q_power(factor):
        if bool(factor.is_type(AtomType.Pow)):
            base, power = list(factor)
            match = match_q(base)
            if match is None:
                return None
            power = int(power.to_canonical_string())
            if power <= 0:
                return None
            edge, slot = match
            return edge, slot, power
        match = match_q(factor)
        if match is None:
            return None
        edge, slot = match
        return edge, slot, 1

    def first_repeat(expr):
        for term in terms(expr):
            term_factors = factors(term)
            infos = []
            counts = Counter()
            for i, factor in enumerate(term_factors):
                info = q_power(factor)
                if info is None:
                    continue
                edge, slot, power = info
                infos.append((i, edge, slot, power))
                counts[edge] += power
            repeated = {
                edge
                for edge, count in counts.items()
                if count > 1 and edge in non_cut_ids
            }
            for info in infos:
                if info[1] in repeated:
                    return term, term_factors, info
        return None

    def replace_factor(expr, term, term_factors, info, replacement):
        factor_index, edge, slot, power = info
        new_factor = replacement
        if power > 1:
            new_factor *= q(edge, slot) ** E(str(power - 1))
        out = E("1")
        for i, factor in enumerate(term_factors):
            out *= new_factor if i == factor_index else factor
        return (expr - term + out).expand()

    def solve(expr, depth, seen):
        if depth > 512:
            raise ValueError("Could not remove repeated non-cut edge momenta.")
        repeat = first_repeat(expr)
        if repeat is None:
            return expr
        term, term_factors, info = repeat
        edge = info[1]
        for rule in rules_for(edge):
            candidate = replace_factor(expr, term, term_factors, info, rule(info[2]))
            key = candidate.to_canonical_string()
            if key in seen:
                continue
            seen.add(key)
            try:
                return solve(candidate, depth + 1, seen)
            except ValueError:
                continue
        raise ValueError(f"Could not remove repeated non-cut edge momentum {edge}.")

    numerator = numerator.expand()
    return solve(numerator, 0, {numerator.to_canonical_string()})


class RoutedIntegrand(object):
    def __init__(
        self,
        integrand,
        cut_graph,
        replacements,
        emr_integrand,
        type,
        ir_limit,
        t_derivative=False,
    ):
        self.emr_integrand = emr_integrand
        self.integrand = integrand
        self.cut_graph = cut_graph
        self.replacements = replacements
        self.approximation_type = type
        self.ir_limit = ir_limit
        self.t_derivative = t_derivative


# This class is responsible for generating the CFF representation of the cut graph


class EMRIntegrandConstructor(object):
    def __init__(self, params, name, L):
        self.L = L
        self.params = params
        self.name = name
        self.gl_worker = GammaLoopAPI(
            pjoin(PYGLOOP_FOLDER, "outputs", "gammaloop_states", self.name),
            # log_file_name=self.name,
            # log_level=gl_log_level,
        )
        # GAMMALOOP_STATE_FOLDER
        self.gl_worker.run("import model sm-default.json")

    # Get the numerator of the graph

    def get_numerator(self, graph) -> Expression:
        symmetry_factor = Es(graph.get("overall_factor_evaluated"))

        num = _raw_graph_numerator(graph)
        num = _rewrite_repeated_non_cut_edge_momentum_powers(num, graph)

        print("here" * 10)

        num = num

        # print("now here" * 10)
        simplified = simplify_metrics(simplify_gamma(simplify_color(num))).expand()
        # print(simplified)
        res = _strip_namespaces_structurally(simplified)
        res = substitute_process_couplings(res, self.name, self.L).expand()

        # res = E(_canonicalize_symbolica_display_string(str(simplified)))
        out = res.replace(
            E("Q(y_,mink(4,x_))") * E("Q(z_,mink(4,x_))"), E("sp(y_,z_)"), repeat=True
        )
        out = out.replace(
            E("Qp(y_,mink(4,x_))") * E("Q(z_,mink(4,x_))"),
            E("spp(qp(y_),z_)"),
            repeat=True,
        )
        out = out.replace(
            E("Qp(y_,mink(4,x_))") * E("Qp(z_,mink(4,x_))"),
            E("spp(qp(y_),qp(z_))"),
            repeat=True,
        )

        # print(out)

        # out = out.replace(E("sp(0,7)"), E("0"))

        return symmetry_factor * out

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
            e.get_attributes().pop("lmb_id", None)

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
            if e_atts.get("is_cut", 0) != 0:
                e_atts["is_cut_DY"] = e_atts["is_cut"]
            e_atts.pop("is_cut", None)
            e_atts.pop("source", None)
            e_atts.pop("num", None)
            e_atts.pop("sink", None)
            e_atts.pop("is_dummy", None)
            e_atts.pop("dir_in_cycle", None)

    # The following function takes a cut graph and gives out the two amplitude graphs which, glued together
    # give back the origi   q   nal cut graph. In order to do so it has to check if edges of the cut_graph are contained
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
                    new_graphs[i].del_edge(src, dest, int(e_atts["id"]))
                elif src_key in comps[i] and dest_key not in comps[i]:
                    new_graphs[i].del_edge(src, dest, int(e_atts["id"]))
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
                    new_graphs[i].del_edge(src, dest, int(e_atts["id"]))
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
                    new_graphs[i].del_edge(src, dest, int(e_atts["id"]))
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
                    new_graphs[i].del_edge(src, dest, int(e_atts["id"]))
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

    def get_cff(
        self, cut_graph, s_split_graphs, s_channel_edges, numerator, get_residues=False
    ):

        # print("numerator here" * 5)
        # print(numerator)
        numerator = numerator.replace(
            E("spp(qp(x_),qp(y_))"),
            E("sigma(x_)*sigma(y_)*En(x_)*En(y_)-sp3(q(x_),q(y_))"),
        )
        numerator = numerator.replace(
            E("spp(qp(x_),y_)"), E("sigma(x_)*sigma(y_)*En(x_)*En(y_)-sp3(q(x_),q(y_))")
        )
        numerator = numerator.replace(
            E("sp(x_,y_)"), E("sigma(x_)*sigma(y_)*En(x_)*En(y_)-sp3(q(x_),q(y_))")
        )
        #
        # numerator = numerator.replace(
        #    E("spp(qp(x_),qp(y_))"),
        #    E("sigma(x_)*sigma(y_)*E(x_)*E(y_)-sp3(q(x_),q(y_))"),
        # )
        # numerator = numerator.replace(
        #    E("spp(qp(x_),y_)"), E("sigma(x_)*sigma(y_)*E(x_)*E(y_)-sp3(q(x_),q(y_))")
        # )
        # numerator = numerator.replace(
        #    E("sp(x_,y_)"), E("sigma(x_)*sigma(y_)*E(x_)*E(y_)-sp3(q(x_),q(y_))")
        # )

        numerator = numerator.replace(E("sigma(1000)"), E("1"))
        numerator = numerator.replace(E("sp3(q(1000), x___)"), E("0"))
        numerator = numerator.replace(E("sp3(x___, q(1000))"), E("0"))
        numerator = numerator.replace(E("En(1000)"), E("1"))

        # numerator = numerator.replace(E("sigma(6)"), E("-sigma(6)"))

        # if len(cut_graph.partition[1]) == 2:
        #    numerator = numerator.replace(E("E(7)"), E("-E(7)"))

        # print(numerator)

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

        e_surfaces = set()

        # previous_cff = numerator

        previous_cff = numerator

        edges_to_reverse = []

        # print("graphs")
        # for n_graph, g in enumerate(split_graphs_gt2_non_ext):
        #    print("n_graph ", n_graph)
        #    g_rep = g.replacements
        #    for i in range(0, len(g.graph.get_edges())):
        #        print(g_rep[i][1])

        # print("emr cutgraph")
        # for e in cut_g_edges:
        #    print(e)
        # print("---------")
        edges_to_reverse = []
        for n_graph, g in enumerate(split_graphs_gt2_non_ext):
            cff_g = self.get_CFF(g.graph, [], [])
            new_cff = E("0")
            g_rep = g.replacements

            for cffterm in cff_g.expressions:
                cff_term = previous_cff * cffterm.expression

                # print("çççççççç")
                # print("n_graph:", n_graph)
                # print("len:", len(cffterm.orientation))
                for o, i in zip(cffterm.orientation, range(len(cffterm.orientation))):
                    id_in_original_graph = g_rep[i][1]
                    this_e_atts = cut_g_edges[
                        int(id_in_original_graph)
                    ].get_attributes()
                    if o.is_reversed():
                        cff_term = cff_term.replace(
                            E(f"sigma({id_in_original_graph})"), E("-1")
                        )
                    if o.is_default():
                        cff_term = cff_term.replace(
                            E(f"sigma({id_in_original_graph})"), E("1")
                        )
                    if this_e_atts.get("is_cut_DY", 0) == -1:
                        # print("***")
                        # print(this_e_atts["id"])
                        # print(id_in_original_graph)
                        edges_to_reverse.append(id_in_original_graph)
                # print(edges_to_reverse)
                for etas in cff_g.e_surfaces:
                    eta = etas.expression
                    for rep in g.replacements:
                        eta = eta.replace(E(f"pygloop::E({rep[0]})"), E(f"E({rep[1]})"))

                    # tt~change: fix eta here with minus sign
                    cff_term = cff_term.replace(E(f"pygloop::η({etas.id})"), -eta)

                    if get_residues:
                        residue_eta = deepcopy(eta)
                        for id in set(edges_to_reverse):
                            residue_eta = residue_eta.replace(
                                E(f"E({id})"), -E(f"E({id})")
                            )
                            residue_eta = residue_eta.replace(
                                E(f"En({id})"), -E(f"En({id})")
                            )
                        e_surfaces.add(residue_eta)

                # print(f"cff term for graph {n_graph} before: ", cff_term)

                # for id in set(edges_to_reverse):
                #    cff_term = cff_term.replace(E(f"E({id})"), -E(f"E({id})"))

                # print(f"cff term for graph {n_graph} after: ", cff_term)
                new_cff += cff_term

            previous_cff = new_cff

        for id in set(edges_to_reverse):
            previous_cff = previous_cff.replace(E(f"E({id})"), E(f"-E({id})"))
            previous_cff = previous_cff.replace(E(f"En({id})"), E(f"-En({id})"))

        # print("cff fresh out of the cff algorithm")
        # print(edges_to_reverse)

        # print(previous_cff)

        # Multiplies by non-partial-fractioned s-channel propagators and sets the s-channel particle's
        # energy in terms of other cut particles by energy conservation. The logic is weak for many s-channel
        # propagators since you can have a subgraph sandwiched between two s-channel propagators (TODO: FIX).
        # Also sets the sign of the propagator sigma(i) by one (i.e. according to original orientation)

        popping_edges = deepcopy(s_channel_edges)

        def _is_exact_zero(expr):
            return str(expr.expand()) == "0"

        while len(popping_edges) > 0:
            current_s_edge = popping_edges.pop()
            s_edge_atts = current_s_edge.get_attributes()
            candidates = []
            for g in s_split_graphs:
                for e in g.graph.get_edges():
                    e_atts = e.get_attributes()
                    e_src = e.get_source()
                    if s_edge_atts["name"] != e_atts["name"]:
                        continue

                    denom = E("0")
                    denom_num = E("0")
                    for ep in g.graph.get_edges():
                        ep_src = ep.get_source()
                        ep_dest = ep.get_destination()
                        ep_atts = ep.get_attributes()
                        if ep_src.startswith("ext") and ep != e:
                            cut_sign = ep_atts.get("is_cut_DY", 0)
                            denom += cut_sign * E(
                                f"E({g.replacements[ep_atts['id']][1]})"
                            )
                            denom_num += cut_sign * E(
                                f"En({g.replacements[ep_atts['id']][1]})"
                            )
                        if ep_dest.startswith("ext") and ep != e:
                            cut_sign = ep_atts.get("is_cut_DY", 0)
                            denom -= cut_sign * E(
                                f"E({g.replacements[ep_atts['id']][1]})"
                            )
                            denom_num -= cut_sign * E(
                                f"En({g.replacements[ep_atts['id']][1]})"
                            )
                    candidates.append((g, e_atts, e_src, denom, denom_num))

            selected_candidate = None
            for candidate in candidates:
                if not _is_exact_zero(candidate[3]):
                    selected_candidate = candidate
                    break

            if selected_candidate is None:
                denom_report = [str(candidate[3].expand()) for candidate in candidates]
                raise ValueError(
                    "Could not reconstruct nonzero s-channel denominator for "
                    f"{s_edge_atts.get('name')}; candidates={denom_report}"
                )

            g, e_atts, e_src, denom, denom_num = selected_candidate

            # NEW: for cut s-channel propagators
            if e_atts.get("is_cut_DY", None) is not None:
                previous_cff = previous_cff / denom
            else:
                previous_cff = previous_cff / denom**2

            sign = 1 if e_src.startswith("ext") else -1

            if e_atts.get("is_cut_DY") is not None:
                sign = e_atts.get("is_cut_DY") * sign

            previous_cff = previous_cff.replace(
                E(f"E({g.replacements[e_atts['id']][1]})"), -sign * denom
            )
            previous_cff = previous_cff.replace(
                E(f"En({g.replacements[e_atts['id']][1]})"),
                -sign * denom_num,
            )

            if get_residues:
                e_surfaces = {
                    e_surf.replace(
                        E(f"E({g.replacements[e_atts['id']][1]})"),
                        -sign * denom,
                    )
                    for e_surf in e_surfaces
                }

            previous_cff = previous_cff.replace(
                E(f"sigma({g.replacements[e_atts['id']][1]})"), E("1")
            )

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

        total_cff = previous_cff * energies

        if get_residues:
            delta = E("δ")
            residues = []
            for eta in e_surfaces:
                eta_expanded = eta.expand()
                energies = (
                    list(eta_expanded) if eta_expanded.is_type(AtomType.Add) else [eta]
                )
                eN = eta.replace(E("E(x___)"), E("1"))
                if len(energies) > 1 and eN < len(energies):
                    pivot = energies[0]
                    # if pivot.match(E("-E(x___)")) is not None:
                    if pivot.format_plain().lstrip().startswith("-"):
                        patt = pivot.replace(E("-E(x___)"), E("E(x___)"))
                        repl = sum(en for en in energies[1:]) - delta
                    else:
                        patt = pivot
                        repl = -sum(en for en in energies[1:]) + delta
                    res_i = deepcopy(total_cff)
                    res_i = (
                        res_i
                        .replace(patt, repl)
                        .series(delta, 0, -1)
                        .to_expression()
                        .replace(delta, E("1"))
                    )
                    residues.append((eta, res_i))
                elif eN > len(energies):
                    raise ValueError("really weird stuff happening with e surfaces")
            return residues

        total_cff = total_cff.replace(E("Q(x_,0)"), E("E(x_)"))

        # total_cff = total_cff.replace(E("E(6)"), E("-E(6)"))

        # print(total_cff)

        return total_cff

    # Use all the previous functions to get the cff for the cut graph

    def _normalise_global_i(self, num: Expression):
        num = E(str(num.expand()))
        if "𝑖" not in num.format_plain():
            return num

        candidate = E(str((num / E("1i")).expand()))
        if "𝑖" not in candidate.format_plain():
            return candidate

        return num

    def get_integrand(self, cut_graph: routed_cut_graph, get_residues=False):

        # Derives numerator, eliminates useless labels, get left and right graphs and further
        # splits them if they have s-channel propagators.

        print("got to num")

        num = self.get_numerator(cut_graph.graph)

        # print(num.replace(E("sp(x_,y_)"),E("1")))

        print("and beyond num")

        # print("NUM before contraction:   " , num)

        num = num.replace(E("Q(x_,mink(y_,z_))^2"), E("sp(x_,x_)"))  # * E("1i")

        self.normalise_graph(cut_graph.graph)

        graph_L, graph_R = self.get_LR_graphs(cut_graph)

        s_split_graphs_L, s_channel_edges_L = self.split_s_channels(graph_L)
        s_split_graphs_R, s_channel_edges_R = self.split_s_channels(graph_R)

        ## DEBUG: set numerator to 1
        # num = E("1")
        # print("NUMERATORRRRRRRR")
        # print(num)

        print("got to cff construction")

        cut_graph_cff = self.get_cff(
            cut_graph,
            s_split_graphs_L + s_split_graphs_R,
            s_channel_edges_L + s_channel_edges_R,
            num,
            get_residues,
        )

        print("and beyond cff construction")

        return cut_graph_cff


class UltraVioletSubtraction(object):
    def __init__(
        self,
        emr_integrand,
        cut_graph,
        L,
        emr_processor=None,
        integrated_cut_graph=None,
        disable_integrated_uv_cts=True,
    ):
        self.cut_graph = cut_graph
        self.emr_integrand = emr_integrand
        self.sp3D = S("sp3D", is_linear=True, is_symmetric=True)
        self.L = L
        self.emr_processor = emr_processor
        self.integrated_cut_graph = integrated_cut_graph
        self.disable_integrated_uv_cts = bool(disable_integrated_uv_cts)

    # Focuses on cycles, and not unions of cycles. Specialised to NLO
    def enumerate_spinneys(self):

        cycles = get_simple_cycles(self.cut_graph.graph)

        cut_edges = set(self.cut_graph.initial_cut).union(set(self.cut_graph.final_cut))
        divergent_cycles = []

        for cycle in cycles:
            cut_cycle_edges = set(cycle).intersection(cut_edges)
            if len(cut_cycle_edges) == 0:  ## FIX: SHOULD BE == 0
                dod = 0
                visited_nodes = set()

                for e in cycle:
                    e_atts = e.get_attributes()
                    dod += int(_strip_quotes(str(e_atts["dod"])))
                    visited_nodes.add(_node_key(e.get_source()))
                    visited_nodes.add(_node_key(e.get_destination()))

                for v in self.cut_graph.graph.get_nodes():
                    v_atts = v.get_attributes()
                    if _node_key(v.get_name()) in visited_nodes:
                        dod += int(_strip_quotes(str(v_atts["dod"])))

                if dod + 4 >= 0:
                    divergent_cycles.append([cycle, dod + 4])

        return divergent_cycles

    def replace_energies(self, integrand, graph):

        for e in graph.get_edges():
            e_atts = e.get_attributes()
            eid_raw = e_atts["id"]
            eid = _strip_quotes(eid_raw) if isinstance(eid_raw, str) else eid_raw
            target = E(f"E({eid})")
            target_num = E(f"En({eid})")
            particle = _strip_quotes(str(e_atts["particle"]))
            replacement = (
                self.sp3D(E(f"q({eid})"), E(f"q({eid})")) + E(f"m({eid})") ** 2
            ) ** E("1/2")
            integrand = integrand.replace(target, replacement)
            integrand = integrand.replace(target_num, replacement)

        return integrand

    def route_integrand(self, integrand, graph):

        for e in graph.get_edges():
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

    # For now we construct the counter-term associated to a cycle. This ignores loop-induced.
    def construct_counter_term(self, cycle, dod):

        lam = S("λ", is_scalar=True)
        mUV = E("mUV")

        # uv_loop_momentum = next(iter(cycle))
        uv_loop_momentum = min(
            cycle, key=lambda e: int(_strip_quotes(str(e.get_attributes()["id"])))
        )
        lmb = [uv_loop_momentum] + self.cut_graph.final_cut[:-1]

        lmb_id = []
        for e in lmb:
            e_atts = e.get_attributes()
            lmb_id.append(e_atts["id"])

        uv_graph = change_routing(deepcopy(self.cut_graph.graph), lmb_id)

        routed_integrand = self.replace_energies(self.emr_integrand, uv_graph)
        routed_integrand = self.route_integrand(routed_integrand, uv_graph)

        parametrised_integrand = routed_integrand.replace(E("k(0)"), E("k(0)") / lam)

        for e in cycle:
            e_atts = e.get_attributes()
            # This mass substitution also acts on the numerator, which might be dangerous
            # On a second thought this might be needed to reproduce the 4D version of the UV ct
            parametrised_integrand = parametrised_integrand.replace(
                E(f"m({e_atts['id']})") ** 2,
                1 / lam**2 * mUV**2 + (E(f"m({e_atts['id']})") ** 2 - mUV**2),
            )

        for e in self.cut_graph.graph.get_edges():
            e_atts = e.get_attributes()
            e_id = e_atts["id"]
            e_particle = _strip_quotes(str(e_atts["particle"]))
            if e_particle in ["d", "d~", "g", "ghG", "ghG~"]:
                mass = E("0")
            else:
                mass = E(f"m({e_particle})")
            parametrised_integrand = parametrised_integrand.replace(
                E(f"m({e_atts['id']})"), mass
            )

        print("DOD" * 10)
        print(dod)
        dod = 0
        expanded_integrand = (
            (1 / lam**3 * parametrised_integrand)
            .series(lam, 0, dod)
            .to_expression()
            .replace(lam, E("1"))
        )

        # Go back to previous basis. Use the routed graph edge, not the boundary-edge
        # object in final_cut, because the latter can miss routing attributes.
        edge_by_id = {
            _strip_quotes(str(edge.get_attributes()["id"])): edge
            for edge in self.cut_graph.graph.get_edges()
        }
        basis_replacements = []
        for j, e in enumerate(lmb):
            edge_id = _strip_quotes(str(e.get_attributes()["id"]))
            basis_edge = edge_by_id.get(edge_id)
            if basis_edge is None:
                raise ValueError(f"UV basis edge {edge_id} not found in cut graph")
            e_atts = basis_edge.get_attributes()
            routing_items = E("0")
            for i in range(self.L + 1):
                key = f"routing_k{i}"
                if key in e_atts:
                    routing_items += E(f"{e_atts[key]}*k[{i}]")
            for i in range(0, 2):
                key = f"routing_p{i + 1}"
                if key in e_atts:
                    routing_items += E(f"{e_atts[key]}*p[{i + 1}]")
            basis_replacements.append((E(f"k({j})"), routing_items))

        tmp_symbols = [E(f"__tmp_uv_{j}") for j in range(len(basis_replacements))]
        for (pattern, _), tmp_symbol in zip(basis_replacements, tmp_symbols):
            expanded_integrand = expanded_integrand.replace(pattern, tmp_symbol)
        for tmp_symbol, (_, replacement) in zip(tmp_symbols, basis_replacements):
            expanded_integrand = expanded_integrand.replace(tmp_symbol, replacement)

        return -expanded_integrand

    def construct_integrated_counter_term(self, cycle, dod):
        if self.disable_integrated_uv_cts:
            return None

        return construct_integrated_uv_counter_term(
            self,
            cycle,
            dod,
            routed_cut_graph,
            RoutedIntegrand,
        )

    def construct_uv_counter_terms(self):
        spinneys = self.enumerate_spinneys()
        counterms = []

        # check copies and deepcopies
        for cycle in spinneys:
            uv_ct = self.construct_counter_term(cycle[0], cycle[1])
            routed_uv_ct = RoutedIntegrand(
                uv_ct, self.cut_graph, [], self.emr_integrand, "uv", "uv"
            )
            counterms.append(routed_uv_ct)
            if not self.disable_integrated_uv_cts:
                integrated_uv_ct = self.construct_integrated_counter_term(
                    cycle[0], cycle[1]
                )
                if integrated_uv_ct is not None:
                    counterms.append(integrated_uv_ct)

        return counterms


class ThresholdSubtractor(object):
    def __init__(self, routed_cut_graph, params, name, L, theta_support=True):
        self.routed_cut_graph = routed_cut_graph
        self.emr_processor = EMRIntegrandConstructor(params, name, L)
        self.sp3D = S("sp3D", is_linear=True, is_symmetric=True)
        self.residues = self.emr_processor.get_integrand(routed_cut_graph, True)
        self.L = L
        self.theta_support = theta_support
        self.name = name

    # Here we can do something that is sort of process-specific...
    def filter_e_surfaces(self):
        filtered_e_surf = []

        for esurf, residue in self.residues:
            esurf = esurf.expand()
            if not esurf.is_type(AtomType.Add):
                raise ValueError(
                    "problem: single energy e-surface in threshold approximator"
                )
            energy_ids = list(esurf)

            energy_ids = [
                en.replace(E("-E(x___)"), E("x___")).replace(E("E(x___)"), E("x___"))
                for en in energy_ids
            ]

            positive_ids = []
            negative_ids = []
            for e in self.routed_cut_graph.graph.get_edges():
                e_atts = e.get_attributes()
                if (
                    int(e_atts["id"]) in energy_ids
                    and e_atts.get("is_cut_DY") is not None
                ):
                    positive_ids.append(e_atts["id"])
                elif (
                    int(e_atts["id"]) in energy_ids and e_atts.get("is_cut_DY") is None
                ):
                    negative_ids.append(e_atts["id"])

            external_momentum = [0, 0]
            outgoing_mass = 0
            incoming_mass = 0
            for e in self.routed_cut_graph.graph.get_edges():
                e_atts = e.get_attributes()
                particle = _strip_quotes(str(e_atts["particle"]))
                if e_atts["id"] in positive_ids:
                    external_momentum[0] += (
                        int(e_atts["routing_p1"]) * e_atts["is_cut_DY"]
                    )
                    external_momentum[1] += (
                        int(e_atts["routing_p2"]) * e_atts["is_cut_DY"]
                    )
                    if particle in ["t", "t~"]:
                        outgoing_mass += 1
                if e_atts["id"] in positive_ids:
                    if particle in ["t", "t~"]:
                        incoming_mass += 1

            if (
                external_momentum[0] * external_momentum[1] == 1
                and external_momentum[0] == external_momentum[1]
                and len(negative_ids) > 1
            ):
                if outgoing_mass == 1 and incoming_mass == 1:
                    continue
                filtered_e_surf.append((negative_ids, esurf, residue))

        return filtered_e_surf

    def replace_energies(self, integrand, cut_graph):

        for e in cut_graph.get_edges():
            e_atts = e.get_attributes()
            eid_raw = e_atts["id"]
            eid = _strip_quotes(eid_raw) if isinstance(eid_raw, str) else eid_raw
            target = E(f"E({eid})")
            target_num = E(f"En({eid})")
            particle = _strip_quotes(str(e_atts["particle"]))
            if particle not in ["d", "d~", "g", "ghG", "ghG~"]:
                replacement = (
                    self.sp3D(E(f"q({eid})"), E(f"q({eid})")) + E(f"m({particle})") ** 2
                ) ** E("1/2")
            else:
                # new stuff
                k_keys = [f"routing_k{i}" for i in range(0, self.L)]
                zero_routing = all(e_atts[kk] == "0" for kk in k_keys)
                mass = E("0")
                if (
                    zero_routing
                    and e_atts["routing_p1"] == "0"
                    and e_atts["routing_p2"] != "0"
                ):
                    mass = E("0")
                if (
                    zero_routing
                    and e_atts["routing_p1"] != "0"
                    and e_atts["routing_p2"] == "0"
                ):
                    mass = E("0")

                replacement = (self.sp3D(E(f"q({eid})"), E(f"q({eid})")) + mass) ** E(
                    "1/2"
                )

            integrand = integrand.replace(target, replacement)
            integrand = integrand.replace(target_num, replacement)

        return integrand

    # Substituted the emr momenta by their linear decomposition in terms of
    # loop variables and external momenta (p1 and p2)

    def route_integrand(self, integrand, cut_graph):

        for e in cut_graph.get_edges():
            e_atts = e.get_attributes()
            routing_items = E("0")
            for i in range(self.L):
                key = f"routing_k{i}"
                if key in e_atts:
                    routing_items += E(f"{e_atts[key]}*k[{i}]")
            for i in range(2):
                key = f"routing_p{i + 1}"
                if key in e_atts:
                    routing_items += E(f"{e_atts[key]}*p[{i + 1}]")
            integrand = integrand.replace(E(f"q({e_atts['id']})"), routing_items)

        integrand = integrand.replace(
            E("sp3(x___,y___)"), self.sp3D(S("x___"), S("y___"))
        )
        return integrand

    def construct_threshold_counter_term(
        self, emr_residue_integrand, threshold_graph, threshold_ids
    ):

        collinear_momentum = E("0")

        for e in threshold_graph.graph.get_edges():
            e_atts = e.get_attributes()
            if e_atts["id"] == threshold_ids[0]:
                collinear_momentum = (
                    collinear_momentum
                    + E(f"{e_atts['routing_p1']}*p(1)")
                    + E(f"{e_atts['routing_p2']}*p(2)")
                )
                k_sign = E(f"{e_atts['routing_k0']}*k(0)")

        # for ttbar: careful of k0 sign

        lmb_ids = threshold_ids[:-1] + [
            e.get_attributes()["id"] for e in threshold_graph.final_cut[:-1]
        ]

        threshold_graph_routed = change_routing(
            deepcopy(threshold_graph.graph), lmb_ids
        )

        sqrts = S("s") ** E("1/2")

        threshold_integrand = self.replace_energies(
            emr_residue_integrand, threshold_graph_routed
        )
        threshold_integrand = self.route_integrand(
            threshold_integrand, threshold_graph_routed
        )  # .replace(r, r.exp())

        shifts = []
        masses = []

        for id in threshold_ids:
            shift = E("0")
            for e in threshold_graph_routed.get_edges():
                e_atts = e.get_attributes()
                if e_atts["id"] == id:
                    routing_items = E("0")
                    for i in range(1, self.L + 1):
                        key = f"routing_k{i}"
                        if key in e_atts:
                            routing_items += E(f"{e_atts[key]}*k[{i}]")
                    for i in range(2):
                        key = f"routing_p{i + 1}"
                        if key in e_atts:
                            routing_items += E(f"{e_atts[key]}*p[{i + 1}]")
                    shift = routing_items
                    mass = E("0")
                    e_particle = _strip_quotes(str(e_atts["particle"]))
                    if e_particle in ["d", "d~", "g", "ghG", "ghG~"]:
                        mass = E("0")
                    else:
                        mass = E(f"m({e_particle})")
            shifts.append(shift)
            masses.append(mass)

        r = S("r", is_scalar=True)
        rexp = S("rexp", is_scalar=True)
        khat = S("khat")
        patt = E("k(0)")
        rep = rexp * khat

        # dot products
        # kp1 = self.sp3D(E("k(0)"), shifts[0])
        # kp2 = self.sp3D(E("k(0)"), shifts[1])
        # kk = self.sp3D(E("k(0)"), E("k(0)"))
        kp1 = self.sp3D(khat, shifts[0])
        kp2 = self.sp3D(khat, shifts[1])
        kk = self.sp3D(khat, khat)
        p1p1 = self.sp3D(shifts[0], shifts[0])
        p2p2 = self.sp3D(shifts[1], shifts[1])
        m1sq = masses[0] ** 2
        m2sq = masses[1] ** 2

        A = 4 * (sqrts**2) * kk - kp1**2 + 2 * kp1 * kp2 - kp2**2

        B = (
            -2 * m1sq * kp1
            + 2 * m2sq * kp1
            + 2 * (sqrts**2) * kp1
            + 2 * m1sq * kp2
            - 2 * m2sq * kp2
            + 2 * (sqrts**2) * kp2
            - 2 * kp1 * p1p1
            + 2 * kp2 * p1p1
            + 2 * kp1 * p2p2
            - 2 * kp2 * p2p2
        )

        C = (
            -(m1sq**2)
            + 2 * m1sq * m2sq
            - m2sq**2
            + 2 * m1sq * (sqrts**2)
            + 2 * m2sq * (sqrts**2)
            - (sqrts**4)
            - 2 * m1sq * p1p1
            + 2 * m2sq * p1p1
            + 2 * (sqrts**2) * p1p1
            - (p1p1**2)
            + 2 * m1sq * p2p2
            - 2 * m2sq * p2p2
            + 2 * (sqrts**2) * p2p2
            + 2 * p1p1 * p2p2
            - (p2p2**2)
        )

        rstar = ((-B + (B**2 - 4 * A * C) ** E("1/2")) / (2 * A)).log()

        derivative = (2 * (2 * rstar).exp() * kk + (rstar).exp() * kp1) / (
            2.0
            * ((2 * rstar).exp() * kk + (rstar).exp() * kp1 + m1sq + p1p1) ** E("1/2")
        ) + (2 * (2 * rstar).exp() * kk + (rstar).exp() * kp2) / (
            2.0
            * ((2 * rstar).exp() * kk + (rstar).exp() * kp2 + m2sq + p2p2) ** E("1/2")
        )

        threshold_integrand = threshold_integrand.replace(patt, rep)

        threshold_integrand = (
            (threshold_integrand.replace(rexp, r.exp()).replace(r, rstar))
            / (r - rstar)
            / derivative
        )

        inv_knorm = S("knorm", is_scalar=True)
        threshold_integrand = (
            threshold_integrand
            .replace(rexp, r.exp())
            .replace(r, (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")).log())
            .replace(khat, E("k(0)") * inv_knorm)
            .replace(inv_knorm, 1 / (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")))
        )

        threshold_integrand = threshold_integrand.replace(
            E("s"), 4 * (E("p(1,1)") ** 2 + E("p(1,2)") ** 2 + E("p(1,3)") ** 2)
        )

        rep_r = rexp * khat

        if self.theta_support == True:
            repl_x = self.sp3D(rep_r, collinear_momentum) / self.sp3D(
                collinear_momentum, collinear_momentum
            )

            x = S("x", is_scalar=True, is_positive=True)

            repl_kperp = -x * collinear_momentum + rep_r

            # NEW: CUT THRESHOLD CUTTING REGION BY 4

            theta1 = (
                E(f"Θ(({self.sp3D(repl_kperp, repl_kperp)})-(x*(1-x))*Lambdasq/16)")
                .replace(x, repl_x)
                .replace(rexp, r.exp())
                .replace(r, (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")).log())
                .replace(khat, E("k(0)") * inv_knorm)
                .replace(inv_knorm, 1 / (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")))
            )
            theta2 = (
                # E(f"Θ(({self.sp3D(repl_kperp, repl_kperp)})-(x*(1-x))*Lambdasq)")
                E(f"Θ(({self.sp3D(repl_kperp, repl_kperp)})-(x*(1-x))*Lambdasq/16)")
                .replace(x, repl_x)
                .replace(rexp, r.exp())
                .replace(r, 2 * rstar - r)
                .replace(r, (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")).log())
                .replace(khat, E("k(0)") * inv_knorm)
                .replace(inv_knorm, 1 / (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")))
                .replace(
                    E("s"), 4 * (E("p(1,1)") ** 2 + E("p(1,2)") ** 2 + E("p(1,3)") ** 2)
                )
            )

            print(theta1)
            print(theta2)
        else:
            theta1 = E("1")
            theta2 = E("1")

        if self.name == "DY":
            hr = (
                # (-((r - rstar) ** 2) - 1 / r**2 + 1 / rstar**2)
                (-((r - rstar) ** 2))
                .exp()
                .replace(r, (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")).log())
                .replace(khat, E("k(0)") * inv_knorm)
                .replace(inv_knorm, 1 / (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")))
                .replace(
                    E("s"), 4 * (E("p(1,1)") ** 2 + E("p(1,2)") ** 2 + E("p(1,3)") ** 2)
                )
            )
        else:
            hr = (
                # (-((r - rstar) ** 2) - 1 / r**2 + 1 / rstar**2)
                (-((r - rstar) ** 2) - 1 / r**2 + 1 / rstar**2)
                .exp()
                .replace(r, (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")).log())
                .replace(khat, E("k(0)") * inv_knorm)
                .replace(inv_knorm, 1 / (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")))
                .replace(
                    E("s"), 4 * (E("p(1,1)") ** 2 + E("p(1,2)") ** 2 + E("p(1,3)") ** 2)
                )
            )

        jacobian_correction = (
            (-3 * (r - rstar))
            .exp()
            .replace(r, (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")).log())
            .replace(khat, E("k(0)") * inv_knorm)
            .replace(inv_knorm, 1 / (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")))
            .replace(
                E("s"), 4 * (E("p(1,1)") ** 2 + E("p(1,2)") ** 2 + E("p(1,3)") ** 2)
            )
        )

        threshold_integrand = (
            jacobian_correction * threshold_integrand * theta1 * theta2 * hr
        )

        # Go back to previous basis
        edge_by_id = {
            _strip_quotes(str(edge.get_attributes()["id"])): edge
            for edge in threshold_graph.graph.get_edges()
        }
        basis_replacements = []
        for j, id in enumerate(lmb_ids):
            basis_edge = edge_by_id.get(_strip_quotes(str(id)))
            if basis_edge is None:
                raise ValueError(f"Threshold basis edge {id} not found in cut graph")
            e_atts = basis_edge.get_attributes()
            routing_items = E("0")
            for i in range(self.L + 1):
                key = f"routing_k{i}"
                if key in e_atts:
                    routing_items += E(f"{e_atts[key]}*k[{i}]")
            for i in range(2):
                key = f"routing_p{i + 1}"
                if key in e_atts:
                    routing_items += E(f"{e_atts[key]}*p[{i + 1}]")
            basis_replacements.append((E(f"k({j})"), routing_items))

        tmp_symbols = [E(f"__tmp_th_{j}") for j in range(len(basis_replacements))]
        for (pattern, _), tmp_symbol in zip(basis_replacements, tmp_symbols):
            threshold_integrand = threshold_integrand.replace(pattern, tmp_symbol)
        for tmp_symbol, (_, replacement) in zip(tmp_symbols, basis_replacements):
            threshold_integrand = threshold_integrand.replace(tmp_symbol, replacement)

        return RoutedIntegrand(
            -threshold_integrand,
            threshold_graph,
            [],
            E("0"),
            "threshold",
            "threshold",
        )

    def construct_threshold_counter_terms(self):

        filtered_e_surfs = self.filter_e_surfaces()
        threshold_cts = []

        print("filtered e surfs")
        print(filtered_e_surfs)

        for thresh_ids, e_surf, residue in filtered_e_surfs:
            if len(thresh_ids) > 2:
                raise ValueError("not ready for two-loop threshold subtraction yet...")

            threshold_cts.append(
                self.construct_threshold_counter_term(
                    residue, deepcopy(self.routed_cut_graph), thresh_ids
                )
            )

        return threshold_cts


class Approximator(object):
    def __init__(self):
        self.sp3D = S("sp3D", is_linear=True, is_symmetric=True)

    def collinear_approximation(self, integrand, momentum, k_id, direction, order=-2):

        # kperp_sq = E(f"sp3D(k_perp({k_id[0]}),k_perp({k_id[0]}))")
        # coll_en = x * E("sp3D(p(1),p(1))^(1/2)") + lam**2 * kperp_sq / (
        #    2 * x * E("sp3D(p(1),p(1))^(1/2)")
        # )
        # a_coll_en = (1 - x) * E("sp3D(p(1),p(1))^(1/2)") + lam**2 * kperp_sq / (
        #    2 * (1 - x) * E("sp3D(p(1),p(1))^(1/2)")
        # )

        x = S("x", is_scalar=True, is_positive=True)
        lam = S("λ", is_scalar=True)

        # Let s*q(i) be the vector that should become collinear to p(1). s encodes the cut orientation. We
        # write s*q(i)=s*(q(i)-a*k(j))+a*s*k(j)= x*p(1)+lam*k_perp(j) and solve in k(j), giving
        # k(j)=a*s*x*p(1)+a*s*lam*k_perp(j)-a*(q(i)-a*k(j)). Now s=momentum[1] and a=k_id[1] and j=k_id[0].

        repl = k_id[1] * (
            momentum[1] * x * direction
            - (momentum[0] - k_id[1] * E(f"k({k_id[0]})"))
            + lam * E(f"k_perp({k_id[0]})")
        )

        integrand = integrand.replace(E(f"k({k_id[0]})"), repl)
        integrand = integrand.replace(
            self.sp3D(E(f"k_perp({k_id[0]})"), E("p(x_)")), E("0")
        )

        # Only consider leading-virtuality contribution

        # FIX ::::::::: REMEMBER TO CHANGE BACK TO -2 and overall -1 sign
        # integrand = integrand.series(lam, 0, order).to_expression().replace(lam, 1)
        print("got to expansion")
        integrand = integrand.series(lam, 0, -2).to_expression().replace(lam, 1)
        print("and beyond")

        # Invert back the collinear parametrisation. Since s*q(i)= x*p(1)+lam*k_perp(j), we have
        # x=s*q(i).p(1)/p(1).p(1)

        repl_x = (
            momentum[1]
            * self.sp3D(momentum[0], direction)
            / self.sp3D(direction, direction)
        )

        repl_kperp = -momentum[1] * x * direction + momentum[0]

        # integrand = integrand.replace(
        #    self.sp3D(E(f"k_perp({k_id[0]})"), E(f"k_perp({k_id[0]})")),
        #    self.sp3D(repl_kperp, repl_kperp),
        # ).replace(x, repl_x)

        # new: normalisation for general ecm: THIS IS FOR RAISED PROPAGATORS ONLY
        # integrand = integrand * (self.sp3D(E("p(1)"), E("p(1)"))) ** E("1/2")

        integrand = integrand.replace(
            E(f"k_perp({k_id[0]})"),
            repl_kperp,
        ).replace(x, repl_x)

        print("the collinear replacement happened")

        return integrand, repl, repl_x, repl_kperp

    def soft_approximation(self, integrand, momentum, k_id):

        lam = S("λ", is_scalar=True)

        repl = k_id[1] * (-(momentum[0] - k_id[1] * E(f"k({k_id[0]})"))) + k_id[
            1
        ] * lam * E("qsoft")

        integrand = integrand.replace(E(f"k({k_id[0]})"), repl)

        integrand = integrand.series(lam, 0, -3).to_expression().replace(lam, 1)

        integrand = integrand.replace(E("qsoft"), momentum[0])

        return integrand, repl


class LoopIntegrandConstructor(object):
    def __init__(self, params, name, L, channel=None, disable_integrated_uv_cts=True):
        self.L = L
        self.params = params
        self.name = name
        self.emr_processor = EMRIntegrandConstructor(params, name, L)
        self.sp3D = S("sp3D", is_linear=True, is_symmetric=True)
        self.approximator = Approximator()
        self.channel = channel
        self.disable_integrated_uv_cts = bool(disable_integrated_uv_cts)

    # Replaces energies by their expression in terms of the emr momenta and particle masses

    def replace_energies(self, integrand, cut_graph):

        for e in cut_graph.graph.get_edges():
            e_atts = e.get_attributes()
            eid_raw = e_atts["id"]
            eid = _strip_quotes(eid_raw) if isinstance(eid_raw, str) else eid_raw
            target = E(f"E({eid})")
            target_num = E(f"En({eid})")
            particle = _strip_quotes(str(e_atts["particle"]))
            if particle not in ["d", "d~", "g", "ghG", "ghG~"]:
                replacement = (
                    self.sp3D(E(f"q({eid})"), E(f"q({eid})")) + E(f"m({particle})") ** 2
                ) ** E("1/2")
                replacement_num = replacement
            else:
                # new stuff
                k_keys = [f"routing_k{i}" for i in range(0, self.L)]
                zero_routing = all(e_atts[kk] == "0" for kk in k_keys)
                mass = E("0")
                if (
                    zero_routing
                    and e_atts["routing_p1"] == "0"
                    and e_atts["routing_p2"] != "0"
                ):
                    mass = E("p2sq")
                if (
                    zero_routing
                    and e_atts["routing_p1"] != "0"
                    and e_atts["routing_p2"] == "0"
                ):
                    mass = E("p1sq")

                replacement = (self.sp3D(E(f"q({eid})"), E(f"q({eid})")) + mass) ** E(
                    "1/2"
                )
                replacement_num = (self.sp3D(E(f"q({eid})"), E(f"q({eid})"))) ** E(
                    "1/2"
                )

            integrand = integrand.replace(target, replacement)
            integrand = integrand.replace(target_num, replacement_num)

        return integrand

    # Substituted the emr momenta by their linear decomposition in terms of
    # loop variables and external momenta (p1 and p2)

    def route_integrand(self, integrand, cut_graph):

        for e in cut_graph.graph.get_edges():
            e_atts = e.get_attributes()
            routing_items = E("0")
            for i in range(self.L):
                key = f"routing_k{i}"
                if key in e_atts:
                    routing_items += E(f"{e_atts[key]}*k({i})")
            for i in range(2):
                key = f"routing_p{i + 1}"
                if key in e_atts:
                    routing_items += E(f"{e_atts[key]}*p({i + 1})")
            integrand = integrand.replace(E(f"q({e_atts['id']})"), routing_items)

        integrand = integrand.replace(
            E("sp3(x___,y___)"), self.sp3D(S("x___"), S("y___"))
        )
        return integrand

    # Checks if two edges have the same routing, implying they form a raised propagator.

    def _routing_sign_match(self, e: pydot.Edge, ep: pydot.Edge):
        a = e.get_attributes()
        b = ep.get_attributes()

        # collect all routing keys (p1, p2 and any k*)
        keys = [k for k in set(a.keys()) | set(b.keys()) if k.startswith("routing_")]

        def val(attrs, k):
            return E(attrs.get(k, "0"))

        same = all(val(a, k) == val(b, k) for k in keys)
        opp = all(val(a, k) == -val(b, k) for k in keys)
        if same:
            return "same"
        if opp:
            return "opp"
        return None

    # Changes the routing of a graph based on an input lmb choice.

    def canonicalise_energies(self, integrand, cut_graph):

        rep = E("0")

        f_cut_set = set(cut_graph.final_cut)
        i_cut_set = set(cut_graph.initial_cut)
        cut_union = f_cut_set.union(i_cut_set)
        cut_intersection = f_cut_set.intersection(i_cut_set)

        first_f = True
        patt = None

        e_to_sub = next(iter(f_cut_set - i_cut_set))

        rep = E("0")

        for e in cut_graph.initial_cut:
            e_atts = e.get_attributes()
            rep += E(f"E({e_atts['id']})")

        for e in cut_graph.final_cut:
            e_atts = e.get_attributes()
            rep -= E(f"E({e_atts['id']})")

        e_to_sub_atts = e_to_sub.get_attributes()
        patt = E(f"E({e_to_sub_atts['id']})")
        rep = rep + E(f"E({e_to_sub_atts['id']})")

        print("canoniucalisation: replacing ", patt, " by ", rep)

        integrand = integrand.replace(patt, rep)

        return integrand

    def concretise_scalar_products(self, integrand):

        return integrand.replace(
            E("sp3D(w_(x_),z_(y_))"),
            E("w_(x_,1)*z_(y_,1)+w_(x_,2)*z_(y_,2)+w_(x_,3)*z_(y_,3)"),
        )

    # Approximates the integrand at leading virtuality. For parton model diagrams, it simply replaces the
    # energies and routes the integrand, giving an expression in terms of loop momenta. No approximation is
    # performed. For partitions of the type [i_1]_[i_2,i_3,...], takes the limit p2sq->0 by setting i_2,i_3,...
    # collinear to p2 and expanding for small transverse momenta around this collinear configuration. Same for
    # [i_2,i_3,...]_[i_1] with p2sq substituted with p1sq.

    def leading_virtuality_expansion(self, integrand, cut_graph, raised_cut):
        emr_integrand = deepcopy(integrand)
        partition = cut_graph.partition

        routed_integrands = []

        if len(partition[0]) == 1 and len(partition[1]) == 1:
            integrand = self.replace_energies(integrand, cut_graph)
            integrand = integrand.replace(E("p1sq"), E("0"))
            integrand = integrand.replace(E("p2sq"), E("0"))
            integrand = self.route_integrand(integrand, cut_graph)
            routed_integrand = RoutedIntegrand(
                integrand, cut_graph, [], emr_integrand, "PM", []
            )
            routed_integrands.append(routed_integrand)
            return routed_integrands

        elif len(partition[0]) > 1 and len(partition[1]) == 1:
            x = S("x", is_scalar=True, is_positive=True)
            lam = S("λ", is_scalar=True)
            momentum = E("0")

            coll_moms = []

            # Find collinear momenta and particles.
            first = True
            for ep in partition[0]:
                ep_atts = ep.get_attributes()
                id = ep_atts["id"]
                coll_moms.append(id)
                for e in cut_graph.graph.get_edges():
                    e_atts = e.get_attributes()
                    if e_atts["id"] == id:
                        k_keys = ["routing_k" + str(i) for i in range(0, self.L)]
                        loop_coeff = [E(e_atts[rout]) for rout in k_keys]
                        if first:
                            k_id = next(
                                (i, c)
                                for i, c in enumerate(loop_coeff)
                                if str(c) != "0"
                            )
                            momentum = [
                                (
                                    sum(
                                        loop_coeff[i] * E(f"k({i})")
                                        for i in range(self.L)
                                    )
                                    + E(e_atts["routing_p1"]) * E("p(1)")
                                    + E(e_atts["routing_p2"]) * E("p(2)")
                                ),
                                e_atts["is_cut_DY"],
                            ]
                            first = False

            # In order to make the collinear replacement always work, replace a final-state energy
            # by energy conservation.

            integrand = self.canonicalise_energies(integrand, cut_graph)

            # integrand = integrand.replace(E(f"E({coll_moms[0]})"), coll_en)
            # integrand = integrand.replace(E(f"E({coll_moms[1]})"), a_coll_en)

            # integrand = integrand * 1 / (E("4*E(0)*(E(7)+E(8)-E(0))"))

            integrand = self.replace_energies(integrand, cut_graph)

            if len(raised_cut) > 0:
                integrand = integrand * (self.sp3D(E("p(1)"), E("p(1)"))) ** E("1/2")
                print("got to derivative")
                integrand = E("1/2") * integrand.derivative(E("p1sq"))
                print("and beyond")

            integrand = integrand.replace(E("p1sq"), E("0"))
            integrand = integrand.replace(E("p2sq"), E("0"))

            integrand = self.route_integrand(integrand, cut_graph)

            integrand, repl, repl_x, repl_kperp = (
                self.approximator.collinear_approximation(
                    integrand, momentum, k_id, E("p(1)")
                )
            )

            print("expanded expression is available")

            thetaLambdasq = E(
                f"Θ(Lambdasq-({self.sp3D(repl_kperp, repl_kperp)})/(x*(1-x)))"
            ).replace(x, repl_x)

            integrand = (
                integrand * E(f"Θ({repl_x})") * E(f"Θ(1-{repl_x})") * thetaLambdasq
            )

            routed_integrand = RoutedIntegrand(
                integrand,
                cut_graph,
                [
                    E(f"k({k_id[0]})"),
                    repl.replace(x, repl_x).series(lam, 0, 0).to_expression(),
                ],
                emr_integrand,
                "collinear",
                [
                    E(f"k({k_id[0]})"),
                    repl.replace(x, repl_x).series(lam, 0, 0).to_expression(),
                ],
            )
            routed_integrands.append(routed_integrand)

            print("routed integrand constructed")

            return routed_integrands

        elif len(partition[0]) == 1 and len(partition[1]) > 1:
            x = S("x", is_scalar=True, is_positive=True)
            lam = S("λ", is_scalar=True)
            momentum = E("0")

            coll_moms = []

            # Find collinear momenta and particles.

            first = True
            for ep in partition[1]:
                ep_atts = ep.get_attributes()
                id = ep_atts["id"]
                coll_moms.append(id)
                for e in cut_graph.graph.get_edges():
                    e_atts = e.get_attributes()
                    if e_atts["id"] == id:
                        k_keys = ["routing_k" + str(i) for i in range(self.L)]
                        loop_coeff = [E(e_atts[rout]) for rout in k_keys]
                        if first:
                            k_id = next(
                                (i, c)
                                for i, c in enumerate(loop_coeff)
                                if str(c) != "0"
                            )
                            momentum = [
                                (
                                    sum(
                                        loop_coeff[i] * E(f"k({i})")
                                        for i in range(self.L)
                                    )
                                    + E(e_atts["routing_p1"]) * E("p(1)")
                                    + E(e_atts["routing_p2"]) * E("p(2)")
                                ),
                                e_atts["is_cut_DY"],
                            ]
                            first = False

            # In order to make the collinear replacement always work, replace a final-state energy
            # by energy conservation.

            integrand = self.canonicalise_energies(integrand, cut_graph)

            # integrand = integrand.replace(E(f"E({coll_moms[0]})"), coll_en)
            # integrand = integrand.replace(E(f"E({coll_moms[1]})"), a_coll_en)

            # integrand = integrand / (E("4*E(0)*(E(7)+E(8)-E(0))"))

            integrand = self.replace_energies(integrand, cut_graph)

            if len(raised_cut) > 0:
                integrand = integrand * (self.sp3D(E("p(2)"), E("p(2)"))) ** E("1/2")
                print("got to derivative")
                integrand = E("1/2") * integrand.derivative(E("p2sq"))

            integrand = integrand.replace(E("p1sq"), E("0"))
            integrand = integrand.replace(E("p2sq"), E("0"))

            integrand = self.route_integrand(integrand, cut_graph)

            integrand_old = deepcopy(integrand)

            integrand, repl, repl_x, repl_kperp = (
                self.approximator.collinear_approximation(
                    integrand, momentum, k_id, E("p(2)")
                )
            )

            thetaLambdasq = E(
                f"Θ(Lambdasq-({self.sp3D(repl_kperp, repl_kperp)})/(x*(1-x)))"
            ).replace(x, repl_x)
            integrand = (
                integrand * E(f"Θ({repl_x})") * E(f"Θ(1-{repl_x})") * thetaLambdasq
            )

            # HACKED OLD
            routed_integrand = RoutedIntegrand(
                integrand,
                cut_graph,
                [
                    E(f"k({k_id[0]})"),
                    repl.replace(x, repl_x).series(lam, 0, 0).to_expression(),
                ],
                emr_integrand,
                "anti-collinear",
                [
                    E(f"k({k_id[0]})"),
                    repl.replace(x, repl_x).series(lam, 0, 0).to_expression(),
                ],
            )
            routed_integrands.append(routed_integrand)

            return routed_integrands

        elif len(partition[0]) > 1 and len(partition[1]) > 1:
            lam = S("λ", is_scalar=True)
            x = S("x", is_scalar=True, is_positive=True)

            soft_edge = list(set(partition[0]).intersection(partition[1]))
            hard_edge1 = [e for e in partition[0] if e not in soft_edge]
            hard_edge2 = [e for e in partition[1] if e not in soft_edge]

            if len(soft_edge) != 1 or len(hard_edge1) != 1 or len(hard_edge2) != 1:
                raise ValueError(
                    "Big problem with soft approximation!!! (or your desires are too demanding for this poor, little code)"
                )

            id = soft_edge[0].get_attributes()["id"]
            id_hard1 = hard_edge1[0].get_attributes()["id"]
            id_hard2 = hard_edge2[0].get_attributes()["id"]
            momentum = []
            hard_momentum1 = []
            hard_momentum2 = []
            for e in cut_graph.graph.get_edges():
                e_atts = e.get_attributes()
                if e_atts["id"] == id:
                    k_keys = ["routing_k" + str(i) for i in range(0, self.L)]
                    loop_coeff = [E(e_atts[rout]) for rout in k_keys]
                    k_id = next(
                        (i, c) for i, c in enumerate(loop_coeff) if str(c) != "0"
                    )
                    momentum = [
                        (
                            sum(loop_coeff[i] * E(f"k({i})") for i in range(0, self.L))
                            + E(e_atts["routing_p1"]) * E("p(1)")
                            + E(e_atts["routing_p2"]) * E("p(2)")
                        ),
                        e_atts["is_cut_DY"],
                    ]
                if e_atts["id"] == id_hard1:
                    k_keys = ["routing_k" + str(i) for i in range(0, self.L)]
                    loop_coeff = [E(e_atts[rout]) for rout in k_keys]
                    hard_momentum1 = [
                        (
                            sum(loop_coeff[i] * E(f"k({i})") for i in range(0, self.L))
                            + E(e_atts["routing_p1"]) * E("p(1)")
                            + E(e_atts["routing_p2"]) * E("p(2)")
                        ),
                        e_atts["is_cut_DY"],
                    ]
                if e_atts["id"] == id_hard2:
                    k_keys = ["routing_k" + str(i) for i in range(0, self.L)]
                    loop_coeff = [E(e_atts[rout]) for rout in k_keys]
                    hard_momentum2 = [
                        (
                            sum(loop_coeff[i] * E(f"k({i})") for i in range(0, self.L))
                            + E(e_atts["routing_p1"]) * E("p(1)")
                            + E(e_atts["routing_p2"]) * E("p(2)")
                        ),
                        e_atts["is_cut_DY"],
                    ]

            # integrand = E("(-(E(0)+E(2)+E(3))+E(0)+E(5))^-1*E(2)^(-2)")
            base_graph_name = _strip_quotes(str(cut_graph.graph.get("base_graph_name")))

            if (
                base_graph_name in ["GL07", "GL09"]
                and self.channel == (1, -1)
                or self.channel == (-1, 1)
            ):
                integrand = integrand.replace(
                    E("E(5)"), self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")
                )
                integrand = integrand.replace(
                    E("E(4)"), self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")
                )

            # print("hacked integrand")
            # print(integrand)

            integrand = self.canonicalise_energies(integrand, cut_graph)

            # print("soft emr integrand")
            # print(integrand)

            integrand = self.replace_energies(integrand, cut_graph)
            integrand = integrand.replace(E("p1sq"), E("0"))
            integrand = integrand.replace(E("p2sq"), E("0"))
            integrand = self.route_integrand(integrand, cut_graph)

            # print("routed emr integrand")
            # print(integrand)

            # Factor of 1/s for soft and soft-collinear for virtual DY diagram

            factor = 1
            if self.name == "DY" and len(cut_graph.final_cut) == 1:
                factor = 1 / (4 * self.sp3D(E("p(1)"), E("p(2)")))

            # if self.name=="tt~":
            # integrand = integrand.replace(E("p(2)"), -E("p(1)"))
            # momentum[0] = momentum[0].replace(E("p(2)"), -E("p(1)"))

            print("input-" * 10)
            print(f"hard particles are {id_hard1} and {id_hard2}")
            print(deepcopy(momentum))
            print(deepcopy(k_id))

            soft_integrand, repl_s = self.approximator.soft_approximation(
                deepcopy(integrand), deepcopy(momentum), deepcopy(k_id)
            )

            # print("soft approximation")
            # print(soft_integrand)

            soft_collinear_integrand1, repl1, repl1_x, repl_kperp1 = (
                self.approximator.collinear_approximation(
                    deepcopy(soft_integrand),
                    momentum,
                    k_id,
                    E("p(1)"),
                )
            )

            soft_collinear_integrand2, repl2, repl2_x, repl_kperp2 = (
                self.approximator.collinear_approximation(
                    deepcopy(soft_integrand),
                    momentum,
                    k_id,
                    E("p(2)"),
                )
            )

            if self.name == "tt~" and len(cut_graph.final_cut) == 2:
                soft_collinear_integrand1 = -soft_collinear_integrand1
                soft_collinear_integrand2 = -soft_collinear_integrand2
                soft_integrand = -soft_integrand

            # integrand = (
            #    soft_integrand + soft_collinear_integrand1 + soft_collinear_integrand2
            # )  # * E(f"Θ({repl_x})") * E(f"Θ(1-{repl_x})")

            # need to construct propagators
            propsoft1 = 2 * (
                (self.sp3D(momentum[0], momentum[0])) ** E("1/2")
                * (self.sp3D(E("p(1)"), E("p(1)"))) ** E("1/2")
                - momentum[1] * self.sp3D(momentum[0], E("p(1)"))
            )
            propsoft2 = 2 * (
                (self.sp3D(momentum[0], momentum[0])) ** E("1/2")
                * (self.sp3D(E("p(2)"), E("p(2)"))) ** E("1/2")
                - momentum[1] * self.sp3D(momentum[0], E("p(2)"))
            )

            thetaSoft = E(f"Θ(Lambdasq-{propsoft1})") * E(f"Θ(Lambdasq-{propsoft2})")  #

            routed_integrand_soft = RoutedIntegrand(
                -factor * soft_integrand * thetaSoft,
                cut_graph,
                [
                    E(f"k({k_id[0]})"),
                    repl_s.series(lam, 0, 0).to_expression(),
                ],
                emr_integrand,
                "soft",
                [
                    E(f"k({k_id[0]})"),
                    repl_s.series(lam, 0, 0).to_expression(),
                ],
            )
            routed_integrands.append(routed_integrand_soft)

            thetacollinear1 = (
                E(f"Θ(Lambdasq-({self.sp3D(repl_kperp1, repl_kperp1)})/(x))").replace(
                    x, repl1_x
                )
                * E(f"Θ(Lambdasq-4*{self.sp3D(E('p(1)'), E('p(1)'))}*x)").replace(
                    x, repl1_x
                )
                * E(f"Θ({repl1_x})")
                # * E(f"Θ(1-{repl1_x})")
            )

            routed_integrand_collinear1 = RoutedIntegrand(
                factor * soft_collinear_integrand1 * thetacollinear1,
                cut_graph,
                [
                    E(f"k({k_id[0]})"),
                    repl_s.series(lam, 0, 0).to_expression(),
                    # repl1.replace(x, repl1_x).series(lam, 0, 0).to_expression(),
                ],
                emr_integrand,
                "soft-collinear",
                [
                    E(f"k({k_id[0]})"),
                    repl1.replace(x, repl1_x).series(lam, 0, 0).to_expression(),
                ],
            )
            routed_integrands.append(routed_integrand_collinear1)

            thetacollinear2 = (
                E(f"Θ(Lambdasq-({self.sp3D(repl_kperp2, repl_kperp2)})/(x))").replace(
                    x, repl2_x
                )
                * E(f"Θ(Lambdasq-4*{self.sp3D(E('p(1)'), E('p(1)'))}*(x))").replace(
                    x, repl2_x
                )
                * E(f"Θ({repl2_x})")
                # * E(f"Θ(1-{repl2_x})")
            )

            routed_integrand_collinear2 = RoutedIntegrand(
                factor * soft_collinear_integrand2 * thetacollinear2,
                cut_graph,
                [
                    E(f"k({k_id[0]})"),
                    repl_s.series(lam, 0, 0).to_expression(),
                    # repl2.replace(x, repl2_x).series(lam, 0, 0).to_expression(),
                ],
                emr_integrand,
                "soft-anti-collinear",
                [
                    E(f"k({k_id[0]})"),
                    repl2.replace(x, repl2_x).series(lam, 0, 0).to_expression(),
                ],
            )
            routed_integrands.append(routed_integrand_collinear2)

            return routed_integrands

        else:
            raise ValueError("Big problem if you get here :( ")

    # Eliminates raised propagators by multiplying the relevant diagrams by the raised denominator

    def eliminate_raised_cuts(self, emr_representation, cut_graph):

        raised_cut = []
        g_edges = cut_graph.graph.get_edges()
        init_cut_ids = [e.get_attributes()["id"] for e in cut_graph.initial_cut]
        for e, i in zip(g_edges, range(len(g_edges))):
            e_atts = e.get_attributes()
            for ep, j in zip(g_edges, range(len(g_edges))):
                ep_atts = ep.get_attributes()
                relation = self._routing_sign_match(e, ep)
                if (
                    j > i
                    and relation is not None
                    and (
                        e_atts.get("is_cut_DY", None) is not None
                        or ep_atts.get("is_cut_DY", None) is not None
                    )
                ):
                    if e_atts["id"] not in init_cut_ids:
                        raised_cut.append([
                            e_atts["id"],
                            ep_atts["id"],
                            relation,
                            _strip_quotes(str(e_atts["particle"])),
                        ])
                    else:
                        raised_cut.append([
                            ep_atts["id"],
                            e_atts["id"],
                            relation,
                            _strip_quotes(str(e_atts["particle"])),
                        ])

        print("RAISED CUTS ARE: ", raised_cut)

        # compute energy conservation condition (specialised to "DY")
        initial_cut_ids = [e.get_attributes()["id"] for e in cut_graph.initial_cut]
        final_cut_ids = [e.get_attributes()["id"] for e in cut_graph.final_cut]

        if self.name == "DY":
            photon_id = [
                e.get_attributes()["id"]
                for e in cut_graph.final_cut
                if _strip_quotes(str(e.get_attributes()["particle"])) == "a"
            ]

            if len(photon_id) != 1:
                raise ValueError(
                    "problem with final state gamma in raised cut treatment"
                )

            repl = (
                sum(E(f"E({id})") for id in initial_cut_ids)
                - sum(E(f"E({id})") for id in final_cut_ids)
                + E(f"E({photon_id[0]})")
            ).expand()

            emr_representation = emr_representation.replace(
                E(f"E({photon_id[0]})"), repl
            )

        if self.name == "tt~":
            if len(raised_cut) > 0:
                if raised_cut[0][3] not in ["t", "t~"]:
                    tt_id = [
                        e.get_attributes()["id"]
                        for e in cut_graph.final_cut
                        if _strip_quotes(str(e.get_attributes()["particle"])) == "t"
                        or _strip_quotes(str(e.get_attributes()["particle"])) == "t~"
                    ]

                    if len(tt_id) != 2:
                        raise ValueError(
                            "problem with final state tt in raised cut treatment"
                        )

                    repl = (
                        sum(E(f"E({id})") for id in initial_cut_ids)
                        - sum(E(f"E({id})") for id in final_cut_ids)
                        + E(f"E({tt_id[0]})")
                    ).expand()
                    emr_representation = emr_representation.replace(
                        E(f"E({tt_id[0]})"), repl
                    )
                else:
                    repl = (
                        sum(E(f"E({id})") for id in initial_cut_ids)
                        - sum(E(f"E({id})") for id in final_cut_ids)
                        + E(f"E({initial_cut_ids[0]})")
                    ).expand()
                    emr_representation = emr_representation.replace(
                        E(f"E({initial_cut_ids[0]})"), repl
                    )

        base_graph_name = _strip_quotes(str(cut_graph.graph.get("base_graph_name")))
        edge_by_id = {e.get_attributes()["id"]: e for e in cut_graph.graph.get_edges()}

        is_final_raised = False

        if len(raised_cut) > 0:
            for cut in raised_cut:
                if cut[2] == "opp" and cut[3] != "g":
                    # minus sign is because the correct way to correct for an opposite routing of the
                    # raised propagator would be to actually switch the sign of the energy in the
                    # numerator only, which at this point is difficult to access.

                    if not (
                        self.L == 2
                        and (self.channel == (-1, 1) or self.channel == (1, -1))
                        and base_graph_name in ["GL06", "GL08"]
                    ):
                        emr_representation = -emr_representation.replace(
                            E(f"q({cut[0]})"),
                            -E(f"q({cut[0]})"),
                        )

                # emr_representation = emr_representation.replace(E("q(6)"), E("-q(6)"))

                # print("EMR AFTER LIMITTTTT")
                # print(emr_representation)

                if cut[3] in ["t", "t~"]:
                    print("HEREEEEEEE" * 10)
                    print(cut)
                    cut_repeated_ids = [
                        eid
                        for eid in [cut[0], cut[1]]
                        if edge_by_id[eid].get_attributes().get("is_cut_DY") is not None
                    ]
                    cut_repeated_id = cut[0]
                    other_repeated_id = cut[1]
                    if len(cut_repeated_ids) == 1:
                        cut_repeated_id = cut_repeated_ids[0]
                        other_repeated_id = (
                            cut[1] if cut_repeated_id == cut[0] else cut[0]
                        )
                    emr_representation = (
                        emr_representation
                        * (E(f"E({cut_repeated_id})") - E(f"E({other_repeated_id})"))
                        * (2 * E(f"E({cut_repeated_id})"))
                    )
                    emr_representation = emr_representation.replace(
                        E(f"E({cut_repeated_id})"),
                        E(f"E({other_repeated_id})") + E("same"),
                    )
                    emr_representation = emr_representation.series(
                        E("same"), 0, 0
                    ).to_expression()
                    is_final_raised = True
                    if len(cut_repeated_ids) == 1:
                        repl = (
                            sum(E(f"E({id})") for id in initial_cut_ids)
                            - sum(
                                E(f"E({id})")
                                for id in final_cut_ids
                                if id != cut_repeated_id
                            )
                        ).expand()
                        emr_representation = -emr_representation.replace(
                            E(f"E({other_repeated_id})"), repl
                        )

                        emr_representation = -emr_representation.replace(
                            E(f"En({cut_repeated_id})"), repl
                        )

                        emr_representation = -emr_representation.replace(
                            E(f"En({other_repeated_id})"), repl
                        )

                        emr_representation = emr_representation / (
                            E(f"E({cut_repeated_id})")
                        )

                        print("check repetated energyyy")
                        print(emr_representation)
                else:
                    print("HEREEEWWWWWW" * 10)
                    print(cut)
                    emr_representation = (
                        emr_representation
                        * (E(f"E({cut[0]})") - E(f"E({cut[1]})"))
                        * (2)
                    )
                    emr_representation = emr_representation.replace(
                        E(f"E({cut[0]})"), E(f"E({cut[1]})") + E("same")
                    )
                    emr_representation = emr_representation.series(
                        E("same"), 0, 0
                    ).to_expression()

        # HACKY HACK THAT IS NEW
        # emr_representation = emr_representation.replace(E("q(6)"), E("-q(6)"))

        # emr_representation = emr_representation.replace(E("q(8)"), E("-q(8)"))

        return emr_representation, raised_cut, is_final_raised

    def modify_t_channel_gluon_numerator(self, cut_graph):

        has_raised_t_channel_gluon = [False, 0, 0, None]
        g_edges = cut_graph.graph.get_edges()
        for e, i in zip(g_edges, range(len(g_edges))):
            e_atts = e.get_attributes()
            k_keys = ["routing_k" + str(i) for i in range(0, self.L)]
            p_keys = ["routing_p1", "routing_p2"]
            for ep, j in zip(g_edges, range(len(g_edges))):
                ep_atts = ep.get_attributes()
                relation = self._routing_sign_match(e, ep)
                if (
                    j > i
                    and relation is not None
                    and _strip_quotes(str(e_atts.get("particle"))) == "g"
                    and _strip_quotes(str(ep_atts.get("particle"))) == "g"
                    and not all(e_atts.get(key) == "0" for key in k_keys)
                    and not all(e_atts.get(key) != "0" for key in p_keys)
                ):
                    has_raised_t_channel_gluon = [True, e, ep, relation]

        if has_raised_t_channel_gluon[0]:
            e1 = has_raised_t_channel_gluon[1]
            e2 = has_raised_t_channel_gluon[2]
            repeated_relation = has_raised_t_channel_gluon[3]
            k_keys = [f"routing_k{i}" for i in range(0, self.L)]

            def _is_pure_external_edge(edge):
                edge_atts = edge.get_attributes()
                has_p1 = _strip_quotes(str(edge_atts.get("routing_p1", "0"))) != "0"
                has_p2 = _strip_quotes(str(edge_atts.get("routing_p2", "0"))) != "0"
                has_single_external_momentum = has_p1 != has_p2
                has_no_loop_momentum = all(
                    _strip_quotes(str(edge_atts.get(key, "0"))) == "0" for key in k_keys
                )
                return has_single_external_momentum and has_no_loop_momentum

            def _has_external_with_orientation(candidate_edge, orientation):
                candidate_nodes = [
                    _base_node(candidate_edge.get_source()),
                    _base_node(candidate_edge.get_destination()),
                ]

                for node_name in candidate_nodes:
                    for incident_edge in boundary_edges(cut_graph.graph, {node_name}):
                        incident_atts = incident_edge.get_attributes()
                        if incident_atts["id"] == candidate_edge.get_attributes()["id"]:
                            continue
                        if not _is_pure_external_edge(incident_edge):
                            continue

                        incident_node = (
                            _base_node(incident_edge.get_destination())
                            if orientation == "injecting"
                            else _base_node(incident_edge.get_source())
                        )
                        if incident_node == node_name:
                            return True

                return False

            e1_has_injecting_external = _has_external_with_orientation(e1, "injecting")
            e2_has_injecting_external = _has_external_with_orientation(e2, "injecting")
            if e1_has_injecting_external == e2_has_injecting_external:
                raise ValueError(
                    "could not uniquely order raised gluon edges by external injection"
                )
            if not e1_has_injecting_external:
                e1, e2 = e2, e1

            e1_atts = e1.get_attributes()
            e2_atts = e2.get_attributes()

            def _is_cut_edge(edge_atts):
                return _strip_quotes(str(edge_atts.get("is_cut", "0"))) not in (
                    "0",
                    "0.0",
                )

            if not _has_external_with_orientation(e2, "departing"):
                raise ValueError(
                    "the second raised gluon edge has no departing pure external edge"
                )

            # Determine the corresponding IR limit
            e1_s = e1.get_source()
            e1_d = e1.get_destination()
            e2_s = e2.get_source()
            e2_d = e2.get_destination()

            # kinda brittle and kinda not brittle (assumes a specific routing)
            vertices1 = [e1_s, e1_d]
            vertices2 = [e2_s, e2_d]

            # vertices1 = [e1_s, e1_d]
            # vertices2 = [e2_s, e2_d]

            overall_sign1 = 1
            overall_sign2 = -1
            if e1_atts.get("routing_p1") == "1":
                if e2_atts.get("routing_p1") == "1":
                    vertices2 = [e2_d, e2_s]
                    overall_sign2 = 1
            elif e1_atts.get("routing_p1") == "-1":
                vertices1 = [e1_d, e1_s]
                overall_sign1 = -1
                if e2_atts.get("routing_p1") == "1":
                    vertices2 = [e2_d, e2_s]
                    overall_sign2 = 1
            elif e1_atts.get("routing_p2") == "1":
                if e2_atts.get("routing_p2") == "1":
                    vertices2 = [e2_d, e2_s]
                    overall_sign2 = 1
            elif e1_atts.get("routing_p2") == "-1":
                vertices1 = [e1_d, e1_s]
                overall_sign1 = -1
                if e2_atts.get("routing_p2") == "1":
                    vertices2 = [e2_d, e2_s]
                    overall_sign2 = 1
            else:
                raise ValueError("routing in gluonic t-channel exchange has a problem")

            # if repeated_relation == "opp":
            #    vertices2 = [vertices2[1], vertices2[0]]

            indices1 = []
            for e in boundary_edges(cut_graph.graph, {_base_node(vertices1[0])}):
                e_atts = e.get_attributes()
                if e_atts["id"] != e1_atts["id"]:
                    if _base_node(vertices1[0]) == _base_node(e.get_source()):
                        indices1.append((e_atts["id"], overall_sign1))
                    else:
                        indices1.append((e_atts["id"], -overall_sign1))

            indices2 = []
            for e in boundary_edges(cut_graph.graph, {_base_node(vertices2[0])}):
                e_atts = e.get_attributes()
                if e_atts["id"] != e2_atts["id"]:
                    if _base_node(vertices2[0]) == _base_node(e.get_source()):
                        indices2.append((e_atts["id"], overall_sign2))
                    else:
                        indices2.append((e_atts["id"], -overall_sign2))

            target_node_to_edge_ids = {}
            target_node_to_edge_ids.setdefault(_base_node(vertices1[1]), []).append(
                e1_atts["id"]
            )
            target_node_to_edge_ids.setdefault(_base_node(vertices2[1]), []).append(
                e2_atts["id"]
            )
            for node in cut_graph.graph.get_nodes():
                node_name = _strip_quotes(str(node.get_name()))
                node_int_id = _strip_quotes(str(node.get_attributes().get("int_id")))
                if node_int_id != "V_36" or node_name not in target_node_to_edge_ids:
                    continue

                node_num = node.get("num")
                if not node_num:
                    continue

                print("substituting momentum conservation condition")

                incident_edges = boundary_edges(cut_graph.graph, {node_name})
                num = Es(node_num)

                for target_edge_id in target_node_to_edge_ids[node_name]:
                    target_edge = next(
                        (
                            e
                            for e in incident_edges
                            if e.get_attributes()["id"] == target_edge_id
                        ),
                        None,
                    )
                    if target_edge is None:
                        raise ValueError(
                            f"target edge {target_edge_id} not incident to {node_name}"
                        )

                    edge_id_pattern = Es(f"Q({target_edge_id},y___)")
                    edge_id_replace = Es("0")
                    for incident_edge in incident_edges:
                        incident_edge_id = incident_edge.get_attributes()["id"]
                        if incident_edge_id == target_edge_id:
                            continue
                        edge_sign = (
                            1
                            if _base_node(incident_edge.get_source()) == node_name
                            else -1
                        )
                        edge_id_replace += edge_sign * Es(f"Q({incident_edge_id},y___)")
                    if _base_node(target_edge.get_source()) == node_name:
                        edge_id_replace = -edge_id_replace

                    num = num.replace(edge_id_pattern, edge_id_replace)

                node.get_attributes()["num"] = expr_to_string(num)

            # Finally construct the counter-terms
            #


            for e in cut_graph.graph.get_edges():
                e_atts = e.get_attributes()
                if e_atts["id"] == e1_atts["id"]:
                    ##
                    if _is_cut_edge(e1_atts) or _is_cut_edge(e2_atts):
                        ##
                        e_atts["num"] = (
                            f"-1𝑖*(spenso::g(spenso::coad(8,hedge({_parse_port(e.get_destination())})),spenso::coad(8,hedge({_parse_port(e.get_source())})))*spenso::g(spenso::mink(4,hedge({_parse_port(e.get_destination())})),spenso::mink(4,hedge({_parse_port(e.get_source())})))-{overall_sign1}*1/Q({e_atts['id']},0)*Qp({e_atts['id']},spenso::mink(4,hedge({_parse_port(vertices1[1])})))*Q(1000,spenso::mink(4,hedge({_parse_port(vertices1[0])})))*spenso::g(spenso::coad(8,hedge({_parse_port(e.get_destination())})),spenso::coad(8,hedge({_parse_port(e.get_source())}))))"
                        )
                        print("- IN CUT -" * 10)
                    else:
                        e_atts["num"] = (
                            f"-1𝑖*(spenso::g(spenso::coad(8,hedge({_parse_port(e.get_destination())})),spenso::coad(8,hedge({_parse_port(e.get_source())})))*spenso::g(spenso::mink(4,hedge({_parse_port(e.get_destination())})),spenso::mink(4,hedge({_parse_port(e.get_source())})))+(-1)*(-1)*1/({indices1[0][1]}*Q({indices1[0][0]},0)+{indices1[1][1]}*Q({indices1[1][0]},0))*Qp({e_atts['id']},spenso::mink(4,hedge({_parse_port(vertices1[1])})))*Q(1000,spenso::mink(4,hedge({_parse_port(vertices1[0])}))*spenso::g(spenso::coad(8,hedge({_parse_port(e.get_destination())})),spenso::coad(8,hedge({_parse_port(e.get_source())}))))"
                        )
                        print("- NOT IN CUT -" * 10)
                        print("sign 1")
                        print(overall_sign1)
                        print(e_atts["num"])

                if e_atts["id"] == e2_atts["id"]:
                    if _is_cut_edge(e2_atts) or _is_cut_edge(e1_atts):
                        e_atts["num"] = (
                            f"-1𝑖*(spenso::g(spenso::coad(8,hedge({_parse_port(e.get_destination())})),spenso::coad(8,hedge({_parse_port(e.get_source())})))*spenso::g(spenso::mink(4,hedge({_parse_port(e.get_destination())})),spenso::mink(4,hedge({_parse_port(e.get_source())})))-{overall_sign2}*1/Q({e_atts['id']},0)*Qp({e_atts['id']},spenso::mink(4,hedge({_parse_port(vertices2[1])})))*Q(1000,spenso::mink(4,hedge({_parse_port(vertices2[0])})))*spenso::g(spenso::coad(8,hedge({_parse_port(e.get_destination())})),spenso::coad(8,hedge({_parse_port(e.get_source())}))))"
                        )
                        print("- IN CUT -" * 10)
                    else:
                        e_atts["num"] = (
                            f"-1𝑖*(spenso::g(spenso::coad(8,hedge({_parse_port(e.get_destination())})),spenso::coad(8,hedge({_parse_port(e.get_source())})))*spenso::g(spenso::mink(4,hedge({_parse_port(e.get_destination())})),spenso::mink(4,hedge({_parse_port(e.get_source())})))+(-1)*(1)*1/({indices2[0][1]}*Q({indices2[0][0]},0)+{indices2[1][1]}*Q({indices2[1][0]},0))*Qp({e_atts['id']},spenso::mink(4,hedge({_parse_port(vertices2[1])})))*Q(1000,spenso::mink(4,hedge({_parse_port(vertices2[0])})))*spenso::(spenso::coad(8,hedge({_parse_port(e.get_destination())})),spenso::coad(8,hedge({_parse_port(e.get_source())}))))"
                        )
                        print("- NOT IN CUT -" * 10)
                        print("sign 2")
                        print(overall_sign2)
                        print(e_atts["num"])

        return cut_graph

    # Derive cff, set the lmb so that the loop momentum coincides with the photon (for DY), and
    # derive the approximated representation.

    def get_integrand(self, cut_graph):

        # FIX: cut graph logic and overwriting
        orig_cut_graph = deepcopy(cut_graph)
        skip_threshold_cts = False
        gluonic_t_channel = True

        # emr_integrand_tmp = self.emr_processor.get_integrand(orig_cut_graph)

        if len(cut_graph.final_cut) > 1 and self.name == "DY":
            lmb_choice = []
            theta_flag = True
            for e in cut_graph.final_cut:
                e_atts = e.get_attributes()
                if _strip_quotes(str(e_atts["particle"])) == "a":
                    lmb_choice.append(e_atts["id"])
            cut_graph.graph = change_routing(cut_graph.graph, lmb_choice)
            orig_cut_graph.graph = change_routing(orig_cut_graph.graph, lmb_choice)

        if len(cut_graph.final_cut) == 1 and self.name == "DY":
            theta_flag = True
            lmb_choice = []
            for e in cut_graph.graph.get_edges():
                e_atts = e.get_attributes()
                if _strip_quotes(str(e_atts["particle"])) == "g":
                    lmb_choice.append(e_atts["id"])
            cut_graph.graph = change_routing(cut_graph.graph, lmb_choice)
            orig_cut_graph.graph = change_routing(orig_cut_graph.graph, lmb_choice)

        if self.name == "tt~":
            lmb_choice = []

            if self.L == 1:
                theta_flag = True
                lmb_choice = [3]

            if self.L == 2:
                ## only for q g channel
                if self.channel == (1, 0) or self.channel == (0, 1):
                    theta_flag = True
                    lmb_choice = [
                        (
                            e.get_attributes()["id"],
                            _strip_quotes(str(e.get_attributes()["particle"])),
                        )
                        for e in cut_graph.final_cut
                    ]

                    new_lmb_choice = []
                    for id, part in lmb_choice:
                        if part in ["d", "d~", "g"]:
                            new_lmb_choice.append(id)

                    massives = []
                    for id, part in lmb_choice:
                        if part not in ["d", "d~", "g"]:
                            massives.append(id)

                    new_lmb_choice.extend(sorted(massives))

                    lmb_choice = new_lmb_choice[:-1]

                if self.channel == (1, -1) or self.channel == (-1, 1):
                    base_graph_name = _strip_quotes(
                        str(cut_graph.graph.get("base_graph_name"))
                    )

                    if base_graph_name in [
                        "GL00",
                        "GL01",
                        "GL02",
                        "GL03",
                        "GL05",
                        "GL13",
                        "GL18",
                    ]:
                        theta_flag = False
                        lmb_choice = [6, 3]

                    if base_graph_name in [
                        "GL17",
                    ]:
                        theta_flag = False
                        lmb_choice = [3, 6]
                        skip_threshold_cts = True

                    if base_graph_name in ["GL04", "GL15"]:
                        theta_flag = False
                        lmb_choice = [8, 3]

                    if base_graph_name in ["GL06", "GL08", "GL11"]:
                        theta_flag = False
                        lmb_choice = [7, 3]

                    if base_graph_name in ["GL07"]:
                        theta_flag = False
                        gluonic_t_channel = False
                        lmb_choice = [7, 3]

                    if base_graph_name in ["GL09"]:
                        theta_flag = True
                        lmb_choice = [8, 3]

                    if base_graph_name == "GL10":
                        lmb_choice = []
                        theta_flag = False
                        t_count = 0
                        d_count = 0
                        for e in cut_graph.graph.get_edges():
                            e_atts = e.get_attributes()
                            e_part = _strip_quotes(str(e_atts["particle"]))
                            is_cut = _strip_quotes(str(e_atts.get("is_cut", "")))

                            if (e_part == "t" or e_part == "t~") and t_count == 0:
                                lmb_choice.append(e_atts["id"])
                                t_count += 1
                            elif (
                                (e_part == "d" or e_part == "d~")
                                and d_count == 0
                                and is_cut == "0"
                            ):
                                lmb_choice.append(e_atts["id"])
                                d_count += 1

            print(lmb_choice)
            # lmb_choice = [7, 2]
            # lmb_choice = [2, 7]

            cut_graph.graph = change_routing(cut_graph.graph, lmb_choice)
            orig_cut_graph.graph = change_routing(orig_cut_graph.graph, lmb_choice)

        print("heyyy")
        if gluonic_t_channel:
            cut_graph = self.modify_t_channel_gluon_numerator(cut_graph)

        print(cut_graph.graph)
        emr_integrand = self.emr_processor.get_integrand(cut_graph)

        # print(emr_integrand)

        loop_integrand, raised_cut, is_final_raised = self.eliminate_raised_cuts(
            emr_integrand, cut_graph
        )

        uv_approximator = UltraVioletSubtraction(
            loop_integrand,
            deepcopy(cut_graph),
            self.L,
            self.emr_processor,
            deepcopy(orig_cut_graph),
            disable_integrated_uv_cts=self.disable_integrated_uv_cts,
        )
        uv_ct = uv_approximator.construct_uv_counter_terms()

        threshold_approximator = ThresholdSubtractor(
            deepcopy(orig_cut_graph), self.params, self.name, self.L, theta_flag
        )
        threshold_cts = threshold_approximator.construct_threshold_counter_terms()

        if skip_threshold_cts:
            threshold_cts = []
        # print("this emr")
        # print(emr_integrand)

        loop_integrand = self.leading_virtuality_expansion(
            loop_integrand, cut_graph, raised_cut
        )

        print(len(loop_integrand))

        print("*** " * 10)
        # for lp in loop_integrand:
        #    print(lp.integrand)

        if is_final_raised:
            for ct in uv_ct:
                ct.t_derivative = is_final_raised
            for lp in loop_integrand:
                lp.t_derivative = is_final_raised
            for ct in threshold_cts:
                ct.t_derivative = is_final_raised

        loop_integrand = (
            loop_integrand + threshold_cts + uv_ct  #   # + soft_t_channel_gluon_cts
        )

        print("returning final integrand")

        return loop_integrand
