import json
import os
from copy import deepcopy
from functools import lru_cache

import pydot
from symbolica import AtomType, E, Expression, S  # pyright: ignore
from symbolica.community.idenso import (  # pyright: ignore
    simplify_gamma,
    simplify_metrics,
)

from processes.dy.dy_graph_utils import _base_node, _strip_quotes

pjoin = os.path.join

UV_INTEGRATED_COUNTERTERMS_PATH = pjoin(
    os.path.dirname(__file__),
    "uv_integrated_counterterms",
    "generated_integrands",
    "uv_integrated_counterterms_1L.json",
)


def Es(expr: str) -> Expression:
    return E(expr.replace('"', ""), default_namespace="gammalooprs")


def _normalise_uv_particle(particle: str) -> str:
    particle = _strip_quotes(str(particle))
    if particle == "ghG~":
        return "ghG"
    if particle.endswith("~"):
        return particle[:-1]
    return particle


def _normalise_uv_external_particle(particle: str) -> str:
    particle = _strip_quotes(str(particle))
    if particle == "ghG~":
        return "ghG"
    return particle


UV_PARTICLE_ORDER = {
    "d": 0,
    "d~": 1,
    "g": 2,
    "ghG": 3,
    "t": 4,
}


def _uv_particle_sort_key(particle: str) -> tuple[int, str]:
    return UV_PARTICLE_ORDER.get(particle, 100), particle


def _uv_particle_multiset(particles, normalizer) -> tuple[str, ...]:
    return tuple(sorted((normalizer(p) for p in particles), key=_uv_particle_sort_key))


def _external_particles_from_process(process: str) -> list[str]:
    if process == "g > g":
        return ["g", "g"]
    if process == "d > d":
        return ["d", "d"]
    if process == "d d~ > g":
        return ["d", "d~", "g"]
    return []


def _expanded_add_terms(expr: Expression) -> list[Expression]:
    expr = expr.expand()
    if bool(expr.is_type(AtomType.Add)):
        return list(expr)
    return [expr]


def _finite_uv_integrated_expression(expr: Expression) -> Expression:
    finite = E("0")
    for term in _expanded_add_terms(expr):
        if "ε" not in str(term):
            finite += term

    return (
        ((finite / E("𝑖")).expand())
        .replace(E("muvsq"), E("mUV") ** 2)
        .replace(E("𝜋"), E("3.141592653589793238462643383279502884"))
    )


@lru_cache(maxsize=1)
def _uv_integrated_counterterm_table():
    if not os.path.exists(UV_INTEGRATED_COUNTERTERMS_PATH):
        return {}

    with open(UV_INTEGRATED_COUNTERTERMS_PATH, encoding="utf-8") as handle:
        entries = json.load(handle)

    counterterms = {}
    for entry in entries:
        external_particles = entry.get("external_particles") or _external_particles_from_process(
            entry["process"]
        )
        internal_particles = entry.get(
            "internal_particles_multiset", entry["internal_particles"]
        )
        key = (
            entry["process"],
            _uv_particle_multiset(
                external_particles, _normalise_uv_external_particle
            ),
            _uv_particle_multiset(internal_particles, _normalise_uv_particle),
        )
        counterterms.setdefault(key, []).append(
            {
                "entry": entry,
                "finite_counterterm": _finite_uv_integrated_expression(
                    E(entry["vakint_analytic_total"].replace('"', ""))
                ),
            }
        )
    return counterterms


def _uv_process_from_external_particles(particles: list[str]) -> str | None:
    external_multiset = _uv_particle_multiset(
        particles, _normalise_uv_external_particle
    )
    if external_multiset == ("g", "g"):
        return "g > g"
    if (
        len(particles) == 2
        and _uv_particle_multiset(particles, _normalise_uv_particle) == ("d", "d")
    ):
        return "d > d"
    if external_multiset == ("d", "d~", "g"):
        return "d d~ > g"
    return None


def _direct_external_momentum_label(lmb_rep: str | None) -> str | None:
    if lmb_rep is None:
        return None
    expr = Es(lmb_rep)
    if not bool(expr.is_type(AtomType.Fn)):
        return None
    if expr.get_name().rsplit("::", 1)[-1] != "P" or len(expr) != 2:
        return None
    label = expr[0].to_canonical_string()
    if not label.isdigit():
        return None
    return str(int(label) + 1)


def _edge_slot_from_vakint_index(index: int) -> str | None:
    if index < 1_000_000:
        return None
    shifted = index - 1_000_000
    edge_id = shifted // 1_000
    direction = shifted % 1_000
    return f"mink(4,edge({edge_id},{direction}))"


def _slot_from_vakint_index(
    index_expr: Expression, reference_to_actual_port: dict[str, str]
) -> str | None:
    index = index_expr.to_canonical_string()
    if not index.isdigit():
        return None
    index_int = int(index)
    edge_slot = _edge_slot_from_vakint_index(index_int)
    if edge_slot is not None:
        return edge_slot
    if index_int < 11:
        return None
    reference_port = str(index_int - 11)
    actual_port = reference_to_actual_port.get(reference_port)
    if actual_port is not None:
        return f"mink(4,hedge({actual_port}))"
    return f"mink(4,hedge({reference_port}))"


def _remap_vakint_metric_slots(
    expr: Expression, reference_to_actual_port: dict[str, str]
) -> Expression:
    for match in list(expr.match(E("g(left_,right_)"))):
        left = match[S("left_")]
        right = match[S("right_")]
        left_slot = _slot_from_vakint_index(left, reference_to_actual_port)
        right_slot = _slot_from_vakint_index(right, reference_to_actual_port)
        if left_slot is None or right_slot is None:
            continue
        expr = expr.replace(
            E(f"g({left.to_canonical_string()},{right.to_canonical_string()})"),
            E(f"g({left_slot},{right_slot})"),
        )
    return expr


def _remap_vakint_momenta(
    expr: Expression,
    momentum_source_edges: dict[str, str],
    reference_to_actual_port: dict[str, str],
) -> Expression:
    for source_label, edge_id in momentum_source_edges.items():
        expr = expr.replace(
            E(f"p({source_label},dot_dummy_ind(x_))^2"),
            E(f"Q({edge_id},mink(4,rho))*Q({edge_id},mink(4,rho))"),
            repeat=True,
        )

    for match in list(expr.match(E("p(source_,slot_)"))):
        source_label = match[S("source_")].to_canonical_string()
        edge_id = momentum_source_edges.get(source_label)
        if edge_id is None:
            continue
        slot = _slot_from_vakint_index(match[S("slot_")], reference_to_actual_port)
        if slot is None:
            continue
        expr = expr.replace(
            E(
                f"p({source_label},{match[S('slot_')].to_canonical_string()})"
            ),
            E(f"Q({edge_id},{slot})"),
        )
    return expr


def _remap_external_hedges(
    expr: Expression, reference_to_actual_port: dict[str, str]
) -> Expression:
    tmp_ports = {}
    for i, reference_port in enumerate(reference_to_actual_port):
        tmp_port = str(900_000 + i)
        tmp_ports[tmp_port] = reference_to_actual_port[reference_port]
        expr = expr.replace(E(f"hedge({reference_port})"), E(f"hedge({tmp_port})"))

    for tmp_port, actual_port in tmp_ports.items():
        expr = expr.replace(E(f"hedge({tmp_port})"), E(f"hedge({actual_port})"))
    return expr


def _contract_lorentz_metrics(expr: Expression) -> Expression:
    replacements = [
        (
            E("g(mink(4,left_),mink(4,right_))*gamma(a_,b_,mink(4,right_))"),
            E("gamma(a_,b_,mink(4,left_))"),
        ),
        (
            E("g(mink(4,left_),mink(4,right_))*gamma(a_,b_,mink(4,left_))"),
            E("gamma(a_,b_,mink(4,right_))"),
        ),
        (
            E("g(mink(4,left_),mink(4,right_))*Q(edge_,mink(4,right_))"),
            E("Q(edge_,mink(4,left_))"),
        ),
        (
            E("g(mink(4,left_),mink(4,right_))*Q(edge_,mink(4,left_))"),
            E("Q(edge_,mink(4,right_))"),
        ),
        (
            E(
                "g(mink(4,left_),mink(4,right_))"
                "*g(mink(4,right_),mink(4,other_))"
            ),
            E("g(mink(4,left_),mink(4,other_))"),
        ),
        (
            E(
                "g(mink(4,left_),mink(4,right_))"
                "*g(mink(4,left_),mink(4,other_))"
            ),
            E("g(mink(4,right_),mink(4,other_))"),
        ),
    ]

    for _ in range(8):
        previous = expr.to_canonical_string()
        for pattern, replacement in replacements:
            expr = expr.replace(pattern, replacement, repeat=True)
        if expr.to_canonical_string() == previous:
            break
    return expr.expand()


def _match_uv_external_edges(reference_edges, surviving_ports, boundary_edges):
    if not reference_edges or len(reference_edges) != len(surviving_ports):
        return None

    actual_edges = []
    for port, edge in zip(surviving_ports, boundary_edges, strict=True):
        actual_edges.append(
            {
                "port": port,
                "edge": edge,
                "particle": _normalise_uv_external_particle(
                    _strip_quotes(str(edge.get_attributes().get("particle", "")))
                ),
            }
        )

    reference_particles = [
        _normalise_uv_external_particle(edge.get("particle", ""))
        for edge in reference_edges
    ]
    if len(reference_particles) == len(set(reference_particles)):
        actual_by_particle = {edge["particle"]: edge for edge in actual_edges}
        if set(actual_by_particle) != set(reference_particles):
            return None
        return [
            {
                "reference": reference_edge,
                "actual_port": actual_by_particle[
                    _normalise_uv_external_particle(reference_edge.get("particle", ""))
                ]["port"],
                "actual_edge": actual_by_particle[
                    _normalise_uv_external_particle(reference_edge.get("particle", ""))
                ]["edge"],
            }
            for reference_edge in reference_edges
        ]

    return [
        {
            "reference": reference_edge,
            "actual_port": actual_edge["port"],
            "actual_edge": actual_edge["edge"],
        }
        for reference_edge, actual_edge in zip(reference_edges, actual_edges, strict=True)
    ]


def _remap_uv_integrated_expression(expr: Expression, external_mappings) -> Expression:
    reference_to_actual_port = {
        str(mapping["reference"]["port"]): str(mapping["actual_port"])
        for mapping in external_mappings
    }
    momentum_source_edges = {}
    for mapping in external_mappings:
        source_label = _direct_external_momentum_label(
            mapping["reference"].get("lmb_rep")
        )
        if source_label is None or source_label in momentum_source_edges:
            continue
        momentum_source_edges[source_label] = _strip_quotes(
            str(mapping["actual_edge"].get_attributes()["id"])
        )

    expr = _remap_vakint_metric_slots(expr, reference_to_actual_port)
    expr = _remap_vakint_momenta(expr, momentum_source_edges, reference_to_actual_port)
    expr = _remap_external_hedges(expr, reference_to_actual_port)
    expr = _contract_lorentz_metrics(expr)
    expr = simplify_gamma(simplify_metrics(expr)).expand()
    return _contract_lorentz_metrics(expr)


def _format_uv_integrated_numerator(expr: Expression) -> str:
    expr = expr.replace(
        E("Q(edge_,mink(4,rho))^2"),
        E("Q(edge_,mink(4,rho))*Q(edge_,mink(4,rho))"),
        repeat=True,
    )

    args__ = S("args__")
    spenso_functions = ["g", "mink", "coad", "cof", "dind", "bis", "gamma", "t"]
    for function_name in spenso_functions:
        placeholder = S(f"uv_spenso_{function_name}")
        expr = expr.replace(
            S(function_name)(args__),
            placeholder(args__),
            allow_new_wildcards_on_rhs=True,
        )

    expr_str = " ".join(expr.to_canonical_string().split())
    for prefix, replacement in (
        ("python::{}::", ""),
        ("gammalooprs::{}::", ""),
        ("symbolica::{}::", ""),
    ):
        expr_str = expr_str.replace(prefix, replacement)
    for function_name in spenso_functions:
        expr_str = expr_str.replace(
            f"uv_spenso_{function_name}(", f"spenso::{function_name}("
        )
    return expr_str.replace("+-", "-")


def construct_integrated_counter_term(
    subtraction,
    cycle,
    _dod,
    routed_cut_graph_cls,
    routed_integrand_cls,
):
    if subtraction.emr_processor is None or subtraction.L != 2:
        return None

    contraction_cut_graph = (
        subtraction.integrated_cut_graph
        if subtraction.integrated_cut_graph is not None
        else subtraction.cut_graph
    )

    cycle_ids = {_strip_quotes(str(e.get_attributes()["id"])) for e in cycle}
    cycle_nodes = {_base_node(e.get_source()) for e in cycle} | {
        _base_node(e.get_destination()) for e in cycle
    }

    surviving_ports = []
    surviving_port_edges = {}
    for e in contraction_cut_graph.graph.get_edges():
        e_id = _strip_quotes(str(e.get_attributes()["id"]))
        if e_id in cycle_ids:
            continue
        for endpoint in [str(e.get_source()), str(e.get_destination())]:
            if _base_node(endpoint) in cycle_nodes and ":" in endpoint:
                port = endpoint.split(":", 1)[1]
                if port not in surviving_ports:
                    surviving_ports.append(port)
                surviving_port_edges.setdefault(port, e)

    if len(surviving_ports) not in {2, 3}:
        return None

    boundary_edges = [surviving_port_edges[port] for port in surviving_ports]
    external_particles = [
        _strip_quotes(str(edge.get_attributes().get("particle", "")))
        for edge in boundary_edges
    ]
    uv_process = _uv_process_from_external_particles(external_particles)
    if uv_process is None:
        return None

    internal_particles = _uv_particle_multiset(
        [
            _strip_quotes(str(edge.get_attributes().get("particle", "")))
            for edge in cycle
        ],
        _normalise_uv_particle,
    )
    external_particle_multiset = _uv_particle_multiset(
        external_particles, _normalise_uv_external_particle
    )
    candidates = _uv_integrated_counterterm_table().get(
        (uv_process, external_particle_multiset, internal_particles), []
    )
    if not candidates:
        return None

    if len(candidates) > 1:
        actual_sequence = tuple(
            _normalise_uv_particle(
                _strip_quotes(str(edge.get_attributes().get("particle", "")))
            )
            for edge in sorted(
                cycle,
                key=lambda cycle_edge: int(
                    _strip_quotes(str(cycle_edge.get_attributes()["id"]))
                ),
            )
        )
        candidates = [
            candidate
            for candidate in candidates
            if tuple(
                _normalise_uv_particle(p)
                for p in candidate["entry"].get("internal_particle_sequence", [])
            )
            == actual_sequence
        ]
    if len(candidates) != 1:
        return None

    counterterm_entry = candidates[0]["entry"]
    finite_counterterm = candidates[0]["finite_counterterm"]
    if finite_counterterm is None:
        return None
    if finite_counterterm.to_canonical_string() == "0":
        return None

    external_mappings = _match_uv_external_edges(
        counterterm_entry.get("external_edges", []),
        surviving_ports,
        boundary_edges,
    )
    if external_mappings is None:
        return None

    tensor_numerator = _format_uv_integrated_numerator(
        _remap_uv_integrated_expression(
            finite_counterterm,
            external_mappings,
        )
    )

    contracted_graph = pydot.Dot(
        graph_type="digraph",
        name=f"{contraction_cut_graph.graph.get_name()}_uvint",
    )
    for key, value in contraction_cut_graph.graph.get_attributes().items():
        contracted_graph.set(key, value)
    for node in contraction_cut_graph.graph.get_nodes():
        node_name = _strip_quotes(str(node.get_name()))
        if node_name in cycle_nodes or node_name in ["node", "edge", "graph"]:
            continue
        contracted_graph.add_node(deepcopy(node))
    contracted_graph.add_node(
        pydot.Node(
            "UVCT",
            dod="0",
            int_id="UV_CONTRACT",
            num=tensor_numerator,
        )
    )

    for edge in contraction_cut_graph.graph.get_edges():
        e_id = _strip_quotes(str(edge.get_attributes()["id"]))
        if e_id in cycle_ids:
            continue
        src = str(edge.get_source())
        dst = str(edge.get_destination())
        if _base_node(src) in cycle_nodes:
            src = f"UVCT:{src.split(':', 1)[1]}" if ":" in src else "UVCT"
        if _base_node(dst) in cycle_nodes:
            dst = f"UVCT:{dst.split(':', 1)[1]}" if ":" in dst else "UVCT"
        contracted_graph.add_edge(
            pydot.Edge(src, dst, **deepcopy(edge.get_attributes()))
        )

    contracted_edges_by_id = {
        _strip_quotes(str(edge.get_attributes()["id"])): edge
        for edge in contracted_graph.get_edges()
    }

    def contracted_cut_edges(cut_edges):
        return [
            contracted_edges_by_id[_strip_quotes(str(edge.get_attributes()["id"]))]
            for edge in cut_edges
            if _strip_quotes(str(edge.get_attributes()["id"]))
            in contracted_edges_by_id
        ]

    contracted_cut_graph = routed_cut_graph_cls(
        contracted_graph,
        contracted_cut_edges(contraction_cut_graph.initial_cut),
        contracted_cut_edges(contraction_cut_graph.final_cut),
        contraction_cut_graph.partition,
    )
    contracted_emr = subtraction.emr_processor.get_integrand(
        deepcopy(contracted_cut_graph)
    )
    routed_integrand = subtraction.replace_energies(contracted_emr, contracted_graph)
    routed_integrand = subtraction.route_integrand(routed_integrand, contracted_graph)

    for edge in contracted_graph.get_edges():
        e_atts = edge.get_attributes()
        e_particle = _strip_quotes(str(e_atts["particle"]))
        if e_particle in ["d", "d~", "g", "ghG", "ghG~"]:
            mass = E("0")
        else:
            mass = E(f"m({e_particle})")
        routed_integrand = routed_integrand.replace(E(f"m({e_atts['id']})"), mass)

    uv_loop_momentum = min(
        cycle, key=lambda e: int(_strip_quotes(str(e.get_attributes()["id"])))
    )
    uv_loop_atts = uv_loop_momentum.get_attributes()
    uv_loop_indices = [
        i
        for i in range(subtraction.L)
        if uv_loop_atts.get(f"routing_k{i}") not in [None, "0"]
    ]
    if len(uv_loop_indices) != 1:
        return None

    routed_integrand *= E("1") / (
        subtraction.sp3D(
            E(f"k({uv_loop_indices[0]})"), E(f"k({uv_loop_indices[0]})")
        )
        + E("mUV") ** 2
    )

    return routed_integrand_cls(
        routed_integrand,
        contracted_cut_graph,
        [],
        contracted_emr,
        "uv_int",
        "uv",
    )
