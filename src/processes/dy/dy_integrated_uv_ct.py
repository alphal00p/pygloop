import json
import os
import re
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache

import pydot
from symbolica import AtomType, E, Expression, Replacement, S  # pyright: ignore
from symbolica.community.idenso import (  # pyright: ignore
    simplify_color,
    simplify_gamma,
    simplify_metrics,
)

from processes.dy.dy_graph_utils import _base_node, _strip_quotes

pjoin = os.path.join

UV_INTEGRATED_COUNTERTERMS_PATH = pjoin(
    os.path.dirname(__file__),
    "table_uv_ct.json",
)


def Es(expr: str) -> Expression:
    return E(expr.replace('"', ""), default_namespace="gammalooprs")


def Eu(expr: str) -> Expression:
    return E(expr.replace('"', ""))


def _normalise_uv_particle(particle: str) -> str:
    particle = _strip_quotes(str(particle))
    if particle in {"gh", "gh~"}:
        return "ghG"
    if particle == "ghG~":
        return "ghG"
    if particle.endswith("~"):
        return particle[:-1]
    return particle


def _normalise_uv_external_particle(particle: str) -> str:
    particle = _strip_quotes(str(particle))
    if particle in {"gh", "gh~"}:
        return "ghG"
    if particle == "ghG~":
        return "ghG"
    if particle == "t~":
        return "t"
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
    if process == "t > t":
        return ["t", "t"]
    if process == "d d~ > g":
        return ["d", "d~", "g"]
    if process == "t t~ > g":
        return ["t", "t~", "g"]
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


def _compact_external_particle(entry_particle) -> str:
    if isinstance(entry_particle, str):
        return entry_particle
    if (
        isinstance(entry_particle, list)
        and len(entry_particle) == 2
        and isinstance(entry_particle[0], str)
    ):
        return entry_particle[0]
    raise ValueError(f"invalid compact external particle entry: {entry_particle!r}")


def _compact_external_labels(entry_particle) -> list[str]:
    if not isinstance(entry_particle, list) or len(entry_particle) != 2:
        return []
    labels = entry_particle[1]
    if not isinstance(labels, list):
        raise ValueError(f"invalid compact external labels: {entry_particle!r}")
    return [str(label) for label in labels]


def _compact_process_from_entry(entry: dict) -> str:
    if "process" in entry:
        return entry["process"]
    particles = [_compact_external_particle(p) for p in entry["external_particles"]]
    process = _uv_process_from_external_particles(particles)
    if process is None:
        raise ValueError(
            f"could not infer UV process from external particles: {particles!r}"
        )
    return process


def _compact_external_edges(entry: dict) -> list[dict[str, object]]:
    return [
        {
            "particle": _compact_external_particle(entry_particle),
            "labels": _compact_external_labels(entry_particle),
            "port": str(index),
        }
        for index, entry_particle in enumerate(entry["external_particles"])
    ]


def _compact_integrated_expression(expr: str) -> Expression:
    expr = expr.replace("**", "^")
    expr = expr.replace("Log", "log")
    expr = re.sub(r"Power\(([^(),]+),([^(),]+)\)", r"(\1^\2)", expr)
    expr = expr.replace("Pi", "3.141592653589793238462643383279502884")
    parsed = Eu(expr)
    return parsed.replace(Eu("muvsq"), Eu("mUV") ** 2)


def _counterterm_from_generated_entry(entry: dict) -> dict[str, object]:
    return {
        "entry": entry,
        "finite_counterterm": _finite_uv_integrated_expression(
            E(entry["vakint_analytic_total"].replace('"', ""))
        ),
        "source": "generated",
    }


def _counterterm_from_compact_entry(entry: dict) -> dict[str, object]:
    external_edges = _compact_external_edges(entry)
    normalized_entry = dict(entry)
    normalized_entry["process"] = _compact_process_from_entry(entry)
    normalized_entry["external_edges"] = external_edges
    normalized_entry["external_particles"] = [
        edge["particle"] for edge in external_edges
    ]
    normalized_entry["internal_particle_sequence"] = entry.get(
        "internal_particle_sequence", entry["internal_particles"]
    )
    normalized_entry["internal_particles_multiset"] = entry.get(
        "internal_particles_multiset", entry["internal_particles"]
    )
    return {
        "entry": normalized_entry,
        "finite_counterterm": _compact_integrated_expression(entry["integrated_ct"]),
        "source": "compact",
    }


def _load_uv_counterterm_entry(entry: dict) -> dict[str, object]:
    if "vakint_analytic_total" in entry:
        return _counterterm_from_generated_entry(entry)
    if "integrated_ct" in entry:
        return _counterterm_from_compact_entry(entry)
    raise ValueError(
        "UV integrated counterterm entry must contain either "
        "'vakint_analytic_total' or 'integrated_ct'"
    )


@lru_cache(maxsize=1)
def _uv_integrated_counterterm_table():
    if not os.path.exists(UV_INTEGRATED_COUNTERTERMS_PATH):
        return {}

    with open(UV_INTEGRATED_COUNTERTERMS_PATH, encoding="utf-8") as handle:
        entries = json.load(handle)

    counterterms = {}
    for entry in entries:
        counterterm = _load_uv_counterterm_entry(entry)
        entry = counterterm["entry"]
        external_particles = entry.get(
            "external_particles"
        ) or _external_particles_from_process(entry["process"])
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
        counterterms.setdefault(key, []).append(counterterm)
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
    if (
        len(particles) == 2
        and _uv_particle_multiset(particles, _normalise_uv_particle) == ("t", "t")
    ):
        return "t > t"
    if external_multiset == ("d", "d~", "g"):
        return "d d~ > g"
    if (
        len(particles) == 3
        and _uv_particle_multiset(particles, _normalise_uv_particle)
        == ("g", "t", "t")
    ):
        return "t t~ > g"
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
    replacements = []
    for prefix in ["", "spenso::"]:
        g = f"{prefix}g"
        mink = f"{prefix}mink"
        gamma = f"{prefix}gamma"
        replacements.extend(
            [
                (
                    E(
                        f"{g}({mink}(4,left_),{mink}(4,right_))"
                        f"*{gamma}(a_,b_,{mink}(4,right_))"
                    ),
                    E(f"{gamma}(a_,b_,{mink}(4,left_))"),
                ),
                (
                    E(
                        f"{g}({mink}(4,left_),{mink}(4,right_))"
                        f"*{gamma}(a_,b_,{mink}(4,left_))"
                    ),
                    E(f"{gamma}(a_,b_,{mink}(4,right_))"),
                ),
                (
                    E(
                        f"{g}({mink}(4,left_),{mink}(4,right_))"
                        f"*Q(edge_,{mink}(4,right_))"
                    ),
                    E(f"Q(edge_,{mink}(4,left_))"),
                ),
                (
                    E(
                        f"{g}({mink}(4,left_),{mink}(4,right_))"
                        f"*Q(edge_,{mink}(4,left_))"
                    ),
                    E(f"Q(edge_,{mink}(4,right_))"),
                ),
                (
                    E(
                        f"{g}({mink}(4,left_),{mink}(4,right_))"
                        f"*{g}({mink}(4,right_),{mink}(4,other_))"
                    ),
                    E(f"{g}({mink}(4,left_),{mink}(4,other_))"),
                ),
                (
                    E(
                        f"{g}({mink}(4,left_),{mink}(4,right_))"
                        f"*{g}({mink}(4,left_),{mink}(4,other_))"
                    ),
                    E(f"{g}({mink}(4,right_),{mink}(4,other_))"),
                ),
            ]
        )

    for _ in range(8):
        previous = expr.to_canonical_string()
        for pattern, replacement in replacements:
            expr = expr.replace(pattern, replacement, repeat=True)
        if expr.to_canonical_string() == previous:
            break
    return expr.expand()


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
        expr = expr.replace(
            old(args__),
            new(args__),
            allow_new_wildcards_on_rhs=True,
        )
        atom_repls.append(Replacement(old, new))

    if atom_repls:
        expr = expr.replace_multiple(atom_repls)

    return expr


def _contract_lorentz_momentum_pairs(expr: Expression) -> Expression:
    expr = expr.replace(
        E("Q(y_,mink(4,x_))") * E("Q(z_,mink(4,x_))"),
        E("sp(y_,z_)"),
        repeat=True,
    )
    expr = expr.replace(
        E("Qp(y_,mink(4,x_))") * E("Q(z_,mink(4,x_))"),
        E("spp(qp(y_),z_)"),
        repeat=True,
    )
    return expr.replace(
        E("Qp(y_,mink(4,x_))") * E("Qp(z_,mink(4,x_))"),
        E("spp(qp(y_),qp(z_))"),
        repeat=True,
    )


def _raw_uv_int_graph_numerator(graph) -> Expression:
    numerator = E("1")
    for node in graph.get_nodes():
        if node.get_name() in ["edge", "node"]:
            continue
        node_numerator = node.get("num")
        if node_numerator:
            numerator *= Es(node_numerator)
    for edge in graph.get_edges():
        edge_numerator = edge.get("num")
        if edge_numerator:
            numerator *= Es(edge_numerator)
    return numerator


def _graph_without_numerators(graph):
    graph_without_numerators = deepcopy(graph)
    for node in graph_without_numerators.get_nodes():
        node.get_attributes().pop("num", None)
    for edge in graph_without_numerators.get_edges():
        edge.get_attributes().pop("num", None)
    return graph_without_numerators


def _close_uv_int_tensor_numerator(expr: Expression) -> Expression:
    for _ in range(8):
        previous = expr.to_canonical_string()
        expr = _contract_lorentz_metrics(expr)
        expr = simplify_metrics(simplify_gamma(simplify_color(expr))).expand()
        expr = _contract_lorentz_metrics(expr)
        if expr.to_canonical_string() == previous:
            break
    expr = _strip_namespaces_structurally(expr)
    expr = _contract_lorentz_metrics(expr)
    expr = _contract_lorentz_momentum_pairs(expr).expand()
    return expr


def _open_lorentz_momenta(expr: Expression) -> list[str]:
    momenta = []
    for pattern in [
        E("Q(edge_,mink(4,slot_))"),
        E("Qp(edge_,mink(4,slot_))"),
    ]:
        for match in expr.match(pattern):
            edge = match[S("edge_")].to_canonical_string()
            slot = match[S("slot_")].to_canonical_string()
            momenta.append(f"{edge}:{slot}")
    return list(dict.fromkeys(momenta))


def _uv_int_numerator_factorisation(graph):
    closed_numerator = _close_uv_int_tensor_numerator(
        _raw_uv_int_graph_numerator(graph)
    )
    return _graph_without_numerators(graph), closed_numerator


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
    if (
        len(reference_particles) == 3
        and reference_particles.count("t") == 2
        and reference_particles.count("g") == 1
    ):
        actual_t_edges = [edge for edge in actual_edges if edge["particle"] == "t"]
        actual_g_edges = [edge for edge in actual_edges if edge["particle"] == "g"]
        if len(actual_t_edges) != 2 or len(actual_g_edges) != 1:
            return None

        top_edges = iter(actual_t_edges)
        external_mappings = []
        for reference_edge, reference_particle in zip(
            reference_edges, reference_particles, strict=True
        ):
            actual_edge = actual_g_edges[0] if reference_particle == "g" else next(top_edges)
            external_mappings.append(
                {
                    "reference": reference_edge,
                    "actual_port": actual_edge["port"],
                    "actual_edge": actual_edge["edge"],
                }
            )
        return external_mappings

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


_COMPACT_FREE_LORENTZ_LABELS = ("mu", "nu", "rho", "sigma", "alpha", "beta")


def _port_suffix(endpoint: object) -> str | None:
    endpoint = str(endpoint)
    if ":" not in endpoint:
        return None
    return endpoint.split(":", 1)[1]


def _compact_fermion_color_slot(edge, actual_port: str) -> str:
    if _port_suffix(edge.get_destination()) == actual_port:
        return f"cof(3,hedge({actual_port}))"
    return f"dind(cof(3,hedge({actual_port})))"


def _external_routing_component(edge, key: str) -> Fraction:
    value = _strip_quotes(str(edge.get_attributes().get(key, "0"))).strip()
    if value in {"", "+0", "-0"}:
        value = "0"
    return Fraction(value)


def _external_edges_have_opposite_routing(edge_a, edge_b) -> bool:
    edge_a_atts = edge_a.get_attributes()
    edge_b_atts = edge_b.get_attributes()
    routing_keys = sorted(
        {
            key
            for key in set(edge_a_atts) | set(edge_b_atts)
            if key.startswith("routing_k") or key in {"routing_p1", "routing_p2"}
        }
    )
    if not routing_keys:
        return False
    return all(
        _external_routing_component(edge_a, key)
        == -_external_routing_component(edge_b, key)
        for key in routing_keys
    )


def _compact_remapping_maps(external_mappings):
    index_slots = {}
    momentum_edges = {}
    momentum_edge_by_slot = {}
    momentum_actual_edges = {}
    momentum_slots = {}

    for mapping in external_mappings:
        reference = mapping["reference"]
        labels = reference.get("labels", [])
        if not labels:
            continue
        particle = _normalise_uv_external_particle(reference.get("particle", ""))
        actual_port = str(mapping["actual_port"])
        edge_id = _strip_quotes(str(mapping["actual_edge"].get_attributes()["id"]))

        if particle == "g" and len(labels) >= 3:
            index_slots[str(labels[0])] = f"mink(4,hedge({actual_port}))"
            momentum_edges.setdefault(str(labels[1]), []).append(edge_id)
            momentum_actual_edges.setdefault(str(labels[1]), []).append(
                mapping["actual_edge"]
            )
            momentum_slots.setdefault(str(labels[1]), []).append(str(labels[0]))
            momentum_edge_by_slot[(str(labels[1]), str(labels[0]))] = edge_id
            index_slots[str(labels[2])] = f"coad(8,hedge({actual_port}))"
        elif particle == "d" and len(labels) >= 3:
            index_slots[str(labels[0])] = f"bis(4,hedge({actual_port}))"
            momentum_edges.setdefault(str(labels[1]), []).append(edge_id)
            momentum_actual_edges.setdefault(str(labels[1]), []).append(
                mapping["actual_edge"]
            )
            momentum_slots.setdefault(str(labels[1]), []).append(str(labels[0]))
            momentum_edge_by_slot[(str(labels[1]), str(labels[0]))] = edge_id
            index_slots[str(labels[2])] = f"cof(3,hedge({actual_port}))"
        elif particle == "t" and len(labels) >= 3:
            index_slots[str(labels[0])] = f"bis(4,hedge({actual_port}))"
            momentum_edges.setdefault(str(labels[1]), []).append(edge_id)
            momentum_actual_edges.setdefault(str(labels[1]), []).append(
                mapping["actual_edge"]
            )
            momentum_slots.setdefault(str(labels[1]), []).append(str(labels[0]))
            momentum_edge_by_slot[(str(labels[1]), str(labels[0]))] = edge_id
            index_slots[str(labels[2])] = _compact_fermion_color_slot(
                mapping["actual_edge"],
                actual_port,
            )
        elif particle == "d~" and len(labels) >= 3:
            index_slots[str(labels[0])] = f"bis(4,hedge({actual_port}))"
            momentum_edges.setdefault(str(labels[1]), []).append(edge_id)
            momentum_actual_edges.setdefault(str(labels[1]), []).append(
                mapping["actual_edge"]
            )
            momentum_slots.setdefault(str(labels[1]), []).append(str(labels[0]))
            momentum_edge_by_slot[(str(labels[1]), str(labels[0]))] = edge_id
            index_slots[str(labels[2])] = f"dind(cof(3,hedge({actual_port})))"

    momentum_signs = {}
    momentum_factors = {}
    for momentum_label, edge_ids in momentum_edges.items():
        signs = [1] * len(edge_ids)
        actual_edges = momentum_actual_edges.get(momentum_label, [])
        if (
            len(edge_ids) == 2
            and len(actual_edges) == 2
            and _external_edges_have_opposite_routing(actual_edges[0], actual_edges[1])
        ):
            signs[1] = -1
        momentum_factors[momentum_label] = list(zip(edge_ids, signs, strict=True))
        for slot_label, sign in zip(
            momentum_slots.get(momentum_label, []),
            signs,
            strict=True,
        ):
            momentum_signs[(momentum_label, slot_label)] = sign

    return index_slots, momentum_factors, momentum_edge_by_slot, momentum_signs


def _compact_label_maps(external_mappings):
    (
        index_slots,
        momentum_factors,
        _momentum_edge_by_slot,
        _momentum_signs,
    ) = _compact_remapping_maps(external_mappings)
    return index_slots, {
        momentum_label: factors[0][0]
        for momentum_label, factors in momentum_factors.items()
    }


def _signed_compact_q(edge_id: str, slot: str, sign: int) -> Expression:
    q = Eu(f"Q({edge_id},{slot})")
    if sign < 0:
        return E("-1") * q
    return q


def _replace_compact_momentum_functions(
    expr: Expression,
    index_slots: dict[str, str],
    momentum_factors: dict[str, list[tuple[str, int]]],
    momentum_edge_by_slot: dict[tuple[str, str], str],
    momentum_signs: dict[tuple[str, str], int],
) -> Expression:
    for momentum_label, factors in momentum_factors.items():
        fallback_edge_id, fallback_sign = factors[0]
        for slot_label, slot in index_slots.items():
            edge_id = momentum_edge_by_slot.get(
                (momentum_label, slot_label),
                fallback_edge_id,
            )
            sign = momentum_signs.get((momentum_label, slot_label), fallback_sign)
            expr = expr.replace(
                Eu(f"{momentum_label}({slot_label})"),
                _signed_compact_q(edge_id, slot, sign),
                repeat=True,
            )
        for lorentz_label in _COMPACT_FREE_LORENTZ_LABELS:
            expr = expr.replace(
                Eu(f"{momentum_label}({lorentz_label})"),
                _signed_compact_q(
                    fallback_edge_id,
                    f"mink(4,{lorentz_label})",
                    fallback_sign,
                ),
                repeat=True,
            )

    for left_label, left_factors in momentum_factors.items():
        for right_label, right_factors in momentum_factors.items():
            left_edge, left_sign = left_factors[0]
            right_edge, right_sign = right_factors[0]
            if left_label == right_label and len(left_factors) >= 2:
                if left_factors[0][1] == left_factors[1][1]:
                    right_edge, right_sign = left_factors[0]
                else:
                    right_edge, right_sign = left_factors[1]
            expr = expr.replace(
                Eu(f"sp({left_label},{right_label})"),
                (
                    _signed_compact_q(left_edge, "mink(4,rho)", left_sign)
                    * _signed_compact_q(
                        right_edge,
                        "mink(4,rho)",
                        right_sign,
                    )
                ),
                repeat=True,
            )
    return expr


def _replace_compact_spin_color_functions(
    expr: Expression,
    index_slots: dict[str, str],
    transpose_gamma_spinors: bool = False,
    transpose_color_spinors: bool = False,
    replace_free_lorentz_gamma: bool = False,
    replace_spin_delta: bool = False,
) -> Expression:
    labels = set(index_slots)
    for first in labels:
        for second in labels:
            for third in labels:
                if transpose_gamma_spinors:
                    gamma_replacement = Eu(
                        f"gamma({index_slots[first]},"
                        f"{index_slots[third]},"
                        f"{index_slots[second]})"
                    )
                else:
                    gamma_replacement = Eu(
                        f"gamma({index_slots[third]},"
                        f"{index_slots[first]},"
                        f"{index_slots[second]})"
                    )
                expr = expr.replace(
                    Eu(f"gamma({first},{second},{third})"),
                    gamma_replacement,
                    repeat=True,
                )
                expr = expr.replace(
                    Eu(f"T({first},{second},{third})"),
                    (
                        Eu(
                            f"t({index_slots[second]},"
                            f"{index_slots[third]},"
                            f"{index_slots[first]})"
                        )
                        if transpose_color_spinors
                        else Eu(
                            f"t({index_slots[second]},"
                            f"{index_slots[first]},"
                            f"{index_slots[third]})"
                        )
                    ),
                    repeat=True,
                )
        if replace_spin_delta and index_slots[first].startswith("bis("):
            for second in labels:
                if not index_slots[second].startswith("bis("):
                    continue
                expr = expr.replace(
                    Eu(f"delta({first},{second})"),
                    Eu(f"g({index_slots[first]},{index_slots[second]})"),
                    repeat=True,
                )
        if replace_free_lorentz_gamma and index_slots[first].startswith("bis("):
            for third in labels:
                if not index_slots[third].startswith("bis("):
                    continue
                for lorentz_label in _COMPACT_FREE_LORENTZ_LABELS:
                    expr = expr.replace(
                        Eu(f"gamma({first},{lorentz_label},{third})"),
                        Eu(
                            f"gamma({index_slots[first]},"
                            f"{index_slots[third]},"
                            f"mink(4,{lorentz_label}))"
                        ),
                        repeat=True,
                    )
    return expr


def _replace_compact_index_labels(
    expr: Expression,
    index_slots: dict[str, str],
) -> Expression:
    for label, slot in sorted(
        index_slots.items(), key=lambda item: len(item[0]), reverse=True
    ):
        expr = expr.replace(Eu(label), Eu(slot), repeat=True)
    return expr


def _actual_port_is_source(mapping) -> bool:
    return _port_suffix(mapping["actual_edge"].get_source()) == str(
        mapping["actual_port"]
    )


def _actual_port_is_destination(mapping) -> bool:
    return _port_suffix(mapping["actual_edge"].get_destination()) == str(
        mapping["actual_port"]
    )


def _transpose_compact_top_gluon_spinors(external_mappings) -> bool:
    external_particles = tuple(
        _normalise_uv_external_particle(mapping["reference"].get("particle", ""))
        for mapping in external_mappings
    )
    return (
        external_particles == ("t", "t", "g")
        and _actual_port_is_source(external_mappings[0])
        and _actual_port_is_destination(external_mappings[1])
    )


def _remap_compact_uv_integrated_expression(
    expr: Expression, external_mappings
) -> Expression:
    (
        index_slots,
        momentum_factors,
        momentum_edge_by_slot,
        momentum_signs,
    ) = _compact_remapping_maps(external_mappings)
    expr = _replace_compact_momentum_functions(
        expr,
        index_slots,
        momentum_factors,
        momentum_edge_by_slot,
        momentum_signs,
    )
    external_particles = tuple(
        _normalise_uv_external_particle(mapping["reference"].get("particle", ""))
        for mapping in external_mappings
    )
    transpose_top_gluon_spinors = _transpose_compact_top_gluon_spinors(
        external_mappings
    )
    expr = _replace_compact_spin_color_functions(
        expr,
        index_slots,
        transpose_gamma_spinors=external_particles == ("d", "d~", "g")
        or transpose_top_gluon_spinors,
        transpose_color_spinors=transpose_top_gluon_spinors,
        replace_free_lorentz_gamma=external_particles == ("t", "t"),
        replace_spin_delta=external_particles == ("t", "t"),
    )
    expr = _replace_compact_index_labels(expr, index_slots)
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


def _format_compact_uv_integrated_numerator(expr: Expression) -> str:
    expr_str = " ".join(expr.to_canonical_string().split())
    for prefix, replacement in (
        ("python::{}::", ""),
        ("gammalooprs::{}::", ""),
        ("symbolica::{}::", ""),
    ):
        expr_str = expr_str.replace(prefix, replacement)

    for function_name in ["gamma", "mink", "coad", "cof", "dind", "bis", "g", "t"]:
        expr_str = re.sub(
            rf"(?<![A-Za-z0-9_:]){function_name}\(",
            f"spenso::{function_name}(",
            expr_str,
        )

    expr_str = re.sub(
        r"\(?(Q\([0-9]+,spenso::mink\(4,rho\)\))\)?\^2",
        r"\1*\1",
        expr_str,
    )
    return expr_str.replace("+-", "-")


def _edge_loop_indices(edge, n_loops: int) -> set[int]:
    edge_atts = edge.get_attributes()
    loop_indices = set()
    for i in range(n_loops):
        value = edge_atts.get(f"routing_k{i}")
        if value is None:
            continue
        if _strip_quotes(str(value)).strip() in {"0", "+0", "-0"}:
            continue
        loop_indices.add(i)
    return loop_indices


def _common_uv_loop_indices(cycle, n_loops: int) -> list[int]:
    common_indices = None
    for edge in cycle:
        edge_indices = _edge_loop_indices(edge, n_loops)
        common_indices = (
            set(edge_indices)
            if common_indices is None
            else common_indices & edge_indices
        )
    return sorted(common_indices or [])


def _routing_component(edge, key: str) -> str:
    return _strip_quotes(str(edge.get_attributes().get(key, "0"))).strip()


def _same_routing(edge_a, edge_b, n_loops: int) -> bool:
    keys = [f"routing_k{i}" for i in range(n_loops + 1)]
    keys.extend(["routing_p1", "routing_p2"])
    return all(
        _routing_component(edge_a, key) == _routing_component(edge_b, key)
        for key in keys
    )


def _repeated_top_two_point_edges(external_mappings, n_loops: int):
    if len(external_mappings) != 2:
        return None
    particles = [
        _normalise_uv_external_particle(mapping["reference"].get("particle", ""))
        for mapping in external_mappings
    ]
    if particles != ["t", "t"]:
        return None

    edges = [mapping["actual_edge"] for mapping in external_mappings]
    if not _same_routing(edges[0], edges[1], n_loops):
        return None
    return tuple(
        _strip_quotes(str(edge.get_attributes()["id"]))
        for edge in edges
    )


def _regularise_repeated_two_point_integrated_ct(
    expr: Expression,
    external_mappings,
    n_loops: int,
    cut_graph=None,
) -> Expression:
    repeated_edges = _repeated_top_two_point_edges(external_mappings, n_loops)
    if repeated_edges is None:
        return expr

    repeated_id, other_id = repeated_edges
    cut_repeated_ids = []
    if cut_graph is not None:
        edge_by_id = {
            _strip_quotes(str(edge.get_attributes()["id"])): edge
            for edge in cut_graph.graph.get_edges()
        }
        cut_repeated_ids = [
            edge_id
            for edge_id in repeated_edges
            if edge_by_id[edge_id].get_attributes().get("is_cut_DY") is not None
        ]
        if len(cut_repeated_ids) != 1:
            cut_repeated_ids = [
                edge_id
                for edge_id in repeated_edges
                if _strip_quotes(
                    str(edge_by_id[edge_id].get_attributes().get("is_cut", "0"))
                )
                not in {"", "0", "None"}
            ]
        if len(cut_repeated_ids) == 1:
            repeated_id = cut_repeated_ids[0]
            other_id = (
                repeated_edges[1]
                if repeated_id == repeated_edges[0]
                else repeated_edges[0]
            )

    same = E("__uv_int_same")
    repeated_energy = E(f"E({repeated_id})")
    other_energy = E(f"E({other_id})")
    expr = (
        (expr * (repeated_energy - other_energy) * (E("2") * repeated_energy))
        .replace(repeated_energy, other_energy + same)
        .series(same, 0, 0)
        .to_expression()
    )
    if cut_graph is None or len(cut_repeated_ids) != 1:
        return expr

    initial_cut_ids = [
        _strip_quotes(str(edge.get_attributes()["id"]))
        for edge in cut_graph.initial_cut
    ]
    final_cut_ids = [
        _strip_quotes(str(edge.get_attributes()["id"]))
        for edge in cut_graph.final_cut
    ]
    replacement = (
        sum(E(f"E({edge_id})") for edge_id in initial_cut_ids)
        - sum(
            E(f"E({edge_id})")
            for edge_id in final_cut_ids
            if edge_id != repeated_id
        )
    ).expand()

    expr = -expr.replace(E(f"E({other_id})"), replacement)
    expr = -expr.replace(E(f"En({repeated_id})"), replacement)
    expr = -expr.replace(E(f"En({other_id})"), replacement)
    return expr / E(f"E({repeated_id})")


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

    if candidates[0].get("source") == "compact":
        tensor_numerator = _format_compact_uv_integrated_numerator(
            _remap_compact_uv_integrated_expression(
                finite_counterterm,
                external_mappings,
            )
        )
    else:
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
        deepcopy(contracted_cut_graph),
        numerator_factorisation=_uv_int_numerator_factorisation,
    )
    contracted_emr = _regularise_repeated_two_point_integrated_ct(
        contracted_emr,
        external_mappings,
        subtraction.L,
        contracted_cut_graph,
    )
    open_lorentz_momenta = _open_lorentz_momenta(contracted_emr)
    if open_lorentz_momenta:
        graph_name = contracted_graph.get("base_graph_name") or contracted_graph.get_name()
        raise ValueError(
            "Integrated UV counterterm numerator for "
            f"{graph_name} still contains open Lorentz momentum components "
            "after EMR construction: "
            f"{', '.join(open_lorentz_momenta[:8])}"
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

    uv_loop_indices = _common_uv_loop_indices(cycle, subtraction.L)
    if len(uv_loop_indices) != 1:
        return None

    normalisation=E("mUV")/(3.141592653589793238462643383279502884197169399375105820974944592)**2

    normalising_tadpole = normalisation*E("1") / (
        subtraction.sp3D(
            E(f"k({uv_loop_indices[0]})"), E(f"k({uv_loop_indices[0]})")
        )
        + E("mUV") ** 2
    )**2

    routed_integrand *= -normalising_tadpole / E("4")

    return routed_integrand_cls(
        routed_integrand,
        contracted_cut_graph,
        [],
        contracted_emr,
        "uv_int",
        "uv",
    )
