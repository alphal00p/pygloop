from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from itertools import permutations, product
from typing import Any

import pydot


def strip_quotes(value: Any) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    return text


def endpoint_node(endpoint: str) -> str:
    return strip_quotes(endpoint).split(":", 1)[0]


def is_external_node(node_name: str) -> bool:
    return endpoint_node(node_name).startswith("ext")


def normalise_particle(particle: Any) -> str:
    particle_name = strip_quotes(particle)
    aliases = {
        "H": "h",
        "h": "h",
        "d": "d",
        "d~": "d~",
        "g": "g",
        "t": "t",
        "t~": "t~",
    }
    return aliases.get(particle_name, particle_name)


def edge_particle(edge: pydot.Edge) -> str:
    return normalise_particle(edge.get_attributes().get("particle", ""))


def node_name(node: pydot.Node) -> str:
    return strip_quotes(node.get_name())


def graph_name(graph: pydot.Dot) -> str:
    return strip_quotes(graph.get_name())


def node_int_id(node: pydot.Node) -> str:
    return strip_quotes(node.get_attributes().get("int_id", ""))


def graph_internal_nodes(graph: pydot.Dot) -> dict[str, pydot.Node]:
    out: dict[str, pydot.Node] = {}
    for node in graph.get_nodes():
        name = node_name(node)
        if name in {"node", "edge", "graph"} or is_external_node(name):
            continue
        out[name] = node
    return out


def graph_internal_edges(graph: pydot.Dot) -> list[pydot.Edge]:
    edges: list[pydot.Edge] = []
    for edge in graph.get_edges():
        source = endpoint_node(edge.get_source())
        destination = endpoint_node(edge.get_destination())
        if not is_external_node(source) and not is_external_node(destination):
            edges.append(edge)
    return edges


def graph_external_edges(graph: pydot.Dot) -> list[pydot.Edge]:
    edges: list[pydot.Edge] = []
    for edge in graph.get_edges():
        source = endpoint_node(edge.get_source())
        destination = endpoint_node(edge.get_destination())
        if is_external_node(source) or is_external_node(destination):
            edges.append(edge)
    return edges


def non_external_endpoint(edge: pydot.Edge) -> str | None:
    source = endpoint_node(edge.get_source())
    destination = endpoint_node(edge.get_destination())
    source_is_external = is_external_node(source)
    destination_is_external = is_external_node(destination)
    if source_is_external and not destination_is_external:
        return destination
    if destination_is_external and not source_is_external:
        return source
    return None


@dataclass
class GraphSelection:
    graph: pydot.Dot
    canonical_key: str
    topology_hash: str


@dataclass
class RejectedGraph:
    name: str
    reason: str


@dataclass
class SelectionReport:
    accepted: list[GraphSelection] = field(default_factory=list)
    rejected: list[RejectedGraph] = field(default_factory=list)
    groups: dict[str, list[str]] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "accepted_graphs": [
                {
                    "name": graph_name(item.graph),
                    "topology_hash": item.topology_hash,
                    "canonical_key": item.canonical_key,
                    "group_id": item.graph.get_attributes().get("group_id"),
                    "group_master": strip_quotes(
                        item.graph.get_attributes().get("is_group_master", "false")
                    ),
                }
                for item in self.accepted
            ],
            "rejected_graphs": [
                {"name": item.name, "reason": item.reason} for item in self.rejected
            ],
            "groups": self.groups,
            "counts": {
                "accepted": len(self.accepted),
                "rejected": len(self.rejected),
                "groups": len(self.groups),
            },
        }


@dataclass(frozen=True)
class TopologySelectorConfig:
    initial_state: tuple[str, str]
    final_state: tuple[str, ...]
    light_quark_gluon_vertex_id: str
    top_gluon_vertex_id: str
    top_higgs_vertex_id: str
    light_quark_particles: tuple[str, ...] = ("d", "d~")
    gluon_particle: str = "g"
    top_particles: tuple[str, ...] = ("t", "t~")

    @property
    def final_state_normalised(self) -> tuple[str, ...]:
        return tuple(normalise_particle(p) for p in self.final_state)

    @property
    def initial_state_normalised(self) -> tuple[str, str]:
        return tuple(normalise_particle(p) for p in self.initial_state)  # type: ignore[return-value]


def _node_by_int_id(nodes: dict[str, pydot.Node], int_id: str) -> list[str]:
    return [name for name, node in nodes.items() if node_int_id(node) == int_id]


def _external_particle_counts(graph: pydot.Dot) -> dict[str, int]:
    counts: dict[str, int] = {}
    for edge in graph_external_edges(graph):
        particle = edge_particle(edge)
        counts[particle] = counts.get(particle, 0) + 1
    return counts


def _top_subgraph_is_cycle(top_edges: list[pydot.Edge], top_vertices: set[str]) -> bool:
    if len(top_edges) != len(top_vertices):
        return False
    degree = {vertex: 0 for vertex in top_vertices}
    adjacency = {vertex: set() for vertex in top_vertices}
    for edge in top_edges:
        source = endpoint_node(edge.get_source())
        destination = endpoint_node(edge.get_destination())
        if source not in top_vertices or destination not in top_vertices:
            return False
        degree[source] += 1
        degree[destination] += 1
        adjacency[source].add(destination)
        adjacency[destination].add(source)
    if any(value != 2 for value in degree.values()):
        return False
    stack = [next(iter(top_vertices))]
    visited: set[str] = set()
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(adjacency[current] - visited)
    return visited == top_vertices


def graph_rejection_reason(
    graph: pydot.Dot, selector_config: TopologySelectorConfig
) -> str | None:
    nodes = graph_internal_nodes(graph)
    internal_edges = graph_internal_edges(graph)
    external_edges = graph_external_edges(graph)
    particle_counts = _external_particle_counts(graph)

    for particle in selector_config.initial_state_normalised:
        if particle_counts.get(particle, 0) != 1:
            return f"expected exactly one external {particle}"
    for particle in sorted(set(selector_config.final_state_normalised)):
        expected = selector_config.final_state_normalised.count(particle)
        if particle_counts.get(particle, 0) != expected:
            return f"expected {expected} external {particle}"

    q_vertices = _node_by_int_id(
        nodes, selector_config.light_quark_gluon_vertex_id
    )
    top_gluon_vertices = _node_by_int_id(nodes, selector_config.top_gluon_vertex_id)
    top_higgs_vertices = _node_by_int_id(nodes, selector_config.top_higgs_vertex_id)

    if len(q_vertices) != 2:
        return f"expected 2 light-quark/gluon vertices, found {len(q_vertices)}"
    if len(top_gluon_vertices) != 2:
        return f"expected 2 top/gluon vertices, found {len(top_gluon_vertices)}"
    if len(top_higgs_vertices) != len(selector_config.final_state_normalised):
        return (
            f"expected {len(selector_config.final_state_normalised)} top/H vertices, "
            f"found {len(top_higgs_vertices)}"
        )

    gluon_edges = [
        edge
        for edge in internal_edges
        if edge_particle(edge) == selector_config.gluon_particle
    ]
    light_quark_edges = [
        edge
        for edge in internal_edges
        if edge_particle(edge) in selector_config.light_quark_particles
    ]
    top_edges = [
        edge for edge in internal_edges if edge_particle(edge) in selector_config.top_particles
    ]

    if len(gluon_edges) != 2:
        return f"expected 2 internal gluon bridges, found {len(gluon_edges)}"
    if len(light_quark_edges) != 1:
        return f"expected 1 internal massless quark edge, found {len(light_quark_edges)}"
    if len(top_edges) != len(selector_config.final_state_normalised) + 2:
        return (
            f"expected {len(selector_config.final_state_normalised) + 2} top-loop edges, "
            f"found {len(top_edges)}"
        )

    q_set = set(q_vertices)
    top_gluon_set = set(top_gluon_vertices)
    top_vertex_set = set(top_gluon_vertices + top_higgs_vertices)

    light_quark_edge = light_quark_edges[0]
    if {
        endpoint_node(light_quark_edge.get_source()),
        endpoint_node(light_quark_edge.get_destination()),
    } != q_set:
        return "internal massless quark edge does not connect the two initial-state vertices"

    touched_q_vertices: set[str] = set()
    touched_top_gluon_vertices: set[str] = set()
    for edge in gluon_edges:
        source = endpoint_node(edge.get_source())
        destination = endpoint_node(edge.get_destination())
        source_is_q = source in q_set
        destination_is_q = destination in q_set
        source_is_top_gluon = source in top_gluon_set
        destination_is_top_gluon = destination in top_gluon_set
        if not (
            (source_is_q and destination_is_top_gluon)
            or (destination_is_q and source_is_top_gluon)
        ):
            return "internal gluon bridge does not connect a quark vertex to a top-loop gluon vertex"
        touched_q_vertices.add(source if source_is_q else destination)
        touched_top_gluon_vertices.add(
            source if source_is_top_gluon else destination
        )
    if touched_q_vertices != q_set:
        return "not every initial-state quark vertex has one gluon bridge"
    if touched_top_gluon_vertices != top_gluon_set:
        return "not every top-loop gluon vertex has one gluon bridge"

    if not _top_subgraph_is_cycle(top_edges, top_vertex_set):
        return "top subgraph is not a single pentagon loop"

    initial_attachment_counts = {vertex: [] for vertex in q_vertices}
    final_attachment_counts = {vertex: 0 for vertex in top_higgs_vertices}
    for edge in external_edges:
        particle = edge_particle(edge)
        endpoint = non_external_endpoint(edge)
        if endpoint is None:
            continue
        if particle in selector_config.initial_state_normalised and endpoint in q_set:
            initial_attachment_counts[endpoint].append(particle)
        if particle in selector_config.final_state_normalised and endpoint in final_attachment_counts:
            final_attachment_counts[endpoint] += 1
    if sorted(len(values) for values in initial_attachment_counts.values()) != [1, 1]:
        return "initial d and d~ are not attached one per light-quark vertex"
    if any(value != 1 for value in final_attachment_counts.values()):
        return "each top/H vertex must carry exactly one external Higgs"

    return None


def _node_labels(graph: pydot.Dot, selector_config: TopologySelectorConfig) -> dict[str, str]:
    nodes = graph_internal_nodes(graph)
    labels: dict[str, str] = {}
    initial_attachments: dict[str, str] = {}
    for edge in graph_external_edges(graph):
        particle = edge_particle(edge)
        endpoint = non_external_endpoint(edge)
        if endpoint is not None and particle in selector_config.initial_state_normalised:
            initial_attachments[endpoint] = particle

    for name, node in nodes.items():
        int_id = node_int_id(node)
        if int_id == selector_config.light_quark_gluon_vertex_id:
            labels[name] = f"qg:{initial_attachments.get(name, 'unknown')}"
        elif int_id == selector_config.top_gluon_vertex_id:
            labels[name] = "tg"
        elif int_id == selector_config.top_higgs_vertex_id:
            labels[name] = "th"
        else:
            labels[name] = f"other:{int_id}"
    return labels


def canonical_topology_key(
    graph: pydot.Dot, selector_config: TopologySelectorConfig
) -> str:
    labels = _node_labels(graph, selector_config)
    names_by_label: dict[str, list[str]] = {}
    for name, label in labels.items():
        names_by_label.setdefault(label, []).append(name)

    label_order = sorted(names_by_label)
    canonical_targets: dict[str, list[str]] = {}
    next_id = 0
    for label in label_order:
        size = len(names_by_label[label])
        canonical_targets[label] = [f"{label}#{next_id + i}" for i in range(size)]
        next_id += size

    choices = []
    for label in label_order:
        source_names = sorted(names_by_label[label])
        targets = canonical_targets[label]
        choices.append(
            [
                dict(zip(source_names, permuted_targets))
                for permuted_targets in permutations(targets)
            ]
        )

    edge_records: list[tuple[str, str, str]] = []
    for edge in graph.get_edges():
        source = endpoint_node(edge.get_source())
        destination = endpoint_node(edge.get_destination())
        particle = edge_particle(edge)
        if is_external_node(source) and destination in labels:
            edge_records.append((destination, f"ext:{particle}", "external"))
        elif is_external_node(destination) and source in labels:
            edge_records.append((source, f"ext:{particle}", "external"))
        elif source in labels and destination in labels:
            edge_records.append((source, destination, f"internal:{particle}"))

    serialisations: list[str] = []
    for parts in product(*choices):
        mapping: dict[str, str] = {}
        for part in parts:
            mapping.update(part)
        mapped_edges: list[tuple[str, str, str]] = []
        for left, right, label in edge_records:
            if label == "external":
                mapped_edges.append((mapping[left], right, label))
            else:
                a = mapping[left]
                b = mapping[right]
                if b < a:
                    a, b = b, a
                mapped_edges.append((a, b, label))
        serialisations.append(repr(sorted(mapped_edges)))

    return min(serialisations)


def topology_hash(canonical_key: str) -> str:
    return hashlib.sha1(canonical_key.encode("utf-8")).hexdigest()[:16]


def select_top_pentagon_isr_graphs(
    graphs: list[pydot.Dot], selector_config: TopologySelectorConfig
) -> SelectionReport:
    report = SelectionReport()
    for graph in graphs:
        reason = graph_rejection_reason(graph, selector_config)
        if reason is not None:
            report.rejected.append(RejectedGraph(graph_name(graph), reason))
            continue
        graph_copy = copy.deepcopy(graph)
        strip_gammaloop_autogenerated_attributes(graph_copy)
        canonical_key = canonical_topology_key(graph_copy, selector_config)
        report.accepted.append(
            GraphSelection(
                graph=graph_copy,
                canonical_key=canonical_key,
                topology_hash=topology_hash(canonical_key),
            )
        )

    accepted_by_key: dict[str, list[GraphSelection]] = {}
    for accepted in report.accepted:
        accepted_by_key.setdefault(accepted.canonical_key, []).append(accepted)

    for group_id, canonical_key in enumerate(sorted(accepted_by_key)):
        group = sorted(accepted_by_key[canonical_key], key=lambda item: graph_name(item.graph))
        group_names = [graph_name(item.graph) for item in group]
        report.groups[str(group_id)] = group_names
        for index, item in enumerate(group):
            item.graph.set("group_id", str(group_id))
            item.graph.set("is_group_master", "true" if index == 0 else "false")

    return report


def parse_dot_graphs(dot_data: str) -> list[pydot.Dot]:
    graphs = pydot.graph_from_dot_data(dot_data)
    if graphs is None:
        raise ValueError("No graphs found in DOT data.")
    return graphs


def dot_graphs_to_string(graphs: list[pydot.Dot]) -> str:
    return "\n\n".join(graph.to_string() for graph in graphs)


def strip_gammaloop_autogenerated_attributes(graph: pydot.Dot) -> None:
    for attrs in [graph.get_attributes()]:
        _strip_autogenerated_attributes(attrs)
    for node in graph.get_nodes():
        _strip_autogenerated_attributes(node.get_attributes())
    for edge in graph.get_edges():
        _strip_autogenerated_attributes(edge.get_attributes())


def _strip_autogenerated_attributes(attributes: dict[str, Any]) -> None:
    for key in ("num_autogen", "dod_autogen", "name_autogen"):
        attributes.pop(key, None)
