"""Closed-ghost-loop metadata derived from generated DY graph topology."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import pydot

from processes.dy.dy_graph_utils import _strip_quotes
from utils.utils import pygloopException


def ghost_particle_names_from_model_metadata(
    model_metadata: str | bytes | Mapping[str, Any],
) -> frozenset[str]:
    """Return all particle names carrying nonzero model ghost number."""

    if isinstance(model_metadata, bytes):
        model_metadata = model_metadata.decode("utf-8")
    if isinstance(model_metadata, str):
        try:
            parsed: Any = json.loads(model_metadata)
        except json.JSONDecodeError as exc:
            raise pygloopException(
                f"Could not parse GammaLoop model metadata: {exc}"
            ) from exc
    elif isinstance(model_metadata, Mapping):
        parsed = dict(model_metadata)
    else:
        raise pygloopException(
            "GammaLoop model metadata must be a JSON object or JSON string."
        )

    particles = parsed.get("particles") if isinstance(parsed, Mapping) else None
    if not isinstance(particles, list):
        raise pygloopException("GammaLoop model metadata is missing particles.")

    names: set[str] = set()
    for particle in particles:
        if not isinstance(particle, Mapping):
            raise pygloopException("Malformed particle entry in GammaLoop model metadata.")
        try:
            ghost_number = float(particle.get("ghost_number", 0))
        except (TypeError, ValueError) as exc:
            raise pygloopException(
                "Particle ghost_number metadata must be numeric."
            ) from exc
        if not math.isfinite(ghost_number):
            raise pygloopException("Particle ghost_number metadata must be finite.")
        if ghost_number == 0.0:
            continue
        for key in ("name", "antiname"):
            name = particle.get(key)
            if name is not None and str(name).strip():
                names.add(str(name).strip())
    return frozenset(names)


def _base_endpoint_name(endpoint: Any) -> str:
    return _strip_quotes(str(endpoint)).split(":", 1)[0]


def closed_ghost_loop_count(
    graph: pydot.Dot | Any,
    ghost_particle_names: set[str] | frozenset[str],
) -> int:
    """Count independent cycles in the graph's ghost-only multigraph.

    For each connected ghost component the cycle rank is ``E - V + 1``.
    This handles parallel ghost/antighost propagators, self-loops, and multiple
    disconnected closed ghost loops without relying on particle-name prefixes.
    """

    dot = getattr(graph, "dot", graph)
    if not isinstance(dot, pydot.Dot):
        raise pygloopException("Closed ghost-loop counting requires a pydot graph.")
    ghost_names = {str(name) for name in ghost_particle_names}
    ghost_edges: list[tuple[str, str]] = []
    for edge in dot.get_edges():
        particle = _strip_quotes(str(edge.get_attributes().get("particle", "")))
        if particle not in ghost_names:
            continue
        ghost_edges.append(
            (
                _base_endpoint_name(edge.get_source()),
                _base_endpoint_name(edge.get_destination()),
            )
        )
    if not ghost_edges:
        return 0

    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for source, destination in ghost_edges:
        union(source, destination)
    vertices = set(parent)
    components = len({find(vertex) for vertex in vertices})
    cycle_rank = len(ghost_edges) - len(vertices) + components
    if cycle_rank < 0:
        raise pygloopException("The ghost-only graph has an invalid negative cycle rank.")
    return cycle_rank
