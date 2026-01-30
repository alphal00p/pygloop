import copy
import re
from collections import deque
from itertools import product
from typing import List, Optional, Set, Tuple

import pydot



# == Parsing ===================================================================
# Helpers for turning DOT endpoints like '"a:3"' into canonical node/port forms.
# ==============================================================================

# Remove surrounding double quotes if present.
def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s

# True if the vertex name matches "ext" followed by digits (ignoring quotes).
def _is_ext(name: str) -> bool:
    return bool(re.fullmatch(r"ext\d+", _strip_quotes(name)))

# pydot vertices are in the form "v:port", where port is an int; returns (v,port)
def _parse_port_endpoint(endpoint: str) -> Optional[Tuple[str, int]]:
    ep = _strip_quotes(endpoint)
    m = re.fullmatch(r"([^:]+):(\d+)", ep)
    if not m:
        return None
    return m.group(1), int(m.group(2))

# pydot vertices are in the form "v:port", where port is an int; return v
def _base_node(endpoint: str) -> str:
    ep = _strip_quotes(endpoint)
    parsed = _parse_port_endpoint(ep)
    return parsed[0] if parsed else ep

# pydot vertices are in the form "v:port"; return v (optionally collapse ports)
def _node_key(endpoint: str, collapse_ports: bool = True) -> str:
    ep = _strip_quotes(endpoint)
    if collapse_ports and ":" in ep:
        return ep.split(":", 1)[0]
    return ep

# pydot vertices are in the form "v:port"; return port
def _parse_port(endpoint: str) -> Optional[int]:
    """Return b from 'a:b' (possibly quoted), else None."""
    ep = _strip_quotes(endpoint)
    m = re.fullmatch(r"[^:]+:(\d+)", ep)
    return None if not m else int(m.group(1))



# == Edge attribute access ==================================
# Helpers for extracting/removing attributes from pydot edges
# ===========================================================

# Return e.get_attributes()["is_cut"] as an int (defaults to 0), stripping quotes if needed.
def _edge_is_cut_value(e: pydot.Edge) -> int:
    attrs = e.get_attributes() or {}
    val = attrs.get("is_cut", "0")
    return int(_strip_quotes(str(val)))

# Return e.get_attributes()["particle"] as an int (defaults to 0), stripping quotes if needed.
def _edge_particle(e: pydot.Edge) -> str:
    attrs = e.get_attributes() or {}
    p = attrs.get("particle", "")
    return _strip_quotes(str(p))

# Return e.get_attributes()["id"] as an int
def edge_id_int(e: pydot.Edge) -> int:
    attrs = e.get_attributes() or {}
    try:
        return int(_strip_quotes(str(attrs.get("id", ""))))
    except Exception:
        return 10**18

# Removes an edge attribute
def remove_edge_attr(e: pydot.Edge, key: str) -> None:
    if e is None:
        raise ValueError("remove_edge_attr got None edge")

    attrs = e.get_attributes() or {}
    attrs.pop(key, None)  # in-place usually works in pydot
    return e

# Sorts a list of edges by the integer value of the port
def sort_half_edges_by_port(
    edges: List[pydot.Edge],
    non_ext_endpoint_fn,
    reverse: bool = False,
) -> List[pydot.Edge]:
    def key(e: pydot.Edge):
        end = non_ext_endpoint_fn(e)
        b = _parse_port(end)
        if b is None:
            raise ValueError(f"Non-ext endpoint is not of form a:b: {end}")
        return (b, _strip_quotes(end), edge_id_int(e))

    return sorted(edges, key=key, reverse=reverse)

def _all_node_names(graph):
    seen = set()
    out = []
    for e in graph.get_edges():
        for raw in (e.get_source(), e.get_destination()):
            name = _base_node(raw)
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out



# == Graph topology helpers =======================================
# Helpers for extracting boundaries of cuts or the adjacency matrix
# =================================================================

# Given an edge and a node contained in the edge, returns the other node
def other_node(edge: pydot.Edge, node: str) -> Optional[str]:
    u = _node_key(edge.get_source())
    v = _node_key(edge.get_destination())
    if u == node:
        return v
    if v == node:
        return u
    return None

# Given a graph and a cut (set of vertices of the graph), returns the boundary of the cut
def boundary_edges(graph: pydot.Dot, S: set[str]) -> List[pydot.Edge]:
    out = []
    edges = graph.get_edges()
    nodes = set()
    for e in edges:
        nodes.add(_base_node(e.get_source()))
        nodes.add(_base_node(e.get_destination()))
    if not set(S).issubset(nodes):
        raise ValueError("cut contains vertices not present in graph")
    for e in edges:
        u = _base_node(e.get_source())
        v = _base_node(e.get_destination())
        if (u in S and v not in S) or (u not in S and v in S):
            out.append(e)
    return out

# Returns the adjacency matrix of the graph
def get_vertex_to_edges_map(graph: pydot.Dot) -> dict[str, List[pydot.Edge]]:
    adj = {}
    edges = graph.get_edges()
    for e in edges:
        u = _node_key(e.get_source())
        v = _node_key(e.get_destination())
        adj.setdefault(u, []).append(e)
        adj.setdefault(v, []).append(e)
    return adj

# Returns the spanning tree of the graph; the algorithm is stupid but I like it
# because it uses the boundary function
def get_spanning_tree(graph: pydot.Dot) -> List[pydot.Edge]:
    edges = graph.get_edges()
    if not edges:
        return []

    root = _node_key(edges[0].get_source())
    S = {root}
    T = []

    changed = True
    while changed:
        changed = False
        for e in boundary_edges(graph, S):
            u = _node_key(e.get_source())
            v = _node_key(e.get_destination())

            if u in S and v not in S:
                S.add(v)
                T.append(e)
                changed = True
                break
            if v in S and u not in S:
                S.add(u)
                T.append(e)
                changed = True
                break

    return T

# Returns the cycle basis of the graph
def get_cycle_basis(graph: pydot.Dot) -> List[Set[pydot.Edge]]:
    T = set(get_spanning_tree(graph))
    L = set(graph.get_edges()) - set(T)

    cycles = []

    for e in L:
        cycle = T.union({e})
        check = True
        while check:
            counter = 0
            for e in cycle:
                if (
                    len(set(boundary_edges(graph, {_node_key(e.get_destination())})).intersection(cycle)) == 1
                    or len(set(boundary_edges(graph, {_node_key(e.get_source())})).intersection(cycle)) == 1
                ):
                    cycle = cycle - {e}
                    break
                else:
                    counter += 1

            if counter == len(cycle):
                check = False
        cycles.append(cycle)

    return cycles

# Returns all possible simple cycles of the graph
def get_simple_cycles(graph: pydot.Dot) -> List[Set[pydot.Edge]]:
    edges = list(copy.deepcopy(graph.get_edges()))
    if not edges:
        return []

    adj = get_vertex_to_edges_map(graph)

    nodes = sorted(adj.keys())
    seen = set()
    cycles = []

    def other_node(edge: pydot.Edge, node: str) -> Optional[str]:
        u = _node_key(edge.get_source())
        v = _node_key(edge.get_destination())
        if u == node:
            return v
        if v == node:
            return u
        return None

    def dfs(start, current, parent_edge, path_nodes, path_edges):
        for edge in adj[current]:
            if edge is parent_edge:
                continue
            nxt = other_node(edge, current)
            if nxt is None:
                continue
            if nxt == start:
                if len(path_nodes) >= 2:
                    key = frozenset(id(e) for e in path_edges + [edge])
                    if key not in seen:
                        seen.add(key)
                        cycles.append(set(path_edges + [edge]))
                continue
            if nxt in path_nodes:
                continue
            if nxt < start:
                continue
            dfs(start, nxt, edge, path_nodes + [nxt], path_edges + [edge])

    for start in nodes:
        dfs(start, start, None, [start], [])

    return cycles

# Given a simple cycle, output the corresponding directed cycle
def _get_directed_cycle(graph: pydot.Dot, cycle: List[pydot.Edge]) -> List[pydot.Edge]:
    edges = list(cycle)
    if not edges:
        return copy.deepcopy(cycle)

    adj = {}
    for e in edges:
        u = _node_key(e.get_source())
        v = _node_key(e.get_destination())
        adj.setdefault(u, []).append(e)
        adj.setdefault(v, []).append(e)

    start = min(adj.keys())
    first_edge = min(adj[start], key=lambda e: other_node(e, start))

    visited = set()
    current = start
    prev_edge = None
    ordered = []

    while True:
        if prev_edge is None:
            edge = first_edge
        else:
            candidates = [e for e in adj[current] if e is not prev_edge]
            if not candidates:
                raise ValueError("Cycle is not well-formed")
            edge = min(candidates, key=lambda e: other_node(e, current))

        if edge in visited:
            break
        visited.add(edge)

        next_node = other_node(edge, current)
        if next_node is None:
            raise ValueError("Cycle is not well-formed")

        src = _node_key(edge.get_source())
        dst = _node_key(edge.get_destination())
        direction = 1 if (src == current and dst == next_node) else -1
        edge.set("dir_in_cycle", direction)
        ordered.append((edge, direction))

        prev_edge = edge
        current = next_node
        if current == start:
            break
        if len(visited) > len(edges):
            raise ValueError("Cycle is not well-formed")

    if len(visited) != len(edges):
        raise ValueError("Cycle is not well-formed")

    # Orient the cycle so a reference is_cut==1 edge has dir_in_cycle == 1.
    flip = False
    for (e, d) in ordered:
        if _edge_is_cut_value(e) == 1 and d == 1:
            flip = True
    if flip:
        for (e, d) in ordered:
            e.set("dir_in_cycle", -d)

    return copy.deepcopy(cycle)

# Returns all directed cycles of a graph
def get_directed_cycles(graph: pydot.Dot, cycles=None) -> List[Set[pydot.Edge]]:
    if cycles is None:
        cycles = get_simple_cycles(graph)
    directed_cycles = []
    for cycle in cycles:
        #print("*****************cycle")
        #for e in cycle:
        #    print(e)
        directed_cycles.append(_get_directed_cycle(graph, cycle))
    return directed_cycles

# Checks if the subgraph induced by a cut node_subset_input of the graph is connected
def is_connected(graph, node_subset_input) -> bool:

    subset = set(node_subset_input.copy())
    if not subset:
        return True
    if len(subset) == 1:
        return True

    start = next(iter(subset))
    visited = {start}
    q = deque([start])

    while q:
        x = q.popleft()

        # scan edges to find neighbors of x inside subset
        for e in graph.get_edges():
            u = _base_node(e.get_source())
            v = _base_node(e.get_destination())

            # only consider induced edges (both endpoints in subset)
            if u not in subset or v not in subset:
                continue

            if u == x and v not in visited:
                visited.add(v)
                q.append(v)
            elif v == x and u not in visited:
                visited.add(u)
                q.append(u)

        if len(visited) == len(subset):
            return True

    return len(visited) == len(subset)



# == Set helpers =============
# Helpers for set manipulation
# ============================

# Given a list V, returns all lists V1, V2 such that V1+V2=V, V1/V2 and V2/V1 are non-empty
# Some more rules apply.
def all_pairs(V: List[str]) -> List[List[str]]:
    V = list(V)
    seen = set()
    def _key(v):
        if hasattr(v, "get_attributes"):
            return v.get_attributes().get("id", str(v))
        return v
    for labels in product("ABC", repeat=len(V)):
        if "A" not in labels or "B" not in labels:
            continue
        V1 = [v for v, lab in zip(V, labels) if lab in ("A", "C")]
        V2 = [v for v, lab in zip(V, labels) if lab in ("B", "C")]
        if len(set(V1) - set(V2)) == 0 or len(set(V2) - set(V1)) == 0:
            continue
        k1 = tuple(sorted((_key(v) for v in V1)))
        k2 = tuple(sorted((_key(v) for v in V2)))
        sig = (k1, k2)
        if sig in seen:
            continue
        seen.add(sig)

        yield V1, V2
