from __future__ import annotations

import sys
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import pydot
from symbolica import E, Expression

from utils.cff import CFFStructure, EdgeOrientation


GENERALISED_LTD_ROOT = Path("/home/zeno/generalised_ltd")
if str(GENERALISED_LTD_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERALISED_LTD_ROOT))

from src import graph_io as GLTD_GRAPH_IO  # type: ignore  # noqa: E402
from src import structure as GLTD_STRUCTURE  # type: ignore  # noqa: E402
from src import validator as GLTD_VALIDATOR  # type: ignore  # noqa: E402
from src.api import _build_bundle  # type: ignore  # noqa: E402
from src.orientation_bundle import normalize_energy_degree_bounds  # type: ignore  # noqa: E402


def _strip_quotes(value: Any) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _node_base(name: Any) -> str:
    return _strip_quotes(name).split(":", 1)[0]


def _external_node_names(graph: pydot.Dot) -> set[str]:
    out = set()
    for node in graph.get_nodes():
        name = _node_base(node.get_name())
        if not name or name in {"graph", "node", "edge"}:
            continue
        attrs = node.get_attributes()
        shape = _strip_quotes(attrs.get("shape", "")).lower()
        style = _strip_quotes(attrs.get("style", "")).lower()
        if name.startswith("ext") or shape == "box" or style == "invis":
            out.add(name)
    return out


def _edge_sort_key(edge: pydot.Edge) -> tuple[int, str]:
    attrs = edge.get_attributes()
    edge_id = _strip_quotes(attrs.get("id", "0"))
    try:
        return int(edge_id), edge_id
    except ValueError:
        return 0, edge_id


def _routing_coeff(attrs: dict[str, Any], key: str) -> int:
    raw = _strip_quotes(attrs.get(key, "0"))
    frac = Fraction(raw)
    if frac.denominator != 1:
        raise ValueError(
            f"generalised_ltd CFF mode only supports integer routing coefficients; "
            f"got {key}={raw}."
        )
    return int(frac)


def _routing_keys(graph: pydot.Dot) -> tuple[list[str], list[str]]:
    keys = {
        key
        for edge in graph.get_edges()
        for key in edge.get_attributes()
        if key.startswith("routing_")
    }
    k_keys = sorted(
        [key for key in keys if key.startswith("routing_k")],
        key=lambda key: int(key.removeprefix("routing_k")),
    )
    p_keys = sorted(
        [key for key in keys if key.startswith("routing_p")],
        key=lambda key: int(key.removeprefix("routing_p")),
    )
    return k_keys, p_keys


def _mass_key(attrs: dict[str, Any]) -> str | None:
    explicit = _strip_quotes(attrs.get("mass", ""))
    if explicit:
        return explicit

    particle = _strip_quotes(attrs.get("particle", ""))
    if particle in {"", "a", "d", "d~", "g", "ghG", "ghG~"}:
        return None
    return particle


def _expr_number(value: Any) -> Expression:
    return E(str(value))


def _zero_tuple(size: int) -> tuple[int, ...]:
    return tuple(0 for _idx in range(size))


def _independent_column_indices(rows: list[tuple[int, ...]]) -> tuple[int, ...]:
    if not rows:
        return tuple()

    matrix = [[Fraction(value) for value in row] for row in rows]
    row_count = len(matrix)
    col_count = len(matrix[0]) if matrix[0] else 0
    pivot_row = 0
    pivots = []
    for col in range(col_count):
        pivot = None
        for row in range(pivot_row, row_count):
            if matrix[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue

        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        pivot_value = matrix[pivot_row][col]
        matrix[pivot_row] = [value / pivot_value for value in matrix[pivot_row]]
        for row in range(row_count):
            if row == pivot_row:
                continue
            factor = matrix[row][col]
            if factor:
                matrix[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(matrix[row], matrix[pivot_row])
                ]

        pivots.append(col)
        pivot_row += 1
        if pivot_row == row_count:
            break

    return tuple(pivots)


@dataclass(frozen=True)
class _ExternalEnergy:
    denominator: Expression
    numerator: Expression


class GeneralisedCFFSurface:
    def __init__(
        self,
        surface: dict[str, Any],
        internal_edge_map: dict[int, int],
        external_energy_map: dict[int, _ExternalEnergy],
    ):
        self.id = int(surface["id"])
        self.kind = str(surface.get("k", "e"))
        self.surface = surface
        self.internal_edge_map = internal_edge_map
        self.external_energy_map = external_energy_map
        self.expression = self.get_expression(use_numerator_energies=False)
        self.numerator_expression = self.get_expression(use_numerator_energies=True)

    def _external_energy(self, ext_id: int, use_numerator_energies: bool) -> Expression:
        energy = self.external_energy_map[int(ext_id)]
        return energy.numerator if use_numerator_energies else energy.denominator

    def get_expression(self, use_numerator_energies: bool) -> Expression:
        data = self.surface["e"]

        out = _expr_number(data.get("c", "0"))
        for edge_id, coeff in data.get("i", []):
            local_id = self.internal_edge_map[int(edge_id)]
            if use_numerator_energies:
                energy = E(f"En({local_id})")
            else:
                energy = CFFStructure.SB["E"](local_id)
            out += _expr_number(coeff) * energy
        for ext_id, coeff in data.get("x", []):
            out += _expr_number(coeff) * self._external_energy(
                int(ext_id),
                use_numerator_energies,
            )

        uniform_coeff = int(data.get("m", 0) or 0)
        if uniform_coeff:
            out += _expr_number(uniform_coeff) * E("M")
        return out

    def __str__(self) -> str:
        return self.expression.format(show_namespaces=False)


class GeneralisedCFFTerm:
    def __init__(
        self,
        id: int,
        orientation: tuple[EdgeOrientation, ...],
        expression: Expression,
        edge_q0: list[Expression],
    ):
        self.id = id
        self.orientation = orientation
        self.expression = expression
        self.edge_q0 = edge_q0


class GeneralisedCFFStructure:
    def __init__(
        self,
        graph: pydot.Dot,
        energy_degree_bounds: list[int] | tuple[int, ...] | dict[int, int] | None = None,
    ):
        self.graph = graph
        self.parsed, self.internal_edge_map, self.external_edge_map = self._parse_graph(graph)
        self.energy_degree_bounds = self._remap_energy_degree_bounds(
            energy_degree_bounds,
        )
        self.external_energy_map = self._external_energy_map()
        self.data = self._build_structure()
        self.e_surfaces: list[GeneralisedCFFSurface] = []
        self.h_surfaces: list[GeneralisedCFFSurface] = []
        self.expressions: list[GeneralisedCFFTerm] = []
        self._build_wrappers()

    def _remap_energy_degree_bounds(
        self,
        energy_degree_bounds: list[int] | tuple[int, ...] | dict[int, int] | None,
    ) -> list[int] | dict[int, int] | None:
        if energy_degree_bounds is None:
            return None
        if isinstance(energy_degree_bounds, dict):
            mapped = {}
            if "*" in energy_degree_bounds:
                mapped["*"] = int(energy_degree_bounds["*"])
            for parsed_id, local_id in self.internal_edge_map.items():
                if local_id in energy_degree_bounds:
                    mapped[parsed_id] = int(energy_degree_bounds[local_id])
                elif str(local_id) in energy_degree_bounds:
                    mapped[parsed_id] = int(energy_degree_bounds[str(local_id)])
            return mapped if any(int(value) > 1 for value in mapped.values()) else None

        bounds = list(energy_degree_bounds)
        mapped_bounds = []
        for parsed_id in range(len(self.internal_edge_map)):
            local_id = self.internal_edge_map[parsed_id]
            mapped_bounds.append(int(bounds[local_id]) if local_id < len(bounds) else 0)
        return mapped_bounds if any(bound > 1 for bound in mapped_bounds) else None

    def _parse_graph(
        self, graph: pydot.Dot
    ) -> tuple[GLTD_GRAPH_IO.ParsedGraph, dict[int, int], dict[int, int]]:
        edges = sorted(graph.get_edges(), key=_edge_sort_key)
        k_keys, p_keys = _routing_keys(graph)
        if not k_keys:
            raise ValueError("generalised_ltd CFF mode requires routed graph edges.")

        external_edge_items = []
        internal_edge_items = []
        external_nodes = _external_node_names(graph)
        for edge in edges:
            src = _node_base(edge.get_source())
            dst = _node_base(edge.get_destination())
            src_ext = src in external_nodes or src.startswith("ext")
            dst_ext = dst in external_nodes or dst.startswith("ext")
            if src_ext ^ dst_ext:
                external_edge_items.append(edge)
            else:
                internal_edge_items.append(edge)

        node_names = sorted(
            {
                _node_base(endpoint)
                for edge in internal_edge_items
                for endpoint in (edge.get_source(), edge.get_destination())
                if (
                    _node_base(endpoint) not in external_nodes
                    and not _node_base(endpoint).startswith("ext")
                )
            }
        )
        node_name_to_internal = {name: idx for idx, name in enumerate(node_names)}
        edge_ext_names = tuple(
            f"x{_strip_quotes(edge.get_attributes().get('id', idx))}"
            for idx, edge in enumerate(external_edge_items)
        )
        physical_ext_names = tuple(f"p{idx + 1}" for idx in range(len(p_keys)))
        ext_names = (*edge_ext_names, *physical_ext_names)

        ext_routings = [
            (
                tuple(_routing_coeff(edge.get_attributes(), key) for key in k_keys),
                tuple(_routing_coeff(edge.get_attributes(), key) for key in p_keys),
            )
            for edge in external_edge_items
        ]
        node_ext_potentials = self._node_external_potentials(
            external_edge_items,
            external_nodes,
        )
        for physical_id in range(len(p_keys)):
            ext_routings.append(
                (
                    tuple(0 for _key in k_keys),
                    tuple(1 if idx == physical_id else 0 for idx in range(len(p_keys))),
                )
            )

        internal_edge_records = []
        internal_edge_map = {}
        for parsed_id, edge in enumerate(internal_edge_items):
            attrs = edge.get_attributes()
            local_id = int(_strip_quotes(attrs.get("id", parsed_id)))
            internal_edge_map[parsed_id] = local_id
            original_loop_coeffs = tuple(_routing_coeff(attrs, key) for key in k_keys)
            original_ext_coeffs = tuple(_routing_coeff(attrs, key) for key in p_keys)
            ext_coeffs = self._edge_external_coefficients(
                edge,
                original_ext_coeffs,
                ext_routings,
                node_ext_potentials,
                external_nodes,
            )
            loop_coeffs = tuple(
                original_loop_coeffs[loop_id]
                - sum(
                    ext_coeffs[ext_id] * ext_routings[ext_id][0][loop_id]
                    for ext_id in range(len(ext_coeffs))
                )
                for loop_id in range(len(k_keys))
            )
            internal_edge_records.append(
                (parsed_id, edge, local_id, loop_coeffs, ext_coeffs)
            )

        active_loop_indices = _independent_column_indices(
            [record[3] for record in internal_edge_records]
        )
        loop_names = tuple(f"k{i + 1}" for i in range(len(active_loop_indices)))

        internal_edges = []
        for parsed_id, edge, local_id, loop_coeffs, ext_coeffs in internal_edge_records:
            reduced_loop_coeffs = tuple(loop_coeffs[idx] for idx in active_loop_indices)
            label = self._format_label(
                reduced_loop_coeffs,
                ext_coeffs,
                loop_names,
                ext_names,
            )
            attrs = edge.get_attributes()
            internal_edges.append(
                GLTD_GRAPH_IO.ParsedGraphInternalEdge(
                    edge_id=parsed_id,
                    tail=node_name_to_internal[_node_base(edge.get_source())],
                    head=node_name_to_internal[_node_base(edge.get_destination())],
                    label=label,
                    mass_key=_mass_key(attrs),
                    signature=(reduced_loop_coeffs, ext_coeffs),
                    had_pow=False,
                )
            )

        external_edges = []
        external_edge_map = {}
        for ext_id, edge in enumerate(external_edge_items):
            attrs = edge.get_attributes()
            local_id = int(_strip_quotes(attrs.get("id", ext_id)))
            external_edge_map[ext_id] = local_id
            src = _node_base(edge.get_source())
            dst = _node_base(edge.get_destination())
            ext_coeffs = [0] * len(ext_names)
            ext_coeffs[ext_id] = 1
            external_edges.append(
                GLTD_GRAPH_IO.ParsedGraphExternalEdge(
                    edge_id=ext_id,
                    source=src,
                    destination=dst,
                    label=edge_ext_names[ext_id],
                    ext_coeffs=tuple(ext_coeffs),
                )
            )

        return (
            GLTD_GRAPH_IO.ParsedGraph(
                internal_edges=tuple(internal_edges),
                external_edges=tuple(external_edges),
                loop_names=loop_names,
                ext_names=ext_names,
                node_name_to_internal=node_name_to_internal,
            ),
            internal_edge_map,
            external_edge_map,
        )

    @staticmethod
    def _node_external_potentials(
        external_edge_items: list[pydot.Edge],
        external_nodes: set[str],
    ) -> dict[str, tuple[int, ...]] | None:
        size = len(external_edge_items)
        potentials: dict[str, tuple[int, ...]] = {}
        for ext_id, edge in enumerate(external_edge_items):
            src = _node_base(edge.get_source())
            dst = _node_base(edge.get_destination())
            src_ext = src in external_nodes or src.startswith("ext")
            dst_ext = dst in external_nodes or dst.startswith("ext")
            unit = [0] * size
            unit[ext_id] = 1
            if src_ext and not dst_ext:
                node = dst
                potential = tuple(-value for value in unit)
            elif dst_ext and not src_ext:
                node = src
                potential = tuple(unit)
            else:
                continue
            existing = potentials.get(node)
            if existing is None:
                potentials[node] = potential
            else:
                potentials[node] = tuple(
                    existing[idx] + potential[idx] for idx in range(size)
                )
        return potentials

    @staticmethod
    def _edge_external_coefficients(
        edge: pydot.Edge,
        original_ext_coeffs: tuple[int, ...],
        ext_routings: list[tuple[tuple[int, ...], tuple[int, ...]]],
        node_ext_potentials: dict[str, tuple[int, ...]] | None,
        external_nodes: set[str],
    ) -> tuple[int, ...]:
        actual_external_count = len(ext_routings) - len(original_ext_coeffs)
        src = _node_base(edge.get_source())
        dst = _node_base(edge.get_destination())
        src_ext = src in external_nodes or src.startswith("ext")
        dst_ext = dst in external_nodes or dst.startswith("ext")
        if (
            node_ext_potentials is not None
            and not src_ext
            and not dst_ext
            and src in node_ext_potentials
            and dst in node_ext_potentials
        ):
            actual_coeffs = tuple(
                node_ext_potentials[src][idx] - node_ext_potentials[dst][idx]
                for idx in range(actual_external_count)
            )
            residual = tuple(
                original_ext_coeffs[p_id]
                - sum(
                    actual_coeffs[ext_id] * ext_routings[ext_id][1][p_id]
                    for ext_id in range(actual_external_count)
                )
                for p_id in range(len(original_ext_coeffs))
            )
            return (*actual_coeffs, *residual)

        return GeneralisedCFFStructure._external_coefficients(
            original_ext_coeffs,
            [routing[1] for routing in ext_routings],
        )

    def _external_energy_map(self) -> dict[int, _ExternalEnergy]:
        out = {}
        for ext_id, local_id in self.external_edge_map.items():
            out[ext_id] = _ExternalEnergy(
                denominator=CFFStructure.SB["E"](local_id),
                numerator=E(f"En({local_id})"),
            )

        physical_offset = len(self.external_edge_map)
        for physical_id in range(len(self.parsed.ext_names) - physical_offset):
            external_id = physical_offset + physical_id
            momentum_id = physical_id + 1
            energy = E(
                f"(sp3D(p({momentum_id}),p({momentum_id})))^(1/2)"
            )
            out[external_id] = _ExternalEnergy(
                denominator=energy,
                numerator=energy,
            )
        return out

    @staticmethod
    def _external_coefficients(
        target: tuple[int, ...],
        external_vectors: list[tuple[int, ...]],
    ) -> tuple[int, ...]:
        if not external_vectors:
            if any(target):
                raise ValueError(
                    "Could not represent graph external routing: no external edges."
                )
            return tuple()

        row_count = len(target)
        col_count = len(external_vectors)
        rows = [
            [
                Fraction(external_vectors[col][row])
                for col in range(col_count)
            ]
            + [Fraction(target[row])]
            for row in range(row_count)
        ]

        pivot_cols = []
        pivot_row = 0
        for col in range(col_count):
            pivot = None
            for row in range(pivot_row, row_count):
                if rows[row][col] != 0:
                    pivot = row
                    break
            if pivot is None:
                continue
            rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
            pivot_value = rows[pivot_row][col]
            rows[pivot_row] = [value / pivot_value for value in rows[pivot_row]]
            for row in range(row_count):
                if row == pivot_row:
                    continue
                factor = rows[row][col]
                if factor:
                    rows[row] = [
                        value - factor * pivot_value
                        for value, pivot_value in zip(rows[row], rows[pivot_row])
                    ]
            pivot_cols.append(col)
            pivot_row += 1
            if pivot_row == row_count:
                break

        for row in range(row_count):
            if all(rows[row][col] == 0 for col in range(col_count)) and rows[row][-1] != 0:
                raise ValueError(
                    f"Could not represent graph external routing {target} "
                    f"with external vectors {external_vectors}."
                )

        solution = [Fraction(0)] * col_count
        for row, col in enumerate(pivot_cols):
            solution[col] = rows[row][-1]
        if any(value.denominator != 1 for value in solution):
            raise ValueError(
                f"Non-integral external routing decomposition for {target}: {solution}."
            )
        return tuple(int(value) for value in solution)

    @staticmethod
    def _format_label(
        loop_coeffs: tuple[int, ...],
        ext_coeffs: tuple[int, ...],
        loop_names: tuple[str, ...],
        ext_names: tuple[str, ...],
    ) -> str:
        terms = []
        for coeff, name in [*zip(loop_coeffs, loop_names), *zip(ext_coeffs, ext_names)]:
            if coeff == 0:
                continue
            if coeff == 1:
                terms.append(name)
            elif coeff == -1:
                terms.append(f"-{name}")
            else:
                terms.append(f"{coeff}*{name}")
        return " + ".join(terms).replace("+ -", "- ") if terms else "0"

    def _build_structure(self) -> dict[str, Any]:
        validation = GLTD_VALIDATOR.validate_parsed_graph(self.parsed)
        bundle, backend = _build_bundle(
            self.parsed,
            "cff",
            energy_degree_bounds=self.energy_degree_bounds,
        )
        data = GLTD_STRUCTURE.minimal_structure_from_bundle(
            bundle,
            self.parsed,
            backend,
            "cff",
            validation,
        )
        data["graph"].update(GLTD_GRAPH_IO.graph_info(self.parsed))
        if self.energy_degree_bounds is not None:
            bounds = normalize_energy_degree_bounds(
                self.energy_degree_bounds,
                len(self.parsed.internal_edges),
            )
            data["graph"]["energy_degree_bounds"] = list(bounds or [])
        return data

    def _build_wrappers(self) -> None:
        for surface in self.data["surfaces"]:
            wrapper = GeneralisedCFFSurface(
                surface,
                self.internal_edge_map,
                self.external_energy_map,
            )
            if wrapper.kind == "h":
                self.h_surfaces.append(wrapper)
            else:
                self.e_surfaces.append(wrapper)

        for orientation in self.data["orientations"]:
            variants = orientation.get("variants") or [orientation]
            max_local_id = max(self.internal_edge_map.values(), default=-1)
            if self.external_edge_map:
                max_local_id = max(max_local_id, max(self.external_edge_map.values()))
            edge_q0 = [E("0")] * (max_local_id + 1)
            orientation_items = [EdgeOrientation.UNDIRECTED] * (max_local_id + 1)
            for local_id in self.external_edge_map.values():
                edge_q0[local_id] = E(f"En({local_id})")
                orientation_items[local_id] = EdgeOrientation.DEFAULT
            for parsed_id, edge_expr in enumerate(orientation["edge_q0"]):
                local_id = self.internal_edge_map[int(parsed_id)]
                edge_q0[local_id] = self._linear_expr(
                    edge_expr,
                    use_numerator_energies=True,
                )
            for parsed_id, sign in enumerate(orientation.get("edge_signs", [])):
                local_id = self.internal_edge_map[int(parsed_id)]
                orientation_items[local_id] = self._edge_orientation(sign)
            orientation_tuple = tuple(orientation_items)

            for variant in variants:
                expr = _expr_number(variant["pref"])
                expr *= self._relative_half_edge_factor(variant)
                for surface_id in variant.get("num_surfaces", []):
                    expr *= self._surface_by_id(int(surface_id)).numerator_expression
                expr *= self._tree_expression(variant["tree"])
                self.expressions.append(
                    GeneralisedCFFTerm(
                        id=len(self.expressions),
                        orientation=orientation_tuple,
                        expression=expr,
                        edge_q0=edge_q0,
                    )
                )

    @staticmethod
    def _edge_orientation(sign: int) -> EdgeOrientation:
        if int(sign) > 0:
            return EdgeOrientation.DEFAULT
        if int(sign) < 0:
            return EdgeOrientation.REVERSED
        return EdgeOrientation.UNDIRECTED

    def _surface_by_id(self, surface_id: int) -> GeneralisedCFFSurface:
        for surface in [*self.e_surfaces, *self.h_surfaces]:
            if surface.id == surface_id:
                return surface
        raise KeyError(f"Unknown generalised_ltd surface id {surface_id}")

    def _linear_expr(
        self,
        data: dict[str, Any],
        use_numerator_energies: bool,
    ) -> Expression:
        out = _expr_number(data.get("c", "0"))
        for edge_id, coeff in data.get("i", []):
            local_id = self.internal_edge_map[int(edge_id)]
            if use_numerator_energies:
                energy = E(f"En({local_id})")
            else:
                energy = CFFStructure.SB["E"](local_id)
            out += _expr_number(coeff) * energy
        for ext_id, coeff in data.get("x", []):
            external_energy = self.external_energy_map[int(ext_id)]
            energy = (
                external_energy.numerator
                if use_numerator_energies
                else external_energy.denominator
            )
            out += _expr_number(coeff) * energy

        uniform_coeff = int(data.get("m", 0) or 0)
        if uniform_coeff:
            out += _expr_number(uniform_coeff) * E("M")
        return out

    def _relative_half_edge_factor(self, variant: dict[str, Any]) -> Expression:
        counts = Counter(self.internal_edge_map[int(edge)] for edge in variant["half_edges"])
        out = E("1")
        for local_id in self.internal_edge_map.values():
            power = counts.get(local_id, 0) - 1
            half_edge = _expr_number(2) * CFFStructure.SB["E"](local_id)
            if power > 0:
                out /= half_edge**power
            elif power < 0:
                out *= half_edge ** (-power)
        uniform_power = int(variant.get("uniform_scale_power", 0) or 0)
        if uniform_power:
            out /= E("M")**uniform_power
        return out

    def _tree_expression(self, tree: dict[str, Any]) -> Expression:
        def node_expression(node_id: int) -> Expression:
            node = tree["nodes"][node_id]
            factor = E("1")
            for surface_id in node.get("surfaces", []):
                factor /= self._surface_by_id(int(surface_id)).expression
            children = node.get("children", [])
            if not children:
                return factor

            subtotal = E("0")
            for child in children:
                subtotal += node_expression(int(child))
            return factor * subtotal

        out = E("0")
        for root in tree["roots"]:
            out += node_expression(int(root))
        return out

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


def build_generalised_cff_structure(
    graph: pydot.Dot,
    energy_degree_bounds: list[int] | tuple[int, ...] | dict[int, int] | None = None,
) -> GeneralisedCFFStructure:
    return GeneralisedCFFStructure(
        deepcopy(graph),
        energy_degree_bounds=energy_degree_bounds,
    )
