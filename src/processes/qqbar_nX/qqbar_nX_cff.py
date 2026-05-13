from __future__ import annotations

import contextlib
import io
import json
import math
import os
import subprocess
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Any

import pydot
from symbolica import E, Expression, Replacement  # pyright: ignore
from symbolica.community.idenso import cook_indices, simplify_color  # pyright: ignore
from symbolica.community.spenso import TensorLibrary, TensorNetwork  # pyright: ignore

from processes.qqbar_nX.qqbar_nX_graphs import graph_name, strip_quotes
from utils.polarizations import ixxxxx, oxxxxx
from utils.utils import DotGraph, Es, ParamBuilder, PygloopEvaluator, pygloopException


LMB_TERM_RE = __import__("re").compile(
    r"(?P<coef>[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)\*)?)"
    r"(?P<kind>[KP])\((?P<index>\d+),"
)


SB = {
    "E": E("pygloop::E"),
    "Qspatial": E("pygloop::Qspatial"),
    "uniform_scale": E("pygloop::M"),
}
DECIMAL_PI = Decimal(
    "3.141592653589793238462643383279502884197169399375105820974944592307816406286"
)


def _energy_symbol(edge_id: int) -> Expression:
    return E(f"pygloop::E({edge_id})")


def _qspatial_symbol(edge_id: int, component: int) -> Expression:
    return E(f"pygloop::Qspatial({edge_id},{component})")


def _external_symbol(external_id: int, component: int) -> Expression:
    return E(f"gammalooprs::P({external_id},{component})")


def _loop_spatial_symbol(loop_id: int, component: int) -> Expression:
    return E(f"gammalooprs::K({loop_id},{component})")


def _u_symbol(component: int) -> Expression:
    return E(f"gammalooprs::u(0,{component})")


def _vbar_symbol(component: int) -> Expression:
    return E(f"gammalooprs::vbar(1,{component})")


def _build_param_builder(symbols: list[Expression]) -> ParamBuilder:
    param_builder = ParamBuilder()
    param_builder.add_parameter_list((E("dummy"),), len(symbols))
    param_builder.order = []
    param_builder.positions = {}
    for index, symbol in enumerate(symbols):
        head = (symbol,)
        param_builder.order.append(head)
        param_builder.positions[head] = (index, index + 1)
        param_builder.np[index] = 0.0
    return param_builder


def _set_param(
    param_builder: ParamBuilder,
    symbol: Expression,
    value: complex | float,
) -> None:
    param_builder.set_parameter(
        (symbol,), complex(value), check_phase_flag_consistency=False
    )


def _decimal_from_number(value: Decimal | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise pygloopException(f"Cannot convert non-finite float {value} to Decimal.")
        return Decimal(repr(value))
    return Decimal(str(value))


def _decimal_complex_parts(value: complex | float | int | Decimal) -> tuple[Decimal, Decimal]:
    if isinstance(value, complex):
        return _decimal_from_number(value.real), _decimal_from_number(value.imag)
    if (
        hasattr(value, "real")
        and hasattr(value, "imag")
        and not isinstance(value, (Decimal, float, int))
    ):
        complex_value = complex(value)
        return (
            _decimal_from_number(complex_value.real),
            _decimal_from_number(complex_value.imag),
        )
    return _decimal_from_number(value), Decimal(0)


def _decimal_sqrt(value: Decimal, precision: int) -> Decimal:
    with localcontext() as context:
        context.prec = max(precision + 10, 50)
        return +value.sqrt()


def _is_dummy_edge(edge: pydot.Edge) -> bool:
    return strip_quotes(edge.get_attributes().get("is_dummy", "false")).lower() == "true"


def _parse_lmb_representation(lmb_rep: str) -> dict[str, dict[int, float]]:
    cleaned = strip_quotes(lmb_rep).replace(" ", "")
    if cleaned in {"", "0"}:
        return {"K": {}, "P": {}}
    coefficients: dict[str, dict[int, float]] = {"K": {}, "P": {}}
    normalized = cleaned
    if normalized[0] not in "+-":
        normalized = "+" + normalized
    for match in LMB_TERM_RE.finditer(normalized):
        raw_coef = match.group("coef")
        if raw_coef in {"", "+"}:
            coef = 1.0
        elif raw_coef == "-":
            coef = -1.0
        else:
            coef = float(raw_coef.rstrip("*"))
        kind = match.group("kind")
        index = int(match.group("index"))
        coefficients[kind][index] = coefficients[kind].get(index, 0.0) + coef
    return coefficients


def _json_linear_combination_to_expr(data: dict[str, Any]) -> Expression:
    expression = E(str(data.get("constant", "0")))
    uniform_coeff = str(data.get("uniform_scale_coeff", "0"))
    if uniform_coeff != "0":
        expression += E(f"({uniform_coeff})") * SB["uniform_scale"]
    for edge_id, coeff in data.get("internal_terms", []):
        expression += E(f"({coeff})") * _energy_symbol(int(edge_id))
    for ext_id, coeff in data.get("external_terms", []):
        expression += E(f"({coeff})") * _external_symbol(int(ext_id), 0)
    return expression


class CFFMetaExpression:
    def __init__(self, data: dict[str, Any]):
        expression = data.get("expression")
        if not isinstance(expression, dict):
            raise pygloopException("Malformed CFF JSON: missing expression object.")
        self.raw = data
        self.expression = expression
        self.surfaces = expression.get("surfaces", {}).get("linear_surface_cache", [])
        self.orientations = expression.get("orientations", [])

    def surface_expr(self, surface_id: int) -> Expression:
        try:
            surface = self.surfaces[surface_id]
        except IndexError as exc:
            raise pygloopException(f"CFF surface id {surface_id} is unavailable.") from exc
        return _json_linear_combination_to_expr(surface["expression"])

    def denominator_tree_inverse(self, tree: dict[str, Any]) -> Expression:
        nodes = tree.get("nodes")
        if not nodes:
            return E("1")
        by_id = {int(node["node_id"]): node for node in nodes}

        def visit(node_id: int) -> Expression:
            node = by_id[node_id]
            child_expr = E("0")
            for child_id in node.get("children", []):
                child_expr += visit(int(child_id))
            if not node.get("children"):
                child_expr = E("1")

            data = node.get("data")
            if data == "Unit":
                return child_expr
            if isinstance(data, dict) and "Linear" in data:
                return child_expr / self.surface_expr(int(data["Linear"]))
            raise pygloopException(f"Unsupported CFF denominator-tree node: {data!r}")

        return visit(0)


def _graph_scalar_numerator(dot_graph: pydot.Dot) -> Expression:
    graph = DotGraph(dot_graph)
    numerator = simplify_color(
        graph.get_numerator(include_overall_factor=True) * graph.get_projector()
    )
    numerator = numerator.replace(
        E("spenso::projm(x_,y_)+spenso::projp(x_,y_)"),
        E("spenso::g(x_,y_)"),
        repeat=True,
    )
    hep_library = TensorLibrary.hep_lib_atom()  # type: ignore
    tensor_network = TensorNetwork(cook_indices(numerator), hep_library)
    tensor_network.execute(hep_library)
    expression = tensor_network.result_scalar()
    expression = expression.replace(E("spenso::cind(x_)"), E("x_"), repeat=True)
    expression = expression.replace(E("spenso::TR"), E("1/2"), repeat=True)
    expression = expression.replace(E("spenso::CF"), E("4/3"), repeat=True)
    return expression


def _edge_q0_replacements(
    cff: CFFMetaExpression, orientation: dict[str, Any]
) -> list[Replacement]:
    replacements = []
    for edge_id, energy_map in enumerate(orientation.get("edge_energy_map", [])):
        replacements.append(
            Replacement(
                E(f"gammalooprs::Q({edge_id},0)"),
                _json_linear_combination_to_expr(energy_map),
            )
        )
    return replacements


def _spatial_q_replacements(max_edge_id: int) -> list[Replacement]:
    replacements = []
    for edge_id in range(max_edge_id + 1):
        for component in range(1, 4):
            replacements.append(
                Replacement(
                    E(f"gammalooprs::Q({edge_id},{component})"),
                    _qspatial_symbol(edge_id, component - 1),
                )
            )
    return replacements


def _cff_weight_for_variant(
    cff: CFFMetaExpression, variant: dict[str, Any]
) -> Expression:
    weight = Es(str(variant.get("prefactor", "1")))
    for edge_id in variant.get("half_edges", []):
        weight /= E("2") * _energy_symbol(int(edge_id))
    for surface_id in variant.get("numerator_surfaces", []):
        weight *= cff.surface_expr(int(surface_id))
    weight *= cff.denominator_tree_inverse(variant.get("denominator", {"nodes": []}))
    uniform_power = int(variant.get("uniform_scale_power", 0))
    if uniform_power:
        weight *= SB["uniform_scale"] ** E(str(uniform_power))
    return weight


def build_cff_integrand_expression(
    dot_graph: pydot.Dot,
    cff_data: dict[str, Any],
    *,
    orientation_id: int | None = None,
) -> Expression:
    cff = CFFMetaExpression(cff_data)
    graph = DotGraph(dot_graph)
    internal_edges = graph.get_internal_edges()
    max_internal_edge_id = max(
        (int(strip_quotes(edge.get("id"))) for edge in internal_edges),
        default=0,
    )
    numerator = _graph_scalar_numerator(dot_graph)
    numerator = numerator.replace_multiple(
        _spatial_q_replacements(max_internal_edge_id), repeat=True
    )

    total = E("0")
    orientations = cff.orientations
    if orientation_id is not None:
        if orientation_id < 0 or orientation_id >= len(orientations):
            raise pygloopException(
                f"CFF orientation {orientation_id} is outside [0,{len(orientations)})."
            )
        orientations = [orientations[orientation_id]]

    for orientation in orientations:
        localized_numerator = numerator.replace_multiple(
            _edge_q0_replacements(cff, orientation), repeat=True
        )
        for variant in orientation.get("variants", []):
            total += localized_numerator * _cff_weight_for_variant(cff, variant)
    return total


def _edge_mass(edge: pydot.Edge, model_values: dict[str, complex]) -> float:
    mass = edge.get_attributes().get("mass")
    if mass is not None:
        try:
            return abs(float(strip_quotes(mass)))
        except ValueError:
            pass
    particle = strip_quotes(edge.get_attributes().get("particle", ""))
    if particle in {"t", "t~"}:
        return abs(float(model_values.get("UFO::MT", 0.0).real))
    return 0.0


def _edge_mass_decimal(
    edge: pydot.Edge,
    model_values: dict[str, complex],
    external_momenta: list[list[Decimal]] | None,
    precision: int,
) -> Decimal:
    mass = edge.get_attributes().get("mass")
    if mass is not None:
        mass_text = strip_quotes(mass)
        if not mass_text:
            mass = None
        else:
            try:
                return abs(_decimal_from_number(mass_text))
            except Exception:
                if external_momenta is None:
                    raise pygloopException(
                        f"Cannot evaluate symbolic mass expression {mass_text!r} "
                        "without external momenta."
                    )
                expression = Es(mass_text)
                replacements = []
                for edge_id, momentum in enumerate(external_momenta):
                    for component, value in enumerate(momentum):
                        replacements.append(
                            Replacement(
                                E(f"gammalooprs::Q({edge_id},spenso::cind({component}))"),
                                E(str(value)),
                            )
                        )
                        replacements.append(
                            Replacement(
                                E(f"gammalooprs::Q({edge_id},{component})"),
                                E(str(value)),
                            )
                        )
                localized = expression.replace_multiple(replacements, repeat=True)
                return abs(localized.evaluate_with_prec({}, {}, precision))

    particle = strip_quotes(edge.get_attributes().get("particle", ""))
    if particle in {"t", "t~"}:
        return abs(_decimal_from_number(model_values.get("UFO::MT", 0.0).real))
    return Decimal(0)


@dataclass
class CFFGraphEvaluator:
    name: str
    evaluator: PygloopEvaluator
    dot_graph: pydot.Dot
    external_count: int
    lmb_external_edge_map: dict[int, int]
    model_values: dict[str, complex]
    normalization: complex
    n_loops: int
    uniform_scale: float = 1.0
    _decimal_values: list[tuple[Decimal, Decimal]] | None = None

    def _reset_decimal_values(self) -> None:
        self._decimal_values = [
            _decimal_complex_parts(complex(value))
            for value in self.evaluator.param_builder.np.tolist()
        ]

    def _set_param_value(
        self,
        symbol: Expression,
        value: complex | float,
        decimal_value: tuple[Decimal, Decimal] | Decimal | None = None,
    ) -> None:
        _set_param(self.evaluator.param_builder, symbol, value)
        if self._decimal_values is None:
            return
        if decimal_value is None:
            real, imag = _decimal_complex_parts(value)
        elif isinstance(decimal_value, tuple):
            real, imag = decimal_value
        else:
            real, imag = decimal_value, Decimal(0)
        position = self.evaluator.param_builder.positions.get((symbol,))
        if position is None:
            raise pygloopException(
                f"CFF evaluator for {self.name} has no parameter {symbol}."
            )
        start, stop = position
        if stop - start != 1:
            raise pygloopException(
                f"CFF evaluator parameter {symbol} has unexpected width {stop - start}."
            )
        self._decimal_values[start] = (real, imag)

    def set_kinematics(
        self,
        *,
        external_momenta: list[list[float]],
        loop_spatial_momenta: list[list[float]],
        helicities: list[int],
        decimal_digit_precision: int | None = None,
    ) -> None:
        if len(external_momenta) != self.external_count:
            raise pygloopException(
                f"CFF evaluator for {self.name} expected {self.external_count} "
                f"external momenta, got {len(external_momenta)}."
            )
        if len(loop_spatial_momenta) != 2:
            raise pygloopException("qqbar_nX CFF test expects two loop 3-momenta.")

        if decimal_digit_precision is not None:
            self._reset_decimal_values()
        external_decimal = [
            [_decimal_from_number(component) for component in momentum]
            for momentum in external_momenta
        ]
        loop_spatial_decimal = [
            [_decimal_from_number(component) for component in momentum]
            for momentum in loop_spatial_momenta
        ]

        param_builder = self.evaluator.param_builder
        for ext_id, momentum in enumerate(external_momenta):
            for component, value in enumerate(momentum):
                self._set_param_value(
                    _external_symbol(ext_id, component),
                    value,
                    external_decimal[ext_id][component],
                )

        for loop_id, momentum in enumerate(loop_spatial_momenta):
            if len(momentum) != 3:
                raise pygloopException("Loop spatial momentum must have three entries.")
            for component, value in enumerate(momentum):
                self._set_param_value(
                    _loop_spatial_symbol(loop_id, component + 1),
                    value,
                    loop_spatial_decimal[loop_id][component],
                )

        graph = DotGraph(self.dot_graph)
        spatial_external = {
            index: [momentum[1], momentum[2], momentum[3]]
            for index, momentum in enumerate(external_momenta)
        }
        spatial_lmb_external: dict[int, list[float]] = {}
        for lmb_external_id, edge_id in self.lmb_external_edge_map.items():
            try:
                spatial_lmb_external[lmb_external_id] = spatial_external[edge_id]
            except KeyError as exc:
                raise pygloopException(
                    f"CFF evaluator for {self.name} has no momentum for external "
                    f"edge e{edge_id} mapped from LMB external P({lmb_external_id})."
                ) from exc
        spatial_lmb_external_decimal: dict[int, list[Decimal]] = {}
        for lmb_external_id, edge_id in self.lmb_external_edge_map.items():
            try:
                spatial_lmb_external_decimal[lmb_external_id] = [
                    external_decimal[edge_id][1],
                    external_decimal[edge_id][2],
                    external_decimal[edge_id][3],
                ]
            except KeyError as exc:
                raise pygloopException(
                    f"CFF evaluator for {self.name} has no Decimal momentum for "
                    f"external edge e{edge_id} mapped from LMB external "
                    f"P({lmb_external_id})."
                ) from exc
        for edge in graph.get_internal_edges():
            if _is_dummy_edge(edge):
                continue
            edge_id = int(strip_quotes(edge.get("id")))
            coeffs = _parse_lmb_representation(edge.get_attributes().get("lmb_rep", "0"))
            q = [0.0, 0.0, 0.0]
            q_decimal = [Decimal(0), Decimal(0), Decimal(0)]
            for loop_id, coeff in coeffs["K"].items():
                coeff_decimal = _decimal_from_number(coeff)
                for component in range(3):
                    q[component] += coeff * loop_spatial_momenta[loop_id][component]
                    q_decimal[component] += (
                        coeff_decimal * loop_spatial_decimal[loop_id][component]
                    )
            for ext_id, coeff in coeffs["P"].items():
                if ext_id not in spatial_lmb_external:
                    raise pygloopException(
                        f"CFF evaluator for {self.name} cannot resolve LMB external "
                        f"P({ext_id}) in edge e{edge_id}."
                    )
                coeff_decimal = _decimal_from_number(coeff)
                for component in range(3):
                    q[component] += coeff * spatial_lmb_external[ext_id][component]
                    q_decimal[component] += (
                        coeff_decimal
                        * spatial_lmb_external_decimal[ext_id][component]
                    )
            mass = _edge_mass(edge, self.model_values)
            mass_decimal = _edge_mass_decimal(
                edge,
                self.model_values,
                external_decimal,
                decimal_digit_precision or 80,
            )
            energy_decimal = _decimal_sqrt(
                sum(component * component for component in q_decimal)
                + mass_decimal * mass_decimal,
                decimal_digit_precision or 80,
            )
            energy = math.sqrt(sum(component * component for component in q) + mass * mass)
            self._set_param_value(_energy_symbol(edge_id), energy, energy_decimal)
            for component, value in enumerate(q):
                self._set_param_value(
                    _qspatial_symbol(edge_id, component),
                    value,
                    q_decimal[component],
                )

        if len(helicities) < 2:
            raise pygloopException("qqbar_nX CFF test needs two initial helicities.")
        with contextlib.redirect_stdout(io.StringIO()):
            u_spinor = ixxxxx(external_momenta[0], 0.0, helicities[0], 1)[2:]
        vbar_spinor = oxxxxx(external_momenta[1], 0.0, helicities[1], -1)[2:]
        for component, value in enumerate(u_spinor):
            self._set_param_value(
                _u_symbol(component),
                value,
                _decimal_complex_parts(value),
            )
        for component, value in enumerate(vbar_spinor):
            self._set_param_value(
                _vbar_symbol(component),
                value,
                _decimal_complex_parts(value),
            )

        for symbol_name, value in self.model_values.items():
            self._set_param_value(
                E(symbol_name),
                value,
                _decimal_complex_parts(value),
            )
        self._set_param_value(
            SB["uniform_scale"],
            self.uniform_scale,
            _decimal_from_number(self.uniform_scale),
        )

    def evaluate(self) -> complex:
        return self.normalization * complex(self.evaluator.evaluate(eager=True)[0])

    def evaluate_with_prec(
        self,
        decimal_digit_precision: int,
        *,
        use_runtime_float_inputs: bool = True,
    ) -> tuple[Decimal, Decimal]:
        if self._decimal_values is None:
            raise pygloopException(
                "CFF evaluator arbitrary-precision evaluation requires "
                "set_kinematics(..., decimal_digit_precision=N)."
            )
        decimal_values = (
            [
                _decimal_complex_parts(complex(value))
                for value in self.evaluator.param_builder.np.tolist()
            ]
            if use_runtime_float_inputs
            else self._decimal_values
        )
        output = self.evaluator.get_eager_evaluator().evaluate_complex_with_prec(
            decimal_values,
            decimal_digit_precision,
        )
        raw_re, raw_im = output[0]
        with localcontext() as context:
            context.prec = max(decimal_digit_precision + 10, 50)
            denominator = ((Decimal(-2) * DECIMAL_PI) ** 3) ** self.n_loops
            scale = Decimal(1) / denominator
            # normalization = -i / ((-2*pi)^3)^n_loops
            return +(raw_im * scale), +(-raw_re * scale)


def build_cff_graph_evaluator(
    dot_graph: pydot.Dot,
    cff_data: dict[str, Any],
    *,
    external_count: int,
    model_values: dict[str, complex],
    orientation_id: int | None = None,
) -> CFFGraphEvaluator:
    expression = build_cff_integrand_expression(
        dot_graph, cff_data, orientation_id=orientation_id
    )
    graph = DotGraph(dot_graph)
    lmb_external_edge_map: dict[int, int] = {}
    for edge in graph.get_external_edges():
        if _is_dummy_edge(edge):
            continue
        edge_id = int(strip_quotes(edge.get("id")))
        coeffs = _parse_lmb_representation(edge.get_attributes().get("lmb_rep", "0"))
        if len(coeffs["P"]) == 1 and not coeffs["K"]:
            ((external_id, coeff),) = coeffs["P"].items()
            if abs(coeff - 1.0) < 1.0e-12:
                lmb_external_edge_map[external_id] = edge_id
    max_internal_edge_id = max(
        (int(strip_quotes(edge.get("id"))) for edge in graph.get_internal_edges()),
        default=0,
    )
    symbols: list[Expression] = []
    real_symbols: list[Expression] = []
    for edge_id in range(max_internal_edge_id + 1):
        symbol = _energy_symbol(edge_id)
        symbols.append(symbol)
        real_symbols.append(symbol)
        for component in range(3):
            symbol = _qspatial_symbol(edge_id, component)
            symbols.append(symbol)
            real_symbols.append(symbol)
    for ext_id in range(external_count):
        for component in range(4):
            symbol = _external_symbol(ext_id, component)
            symbols.append(symbol)
            real_symbols.append(symbol)
    for loop_id in range(2):
        for component in range(1, 4):
            symbol = _loop_spatial_symbol(loop_id, component)
            symbols.append(symbol)
            real_symbols.append(symbol)
    for component in range(4):
        symbols.append(_u_symbol(component))
        symbols.append(_vbar_symbol(component))
    for symbol_name in model_values:
        symbol = E(symbol_name)
        symbols.append(symbol)
        if abs(model_values[symbol_name].imag) == 0.0:
            real_symbols.append(symbol)
    symbols.append(SB["uniform_scale"])
    real_symbols.append(SB["uniform_scale"])

    param_builder = _build_param_builder(symbols)
    for symbol in real_symbols:
        param_builder.force_parameters_to_real([(symbol,)])
    evaluator = expression.evaluator(
        constants={},
        functions={},
        params=param_builder.get_parameters(),
        iterations=10,
        n_cores=1,
        verbose=False,
    )
    name = graph_name(dot_graph)
    n_loops = len(cff_data.get("graph", {}).get("loop_names", []))
    normalization = -1j / (((-2.0 * math.pi) ** 3) ** n_loops)
    pygloop_evaluator = PygloopEvaluator(evaluator, param_builder, f"{name}_cff_meta")
    pygloop_evaluator.freeze_input_phases()
    return CFFGraphEvaluator(
        name=name,
        evaluator=pygloop_evaluator,
        dot_graph=dot_graph,
        external_count=external_count,
        lmb_external_edge_map=lmb_external_edge_map,
        model_values=model_values,
        normalization=normalization,
        n_loops=n_loops,
    )


def load_or_build_cff_json(
    *,
    api: Any | None,
    gammaloop_cli_path: str,
    state_folder: str,
    integrand_name: str,
    process_name: str,
    graph_name_value: str,
    json_path: str,
    representation: str = "cff",
) -> dict[str, Any]:
    if api is not None and hasattr(api, "get_cff_expression"):
        data = api.get_cff_expression(  # type: ignore[attr-defined]
            process_name=process_name,
            integrand_name=integrand_name,
            graph_name=graph_name_value,
            representation=representation,
        )
        return json.loads(data) if isinstance(data, str) else data

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    command = [
        gammaloop_cli_path,
        "-s",
        state_folder,
        "-o",
        "3Drep",
        "build",
        "-p",
        f"name:{process_name}",
        "-i",
        integrand_name,
        "-g",
        graph_name_value,
        "--representation",
        representation,
        "--json-out",
        json_path,
        "--no-pretty",
        "--no-color",
    ]
    subprocess.run(command, check=True, cwd=os.getcwd())
    with open(json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)
