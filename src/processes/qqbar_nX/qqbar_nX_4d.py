from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any

import pydot
from symbolica import E, Expression, Replacement  # pyright: ignore
from symbolica.community.idenso import cook_indices, simplify_color  # pyright: ignore
from symbolica.community.spenso import TensorLibrary, TensorNetwork  # pyright: ignore

from processes.qqbar_nX.qqbar_nX_graphs import graph_name, strip_quotes
from utils.polarizations import ixxxxx, oxxxxx
from utils.utils import DotGraph, Es, ParamBuilder, PygloopEvaluator, pygloopException


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


def _is_dummy_edge(edge: pydot.Edge) -> bool:
    value = edge.get_attributes().get("is_dummy", "false")
    return strip_quotes(value).lower() == "true"


def _propagator_denominators_4d(graph: DotGraph) -> Expression:
    denominator = E("1")
    for edge in graph.get_internal_edges():
        if _is_dummy_edge(edge):
            continue
        edge_id = strip_quotes(edge.get("id"))
        edge_denominator = E(f"gammalooprs::Q({edge_id},spenso::cind(0))^2")
        edge_denominator -= E(f"gammalooprs::Q({edge_id},spenso::cind(1))^2")
        edge_denominator -= E(f"gammalooprs::Q({edge_id},spenso::cind(2))^2")
        edge_denominator -= E(f"gammalooprs::Q({edge_id},spenso::cind(3))^2")
        mass = edge.get_attributes().get("mass")
        if mass is not None:
            edge_denominator -= Es(f"{strip_quotes(mass)}^2")
        denominator *= edge_denominator
    return denominator


def _replace_ose_with_four_dimensional_spatial_energy(
    expression: Expression, graph: DotGraph
) -> Expression:
    replacements: list[Replacement] = []
    for edge in graph.get_internal_edges():
        edge_id = strip_quotes(edge.get("id"))
        energy_squared = E(f"gammalooprs::Q({edge_id},spenso::cind(1))^2")
        energy_squared += E(f"gammalooprs::Q({edge_id},spenso::cind(2))^2")
        energy_squared += E(f"gammalooprs::Q({edge_id},spenso::cind(3))^2")
        mass = edge.get_attributes().get("mass")
        if mass is not None:
            energy_squared += Es(f"{strip_quotes(mass)}^2")
        on_shell_energy = energy_squared ** E("1/2")
        replacements.append(Replacement(E(f"gammalooprs::OSE({edge_id})"), on_shell_energy))
        replacements.append(Replacement(E(f"OSE({edge_id})"), on_shell_energy))
    return expression.replace_multiple(replacements)


def _xi_symbol(name: str) -> Expression:
    if "::" in name:
        return E(name)
    return E(f"gammalooprs::{name}")


def _scalar_expression_from_graph(dot_graph: pydot.Dot) -> Expression:
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
    expression = tensor_network.result_scalar() / _propagator_denominators_4d(graph)
    expression = _replace_ose_with_four_dimensional_spatial_energy(expression, graph)

    expression = expression.replace_multiple(
        [Replacement(lhs, rhs) for lhs, rhs in graph.get_emr_replacements()]
    )
    expression = expression.replace(E("spenso::cind(x_)"), E("x_"), repeat=True)
    expression = expression.replace(E("spenso::TR"), E("1/2"), repeat=True)
    expression = expression.replace(E("spenso::CF"), E("4/3"), repeat=True)
    return expression


@dataclass
class FourDGraphEvaluator:
    name: str
    evaluator: PygloopEvaluator
    external_count: int
    model_values: dict[str, complex]
    xi_parameter_names: tuple[str, str, str, str]
    xi_default_values: tuple[float, float, float, float]

    def set_kinematics(
        self,
        *,
        external_momenta: list[list[float]],
        loop_momenta: list[list[float]],
        helicities: list[int],
    ) -> None:
        if len(external_momenta) != self.external_count:
            raise pygloopException(
                f"4D evaluator for {self.name} expected {self.external_count} "
                f"external momenta, got {len(external_momenta)}."
            )
        if len(loop_momenta) != 2:
            raise pygloopException("qqbar_nX 4D test expects two loop momenta.")

        param_builder = self.evaluator.param_builder
        for ext_id, momentum in enumerate(external_momenta):
            if len(momentum) != 4:
                raise pygloopException("External 4D momentum must have four entries.")
            for component, value in enumerate(momentum):
                _set_param(
                    param_builder,
                    E(f"gammalooprs::P({ext_id},{component})"),
                    value,
                )

        for loop_id, momentum in enumerate(loop_momenta):
            if len(momentum) != 4:
                raise pygloopException("Loop 4D momentum must have four entries.")
            for component, value in enumerate(momentum):
                _set_param(
                    param_builder,
                    E(f"gammalooprs::K({loop_id},{component})"),
                    value,
                )

        if len(helicities) < 2:
            raise pygloopException("qqbar_nX 4D test needs two initial helicities.")
        with contextlib.redirect_stdout(io.StringIO()):
            u_spinor = ixxxxx(external_momenta[0], 0.0, helicities[0], 1)[2:]
        vbar_spinor = oxxxxx(external_momenta[1], 0.0, helicities[1], -1)[2:]
        for component, value in enumerate(u_spinor):
            _set_param(param_builder, E(f"gammalooprs::u(0,{component})"), value)
        for component, value in enumerate(vbar_spinor):
            _set_param(param_builder, E(f"gammalooprs::vbar(1,{component})"), value)

        for symbol_name, value in self.model_values.items():
            _set_param(param_builder, E(symbol_name), value)
        for symbol_name, value in zip(
            self.xi_parameter_names, self.xi_default_values, strict=True
        ):
            _set_param(param_builder, _xi_symbol(symbol_name), value)

    def evaluate(self) -> complex:
        return complex(self.evaluator.evaluate(eager=True)[0])


def build_4d_graph_evaluator(
    dot_graph: pydot.Dot,
    *,
    external_count: int,
    model_values: dict[str, complex],
    xi_parameter_names: tuple[str, str, str, str],
    xi_default_values: tuple[float, float, float, float],
) -> FourDGraphEvaluator:
    expression = _scalar_expression_from_graph(dot_graph)

    symbols: list[Expression] = []
    for ext_id in range(external_count):
        for component in range(4):
            symbols.append(E(f"gammalooprs::P({ext_id},{component})"))
    for loop_id in range(2):
        for component in range(4):
            symbols.append(E(f"gammalooprs::K({loop_id},{component})"))
    for component in range(4):
        symbols.append(E(f"gammalooprs::u(0,{component})"))
    for component in range(4):
        symbols.append(E(f"gammalooprs::vbar(1,{component})"))
    for symbol_name in model_values:
        symbols.append(E(symbol_name))
    for symbol_name in xi_parameter_names:
        symbols.append(_xi_symbol(symbol_name))

    param_builder = _build_param_builder(symbols)
    evaluator = expression.evaluator(
        constants={},
        functions={},
        params=param_builder.get_parameters(),
        iterations=10,
        n_cores=1,
        verbose=False,
    )
    return FourDGraphEvaluator(
        name=graph_name(dot_graph),
        evaluator=PygloopEvaluator(evaluator, param_builder, graph_name(dot_graph)),
        external_count=external_count,
        model_values=model_values,
        xi_parameter_names=xi_parameter_names,
        xi_default_values=xi_default_values,
    )
