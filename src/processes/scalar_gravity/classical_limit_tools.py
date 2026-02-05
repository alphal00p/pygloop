from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from symbolica import E, Evaluator, Expression, Replacement, S  # isort: skip # noqa: F401 # type: ignore
from symbolica.community.idenso import (  # type: ignore
    cook_indices,  # noqa: F401
    simplify_color,  # noqa: F401
    simplify_gamma,  # noqa: F401
    simplify_metrics,  # noqa: F401
)  # isort: skip # noqa: F401
from symbolica.community.spenso import (  # noqa: F401 # type: ignore
    TensorLibrary,
    TensorNetwork,
)
from ufo_model_loader.commands import Model  # noqa: F401 # type: ignore

from utils.utils import DotGraph, DotGraphs, expr_to_string

if TYPE_CHECKING:
    from processes.scalar_gravity.scalar_gravity import ScalarGravity


class ClassicalLimitProcessor(object):
    def __init__(self, process: ScalarGravity):
        self.process = process

    def get_color_projector(self) -> Expression:
        return E("1")

    def classical_limit_in_numerator(self, graph: DotGraph) -> None:
        for v in graph.dot.get_nodes():
            int_id = v.get_attributes().get("int_id", "").strip().strip('"')

            if int_id.startswith("V_S1S1"):
                num = v.get_attributes()["num"].strip().strip('"')
                replaced_num = expr_to_string(
                    E(num).replace(
                        E("Q(x_,spenso::mink(4,y_))"),
                        E("Q(0,spenso::mink(4,y_))"),
                    )
                )
                v.get_attributes()["num"] = replaced_num
            if int_id.startswith("V_S2S2"):
                num = v.get_attributes()["num"].strip().strip('"')
                replaced_num = expr_to_string(
                    E(num).replace(
                        E("Q(x_,spenso::mink(4,y_))"),
                        E("Q(1,spenso::mink(4,y_))"),
                    )
                )
                v.get_attributes()["num"] = replaced_num

    def delocalize_numerators(self, g: DotGraph) -> None:
        attrs = g.get_attributes()
        attrs["num"] = f'"{expr_to_string(g.get_numerator())}"'
        g.set_local_numerators_to_one()

    def adjust_projectors(self, g: DotGraph) -> None:
        attrs = g.get_attributes()
        attrs["projector"] = (
            f'"{expr_to_string(g.get_projector() * self.get_color_projector())}"'
        )
        return

    def set_group_id(self, g: DotGraph, group_id: int, is_master: bool = False) -> None:
        attrs = g.get_attributes()
        attrs["group_id"] = f'"{group_id}"'
        if is_master:
            attrs["group_master"] = '"true"'
        else:
            attrs["group_master"] = '"false"'

    def generate_UV_CTs(self, g: DotGraph, group_id: int) -> list[DotGraph]:
        match g.dot.get_name():
            case "v_diagram":
                fake_uv: DotGraph = copy.deepcopy(g)
                fake_uv.dot.set_name(f"{g.dot.get_name()}_UV_1")
                self.set_group_id(fake_uv, group_id, is_master=False)
                return [fake_uv]
            case _:
                # TODO
                return []

    def fix_higher_power_energies(self, g: DotGraph) -> None:
        match g.dot.get_name():
            case "v_diagram":
                # num = g.get_numerator()
                # TODO
                pass
            case _:
                raise NotImplementedError(
                    f"Higher power energy fix not implemented for graph {g.dot.get_name()}"
                )

    def process_graphs(self, graphs: DotGraphs) -> DotGraphs:
        processed_graphs = DotGraphs()

        for group_id, g_input in enumerate(graphs):
            g: DotGraph = copy.deepcopy(g_input)

            # Add the main graph
            self.set_group_id(g, group_id, is_master=True)
            self.classical_limit_in_numerator(g)
            self.adjust_projectors(g)
            # self.delocalize_numerators(g)
            self.fix_higher_power_energies(g)
            processed_graphs.append(g)

            # As an example, add a fake UV equal to the original graph
            processed_graphs.extend(self.generate_UV_CTs(g, group_id))

        return processed_graphs

    def remove_raised_power(self, graph: DotGraph):
        return graph
