from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from symbolica import E, Evaluator, Expression, Replacement, S  # isort: skip # noqa: F401
from symbolica.community.idenso import simplify_gamma, simplify_metrics, simplify_color, cook_indices  # isort: skip # noqa: F401
from symbolica.community.spenso import TensorLibrary, TensorNetwork  # noqa: F401
from ufo_model_loader.commands import Model  # noqa: F401

from utils.utils import DotGraph, DotGraphs, expr_to_string

if TYPE_CHECKING:
    from processes.scalar_gravity.scalar_gravity import ScalarGravity


class ClassicalLimitProcessor(object):
    def __init__(self, process: ScalarGravity):
        self.process = process

    def get_color_projector(self) -> Expression:
        return E("1")

    def process_graphs(self, graphs: DotGraphs) -> DotGraphs:
        processed_graphs = DotGraphs()

        for group_id, g_input in enumerate(graphs):
            g: DotGraph = copy.deepcopy(g_input)

            # Add the main graph
            attrs = g.get_attributes()
            attrs["group_master"] = '"true"'
            attrs["group_id"] = f'"{group_id}"'
            attrs["num"] = f'"{expr_to_string(g.get_numerator())}"'
            attrs["projector"] = f'"{expr_to_string(g.get_projector() * self.get_color_projector())}"'
            g.set_local_numerators_to_one()
            processed_graphs.append(g)

            # As an example, add a fake UV equal to the original graph
            fake_uv: DotGraph = copy.deepcopy(g_input)
            fake_uv.dot.set_name(f"{g.dot.get_name()}_UV_1")
            attrs = fake_uv.get_attributes()
            attrs["group_id"] = f'"{group_id}"'
            attrs["num"] = f'"{expr_to_string(g.get_numerator())}"'
            attrs["projector"] = f'"{expr_to_string(g.get_projector() * self.get_color_projector())}"'
            g.set_local_numerators_to_one()
            processed_graphs.append(fake_uv)

        return processed_graphs
