from __future__ import annotations

import contextlib
import io
import logging
import math
import sys
from copy import deepcopy
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from processes.dy.dy import DY  # noqa: E402
from processes.dy.dy_classes import DYDotGraphs  # noqa: E402
from processes.dy.dy_infrared_test import approach_point  # noqa: E402
from processes.dy.dy_integrand import (  # noqa: E402
    LoopIntegrandConstructor,
    routed_cut_graph,
)
from utils.vectors import LorentzVector  # noqa: E402


class DYGraphReferenceProcess(DY):
    name = "DY_GRAPH_REFERENCE_TEST"


GENERATE_CONFIG = ROOT / "configs" / "DY" / "generate.toml"
RUNTIME_CONFIG = ROOT / "configs" / "DY" / "runtime.toml"
POINT_Z = 0.6
POINT_KS = [
    math.sqrt(POINT_Z)
    * np.array([1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)])
]
POINT_P1 = [0.0, 0.0, 1.0]
POINT_P2 = [0.0, 0.0, -1.0]
REFERENCE_SUMS = {
    ("ddx", "GL09"): Decimal("-3.324000708917661371207269448"),
    ("ddx", "GL02"): Decimal("0.8187057880619492018150548758"),
    ("ddx", "GL17"): Decimal("-1.108000236305887945763734433"),
    ("dg", "GL04"): Decimal("-2.689173062828762343237578239"),
    ("dg", "GL10"): Decimal("-0.4232184307259338213278532962"),
    ("dg", "GL11"): Decimal("-0.1711954513949885311089702841"),
}
REFERENCE_ROUTED_GRAPH_COUNTS = {
    ("ddx", "GL09"): 8,
    ("ddx", "GL02"): 16,
    ("ddx", "GL17"): 6,
    ("dg", "GL04"): 6,
    ("dg", "GL10"): 4,
    ("dg", "GL11"): 2,
}
GRAPH_COMMANDS = {
    ("ddx", "GL09"): (
        "generate amp d d~ > d d~ | d d~ g a QED==2 [{1}] "
        "--only-diagrams --numerator-grouping only_detect_zeroes "
        "--select-graphs GL09 -p DY_GRAPH_REF_DDX_GL09 "
        "-i DY_GRAPH_REF_DDX_GL09_generated_graphs"
    ),
    ("ddx", "GL02"): (
        "generate amp d d~ > d d~ | d d~ g a QED==2 [{1}] "
        "--only-diagrams --numerator-grouping only_detect_zeroes "
        "--select-graphs GL02 -p DY_GRAPH_REF_DDX_GL02 "
        "-i DY_GRAPH_REF_DDX_GL02_generated_graphs"
    ),
    ("ddx", "GL17"): (
        "generate amp d d~ > d d~ | d d~ g a QED==2 [{1}] "
        "--only-diagrams --numerator-grouping only_detect_zeroes "
        "--select-graphs GL17 -p DY_GRAPH_REF_DDX_GL17 "
        "-i DY_GRAPH_REF_DDX_GL17_generated_graphs"
    ),
    ("dg", "GL04"): (
        "generate amp d g > d g | d d~ g a QED==2 [{1}] "
        "--only-diagrams --numerator-grouping only_detect_zeroes "
        "--select-graphs GL04 -p DY_GRAPH_REF_DG_GL04 "
        "-i DY_GRAPH_REF_DG_GL04_generated_graphs"
    ),
    ("dg", "GL10"): (
        "generate amp d g > d g | d d~ g a QED==2 [{1}] "
        "--only-diagrams --numerator-grouping only_detect_zeroes "
        "--select-graphs GL10 -p DY_GRAPH_REF_DG_GL10 "
        "-i DY_GRAPH_REF_DG_GL10_generated_graphs"
    ),
    ("dg", "GL11"): (
        "generate amp d g > d g | d d~ g a QED==2 [{1}] "
        "--only-diagrams --numerator-grouping only_detect_zeroes "
        "--select-graphs GL11 -p DY_GRAPH_REF_DG_GL11 "
        "-i DY_GRAPH_REF_DG_GL11_generated_graphs"
    ),
}


def _default_ps_point() -> list[LorentzVector]:
    return [
        LorentzVector(500.0, 0.0, 0.0, 500.0),
        LorentzVector(500.0, 0.0, 0.0, -500.0),
        LorentzVector(
            438.5555662246945,
            155.3322001835378,
            348.0160396513587,
            -177.3773615718412,
        ),
        LorentzVector(
            356.3696374921922,
            -16.80238900851100,
            -318.7291102436005,
            97.48719163688098,
        ),
        LorentzVector(
            205.0747962831133,
            -138.5298111750267,
            -29.28692940775817,
            79.89016993496030,
        ),
    ]


def _build_process() -> DYGraphReferenceProcess:
    with contextlib.redirect_stdout(io.StringIO()):
        return DYGraphReferenceProcess(
            m_top=173.0,
            m_higgs=125.0,
            ps_point=_default_ps_point(),
            n_loops=1,
            clean=True,
            logger_level=logging.CRITICAL,
            skip_ps_validation=True,
            toml_config_path=str(GENERATE_CONFIG),
            runtime_toml_config_path=str(RUNTIME_CONFIG),
        )


def _generated_output_name(command: str) -> str:
    return command.split(" -i ", 1)[1]


def _graph_reference_data(
    process: DYGraphReferenceProcess, command: str
) -> tuple[int, Decimal]:
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        process.gl_worker.run(command)

    output_name = _generated_output_name(command)
    amplitudes, _cross_sections = process.gl_worker.list_outputs()
    if output_name not in amplitudes:
        raise AssertionError(f"Generated DY output '{output_name}' was not found.")

    dot_graphs = DYDotGraphs(
        dot_str=process.gl_worker.get_dot_files(
            process_id=amplitudes[output_name],
            integrand_name=output_name,
        )
    )
    filtered_graphs = DYDotGraphs()
    filtered_graphs.extend(deepcopy(dot_graphs.filter_particle_definition(["a"])))
    if len(filtered_graphs) != 1:
        raise AssertionError(
            f"Expected exactly one filtered graph for '{output_name}', got {len(filtered_graphs)}."
        )

    loop_processor = LoopIntegrandConstructor([], "DY", 1)
    routed_graph_count = 0
    routed_integrands = []
    with contextlib.redirect_stdout(capture):
        for graph in filtered_graphs:
            vacuum_graph = graph.get_vacuum_graph()
            routed_graphs = vacuum_graph.cut_graphs_with_routing_leading_virtuality(
                [], ["a"]
            )
            routed_graph_count += len(routed_graphs)
            for gg in routed_graphs:
                cut_graph = deepcopy(routed_cut_graph(gg[3], gg[0], gg[1], gg[2]))
                routed_integrands.extend(
                    loop_processor.get_integrand(deepcopy(cut_graph))
                )
        evaluator_bundle = approach_point(1, "DY", routed_integrands)

    total = Decimal(0)
    for evaluator in evaluator_bundle.evaluators:
        value = evaluator.eval(
            POINT_KS,
            POINT_P1,
            POINT_P2,
            POINT_Z,
            mode="arb",
            decimal_digit_precision=64,
        )
        total += value
    return routed_graph_count, total


@lru_cache(maxsize=1)
def _current_graph_results() -> dict[tuple[str, str], tuple[int, Decimal]]:
    process = _build_process()
    results: dict[tuple[str, str], tuple[int, Decimal]] = {}
    for key, command in GRAPH_COMMANDS.items():
        results[key] = _graph_reference_data(process, command)
    return results


def _assert_decimal_close(actual: Decimal, expected: Decimal) -> None:
    scale = max(Decimal(1), abs(expected))
    tolerance = Decimal("1e-24") * scale
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"Unexpected cut sum: actual={actual}, expected={expected}, tolerance={tolerance}"
        )


@pytest.mark.slow
def test_dy_graph_cut_sums_match_current_references():
    for key, expected in REFERENCE_SUMS.items():
        _actual_count, actual = _current_graph_results()[key]
        _assert_decimal_close(actual, expected)


@pytest.mark.slow
def test_dy_routed_graph_counts_match_current_references():
    for key, expected in REFERENCE_ROUTED_GRAPH_COUNTS.items():
        actual_count, _actual_sum = _current_graph_results()[key]
        assert actual_count == expected
