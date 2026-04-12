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
from processes.dy.dy_classes import (  # noqa: E402
    DYDotGraphs,
    VacuumDotGraph,
    canonicalise_vacuum_graph,
)
from processes.dy.dy_evaluators import DYCompiledBundle, evaluate_integrand  # noqa: E402
from processes.dy.dy_integrand import (  # noqa: E402
    LoopIntegrandConstructor,
    routed_cut_graph,
)
from utils.vectors import LorentzVector, Vector  # noqa: E402


class DYGraphCutSumReferenceProcess(DY):
    name = "DY_GRAPH_CUT_SUM_REFERENCE_TEST"


class TTGraphChannelReferenceProcess(DY):
    name = "TT_GRAPH_CHANNEL_REFERENCE_TEST"


GENERATE_CONFIG = ROOT / "configs" / "DY" / "generate.toml"
RUNTIME_CONFIG = ROOT / "configs" / "DY" / "runtime.toml"
POINT_Z = 0.6
POINTS = {
    "symmetric": {
        "ks": [
            math.sqrt(POINT_Z)
            * np.array([0.1 / math.sqrt(3), 0.1 / math.sqrt(3), 0.1 / math.sqrt(3)])
        ],
        "p1": [0.0, 0.0, 1.0],
        "p2": [0.0, 0.0, -1.0],
        "z": POINT_Z,
    },
    "anisotropic": {
        "ks": [
            math.sqrt(POINT_Z)
            * np.array([0.1 / math.sqrt(3), 0.1 / math.sqrt(3), 1.0 / math.sqrt(3)])
        ],
        "p1": [0.0, 0.0, 1.0],
        "p2": [0.0, 0.0, -1.0],
        "z": POINT_Z,
    },
}
OBSERVABLE_PARAMS = {
    "zmin": 0.0,
    "zmax": 1.0,
    "Lambdasq": 2.0,
    "mUV": 1.0,
}
CHANNEL_COMMANDS = {
    "ddx": {
        "base_name": "DY_GRAPH_CUT_SUM_REFERENCE_TEST_DDX",
        "graphs_name": "DY_GRAPH_CUT_SUM_REFERENCE_TEST_DDX_generated_graphs",
        "particle_filter": ["a"],
        "command": (
            "generate xs d d~ > a | d d~ g a QED^2==2 [{{1}} QCD=1] "
            "--only-diagrams --numerator-grouping only_detect_zeroes "
            "-p DY_GRAPH_CUT_SUM_REFERENCE_TEST_DDX "
            "-i DY_GRAPH_CUT_SUM_REFERENCE_TEST_DDX_generated_graphs "
            "--max-multiplicity-for-fast-cut-filter 99"
        ),
    },
    "dg": {
        "base_name": "DY_GRAPH_CUT_SUM_REFERENCE_TEST_DG",
        "graphs_name": "DY_GRAPH_CUT_SUM_REFERENCE_TEST_DG_generated_graphs",
        "particle_filter": ["a"],
        "command": (
            "generate xs d g > a | d d~ g a QED^2==2 [{{1}} QCD=1] "
            "--only-diagrams --numerator-grouping only_detect_zeroes "
            "-p DY_GRAPH_CUT_SUM_REFERENCE_TEST_DG "
            "-i DY_GRAPH_CUT_SUM_REFERENCE_TEST_DG_generated_graphs "
            "--max-multiplicity-for-fast-cut-filter 99"
        ),
    },
}
GGTT_COMMAND = {
    "base_name": "TT_GRAPH_CHANNEL_REFERENCE_TEST",
    "graphs_name": "TT_GRAPH_CHANNEL_REFERENCE_TEST_generated_graphs",
    "command": (
        "generate xs g g > t t~ | d d~ g t t~ [{{1}} QCD=1] "
        "--only-diagrams --numerator-grouping "
        "group_identical_graphs_up_to_scalar_rescaling "
        "--symmetrize-left-right-states true "
        "--symmetrize-initial-states true "
        "-p TT_GRAPH_CHANNEL_REFERENCE_TEST "
        "-i TT_GRAPH_CHANNEL_REFERENCE_TEST_generated_graphs "
        "--max-multiplicity-for-fast-cut-filter 99"
    ),
}
REFERENCE_SUMS: dict[tuple[str, str], dict[str, Decimal]] = {
    ("ddx", "GL1"): {
        "anisotropic": Decimal("11.06485143967556911630848559"),
        "symmetric": Decimal("-1185.267710416734201491426698"),
    },
    ("ddx", "GL2"): {
        "anisotropic": Decimal("11.06485143967556911630848571"),
        "symmetric": Decimal("-1185.267710416734201491426698"),
    },
    ("ddx", "GL3"): {
        "anisotropic": Decimal("-3.426357200058932674774999939"),
        "symmetric": Decimal("6.602166557884883592640098374"),
    },
    ("ddx", "GL4"): {
        "anisotropic": Decimal("-36.82655303339209532432538534"),
        "symmetric": Decimal("-63.37843232994971427725117457"),
    },
    ("ddx", "GL6"): {
        "anisotropic": Decimal("-3.426357200058932674774999910"),
        "symmetric": Decimal("6.602166557884883592640098369"),
    },
    ("ddx", "GL7"): {
        "anisotropic": Decimal("-36.82655303339209532432538534"),
        "symmetric": Decimal("-63.37843232994971427725117457"),
    },
    ("dg", "GL0"): {
        "anisotropic": Decimal("-6.179030686855646589789820527"),
        "symmetric": Decimal("-30.53939886477111631220712552"),
    },
    ("dg", "GL1"): {
        "anisotropic": Decimal("-2.790888316442310638561596182"),
        "symmetric": Decimal("3.421606828727281949577332126"),
    },
    ("dg", "GL2"): {
        "anisotropic": Decimal("-2.790888316442310638561596230"),
        "symmetric": Decimal("3.421606828727281949577332123"),
    },
    ("dg", "GL3"): {
        "anisotropic": Decimal("0.5929447457977601929020119492"),
        "symmetric": Decimal("1.712180072456517031351684803"),
    },
}
GGTT_POINTS = {
    "probe_a": {
        "ks": [Vector(150.0, -40.0, 220.0)],
        "p1": Vector(0.0, 0.0, 500.0),
        "p2": Vector(0.0, 0.0, -500.0),
        "z": POINT_Z,
    },
    "probe_b": {
        "ks": [Vector(250.0, 60.0, -90.0)],
        "p1": Vector(0.0, 0.0, 500.0),
        "p2": Vector(0.0, 0.0, -500.0),
        "z": POINT_Z,
    },
}
GGTT_REFERENCE_SUMS: dict[str, dict[str, Decimal]] = {
    "graph_0": {
        "probe_a": Decimal("4.535837345765991e-12"),
        "probe_b": Decimal("1.5242388435430973e-12"),
    },
    "graph_1": {
        "probe_a": Decimal("-3.6164341239834297e-14"),
        "probe_b": Decimal("-1.6754230169188702e-14"),
    },
    "graph_2": {
        "probe_a": Decimal("1.6432398306166366e-12"),
        "probe_b": Decimal("1.4646399443508783e-12"),
    },
    "graph_3": {
        "probe_a": Decimal("-2.3712594247880328e-12"),
        "probe_b": Decimal("-2.010861742122394e-12"),
    },
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


def _graph_name(graph) -> str:
    return str(graph.dot.get_name()).strip('"')


def _build_process() -> DYGraphCutSumReferenceProcess:
    with contextlib.redirect_stdout(io.StringIO()):
        return DYGraphCutSumReferenceProcess(
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


def _generate_channel_graphs(
    process: DYGraphCutSumReferenceProcess, channel: str
) -> DYDotGraphs:
    spec = CHANNEL_COMMANDS[channel]
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        process.gl_worker.run(spec["command"])
        process.gl_worker.run("save state -o")

    amplitudes, cross_sections = process.gl_worker.list_outputs()
    if (
        spec["graphs_name"] not in amplitudes
        and spec["graphs_name"] not in cross_sections
    ):
        raise AssertionError(f"Generated output '{spec['graphs_name']}' was not found.")
    process_id = (
        amplitudes[spec["graphs_name"]]
        if spec["graphs_name"] in amplitudes
        else cross_sections[spec["graphs_name"]]
    )

    dot_str = process.gl_worker.get_dot_files(
        process_id=process_id,
        integrand_name=spec["graphs_name"],
    )
    return DYDotGraphs(dot_str=dot_str)


def _filtered_graphs(graphs: DYDotGraphs, channel: str) -> DYDotGraphs:
    particle_filter = CHANNEL_COMMANDS[channel]["particle_filter"]
    filtered = DYDotGraphs()
    filtered.extend(deepcopy(graphs.filter_particle_definition(particle_filter)))
    return filtered


def _routed_integrands_for_graph(graph, loop_processor) -> list:
    vac_g = canonicalise_vacuum_graph(deepcopy(graph))
    vacuum_g = VacuumDotGraph(deepcopy(vac_g.dot))
    routed_graphs = vacuum_g.cut_graphs_with_routing_leading_virtuality([], ["a"])

    routed_integrands = []
    for gg in routed_graphs:
        cut_graph = deepcopy(routed_cut_graph(gg[3], gg[0], gg[1], gg[2]))
        routed_integrands.extend(loop_processor.get_integrand(deepcopy(cut_graph)))
    return routed_integrands


def _evaluate_cut_sum(routed_integrands: list, point: dict[str, object]) -> Decimal:
    evaluators = [
        evaluate_integrand(
            1,
            "DY",
            deepcopy(routed_integrand),
            n_hornerscheme_iterations=1000,
            n_cpe_iterations=10000,
            observable_params=OBSERVABLE_PARAMS,
        )
        for routed_integrand in routed_integrands
    ]

    total = Decimal(0)
    for evaluator in evaluators:
        total += evaluator.eval(
            point["ks"],
            point["p1"],
            point["p2"],
            point["z"],
            mode="arb",
            decimal_digit_precision=64,
        )
    return total


@lru_cache(maxsize=1)
def _current_graph_results() -> dict[tuple[str, str], dict[str, Decimal]]:
    results: dict[tuple[str, str], dict[str, Decimal]] = {}
    process = _build_process()

    for channel in CHANNEL_COMMANDS:
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            graphs = _generate_channel_graphs(process, channel)
            filtered = _filtered_graphs(graphs, channel)
            loop_processor = LoopIntegrandConstructor([], process.process_name, 1)

            for graph in filtered:
                routed_integrands = _routed_integrands_for_graph(graph, loop_processor)
                graph_key = (channel, _graph_name(graph))
                results[graph_key] = {
                    point_name: _evaluate_cut_sum(routed_integrands, point)
                    for point_name, point in POINTS.items()
                }

    return results


def _assert_decimal_close(actual: Decimal, expected: Decimal) -> None:
    scale = max(Decimal(1), abs(expected))
    tolerance = Decimal("1e-12") * scale
    if abs(abs(actual) - abs(expected)) > tolerance:
        raise AssertionError(
            f"Unexpected cut sum: actual={actual}, expected={expected}, tolerance={tolerance}"
        )


def _build_ggtt_process() -> TTGraphChannelReferenceProcess:
    with contextlib.redirect_stdout(io.StringIO()):
        return TTGraphChannelReferenceProcess(
            m_top=173.0,
            m_higgs=125.0,
            ps_point=_default_ps_point(),
            n_loops=1,
            clean=True,
            logger_level=logging.CRITICAL,
            skip_ps_validation=True,
            toml_config_path=str(GENERATE_CONFIG),
            runtime_toml_config_path=str(RUNTIME_CONFIG),
            final_state=["t", "t"],
            process_name="tt~",
        )


def _generate_ggtt_graphs(process: TTGraphChannelReferenceProcess) -> DYDotGraphs:
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        process.gl_worker.run(GGTT_COMMAND["command"])
        process.gl_worker.run("save state -o")

    amplitudes, cross_sections = process.gl_worker.list_outputs()
    if (
        GGTT_COMMAND["graphs_name"] not in amplitudes
        and GGTT_COMMAND["graphs_name"] not in cross_sections
    ):
        raise AssertionError(
            f"Generated output '{GGTT_COMMAND['graphs_name']}' was not found."
        )
    process_id = (
        amplitudes[GGTT_COMMAND["graphs_name"]]
        if GGTT_COMMAND["graphs_name"] in amplitudes
        else cross_sections[GGTT_COMMAND["graphs_name"]]
    )

    dot_str = process.gl_worker.get_dot_files(
        process_id=process_id,
        integrand_name=GGTT_COMMAND["graphs_name"],
    )
    return DYDotGraphs(dot_str=dot_str)


@lru_cache(maxsize=1)
def _current_ggtt_graph_results() -> dict[str, dict[str, Decimal]]:
    process = _build_ggtt_process()
    with contextlib.redirect_stdout(io.StringIO()):
        process.process_1L_generated_graphs(_generate_ggtt_graphs(process))

    bundle = DYCompiledBundle.load("tt~", process.get_integrand_name())
    results: dict[str, dict[str, Decimal]] = {}
    for channel_index, channel_name in enumerate(bundle.graph_channel_names()):
        point_values: dict[str, Decimal] = {}
        for point_name, point in GGTT_POINTS.items():
            value = bundle.evaluate(
                point["ks"],
                point["p1"],
                point["p2"],
                point["z"],
                mode="compiled",
                channel_selector=channel_index,
            )
            if not math.isclose(value.imag, 0.0, abs_tol=1.0e-30):
                raise AssertionError(
                    f"Expected a real ggtt graph value for '{channel_name}', got {value}."
                )
            point_values[point_name] = Decimal(repr(value.real))
        results[channel_name] = point_values

    return results


def _assert_small_decimal_close(actual: Decimal, expected: Decimal) -> None:
    tolerance = max(Decimal("1e-18"), Decimal("1e-9") * abs(expected))
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"Unexpected ggtt graph value: actual={actual}, expected={expected}, tolerance={tolerance}"
        )


@pytest.mark.slow
def test_dy_graph_cut_sums_match_references():
    current = _current_graph_results()
    assert set(current) == set(REFERENCE_SUMS)

    for graph_key, expected_points in REFERENCE_SUMS.items():
        actual_points = current[graph_key]
        assert set(actual_points) == set(expected_points)
        for point_name, expected_value in expected_points.items():
            _assert_decimal_close(actual_points[point_name], expected_value)


@pytest.mark.slow
def test_ggtt_graph_channels_match_references():
    current = _current_ggtt_graph_results()
    assert set(current) == set(GGTT_REFERENCE_SUMS)

    for graph_name, expected_points in GGTT_REFERENCE_SUMS.items():
        actual_points = current[graph_name]
        assert set(actual_points) == set(expected_points)
        for point_name, expected_value in expected_points.items():
            _assert_small_decimal_close(actual_points[point_name], expected_value)
