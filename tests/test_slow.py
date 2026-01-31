from __future__ import annotations

import sys
from pathlib import Path

import pytest
from tests.helpers import assert_complex_close, run, run_inspect_setup  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))  # noqa: E402

# Module-level thresholds for comparisons
REL_TOL = 1e-9
ABS_TOL_ZERO = 1e-12


@pytest.mark.slow
def test_gammaloop_non_optimized_1loop():
    run_inspect_setup(
        setup_names=["gammaloop_non_optimized"],
        sample=[100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        target=complex(-1.750456470996124e-25 - 1.1076985458465256e-26j),
        rel_tol=REL_TOL,
        abs_tol_zero=ABS_TOL_ZERO,
        n_loops=2,
        debug=True,
        verbosity="info",
    )


# @pytest.mark.slow
# def test_gammaloop_optimized_1loop():
#     run_inspect_setup(
#         setup_names=["gammaloop_optimized"],
#         sample=[100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
#         target=complex(-1.750456470996124e-25 - 1.1076985458465256e-26j),
#         rel_tol=REL_TOL,
#         abs_tol_zero=ABS_TOL_ZERO,
#         n_loops=2,
#         debug=True,
#         verbosity="info",
#     )


@pytest.mark.slow
def test_spenso_function_map_non_optimized_1loop():
    run_inspect_setup(
        setup_names=["spenso_function_map_non_optimized"],
        sample=[100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        target=complex(-1.750456470996124e-25 - 1.1076985458465256e-26j),
        rel_tol=REL_TOL,
        abs_tol_zero=ABS_TOL_ZERO,
        n_loops=2,
        debug=True,
        verbosity="info",
    )


@pytest.mark.slow
def test_spenso_function_map_optimized_1loop():
    run_inspect_setup(
        setup_names=["spenso_function_map_optimized"],
        sample=[100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        target=complex(-1.750456470996124e-25 - 1.1076985458465256e-26j),
        rel_tol=REL_TOL,
        abs_tol_zero=ABS_TOL_ZERO,
        n_loops=2,
        debug=True,
        verbosity="info",
    )
