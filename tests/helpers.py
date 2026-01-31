from __future__ import annotations

import math
from typing import Any

from pygloop import main


def run(
    cli_args: list[tuple[str, str | tuple[str, ...] | None]],
    cmd: str,
    cmd_args: list[tuple[str, str | tuple[str, ...] | None]],
    debug: bool = False,
) -> dict:
    argv = [[a] if v is None else [a, v] for a, v in cli_args] + [[cmd]] + [[a] if v is None else [a, v] for a, v in cmd_args]
    argv = []
    for a, v in cli_args:
        argv.append(a)
        if isinstance(v, tuple):
            argv.extend(v)
        elif v is not None:
            argv.append(v)
    argv.append(cmd)
    for a, v in cmd_args:
        argv.append(a)
        if isinstance(v, tuple):
            argv.extend(v)
        elif v is not None:
            argv.append(v)
    if debug:
        print(f"Running the following command:\n./bin/pygloop {' '.join(argv)}")
    result = main(argv)
    assert isinstance(result, dict)
    return result


def _component_close(actual: float, expected: float, rel_tol: float, abs_tol_zero: float) -> bool:
    # Use an absolute tolerance when the target is zero, otherwise a relative tolerance scaled to the target magnitude.
    abs_tol = abs_tol_zero if expected == 0 else rel_tol * abs(expected)
    return (
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=abs_tol)
        if expected == 0
        else math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol)
    )


def complex_close(actual: complex | Any, expected: complex | Any, rel_tol: float, abs_tol_zero: float) -> bool:
    """Compare complex numbers component-wise with mixed rel/abs tolerances."""
    if not isinstance(actual, complex) or not isinstance(expected, complex):
        return False
    return _component_close(actual.real, expected.real, rel_tol, abs_tol_zero) and _component_close(actual.imag, expected.imag, rel_tol, abs_tol_zero)


def assert_complex_close(actual: complex, expected: complex, rel_tol: float, abs_tol_zero: float) -> None:
    if not complex_close(actual, expected, rel_tol, abs_tol_zero):
        raise AssertionError(f"Complex numbers differ: actual={actual!r}, expected={expected!r}, rel_tol={rel_tol}, abs_tol_zero={abs_tol_zero}")


def run_inspect_setup(
    setup_names: list[str],
    sample: list[float],
    target: complex,
    rel_tol: float = 1e-9,
    abs_tol_zero: float = 1e-12,
    n_loops: int = 1,
    debug=True,
    verbosity: str = "info",
):
    test_result = {
        "gammaloop_non_optimized": {},
        "gammaloop_optimized": {},
        "spenso_function_map_non_optimized": {},
        "spenso_function_map_optimized": {},
    }
    test_result = {name: {} for name in setup_names}

    for run_description in test_result.keys():
        match run_description:
            case "gammaloop_non_optimized":
                test_result[run_description]["generation"] = run(
                    [
                        ("--verbosity", verbosity),
                        ("--overwrite-process-basename", f"TEST_GGHHH_{n_loops}L_{run_description}"),
                        (
                            "--gammaloop-settings",
                            (
                                "set global kv global.generation.evaluator.iterative_orientation_optimization=false",
                                "set global kv global.generation.threshold_subtraction.enable_thresholds=false",
                            ),
                        ),
                        ("--m_top", "1000.0"),
                        ("--process", "gghhh"),
                        ("--n_loops", str(n_loops)),
                        ("--clean", None),
                    ],
                    "generate",
                    [
                        ("-t", "gammaloop"),
                    ],
                    debug=debug,
                )
                test_result[run_description]["inspect"] = run(
                    [
                        ("--verbosity", verbosity),
                        ("--overwrite-process-basename", f"TEST_GGHHH_{n_loops}L_{run_description}"),
                        ("--m_top", "1000.0"),
                        ("--process", "gghhh"),
                        ("--n_loops", str(n_loops)),
                        ("-ii", "gammaloop"),
                    ],
                    "inspect",
                    [
                        ("-p", tuple([f"{k_i:.16e}" for k_i in sample])),
                    ],
                    debug=debug,
                )
                assert_complex_close(test_result[run_description]["inspect"]["inspect_result"], target, rel_tol, abs_tol_zero)
            case "gammaloop_optimized":
                test_result[run_description]["generation"] = run(
                    [
                        ("--verbosity", verbosity),
                        ("--overwrite-process-basename", f"TEST_GGHHH_{n_loops}L_{run_description}"),
                        (
                            "--gammaloop-settings",
                            (
                                "set global kv global.generation.evaluator.iterative_orientation_optimization=true",
                                "set global kv global.generation.threshold_subtraction.enable_thresholds=false",
                            ),
                        ),
                        ("--m_top", "1000.0"),
                        ("--process", "gghhh"),
                        ("--n_loops", str(n_loops)),
                        ("--clean", None),
                    ],
                    "generate",
                    [
                        ("-t", "gammaloop"),
                    ],
                    debug=debug,
                )
                test_result[run_description]["inspect"] = run(
                    [
                        ("--verbosity", verbosity),
                        ("--overwrite-process-basename", f"TEST_GGHHH_{n_loops}L_{run_description}"),
                        ("--m_top", "1000.0"),
                        ("--process", "gghhh"),
                        ("--n_loops", str(n_loops)),
                        ("-ii", "gammaloop"),
                    ],
                    "inspect",
                    [
                        ("-p", tuple([f"{k_i:.16e}" for k_i in sample])),
                    ],
                    debug=debug,
                )
                assert_complex_close(test_result[run_description]["inspect"]["inspect_result"], target, rel_tol, abs_tol_zero)

            case "spenso_function_map_non_optimized":
                test_result[run_description]["generation"] = run(
                    [
                        ("--verbosity", verbosity),
                        ("--overwrite-process-basename", f"TEST_GGHHH_{n_loops}L_{run_description}"),
                        ("--general_settings", ("COMPLEXIFY_EVALUATOR=False", "FREEZE_INPUT_PHASES=False")),
                        ("--m_top", "1000.0"),
                        ("--process", "gghhh"),
                        ("--n_loops", str(n_loops)),
                        ("--clean", None),
                    ],
                    "generate",
                    [
                        ("-t", "spenso"),
                        ("-g", "function_map"),
                    ],
                    debug=debug,
                )
                test_result[run_description]["inspect"] = run(
                    [
                        ("--verbosity", verbosity),
                        ("--overwrite-process-basename", f"TEST_GGHHH_{n_loops}L_{run_description}"),
                        ("--general_settings", ("COMPLEXIFY_EVALUATOR=False", "FREEZE_INPUT_PHASES=False")),
                        ("--m_top", "1000.0"),
                        ("--process", "gghhh"),
                        ("--n_loops", str(n_loops)),
                        ("-ii", "spenso_summed"),
                    ],
                    "inspect",
                    [
                        ("-p", tuple([f"{k_i:.16e}" for k_i in sample])),
                    ],
                    debug=debug,
                )
                assert_complex_close(test_result[run_description]["inspect"]["inspect_result"], target, rel_tol, abs_tol_zero)

            case "spenso_function_map_optimized":
                test_result[run_description]["generation"] = run(
                    [
                        ("--verbosity", verbosity),
                        ("--overwrite-process-basename", f"TEST_GGHHH_{n_loops}L_{run_description}"),
                        ("--general_settings", ("COMPLEXIFY_EVALUATOR=False", "FREEZE_INPUT_PHASES=True")),
                        ("--m_top", "1000.0"),
                        ("--process", "gghhh"),
                        ("--n_loops", str(n_loops)),
                        ("--clean", None),
                    ],
                    "generate",
                    [
                        ("-t", "spenso"),
                        ("-g", "function_map"),
                    ],
                    debug=debug,
                )
                test_result[run_description]["inspect"] = run(
                    [
                        ("--verbosity", verbosity),
                        ("--overwrite-process-basename", f"TEST_GGHHH_{n_loops}L_{run_description}"),
                        ("--general_settings", ("COMPLEXIFY_EVALUATOR=False", "FREEZE_INPUT_PHASES=True")),
                        ("--m_top", "1000.0"),
                        ("--process", "gghhh"),
                        ("--n_loops", str(n_loops)),
                        ("-ii", "spenso_summed"),
                    ],
                    "inspect",
                    [
                        ("-p", tuple([f"{k_i:.16e}" for k_i in sample])),
                    ],
                    debug=debug,
                )
                assert_complex_close(test_result[run_description]["inspect"]["inspect_result"], target, rel_tol, abs_tol_zero)

            case _:
                raise ValueError(f"Unknown run description: {run_description}")
