#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))  # noqa: E402

from prettytable import PrettyTable  # isort: skip # noqa: E402
from pygloop import main  # isort: skip # noqa: E402


def run(
    cli_args: Sequence[tuple[str, str | tuple[str, ...] | None]],
    cmd: str,
    cmd_args: Sequence[tuple[str, str | tuple[str, ...] | None]],
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


def print_profiling_summary(profiling_result: dict) -> None:
    green = "\033[92m"
    red = "\033[91m"
    reset = "\033[0m"

    def _extract_bench_stats(results: dict) -> dict:
        if "bench_stats" in results:
            return results["bench_stats"]
        return results["bench"]["bench_stats"]

    def _extract_disk_size(results: dict) -> float | None:
        if "disk_size" in results:
            return results["disk_size"]
        return results["bench"]["disk_size"]

    def _format_metric(
        value: float | None,
        baseline: float | None,
        is_baseline: bool,
        decimals: int,
        good_if_lower: bool | None,
    ) -> str:
        if value is None:
            return "n/a"
        value_str = f"{value:.{decimals}f}"
        if is_baseline:
            return value_str
        if baseline is None or baseline == 0:
            return f"{value_str} (x n/a)"
        multiplier = value / baseline
        multiplier_str = f"x {multiplier:.2f}"
        if good_if_lower is None:
            colored_multiplier = multiplier_str
        else:
            is_good = multiplier < 1 if good_if_lower else multiplier > 1
            color = green if is_good else red
            colored_multiplier = f"{color}{multiplier_str}{reset}"
        return f"{value_str} ({colored_multiplier})"

    table = PrettyTable()
    table.field_names = [
        "Integrand type",
        "Generation time (s)",
        "Bench median (ms)",
        "Disk size (MB)",
    ]

    baseline_values: tuple[float | None, float | None, float | None] | None = None
    for idx, (run_description, results) in enumerate(profiling_result.items()):
        generation_time_in_s: float | None
        bench_time_ms: float | None
        disk_size_in_MB: float | None

        if "generation" in results:
            generation_time_in_s = results["generation"]["cmd_runtime"]
        else:
            generation_time_in_s = None

        try:
            bench_stats = _extract_bench_stats(results)
            bench_time_ms = bench_stats["median_s"] * 1e3
        except Exception:
            bench_time_ms = None

        try:
            disk_size = _extract_disk_size(results)
            disk_size_in_MB = None if disk_size is None else disk_size / 1_000_000.0
        except Exception:
            disk_size_in_MB = None

        if baseline_values is None:
            baseline_values = (generation_time_in_s, bench_time_ms, disk_size_in_MB)

        table.add_row(
            [
                run_description,
                _format_metric(generation_time_in_s, baseline_values[0], idx == 0, 3, True),
                _format_metric(bench_time_ms, baseline_values[1], idx == 0, 3, True),
                _format_metric(disk_size_in_MB, baseline_values[2], idx == 0, 2, True),
            ]
        )

    print(table)


RUN_SCENARIOS_IMPLEMENTED = [
    "gammaloop_optimized",
    "gammaloop_non_optimized",
    "spenso_function_map_non_optimized",
    "spenso_function_map_optimized",
    "spenso_function_map_symjit_non_optimized",
    "spenso_function_map_symjit_optimized",
    "spenso_merging_optimized",
    "spenso_summing_optimized",
    "spenso_parametric_optimized",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile pygloop generation and bench runs.")
    # fmt: off
    parser.add_argument("--n-loops", "-l", type=int, default=1, help="Number of loops to profile.")
    parser.add_argument(
        "--results-dump-path", "-r", type=str, default="./profiling_results/profiling_results.txt", help="Path to dump or recycle profiling results from."
    )
    parser.add_argument("--verbosity", "-v", type=str, choices=["debug", "info", "critical"], default="info",
        help="Set verbosity level",
    )
    parser.add_argument("--process", "-p", type=str, choices=["gghhh", "template_process", "dy", "scalar_gravity"], default="gghhh",
        help="Process to consider. Default = %(default)s",
    )  # fmt: off
    parser.add_argument("--rerun", "-rr", default=False, action="store_true", help="Rerun the profiling even if results file exists.")
    parser.add_argument("--debug", "-d", default=False, action="store_true", help="Enable debug mode")
    parser.add_argument("--no-generation", "-ng", default=False, action="store_true", help="Skip process generation.")
    parser.add_argument("--target_time", "-t", type=float, default=1.0,
        help="Target time for the timing profile per repeat. Default = %(default)s",
    )  # fmt: off
    parser.add_argument("--setups", "-s", type=str, nargs="*", choices=RUN_SCENARIOS_IMPLEMENTED, default=None, help="Setups to profile. If not set, all setups are profiled.")
    parser.add_argument("--veto-setups", "-vs", type=str, nargs="*", choices=RUN_SCENARIOS_IMPLEMENTED, default=None, help="Setups to skip profiling for.")
    parser.add_argument("--n-iterations-hornerscheme", "-nhorner", type=int, default=None,
        help="Number of iterations for the Horner scheme optimization. Default = %(default)s",
    )  # fmt: off
    parser.add_argument("--n-iterations-cpe", "-ncpe", type=int, default=None,
        help="Number of iterations for the CPE optimization. Default = until exhaustion",
    )  # fmt: off
    # fmt: on
    args = parser.parse_args()

    if not os.path.isdir(os.path.dirname(args.results_dump_path)):
        os.makedirs(os.path.dirname(args.results_dump_path))

    if os.path.isfile(args.results_dump_path):
        if not args.rerun:
            with open(args.results_dump_path, "r") as f:
                profiling_result = eval(f.read())
            print_profiling_summary(profiling_result)
            sys.exit(0)
        else:
            print(f"Rerunning profiling and overwriting existing results at {args.results_dump_path}.")

    profiling_result = {k: {} for k in RUN_SCENARIOS_IMPLEMENTED}
    if args.setups is not None:
        profiling_result = {key: profiling_result[key] for key in args.setups}
    if args.veto_setups is not None:
        for veto_setup in args.veto_setups:
            if veto_setup in profiling_result:
                del profiling_result[veto_setup]

    for run_description in profiling_result.keys():
        match run_description:
            case "gammaloop_non_optimized":
                if not args.no_generation:
                    profiling_result[run_description]["generation"] = run(
                        [
                            ("--verbosity", args.verbosity),
                            ("--overwrite-process-basename", f"PROFILE_GGHHH_{args.n_loops}L_{run_description}"),
                            (
                                "--gammaloop-settings",
                                (
                                    "set global kv global.generation.evaluator.iterative_orientation_optimization=false",
                                    "set global kv global.generation.threshold_subtraction.enable_thresholds=false",
                                ),
                            ),
                            ("--m_top", "1000.0"),
                            ("--process", args.process),
                            ("--n_loops", str(args.n_loops)),
                            ("--clean", None),
                        ],
                        "generate",
                        [
                            ("-t", "gammaloop"),
                        ]
                        + ([("--n-iterations-cpe", f"{args.n_iterations_cpe}")] if args.n_iterations_cpe is not None else [])
                        + (
                            [("--n-iterations-hornerscheme", f"{args.n_iterations_hornerscheme}")]
                            if args.n_iterations_hornerscheme is not None
                            else []
                        ),
                        debug=args.debug,
                    )
                profiling_result[run_description]["bench"] = run(
                    [
                        ("--verbosity", args.verbosity),
                        ("--overwrite-process-basename", f"PROFILE_GGHHH_{args.n_loops}L_{run_description}"),
                        ("--m_top", "1000.0"),
                        ("--process", args.process),
                        ("--n_loops", str(args.n_loops)),
                        ("-ii", "gammaloop"),
                    ],
                    "bench",
                    [("--target_time", f"{args.target_time:.1f}")],
                    debug=args.debug,
                )
            case "gammaloop_optimized":
                if not args.no_generation:
                    profiling_result[run_description]["generation"] = run(
                        [
                            ("--verbosity", args.verbosity),
                            ("--overwrite-process-basename", f"PROFILE_GGHHH_{args.n_loops}L_{run_description}"),
                            (
                                "--gammaloop-settings",
                                (
                                    "set global kv global.generation.evaluator.iterative_orientation_optimization=true",
                                    "set global kv global.generation.threshold_subtraction.enable_thresholds=false",
                                ),
                            ),
                            ("--m_top", "1000.0"),
                            ("--process", args.process),
                            ("--n_loops", str(args.n_loops)),
                            ("--clean", None),
                        ],
                        "generate",
                        [
                            ("-t", "gammaloop"),
                        ]
                        + ([("--n-iterations-cpe", f"{args.n_iterations_cpe}")] if args.n_iterations_cpe is not None else [])
                        + (
                            [("--n-iterations-hornerscheme", f"{args.n_iterations_hornerscheme}")]
                            if args.n_iterations_hornerscheme is not None
                            else []
                        ),
                        debug=args.debug,
                    )
                profiling_result[run_description]["bench"] = run(
                    [
                        ("--verbosity", args.verbosity),
                        ("--overwrite-process-basename", f"PROFILE_GGHHH_{args.n_loops}L_{run_description}"),
                        ("--m_top", "1000.0"),
                        ("--process", args.process),
                        ("--n_loops", str(args.n_loops)),
                        ("-ii", "gammaloop"),
                    ],
                    "bench",
                    [("--target_time", f"{args.target_time:.1f}")],
                    debug=args.debug,
                )

            case "spenso_function_map_non_optimized":
                if not args.no_generation:
                    profiling_result[run_description]["generation"] = run(
                        [
                            ("--verbosity", args.verbosity),
                            ("--overwrite-process-basename", f"PROFILE_GGHHH_{args.n_loops}L_{run_description}"),
                            ("--general_settings", ("COMPLEXIFY_EVALUATOR=False", "FREEZE_INPUT_PHASES=False")),
                            ("--m_top", "1000.0"),
                            ("--process", args.process),
                            ("--n_loops", str(args.n_loops)),
                            ("--clean", None),
                        ],
                        "generate",
                        [
                            ("-t", "spenso"),
                            ("-g", "function_map"),
                        ]
                        + ([("--n-iterations-cpe", f"{args.n_iterations_cpe}")] if args.n_iterations_cpe is not None else [])
                        + (
                            [("--n-iterations-hornerscheme", f"{args.n_iterations_hornerscheme}")]
                            if args.n_iterations_hornerscheme is not None
                            else []
                        ),
                        debug=args.debug,
                    )
                profiling_result[run_description]["bench"] = run(
                    [
                        ("--verbosity", args.verbosity),
                        ("--overwrite-process-basename", f"PROFILE_GGHHH_{args.n_loops}L_{run_description}"),
                        ("--general_settings", ("COMPLEXIFY_EVALUATOR=False", "FREEZE_INPUT_PHASES=False")),
                        ("--m_top", "1000.0"),
                        ("--process", args.process),
                        ("--n_loops", str(args.n_loops)),
                        ("-ii", "spenso_summed"),
                    ],
                    "bench",
                    [("--target_time", f"{args.target_time:.1f}")],
                    debug=args.debug,
                )
            case "spenso_function_map_optimized":
                if not args.no_generation:
                    profiling_result[run_description]["generation"] = run(
                        [
                            ("--verbosity", args.verbosity),
                            ("--overwrite-process-basename", f"PROFILE_GGHHH_{args.n_loops}L_{run_description}"),
                            ("--general_settings", ("COMPLEXIFY_EVALUATOR=False", "FREEZE_INPUT_PHASES=True")),
                            ("--m_top", "1000.0"),
                            ("--process", args.process),
                            ("--n_loops", str(args.n_loops)),
                            ("--clean", None),
                        ],
                        "generate",
                        [
                            ("-t", "spenso"),
                            ("-g", "function_map"),
                        ]
                        + ([("--n-iterations-cpe", f"{args.n_iterations_cpe}")] if args.n_iterations_cpe is not None else [])
                        + (
                            [("--n-iterations-hornerscheme", f"{args.n_iterations_hornerscheme}")]
                            if args.n_iterations_hornerscheme is not None
                            else []
                        ),
                        debug=args.debug,
                    )
                profiling_result[run_description]["bench"] = run(
                    [
                        ("--verbosity", args.verbosity),
                        ("--overwrite-process-basename", f"PROFILE_GGHHH_{args.n_loops}L_{run_description}"),
                        ("--general_settings", ("COMPLEXIFY_EVALUATOR=False", "FREEZE_INPUT_PHASES=True")),
                        ("--m_top", "1000.0"),
                        ("--process", args.process),
                        ("--n_loops", str(args.n_loops)),
                        ("-ii", "spenso_summed"),
                    ],
                    "bench",
                    [("--target_time", f"{args.target_time:.1f}")],
                    debug=args.debug,
                )

            case "spenso_merging_optimized":
                if not args.no_generation:
                    profiling_result[run_description]["generation"] = run(
                        [
                            ("--verbosity", args.verbosity),
                            ("--overwrite-process-basename", f"PROFILE_GGHHH_{args.n_loops}L_{run_description}"),
                            ("--general_settings", ("COMPLEXIFY_EVALUATOR=False", "FREEZE_INPUT_PHASES=True")),
                            ("--m_top", "1000.0"),
                            ("--process", args.process),
                            ("--n_loops", str(args.n_loops)),
                            ("--clean", None),
                        ],
                        "generate",
                        [
                            ("-t", "spenso"),
                            ("-g", "merging"),
                        ]
                        + ([("--n-iterations-cpe", f"{args.n_iterations_cpe}")] if args.n_iterations_cpe is not None else [])
                        + (
                            [("--n-iterations-hornerscheme", f"{args.n_iterations_hornerscheme}")]
                            if args.n_iterations_hornerscheme is not None
                            else []
                        ),
                        debug=args.debug,
                    )
                profiling_result[run_description]["bench"] = run(
                    [
                        ("--verbosity", args.verbosity),
                        ("--overwrite-process-basename", f"PROFILE_GGHHH_{args.n_loops}L_{run_description}"),
                        ("--general_settings", ("COMPLEXIFY_EVALUATOR=False", "FREEZE_INPUT_PHASES=True")),
                        ("--m_top", "1000.0"),
                        ("--process", args.process),
                        ("--n_loops", str(args.n_loops)),
                        ("-ii", "spenso_summed"),
                    ],
                    "bench",
                    [("--target_time", f"{args.target_time:.1f}")],
                    debug=args.debug,
                )

            case "spenso_summing_optimized":
                if not args.no_generation:
                    profiling_result[run_description]["generation"] = run(
                        [
                            ("--verbosity", args.verbosity),
                            ("--overwrite-process-basename", f"PROFILE_GGHHH_{args.n_loops}L_{run_description}"),
                            ("--general_settings", ("COMPLEXIFY_EVALUATOR=False", "FREEZE_INPUT_PHASES=True")),
                            ("--m_top", "1000.0"),
                            ("--process", args.process),
                            ("--n_loops", str(args.n_loops)),
                            ("--clean", None),
                        ],
                        "generate",
                        [
                            ("-t", "spenso"),
                            ("-g", "summing"),
                        ]
                        + ([("--n-iterations-cpe", f"{args.n_iterations_cpe}")] if args.n_iterations_cpe is not None else [])
                        + (
                            [("--n-iterations-hornerscheme", f"{args.n_iterations_hornerscheme}")]
                            if args.n_iterations_hornerscheme is not None
                            else []
                        ),
                        debug=args.debug,
                    )
                profiling_result[run_description]["bench"] = run(
                    [
                        ("--verbosity", args.verbosity),
                        ("--overwrite-process-basename", f"PROFILE_GGHHH_{args.n_loops}L_{run_description}"),
                        ("--general_settings", ("COMPLEXIFY_EVALUATOR=False", "FREEZE_INPUT_PHASES=True")),
                        ("--m_top", "1000.0"),
                        ("--process", args.process),
                        ("--n_loops", str(args.n_loops)),
                        ("-ii", "spenso_summed"),
                    ],
                    "bench",
                    [("--target_time", f"{args.target_time:.1f}")],
                    debug=args.debug,
                )

            case "spenso_parametric_optimized":
                if not args.no_generation:
                    profiling_result[run_description]["generation"] = run(
                        [
                            ("--verbosity", args.verbosity),
                            ("--overwrite-process-basename", f"PROFILE_GGHHH_{args.n_loops}L_{run_description}"),
                            ("--general_settings", ("COMPLEXIFY_EVALUATOR=False", "FREEZE_INPUT_PHASES=True")),
                            ("--m_top", "1000.0"),
                            ("--process", args.process),
                            ("--n_loops", str(args.n_loops)),
                            ("--clean", None),
                        ],
                        "generate",
                        [
                            ("-t", "spenso"),
                        ]
                        + ([("--n-iterations-cpe", f"{args.n_iterations_cpe}")] if args.n_iterations_cpe is not None else [])
                        + (
                            [("--n-iterations-hornerscheme", f"{args.n_iterations_hornerscheme}")]
                            if args.n_iterations_hornerscheme is not None
                            else []
                        ),
                        debug=args.debug,
                    )
                profiling_result[run_description]["bench"] = run(
                    [
                        ("--verbosity", args.verbosity),
                        ("--overwrite-process-basename", f"PROFILE_GGHHH_{args.n_loops}L_{run_description}"),
                        ("--general_settings", ("COMPLEXIFY_EVALUATOR=False", "FREEZE_INPUT_PHASES=True")),
                        ("--m_top", "1000.0"),
                        ("--process", args.process),
                        ("--n_loops", str(args.n_loops)),
                        ("-ii", "spenso_parametric"),
                    ],
                    "bench",
                    [("--target_time", f"{args.target_time:.1f}")],
                    debug=args.debug,
                )

            case "spenso_function_map_symjit_non_optimized":
                if not args.no_generation:
                    profiling_result[run_description]["generation"] = run(
                        [
                            ("--verbosity", args.verbosity),
                            ("--integrand-evaluator-compiler", "symjit"),
                            ("--overwrite-process-basename", f"PROFILE_GGHHH_{args.n_loops}L_{run_description}"),
                            ("--general_settings", ("COMPLEXIFY_EVALUATOR=False", "FREEZE_INPUT_PHASES=False")),
                            ("--m_top", "1000.0"),
                            ("--process", args.process),
                            ("--n_loops", str(args.n_loops)),
                            ("--clean", None),
                        ],
                        "generate",
                        [
                            ("-t", "spenso"),
                            ("-g", "function_map"),
                        ]
                        + ([("--n-iterations-cpe", f"{args.n_iterations_cpe}")] if args.n_iterations_cpe is not None else [])
                        + (
                            [("--n-iterations-hornerscheme", f"{args.n_iterations_hornerscheme}")]
                            if args.n_iterations_hornerscheme is not None
                            else []
                        ),
                        debug=args.debug,
                    )
                profiling_result[run_description]["bench"] = run(
                    [
                        ("--verbosity", args.verbosity),
                        ("--integrand-evaluator-compiler", "symjit"),
                        ("--overwrite-process-basename", f"PROFILE_GGHHH_{args.n_loops}L_{run_description}"),
                        ("--general_settings", ("COMPLEXIFY_EVALUATOR=False", "FREEZE_INPUT_PHASES=False")),
                        ("--m_top", "1000.0"),
                        ("--process", args.process),
                        ("--n_loops", str(args.n_loops)),
                        ("-ii", "spenso_summed"),
                    ],
                    "bench",
                    [("--target_time", f"{args.target_time:.1f}")],
                    debug=args.debug,
                )
            case "spenso_function_map_symjit_optimized":
                if not args.no_generation:
                    profiling_result[run_description]["generation"] = run(
                        [
                            ("--verbosity", args.verbosity),
                            ("--integrand-evaluator-compiler", "symjit"),
                            ("--overwrite-process-basename", f"PROFILE_GGHHH_{args.n_loops}L_{run_description}"),
                            ("--general_settings", ("COMPLEXIFY_EVALUATOR=False", "FREEZE_INPUT_PHASES=True")),
                            ("--m_top", "1000.0"),
                            ("--process", args.process),
                            ("--n_loops", str(args.n_loops)),
                            ("--clean", None),
                        ],
                        "generate",
                        [
                            ("-t", "spenso"),
                            ("-g", "function_map"),
                        ]
                        + ([("--n-iterations-cpe", f"{args.n_iterations_cpe}")] if args.n_iterations_cpe is not None else [])
                        + (
                            [
                                ("--n-iterations-hornerscheme", f"{args.n_iterations_hornerscheme}"),
                            ]
                            if args.n_iterations_hornerscheme is not None
                            else []
                        ),
                        debug=args.debug,
                    )
                profiling_result[run_description]["bench"] = run(
                    [
                        ("--verbosity", args.verbosity),
                        ("--integrand-evaluator-compiler", "symjit"),
                        ("--overwrite-process-basename", f"PROFILE_GGHHH_{args.n_loops}L_{run_description}"),
                        ("--general_settings", ("COMPLEXIFY_EVALUATOR=False", "FREEZE_INPUT_PHASES=True")),
                        ("--m_top", "1000.0"),
                        ("--process", args.process),
                        ("--n_loops", str(args.n_loops)),
                        ("-ii", "spenso_summed"),
                    ],
                    "bench",
                    [("--target_time", f"{args.target_time:.1f}")],
                    debug=args.debug,
                )

            case _:
                raise ValueError(f"Unknown run description: {run_description}")

    with open(args.results_dump_path, "w") as f:
        f.write(repr(profiling_result))

    print_profiling_summary(profiling_result)
