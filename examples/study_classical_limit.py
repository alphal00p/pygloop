#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))  # noqa: E402

from prettytable import PrettyTable  # isort: skip # noqa: E402
from pygloop import main  # isort: skip # noqa: E402
from pygloop.utils.phase_space_generation import (
    generate_two_to_two_at_fixed_t_negative_s,
    generate_two_to_two_at_fixed_t_positive_s,
    verify_ps_point,
)


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


def print_summary(table: dict[str, Any]) -> None:
    print("\nProfiling Summary:")
    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Study classical limit approach")
    # fmt: off
    parser.add_argument("--n-loops", "-l", type=int, default=1, help="Number of loops to profile.")
    parser.add_argument("--m_bh1", type=float, default=1.0,
        help="Mass of the first black hole. Default: %(default)s",
    )  # fmt: off
    parser.add_argument("--m_bh2", type=float, default=1.0,
        help="Mass of the second black hole. Default: %(default)s",
    )  # fmt: off
    parser.add_argument("--sqrt_s", type=float, default=10.0,
        help="Scattering energy. Default: %(default)s",
    )  # fmt: off
    parser.add_argument("--diagrams", "-d", type=str, nargs="*", default=["v_diagram"], help="Specify which diagram to run.")
    parser.add_argument(
        "--results-dump-path", "-r", type=str, default="./classical_limit_results/classical_limit_results.txt", help="Path to dump or recycle profiling results from."
    )
    parser.add_argument("--verbosity", "-v", type=str, choices=["debug", "info", "critical"], default="info",
        help="Set verbosity level",
    )
    parser.add_argument("--rerun", "-rr", default=False, action="store_true", help="Rerun the profiling even if results file exists.")
    parser.add_argument("--debug", "-d", default=False, action="store_true", help="Enable debug mode")
    parser.add_argument("--no-generation", "-ng", default=False, action="store_true", help="Skip process generation.")
    parser.add_argument("--enable-threshold", "-et", default=False, action="store_true", help="Enable threshold subtraction.")
    parser.add_argument("--restart_integration", "-ri", default=False, action="store_true", help="Restart integration.")
    # fmt: on
    args = parser.parse_args()

    if not os.path.isdir(os.path.dirname(args.results_dump_path)):
        os.makedirs(os.path.dirname(args.results_dump_path))

    if os.path.isfile(args.results_dump_path):
        if not args.rerun:
            with open(args.results_dump_path, "r") as f:
                classical_limit_results = eval(f.read())
            print_summary(classical_limit_results)
            sys.exit(0)
        else:
            print(f"Rerunning profiling and overwriting existing results at {args.results_dump_path}.")

    classical_limit_results = {}

    if not args.no_generation:
        classical_limit_results["generation"] = run(
            [
                ("--process", "scalar_gravity"),
                ("--verbosity", args.verbosity),
                ("--m_top", args.m_bh1),
                ("--m_higgs", args.m_bh2),
                ("--overwrite-process-basename", f"CLASSICAL_LIMIT_{args.n_loops}L"),
                (
                    "--gammaloop-settings",
                    (
                        "set global kv global.generation.evaluator.iterative_orientation_optimization=true",
                        f"set global kv global.generation.threshold_subtraction.enable_thresholds={'true' if args.enable_threshold else 'false'}",
                    ),
                ),
                ("--n_loops", str(args.n_loops)),
                ("--clean", None),
            ]+([("--diagrams", tuple(args.diagrams)),] if args.diagrams is not None else []),
            "generate",
            [
                ("-t", "gammaloop"),
            ],
            debug=args.debug,
        )  # fmt: off
    classical_limit_results["integral"] = run(
        [
            ("--process", "scalar_gravity"),
            ("--verbosity", args.verbosity),
            ("--overwrite-process-basename", f"CLASSICAL_LIMIT_{args.n_loops}L"),
            ("--m_top", args.m_bh1),
            ("--m_higgs", args.m_bh2),
            ("--n_loops", str(args.n_loops)),
            ("-ii", "gammaloop"),
        ],
        "integrate",
        []
        + ([("--restart", None)] if args.restart_integration else []),
        debug=args.debug,
    )  # fmt: off

    with open(args.results_dump_path, "w") as f:
        f.write(repr(classical_limit_results))

    print_summary(classical_limit_results)
