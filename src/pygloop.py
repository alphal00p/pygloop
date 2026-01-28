#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import multiprocessing
import random
import re
import sys
import time
from pprint import pformat

from processes.dy.dy import DY
from processes.gghhh.gghhh import GGHHH
from processes.scalar_gravity.scalar_gravity import ScalarGravity
from processes.template_process import TemplateProcess
from utils.utils import (
    SRC_DIR,
    Colour,
    logger,
    pygloopException,
    setup_logging,
    time_function,
)
from utils.vectors import LorentzVector, Vector

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def main(argv: list[str] | None = None) -> int:
    # create the top-level parser
    class FloatArgParser(argparse.ArgumentParser):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._negative_number_matcher = re.compile(r"^-?\d+(\.\d*)?([eE][-+]?\d+)?$")  # type: ignore

    parser = FloatArgParser(prog="pygloop")

    parser.add_argument("--process", "-p", type=str, choices=["gghhh", "template_process", "dy", "scalar_gravity"], default="gghhh",
        help="Process to consider. Default = %(default)s",
    )  # fmt: off
    parser.add_argument("--diagrams", "-d", type=str, nargs="*", default=None,
        help="Diagrams to consider. Default = %(default)s",
    )  # fmt: off

    parser.add_argument("--overwrite-process-basename", "-o", type=str, default=None,
        help="Overwrite the default process basename used for generated files. Default = <DEFAULT_PROCESS_NAME_SET_BY_PROCESS_CLASS>",
    )  # fmt: off

    # Add options common to all subcommands
    _ = parser.add_argument("--verbosity", "-v", type=str, choices=["debug", "info", "critical"], default="info",
        help="Set verbosity level",
    )  # fmt: off

    parser.add_argument("--parameterisation", "-param", type=str, choices=["cartesian", "spherical"], default="spherical",
        help="Parameterisation to employ.",
    )  # fmt: off

    parser.add_argument("--gammaloop-configuration", "-f", default=None,
        help="Specify a toml file containing the gammaloop configuration desired. Default = ./configs/<PROCESS_NAME>/generate.toml",
    )  # fmt: off
    parser.add_argument("--runtime-configuration", "-r", metavar="toml_config_path", default=None,
        help="Specify a toml file containing the integration configuration (only for gammaloop integrator). Default = ./configs/<PROCESS_NAME>/integrate.toml",
    )  # fmt: off
    parser.add_argument("--gammaloop-settings", "-s", metavar="gammaloop_settings", type=str, nargs="*", default=None,
        help='specify gammaloop settings to override toml. Format list of space-separated instructions. -s "set global kv global.n_cores.feyngen=12" "set global kv global.generation.evaluator.iterative_orientation_optimization=false"',
    )  # fmt: off

    parser.add_argument("--m_top", type=float, default=None,
        help="Mass of the internal top quark. Default for gghhh = 173 GeV",
    )  # fmt: off
    _ = parser.add_argument("--m_higgs", type=float, default=None,
        help="Higgs mass. Default for gghhh = 125 GeV",
    )  # fmt: off
    parser.add_argument("--pg1", "-pg1", type=float, nargs=4, default=None,
        help="Four-momentum of the first gluon. Default for gghhh = [500.0, 0.0, 0.0, 500.0] GeV",
    )  # fmt: off
    parser.add_argument("--pg2", "-pg2", type=float, nargs=4, default=None,
        help="Four-momentum of the second gluon. Default for gghhh = [500.0, 0.0, 0.0, -500.0] GeV",
    )  # fmt: off
    parser.add_argument("--ph1", "-ph1", type=float, nargs=4, default=None,
        help="Four-momentum of the first Higgs. Default for gghhh = [0.4385555662246945e03, 0.1553322001835378e03, 0.3480160396513587e03, -0.1773773615718412e03] GeV",
    )  # fmt: off
    parser.add_argument("--ph2", "-ph2", type=float, nargs=4, default=None,
        help="Four-momentum of the second Higgs. Default for gghhh = [0.3563696374921922e03, -0.1680238900851100e02, -0.3187291102436005e03, 0.9748719163688098e02] GeV",
    )  # fmt: off
    parser.add_argument("--ph3", "-ph3", type=float, nargs=4, default=None,
        help="Four-momentum of the third Higgs. Default for gghhh = [0.2050747962831133e03, -0.1385298111750267e03, -0.2928692940775817e02, 0.7989016993496030e02] GeV",
    )  # fmt: off
    parser.add_argument("--helicities", type=int, nargs=5, default=[+1, +1, +0, +0, +0],
        help="Helicities of the particles in the process. Default = %(default)s",
    )  # fmt: off
    parser.add_argument("--n_loops", type=int, choices=[1, 2, 3], default=1,
        help="Number of loops in the process. Default = %(default)s",
    )  # fmt: off
    parser.add_argument("--clean", "-c", action="store_true", default=False,
        help="Clean existing generated states before generating new ones. Default = %(default)s",
    )  # fmt: off

    parser.add_argument("--integrand-implementation", "-ii", type=str, default="gammaloop", choices=["gammaloop", "spenso_parametric", "spenso_summed"],
        help="Integrand implementation to employ. Default = %(default)s",
    )  # fmt: off
    parser.add_argument("--multi_channeling", "-mc", action="store_true", default=False, help="Consider a multi-channeled integrand.")

    # Add subcommands and their options
    subparsers = parser.add_subparsers(title="commands", dest="command", help="Various commands available")

    # create the parser for the "generate" command
    parser_generate = subparsers.add_parser("generate", help="Generate the process.")
    parser_generate.add_argument("--generation-type", "-t", type=str, nargs=1, choices=["gammaloop", "spenso", "all"], default="all",
        help="Select generation type",
    )  # fmt: off
    parser_generate.add_argument(
        "--full_spenso_integrand_strategy",
        "-g",
        type=str,
        choices=["merging", "summing", "function_map"],
        default=None,
        help="Strategy to generate the full spenso integrand when explicitly summing over orientation in the evaluator for performances. Default = %(default)s",
    )
    parser_generate.add_argument("--n-iterations-hornerscheme", "-nhorner", type=int, default=100,
        help="Number of iterations for the Horner scheme optimization. Default = %(default)s",
    )  # fmt: off
    parser_generate.add_argument("--n-iterations-cpe", "-ncpe", type=int, default=None,
        help="Number of iterations for the CPE optimization. Default = until exhaustion",
    )  # fmt: off

    # create the parser for the "inspect" command
    parser_inspect = subparsers.add_parser("inspect", help="Inspect evaluation of a sample point of the integration space.")
    parser_inspect.add_argument("--point", "-p", type=float, nargs="*",
        help="Sample point to inspect",
    )  # fmt: off
    parser_inspect.add_argument("--x_space", action="store_true", default=False,
        help="Inspect a point given in x-space. Default = %(default)s",
    )  # fmt: off
    parser_inspect.add_argument("--full_integrand", action="store_true", default=False,
        help="Inspect the complete integrand, incl. multi-channeling. Default = %(default)s",
    )  # fmt: off

    # create the parser for the "integrate" command
    parser_integrate = subparsers.add_parser("integrate", help="Integrate the loop amplitude.")
    parser_integrate.add_argument("--n_iterations", "-n", type=int, default=10,
        help="Number of iterations to perform. Default = %(default)s",
    )  # fmt: off
    parser_integrate.add_argument("--points_per_iteration", "-ppi", type=int, default=1000,
        help="Number of points per iteration. Default = %(default)s",
    )  # fmt: off
    parser_integrate.add_argument("--integrator", "-it", type=str, default="gammaloop", choices=["naive", "symbolica", "vegas", "gammaloop"],
        help="Integrator selected. Default = %(default)s",
    )  # fmt: off
    parser_integrate.add_argument("--n_cores", "-nc", type=int, default=1,
        help="Number of cores to run with. Default = %(default)s",
    )  # fmt: off

    parser_integrate.add_argument("--target", "-t", type=complex, default=None,
        help="Target value for the integration. Default = %(default)s",
    )  # fmt: off
    parser_integrate.add_argument("--phase", "-p", type=str, default="real", choices=["real", "imag"],
        help="Phase of the amplitude to compute. Default = %(default)s",
    )  # fmt: off
    parser_integrate.add_argument("--seed", "-s", type=int, default=1337,
        help="Specify random seed. Default = %(default)s",
    )  # fmt: off
    parser_integrate.add_argument("--restart", "-r", action="store_true", default=False,
        help="Restart the integration from previous results. Default = %(default)s",
    )  # fmt: off

    # Create the parser for the "plot" command
    parser_plot = subparsers.add_parser("plot", help="Plot the integrand.")
    parser_plot.add_argument("--xs", type=int, nargs=2, default=[0,1],
        help="Chosen 2-dimension projection of the integration space",
    )  # fmt: off
    parser_plot.add_argument("--fixed_x", type=float, default=0.75,
        help="Value of x kept fixed: default = %(default)s",
    )  # fmt: off
    parser_plot.add_argument("--range", "-r", type=float, nargs=2, default=[0.0, 1.0],
        help="range to plot. default = %(default)s",
    )  # fmt: off
    parser_plot.add_argument("--x_space", action="store_true", default=False,
        help="Plot integrand in x-space. Default = %(default)s",
    )  # fmt: off
    parser_plot.add_argument("--3D", "-3D", action="store_true", default=False,
        help="Make a 3D plot. Default = %(default)s",
    )  # fmt: off
    parser_plot.add_argument("--mesh_size", "-ms", type=int, default=300,
        help="Number of bins in meshing: default = %(default)s",
    )  # fmt: off
    parser_plot.add_argument("--nb_cores", "-c", type=int, default=1,
        help="Number of cores to use for plotting. Default = %(default)s",
    )  # fmt: off

    parser_plot = subparsers.add_parser("bench", help="bench the integrand.")
    parser_plot.add_argument("--n_evals", "-n", type=int, default=None,
        help="Number of points to benchmark. Default = %(default)s",
    )  # fmt: off
    parser_plot.add_argument("--target_time", "-t", type=float, default=1.0,
        help="Target time for the timing profile per repeat. Default = %(default)s",
    )  # fmt: off
    parser_plot.add_argument("--repeat", "-r", type=int, default=5,
        help="Number of repeats for the timing profile. Default = %(default)s",
    )  # fmt: off
    args = parser.parse_args(argv)
    setup_logging()

    match args.verbosity:
        case "debug":
            logger.setLevel(logging.DEBUG)
        case "info":
            logger.setLevel(logging.INFO)
        case "critical":
            logger.setLevel(logging.CRITICAL)

    ps_point_is_default = args.pg1 is None or args.pg2 is None or args.ph1 is None or args.ph2 is None or args.ph3 is None
    match args.process:
        case "scalar_gravity":
            if args.m_top is None:
                args.m_top = 1.0
            if args.m_higgs is None:
                args.m_higgs = 1.0
            if ps_point_is_default:
                ps_point = [
                    LorentzVector(2.0, 1.0, 1.0, 1.0),
                    LorentzVector(2.0, -1.0, -1.0, -1.0),
                    LorentzVector(2.0, 1.0, -1.0, 1.0),
                    LorentzVector(2.0, -1.0, 1.0, -1.0),
                ]
            else:
                ps_point = [
                    LorentzVector(args.pg1[0], args.pg1[1], args.pg1[2], args.pg1[3]),
                    LorentzVector(args.pg2[0], args.pg2[1], args.pg2[2], args.pg2[3]),
                    LorentzVector(args.ph1[0], args.ph1[1], args.ph1[2], args.ph1[3]),
                    LorentzVector(args.ph2[0], args.ph2[1], args.ph2[2], args.ph2[3]),
                ]
        case _:
            if args.m_top is None:
                args.m_top = 173.0
            if args.m_higgs is None:
                args.m_higgs = 125.0
            if ps_point_is_default:
                ps_point = [
                    LorentzVector(500.0, 0.0, 0.0, 500.0),
                    LorentzVector(500.0, 0.0, 0.0, -500.0),
                    LorentzVector(438.5555662246945, 155.3322001835378, 348.0160396513587, -177.3773615718412),
                    LorentzVector(356.3696374921922, -16.80238900851100, -318.7291102436005, 97.48719163688098),
                    LorentzVector(205.0747962831133, -138.5298111750267, -29.28692940775817, 79.89016993496030),
                ]
            else:
                ps_point = [
                    LorentzVector(args.pg1[0], args.pg1[1], args.pg1[2], args.pg1[3]),
                    LorentzVector(args.pg2[0], args.pg2[1], args.pg2[2], args.pg2[3]),
                    LorentzVector(args.ph1[0], args.ph1[1], args.ph1[2], args.ph1[3]),
                    LorentzVector(args.ph2[0], args.ph2[1], args.ph2[2], args.ph2[3]),
                    LorentzVector(args.ph3[0], args.ph3[1], args.ph3[2], args.ph3[3]),
                ]

    match args.process:
        case "gghhh":
            if args.overwrite_process_basename is not None:
                GGHHH.name = args.overwrite_process_basename
            process = GGHHH(
                args.m_top,
                args.m_higgs,
                ps_point,
                args.helicities,
                args.n_loops,
                toml_config_path=args.gammaloop_configuration,
                runtime_toml_config_path=args.runtime_configuration,
                clean=args.clean,
                gammaloop_settings=args.gammaloop_settings,
            )
        case "template_process":
            if args.overwrite_process_basename is not None:
                TemplateProcess.name = args.overwrite_process_basename
            process = TemplateProcess(
                args.m_top,
                args.m_higgs,
                ps_point,
                args.helicities,
                args.n_loops,
                toml_config_path=args.gammaloop_configuration,
                runtime_toml_config_path=args.runtime_configuration,
                clean=args.clean,
                gammaloop_settings=args.gammaloop_settings,
            )
        case "scalar_gravity":
            process = ScalarGravity(
                args.m_top,
                args.m_higgs,
                ps_point,
                args.n_loops,
                args.diagrams,
                toml_config_path=args.gammaloop_configuration,
                runtime_toml_config_path=args.runtime_configuration,
                clean=args.clean,
                gammaloop_settings=args.gammaloop_settings,
            )
        case "dy":
            if args.overwrite_process_basename is not None:
                DY.name = args.overwrite_process_basename
            process = DY(
                args.m_top,
                args.m_higgs,
                ps_point,
                args.helicities,
                args.n_loops,
                toml_config_path=args.gammaloop_configuration,
                runtime_toml_config_path=args.runtime_configuration,
                clean=args.clean,
                gammaloop_settings=args.gammaloop_settings,
            )
        case _:
            raise pygloopException(f"Process {args.process} not implemented.")

    match args.command:
        case "generate":
            logger.info("Generating graphs ...")
            process.generate_graphs()
            if "gammaloop" in args.generation_type or "all" in args.generation_type:
                logger.info("Generating gammaloop code ...")
                process.generate_gammaloop_code()
                logger.info("Gammaloop code generation completed.")
            if "spenso" in args.generation_type or "all" in args.generation_type:
                logger.info("Generating spenso code ...")
                process.generate_spenso_code(
                    full_spenso_integrand_strategy=args.full_spenso_integrand_strategy,
                    n_hornerscheme_iterations=args.n_iterations_hornerscheme,
                    n_cpe_iterations=args.n_iterations_cpe,
                )
                logger.info("Spenso code generation completed.")

        case "inspect":
            process.set_log_level(logging.WARNING)
            if args.full_integrand:
                res = process.integrand_xspace(
                    args.point,
                    args.parameterisation,
                    args.integrand_implementation,
                    args.multi_channeling,
                )
                logger.info(
                    f"Full integrand evaluated at xs = [{Colour.BLUE}{', '.join(f'{xi:+.16e}' for xi in args.point)}{Colour.END}] : {Colour.GREEN}{
                        res:+.16e}{Colour.END}"
                )
            else:
                if len(args.point) % 3 != 0:
                    raise pygloopException("Expected a multiple of 3 values for --point.")
                point = [args.point[i : i + 3] for i in range(0, len(args.point), 3)]
                k_to_inspect = []
                jacobian = 1.0
                if args.x_space:
                    for p in point:
                        k, j = process.parameterize(p, args.parameterisation)
                        k_to_inspect.append(k)
                        jacobian *= j
                else:
                    for p in point:
                        k_to_inspect.append(Vector(*p))
                res = process.integrand(k_to_inspect, args.integrand_implementation)
                report = f"Integrand evaluated at loop momentum ks = [{Colour.BLUE}{
                    ','.join('[' + ', '.join(f'{ki:+.16e}' for ki in k.to_list()) + ']' for k in k_to_inspect)
                }{Colour.END}] : {Colour.GREEN}{res:+.16e}{Colour.END}"
                if args.x_space:
                    report += f" (excl. jacobian = {jacobian:+.16e})"
                logger.info(report)
            process.set_log_level(logging.INFO)

        case "integrate":
            if args.seed is not None:
                random.seed(args.seed)
                if args.integrator == "naive" and args.n_cores != 1:
                    logger.info("Note that setting the random seed only ensure reproducible results with the naive integrator and a single core.")

            if args.n_cores > multiprocessing.cpu_count():
                raise pygloopException(
                    f"Number of cores requested ({args.n_cores}) is larger than number of available cores ({multiprocessing.cpu_count()})"
                )

            direct_target = None
            if args.target is not None:
                if args.phase == "real":
                    direct_target = args.target.real
                else:
                    direct_target = args.target.imag

                if args.integrand_implementation != "gammaloop":
                    args.target = direct_target

            t_start = time.time()
            res = process.integrate(**vars(args))  # type: ignore
            integration_time = time.time() - t_start
            # tabs = "\t" * 5
            new_line = "\n"
            logger.info("-" * 80)
            logger.info(
                f"Integration with settings below completed in {Colour.GREEN}{integration_time:.2f}s{Colour.END}:{new_line}"
                f"{new_line.join(f'| {Colour.BLUE}{k:<30s}{Colour.END}: {Colour.GREEN}{pformat(v)}{Colour.END}' for k, v in vars(args).items())}"
                f"{new_line}| {new_line}{res.str_report(direct_target)}"  # type: ignore
            )  # type: ignore
            logger.info("-" * 80)

        case "plot":
            process.plot(**vars(args))

        case "bench":
            process.set_log_level(logging.CRITICAL)
            try:
                disk_size = process.get_size_on_disk(args.integrand_implementation)  # type: ignore
            except Exception:
                disk_size = None

            if disk_size is None:
                logger.info(f"{Colour.BLUE}Size on disk [MB]: {Colour.END}{Colour.RED}N/A{Colour.END}")
            else:
                logger.info(f"{Colour.BLUE}Size on disk [MB]: {Colour.END}{Colour.GREEN}{disk_size / 1_000_000.0:.2f}{Colour.END}")

            def f():
                k_to_inspect = [Vector(*[random.random() for _ in range(3)]) for _ in range(args.n_loops)]
                process.integrand(k_to_inspect, args.integrand_implementation)

            res, st = time_function(f, repeats=args.repeat, target_time=args.target_time, number=args.n_evals, warmup_evals=2)
            # logger.info("Last eval result:", res)
            logger.info(f"{Colour.BLUE}calls / run      : {Colour.END}{st['number']}")
            logger.info(f"{Colour.BLUE}median (µs)      : {Colour.END}{Colour.GREEN}{st['median_s'] * 1e6:.1f}{Colour.END}")
            logger.info(f"{Colour.BLUE}min    (µs)      : {Colour.END}{Colour.GREEN}{st['min_s'] * 1e6:.1f}{Colour.END}")

        case _:
            raise pygloopException(f"Command {args.command} not implemented.")

    return 0


if __name__ == "__main__":
    SystemExit(main())
