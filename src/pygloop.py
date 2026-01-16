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

from processes.gghhh import GGHHH
from utils.utils import (
    SRC_DIR,
    Colour,
    logger,
    pygloopException,
    setup_logging,
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

    parser.add_argument("--m_top", type=float, default=173.0,
        help="Mass of the internal top quark. Default = %(default)s GeV",
    )  # fmt: off
    _ = parser.add_argument("--m_higgs", type=float, default=125.0,
        help="Higgs mass. Default = %(default)s GeV",
    )  # fmt: off
    parser.add_argument("--pg1", "-pg1", type=float, nargs=4, default=[500.0, 0.0, 0.0, 500.0],
        help="Four-momentum of the first gluon. Default = %(default)s GeV",
    )  # fmt: off
    parser.add_argument("--pg2", "-pg2", type=float, nargs=4, default=[500.0, 0.0, 0.0, -500.0],
        help="Four-momentum of the second gluon. Default = %(default)s GeV",
    )  # fmt: off
    parser.add_argument("--ph1", "-ph1", type=float, nargs=4, default=[0.4385555662246945e03, 0.1553322001835378e03, 0.3480160396513587e03, -0.1773773615718412e03],
        help="Four-momentum of the first Higgs. Default = %(default)s GeV",
    )  # fmt: off
    parser.add_argument("--ph2", "-ph2", type=float, nargs=4, default=[0.3563696374921922e03, -0.1680238900851100e02, -0.3187291102436005e03, 0.9748719163688098e02],
        help="Four-momentum of the second Higgs. Default = %(default)s GeV",
    )  # fmt: off
    parser.add_argument("--ph3", "-ph3", type=float, nargs=4, default=[0.2050747962831133e03, -0.1385298111750267e03, -0.2928692940775817e02, 0.7989016993496030e02],
        help="Four-momentum of the third Higgs. Default = %(default)s GeV",
    )  # fmt: off
    parser.add_argument("--helicities", type=int, nargs=5, default=[+1, +1, +0, +0, +0],
        help="Helicities of the particles in the process. Default = %(default)s",
    )  # fmt: off
    parser.add_argument("--n_loops", type=int, choices=[1, 2], default=1,
        help="Number of loops in the process. Default = %(default)s",
    )  # fmt: off
    parser.add_argument("--clean", "-c", action="store_true", default=False,
        help="Clean existing generated states before generating new ones. Default = %(default)s",
    )  # fmt: off
    parser.add_argument("--process", "-p", type=str, choices=["gghhh"], default="gghhh",
        help="Process to consider. Default = %(default)s",
    )  # fmt: off
    parser.add_argument("--integrand-implementation", "-ii", type=str, default="gammaloop", choices=["gammaloop", "spenso"],
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
    parser_generate.add_argument("--x_space", action="store_true", default=False,
        help="Inspect a point given in x-space. Default = %(default)s",
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

    args = parser.parse_args(argv)
    setup_logging()

    match args.verbosity:
        case "debug":
            logger.setLevel(logging.DEBUG)
        case "info":
            logger.setLevel(logging.INFO)
        case "critical":
            logger.setLevel(logging.CRITICAL)

    ps_point = [
        LorentzVector(args.pg1[0], args.pg1[1], args.pg1[2], args.pg1[3]),
        LorentzVector(args.pg2[0], args.pg2[1], args.pg2[2], args.pg2[3]),
        LorentzVector(args.ph1[0], args.ph1[1], args.ph1[2], args.ph1[3]),
        LorentzVector(args.ph2[0], args.ph2[1], args.ph2[2], args.ph2[3]),
        LorentzVector(args.ph3[0], args.ph3[1], args.ph3[2], args.ph3[3]),
    ]

    match args.process:
        case "gghhh":
            process = GGHHH(
                args.m_top,
                args.m_higgs,
                ps_point,
                args.helicities,
                args.n_loops,
                toml_config_path=args.gammaloop_configuration,
                runtime_toml_config_path=args.runtime_configuration,
                clean=args.clean,
            )
        case _:
            raise pygloopException(f"Process {args.process} not implemented.")

    match args.command:
        case "generate":
            logger.info("Generating graphs ...")
            process.generate_graphs()
            if args.generation_type == "gammaloop" or args.generation_type == "all":
                logger.info("Generating gammaloop code ...")
                process.generate_gammaloop_code()
                logger.info("Gammaloop code generation completed.")
            if args.generation_type == "spenso" or args.generation_type == "all":
                logger.info("Generating spenso code ...")
                process.generate_spenso_code()
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
        case _:
            raise pygloopException(f"Command {args.command} not implemented.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
