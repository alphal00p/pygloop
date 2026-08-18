#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import logging
import multiprocessing
import os
import random
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from pprint import pformat

DY = None
GGHHH = None
ScalarGravity = None
TemplateProcess = None

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

from processes.dy.dy_card import (  # noqa: E402
    DYCardError,
    EffectiveDYCard,
    compatibility_diff,
    format_compatibility_diff,
    generation_settings_fingerprint,
    load_dy_card,
    normalise_generation_compatibility_settings,
)
from processes.dy.dy_runtime_parameters import (  # noqa: E402
    DYRuntimeParameterError,
    merge_runtime_parameter_overrides,
    parse_runtime_parameter_assignments,
)
from processes.dy.dy_top_self_energy import (  # noqa: E402
    TOP_SELF_ENERGY_RENORMALISATION_MODES,
    resolve_top_self_energy_renormalisation,
)


def _load_process_class(process_name: str) -> None:
    """Import only the heavyweight process implementation execution selected."""
    global DY, GGHHH, ScalarGravity, TemplateProcess

    if process_name == "dy" and DY is None:
        try:
            from processes.dy.dy import DY as loaded_dy

            DY = loaded_dy
        except Exception as exc:
            print(f"Warning: DY process not available ({exc}).", file=sys.stderr)
    elif process_name == "gghhh" and GGHHH is None:
        try:
            from processes.gghhh.gghhh import GGHHH as loaded_gghhh

            GGHHH = loaded_gghhh
        except Exception as exc:
            print(f"Warning: GGHHH process not available ({exc}).", file=sys.stderr)
    elif process_name == "scalar_gravity" and ScalarGravity is None:
        try:
            from processes.scalar_gravity.scalar_gravity import (
                ScalarGravity as loaded_scalar_gravity,
            )

            ScalarGravity = loaded_scalar_gravity
        except Exception as exc:
            print(
                f"Warning: ScalarGravity process not available ({exc}).",
                file=sys.stderr,
            )
    elif process_name == "template_process" and TemplateProcess is None:
        try:
            from processes.template_process import TemplateProcess as loaded_template

            TemplateProcess = loaded_template
        except Exception as exc:
            print(
                f"Warning: TemplateProcess not available ({exc}).", file=sys.stderr
            )


def _require_process_class(process_name: str, process_class: type | None) -> type:
    if process_class is None:
        raise pygloopException(
            f"Process '{process_name}' is not available in this environment. "
            f"Install its dependencies or pick another process with --process."
        )
    return process_class


def _parse_bool_flag(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected True or False, got '{value}'.")


def _iter_parser_actions(
    parser: argparse.ArgumentParser,
) -> Iterator[argparse.Action]:
    """Yield actions from a parser and all of its subparsers."""
    for action in parser._actions:
        yield action
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                yield from _iter_parser_actions(subparser)


def _explicit_cli_destinations(
    parser: argparse.ArgumentParser, argv: list[str]
) -> set[str]:
    """Return argparse destinations explicitly named on the command line.

    Card merging must distinguish a parser default from an option explicitly
    supplied with that same value.  A second parse with defaults suppressed
    also handles aliases, attached short values, and list/append actions while
    keeping card list replacement semantics.
    """
    actions = list(_iter_parser_actions(parser))
    saved_defaults = [(action, action.default) for action in actions]
    try:
        for action in actions:
            action.default = argparse.SUPPRESS
        explicit_namespace = parser.parse_args(argv)
    finally:
        for action, default in saved_defaults:
            action.default = default
    return set(vars(explicit_namespace)).difference({"command"})


def _runtime_arguments(args: argparse.Namespace) -> dict[str, object]:
    """Exclude card control-plane flags from existing process call namespaces."""
    values = vars(args).copy()
    for destination in (
        "dy_card",
        "dy_card_check",
        "dy_dump_effective_card",
        "dy_allow_unverified_bundle",
        "dy_runtime_parameter_overrides",
        "dy_runtime_parameters",
    ):
        values.pop(destination, None)
    return values


def _verify_card_bundle_compatibility(
    process: object,
    effective_card: EffectiveDYCard,
    *,
    allow_unverified: bool,
) -> None:
    """Fail before sampling when a card does not match the loaded DY bundle."""
    compiled_bundle = getattr(process, "compiled_bundle", None)
    if compiled_bundle is None:
        raise pygloopException(
            "Card-driven DY integration could not load a compiled bundle to verify."
        )
    metadata = compiled_bundle.bundle_metadata
    generated_settings = metadata.get("dy_generation_settings")
    stored_fingerprint = metadata.get("dy_generation_settings_fingerprint")
    if generated_settings is None or stored_fingerprint is None:
        if not allow_unverified:
            raise pygloopException(
                "The selected DY bundle has no generation-card provenance. "
                "Regenerate it with --dy-card or explicitly pass "
                "--dy-allow-unverified-bundle."
            )
        logger.warning(
            "Integrating legacy DY bundle %s without generation-settings "
            "verification because --dy-allow-unverified-bundle was supplied.",
            compiled_bundle.integrand_name,
        )
        return
    if not isinstance(generated_settings, dict) or not isinstance(
        stored_fingerprint, str
    ):
        raise pygloopException(
            "The selected DY bundle contains malformed generation provenance."
        )
    recalculated_fingerprint = generation_settings_fingerprint(generated_settings)
    if recalculated_fingerprint != stored_fingerprint:
        raise pygloopException(
            "The selected DY bundle contains inconsistent generation provenance: "
            "its stored settings do not match its stored fingerprint."
        )

    requested_settings = effective_card.bundle_metadata()["dy_generation_settings"]
    canonical_generated = normalise_generation_compatibility_settings(
        generated_settings
    )
    canonical_requested = normalise_generation_compatibility_settings(
        requested_settings
    )
    if canonical_generated != canonical_requested:
        differences = compatibility_diff(
            canonical_generated,
            canonical_requested,
        )
        if not differences:
            raise pygloopException(
                "The selected DY bundle generation fingerprint is inconsistent "
                "with the requested normalized settings."
            )
        raise pygloopException(
            format_compatibility_diff(compiled_bundle.integrand_name, differences)
        )

    generated_source_hashes = metadata.get("dy_source_hashes")
    if isinstance(generated_source_hashes, dict):
        current_source_hashes = dict(effective_card.source_hashes)
        changed_sources = sorted(
            path
            for path in set(generated_source_hashes) | set(current_source_hashes)
            if generated_source_hashes.get(path) != current_source_hashes.get(path)
        )
        if changed_sources:
            logger.warning(
                "DY bundle source provenance differs from the current checkout "
                "for: %s. Physics settings match, so integration will continue.",
                ", ".join(changed_sources),
            )


def _stamp_generated_dy_bundle(
    process: object, effective_card: EffectiveDYCard
) -> dict[str, object]:
    """Atomically stamp the final serial or merged DY bundle."""
    from processes.dy.dy_evaluators import DYCompiledBundle

    process_name = str(getattr(process, "process_name"))
    integrand_name = str(process.get_integrand_name())
    updates = effective_card.bundle_metadata()
    DYCompiledBundle.augment_metadata(
        process_name,
        integrand_name,
        updates,
    )
    return updates


def _preflight_card_generation(
    process: object,
    effective_card: EffectiveDYCard,
    *,
    clean: bool,
) -> bool:
    """Validate an existing final bundle before a clean-false generation.

    Returns true when matching provenance already exists.  The caller can then
    preserve it if generation recycles the bundle unchanged, avoiding a false
    update of source/card provenance.
    """
    if clean:
        return False
    from processes.dy.dy_evaluators import DYCompiledBundle

    process_name = str(getattr(process, "process_name"))
    integrand_name = str(process.get_integrand_name())
    metadata_path = DYCompiledBundle.metadata_path(process_name, integrand_name)
    if not os.path.isfile(metadata_path):
        return False
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise pygloopException(
            f"Could not inspect existing DY bundle metadata '{metadata_path}': {exc}"
        ) from exc
    generated_settings = metadata.get("dy_generation_settings")
    stored_fingerprint = metadata.get("dy_generation_settings_fingerprint")
    if not isinstance(generated_settings, dict) or not isinstance(
        stored_fingerprint, str
    ):
        raise pygloopException(
            f"Existing bundle {integrand_name} has no verified generation "
            "settings. Use --clean to regenerate it before stamping card provenance."
        )
    if generation_settings_fingerprint(generated_settings) != stored_fingerprint:
        raise pygloopException(
            f"Existing bundle {integrand_name} has corrupt generation provenance. "
            "Use --clean to regenerate it."
        )
    requested_settings = effective_card.bundle_metadata()["dy_generation_settings"]
    canonical_generated = normalise_generation_compatibility_settings(
        generated_settings
    )
    canonical_requested = normalise_generation_compatibility_settings(
        requested_settings
    )
    if canonical_generated != canonical_requested:
        differences = compatibility_diff(
            canonical_generated,
            canonical_requested,
        )
        detail = format_compatibility_diff(integrand_name, differences)
        raise pygloopException(
            f"{detail}\nUse --clean to regenerate instead of recycling this bundle."
        )
    return True


def _finalize_card_generation_metadata(
    process: object,
    effective_card: EffectiveDYCard,
    *,
    matching_provenance_existed: bool,
) -> dict[str, object]:
    """Stamp a new bundle, or preserve truthful provenance on exact reuse."""
    if matching_provenance_existed:
        from processes.dy.dy_evaluators import DYCompiledBundle

        metadata_path = DYCompiledBundle.metadata_path(
            str(getattr(process, "process_name")), str(process.get_integrand_name())
        )
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                current_metadata = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError):
            current_metadata = {}
        if (
            current_metadata.get("dy_generation_settings_fingerprint")
            == effective_card.generation_fingerprint
        ):
            return {
                key: value
                for key, value in current_metadata.items()
                if key.startswith("dy_")
            }
    return _stamp_generated_dy_bundle(process, effective_card)


def main(argv: list[str] | None = None) -> dict[str, object] | int:
    # create the top-level parser
    class FloatArgParser(argparse.ArgumentParser):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._negative_number_matcher = re.compile(
                r"^-?\d+(\.\d*)?([eE][-+]?\d+)?$"
            )  # type: ignore

    parser = FloatArgParser(prog="pygloop")

    parser.add_argument(
        "--dy-card",
        type=Path,
        default=None,
        metavar="PATH",
        help="Load shared DY generation/integration settings from a TOML card.",
    )
    card_exit_group = parser.add_mutually_exclusive_group()
    card_exit_group.add_argument(
        "--dy-card-check",
        action="store_true",
        default=False,
        help="Validate and normalize the selected DY card command, then exit.",
    )
    card_exit_group.add_argument(
        "--dy-dump-effective-card",
        action="store_true",
        default=False,
        help="Print the fully merged effective DY card as canonical TOML, then exit.",
    )
    parser.add_argument(
        "--dy-allow-unverified-bundle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow card-driven integration of a legacy bundle without DY provenance.",
    )

    parser.add_argument("--process", "-p", type=str, choices=["gghhh", "template_process", "dy", "scalar_gravity"], default="gghhh",
        help="Process to consider. Default = %(default)s",
    )  # fmt: off
    parser.add_argument("--general_settings", "-gs", type=str, nargs="*", default=None,
        help="General settings to set as class variables to the process. Default = %(default)s",
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

    parser.add_argument("--parameterisation", "-param", type=str, choices=["cartesian", "spherical", "log_spherical"], default="spherical",
        help="Parameterisation to employ.",
    )  # fmt: off
    parser.add_argument(
        "--dy-skip-ps-validation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="DY only: skip phase-space momentum-conservation validation.",
    )
    parser.add_argument(
        "--dy-rotation-check-digits",
        type=int,
        default=None,
        help="DY only: enable rotational-invariance veto with N relative-accuracy digits (disabled by default).",
    )
    parser.add_argument(
        "--dy-rotation-check-eps",
        type=float,
        default=1e-15,
        help="DY only: absolute tolerance for original/rotated sample agreement.",
    )
    parser.add_argument(
        "--dy-rotation-check-arb-digits",
        type=int,
        default=80,
        help="DY only: decimal precision used for arbitrary-precision retry after a rotation-check failure.",
    )
    parser.add_argument(
        "--dy-fallback-precision",
        "--dy-higher-precision-digits",
        dest="dy_fallback_precision",
        type=int,
        default=None,
        help=(
            "DY only: select the integration-time higher-precision backend. "
            "A value of 32 uses Symbolica DoubleFloat; any other positive value "
            "uses Arb at that many decimal digits. New bundles store both "
            "backends. Defaults to --dy-rotation-check-arb-digits."
        ),
    )
    parser.add_argument(
        "--dy-stability-backend",
        choices=["double-float", "arb"],
        default=None,
        help=(
            "DY only: integration-time higher-precision backend. 'double-float' "
            "selects 32 digits; 'arb' selects --dy-stability-arb-digits."
        ),
    )
    parser.add_argument(
        "--dy-stability-arb-digits",
        type=int,
        default=None,
        help=(
            "DY only: decimal digits for --dy-stability-backend arb "
            "(default: --dy-rotation-check-arb-digits)."
        ),
    )
    parser.add_argument(
        "--dy-stability-rtol",
        type=float,
        default=None,
        help=(
            "DY only: direct relative tolerance for original/rotated final "
            "sample agreement (default: 10^-N from --dy-rotation-check-digits)."
        ),
    )
    parser.add_argument(
        "--dy-stability-atol",
        type=float,
        default=None,
        help=(
            "DY only: direct absolute tolerance for original/rotated final "
            "sample agreement (default: --dy-rotation-check-eps)."
        ),
    )
    parser.add_argument(
        "--dy-theta-tol",
        type=float,
        default=0.0,
        help="DY only: theta support tolerance used in compiled and arbitrary-precision DY evaluation.",
    )
    parser.add_argument(
        "--dy-large-weight-precision",
        type=float,
        default=None,
        help=(
            "DY only: skip the float rotation and validate both original and "
            "rotated samples in higher precision when |final sample| exceeds this."
        ),
    )
    parser.add_argument(
        "--dy-large-weight-clip",
        type=float,
        default=None,
        help=(
            "DY only: set a higher-precision-validated final sample to zero when "
            "its absolute value exceeds this threshold."
        ),
    )
    parser.add_argument(
        "--dy-soft-mirror-edge",
        action="append",
        default=None,
        metavar="[GRAPH:]EDGE",
        help=(
            "DY only: pair samples around the selected soft edge. Shifted edges "
            "use conditional soft-centred spherical coordinates and an antipodal "
            "pair; origin-centred edges retain the beam-equatorial pair. Repeat "
            "GRAPH:EDGE for multi-graph bundles; a bare EDGE is accepted only "
            "for a single-graph bundle."
        ),
    )
    parser.add_argument(
        "--dy-large-weight-threshold",
        type=float,
        default=None,
        help=(
            "DY only: compatibility alias for --dy-large-weight-precision. "
            "With --dy-zero-large-weight-samples it also supplies the clip threshold."
        ),
    )
    parser.add_argument(
        "--dy-zero-large-weight-samples",
        action="store_true",
        default=False,
        help=(
            "DY only: compatibility mode that uses --dy-large-weight-threshold "
            "as --dy-large-weight-clip."
        ),
    )
    parser.add_argument(
        "--dy-integrated-uv-ct-filter",
        choices=["all", "only", "exclude"],
        default="all",
        help=(
            "DY zenos integration only: select integrated UV counterterm terms. "
            "'all' integrates every term, 'only' integrates only integrated UV "
            "counterterms, and 'exclude' integrates everything except them."
        ),
    )
    parser.add_argument(
        "--dy-accept-all-arb-retries",
        action="store_true",
        default=False,
        help=(
            "Deprecated no-op: higher-precision retries always require agreement "
            "between the original and rotated samples."
        ),
    )
    parser.add_argument(
        "--dy-final-state",
        type=str,
        nargs="+",
        default=None,
        help="DY only: final-state particle labels used when routing cut graphs. Example: --dy-final-state a or --dy-final-state t t",
    )
    parser.add_argument(
        "--dy-process-name",
        type=str,
        default=None,
        help="DY only: process label passed to the downstream DY integrand/evaluator pipeline.",
    )
    parser.add_argument(
        "--dy-channel",
        type=int,
        nargs=2,
        default=None,
        metavar=("IN1", "IN2"),
        help="DY only: partonic channel to generate/process at one or two loops, e.g. --dy-channel 0 0 for gg, 0 1 for qg, or 1 -1 for qq~.",
    )
    parser.add_argument(
        "--external_gluon_polarisation",
        "--external-gluon-polarisation",
        type=_parse_bool_flag,
        default=False,
        help="DY generation only: replace cut external p1/p2 gluon metric sums by the axial physical-polarisation projector.",
    )
    parser.add_argument(
        "--dy-symmetrise-p1-p2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "DY generation/evaluation only: use the p1<->p2-symmetrised "
            "initial-state graph construction."
        ),
    )
    parser.add_argument(
        "--dy-parallel-graphs",
        type=int,
        default=1,
        help="DY two-loop generation only: process this many source graphs in parallel. Default 1 preserves serial generation.",
    )
    parser.add_argument(
        "--dy-integrate-beams",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="DY only: sample beam fractions x1 and x2 in [0,1] and evaluate the hard process in the partonic centre-of-mass frame at shat=x1*x2*e_cm^2.",
    )
    parser.add_argument(
        "--dy-beam-parameterisation",
        choices=["x1_x2", "beta_y"],
        default="x1_x2",
        help=(
            "DY ttbar beam convolution only: sample the beam fractions directly "
            "(x1_x2, default) or through threshold-adapted beta and rapidity "
            "coordinates (beta_y)."
        ),
    )
    parser.add_argument(
        "--dy-z-bin",
        type=float,
        nargs=2,
        default=None,
        metavar=("ZMIN", "ZMAX"),
        help=(
            "DY beam convolution only: integrate directly over one physical "
            "z bin ZMIN <= z <= ZMAX, including the bin-width Jacobian."
        ),
    )
    parser.add_argument(
        "--dy-q-min",
        type=float,
        default=None,
        help=(
            "DY beam convolution only: minimum virtual-photon mass Q in GeV, "
            "with Q^2=z*x1*x2*e_cm^2."
        ),
    )
    parser.add_argument(
        "--dy-q-max",
        type=float,
        default=None,
        help=(
            "DY beam convolution only: maximum virtual-photon mass Q in GeV, "
            "with Q^2=z*x1*x2*e_cm^2."
        ),
    )
    parser.add_argument(
        "--dy-physical-normalisation",
        "--dy-physical-normalization",
        dest="dy_physical_normalisation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "DY beam convolution only: multiply by per-beam spin and colour "
            "averages and by 1/(2*pi)^(3*L-1)."
        ),
    )
    parser.add_argument(
        "--dy-integrated-leptonic-phase-space",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "One-loop beam-convoluted DY only: multiply the hard contribution "
            "and scheme counterterm by the coupling-stripped integrated "
            "leptonic phase-space factor 1/(24*pi^2)."
        ),
    )
    parser.add_argument(
        "--dy-pdf-set",
        type=str,
        default=None,
        help="DY beam convolution only: LHAPDF set name to use for PDF weighting.",
    )
    parser.add_argument(
        "--dy-pdf-member",
        type=int,
        default=0,
        help="DY beam convolution only: LHAPDF member index (default: 0).",
    )
    parser.add_argument(
        "--dy-muf-sq",
        type=float,
        default=None,
        help=(
            "DY beam convolution only: factorisation scale squared. If omitted, "
            "equal --dy-lambda-sq and --dy-mur-sq values are used."
        ),
    )
    parser.add_argument(
        "--dy-msbar-scheme-counterterm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Beam-convoluted one-loop DY or two-loop ttbar: add the selected "
            "channel's finite scheme counterterm. The gq kernel includes its "
            "1/(2-2*eps) polarisation factor; qqbar uses the two-leg D_qq "
            "kernel at Lambdasq=1."
        ),
    )
    parser.add_argument(
        "--dy-decoupling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Ordered two-loop qqbar -> ttbar only: add the heavy-flavour "
            "decoupling Born term. This requires scheme conversion."
        ),
    )
    parser.add_argument(
        "--dy-scheme-counterterm-sobol-power",
        type=int,
        default=15,
        help=(
            "DY scheme counterterm only: use 2^POWER points per scrambled "
            "Sobol replica (default: 15)."
        ),
    )
    parser.add_argument(
        "--dy-scheme-counterterm-replicas",
        type=int,
        default=8,
        help=(
            "DY scheme counterterm only: number of independent scrambled "
            "Sobol replicas used for its uncertainty (default: 8)."
        ),
    )
    parser.add_argument(
        "--dy-scheme-counterterm-factor",
        type=float,
        default=None,
        help=(
            "DY scheme counterterm only: override the channel's finite "
            "counterterm factor (default: -2 for one-loop DY qqbar, 4*pi "
            "for two-loop ttbar qg, and alpha_s/2 for two-loop ttbar qqbar)."
        ),
    )
    parser.add_argument(
        "--dy-scheme-born-bundle",
        type=str,
        default=None,
        help=(
            "Two-loop ttbar scheme counterterm only: one-loop ttbar compiled "
            "bundle name used for the Born convolution."
        ),
    )
    parser.add_argument(
        "--dy-scheme-born-bundles",
        nargs="+",
        default=None,
        help=(
            "Two-loop scheme counterterm only: optional one-loop compiled "
            "Born bundle names. Their initial-state channels are inferred "
            "from bundle metadata."
        ),
    )
    parser.add_argument(
        "--dy-scheme-alpha-s",
        type=float,
        default=0.118,
        help=(
            "Two-loop ttbar scheme conversion: separately supplied alpha_s "
            "entering qg kernels or the qqbar D_qq coefficient (default: 0.118)."
        ),
    )
    parser.add_argument(
        "--dy-scheme-counterterm-clip",
        type=float,
        default=None,
        help=(
            "Two-loop ttbar scheme counterterm only: after 32-digit fallback, "
            "zero samples whose absolute fully weighted value exceeds this "
            "threshold."
        ),
    )
    parser.add_argument(
        "--dy-ttbar-pt-min",
        type=float,
        default=None,
        help="DY ttbar only: activate a lower pT cut on the ttbar pair and use this value as the minimum pT.",
    )
    parser.add_argument(
        "--mUV",
        "--dy-muv",
        dest="dy_muv",
        type=float,
        default=None,
        help="DY only: UV mass parameter passed to the zenos runtime evaluator.",
    )
    parser.add_argument(
        "--dy-runtime-parameter",
        dest="dy_runtime_parameter_overrides",
        action="append",
        default=None,
        metavar="NAME=DECIMAL",
        help=(
            "DY zenos integration only: override a versioned bundle runtime "
            "parameter. Repeat for multiple values; decimal text is preserved."
        ),
    )
    parser.add_argument(
        "--dy-lambda-sq",
        type=float,
        default=None,
        help=(
            "DY only: Lambdasq default recorded during generation; when "
            "explicitly supplied for integration, override it at runtime."
        ),
    )
    parser.add_argument(
        "--dy-mur-sq",
        type=float,
        default=None,
        help=(
            "DY only: mursq default recorded during generation; when "
            "explicitly supplied for integration, override it at runtime."
        ),
    )

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
    parser.add_argument("--n_loops", type=int, choices=[1, 2, 3, 4], default=1,
        help="Number of loops in the process. Default = %(default)s",
    )  # fmt: off
    parser.add_argument("--clean", "-c", action=argparse.BooleanOptionalAction, default=False,
        help="Clean existing generated states before generating new ones. Default = %(default)s",
    )  # fmt: off

    parser.add_argument("--integrand-implementation", "-ii", type=str, default="gammaloop", choices=["gammaloop", "zenos", "spenso_parametric", "spenso_summed"],
        help="Integrand implementation to employ. Default = %(default)s",
    )  # fmt: off
    parser.add_argument("--integrand-evaluator-compiler", "-iec", type=str, default="symbolica_only", choices=["symbolica_only", "symjit"],
        help="Compiler to use for the spenso integrand evaluator. Default = %(default)s",
    )  # fmt: off
    parser.add_argument(
        "--multi_channeling",
        "-mc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Consider a multi-channeled integrand.",
    )

    # Add subcommands and their options
    subparsers = parser.add_subparsers(
        title="commands", dest="command", required=True, help="Various commands available"
    )

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
    parser_generate.add_argument(
        "--dy-enable-integrated-uv-cts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="DY generation only: enable integrated UV counterterms. Disabled by default.",
    )
    parser_generate.add_argument(
        "--dy-top-self-energy-os-subtraction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "DY generation only: enable on-shell subtraction for eligible "
            "two-loop ttbar top self-energy insertions. Disabled by default."
        ),
    )
    parser_generate.add_argument(
        "--dy-top-self-energy-renormalisation",
        choices=TOP_SELF_ENERGY_RENORMALISATION_MODES,
        default=None,
        help=(
            "DY generation only: select top-self-energy renormalisation. "
            "The default is no-os; the legacy boolean selects os."
        ),
    )
    parser_generate.add_argument(
        "--dy-check-generation-limits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="DY generation only: construct limit-check evaluators and approach a test limit. Disabled by default.",
    )
    parser_generate.add_argument(
        "--dy-threshold-h-function",
        choices=["gaussian", "inverse_square_damped"],
        default=None,
        help=(
            "DY generation only: select the threshold-counterterm h function. "
            "When omitted, preserve the process default (gaussian for DY and "
            "inverse_square_damped for ttbar)."
        ),
    )

    # create the parser for the "inspect" command
    parser_inspect = subparsers.add_parser(
        "inspect", help="Inspect evaluation of a sample point of the integration space."
    )
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
    parser_integrate = subparsers.add_parser(
        "integrate", help="Integrate the loop amplitude."
    )
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
    parser_integrate.add_argument("--restart", "-r", action=argparse.BooleanOptionalAction, default=False,
        help="Restart the integration from previous results. Default = %(default)s",
    )  # fmt: off
    parser_integrate.add_argument("--run-workspace-name", "-rn", type=str, default=None,
        help="Name of the workspace to use for this run. Default = %(default)s",
    )  # fmt: off
    parser_integrate.add_argument(
        "--dy-integrated-uv-ct-filter",
        choices=["all", "only", "exclude"],
        default=argparse.SUPPRESS,
        help=(
            "DY zenos integration only: select integrated UV counterterm terms. "
            "'all' integrates every term, 'only' integrates only integrated UV "
            "counterterms, and 'exclude' integrates everything except them."
        ),
    )
    parser_integrate.add_argument(
        "--dy-integration-graphs",
        nargs="+",
        default=None,
        metavar="GRAPH",
        help=(
            "DY Symbolica multi-channel integration only: integrate a subset "
            "of compiled graph channels without regenerating the bundle. "
            "Accepts native channel names such as graph_3, zero-based channel "
            "indices, or source graph names such as GL035 when the matching "
            "ordered --diagrams list is supplied."
        ),
    )
    parser_integrate.add_argument(
        "--dy-graph-weights",
        nargs="+",
        type=float,
        default=None,
        metavar="WEIGHT",
        help=(
            "DY Symbolica multi-channel integration only: multiply each full "
            "compiled graph channel by the corresponding integration-time "
            "weight, in bundle graph-channel order."
        ),
    )

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

    parser_bench = subparsers.add_parser("bench", help="bench the integrand.")
    parser_bench.add_argument("--n_evals", "-n", type=int, default=None,
        help="Number of points to benchmark. Default = %(default)s",
    )  # fmt: off
    parser_bench.add_argument("--target_time", "-t", type=float, default=1.0,
        help="Target time for the timing profile per repeat. Default = %(default)s",
    )  # fmt: off
    parser_bench.add_argument("--repeat", "-r", type=int, default=5,
        help="Number of repeats for the timing profile. Default = %(default)s",
    )  # fmt: off
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(cli_argv)
    explicit_cli_destinations = _explicit_cli_destinations(parser, cli_argv)

    if args.dy_card is None and (
        args.dy_card_check or args.dy_dump_effective_card
    ):
        parser.error("--dy-card-check/--dy-dump-effective-card require --dy-card")

    effective_dy_card: EffectiveDYCard | None = None
    loaded_dy_card = None
    if args.dy_card is not None:
        try:
            loaded_dy_card = load_dy_card(args.dy_card)
            effective_dy_card = loaded_dy_card.merge(
                args.command,
                args,
                explicit_destinations=explicit_cli_destinations,
            )
        except DYCardError as exc:
            parser.error(str(exc))
        for destination, value in effective_dy_card.argparse_values.items():
            setattr(args, destination, value)

    top_self_energy_mode_explicit = (
        "dy_top_self_energy_renormalisation" in explicit_cli_destinations
        or loaded_dy_card is not None
        and "generate.top_self_energy_renormalisation"
        in loaded_dy_card.supplied_paths
    )
    top_self_energy_legacy_explicit = (
        "dy_top_self_energy_os_subtraction" in explicit_cli_destinations
        or loaded_dy_card is not None
        and "generate.top_self_energy_os_subtraction"
        in loaded_dy_card.supplied_paths
    )
    if args.process == "dy" and args.command == "generate":
        try:
            resolve_top_self_energy_renormalisation(
                getattr(args, "dy_top_self_energy_renormalisation", None)
                if top_self_energy_mode_explicit
                else None,
                getattr(args, "dy_top_self_energy_os_subtraction", None)
                if top_self_energy_legacy_explicit
                else None,
                legacy_is_explicit=top_self_energy_legacy_explicit,
            )
        except ValueError as exc:
            parser.error(str(exc))

    try:
        cli_runtime_overrides = parse_runtime_parameter_assignments(
            args.dy_runtime_parameter_overrides
        )
        runtime_parameter_overrides = merge_runtime_parameter_overrides(
            getattr(args, "dy_runtime_parameters", None),
            cli_runtime_overrides,
        )
    except DYRuntimeParameterError as exc:
        parser.error(str(exc))
    # Existing scalar flags remain convenient integration-time aliases.  Only
    # explicitly supplied CLI options become overrides; common card values are
    # generation defaults recorded in the bundle metadata.
    if (
        args.command == "integrate"
        and args.process == "dy"
        and args.integrand_implementation == "zenos"
    ):
        scalar_runtime_aliases = {
            "m_top": "m_top",
            "dy_muv": "muv",
            "dy_lambda_sq": "lambda_sq",
            "dy_mur_sq": "mur_sq",
        }
        for destination, parameter_name in scalar_runtime_aliases.items():
            if destination in explicit_cli_destinations:
                runtime_parameter_overrides[parameter_name] = repr(
                    float(getattr(args, destination))
                )
        # The dedicated precision-preserving syntax is the authoritative CLI
        # form when a legacy scalar alias is also present.
        runtime_parameter_overrides.update(cli_runtime_overrides)
    elif args.command != "integrate" and runtime_parameter_overrides:
        parser.error("DY runtime parameter overrides are integration-time settings.")
    if runtime_parameter_overrides and args.process != "dy":
        parser.error("DY runtime parameter overrides require --process dy.")
    if (
        runtime_parameter_overrides
        and args.integrand_implementation != "zenos"
    ):
        parser.error(
            "DY runtime parameter overrides require --integrand_implementation zenos."
        )
    args.dy_runtime_parameters = runtime_parameter_overrides
    if effective_dy_card is not None and args.command == "integrate":
        effective_dy_card = effective_dy_card.with_runtime_parameters(
            runtime_parameter_overrides
        )

    # Keep non-evaluator integration logic (thresholds and PDF scale defaults)
    # aligned with the same resolved user-facing parameters.
    if args.command == "integrate":
        if "m_top" in runtime_parameter_overrides:
            args.m_top = float(runtime_parameter_overrides["m_top"])
        if "lambda_sq" in runtime_parameter_overrides:
            args.dy_lambda_sq = float(runtime_parameter_overrides["lambda_sq"])
        if "mur_sq" in runtime_parameter_overrides:
            args.dy_mur_sq = float(runtime_parameter_overrides["mur_sq"])

    setup_logging()

    apply_stability_cli = (
        effective_dy_card is None or args.command == "integrate"
    )
    if apply_stability_cli and args.dy_stability_backend is not None:
        if args.dy_stability_backend == "double-float":
            selected_fallback_precision = 32
        else:
            selected_fallback_precision = (
                args.dy_stability_arb_digits
                if args.dy_stability_arb_digits is not None
                else args.dy_rotation_check_arb_digits
            )
            if selected_fallback_precision == 32:
                parser.error(
                    "--dy-stability-backend arb cannot use 32 digits because 32 "
                    "selects the DoubleFloat backend."
                )
        if selected_fallback_precision < 2:
            parser.error("DY stability precision must be at least two digits.")
        if (
            effective_dy_card is None
            and args.dy_fallback_precision is not None
            and args.dy_fallback_precision != selected_fallback_precision
        ):
            parser.error(
                "--dy-fallback-precision conflicts with --dy-stability-backend."
            )
        args.dy_fallback_precision = selected_fallback_precision
    elif apply_stability_cli and args.dy_stability_arb_digits is not None:
        parser.error(
            "--dy-stability-arb-digits requires --dy-stability-backend arb."
        )

    match args.verbosity:
        case "debug":
            logger.setLevel(logging.DEBUG)
        case "info":
            logger.setLevel(logging.INFO)
        case "critical":
            logger.setLevel(logging.CRITICAL)

    if effective_dy_card is not None and (
        args.dy_card_check or args.dy_dump_effective_card
    ):
        result: dict[str, object] = {
            "command": args.command,
            "process": args.process,
            "exit_code": 0,
            "status": "card_valid",
            "dy_card_path": effective_dy_card.card_path,
            "dy_source_card_sha256": effective_dy_card.source_card_sha256,
            "dy_effective_settings": effective_dy_card.as_dict(),
            "dy_generation_settings_fingerprint": (
                effective_dy_card.generation_fingerprint
            ),
            "dy_configuration_hashes": dict(
                effective_dy_card.configuration_hashes
            ),
        }
        if args.dy_dump_effective_card:
            print(effective_dy_card.to_toml(), end="")
            result["status"] = "card_dumped"
        else:
            logger.info(
                "Validated DY card %s for command %s (generation fingerprint %s).",
                effective_dy_card.card_path,
                args.command,
                effective_dy_card.generation_fingerprint,
            )
        return result

    _load_process_class(args.process)

    ps_point_is_default = (
        args.pg1 is None
        or args.pg2 is None
        or args.ph1 is None
        or args.ph2 is None
        or args.ph3 is None
    )
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
        case "dy":
            if args.m_top is None:
                args.m_top = 173.0
            if args.m_higgs is None:
                args.m_higgs = 125.0
            default_pg1 = [500.0, 0.0, 0.0, 500.0]
            default_pg2 = [500.0, 0.0, 0.0, -500.0]
            default_ph1 = [
                438.5555662246945,
                155.3322001835378,
                348.0160396513587,
                -177.3773615718412,
            ]
            default_ph2 = [
                356.3696374921922,
                -16.80238900851100,
                -318.7291102436005,
                97.48719163688098,
            ]
            default_ph3 = [
                205.0747962831133,
                -138.5298111750267,
                -29.28692940775817,
                79.89016993496030,
            ]

            pg1 = args.pg1 if args.pg1 is not None else default_pg1
            pg2 = args.pg2 if args.pg2 is not None else default_pg2
            ph1 = args.ph1 if args.ph1 is not None else default_ph1
            ph2 = args.ph2 if args.ph2 is not None else default_ph2
            ph3 = args.ph3 if args.ph3 is not None else default_ph3

            ps_point = [
                LorentzVector(pg1[0], pg1[1], pg1[2], pg1[3]),
                LorentzVector(pg2[0], pg2[1], pg2[2], pg2[3]),
                LorentzVector(ph1[0], ph1[1], ph1[2], ph1[3]),
                LorentzVector(ph2[0], ph2[1], ph2[2], ph2[3]),
                LorentzVector(ph3[0], ph3[1], ph3[2], ph3[3]),
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
            process_class = _require_process_class("gghhh", GGHHH)
        case "template_process":
            process_class = _require_process_class("template_process", TemplateProcess)
        case "scalar_gravity":
            process_class = _require_process_class("scalar_gravity", ScalarGravity)
        case "dy":
            process_class = _require_process_class("dy", DY)
        case _:
            raise pygloopException(f"Process {args.process} not implemented.")

    result: dict[str, object] = {
        "command": args.command,
        "process": args.process,
        "exit_code": 0,
    }
    if effective_dy_card is not None:
        result.update({
            "dy_card_path": effective_dy_card.card_path,
            "dy_source_card_sha256": effective_dy_card.source_card_sha256,
            "dy_effective_settings": effective_dy_card.as_dict(),
            "dy_generation_settings_fingerprint": (
                effective_dy_card.generation_fingerprint
            ),
            "dy_configuration_hashes": dict(
                effective_dy_card.configuration_hashes
            ),
        })
        logger.info(
            "Using DY card %s (source SHA256 %s, generation fingerprint %s).",
            effective_dy_card.card_path,
            effective_dy_card.source_card_sha256,
            effective_dy_card.generation_fingerprint,
        )

    if args.overwrite_process_basename is not None:
        process_class.name = args.overwrite_process_basename  # type: ignore

    if args.general_settings is not None:
        target_class = process_class
        applied_settings: dict[str, object] = {}
        for setting in args.general_settings:
            try:
                key, value = setting.split("=", 1)
            except ValueError as e:
                raise pygloopException(
                    f"Could not parse general setting '{setting}'. Expected format: key=value"
                ) from e
            try:
                parsed_value = ast.literal_eval(value)
            except Exception:
                parsed_value = value
            setattr(target_class, key, parsed_value)
            applied_settings[key] = parsed_value
        logger.info(
            f"Applied general settings to process {process_class.name}: {applied_settings}"
        )
        result["general_settings"] = applied_settings

    match args.process:
        case "gghhh":
            process = process_class(  # type: ignore
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
            process = process_class(  # type: ignore
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
            process = process_class(  # type: ignore
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
            process = process_class(
                args.m_top,
                args.m_higgs,
                ps_point,
                args.helicities,
                args.n_loops,
                toml_config_path=args.gammaloop_configuration,
                runtime_toml_config_path=args.runtime_configuration,
                clean=args.clean,
                gammaloop_settings=args.gammaloop_settings,
                final_state=args.dy_final_state,
                process_name=args.dy_process_name,
                diagrams=args.diagrams,
                dy_channel=args.dy_channel,
                skip_ps_validation=args.dy_skip_ps_validation,
                integrate_beams=args.dy_integrate_beams,
                dy_z_bin=args.dy_z_bin,
                dy_q_min=args.dy_q_min,
                dy_q_max=args.dy_q_max,
                dy_physical_normalisation=args.dy_physical_normalisation,
                dy_integrated_leptonic_phase_space=(
                    args.dy_integrated_leptonic_phase_space
                ),
                external_gluon_polarisation=args.external_gluon_polarisation,
                symmetrise_p1_p2=args.dy_symmetrise_p1_p2,
                disable_integrated_uv_cts=not getattr(
                    args, "dy_enable_integrated_uv_cts", False
                ),
                dy_top_self_energy_os_subtraction=getattr(
                    args, "dy_top_self_energy_os_subtraction", False
                )
                if not top_self_energy_mode_explicit
                else None,
                dy_top_self_energy_renormalisation=getattr(
                    args, "dy_top_self_energy_renormalisation", None
                ),
                dy_check_generation_limits=getattr(
                    args, "dy_check_generation_limits", False
                ),
                dy_threshold_h_function=getattr(
                    args, "dy_threshold_h_function", None
                ),
                dy_parallel_graphs=args.dy_parallel_graphs,
                dy_fallback_precision=(
                    args.dy_fallback_precision
                    if args.dy_fallback_precision is not None
                    else args.dy_rotation_check_arb_digits
                ),
                dy_lambda_sq=args.dy_lambda_sq,
                dy_mur_sq=args.dy_mur_sq,
                dy_pdf_set=args.dy_pdf_set,
                dy_pdf_member=args.dy_pdf_member,
                dy_muf_sq=args.dy_muf_sq,
                dy_msbar_scheme_counterterm=args.dy_msbar_scheme_counterterm,
                dy_decoupling=args.dy_decoupling,
                dy_scheme_counterterm_sobol_power=(
                    args.dy_scheme_counterterm_sobol_power
                ),
                dy_scheme_counterterm_replicas=(
                    args.dy_scheme_counterterm_replicas
                ),
                dy_scheme_counterterm_factor=(
                    args.dy_scheme_counterterm_factor
                ),
                dy_scheme_born_bundle=args.dy_scheme_born_bundle,
                dy_scheme_born_bundles=args.dy_scheme_born_bundles,
                dy_scheme_alpha_s=args.dy_scheme_alpha_s,
                dy_scheme_counterterm_clip=args.dy_scheme_counterterm_clip,
                dy_observable_muv=args.dy_muv,
                dy_runtime_parameters=args.dy_runtime_parameters,
                skip_gl_worker_init=(
                    args.command == "integrate"
                    and args.integrand_implementation == "zenos"
                ),
                load_compiled_bundle=args.command != "generate",
            )
        case _:
            raise pygloopException(f"Process {args.process} not implemented.")

    if effective_dy_card is not None and args.command == "integrate":
        _verify_card_bundle_compatibility(
            process,
            effective_dy_card,
            allow_unverified=args.dy_allow_unverified_bundle,
        )
    if (
        args.process == "dy"
        and args.command == "integrate"
        and args.dy_runtime_parameters
    ):
        compiled_bundle = getattr(process, "compiled_bundle", None)
        if compiled_bundle is None:
            raise pygloopException(
                "DY runtime parameter overrides require a compiled zenos bundle."
            )
        compiled_bundle.resolve_runtime_parameters(args.dy_runtime_parameters)

    integrand_implementation = {
        "integrand_type": args.integrand_implementation,
        "evaluator_compiler": args.integrand_evaluator_compiler,
    }
    if args.process == "dy":
        integrand_implementation["dy_rotation_check_digits"] = (
            args.dy_rotation_check_digits
        )
        integrand_implementation["dy_rotation_check_eps"] = args.dy_rotation_check_eps
        integrand_implementation["dy_rotation_check_arb_digits"] = (
            args.dy_rotation_check_arb_digits
        )
        integrand_implementation["dy_fallback_precision"] = (
            args.dy_fallback_precision
            if args.dy_fallback_precision is not None
            else args.dy_rotation_check_arb_digits
        )
        integrand_implementation["dy_theta_tol"] = args.dy_theta_tol
        integrand_implementation["dy_stability_rtol"] = args.dy_stability_rtol
        integrand_implementation["dy_stability_atol"] = args.dy_stability_atol
        integrand_implementation["dy_large_weight_precision"] = (
            args.dy_large_weight_precision
        )
        integrand_implementation["dy_large_weight_clip"] = args.dy_large_weight_clip
        integrand_implementation["dy_beam_parameterisation"] = (
            args.dy_beam_parameterisation
        )
        integrand_implementation["dy_soft_mirror_edges"] = args.dy_soft_mirror_edge
        integrand_implementation["dy_large_weight_threshold"] = (
            args.dy_large_weight_threshold
        )
        integrand_implementation["dy_zero_large_weight_samples"] = (
            args.dy_zero_large_weight_samples
        )
        integrand_implementation["dy_integrated_uv_ct_filter"] = (
            args.dy_integrated_uv_ct_filter
        )
        integrand_implementation["dy_accept_all_arb_retries"] = (
            args.dy_accept_all_arb_retries
        )
        integrand_implementation["dy_runtime_parameters"] = dict(
            args.dy_runtime_parameters
        )
        integrand_implementation["dy_graph_weights"] = getattr(
            args, "dy_graph_weights", None
        )
        if args.dy_ttbar_pt_min is not None:
            integrand_implementation["dy_ttbar_pt_min"] = args.dy_ttbar_pt_min
    matching_generation_provenance_existed = False
    if effective_dy_card is not None and args.command == "generate":
        matching_generation_provenance_existed = _preflight_card_generation(
            process,
            effective_dy_card,
            clean=args.clean,
        )
    t_start = time.time()
    match args.command:
        case "generate":
            logger.info("Generating graphs ...")
            process.generate_graphs()
            if args.process != "scalar_gravity" and ("gammaloop" in args.generation_type or "all" in args.generation_type):
                logger.info("Generating gammaloop code ...")
                process.generate_gammaloop_code()
                logger.info("Gammaloop code generation completed.")
            if args.process != "scalar_gravity" and ("spenso" in args.generation_type or "all" in args.generation_type):
                logger.info("Generating spenso code ...")
                process.generate_spenso_code(
                    integrand_evaluator_compiler=args.integrand_evaluator_compiler,
                    full_spenso_integrand_strategy=args.full_spenso_integrand_strategy,
                    n_hornerscheme_iterations=args.n_iterations_hornerscheme,
                    n_cpe_iterations=args.n_iterations_cpe,
                )
                logger.info("Spenso code generation completed.")
            if effective_dy_card is not None:
                stamped_metadata = _finalize_card_generation_metadata(
                    process,
                    effective_dy_card,
                    matching_provenance_existed=(
                        matching_generation_provenance_existed
                    ),
                )
                result["dy_bundle_metadata"] = stamped_metadata
            result["status"] = "generated"

        case "inspect":
            process.set_log_level(logging.WARNING)
            if args.full_integrand:
                res = process.integrand_xspace(
                    args.point,
                    args.parameterisation,
                    integrand_implementation,
                    args.multi_channeling,
                )
                logger.info(
                    f"Full integrand evaluated at xs = [{Colour.BLUE}{
                        ', '.join(f'{xi:+.16e}' for xi in args.point)
                    }{Colour.END}] : {Colour.GREEN}{res:+.16e}{Colour.END}"
                )
            else:
                if len(args.point) % 3 != 0:
                    raise pygloopException(
                        "Expected a multiple of 3 values for --point."
                    )
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
                res = process.integrand(k_to_inspect, integrand_implementation)
                report = f"Integrand evaluated at loop momentum ks = [{Colour.BLUE}{
                    ','.join(
                        '[' + ', '.join(f'{ki:+.16e}' for ki in k.to_list()) + ']'
                        for k in k_to_inspect
                    )
                }{Colour.END}] : {Colour.GREEN}{res:+.16e}{Colour.END}"
                if args.x_space:
                    report += f" (excl. jacobian = {jacobian:+.16e})"
                logger.info(report)
            process.set_log_level(logging.INFO)
            result["inspect_result"] = res

        case "integrate":
            if args.seed is not None:
                random.seed(args.seed)
                if args.integrator == "naive" and args.n_cores != 1:
                    logger.info(
                        "Note that setting the random seed only ensure reproducible results with the naive integrator and a single core."
                    )

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

                if integrand_implementation["integrand_type"] != "gammaloop":
                    args.target = direct_target

            t_start = time.time()
            run_opts = _runtime_arguments(args)
            run_opts["integrand_implementation"] = integrand_implementation
            res = process.integrate(**run_opts)  # type: ignore
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
            result["integration_result"] = res
            result["integration_time_s"] = integration_time

        case "plot":
            process.plot(**_runtime_arguments(args))
            result["status"] = "plotted"

        case "bench":
            process.set_log_level(logging.CRITICAL)
            try:
                disk_size = process.get_size_on_disk(integrand_implementation)  # type: ignore
            except Exception:
                disk_size = None

            if disk_size is None:
                logger.info(
                    f"{Colour.BLUE}Size on disk [MB]: {Colour.END}{Colour.RED}N/A{Colour.END}"
                )
            else:
                logger.info(
                    f"{Colour.BLUE}Size on disk [MB]: {Colour.END}{Colour.GREEN}{disk_size / 1_000_000.0:.2f}{Colour.END}"
                )

            def f():
                k_to_inspect = [
                    Vector(*[random.random() for _ in range(3)])
                    for _ in range(args.n_loops)
                ]
                process.integrand(k_to_inspect, integrand_implementation)

            res, st = time_function(
                f,
                repeats=args.repeat,
                target_time=args.target_time,
                number=args.n_evals,
                warmup_evals=2,
            )
            # logger.info("Last eval result:", res)
            logger.info(f"{Colour.BLUE}calls / run      : {Colour.END}{st['number']}")
            logger.info(
                f"{Colour.BLUE}median (µs)      : {Colour.END}{Colour.GREEN}{st['median_s'] * 1e6:.1f}{Colour.END}"
            )
            logger.info(
                f"{Colour.BLUE}min    (µs)      : {Colour.END}{Colour.GREEN}{st['min_s'] * 1e6:.1f}{Colour.END}"
            )
            result["bench_result"] = res
            result["bench_stats"] = st
            result["disk_size"] = disk_size
        case _:
            raise pygloopException(f"Command {args.command} not implemented.")

    cmd_runtime = time.time() - t_start
    result["cmd_runtime"] = cmd_runtime
    logger.info(
        f"Command '{args.command}' completed in {Colour.GREEN}{cmd_runtime:.2f}s{Colour.END}."
    )

    return result


if __name__ == "__main__":
    _main_result = main()
    if isinstance(_main_result, dict):
        SystemExit(_main_result.get("exit_code", 0))
    SystemExit(_main_result)
