from __future__ import annotations

import importlib
import math
import multiprocessing
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from processes.dy.dy_analytic import (
    qqbar_log_z_bin_integral,
    weighted_d1_bin_integral,
)
from utils.utils import pygloopException


@dataclass(frozen=True)
class _SobolChunkTask:
    replica: int
    start: int
    stop: int


@dataclass(frozen=True)
class _SobolChunkResult:
    replica: int
    start: int
    count: int
    sums: tuple[float, ...]
    fallback_count: int
    clipped_count: int
    nonfinite_count: int


_SOBOL_CHUNK_SAMPLES: tuple[Any, ...] | None = None
_SOBOL_CHUNK_EVALUATOR: Callable[
    [Any], tuple[tuple[float, ...], int, int, int]
] | None = None


def _evaluate_sobol_chunk(task: _SobolChunkTask) -> _SobolChunkResult:
    if _SOBOL_CHUNK_SAMPLES is None or _SOBOL_CHUNK_EVALUATOR is None:
        raise RuntimeError("DY Sobol auxiliary worker was not initialised.")
    accumulated: list[float] | None = None
    fallback_count = 0
    clipped_count = 0
    nonfinite_count = 0
    samples = _SOBOL_CHUNK_SAMPLES[task.replica]
    for sample in samples[task.start : task.stop]:
        values, fallback, clipped, nonfinite = _SOBOL_CHUNK_EVALUATOR(sample)
        if accumulated is None:
            accumulated = [0.0 for _value in values]
        elif len(values) != len(accumulated):
            raise RuntimeError("DY Sobol auxiliary changed its output width.")
        for index, value in enumerate(values):
            accumulated[index] += float(value)
        fallback_count += int(fallback)
        clipped_count += int(clipped)
        nonfinite_count += int(nonfinite)
    if accumulated is None:
        raise RuntimeError("DY Sobol auxiliary received an empty chunk.")
    return _SobolChunkResult(
        replica=task.replica,
        start=task.start,
        count=task.stop - task.start,
        sums=tuple(accumulated),
        fallback_count=fallback_count,
        clipped_count=clipped_count,
        nonfinite_count=nonfinite_count,
    )


def _integrate_sobol_chunks(
    *,
    dimensions: int,
    power: int,
    replicas: int,
    seed: int,
    output_width: int,
    evaluator: Callable[[Any], tuple[tuple[float, ...], int, int, int]],
    workers: int,
) -> tuple[list[tuple[float, ...]], int, int, int]:
    """Evaluate fixed scrambled Sobol replicas in fork-shared chunks."""

    try:
        from scipy.stats import qmc
    except ImportError as exc:
        raise pygloopException(
            "DY scheme-counterterm integration requires scipy.stats.qmc."
        ) from exc

    requested_workers = int(workers)
    if requested_workers < 1:
        raise pygloopException("DY auxiliary workers must be positive.")
    try:
        affinity_workers = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity_workers = os.cpu_count() or 1
    worker_count = min(requested_workers, affinity_workers)
    samples_per_replica = 2**power
    target_chunks = max(1, math.ceil(worker_count / replicas))
    chunks_per_replica = min(
        samples_per_replica,
        1 << (target_chunks - 1).bit_length(),
    )
    samples_per_chunk = samples_per_replica // chunks_per_replica
    tasks = tuple(
        _SobolChunkTask(
            replica=replica,
            start=chunk * samples_per_chunk,
            stop=(chunk + 1) * samples_per_chunk,
        )
        for replica in range(replicas)
        for chunk in range(chunks_per_replica)
    )
    worker_count = min(worker_count, len(tasks))

    samples = tuple(
        qmc.Sobol(
            d=dimensions,
            scramble=True,
            seed=int(seed) + replica,
        ).random_base2(power)
        for replica in range(replicas)
    )
    global _SOBOL_CHUNK_SAMPLES
    global _SOBOL_CHUNK_EVALUATOR
    _SOBOL_CHUNK_SAMPLES = samples
    _SOBOL_CHUNK_EVALUATOR = evaluator
    try:
        if worker_count == 1:
            chunk_results = [_evaluate_sobol_chunk(task) for task in tasks]
        else:
            if "fork" not in multiprocessing.get_all_start_methods():
                raise pygloopException(
                    "Parallel DY auxiliary integration requires multiprocessing fork."
                )
            context = multiprocessing.get_context("fork")
            with context.Pool(processes=worker_count) as pool:
                chunk_results = list(
                    pool.imap_unordered(_evaluate_sobol_chunk, tasks, chunksize=1)
                )
    finally:
        _SOBOL_CHUNK_EVALUATOR = None
        _SOBOL_CHUNK_SAMPLES = None

    ordered = sorted(chunk_results, key=lambda row: (row.replica, row.start))
    replica_sums: list[tuple[float, ...]] = []
    for replica in range(replicas):
        rows = [row for row in ordered if row.replica == replica]
        if sum(row.count for row in rows) != samples_per_replica:
            raise RuntimeError(
                f"DY Sobol replica {replica} returned the wrong sample count."
            )
        replica_sums.append(
            tuple(
                math.fsum(row.sums[index] for row in rows)
                for index in range(output_width)
            )
        )
    return (
        replica_sums,
        sum(row.fallback_count for row in ordered),
        sum(row.clipped_count for row in ordered),
        sum(row.nonfinite_count for row in ordered),
    )


DY_CHANNEL_PDG_IDS: dict[tuple[int, int], tuple[int, int]] = {
    (0, 0): (21, 21),
    (0, 1): (21, 1),
    (0, -1): (21, -1),
    (1, 0): (1, 21),
    (-1, 0): (-1, 21),
    (1, -1): (1, -1),
    (-1, 1): (-1, 1),
}

DY_PDF_LUMINOSITY_FAMILIES: dict[
    str, tuple[tuple[int, int], ...]
] = {
    "qqbar": ((1, -1), (-1, 1)),
    "qg": ((1, 0), (0, 1), (-1, 0), (0, -1)),
    "gg": ((0, 0),),
}

DY_INTEGRATED_LEPTONIC_PHASE_SPACE_FACTOR: float = 1.0 / (
    24.0 * math.pi**2
)


@dataclass(frozen=True)
class DYSchemeCountertermResult:
    central_value: float
    error: float
    n_samples: int
    elapsed_time: float
    replica_values: tuple[float, ...]
    fallback_count: int = 0
    clipped_count: int = 0
    nonfinite_count: int = 0
    components: tuple[DYSchemeCountertermComponentResult, ...] = ()
    method: str | None = None
    integration_dimension: int | None = None


@dataclass(frozen=True)
class DYSchemeCountertermComponentResult:
    label: str
    central_value: float
    error: float
    replica_values: tuple[float, ...]


@dataclass(frozen=True)
class DYQQbarSchemeContinuousComponents:
    """Unnormalised numerators of the established continuous ``D_qq`` kernel."""

    regular: float
    log_x_plus: float
    log_one_minus_x_plus: float


@dataclass(frozen=True)
class DYQQbarAuxiliaryResult:
    """Correlated ``D_qq`` and Born estimates for ordered two-loop qqbar."""

    dqq_central_value: float
    dqq_error: float
    born_central_value: float
    born_error: float
    combined_central_value: float
    combined_error: float
    n_samples: int
    elapsed_time: float
    dqq_replica_values: tuple[float, ...]
    born_replica_values: tuple[float, ...]
    combined_replica_values: tuple[float, ...]
    components: tuple[DYSchemeCountertermComponentResult, ...]
    fallback_count: int = 0
    clipped_count: int = 0
    nonfinite_count: int = 0


@dataclass(frozen=True)
class DYGGEndpointCoefficients:
    """Endpoint coefficients for the corrected physical gg conversion."""

    dgg_delta: float
    minus_lsz: float
    top_lsz_total: float


@dataclass(frozen=True)
class DYGGAuxiliaryResult:
    """Correlated physical ``D_gg-LSZ`` and massive-top LSZ estimates."""

    dgg_minus_lsz_central_value: float
    dgg_minus_lsz_error: float
    top_lsz_central_value: float
    top_lsz_error: float
    born_central_value: float
    born_error: float
    combined_central_value: float
    combined_error: float
    n_samples: int
    elapsed_time: float
    dgg_minus_lsz_replica_values: tuple[float, ...]
    top_lsz_replica_values: tuple[float, ...]
    born_replica_values: tuple[float, ...]
    combined_replica_values: tuple[float, ...]
    components: tuple[DYSchemeCountertermComponentResult, ...]
    fallback_count: int = 0
    clipped_count: int = 0
    nonfinite_count: int = 0


@dataclass(frozen=True)
class DYRegularSchemeConvolution:
    """One regular finite kernel convolved with a DY-generated Born bundle."""

    label: str
    born_channel: tuple[int, int]
    physical_normalisation: float
    kernel: Callable[[float], float]
    born_integrand: Callable[
        [float, tuple[float, float, float], bool],
        float,
    ]


def finite_gq_scheme_kernel(xi: float) -> float:
    """Finite g->q kernel with alpha_s/(2*pi) stripped.

    This is the finite part of

        T_F / eps * (xi^2 + (1-xi)^2 - eps)
        * (xi*(1-xi))^(-eps) / (2-2*eps)

    with T_F=1/2.
    """

    value = float(xi)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise pygloopException(
            f"The g->q scheme variable xi must be finite and in (0, 1), got {value}."
        )
    splitting = value**2 + (1.0 - value) ** 2
    return 0.25 * (
        splitting * (1.0 - math.log(value * (1.0 - value))) - 1.0
    )


QG_SCHEME_COUNTERTERM_FACTOR: float = 2.0 * math.pi


def finite_g_to_q_scheme_kernel(
    xi: float,
    lambda_sq: float,
    mu_sq: float,
    alpha_s: float,
) -> float:
    """Finite ``g -> q`` kernel in the two-loop Born-convolution convention.

    This is the finite part of the dimensionally regulated expression supplied
    for the qg ttbar scheme change, including ``T_F=1/2``.  The common
    ``2*pi`` factor applied after integration converts this to the standard
    ``alpha_s/(2*pi) * T_F`` normalisation.
    """

    value = float(xi)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise pygloopException(
            f"The g->q scheme variable xi must be finite and in (0, 1), got {value}."
        )
    lambda_value = _positive_finite_scale(lambda_sq, "Lambdasq")
    mu_value = _positive_finite_scale(mu_sq, "scheme musq")
    alpha_value = _positive_finite_scale(alpha_s, "scheme alpha_s")
    splitting = value**2 + (1.0 - value) ** 2
    logarithm = math.log(
        lambda_value * value * (1.0 - value) / mu_value
    )
    return alpha_value / (8.0 * math.pi**2) * (
        1.0 - splitting + splitting * logarithm
    )


def finite_q_to_g_scheme_kernel(
    xi: float,
    lambda_sq: float,
    mu_sq: float,
    alpha_s: float,
) -> float:
    """Finite ``q -> g`` kernel in the two-loop Born-convolution convention.

    The expression includes ``C_F=4/3`` and leaves the common ``2*pi`` factor
    to be applied after integration, giving ``alpha_s/(2*pi) * C_F``.
    """

    value = float(xi)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise pygloopException(
            f"The q->g scheme variable xi must be finite and in (0, 1), got {value}."
        )
    lambda_value = _positive_finite_scale(lambda_sq, "Lambdasq")
    mu_value = _positive_finite_scale(mu_sq, "scheme musq")
    alpha_value = _positive_finite_scale(alpha_s, "scheme alpha_s")
    splitting = (1.0 + (1.0 - value) ** 2) / value
    logarithm = math.log(
        lambda_value * value * (1.0 - value) / mu_value
    )
    return alpha_value / (3.0 * math.pi**2) * (
        value + splitting * logarithm
    )


QQBAR_SCHEME_DELTA_PI_SQUARED_COEFFICIENT: float = (
    4.0 * math.pi**2 / (18.0 * math.pi)
)
QQBAR_SCHEME_DELTA_RATIONAL_COEFFICIENT: float = -48.0 / (
    18.0 * math.pi
)
QQBAR_SCHEME_DELTA_COEFFICIENT: float = (
    QQBAR_SCHEME_DELTA_PI_SQUARED_COEFFICIENT
    + QQBAR_SCHEME_DELTA_RATIONAL_COEFFICIENT
)
# ``integrate_qqbar_scheme_counterterm`` explicitly sums the two incoming-leg
# actions of the supplied D_qq kernel, which already contains C_F/(2*pi).
# For the z-weighted DY observable, the Born coefficient after stripping
# g_s^2 = 4*pi*alpha_s is 1/(144*pi^2*s_hat).  The common forward-measure
# normalisation used below is 1/(54*pi^3*s_hat), so its exact conversion is
# 3*pi/8.  The minus sign is the source-to-MSbar direction in the forward-cut
# convention.
QQBAR_SCHEME_COUNTERTERM_FACTOR: float = -3.0 * math.pi / 8.0

GG_SCHEME_CA: float = 3.0
GG_SCHEME_TF: float = 0.5
GG_SCHEME_ACTIVE_FLAVOURS: int = 2


def finite_qqbar_scheme_continuous_components(
    xi: float,
    born_at_xi: float,
    born_at_endpoint: float,
) -> DYQQbarSchemeContinuousComponents:
    """Return the three numerators used by the production ``D_qq`` kernel.

    This primitive deliberately preserves the endpoint prescription and the
    arithmetic of :func:`finite_qqbar_scheme_continuous_weight`.  Its fields
    still need the common ``1/(18*pi)`` normalisation.
    """

    value = float(xi)
    born = float(born_at_xi)
    endpoint = float(born_at_endpoint)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise pygloopException(
            f"The q->q scheme variable xi must be finite and in (0, 1), got {value}."
        )
    if not math.isfinite(born) or not math.isfinite(endpoint):
        raise pygloopException("The q->q Born test function must be finite.")

    one_minus_xi = 1.0 - value
    regular = 12.0 * one_minus_xi * born
    log_x_plus = (
        12.0
        * (1.0 + value**2)
        * math.log(value)
        * born
        / one_minus_xi
    )
    if born == endpoint:
        endpoint_subtracted = (value - 1.0) * (value + 1.0) * born
    else:
        endpoint_subtracted = (1.0 + value**2) * born - 2.0 * endpoint
    log_one_minus_x_plus = (
        12.0
        * math.log1p(-value)
        * endpoint_subtracted
        / one_minus_xi
    )
    return DYQQbarSchemeContinuousComponents(
        regular=regular,
        log_x_plus=log_x_plus,
        log_one_minus_x_plus=log_one_minus_x_plus,
    )


def finite_qqbar_scheme_continuous_weight(
    xi: float,
    born_at_xi: float,
    born_at_endpoint: float,
) -> float:
    """Return the regular and plus-distribution parts of one D_qq convolution.

    The supplied finite kernel is evaluated at Lambdasq=1.  The plus
    distributions act on the complete Born test function.  In particular,

      A(x) [g(x)]_+ B(x) = g(x) (A(x) B(x) - A(1) B(1)).

    The delta(1-x) contribution is exposed separately through
    ``QQBAR_SCHEME_DELTA_COEFFICIENT``.
    """

    components = finite_qqbar_scheme_continuous_components(
        xi,
        born_at_xi,
        born_at_endpoint,
    )
    return (
        components.regular
        + components.log_x_plus
        + components.log_one_minus_x_plus
    ) / (18.0 * math.pi)


def finite_qqbar_scheme_weight_components(
    xi: float,
    born_at_xi: float,
    born_at_endpoint: float,
) -> tuple[float, float, float, float, float]:
    """Return separately normalised regular, plus, and endpoint pieces."""

    continuous = finite_qqbar_scheme_continuous_components(
        xi,
        born_at_xi,
        born_at_endpoint,
    )
    normalisation = 18.0 * math.pi
    return (
        continuous.regular / normalisation,
        continuous.log_x_plus / normalisation,
        continuous.log_one_minus_x_plus / normalisation,
        QQBAR_SCHEME_DELTA_PI_SQUARED_COEFFICIENT * born_at_endpoint,
        QQBAR_SCHEME_DELTA_RATIONAL_COEFFICIENT * born_at_endpoint,
    )


DY_QQBAR_AUXILIARY_METHOD = "analytic_xi_sobol_2d_threshold_square_v1"


def finite_qqbar_scheme_bin_components(
    xi_min: float,
    xi_max: float,
    *,
    endpoint_width: float | None = None,
) -> tuple[float, float, float, float, float]:
    """Integrate one D_qq kernel against a constant Born factor on a bin.

    The distributional endpoint and the subtraction below xi_min are included.
    A separately computed endpoint width preserves very narrow threshold bins
    even when subtracting that width from one would round back to one.
    """
    low, high = float(xi_min), float(xi_max)
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high <= 1):
        raise pygloopException("The integrated q->q kernel requires 0 <= low <= high <= 1.")
    if high == 1.0:
        width = 1.0 - low if endpoint_width is None else float(endpoint_width)
        if not math.isfinite(width) or not 0.0 < width <= 1.0:
            raise pygloopException("The integrated endpoint kernel requires a positive width.")
        regular = 0.5 * width**2
        # Integral of (1+xi^2)*log(xi)/(1-xi), expanded at xi=1.
        log_x = (
            width * (-2.0 + width * (0.5 + width * (-2.0/9.0 + width * (-1.0/12.0 - 7.0*width/150.0))))
            if width < 1.0e-4 else qqbar_log_z_bin_integral(low, high)
        )
        logarithm = math.log(width)
        log_one_minus = (
            logarithm**2 - (2.0*width - 0.5*width**2)*logarithm
            + 2.0*width - 0.25*width**2
        )
    elif low == high:
        return (0.0,) * 5
    else:
        regular = (high-low) * (1.0 - 0.5*(high+low))
        log_x = qqbar_log_z_bin_integral(low, high)
        log_one_minus = weighted_d1_bin_integral(low, high)
    factor = 12.0 / (18.0 * math.pi)
    return (
        factor * regular,
        factor * log_x,
        factor * log_one_minus,
        QQBAR_SCHEME_DELTA_PI_SQUARED_COEFFICIENT if high == 1.0 else 0.0,
        QQBAR_SCHEME_DELTA_RATIONAL_COEFFICIENT if high == 1.0 else 0.0,
    )


def finite_qqbar_scheme_lower_limit_remainder(
    xi_min: float,
    born_at_endpoint: float,
) -> float:
    """Return the one-leg plus-distribution remainder below ``xi_min``."""

    lower = float(xi_min)
    endpoint = float(born_at_endpoint)
    if not math.isfinite(lower) or not 0.0 <= lower < 1.0:
        raise pygloopException(
            "The q->q scheme lower limit must be finite and in [0, 1)."
        )
    if not math.isfinite(endpoint):
        raise pygloopException("The q->q Born endpoint must be finite.")
    if lower == 0.0 or endpoint == 0.0:
        return 0.0
    return (
        12.0
        * endpoint
        * math.log1p(-lower) ** 2
        / (18.0 * math.pi)
    )


def pdf_partons_for_channel(channel: tuple[int, int]) -> tuple[int, int]:
    try:
        return DY_CHANNEL_PDG_IDS[tuple(channel)]
    except KeyError as exc:
        raise pygloopException(
            f"No PDF parton mapping is defined for DY channel {tuple(channel)}."
        ) from exc


def initial_state_average_factor(channel: tuple[int, int]) -> float:
    """Return the spin-colour average for the two selected beam partons."""

    factor = 1.0
    for pdg_id in pdf_partons_for_channel(channel):
        if pdg_id == 21:
            colour_average = 1.0 / 8.0
        elif 1 <= abs(pdg_id) <= 6:
            colour_average = 1.0 / 3.0
        else:
            raise pygloopException(
                f"No initial-state average is defined for PDG ID {pdg_id}."
            )
        factor *= 0.5 * colour_average
    return factor


def physical_beam_normalisation_factor(
    channel: tuple[int, int],
    n_loops: int,
) -> float:
    """Return spin/colour averages times the loop-measure normalisation."""

    loop_count = int(n_loops)
    if loop_count < 1:
        raise pygloopException("Physical beam normalisation requires L >= 1.")
    return initial_state_average_factor(channel) / (2.0 * math.pi) ** (
        3 * loop_count - 1
    )


def _positive_finite_scale(value: float, option_name: str) -> float:
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise pygloopException(f"{option_name} must be finite and strictly positive.")
    return scale


def gg_scheme_endpoint_coefficients(
    lambda_sq: float,
    mur_sq: float,
    alpha_s: float,
    m_top: float,
    *,
    active_flavours: int = GG_SCHEME_ACTIVE_FLAVOURS,
) -> DYGGEndpointCoefficients:
    """Return the endpoint coefficients of the corrected physical gg kernel.

    ``dgg_delta`` is the finite Altarelli--Parisi delta coefficient after the
    missing one-half in the cut normalisation is restored.  ``minus_lsz`` uses
    the full ``(2*pi)^(d-1)`` LSZ measure.  Consequently their flavour terms
    cancel at ``L=log(LambdaSq/murSq)=0``.  ``top_lsz_total`` is the massive-top
    external-gluon LSZ correction after summing both gluon legs and interfering
    with the Born amplitude.  It is therefore a cross-section coefficient and
    must multiply the endpoint Born only once.
    """

    lambda_value = _positive_finite_scale(lambda_sq, "Lambdasq")
    mur_value = _positive_finite_scale(mur_sq, "mursq")
    alpha_value = _positive_finite_scale(alpha_s, "scheme alpha_s")
    top_mass = _positive_finite_scale(m_top, "top mass")
    flavour_count = int(active_flavours)
    if flavour_count < 0 or flavour_count != active_flavours:
        raise pygloopException(
            "The gg scheme active-flavour count must be a non-negative integer."
        )

    logarithm = math.log(lambda_value / mur_value)
    ca = GG_SCHEME_CA
    tf_nf = GG_SCHEME_TF * flavour_count
    dgg_delta = alpha_value / (36.0 * math.pi) * (
        ca * (-67.0 + 6.0 * math.pi**2)
        + 20.0 * tf_nf
        + 3.0 * (11.0 * ca - 4.0 * tf_nf) * logarithm
    )
    transverse_0 = (5.0 * ca - 4.0 * tf_nf) / 3.0
    transverse_1 = (ca + 4.0 * tf_nf) / 9.0
    minus_lsz = alpha_value / (4.0 * math.pi) * (
        transverse_1 + transverse_0 * (2.0 - logarithm)
    )
    top_lsz_total = (
        alpha_value
        / (2.0 * math.pi)
        * (-4.0 * GG_SCHEME_TF / 3.0)
        * math.log(mur_value / top_mass**2)
    )
    return DYGGEndpointCoefficients(
        dgg_delta=dgg_delta,
        minus_lsz=minus_lsz,
        top_lsz_total=top_lsz_total,
    )


def finite_gg_scheme_continuous_components(
    xi: float,
    born_at_xi: float,
    born_at_endpoint: float,
    lambda_sq: float,
    mur_sq: float,
    alpha_s: float,
) -> tuple[float, float, float]:
    """Return one-leg regular, D0 and D1 pieces of physical ``D_gg``."""

    value = float(xi)
    born = float(born_at_xi)
    endpoint = float(born_at_endpoint)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise pygloopException(
            f"The g->g scheme variable xi must be finite and in (0, 1), got {value}."
        )
    if not math.isfinite(born) or not math.isfinite(endpoint):
        raise pygloopException("The g->g Born test function must be finite.")
    lambda_value = _positive_finite_scale(lambda_sq, "Lambdasq")
    mur_value = _positive_finite_scale(mur_sq, "mursq")
    alpha_value = _positive_finite_scale(alpha_s, "scheme alpha_s")

    scale_ratio = lambda_value / mur_value
    scale_logarithm = math.log(scale_ratio)
    one_minus_xi = 1.0 - value
    normalisation = alpha_value * GG_SCHEME_CA / math.pi
    regular_kernel = one_minus_xi / value + value * one_minus_xi
    regular = (
        normalisation
        * regular_kernel
        * math.log(scale_ratio * value * one_minus_xi)
        * born
    )
    d0 = normalisation * (
        value * math.log(scale_ratio * value) * born
        - scale_logarithm * endpoint
    ) / one_minus_xi
    d1 = (
        normalisation
        * math.log1p(-value)
        * (value * born - endpoint)
        / one_minus_xi
    )
    return regular, d0, d1


def finite_gg_scheme_lower_limit_remainder(
    xi_min: float,
    born_at_endpoint: float,
    lambda_sq: float,
    mur_sq: float,
    alpha_s: float,
) -> float:
    """Return the one-leg D0/D1 plus-distribution remainder below ``xi_min``."""

    lower = float(xi_min)
    endpoint = float(born_at_endpoint)
    if not math.isfinite(lower) or not 0.0 <= lower < 1.0:
        raise pygloopException(
            "The g->g scheme lower limit must be finite and in [0, 1)."
        )
    if not math.isfinite(endpoint):
        raise pygloopException("The g->g Born endpoint must be finite.")
    lambda_value = _positive_finite_scale(lambda_sq, "Lambdasq")
    mur_value = _positive_finite_scale(mur_sq, "mursq")
    alpha_value = _positive_finite_scale(alpha_s, "scheme alpha_s")
    if lower == 0.0 or endpoint == 0.0:
        return 0.0
    endpoint_logarithm = math.log1p(-lower)
    return (
        alpha_value
        * GG_SCHEME_CA
        / math.pi
        * endpoint
        * (
            math.log(lambda_value / mur_value) * endpoint_logarithm
            + 0.5 * endpoint_logarithm**2
        )
    )


def resolve_factorisation_scale_sq(
    explicit_muf_sq: float | None,
    lambda_sq: float | None,
    mur_sq: float | None,
) -> float:
    if (lambda_sq is None) != (mur_sq is None):
        raise pygloopException(
            "PDF weighting requires both Lambdasq and mursq when either is set."
        )

    common_scale: float | None = None
    if lambda_sq is not None and mur_sq is not None:
        lambda_value = _positive_finite_scale(lambda_sq, "Lambdasq")
        mur_value = _positive_finite_scale(mur_sq, "mursq")
        if lambda_value != mur_value:
            raise pygloopException(
                "PDF weighting requires Lambdasq == mursq; "
                f"got {lambda_value} != {mur_value}."
            )
        common_scale = lambda_value

    if explicit_muf_sq is not None:
        return _positive_finite_scale(explicit_muf_sq, "mufsq")
    if common_scale is None:
        raise pygloopException(
            "PDF weighting requires --dy-muf-sq or equal explicit values for "
            "--dy-lambda-sq and --dy-mur-sq."
        )
    return common_scale


class DYPDFProvider:
    """Lazy LHAPDF adapter for one selected set member."""

    def __init__(
        self,
        set_name: str,
        member: int = 0,
        luminosity_family: str | None = None,
    ):
        self.set_name = str(set_name).strip()
        if not self.set_name:
            raise pygloopException("The DY PDF set name cannot be empty.")
        self.member = int(member)
        if self.member < 0:
            raise pygloopException("The DY PDF member must be non-negative.")
        self.luminosity_family = self.normalise_luminosity_family(
            luminosity_family
        )
        self._pdf: Any | None = None

    @staticmethod
    def normalise_luminosity_family(
        luminosity_family: str | None,
    ) -> str | None:
        if luminosity_family is None:
            return None
        family = str(luminosity_family).strip().lower()
        if family not in DY_PDF_LUMINOSITY_FAMILIES:
            known = ", ".join(sorted(DY_PDF_LUMINOSITY_FAMILIES))
            raise pygloopException(
                f"Unknown DY PDF luminosity family {family!r}; expected {known}."
            )
        return family

    def _load_pdf(self) -> Any:
        if self._pdf is not None:
            return self._pdf
        try:
            lhapdf = importlib.import_module("lhapdf")
        except ImportError as exc:
            raise pygloopException(
                "DY PDF weighting requires the LHAPDF Python bindings for "
                f"integrate.beams.pdf.set={self.set_name!r}, "
                f"integrate.beams.pdf.member={self.member}."
            ) from exc
        try:
            self._pdf = lhapdf.mkPDF(self.set_name, self.member)
        except Exception as exc:
            raise pygloopException(
                "Could not load "
                f"integrate.beams.pdf.set={self.set_name!r}, "
                f"integrate.beams.pdf.member={self.member}: {exc}"
            ) from exc
        return self._pdf

    @staticmethod
    def _validate_x(x: float, beam_label: str) -> float:
        value = float(x)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise pygloopException(
                f"{beam_label} must be finite and in [0, 1], got {value}."
            )
        return value

    def density(
        self,
        pdg_id: int,
        x: float,
        muf_sq: float,
        beam_label: str,
    ) -> float:
        x_value = self._validate_x(x, beam_label)
        if x_value == 0.0:
            return 0.0
        try:
            x_times_density = float(
                self._load_pdf().xfxQ2(int(pdg_id), x_value, float(muf_sq))
            )
        except Exception as exc:
            if isinstance(exc, pygloopException):
                raise
            raise pygloopException(
                f"LHAPDF evaluation failed for parton {pdg_id}, {beam_label}="
                f"{x_value}, mufsq={muf_sq}: {exc}"
            ) from exc
        density = x_times_density / x_value
        if not math.isfinite(density):
            raise pygloopException(
                f"LHAPDF returned a non-finite density for parton {pdg_id}, "
                f"{beam_label}={x_value}, mufsq={muf_sq}."
            )
        return density

    def luminosity(
        self,
        channel: tuple[int, int],
        x1: float,
        x2: float,
        muf_sq: float,
    ) -> float:
        channels = (
            DY_PDF_LUMINOSITY_FAMILIES[self.luminosity_family]
            if self.luminosity_family is not None
            else (tuple(channel),)
        )
        luminosity = math.fsum(
            self.density(parton1, x1, muf_sq, "x1")
            * self.density(parton2, x2, muf_sq, "x2")
            for parton1, parton2 in (
                pdf_partons_for_channel(selected_channel)
                for selected_channel in channels
            )
        )
        if not math.isfinite(luminosity):
            raise pygloopException(
                "Non-finite PDF luminosity for DY channel "
                f"{channel} (family={self.luminosity_family!r})."
            )
        return luminosity


def integrate_gq_scheme_counterterm(
    provider: DYPDFProvider,
    channel: tuple[int, int],
    muf_sq: float,
    e_cm_sq: float,
    z_bin: tuple[float, float] | None,
    q_min: float,
    q_max: float | None,
    physical_normalisation: float,
    sobol_power: int,
    replicas: int,
    seed: int,
) -> DYSchemeCountertermResult:
    """Integrate the finite gq scheme term with replicated scrambled Sobol rules."""

    if tuple(channel) not in {(1, 0), (0, 1)}:
        raise pygloopException(
            "The finite gq scheme counterterm only supports DY channels "
            "(1,0) and (0,1)."
        )
    centre_of_mass_sq = _positive_finite_scale(e_cm_sq, "e_cm^2")
    minimum_q = _positive_finite_scale(q_min, "DY scheme-counterterm QMIN")
    maximum_q = (
        _positive_finite_scale(q_max, "DY scheme-counterterm QMAX")
        if q_max is not None
        else None
    )
    if maximum_q is not None and minimum_q >= maximum_q:
        raise pygloopException(
            "DY scheme-counterterm virtuality bounds require QMIN < QMAX."
        )
    power = int(sobol_power)
    replica_count = int(replicas)
    if not 1 <= power <= 30:
        raise pygloopException(
            "DY scheme-counterterm Sobol power must be between 1 and 30."
        )
    if replica_count < 2:
        raise pygloopException(
            "DY scheme-counterterm integration requires at least two replicas."
        )

    requested_z_min, requested_z_max = z_bin if z_bin is not None else (0.0, 1.0)
    physical_z_min = max(
        float(requested_z_min), minimum_q**2 / centre_of_mass_sq
    )
    physical_z_max = float(requested_z_max)
    sample_count = replica_count * 2**power
    if physical_z_min >= physical_z_max:
        return DYSchemeCountertermResult(
            central_value=0.0,
            error=0.0,
            n_samples=sample_count,
            elapsed_time=0.0,
            replica_values=tuple(0.0 for _ in range(replica_count)),
        )

    try:
        from scipy.stats import qmc
    except ImportError as exc:
        raise pygloopException(
            "DY scheme-counterterm integration requires scipy.stats.qmc."
        ) from exc

    z_width = physical_z_max - physical_z_min
    q_max_sq = maximum_q**2 if maximum_q is not None else None
    replica_values: list[float] = []
    start = time.monotonic()
    for replica in range(replica_count):
        samples = qmc.Sobol(
            d=3,
            scramble=True,
            seed=int(seed) + replica,
        ).random_base2(power)
        accumulated = 0.0
        for u1, u2, uz in samples:
            xi = physical_z_min + z_width * float(uz)
            tau_min = minimum_q**2 / (xi * centre_of_mass_sq)
            if tau_min >= 1.0:
                continue

            log_tau_min = math.log(tau_min)
            x1 = math.exp((1.0 - float(u1)) * log_tau_min)
            # log(tau_min/x1) = u1*log(tau_min), exactly zero at u1=0.
            log_x2_min = float(u1) * log_tau_min
            x2 = math.exp((1.0 - float(u2)) * log_x2_min)
            q_sq = xi * x1 * x2 * centre_of_mass_sq
            if q_max_sq is not None and q_sq > q_max_sq:
                continue

            jacobian_over_x1_x2 = (
                z_width * (-log_tau_min) * (-log_x2_min)
            )
            accumulated += (
                jacobian_over_x1_x2
                * provider.luminosity(channel, x1, x2, muf_sq)
                * finite_gq_scheme_kernel(xi)
            )
        replica_values.append(
            physical_normalisation * accumulated / float(len(samples))
        )

    central_value = math.fsum(replica_values) / replica_count
    variance = math.fsum(
        (value - central_value) ** 2 for value in replica_values
    ) / (replica_count - 1)
    return DYSchemeCountertermResult(
        central_value=central_value,
        error=math.sqrt(variance / replica_count),
        n_samples=sample_count,
        elapsed_time=time.monotonic() - start,
        replica_values=tuple(replica_values),
    )


def integrate_qqbar_scheme_counterterm(
    provider: DYPDFProvider,
    channel: tuple[int, int],
    muf_sq: float,
    e_cm_sq: float,
    z_bin: tuple[float, float] | None,
    q_min: float,
    q_max: float | None,
    physical_normalisation: float,
    sobol_power: int,
    replicas: int,
    seed: int,
) -> DYSchemeCountertermResult:
    """Integrate D_qq analytically in xi, with a two-dimensional PDF integral.

    For this z-weighted DY Born observable at fixed mu_F, the luminosity is
    independent of xi. The Q and z cuts become limits of the exact kernel
    integral. Both incoming legs and the original normalisation are retained.
    """

    if tuple(channel) not in {(1, -1), (-1, 1)}:
        raise pygloopException(
            "The finite qqbar scheme counterterm only supports DY channels "
            "(1,-1) and (-1,1)."
        )
    centre_of_mass_sq = _positive_finite_scale(e_cm_sq, "e_cm^2")
    minimum_q = _positive_finite_scale(q_min, "DY scheme-counterterm QMIN")
    maximum_q = (
        _positive_finite_scale(q_max, "DY scheme-counterterm QMAX")
        if q_max is not None
        else None
    )
    if maximum_q is not None and minimum_q >= maximum_q:
        raise pygloopException(
            "DY scheme-counterterm virtuality bounds require QMIN < QMAX."
        )

    power = int(sobol_power)
    replica_count = int(replicas)
    if not 1 <= power <= 30:
        raise pygloopException(
            "DY scheme-counterterm Sobol power must be between 1 and 30."
        )
    if replica_count < 2:
        raise pygloopException(
            "DY scheme-counterterm integration requires at least two replicas."
        )

    requested_z_min, requested_z_max = z_bin if z_bin is not None else (0.0, 1.0)
    requested_z_min = float(requested_z_min)
    requested_z_max = float(requested_z_max)
    if not (
        math.isfinite(requested_z_min)
        and math.isfinite(requested_z_max)
        and 0.0 <= requested_z_min < requested_z_max <= 1.0
    ):
        raise pygloopException(
            "DY scheme-counterterm z bounds require 0 <= ZMIN < ZMAX <= 1."
        )

    component_labels = (
        "regular",
        "log_x_plus",
        "log_one_minus_x_plus",
        "delta_pi_squared",
        "delta_rational",
    )
    sample_count = replica_count * 2**power
    endpoint_tau = minimum_q**2 / (requested_z_max * centre_of_mass_sq)
    if endpoint_tau >= 1.0:
        zeros = tuple(0.0 for _ in range(replica_count))
        return DYSchemeCountertermResult(
            central_value=0.0,
            error=0.0,
            n_samples=0,
            elapsed_time=0.0,
            replica_values=zeros,
            components=tuple(
                DYSchemeCountertermComponentResult(label, 0.0, 0.0, zeros)
                for label in component_labels
            ),
            method=DY_QQBAR_AUXILIARY_METHOD,
            integration_dimension=2,
        )

    try:
        from scipy.stats import qmc
    except ImportError as exc:
        raise pygloopException(
            "DY scheme-counterterm integration requires scipy.stats.qmc."
        ) from exc

    minimum_q_sq = minimum_q**2
    maximum_q_sq = maximum_q**2 if maximum_q is not None else None
    log_endpoint_tau = math.log(endpoint_tau)
    replica_values: list[float] = []
    component_replica_values: dict[str, list[float]] = {
        label: [] for label in component_labels
    }
    start = time.monotonic()
    for replica in range(replica_count):
        samples = qmc.Sobol(
            d=2,
            scramble=True,
            seed=int(seed) + replica,
        ).random_base2(power)
        component_accumulated = [0.0 for _label in component_labels]
        for u1, u2 in samples:
            u1, u2 = float(u1), float(u2)
            # At u2=0 the xi-integrated kernel grows as log(u2)^2. The
            # quadratic threshold map supplies 2*u2, giving a zero limit.
            if u1 == 0.0 or u2 == 0.0:
                continue
            x1 = math.exp((1.0 - float(u1)) * log_endpoint_tau)
            log_x2_min = u1 * log_endpoint_tau
            x2 = math.exp((1.0 - u2*u2) * log_x2_min)
            threshold_log = -log_x2_min * u2*u2
            partonic_scale_sq = (minimum_q_sq / requested_z_max) * math.exp(threshold_log)
            xi_width = -requested_z_max * math.expm1(-threshold_log)
            xi_min = max(requested_z_min, requested_z_max - xi_width)
            xi_max = requested_z_max
            if maximum_q_sq is not None:
                xi_max = min(xi_max, maximum_q_sq / partonic_scale_sq)
            if xi_min >= xi_max and not (
                xi_max == 1.0 and requested_z_max == 1.0 and xi_width > 0.0
                and requested_z_min <= 1.0-xi_width
            ):
                continue
            one_leg_components = finite_qqbar_scheme_bin_components(
                xi_min, xi_max,
                endpoint_width=(min(1.0-requested_z_min, xi_width) if xi_max == 1.0 else None),
            )
            luminosity = provider.luminosity(channel, x1, x2, muf_sq)
            jacobian_over_x1_x2 = log_endpoint_tau * log_x2_min * 2.0*u2
            common_weight = jacobian_over_x1_x2 * 2.0 * luminosity
            if not math.isfinite(common_weight) or not all(map(math.isfinite, one_leg_components)):
                raise pygloopException("Non-finite analytic-xi DY counterterm evaluation.")
            for index, component in enumerate(one_leg_components):
                component_accumulated[index] += common_weight * component

        component_replicas = tuple(
            physical_normalisation * value / float(len(samples))
            for value in component_accumulated
        )
        replica_values.append(math.fsum(component_replicas))
        for label, value in zip(
            component_labels,
            component_replicas,
            strict=True,
        ):
            component_replica_values[label].append(value)

    central_value = math.fsum(replica_values) / replica_count
    variance = math.fsum(
        (value - central_value) ** 2 for value in replica_values
    ) / (replica_count - 1)
    components = []
    for label in component_labels:
        values = component_replica_values[label]
        component_central = math.fsum(values) / replica_count
        component_variance = math.fsum(
            (value - component_central) ** 2 for value in values
        ) / (replica_count - 1)
        components.append(
            DYSchemeCountertermComponentResult(
                label=label,
                central_value=component_central,
                error=math.sqrt(component_variance / replica_count),
                replica_values=tuple(values),
            )
        )
    return DYSchemeCountertermResult(
        central_value=central_value,
        error=math.sqrt(variance / replica_count),
        n_samples=sample_count,
        elapsed_time=time.monotonic() - start,
        replica_values=tuple(replica_values),
        components=tuple(components),
        method=DY_QQBAR_AUXILIARY_METHOD,
        integration_dimension=2,
    )


def integrate_partonic_qqbar_scheme_counterterm(
    channel: tuple[int, int],
    z_bin: tuple[float, float] | None,
    physical_normalisation: float,
    replicas: int,
) -> DYSchemeCountertermResult:
    """Integrate the one-loop DY ``D_qq`` term at fixed partonic energy.

    The campaign observable is z weighted, so the Born test function is
    constant at fixed ``shat``.  This permits an exact distributional
    integration and avoids introducing a fictitious PDF or a Q-min cut in the
    partonic benchmark.
    """

    if tuple(channel) not in {(1, -1), (-1, 1)}:
        raise pygloopException(
            "The partonic finite qqbar scheme counterterm only supports DY "
            "channels (1,-1) and (-1,1)."
        )
    low, high = z_bin if z_bin is not None else (0.0, 1.0)
    low = float(low)
    high = float(high)
    if not (
        math.isfinite(low)
        and math.isfinite(high)
        and 0.0 <= low < high <= 1.0
    ):
        raise pygloopException(
            "Partonic DY scheme-counterterm z bounds require "
            "0 <= ZMIN < ZMAX <= 1."
        )
    normalisation = float(physical_normalisation)
    if not math.isfinite(normalisation):
        raise pygloopException(
            "Partonic DY scheme-counterterm normalisation must be finite."
        )
    replica_count = int(replicas)
    if replica_count < 2:
        raise pygloopException(
            "DY scheme-counterterm integration requires at least two replicas."
        )

    common = 2.0 * normalisation * 12.0 / (18.0 * math.pi)
    def regular_primitive(value: float) -> float:
        return value - 0.5 * value**2
    component_values = (
        common * (regular_primitive(high) - regular_primitive(low)),
        common * qqbar_log_z_bin_integral(low, high),
        common * weighted_d1_bin_integral(low, high),
        (
            2.0
            * normalisation
            * QQBAR_SCHEME_DELTA_PI_SQUARED_COEFFICIENT
            if high == 1.0
            else 0.0
        ),
        (
            2.0 * normalisation * QQBAR_SCHEME_DELTA_RATIONAL_COEFFICIENT
            if high == 1.0
            else 0.0
        ),
    )
    labels = (
        "regular",
        "log_x_plus",
        "log_one_minus_x_plus",
        "delta_pi_squared",
        "delta_rational",
    )
    components = tuple(
        DYSchemeCountertermComponentResult(
            label=label,
            central_value=value,
            error=0.0,
            replica_values=tuple(value for _ in range(replica_count)),
        )
        for label, value in zip(labels, component_values, strict=True)
    )
    total = math.fsum(component_values)
    return DYSchemeCountertermResult(
        central_value=total,
        error=0.0,
        n_samples=0,
        elapsed_time=0.0,
        replica_values=tuple(total for _ in range(replica_count)),
        components=components,
    )


def integrate_regular_born_scheme_counterterm(
    provider: DYPDFProvider | None,
    channel: tuple[int, int],
    muf_sq: float | None,
    e_cm_sq: float,
    threshold_sq: float,
    convolutions: tuple[DYRegularSchemeConvolution, ...],
    sobol_power: int,
    replicas: int,
    seed: int,
    clip_threshold: float | None = None,
    workers: int = 1,
    integrate_beams: bool = True,
) -> DYSchemeCountertermResult:
    """Convolve regular finite kernels with DY-generated Born integrands.

    Each Sobol sample shares the splitting fraction and loop coordinates across
    all requested Born channels.  Hadronic mode additionally samples common
    beam fractions and applies the selected PDF luminosity.  Partonic mode
    evaluates the same coefficient-function convolution at fixed ``e_cm_sq``.
    This preserves useful correlations between the components while the
    returned uncertainty is computed directly from replicas of their sum.
    """

    if not convolutions:
        raise pygloopException(
            "A regular scheme counterterm requires at least one Born convolution."
        )
    centre_of_mass_sq = _positive_finite_scale(e_cm_sq, "e_cm^2")
    physical_threshold_sq = _positive_finite_scale(
        threshold_sq,
        "scheme-counterterm threshold squared",
    )
    beam_convolution = bool(integrate_beams)
    resolved_muf_sq: float | None = None
    if beam_convolution:
        if provider is None or muf_sq is None:
            raise pygloopException(
                "Hadronic regular scheme-counterterm integration requires "
                "PDF setup."
            )
        resolved_muf_sq = _positive_finite_scale(muf_sq, "mufsq")
    elif provider is not None or muf_sq is not None:
        raise pygloopException(
            "Partonic regular scheme-counterterm integration does not accept PDFs."
        )
    power = int(sobol_power)
    replica_count = int(replicas)
    if not 1 <= power <= 30:
        raise pygloopException(
            "DY scheme-counterterm Sobol power must be between 1 and 30."
        )
    if replica_count < 2:
        raise pygloopException(
            "DY scheme-counterterm integration requires at least two replicas."
        )
    if clip_threshold is not None:
        clip_threshold = _positive_finite_scale(
            clip_threshold,
            "DY scheme-counterterm clipping threshold",
        )
    labels = [convolution.label for convolution in convolutions]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise pygloopException(
            "Regular scheme-counterterm component labels must be non-empty and unique."
        )

    sample_count = replica_count * 2**power
    endpoint_tau = physical_threshold_sq / centre_of_mass_sq
    if endpoint_tau >= 1.0:
        empty_components = tuple(
            DYSchemeCountertermComponentResult(
                label=convolution.label,
                central_value=0.0,
                error=0.0,
                replica_values=tuple(0.0 for _ in range(replica_count)),
            )
            for convolution in convolutions
        )
        return DYSchemeCountertermResult(
            central_value=0.0,
            error=0.0,
            n_samples=sample_count,
            elapsed_time=0.0,
            replica_values=tuple(0.0 for _ in range(replica_count)),
            components=empty_components,
        )

    log_endpoint_tau = math.log(endpoint_tau)
    replica_values: list[float] = []
    component_replica_values: dict[str, list[float]] = {
        convolution.label: [] for convolution in convolutions
    }
    start = time.monotonic()
    def evaluate_sample(
        sample: Any,
    ) -> tuple[tuple[float, ...], int, int, int]:
        if beam_convolution:
            u1, u2, uxi, uk1, uk2, uk3 = sample
            x1 = math.exp((1.0 - float(u1)) * log_endpoint_tau)
            log_x2_min = math.log(endpoint_tau / x1)
            x2 = math.exp((1.0 - float(u2)) * log_x2_min)
            beam_jacobian = (
                x1 * x2 * (-log_endpoint_tau) * (-log_x2_min)
            )
            partonic_scale_sq = x1 * x2 * centre_of_mass_sq
            assert provider is not None and resolved_muf_sq is not None
            luminosity = provider.luminosity(
                channel,
                x1,
                x2,
                resolved_muf_sq,
            )
            common_weight = beam_jacobian * luminosity
        else:
            uxi, uk1, uk2, uk3 = sample
            partonic_scale_sq = centre_of_mass_sq
            common_weight = 1.0
        xi_min = physical_threshold_sq / partonic_scale_sq
        xi_width = 1.0 - xi_min
        if xi_width <= 0.0:
            zeros = tuple(0.0 for _ in convolutions)
            return (*zeros, 0.0), 0, 0, 0
        xi = xi_min + xi_width * float(uxi)
        xi = min(
            math.nextafter(1.0, 0.0),
            max(math.nextafter(xi_min, 1.0), xi),
        )
        scaled_partonic_scale_sq = xi * partonic_scale_sq
        loop_coordinates = (float(uk1), float(uk2), float(uk3))
        common_weight *= xi_width

        def sample_components(use_fallback: bool) -> tuple[float, ...]:
            return tuple(
                common_weight
                * convolution.physical_normalisation
                * convolution.kernel(xi)
                * convolution.born_integrand(
                    scaled_partonic_scale_sq,
                    loop_coordinates,
                    use_fallback,
                )
                for convolution in convolutions
            )

        fallback_count = 0
        clipped_count = 0
        nonfinite_count = 0
        try:
            component_weights = sample_components(False)
            weight = math.fsum(component_weights)
        except Exception:
            component_weights = tuple(math.nan for _ in convolutions)
            weight = math.nan
        needs_fallback = not math.isfinite(weight) or any(
            not math.isfinite(value) for value in component_weights
        )
        if clip_threshold is not None and math.isfinite(weight):
            needs_fallback = needs_fallback or abs(weight) > clip_threshold
        if needs_fallback:
            fallback_count = 1
            try:
                component_weights = sample_components(True)
                weight = math.fsum(component_weights)
            except Exception:
                component_weights = tuple(math.nan for _ in convolutions)
                weight = math.nan

        if not math.isfinite(weight) or any(
            not math.isfinite(value) for value in component_weights
        ):
            nonfinite_count = 1
            component_weights = tuple(0.0 for _ in convolutions)
            weight = 0.0
        elif clip_threshold is not None and abs(weight) > clip_threshold:
            clipped_count = 1
            component_weights = tuple(0.0 for _ in convolutions)
            weight = 0.0
        return (
            (*component_weights, weight),
            fallback_count,
            clipped_count,
            nonfinite_count,
        )

    replica_sums, fallback_count, clipped_count, nonfinite_count = (
        _integrate_sobol_chunks(
            dimensions=6 if beam_convolution else 4,
            power=power,
            replicas=replica_count,
            seed=seed,
            output_width=len(convolutions) + 1,
            evaluator=evaluate_sample,
            workers=workers,
        )
    )
    normalisation = float(2**power)
    for sums in replica_sums:
        replica_values.append(sums[-1] / normalisation)
        for convolution, component_sum in zip(
            convolutions,
            sums[:-1],
            strict=True,
        ):
            component_replica_values[convolution.label].append(
                component_sum / normalisation
            )

    if nonfinite_count == sample_count:
        raise pygloopException(
            "Every regular scheme-counterterm sample remained non-finite after "
            "higher-precision fallback."
        )

    def summarise(values: list[float]) -> tuple[float, float]:
        central = math.fsum(values) / replica_count
        variance = math.fsum(
            (value - central) ** 2 for value in values
        ) / (replica_count - 1)
        return central, math.sqrt(variance / replica_count)

    central_value, error = summarise(replica_values)
    components = []
    for convolution in convolutions:
        values = component_replica_values[convolution.label]
        component_central, component_error = summarise(values)
        components.append(
            DYSchemeCountertermComponentResult(
                label=convolution.label,
                central_value=component_central,
                error=component_error,
                replica_values=tuple(values),
            )
        )
    return DYSchemeCountertermResult(
        central_value=central_value,
        error=error,
        n_samples=sample_count,
        elapsed_time=time.monotonic() - start,
        replica_values=tuple(replica_values),
        fallback_count=fallback_count,
        clipped_count=clipped_count,
        nonfinite_count=nonfinite_count,
        components=tuple(components),
    )


def integrate_ttbar_qqbar_scheme_counterterm(
    provider: DYPDFProvider,
    channel: tuple[int, int],
    muf_sq: float,
    e_cm_sq: float,
    m_top: float,
    physical_normalisation: float,
    born_integrand: Callable[[float, tuple[float, float, float], bool], float],
    sobol_power: int,
    replicas: int,
    seed: int,
    clip_threshold: float | None = None,
) -> DYSchemeCountertermResult:
    """Convolve the finite two-leg D_qq kernel with the ttbar Born integrand.

    ``born_integrand`` receives the partonic centre-of-mass energy squared,
    the three shared loop-parameterisation coordinates, and a flag requesting
    the configured higher-precision fallback.  The same coordinates are used
    at xi and at the endpoint so that the plus subtraction is local in every
    Sobol sample.
    """

    if tuple(channel) not in {(1, -1), (-1, 1)}:
        raise pygloopException(
            "The finite ttbar qqbar scheme counterterm only supports channels "
            "(1,-1) and (-1,1)."
        )
    centre_of_mass_sq = _positive_finite_scale(e_cm_sq, "e_cm^2")
    top_mass = _positive_finite_scale(m_top, "top mass")
    threshold_sq = 4.0 * top_mass**2
    power = int(sobol_power)
    replica_count = int(replicas)
    if not 1 <= power <= 30:
        raise pygloopException(
            "DY scheme-counterterm Sobol power must be between 1 and 30."
        )
    if replica_count < 2:
        raise pygloopException(
            "DY scheme-counterterm integration requires at least two replicas."
        )
    if clip_threshold is not None:
        clip_threshold = _positive_finite_scale(
            clip_threshold,
            "DY scheme-counterterm clipping threshold",
        )

    sample_count = replica_count * 2**power
    endpoint_tau = threshold_sq / centre_of_mass_sq
    if endpoint_tau >= 1.0:
        return DYSchemeCountertermResult(
            central_value=0.0,
            error=0.0,
            n_samples=sample_count,
            elapsed_time=0.0,
            replica_values=tuple(0.0 for _ in range(replica_count)),
        )

    try:
        from scipy.stats import qmc
    except ImportError as exc:
        raise pygloopException(
            "DY scheme-counterterm integration requires scipy.stats.qmc."
        ) from exc

    log_endpoint_tau = math.log(endpoint_tau)
    replica_values: list[float] = []
    fallback_count = 0
    clipped_count = 0
    nonfinite_count = 0
    start = time.monotonic()
    for replica in range(replica_count):
        samples = qmc.Sobol(
            d=6,
            scramble=True,
            seed=int(seed) + replica,
        ).random_base2(power)
        accumulated = 0.0
        for u1, u2, uxi, uk1, uk2, uk3 in samples:
            x1 = math.exp((1.0 - float(u1)) * log_endpoint_tau)
            log_x2_min = math.log(endpoint_tau / x1)
            x2 = math.exp((1.0 - float(u2)) * log_x2_min)
            beam_jacobian = (
                x1
                * x2
                * (-log_endpoint_tau)
                * (-log_x2_min)
            )
            xi = min(
                math.nextafter(1.0, 0.0),
                max(math.nextafter(0.0, 1.0), float(uxi)),
            )
            loop_coordinates = (float(uk1), float(uk2), float(uk3))
            partonic_scale_sq = x1 * x2 * centre_of_mass_sq
            scaled_partonic_scale_sq = xi * partonic_scale_sq
            luminosity = provider.luminosity(channel, x1, x2, muf_sq)

            def sample_weight(use_fallback: bool) -> float:
                born_at_endpoint = born_integrand(
                    partonic_scale_sq,
                    loop_coordinates,
                    use_fallback,
                )
                born_at_xi = 0.0
                if scaled_partonic_scale_sq > threshold_sq:
                    born_at_xi = born_integrand(
                        scaled_partonic_scale_sq,
                        loop_coordinates,
                        use_fallback,
                    )
                one_leg_weight = finite_qqbar_scheme_continuous_weight(
                    xi,
                    born_at_xi,
                    born_at_endpoint,
                )
                one_leg_weight += (
                    QQBAR_SCHEME_DELTA_COEFFICIENT * born_at_endpoint
                )
                return (
                    physical_normalisation
                    * beam_jacobian
                    * luminosity
                    * 2.0
                    * one_leg_weight
                )

            try:
                weight = sample_weight(False)
            except Exception:
                weight = math.nan
            needs_fallback = not math.isfinite(weight) or (
                clip_threshold is not None and abs(weight) > clip_threshold
            )
            if needs_fallback:
                fallback_count += 1
                try:
                    weight = sample_weight(True)
                except Exception:
                    weight = math.nan

            if not math.isfinite(weight):
                nonfinite_count += 1
                weight = 0.0
            elif clip_threshold is not None and abs(weight) > clip_threshold:
                clipped_count += 1
                weight = 0.0
            accumulated += weight

        replica_values.append(accumulated / float(len(samples)))

    if nonfinite_count == sample_count:
        raise pygloopException(
            "Every ttbar scheme-counterterm sample remained non-finite after "
            "higher-precision fallback."
        )
    central_value = math.fsum(replica_values) / replica_count
    variance = math.fsum(
        (value - central_value) ** 2 for value in replica_values
    ) / (replica_count - 1)
    return DYSchemeCountertermResult(
        central_value=central_value,
        error=math.sqrt(variance / replica_count),
        n_samples=sample_count,
        elapsed_time=time.monotonic() - start,
        replica_values=tuple(replica_values),
        fallback_count=fallback_count,
        clipped_count=clipped_count,
        nonfinite_count=nonfinite_count,
    )


def integrate_ttbar_qqbar_auxiliary(
    provider: DYPDFProvider | None,
    channel: tuple[int, int],
    muf_sq: float | None,
    e_cm_sq: float,
    m_top: float,
    dqq_normalisation: float,
    born_normalisation: float,
    born_integrand: Callable[[float, tuple[float, float, float], bool], float],
    sobol_power: int,
    replicas: int,
    seed: int,
    dqq_coefficient: float,
    born_coefficient: float,
    integrate_beams: bool,
    clip_threshold: float | None = None,
    workers: int = 1,
) -> DYQQbarAuxiliaryResult:
    """Integrate correlated two-leg ``D_qq`` and physical Born contributions.

    Partonic mode uses four Sobol coordinates (``xi`` and the three Born
    coordinates).  Hadronic mode prepends ``x1`` and ``x2`` and uses the same
    scrambled samples for every distribution component and for the Born term.
    A fallback or clipping decision always applies to the complete paired
    sample so that replica-level covariance is retained.
    """

    if tuple(channel) not in {(1, -1), (-1, 1)}:
        raise pygloopException(
            "The finite ttbar qqbar auxiliary integration only supports channels "
            "(1,-1) and (-1,1)."
        )
    centre_of_mass_sq = _positive_finite_scale(e_cm_sq, "e_cm^2")
    top_mass = _positive_finite_scale(m_top, "top mass")
    threshold_sq = 4.0 * top_mass**2
    dqq_norm = float(dqq_normalisation)
    born_norm = float(born_normalisation)
    dqq_factor = float(dqq_coefficient)
    born_factor = float(born_coefficient)
    if not all(
        math.isfinite(value)
        for value in (dqq_norm, born_norm, dqq_factor, born_factor)
    ):
        raise pygloopException("DY qqbar auxiliary normalisations must be finite.")

    beam_convolution = bool(integrate_beams)
    resolved_muf_sq: float | None = None
    if beam_convolution:
        if provider is None or muf_sq is None:
            raise pygloopException(
                "Hadronic ttbar qqbar auxiliary integration requires PDF setup."
            )
        resolved_muf_sq = _positive_finite_scale(muf_sq, "mufsq")
    elif provider is not None or muf_sq is not None:
        raise pygloopException(
            "Partonic ttbar qqbar auxiliary integration does not accept PDFs."
        )

    power = int(sobol_power)
    replica_count = int(replicas)
    if not 1 <= power <= 30:
        raise pygloopException(
            "DY scheme-counterterm Sobol power must be between 1 and 30."
        )
    if replica_count < 2:
        raise pygloopException(
            "DY scheme-counterterm integration requires at least two replicas."
        )
    if clip_threshold is not None:
        clip_threshold = _positive_finite_scale(
            clip_threshold,
            "DY scheme-counterterm clipping threshold",
        )

    component_labels = (
        "regular",
        "log_x_plus",
        "log_one_minus_x_plus",
        "lower_limit_remainder",
        "delta_pi_squared",
        "delta_rational",
    )
    sample_count = replica_count * 2**power
    endpoint_tau = threshold_sq / centre_of_mass_sq

    def empty_result() -> DYQQbarAuxiliaryResult:
        zeros = tuple(0.0 for _ in range(replica_count))
        return DYQQbarAuxiliaryResult(
            dqq_central_value=0.0,
            dqq_error=0.0,
            born_central_value=0.0,
            born_error=0.0,
            combined_central_value=0.0,
            combined_error=0.0,
            n_samples=sample_count,
            elapsed_time=0.0,
            dqq_replica_values=zeros,
            born_replica_values=zeros,
            combined_replica_values=zeros,
            components=tuple(
                DYSchemeCountertermComponentResult(label, 0.0, 0.0, zeros)
                for label in component_labels
            ),
        )

    if endpoint_tau >= 1.0:
        return empty_result()

    dimensions = 6 if beam_convolution else 4
    log_endpoint_tau = math.log(endpoint_tau)
    dqq_replica_values: list[float] = []
    born_replica_values: list[float] = []
    combined_replica_values: list[float] = []
    component_replica_values: dict[str, list[float]] = {
        label: [] for label in component_labels
    }
    start = time.monotonic()

    def evaluate_sample(
        sample: Any,
    ) -> tuple[tuple[float, ...], int, int, int]:
        if beam_convolution:
            u1, u2, uxi, uk1, uk2, uk3 = sample
            x1 = math.exp((1.0 - float(u1)) * log_endpoint_tau)
            log_x2_min = math.log(endpoint_tau / x1)
            x2 = math.exp((1.0 - float(u2)) * log_x2_min)
            beam_jacobian = (
                x1 * x2 * (-log_endpoint_tau) * (-log_x2_min)
            )
            partonic_scale_sq = x1 * x2 * centre_of_mass_sq
            assert provider is not None and resolved_muf_sq is not None
            luminosity = provider.luminosity(
                channel,
                x1,
                x2,
                resolved_muf_sq,
            )
            common_weight = beam_jacobian * luminosity
        else:
            uxi, uk1, uk2, uk3 = sample
            partonic_scale_sq = centre_of_mass_sq
            common_weight = 1.0

        xi_min = threshold_sq / partonic_scale_sq
        xi_width = 1.0 - xi_min
        if xi_width <= 0.0:
            return tuple(0.0 for _ in range(9)), 0, 0, 0
        xi = xi_min + xi_width * float(uxi)
        xi = min(
            math.nextafter(1.0, 0.0),
            max(math.nextafter(xi_min, 1.0), xi),
        )
        loop_coordinates = (float(uk1), float(uk2), float(uk3))

        def paired_sample(
            use_fallback: bool,
        ) -> tuple[tuple[float, ...], float, float, float]:
            born_at_endpoint = born_integrand(
                partonic_scale_sq,
                loop_coordinates,
                use_fallback,
            )
            born_at_xi = born_integrand(
                xi * partonic_scale_sq,
                loop_coordinates,
                use_fallback,
            )
            numerators = finite_qqbar_scheme_continuous_components(
                xi,
                born_at_xi,
                born_at_endpoint,
            )
            continuous_scale = (
                2.0
                * dqq_norm
                * common_weight
                * xi_width
                / (18.0 * math.pi)
            )
            endpoint_scale = 2.0 * dqq_norm * common_weight
            component_weights = (
                continuous_scale * numerators.regular,
                continuous_scale * numerators.log_x_plus,
                continuous_scale * numerators.log_one_minus_x_plus,
                endpoint_scale
                * finite_qqbar_scheme_lower_limit_remainder(
                    xi_min,
                    born_at_endpoint,
                ),
                endpoint_scale
                * QQBAR_SCHEME_DELTA_PI_SQUARED_COEFFICIENT
                * born_at_endpoint,
                endpoint_scale
                * QQBAR_SCHEME_DELTA_RATIONAL_COEFFICIENT
                * born_at_endpoint,
            )
            dqq_weight = math.fsum(component_weights)
            born_weight = born_norm * common_weight * born_at_endpoint
            combined_weight = (
                dqq_factor * dqq_weight + born_factor * born_weight
            )
            return (
                component_weights,
                dqq_weight,
                born_weight,
                combined_weight,
            )

        fallback_count = 0
        clipped_count = 0
        nonfinite_count = 0
        try:
            component_weights, dqq_weight, born_weight, combined_weight = (
                paired_sample(False)
            )
        except Exception:
            component_weights = tuple(math.nan for _ in component_labels)
            dqq_weight = born_weight = combined_weight = math.nan
        sample_values = (
            *component_weights,
            dqq_weight,
            born_weight,
            combined_weight,
        )
        needs_fallback = any(not math.isfinite(value) for value in sample_values)
        if clip_threshold is not None and not needs_fallback:
            needs_fallback = max(
                abs(dqq_weight),
                abs(born_weight),
                abs(combined_weight),
            ) > clip_threshold
        if needs_fallback:
            fallback_count = 1
            try:
                (
                    component_weights,
                    dqq_weight,
                    born_weight,
                    combined_weight,
                ) = paired_sample(True)
            except Exception:
                component_weights = tuple(math.nan for _ in component_labels)
                dqq_weight = born_weight = combined_weight = math.nan

        sample_values = (
            *component_weights,
            dqq_weight,
            born_weight,
            combined_weight,
        )
        if any(not math.isfinite(value) for value in sample_values):
            nonfinite_count = 1
            component_weights = tuple(0.0 for _ in component_labels)
            dqq_weight = born_weight = combined_weight = 0.0
        elif clip_threshold is not None and max(
            abs(dqq_weight),
            abs(born_weight),
            abs(combined_weight),
        ) > clip_threshold:
            clipped_count = 1
            component_weights = tuple(0.0 for _ in component_labels)
            dqq_weight = born_weight = combined_weight = 0.0
        return (
            (*component_weights, dqq_weight, born_weight, combined_weight),
            fallback_count,
            clipped_count,
            nonfinite_count,
        )

    replica_sums, fallback_count, clipped_count, nonfinite_count = (
        _integrate_sobol_chunks(
            dimensions=dimensions,
            power=power,
            replicas=replica_count,
            seed=seed,
            output_width=9,
            evaluator=evaluate_sample,
            workers=workers,
        )
    )
    sample_normalisation = float(2**power)
    for sums in replica_sums:
        dqq_replica_values.append(sums[6] / sample_normalisation)
        born_replica_values.append(sums[7] / sample_normalisation)
        combined_replica_values.append(sums[8] / sample_normalisation)
        for label, component_sum in zip(
            component_labels,
            sums[:6],
            strict=True,
        ):
            component_replica_values[label].append(
                component_sum / sample_normalisation
            )

    if nonfinite_count == sample_count:
        raise pygloopException(
            "Every ttbar qqbar auxiliary sample remained non-finite after "
            "higher-precision fallback."
        )

    def summarise(values: list[float]) -> tuple[float, float]:
        central = math.fsum(values) / replica_count
        variance = math.fsum(
            (value - central) ** 2 for value in values
        ) / (replica_count - 1)
        return central, math.sqrt(variance / replica_count)

    dqq_central, dqq_error = summarise(dqq_replica_values)
    born_central, born_error = summarise(born_replica_values)
    combined_central, combined_error = summarise(combined_replica_values)
    components = []
    for label in component_labels:
        values = component_replica_values[label]
        central, error = summarise(values)
        components.append(
            DYSchemeCountertermComponentResult(
                label=label,
                central_value=central,
                error=error,
                replica_values=tuple(values),
            )
        )

    return DYQQbarAuxiliaryResult(
        dqq_central_value=dqq_central,
        dqq_error=dqq_error,
        born_central_value=born_central,
        born_error=born_error,
        combined_central_value=combined_central,
        combined_error=combined_error,
        n_samples=sample_count,
        elapsed_time=time.monotonic() - start,
        dqq_replica_values=tuple(dqq_replica_values),
        born_replica_values=tuple(born_replica_values),
        combined_replica_values=tuple(combined_replica_values),
        components=tuple(components),
        fallback_count=fallback_count,
        clipped_count=clipped_count,
        nonfinite_count=nonfinite_count,
    )


def integrate_ttbar_gg_auxiliary(
    channel: tuple[int, int],
    e_cm_sq: float,
    m_top: float,
    physical_normalisation: float,
    born_integrand: Callable[[float, tuple[float, float, float], bool], float],
    lambda_sq: float,
    mur_sq: float,
    alpha_s: float,
    sobol_power: int,
    replicas: int,
    seed: int,
    clip_threshold: float | None = None,
    *,
    active_flavours: int = GG_SCHEME_ACTIVE_FLAVOURS,
    provider: DYPDFProvider | None = None,
    muf_sq: float | None = None,
    integrate_beams: bool = False,
    workers: int = 1,
) -> DYGGAuxiliaryResult:
    """Integrate the physical ``Dgg-LSZ+top-LSZ`` conversion.

    The generated one-loop forward integrand is converted to a physical Born
    cut before the kernel acts on it.  All distribution pieces, the massless
    LSZ subtraction, the single two-beam massive-top LSZ endpoint and the Born
    diagnostic use identical scrambled Sobol samples, preserving their replica
    covariance.  Hadronic mode additionally shares the two beam fractions and
    their PDF luminosity across every correlated component.
    """

    if tuple(channel) != (0, 0):
        raise pygloopException(
            "The finite ttbar gg auxiliary integration requires channel (0,0)."
        )
    centre_of_mass_sq = _positive_finite_scale(e_cm_sq, "e_cm^2")
    top_mass = _positive_finite_scale(m_top, "top mass")
    normalisation = float(physical_normalisation)
    if not math.isfinite(normalisation):
        raise pygloopException("The physical gg Born normalisation must be finite.")
    lambda_value = _positive_finite_scale(lambda_sq, "Lambdasq")
    mur_value = _positive_finite_scale(mur_sq, "mursq")
    alpha_value = _positive_finite_scale(alpha_s, "scheme alpha_s")
    beam_convolution = bool(integrate_beams)
    resolved_muf_sq: float | None = None
    if beam_convolution:
        if provider is None or muf_sq is None:
            raise pygloopException(
                "Hadronic ttbar gg auxiliary integration requires PDF setup."
            )
        resolved_muf_sq = _positive_finite_scale(muf_sq, "mufsq")
    elif provider is not None or muf_sq is not None:
        raise pygloopException(
            "Partonic ttbar gg auxiliary integration does not accept PDFs."
        )
    endpoint_coefficients = gg_scheme_endpoint_coefficients(
        lambda_value,
        mur_value,
        alpha_value,
        top_mass,
        active_flavours=active_flavours,
    )

    power = int(sobol_power)
    replica_count = int(replicas)
    if not 1 <= power <= 30:
        raise pygloopException(
            "DY scheme-counterterm Sobol power must be between 1 and 30."
        )
    if replica_count < 2:
        raise pygloopException(
            "DY scheme-counterterm integration requires at least two replicas."
        )
    if clip_threshold is not None:
        clip_threshold = _positive_finite_scale(
            clip_threshold,
            "DY scheme-counterterm clipping threshold",
        )

    component_labels = (
        "Dgg_regular",
        "Dgg_log_z_plus",
        "Dgg_log_one_minus_z_plus",
        "Dgg_lower_limit_remainder",
        "Dgg_delta",
        "minus_LSZ",
        "top_LSZ",
    )
    sample_count = replica_count * 2**power
    threshold_sq = 4.0 * top_mass**2
    endpoint_tau = threshold_sq / centre_of_mass_sq

    def empty_result() -> DYGGAuxiliaryResult:
        zeros = tuple(0.0 for _ in range(replica_count))
        return DYGGAuxiliaryResult(
            dgg_minus_lsz_central_value=0.0,
            dgg_minus_lsz_error=0.0,
            top_lsz_central_value=0.0,
            top_lsz_error=0.0,
            born_central_value=0.0,
            born_error=0.0,
            combined_central_value=0.0,
            combined_error=0.0,
            n_samples=sample_count,
            elapsed_time=0.0,
            dgg_minus_lsz_replica_values=zeros,
            top_lsz_replica_values=zeros,
            born_replica_values=zeros,
            combined_replica_values=zeros,
            components=tuple(
                DYSchemeCountertermComponentResult(label, 0.0, 0.0, zeros)
                for label in component_labels
            ),
        )

    if endpoint_tau >= 1.0:
        return empty_result()

    dimensions = 6 if beam_convolution else 4
    log_endpoint_tau = math.log(endpoint_tau)
    dgg_minus_lsz_replica_values: list[float] = []
    top_lsz_replica_values: list[float] = []
    born_replica_values: list[float] = []
    combined_replica_values: list[float] = []
    component_replica_values: dict[str, list[float]] = {
        label: [] for label in component_labels
    }
    start = time.monotonic()

    def evaluate_sample(
        sample: Any,
    ) -> tuple[tuple[float, ...], int, int, int]:
        if beam_convolution:
            u1, u2, uxi, uk1, uk2, uk3 = sample
            x1 = math.exp((1.0 - float(u1)) * log_endpoint_tau)
            log_x2_min = math.log(endpoint_tau / x1)
            x2 = math.exp((1.0 - float(u2)) * log_x2_min)
            beam_jacobian = (
                x1 * x2 * (-log_endpoint_tau) * (-log_x2_min)
            )
            partonic_scale_sq = x1 * x2 * centre_of_mass_sq
            assert provider is not None and resolved_muf_sq is not None
            luminosity = provider.luminosity(
                channel,
                x1,
                x2,
                resolved_muf_sq,
            )
            common_weight = beam_jacobian * luminosity
        else:
            uxi, uk1, uk2, uk3 = sample
            partonic_scale_sq = centre_of_mass_sq
            common_weight = 1.0

        xi_min = threshold_sq / partonic_scale_sq
        xi_width = 1.0 - xi_min
        if xi_width <= 0.0:
            return tuple(0.0 for _ in range(11)), 0, 0, 0
        xi = xi_min + xi_width * float(uxi)
        xi = min(
            math.nextafter(1.0, 0.0),
            max(math.nextafter(xi_min, 1.0), xi),
        )
        loop_coordinates = (float(uk1), float(uk2), float(uk3))

        def paired_sample(
            use_fallback: bool,
        ) -> tuple[tuple[float, ...], float, float, float, float]:
            born_at_endpoint = common_weight * normalisation * born_integrand(
                partonic_scale_sq,
                loop_coordinates,
                use_fallback,
            )
            born_at_xi = common_weight * normalisation * born_integrand(
                xi * partonic_scale_sq,
                loop_coordinates,
                use_fallback,
            )
            continuous = finite_gg_scheme_continuous_components(
                xi,
                born_at_xi,
                born_at_endpoint,
                lambda_value,
                mur_value,
                alpha_value,
            )
            component_weights = (
                2.0 * xi_width * continuous[0],
                2.0 * xi_width * continuous[1],
                2.0 * xi_width * continuous[2],
                2.0
                * finite_gg_scheme_lower_limit_remainder(
                    xi_min,
                    born_at_endpoint,
                    lambda_value,
                    mur_value,
                    alpha_value,
                ),
                2.0 * endpoint_coefficients.dgg_delta * born_at_endpoint,
                2.0 * endpoint_coefficients.minus_lsz * born_at_endpoint,
                endpoint_coefficients.top_lsz_total * born_at_endpoint,
            )
            dgg_minus_lsz_weight = math.fsum(component_weights[:-1])
            top_lsz_weight = component_weights[-1]
            combined_weight = dgg_minus_lsz_weight + top_lsz_weight
            return (
                component_weights,
                dgg_minus_lsz_weight,
                top_lsz_weight,
                born_at_endpoint,
                combined_weight,
            )

        fallback_count = 0
        clipped_count = 0
        nonfinite_count = 0
        try:
            (
                component_weights,
                dgg_minus_lsz_weight,
                top_lsz_weight,
                born_weight,
                combined_weight,
            ) = paired_sample(False)
        except Exception:
            component_weights = tuple(math.nan for _ in component_labels)
            dgg_minus_lsz_weight = math.nan
            top_lsz_weight = math.nan
            born_weight = math.nan
            combined_weight = math.nan
        sample_values = (
            *component_weights,
            dgg_minus_lsz_weight,
            top_lsz_weight,
            born_weight,
            combined_weight,
        )
        needs_fallback = any(not math.isfinite(value) for value in sample_values)
        if clip_threshold is not None and not needs_fallback:
            needs_fallback = max(
                abs(dgg_minus_lsz_weight),
                abs(top_lsz_weight),
                abs(born_weight),
                abs(combined_weight),
            ) > clip_threshold
        if needs_fallback:
            fallback_count = 1
            try:
                (
                    component_weights,
                    dgg_minus_lsz_weight,
                    top_lsz_weight,
                    born_weight,
                    combined_weight,
                ) = paired_sample(True)
            except Exception:
                component_weights = tuple(math.nan for _ in component_labels)
                dgg_minus_lsz_weight = math.nan
                top_lsz_weight = math.nan
                born_weight = math.nan
                combined_weight = math.nan

        sample_values = (
            *component_weights,
            dgg_minus_lsz_weight,
            top_lsz_weight,
            born_weight,
            combined_weight,
        )
        if any(not math.isfinite(value) for value in sample_values):
            nonfinite_count = 1
            component_weights = tuple(0.0 for _ in component_labels)
            dgg_minus_lsz_weight = 0.0
            top_lsz_weight = 0.0
            born_weight = 0.0
            combined_weight = 0.0
        elif clip_threshold is not None and max(
            abs(dgg_minus_lsz_weight),
            abs(top_lsz_weight),
            abs(born_weight),
            abs(combined_weight),
        ) > clip_threshold:
            clipped_count = 1
            component_weights = tuple(0.0 for _ in component_labels)
            dgg_minus_lsz_weight = 0.0
            top_lsz_weight = 0.0
            born_weight = 0.0
            combined_weight = 0.0
        return (
            (
                *component_weights,
                dgg_minus_lsz_weight,
                top_lsz_weight,
                born_weight,
                combined_weight,
            ),
            fallback_count,
            clipped_count,
            nonfinite_count,
        )

    replica_sums, fallback_count, clipped_count, nonfinite_count = (
        _integrate_sobol_chunks(
            dimensions=dimensions,
            power=power,
            replicas=replica_count,
            seed=seed,
            output_width=11,
            evaluator=evaluate_sample,
            workers=workers,
        )
    )
    sample_normalisation = float(2**power)
    for sums in replica_sums:
        dgg_minus_lsz_replica_values.append(sums[7] / sample_normalisation)
        top_lsz_replica_values.append(sums[8] / sample_normalisation)
        born_replica_values.append(sums[9] / sample_normalisation)
        combined_replica_values.append(sums[10] / sample_normalisation)
        for label, component_sum in zip(
            component_labels,
            sums[:7],
            strict=True,
        ):
            component_replica_values[label].append(
                component_sum / sample_normalisation
            )

    if nonfinite_count == sample_count:
        raise pygloopException(
            "Every ttbar gg auxiliary sample remained non-finite after "
            "higher-precision fallback."
        )

    def summarise(values: list[float]) -> tuple[float, float]:
        central = math.fsum(values) / replica_count
        variance = math.fsum(
            (value - central) ** 2 for value in values
        ) / (replica_count - 1)
        return central, math.sqrt(variance / replica_count)

    dgg_minus_lsz_central, dgg_minus_lsz_error = summarise(
        dgg_minus_lsz_replica_values
    )
    top_lsz_central, top_lsz_error = summarise(top_lsz_replica_values)
    born_central, born_error = summarise(born_replica_values)
    combined_central, combined_error = summarise(combined_replica_values)
    components = []
    for label in component_labels:
        values = component_replica_values[label]
        central, error = summarise(values)
        components.append(
            DYSchemeCountertermComponentResult(
                label=label,
                central_value=central,
                error=error,
                replica_values=tuple(values),
            )
        )

    return DYGGAuxiliaryResult(
        dgg_minus_lsz_central_value=dgg_minus_lsz_central,
        dgg_minus_lsz_error=dgg_minus_lsz_error,
        top_lsz_central_value=top_lsz_central,
        top_lsz_error=top_lsz_error,
        born_central_value=born_central,
        born_error=born_error,
        combined_central_value=combined_central,
        combined_error=combined_error,
        n_samples=sample_count,
        elapsed_time=time.monotonic() - start,
        dgg_minus_lsz_replica_values=tuple(dgg_minus_lsz_replica_values),
        top_lsz_replica_values=tuple(top_lsz_replica_values),
        born_replica_values=tuple(born_replica_values),
        combined_replica_values=tuple(combined_replica_values),
        components=tuple(components),
        fallback_count=fallback_count,
        clipped_count=clipped_count,
        nonfinite_count=nonfinite_count,
    )
