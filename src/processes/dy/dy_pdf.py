from __future__ import annotations

import importlib
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

from utils.utils import pygloopException


DY_CHANNEL_PDG_IDS: dict[tuple[int, int], tuple[int, int]] = {
    (0, 0): (21, 21),
    (0, 1): (21, 1),
    (1, 0): (1, 21),
    (1, -1): (1, -1),
    (-1, 1): (-1, 1),
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


QQBAR_SCHEME_DELTA_COEFFICIENT: float = (4.0 * math.pi**2 - 48.0) / (
    18.0 * math.pi
)
QQBAR_SCHEME_COUNTERTERM_FACTOR: float = -2.0


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
    return (regular + log_x_plus + log_one_minus_x_plus) / (18.0 * math.pi)


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

    def __init__(self, set_name: str, member: int = 0):
        self.set_name = str(set_name).strip()
        if not self.set_name:
            raise pygloopException("The DY PDF set name cannot be empty.")
        self.member = int(member)
        if self.member < 0:
            raise pygloopException("The DY PDF member must be non-negative.")
        self._pdf: Any | None = None

    def _load_pdf(self) -> Any:
        if self._pdf is not None:
            return self._pdf
        try:
            lhapdf = importlib.import_module("lhapdf")
        except ImportError as exc:
            raise pygloopException(
                "DY PDF weighting requires the LHAPDF Python bindings."
            ) from exc
        try:
            self._pdf = lhapdf.mkPDF(self.set_name, self.member)
        except Exception as exc:
            raise pygloopException(
                f"Could not load LHAPDF set {self.set_name!r}, member "
                f"{self.member}: {exc}"
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
        parton1, parton2 = pdf_partons_for_channel(channel)
        luminosity = self.density(parton1, x1, muf_sq, "x1") * self.density(
            parton2, x2, muf_sq, "x2"
        )
        if not math.isfinite(luminosity):
            raise pygloopException(
                f"Non-finite PDF luminosity for DY channel {channel}."
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
            log_x2_min = math.log(tau_min / x1)
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
    """Integrate the two-leg finite qqbar scheme term at Lambdasq=1."""

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

    sample_count = replica_count * 2**power
    endpoint_tau = minimum_q**2 / centre_of_mass_sq
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

    minimum_q_sq = minimum_q**2
    maximum_q_sq = maximum_q**2 if maximum_q is not None else None
    log_endpoint_tau = math.log(endpoint_tau)
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
            x1 = math.exp((1.0 - float(u1)) * log_endpoint_tau)
            log_x2_min = math.log(endpoint_tau / x1)
            x2 = math.exp((1.0 - float(u2)) * log_x2_min)
            jacobian_over_x1_x2 = (-log_endpoint_tau) * (-log_x2_min)

            xi = min(
                math.nextafter(1.0, 0.0),
                max(math.nextafter(0.0, 1.0), float(uz)),
            )
            partonic_scale_sq = x1 * x2 * centre_of_mass_sq
            luminosity = provider.luminosity(channel, x1, x2, muf_sq)

            q_sq = xi * partonic_scale_sq
            born_at_xi = luminosity
            if (
                xi < requested_z_min
                or xi > requested_z_max
                or q_sq < minimum_q_sq
                or (maximum_q_sq is not None and q_sq > maximum_q_sq)
            ):
                born_at_xi = 0.0

            born_at_endpoint = luminosity
            if (
                requested_z_max < 1.0
                or partonic_scale_sq < minimum_q_sq
                or (
                    maximum_q_sq is not None
                    and partonic_scale_sq > maximum_q_sq
                )
            ):
                born_at_endpoint = 0.0

            one_leg_weight = finite_qqbar_scheme_continuous_weight(
                xi,
                born_at_xi,
                born_at_endpoint,
            )
            one_leg_weight += (
                QQBAR_SCHEME_DELTA_COEFFICIENT * born_at_endpoint
            )
            accumulated += jacobian_over_x1_x2 * 2.0 * one_leg_weight

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
