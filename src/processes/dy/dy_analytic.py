"""Analytic NLO MSbar coefficient for partonic Drell--Yan production.

The coefficient is the full (not threshold-expanded) qqbar result quoted in
Eq. (3.24) of arXiv:0909.2124, converted from that paper's
``a_s = alpha_s/(4*pi)`` convention to

    Delta(z) = delta(1-z) + alpha_s/(2*pi) * D1(z) + ... .

For the z-weighted observable used by the DY campaign, ``z = Q^2 / shat``
and the Born prefactor is constant while z is integrated at fixed ``shat``.
After combining the explicit ``log(z)`` term with factorisation-scale
evolution, the exact coefficient takes the especially simple form

    D1(z) = CF * [
        4 (1+z^2) (log(1-z)/(1-z))_+
        + 2 log(shat/muf^2) [(1+z^2)/(1-z)]_+
        + (2*pi^2/3-8) delta(1-z)
    ].

Both plus prescriptions below are implemented as distributions, including
for finite bins that end at z=1.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


DY_CF = 4.0 / 3.0


@dataclass(frozen=True)
class DYQQbarNLOBin:
    """Dimensionless integral of the order-alpha_s qqbar coefficient."""

    soft_plus: float
    factorisation_plus: float
    delta: float

    @property
    def total(self) -> float:
        return math.fsum((self.soft_plus, self.factorisation_plus, self.delta))


def _validate_bin(z_min: float, z_max: float) -> tuple[float, float]:
    low = float(z_min)
    high = float(z_max)
    if not (
        math.isfinite(low)
        and math.isfinite(high)
        and 0.0 <= low < high <= 1.0
    ):
        raise ValueError("DY z bin requires 0 <= z_min < z_max <= 1")
    return low, high


def _positive_scale(value: float, label: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return resolved


def _ordinary_d1_primitive(z: float) -> float:
    """Primitive of ``(1+z^2) log(1-z)/(1-z)`` away from z=1."""

    one_minus_z = 1.0 - z
    logarithm = math.log(one_minus_z)
    return -(
        logarithm**2
        - 2.0 * one_minus_z * logarithm
        + 2.0 * one_minus_z
        + 0.5 * one_minus_z**2 * logarithm
        - 0.25 * one_minus_z**2
    )


def weighted_d1_bin_integral(z_min: float, z_max: float) -> float:
    """Integrate ``(1+z^2) [log(1-z)/(1-z)]_+`` over one bin."""

    low, high = _validate_bin(z_min, z_max)
    if high < 1.0:
        return _ordinary_d1_primitive(high) - _ordinary_d1_primitive(low)

    width = 1.0 - low
    logarithm = math.log(width)
    return (
        logarithm**2
        - (2.0 * width - 0.5 * width**2) * logarithm
        + 2.0 * width
        - 0.25 * width**2
    )


def _ordinary_splitting_primitive(z: float) -> float:
    return -2.0 * math.log1p(-z) - z - 0.5 * z**2


def qqbar_splitting_plus_bin_integral(z_min: float, z_max: float) -> float:
    """Integrate ``[(1+z^2)/(1-z)]_+`` over one bin."""

    low, high = _validate_bin(z_min, z_max)
    if high < 1.0:
        return _ordinary_splitting_primitive(high) - _ordinary_splitting_primitive(
            low
        )
    return -_ordinary_splitting_primitive(low)


def _log_z_primitive(z: float) -> float:
    """Primitive of ``(1+z^2) log(z)/(1-z)`` on [0,1]."""

    if z == 0.0:
        return math.pi**2 / 3.0
    if z == 1.0:
        return 1.25
    try:
        from scipy.special import spence
    except ImportError as exc:  # pragma: no cover - scipy is a runtime dependency
        raise RuntimeError("The analytic DY benchmark requires scipy.special") from exc
    logarithm = math.log(z)
    return (
        2.0 * float(spence(z))
        - z * logarithm
        + z
        - 0.5 * z**2 * logarithm
        + 0.25 * z**2
    )


def qqbar_log_z_bin_integral(z_min: float, z_max: float) -> float:
    """Integrate ``(1+z^2) log(z)/(1-z)`` over one bin."""

    low, high = _validate_bin(z_min, z_max)
    return _log_z_primitive(high) - _log_z_primitive(low)


def nlo_qqbar_msbar_bin(
    z_min: float,
    z_max: float,
    *,
    shat: float,
    muf_sq: float,
    colour_factor: float = DY_CF,
) -> DYQQbarNLOBin:
    """Return the full analytic NLO qqbar coefficient integrated over a z bin.

    The returned value does not include ``alpha_s/(2*pi)`` or the Born cross
    section.  ``muf_sq`` is fixed while z varies, matching the MG5 campaign.
    """

    low, high = _validate_bin(z_min, z_max)
    partonic_scale_sq = _positive_scale(shat, "shat")
    factorisation_scale_sq = _positive_scale(muf_sq, "muf_sq")
    cf = float(colour_factor)
    if not math.isfinite(cf):
        raise ValueError("colour_factor must be finite")

    soft = 4.0 * cf * weighted_d1_bin_integral(low, high)
    factorisation = (
        2.0
        * cf
        * math.log(partonic_scale_sq / factorisation_scale_sq)
        * qqbar_splitting_plus_bin_integral(low, high)
    )
    delta = cf * (2.0 * math.pi**2 / 3.0 - 8.0) if high == 1.0 else 0.0
    return DYQQbarNLOBin(
        soft_plus=soft,
        factorisation_plus=factorisation,
        delta=delta,
    )


def photon_dy_born_cross_section(
    shat: float,
    *,
    alpha_ew_inverse: float,
    quark_charge_squared: float = 1.0 / 9.0,
    colours: int = 3,
) -> float:
    """Return the integrated ``q qbar -> gamma* -> l+ l-`` Born cross section.

    The result is in GeV^-2 and is for one ordered partonic initial state.
    """

    partonic_scale_sq = _positive_scale(shat, "shat")
    inverse_alpha = _positive_scale(alpha_ew_inverse, "alpha_ew_inverse")
    charge_squared = _positive_scale(quark_charge_squared, "quark charge squared")
    colour_count = int(colours)
    if colour_count <= 0 or colour_count != colours:
        raise ValueError("colours must be a positive integer")
    alpha_ew = 1.0 / inverse_alpha
    return (
        4.0
        * math.pi
        * alpha_ew**2
        * charge_squared
        / (3.0 * colour_count * partonic_scale_sq)
    )
