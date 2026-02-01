from math import sqrt

from utils.vectors import LorentzVector, Vector  # noqa: F401


def verify_ps_point(
    ps_point: list[LorentzVector],
    target_s: float | None = None,
    target_t=None,
    target_costheta=None,
    target_cosphi=None,
    tolerance: float = 1e-6,
):
    """Verifies that a given phase space point matches the target invariants and angles.

    Conventions must match generate_two_to_two_at_fixed_angle_positive_s:
      - CM frame with p1 along +z
      - theta is the polar angle of p3 wrt +z (i.e. wrt p1 direction)
      - phi is the azimuth of p3 in the x-y plane with sin(phi) chosen >= 0
        (generator uses sinphi = +sqrt(1-cosphi^2), so phi in [0, pi]).
    """
    p1, p2, p3, p4 = ps_point

    s = (p1 + p2).squared()
    t = (p1 - p3).squared()

    p1s = p1.spatial()
    p3s = p3.spatial()

    p1s2 = p1s.squared()
    p3s2 = p3s.squared()

    # cos(theta): angle between spatial p1 and spatial p3 (matches generator's definition)
    if p1s2 <= tolerance**2 or p3s2 <= tolerance**2:
        # At threshold / zero 3-momentum the angles are undefined; pick a benign value.
        costheta = 1.0
    else:
        costheta = p1s.dot(p3s) / sqrt(p1s2 * p3s2)
        costheta = max(-1.0, min(1.0, costheta))  # numeric safety

    # cos(phi): azimuth of p3 around the +z axis (the axes are the fixed x,y of the generator)
    p3_perp2 = p3s.x * p3s.x + p3s.y * p3s.y
    if p3_perp2 <= tolerance**2:
        # p3 has no transverse component -> phi undefined
        cosphi = 1.0
    else:
        cosphi = p3s.x / sqrt(p3_perp2)
        cosphi = max(-1.0, min(1.0, cosphi))  # numeric safety

        # Match the generator's choice sin(phi) = +sqrt(1-cos^2) => p3y >= 0
        assert p3s.y >= -tolerance, f"phi convention mismatch: generator enforces sinphi>=0, but got p3y={p3s.y}"

    if target_s is not None:
        assert abs(s - target_s) < tolerance, f"s mismatch: {s} vs {target_s}"
    if target_t is not None:
        assert abs(t - target_t) < tolerance, f"t mismatch: {t} vs {target_t}"
    if target_costheta is not None:
        assert abs(costheta - target_costheta) < tolerance, f"costheta mismatch: {costheta} vs {target_costheta}"
    if target_cosphi is not None:
        # If phi is undefined (no transverse momentum), there is nothing meaningful to compare.
        if p3_perp2 <= tolerance**2:
            raise AssertionError("cosphi was requested but phi is undefined (p3 has zero transverse momentum).")
        assert abs(cosphi - target_cosphi) < tolerance, f"cosphi mismatch: {cosphi} vs {target_cosphi}"


def generate_two_to_two_at_fixed_angle_positive_s(M1sq, M2sq, M3sq, M4sq, s, costheta, cosphi):
    """Generates a point with given masses, positive s and angles theta and phi between p1 and p3."""
    sintheta = sqrt(1.0 - costheta**2)
    sinphi = sqrt(1.0 - cosphi**2)
    return [
        LorentzVector(
            (M1sq - M2sq + s) / (2.0 * sqrt(s)),
            0,
            0,
            sqrt(M1sq**2 + (M2sq - s) ** 2 - 2 * M1sq * (M2sq + s)) / (2.0 * sqrt(s)),
        ),
        LorentzVector(
            (-M1sq + M2sq + s) / (2.0 * sqrt(s)),
            0,
            0,
            -sqrt(M1sq**2 + (M2sq - s) ** 2 - 2 * M1sq * (M2sq + s)) / (2.0 * sqrt(s)),
        ),
        LorentzVector(
            -(M3sq - M4sq + s) / (2.0 * sqrt(s)),
            (cosphi * sqrt(M3sq**2 + (M4sq - s) ** 2 - 2 * M3sq * (M4sq + s)) * sintheta) / (2.0 * sqrt(s)),
            (sqrt(M3sq**2 + (M4sq - s) ** 2 - 2 * M3sq * (M4sq + s)) * sinphi * sintheta) / (2.0 * sqrt(s)),
            (costheta * sqrt(M3sq**2 + (M4sq - s) ** 2 - 2 * M3sq * (M4sq + s))) / (2.0 * sqrt(s)),
        ),
        LorentzVector(
            -(-M3sq + M4sq + s) / (2.0 * sqrt(s)),
            -(cosphi * sqrt(M3sq**2 + (M4sq - s) ** 2 - 2 * M3sq * (M4sq + s)) * sintheta) / (2.0 * sqrt(s)),
            -(sqrt(M3sq**2 + (M4sq - s) ** 2 - 2 * M3sq * (M4sq + s)) * sinphi * sintheta) / (2.0 * sqrt(s)),
            -(costheta * sqrt(M3sq**2 + (M4sq - s) ** 2 - 2 * M3sq * (M4sq + s))) / (2.0 * sqrt(s)),
        ),
    ]


def generate_two_to_two_at_fixed_t_positive_s(M1sq, M2sq, M3sq, M4sq, s, t, cosphi):
    """Generates a point with given masses, positive s, t and angle phi between p1 and p3."""
    sinphi = sqrt(1.0 - cosphi**2)
    E3 = (M3sq - M4sq + ((M1sq - M2sq + s) / (2.0 * sqrt(s)) + (-M1sq + M2sq + s) / (2.0 * sqrt(s))) ** 2) / (
        2.0 * ((M1sq - M2sq + s) / (2.0 * sqrt(s)) + (-M1sq + M2sq + s) / (2.0 * sqrt(s)))
    )
    E4 = (-M3sq + M4sq + ((M1sq - M2sq + s) / (2.0 * sqrt(s)) + (-M1sq + M2sq + s) / (2.0 * sqrt(s))) ** 2) / (
        2.0 * ((M1sq - M2sq + s) / (2.0 * sqrt(s)) + (-M1sq + M2sq + s) / (2.0 * sqrt(s)))
    )
    p3r = -sqrt(
        (
            (M1sq**2 + (M2sq - s) ** 2 - 2 * M1sq * (M2sq + s))
            * (
                M4sq**2
                + (-M3sq + ((M1sq - M2sq + s) / (2.0 * sqrt(s)) + (-M1sq + M2sq + s) / (2.0 * sqrt(s))) ** 2) ** 2
                - 2 * M4sq * (M3sq + ((M1sq - M2sq + s) / (2.0 * sqrt(s)) + (-M1sq + M2sq + s) / (2.0 * sqrt(s))) ** 2)
            )
        )
        / s
    ) / (
        2.0
        * sqrt(
            ((M1sq**2 + (M2sq - s) ** 2 - 2 * M1sq * (M2sq + s)) * ((M1sq - M2sq + s) / (2.0 * sqrt(s)) + (-M1sq + M2sq + s) / (2.0 * sqrt(s))) ** 2)
            / s
        )
    )
    costheta = (
        -2
        * sqrt(M1sq**2 + (M2sq - s) ** 2 - 2 * M1sq * (M2sq + s))
        * ((M1sq - M2sq + s) / (2.0 * sqrt(s)) + (-M1sq + M2sq + s) / (2.0 * sqrt(s)))
        * (
            ((M1sq - M2sq + s) ** 2 * (-M1sq + M2sq + s)) / (8.0 * s**1.5)
            + ((-M1sq + M2sq + s) * (-M3sq + (M1sq**2 + (M2sq - s) ** 2 - 2 * M1sq * (M2sq + s)) / (4.0 * s) + t)) / (2.0 * sqrt(s))
            + (
                (M1sq - M2sq + s)
                * (-M4sq + (-M1sq + M2sq + s) ** 2 / (4.0 * s) + (M1sq**2 + (M2sq - s) ** 2 - 2 * M1sq * (M2sq + s)) / (4.0 * s) + t)
            )
            / (2.0 * sqrt(s))
        )
    ) / (
        sqrt(s)
        * sqrt(
            ((M1sq**2 + (M2sq - s) ** 2 - 2 * M1sq * (M2sq + s)) * ((M1sq - M2sq + s) / (2.0 * sqrt(s)) + (-M1sq + M2sq + s) / (2.0 * sqrt(s))) ** 2)
            / s
        )
        * sqrt(
            (
                (M1sq**2 + (M2sq - s) ** 2 - 2 * M1sq * (M2sq + s))
                * (
                    M4sq**2
                    + (-M3sq + ((M1sq - M2sq + s) / (2.0 * sqrt(s)) + (-M1sq + M2sq + s) / (2.0 * sqrt(s))) ** 2) ** 2
                    - 2 * M4sq * (M3sq + ((M1sq - M2sq + s) / (2.0 * sqrt(s)) + (-M1sq + M2sq + s) / (2.0 * sqrt(s))) ** 2)
                )
            )
            / s
        )
    )
    # sintheta = sqrt(1.0 - costheta**2)
    return [
        LorentzVector(
            (M1sq - M2sq + s) / (2.0 * sqrt(s)),
            0,
            0,
            sqrt(M1sq**2 + (M2sq - s) ** 2 - 2 * M1sq * (M2sq + s)) / (2.0 * sqrt(s)),
        ),
        LorentzVector(
            (-M1sq + M2sq + s) / (2.0 * sqrt(s)),
            0,
            0,
            -sqrt(M1sq**2 + (M2sq - s) ** 2 - 2 * M1sq * (M2sq + s)) / (2.0 * sqrt(s)),
        ),
        LorentzVector(
            -E3,
            -(cosphi * sqrt(1 - costheta**2) * p3r),
            -(sqrt(1 - costheta**2) * p3r * sinphi),
            -(costheta * p3r),
        ),
        LorentzVector(
            -E4,
            cosphi * sqrt(1 - costheta**2) * p3r,
            sqrt(1 - costheta**2) * p3r * sinphi,
            costheta * p3r,
        ),
    ]


def generate_two_to_two_at_fixed_angle_negative_s(M1sq, M2sq, M3sq, M4sq, minus_s, costheta, cosphi):
    """Generates a point with given masses, positive s and angles theta and phi between p1 and p3."""
    s = -minus_s
    sintheta = sqrt(1.0 - costheta**2)
    sinphi = sqrt(1.0 - cosphi**2)
    # sectheta = 1.0 / sintheta
    cos2theta = 2.0 * costheta**2 - 1.0
    # cos4theta = 2.0 * (2.0 * costheta**2 - 1.0) ** 2 - 1.0
    tantheta = sintheta / costheta

    return [
        LorentzVector(
            sqrt(M1sq**2 + 2 * M1sq * (-M2sq + s) + (M2sq + s) ** 2) / (2.0 * sqrt(s)),
            0,
            0,
            (-M1sq + M2sq + s) / (2.0 * sqrt(s)),
        ),
        LorentzVector(
            -sqrt(M1sq**2 + 2 * M1sq * (-M2sq + s) + (M2sq + s) ** 2) / (2.0 * sqrt(s)),
            0,
            0,
            (M1sq - M2sq + s) / (2.0 * sqrt(s)),
        ),
        LorentzVector(
            -sqrt(M3sq**2 - 2 * M3sq * M4sq + 2 * cos2theta * M3sq * s + (M4sq + s) ** 2) / (2.0 * sqrt(costheta**2 * s)),
            -(cosphi * (-M3sq + M4sq + s) * tantheta) / (2.0 * sqrt(s)),
            -((-M3sq + M4sq + s) * sinphi * tantheta) / (2.0 * sqrt(s)),
            -(-M3sq + M4sq + s) / (2.0 * sqrt(s)),
        ),
        LorentzVector(
            sqrt(M3sq**2 - 2 * M3sq * M4sq + 2 * cos2theta * M3sq * s + (M4sq + s) ** 2) / (2.0 * sqrt(costheta**2 * s)),
            (cosphi * (-M3sq + M4sq + s) * tantheta) / (2.0 * sqrt(s)),
            ((-M3sq + M4sq + s) * sinphi * tantheta) / (2.0 * sqrt(s)),
            -(M3sq - M4sq + s) / (2.0 * sqrt(s)),
        ),
    ]


def generate_two_to_two_at_fixed_t_negative_s(M1sq, M2sq, M3sq, M4sq, minus_s, t, cosphi):
    """Generates a point with given masses, negative s, t and angle phi between p1 and p3."""
    s = -minus_s
    sinphi = sqrt(1.0 - cosphi**2)

    E1 = sqrt(M1sq**2 + 2 * M1sq * (-M2sq + s) + (M2sq + s) ** 2) / (2.0 * sqrt(s))
    p1z = (-M1sq + M2sq + s) / (2.0 * sqrt(s))
    p2z = (M1sq - M2sq + s) / (2.0 * sqrt(s))

    E3 = -(M4sq * p1z + M3sq * p2z + p1z**2 * p2z + p1z * p2z**2 + E1**2 * (p1z + p2z) - p1z * t - p2z * t) / (2.0 * E1 * (p1z + p2z))
    E4 = (M4sq * p1z + M3sq * p2z + p1z**2 * p2z + p1z * p2z**2 + E1**2 * (p1z + p2z) - p1z * t - p2z * t) / (2.0 * E1 * (p1z + p2z))
    p3r = -sqrt(
        E1**4 * (p1z + p2z) ** 2
        + (M4sq * p1z + M3sq * p2z + (p1z + p2z) * (p1z * p2z - t)) ** 2
        + 2 * E1**2 * (p1z + p2z) * (M4sq * p1z - M3sq * (2 * p1z + p2z) + (p1z + p2z) * (p1z * p2z - t))
    ) / (2.0 * sqrt(E1**2 * (p1z + p2z) ** 2))
    p4z = -p1z - p2z
    costheta = (sqrt(E1**2 * (p1z + p2z) ** 2) * (-M3sq + M4sq + (p1z + p2z) ** 2)) / (
        (p1z + p2z)
        * sqrt(
            E1**4 * (p1z + p2z) ** 2
            + (M4sq * p1z + M3sq * p2z + (p1z + p2z) * (p1z * p2z - t)) ** 2
            + 2 * E1**2 * (p1z + p2z) * (M4sq * p1z - M3sq * (2 * p1z + p2z) + (p1z + p2z) * (p1z * p2z - t))
        )
    )

    return [
        LorentzVector(
            E1,
            0,
            0,
            p1z,
        ),
        LorentzVector(
            -E1,
            0,
            0,
            p2z,
        ),
        LorentzVector(
            E3,
            cosphi * sqrt(1 - costheta**2) * p3r,
            sqrt(1 - costheta**2) * p3r * sinphi,
            costheta * p3r,
        ),
        LorentzVector(
            E4,
            -(cosphi * sqrt(1 - costheta**2) * p3r),
            -(sqrt(1 - costheta**2) * p3r * sinphi),
            -(costheta * p3r) + p4z,
        ),
    ]
