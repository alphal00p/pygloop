from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
from typing import Any, Callable, Sequence

from utils.vectors import LorentzVector, Vector


RealInput = float | Decimal | int | str
RotationMatrix = tuple[
    tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]
]


_KNOWN_MASSIVE_SOFT_EDGE_PARTICLES = {
    "h",
    "t",
    "t~",
    "w+",
    "w-",
    "z",
}


def _routing_fraction(value: Any, description: str) -> Fraction:
    try:
        return Fraction(str(value).strip())
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(
            f"Soft-mirror {description} must be an exact rational, got {value!r}."
        ) from exc


@dataclass(frozen=True)
class SoftEdgeRouting:
    """Exact routing needed to reflect one edge about its soft point."""

    source_graph_name: str
    edge_id: str
    particle: str
    loop_coefficients: tuple[Fraction, ...]
    p1_coefficient: Fraction
    p2_coefficient: Fraction
    pivot_loop_index: int

    @classmethod
    def from_metadata(
        cls,
        source_graph_name: str,
        metadata: dict[str, Any],
    ) -> SoftEdgeRouting:
        edge_id = str(metadata.get("edge_id", "")).strip()
        if not edge_id:
            raise ValueError("Soft-mirror edge metadata is missing an edge id.")
        particle = str(metadata.get("particle", "")).strip()
        if particle.lower() in _KNOWN_MASSIVE_SOFT_EDGE_PARTICLES:
            raise ValueError(
                f"Soft-mirror edge {source_graph_name}:{edge_id} carries massive "
                f"particle {particle!r}."
            )

        raw_loop_coefficients = metadata.get("loop_coefficients")
        if not isinstance(raw_loop_coefficients, list) or not raw_loop_coefficients:
            raise ValueError(
                f"Soft-mirror edge {source_graph_name}:{edge_id} has no loop routing."
            )
        loop_coefficients = tuple(
            _routing_fraction(value, f"edge {edge_id} loop coefficient")
            for value in raw_loop_coefficients
        )
        p1_coefficient = _routing_fraction(
            metadata.get("p1_coefficient", "0"),
            f"edge {edge_id} p1 coefficient",
        )
        p2_coefficient = _routing_fraction(
            metadata.get("p2_coefficient", "0"),
            f"edge {edge_id} p2 coefficient",
        )
        first_nonzero = next(
            (
                coefficient
                for coefficient in (
                    *loop_coefficients,
                    p1_coefficient,
                    p2_coefficient,
                )
                if coefficient
            ),
            None,
        )
        if first_nonzero is not None and first_nonzero < 0:
            loop_coefficients = tuple(-value for value in loop_coefficients)
            p1_coefficient = -p1_coefficient
            p2_coefficient = -p2_coefficient
        nonzero_loop_indices = [
            index for index, coefficient in enumerate(loop_coefficients) if coefficient
        ]
        if not nonzero_loop_indices:
            raise ValueError(
                f"Soft-mirror edge {source_graph_name}:{edge_id} carries no loop momentum."
            )

        raw_lmb_id = metadata.get("lmb_id")
        pivot_loop_index: int | None = None
        if raw_lmb_id is not None and str(raw_lmb_id).strip() != "":
            try:
                pivot_loop_index = int(str(raw_lmb_id).strip())
            except ValueError as exc:
                raise ValueError(
                    f"Soft-mirror edge {source_graph_name}:{edge_id} has invalid "
                    f"lmb_id={raw_lmb_id!r}."
                ) from exc
            if (
                pivot_loop_index < 0
                or pivot_loop_index >= len(loop_coefficients)
                or not loop_coefficients[pivot_loop_index]
            ):
                raise ValueError(
                    f"Soft-mirror edge {source_graph_name}:{edge_id} has lmb_id "
                    f"{pivot_loop_index}, incompatible with routing "
                    f"{list(map(str, loop_coefficients))}."
                )
        elif len(nonzero_loop_indices) == 1:
            pivot_loop_index = nonzero_loop_indices[0]
        else:
            raise ValueError(
                f"Soft-mirror edge {source_graph_name}:{edge_id} depends on multiple "
                "loop momenta but is not an LMB edge; the pivot loop is ambiguous."
            )

        return cls(
            source_graph_name=str(source_graph_name),
            edge_id=edge_id,
            particle=particle,
            loop_coefficients=loop_coefficients,
            p1_coefficient=p1_coefficient,
            p2_coefficient=p2_coefficient,
            pivot_loop_index=pivot_loop_index,
        )

    @property
    def label(self) -> str:
        return f"{self.source_graph_name}:{self.edge_id}"

    @property
    def has_external_offset(self) -> bool:
        return bool(self.p1_coefficient or self.p2_coefficient)


def _fraction_for_value(coefficient: Fraction, value: float | Decimal):
    if isinstance(value, Decimal):
        return Decimal(coefficient.numerator) / Decimal(coefficient.denominator)
    return float(coefficient)


def soft_edge_momentum(
    loop_momenta: Sequence[Vector],
    p1: Vector,
    p2: Vector,
    routing: SoftEdgeRouting,
) -> Vector:
    """Evaluate the selected edge's spatial momentum with no precision loss."""
    if len(loop_momenta) != len(routing.loop_coefficients):
        raise ValueError(
            f"Soft-mirror routing {routing.label} expects "
            f"{len(routing.loop_coefficients)} loops, got {len(loop_momenta)}."
        )
    exemplar = loop_momenta[0].to_list()[0]
    zero = Decimal(0) if isinstance(exemplar, Decimal) else 0.0
    momentum = Vector(zero, zero, zero)
    for coefficient, loop_momentum in zip(
        routing.loop_coefficients, loop_momenta, strict=True
    ):
        if coefficient:
            momentum += loop_momentum * _fraction_for_value(coefficient, exemplar)
    if routing.p1_coefficient:
        momentum += p1 * _fraction_for_value(routing.p1_coefficient, exemplar)
    if routing.p2_coefficient:
        momentum += p2 * _fraction_for_value(routing.p2_coefficient, exemplar)
    return momentum


def soft_center_for_rescaling(
    loop_momenta: Sequence[Vector],
    p1: Vector,
    p2: Vector,
    routing: SoftEdgeRouting,
    rescaling_t: float | Decimal,
) -> Vector:
    """Return the raw pivot momentum for which the rescaled edge is soft."""
    if len(loop_momenta) != len(routing.loop_coefficients):
        raise ValueError(
            f"Soft-mirror routing {routing.label} expects "
            f"{len(routing.loop_coefficients)} loops, got {len(loop_momenta)}."
        )
    if rescaling_t <= 0:
        raise ValueError("A shifted soft centre requires strictly positive t.")

    pivot = routing.pivot_loop_index
    exemplar = loop_momenta[pivot].to_list()[0]
    zero = Decimal(0) if isinstance(exemplar, Decimal) else 0.0
    spectator_sum = Vector(zero, zero, zero)
    for index, (coefficient, loop_momentum) in enumerate(
        zip(routing.loop_coefficients, loop_momenta, strict=True)
    ):
        if index != pivot and coefficient:
            spectator_sum += loop_momentum * _fraction_for_value(
                coefficient, exemplar
            )

    external_offset = Vector(zero, zero, zero)
    if routing.p1_coefficient:
        external_offset += p1 * _fraction_for_value(
            routing.p1_coefficient, exemplar
        )
    if routing.p2_coefficient:
        external_offset += p2 * _fraction_for_value(
            routing.p2_coefficient, exemplar
        )

    pivot_coefficient = _fraction_for_value(
        routing.loop_coefficients[pivot], exemplar
    )
    return (spectator_sum + external_offset * (1 / rescaling_t)) * (
        -1 / pivot_coefficient
    )


def reflect_across_beam_equator(value: Vector, p1: Vector, p2: Vector) -> Vector:
    """Flip only the component parallel to the incoming-beam axis."""
    beam_axis = p1 - p2
    axis_components = beam_axis.to_list()
    nonzero_axes = [
        index for index, component in enumerate(axis_components) if component != 0
    ]
    if len(nonzero_axes) == 1:
        reflected = value.to_list()
        index = nonzero_axes[0]
        component = reflected[index]
        reflected[index] = (
            component.copy_negate() if isinstance(component, Decimal) else -component
        )
        return Vector(*reflected)
    axis_squared = beam_axis.squared()
    if axis_squared == 0:
        raise ValueError("Soft-mirror reflection requires a non-zero beam axis.")
    return value - beam_axis * (2 * value.dot(beam_axis) / axis_squared)


def mirror_loop_momenta_for_soft_edge(
    loop_momenta: Sequence[Vector],
    p1: Vector,
    p2: Vector,
    routing: SoftEdgeRouting,
    soft_center: Vector | None = None,
) -> tuple[Vector, ...]:
    """Reflect q_edge about q_edge=0 using a unit-Jacobian involution."""
    pivot = routing.pivot_loop_index
    if soft_center is not None:
        mirrored = list(loop_momenta)
        mirrored[pivot] = soft_center * 2 - mirrored[pivot]
        return tuple(mirrored)
    if routing.has_external_offset:
        raise ValueError(
            f"Shifted soft edge {routing.label} requires its conditional soft centre."
        )
    if all(
        not coefficient
        for index, coefficient in enumerate(routing.loop_coefficients)
        if index != pivot
    ) and not routing.p1_coefficient and not routing.p2_coefficient:
        mirrored = list(loop_momenta)
        mirrored[pivot] = reflect_across_beam_equator(
            mirrored[pivot], p1, p2
        )
        return tuple(mirrored)

    edge_momentum = soft_edge_momentum(loop_momenta, p1, p2, routing)
    reflected_edge_momentum = reflect_across_beam_equator(edge_momentum, p1, p2)
    delta = reflected_edge_momentum - edge_momentum
    pivot_component = loop_momenta[pivot].to_list()[0]
    pivot_coefficient = _fraction_for_value(
        routing.loop_coefficients[pivot], pivot_component
    )
    mirrored = list(loop_momenta)
    mirrored[pivot] = mirrored[pivot] + delta * (1 / pivot_coefficient)
    return tuple(mirrored)


def decimal_from_input(value: RealInput) -> Decimal:
    """Preserve every bit of sampled float input when entering higher precision."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Cannot upcast non-finite input {value!r}.")
        return Decimal.from_float(value)
    if isinstance(value, int):
        return Decimal(value)
    return Decimal(value)


def parameterize_ttbar_beam_fractions(
    u_beta: RealInput,
    u_y: RealInput,
    *,
    m_top: RealInput,
    e_cm: RealInput,
) -> tuple[float | Decimal, float | Decimal, float | Decimal]:
    """Map two unit coordinates to threshold-adapted ttbar beam fractions."""
    use_decimal = any(
        isinstance(value, Decimal) for value in (u_beta, u_y, m_top, e_cm)
    )
    if use_decimal:
        beta_coordinate = decimal_from_input(u_beta)
        rapidity_coordinate = decimal_from_input(u_y)
        mass = decimal_from_input(m_top)
        energy = decimal_from_input(e_cm)
        zero = Decimal(0)
        one = Decimal(1)
        if not all(
            value.is_finite()
            for value in (beta_coordinate, rapidity_coordinate, mass, energy)
        ):
            raise ValueError("The beta-Y beam map requires finite inputs.")
        if not (
            zero <= beta_coordinate <= one
            and zero <= rapidity_coordinate <= one
        ):
            raise ValueError("The beta-Y beam coordinates must lie in [0, 1].")
        if mass <= zero or energy <= zero:
            raise ValueError(
                "The beta-Y beam map requires positive m_top and e_cm."
            )

        tau_threshold = Decimal(4) * mass * mass / (energy * energy)
        if tau_threshold >= one or beta_coordinate == one:
            return one, one, zero
        beta_max = (one - tau_threshold).sqrt()
        beta = beta_max * beta_coordinate
        one_minus_beta_sq = one - beta * beta
        tau = tau_threshold / one_minus_beta_sq
        rapidity_width = -tau.ln()
        rapidity = (rapidity_coordinate - one / Decimal(2)) * rapidity_width
        if rapidity_coordinate == zero:
            x1, x2 = tau, one
        elif rapidity_coordinate == one:
            x1, x2 = one, tau
        else:
            sqrt_tau = tau.sqrt()
            x1 = sqrt_tau * rapidity.exp()
            x2 = sqrt_tau * rapidity.copy_negate().exp()
        jacobian = (
            beta_max
            * (
                Decimal(2)
                * tau_threshold
                * beta
                / (one_minus_beta_sq * one_minus_beta_sq)
            )
            * rapidity_width
        )
        return x1, x2, jacobian

    beta_coordinate = float(u_beta)
    rapidity_coordinate = float(u_y)
    mass = float(m_top)
    energy = float(e_cm)
    if not all(
        math.isfinite(value)
        for value in (beta_coordinate, rapidity_coordinate, mass, energy)
    ):
        raise ValueError("The beta-Y beam map requires finite inputs.")
    if not (
        0.0 <= beta_coordinate <= 1.0
        and 0.0 <= rapidity_coordinate <= 1.0
    ):
        raise ValueError("The beta-Y beam coordinates must lie in [0, 1].")
    if mass <= 0.0 or energy <= 0.0:
        raise ValueError("The beta-Y beam map requires positive m_top and e_cm.")

    tau_threshold = 4.0 * mass * mass / (energy * energy)
    if tau_threshold >= 1.0 or beta_coordinate == 1.0:
        return 1.0, 1.0, 0.0
    beta_max = math.sqrt(1.0 - tau_threshold)
    beta = beta_max * beta_coordinate
    one_minus_beta_sq = 1.0 - beta * beta
    tau = tau_threshold / one_minus_beta_sq
    rapidity_width = -math.log(tau)
    rapidity = (rapidity_coordinate - 0.5) * rapidity_width
    if rapidity_coordinate == 0.0:
        x1, x2 = tau, 1.0
    elif rapidity_coordinate == 1.0:
        x1, x2 = 1.0, tau
    else:
        sqrt_tau = math.sqrt(tau)
        x1 = sqrt_tau * math.exp(rapidity)
        x2 = sqrt_tau * math.exp(-rapidity)
    jacobian = (
        beta_max
        * (2.0 * tau_threshold * beta / (one_minus_beta_sq**2))
        * rapidity_width
    )
    return x1, x2, jacobian


@lru_cache(maxsize=32)
def decimal_pi(decimal_digit_precision: int) -> Decimal:
    """Compute pi with Decimal arithmetic using the Gauss-Legendre algorithm."""
    if decimal_digit_precision < 2:
        raise ValueError("Decimal precision must be at least two digits.")
    with localcontext() as context:
        context.prec = decimal_digit_precision + 10
        one = Decimal(1)
        two = Decimal(2)
        a = one
        b = one / two.sqrt()
        t = Decimal(1) / Decimal(4)
        p = one
        for _ in range(max(8, decimal_digit_precision.bit_length() + 3)):
            next_a = (a + b) / two
            next_b = (a * b).sqrt()
            delta = a - next_a
            t -= p * delta * delta
            a, b = next_a, next_b
            p *= two
            if a == b:
                break
        value = (a + b) * (a + b) / (Decimal(4) * t)
    with localcontext() as context:
        context.prec = decimal_digit_precision
        return +value


def decimal_sin_cos(
    value: Decimal, decimal_digit_precision: int
) -> tuple[Decimal, Decimal]:
    """Evaluate sin and cos without a binary-float intermediate."""
    guard_digits = 12
    with localcontext() as context:
        context.prec = decimal_digit_precision + guard_digits
        pi = decimal_pi(context.prec)
        two_pi = Decimal(2) * pi
        reduced = value % two_pi
        if reduced > pi:
            reduced -= two_pi
        elif reduced < -pi:
            reduced += two_pi

        squared = reduced * reduced
        sin_term = reduced
        cos_term = Decimal(1)
        sin_value = sin_term
        cos_value = cos_term
        cutoff = Decimal(1).scaleb(-(context.prec + 2))
        for index in range(1, 2 * context.prec + 50):
            sin_term *= -squared / Decimal((2 * index) * (2 * index + 1))
            cos_term *= -squared / Decimal((2 * index - 1) * (2 * index))
            sin_value += sin_term
            cos_value += cos_term
            if abs(sin_term) <= cutoff and abs(cos_term) <= cutoff:
                break
        else:
            raise ArithmeticError("Decimal sine/cosine series did not converge.")

    with localcontext() as context:
        context.prec = decimal_digit_precision
        return +sin_value, +cos_value


# Every entry is an exact, non-identity SO(3) rotation. These are signed
# three-cycles: no coordinate axis remains fixed, including for fully collinear
# samples. Signed permutations avoid all trigonometric rounding and remain
# exact for both float and Decimal.
_EXACT_ROTATIONS: tuple[RotationMatrix, ...] = (
    ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    ((0, 1, 0), (0, 0, -1), (-1, 0, 0)),
    ((0, -1, 0), (0, 0, 1), (-1, 0, 0)),
    ((0, -1, 0), (0, 0, -1), (1, 0, 0)),
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    ((0, 0, 1), (-1, 0, 0), (0, -1, 0)),
    ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
    ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
)


def exact_rotation_from_xs(xs: Sequence[float]) -> RotationMatrix:
    """Select a stable rotation deterministically from the sample bit pattern."""
    state = 0xCBF29CE484222325
    for value in xs:
        bits = struct.unpack("!Q", struct.pack("!d", float(value)))[0]
        state ^= bits
        state = (state * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return _EXACT_ROTATIONS[state % len(_EXACT_ROTATIONS)]


def rotate_vector(value: Vector, rotation: RotationMatrix) -> Vector:
    components = value.to_list()

    def component(row: tuple[int, int, int]):
        nonzero = [
            (index, coefficient)
            for index, coefficient in enumerate(row)
            if coefficient
        ]
        if len(nonzero) != 1:
            raise ValueError(f"Rotation row is not a signed permutation: {row!r}.")
        index, coefficient = nonzero[0]
        selected = components[index]
        if coefficient == 1:
            return selected
        if coefficient != -1:
            raise ValueError(f"Unexpected signed-permutation entry {coefficient!r}.")
        return selected.copy_negate() if isinstance(selected, Decimal) else -selected

    return Vector(*(component(row) for row in rotation))


def float_values_agree(
    first: float,
    second: float,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> bool:
    if not math.isfinite(first) or not math.isfinite(second):
        return False
    return abs(first - second) <= absolute_tolerance + relative_tolerance * max(
        abs(first), abs(second)
    )


def decimal_values_agree(
    first: Decimal,
    second: Decimal,
    *,
    relative_tolerance: Decimal,
    absolute_tolerance: Decimal,
    decimal_digit_precision: int,
) -> bool:
    if not first.is_finite() or not second.is_finite():
        return False
    with localcontext() as context:
        context.prec = decimal_digit_precision + 8
        return abs(first - second) <= absolute_tolerance + relative_tolerance * max(
            abs(first), abs(second)
        )


def decimal_center_of_mass_energy(
    first: LorentzVector,
    second: LorentzVector,
    decimal_digit_precision: int,
) -> Decimal:
    with localcontext() as context:
        context.prec = decimal_digit_precision + 12
        p1 = [decimal_from_input(value) for value in first.to_list()]
        p2 = [decimal_from_input(value) for value in second.to_list()]
        total = [left + right for left, right in zip(p1, p2)]
        squared = total[0] * total[0] - sum(
            component * component for component in total[1:]
        )
        # Match DY.__init__, which defines e_cm as sqrt(abs(s)) even when phase-
        # space validation is explicitly disabled.
        return +abs(squared).sqrt()


def _parameterize_decimal(
    coordinates: Sequence[Decimal],
    parameterization: str,
    e_cm: Decimal,
    rescaling: Decimal,
    decimal_digit_precision: int,
    origin: Vector | None = None,
) -> tuple[Vector, Decimal]:
    if len(coordinates) != 3:
        raise ValueError(
            "A loop-momentum parameterisation requires three coordinates."
        )
    x, y, z = coordinates
    one = Decimal(1)
    pi = decimal_pi(decimal_digit_precision + 12)

    if parameterization == "cartesian":
        scale = e_cm * rescaling
        values: list[Decimal] = []
        jacobian = Decimal(1)
        for coordinate in (x, y, z):
            angle = (coordinate - Decimal("0.5")) * pi
            sine, cosine = decimal_sin_cos(angle, decimal_digit_precision + 12)
            values.append(scale * sine / cosine)
            jacobian *= scale * pi / (cosine * cosine)
        momentum = Vector(*values)
        if origin is not None:
            momentum += origin
        return momentum, jacobian

    radius = x / (one - x) * e_cm
    theta = Decimal(2) * pi * y
    phi = pi * z
    sin_theta, cos_theta = decimal_sin_cos(theta, decimal_digit_precision + 12)
    sin_phi, cos_phi = decimal_sin_cos(phi, decimal_digit_precision + 12)
    momentum = Vector(
        radius * cos_theta * sin_phi,
        radius * sin_theta * sin_phi,
        radius * cos_phi,
    )
    if origin is not None:
        momentum += origin
    if parameterization == "spherical":
        jacobian = (
            radius
            * radius
            * sin_phi
            * Decimal(2)
            * pi
            * pi
            * e_cm
            / ((one - x) * (one - x))
        )
        return momentum, jacobian
    if parameterization == "log_spherical":
        jacobian = (
            radius
            * radius
            * radius
            * sin_phi
            * Decimal(2)
            * pi
            * pi
            * (one / x + one / (one - x))
        )
        return momentum, jacobian
    raise ValueError(f"Parameterisation {parameterization!r} is not implemented.")


@dataclass(frozen=True)
class HighPrecisionSample:
    loop_momenta: tuple[Vector, ...]
    p1: Vector
    p2: Vector
    z: Decimal
    jacobian: Decimal
    decimal_digit_precision: int
    soft_center: Vector | None = None

    def rotated(self, rotation: RotationMatrix) -> HighPrecisionSample:
        return HighPrecisionSample(
            loop_momenta=tuple(
                rotate_vector(momentum, rotation) for momentum in self.loop_momenta
            ),
            p1=rotate_vector(self.p1, rotation),
            p2=rotate_vector(self.p2, rotation),
            z=self.z,
            jacobian=self.jacobian,
            decimal_digit_precision=self.decimal_digit_precision,
            soft_center=(
                rotate_vector(self.soft_center, rotation)
                if self.soft_center is not None
                else None
            ),
        )

    def soft_mirrored(self, routing: SoftEdgeRouting) -> HighPrecisionSample:
        with localcontext() as context:
            context.prec = self.decimal_digit_precision + 12
            mirrored_momenta = mirror_loop_momenta_for_soft_edge(
                self.loop_momenta,
                self.p1,
                self.p2,
                routing,
                self.soft_center,
            )
            return HighPrecisionSample(
                loop_momenta=mirrored_momenta,
                p1=self.p1,
                p2=self.p2,
                z=self.z,
                # The momentum-space map is an affine reflection and therefore
                # has unit absolute determinant. Keep the original sampling
                # Jacobian instead of converting the mirrored momenta back to xs.
                jacobian=self.jacobian,
                decimal_digit_precision=self.decimal_digit_precision,
                soft_center=self.soft_center,
            )


def build_high_precision_sample(
    xs: Sequence[float],
    *,
    n_loops: int,
    parameterization: str,
    incoming_momenta: tuple[LorentzVector, LorentzVector],
    expects_z: bool,
    expects_beam_fractions: bool,
    rescaling: RealInput,
    decimal_digit_precision: int,
    beam_parameterisation: str = "x1_x2",
    beam_threshold_mass: RealInput | None = None,
    soft_mirror_routing: SoftEdgeRouting | None = None,
    soft_center_resolver: Callable[
        [Sequence[Vector], Vector, Vector, Decimal], Vector
    ]
    | None = None,
) -> HighPrecisionSample:
    """Rebuild the complete integration point from the original coordinates."""
    if decimal_digit_precision < 2:
        raise ValueError("Higher-precision evaluation needs at least two digits.")
    n_k_vars = 3 * n_loops
    expected_dimension = (
        n_k_vars + int(expects_z) + 2 * int(expects_beam_fractions)
    )
    if len(xs) != expected_dimension:
        raise ValueError(
            f"Expected {expected_dimension} integration coordinates, got {len(xs)}."
        )

    with localcontext() as context:
        context.prec = decimal_digit_precision + 12
        coordinates = [decimal_from_input(value) for value in xs]
        e_cm = decimal_center_of_mass_energy(
            incoming_momenta[0], incoming_momenta[1], context.prec
        )
        scale = decimal_from_input(rescaling)

        jacobian = Decimal(1)
        if expects_z:
            x_z = coordinates[n_k_vars]
            z = x_z / (Decimal(1) - x_z)
            jacobian /= (Decimal(1) - x_z) ** 2
        else:
            z = Decimal(1)

        if expects_beam_fractions:
            beam_offset = n_k_vars + int(expects_z)
            if beam_parameterisation == "x1_x2":
                x1 = coordinates[beam_offset]
                x2 = coordinates[beam_offset + 1]
            elif beam_parameterisation == "beta_y":
                if beam_threshold_mass is None:
                    raise ValueError(
                        "The beta-Y beam map requires the ttbar threshold mass."
                    )
                x1, x2, beam_jacobian = parameterize_ttbar_beam_fractions(
                    coordinates[beam_offset],
                    coordinates[beam_offset + 1],
                    m_top=decimal_from_input(beam_threshold_mass),
                    e_cm=e_cm,
                )
                assert isinstance(x1, Decimal)
                assert isinstance(x2, Decimal)
                assert isinstance(beam_jacobian, Decimal)
                jacobian *= beam_jacobian
            else:
                raise ValueError(
                    f"Beam parameterisation {beam_parameterisation!r} is not implemented."
                )
            beam_energy = e_cm * (x1 * x2).sqrt() / Decimal(2)
            zero = Decimal(0)
            p1 = Vector(zero, zero, beam_energy)
            p2 = Vector(zero, zero, -beam_energy)
        else:
            p1 = Vector(
                *(
                    decimal_from_input(value)
                    for value in incoming_momenta[0].spatial().to_list()
                )
            )
            p2 = Vector(
                *(
                    decimal_from_input(value)
                    for value in incoming_momenta[1].spatial().to_list()
                )
            )

        zero = Decimal(0)
        loop_momenta = [Vector(zero, zero, zero) for _ in range(n_loops)]
        soft_center: Vector | None = None
        shifted_routing = (
            soft_mirror_routing
            if soft_mirror_routing is not None
            and soft_mirror_routing.has_external_offset
            else None
        )
        pivot = shifted_routing.pivot_loop_index if shifted_routing else None
        for loop_index in range(n_loops):
            if loop_index == pivot:
                continue
            momentum, loop_jacobian = _parameterize_decimal(
                coordinates[3 * loop_index : 3 * (loop_index + 1)],
                parameterization,
                e_cm,
                scale,
                decimal_digit_precision,
            )
            loop_momenta[loop_index] = momentum
            jacobian *= loop_jacobian

        if shifted_routing is not None:
            if soft_center_resolver is None:
                raise ValueError(
                    f"Shifted soft edge {shifted_routing.label} requires a "
                    "higher-precision soft-centre resolver."
                )
            soft_center = soft_center_resolver(loop_momenta, p1, p2, z)
            pivot_momentum, pivot_jacobian = _parameterize_decimal(
                coordinates[3 * pivot : 3 * (pivot + 1)],
                parameterization,
                e_cm,
                scale,
                decimal_digit_precision,
                origin=soft_center,
            )
            loop_momenta[pivot] = pivot_momentum
            jacobian *= pivot_jacobian

        return HighPrecisionSample(
            loop_momenta=tuple(loop_momenta),
            p1=p1,
            p2=p2,
            z=+z,
            jacobian=+jacobian,
            decimal_digit_precision=decimal_digit_precision,
            soft_center=soft_center,
        )
