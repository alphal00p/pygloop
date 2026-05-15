from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

import pydot

from processes.qqbar_nX.qqbar_nX_graphs import (
    edge_particle,
    endpoint_node,
    graph_external_edges,
    graph_internal_edges,
    graph_internal_nodes,
    graph_name,
    is_external_node,
    non_external_endpoint,
    node_int_id,
    node_name,
    strip_quotes,
)

EXACT_XI_AUXILIARY_MODE = "exact_xi_topology"
FAKE_XI_P1_IN_EXTERNAL_ID = 2
FAKE_XI_P2_IN_EXTERNAL_ID = 3
FAKE_XI_P1_OUT_EXTERNAL_ID = 4
FAKE_XI_P2_OUT_EXTERNAL_ID = 5
FAKE_XI_PAIR_IDS = {
    "isr_p1": (FAKE_XI_P1_IN_EXTERNAL_ID, FAKE_XI_P1_OUT_EXTERNAL_ID),
    "isr_p2": (FAKE_XI_P2_IN_EXTERNAL_ID, FAKE_XI_P2_OUT_EXTERNAL_ID),
}
FAKE_XI_EXTERNAL_IDS = frozenset(
    {
        FAKE_XI_P1_IN_EXTERNAL_ID,
        FAKE_XI_P2_IN_EXTERNAL_ID,
        FAKE_XI_P1_OUT_EXTERNAL_ID,
        FAKE_XI_P2_OUT_EXTERNAL_ID,
    }
)


def _endpoint_port(endpoint: str) -> int | None:
    parts = strip_quotes(endpoint).split(":", 1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _edge_id(edge: pydot.Edge) -> int:
    value = edge.get_attributes().get("id")
    if value is None:
        raise ValueError("DOT edge without an id cannot be used in qqbar_nX counterterms.")
    return int(strip_quotes(value))


def _edge_endpoints(edge: pydot.Edge) -> tuple[str, int, str, int]:
    source = endpoint_node(edge.get_source())
    destination = endpoint_node(edge.get_destination())
    source_hedge = _endpoint_port(edge.get_source())
    destination_hedge = _endpoint_port(edge.get_destination())
    if source_hedge is None or destination_hedge is None:
        raise ValueError(
            f"Internal edge {edge.to_string().strip()} does not expose both hedge ports."
        )
    return source, source_hedge, destination, destination_hedge


def _h(index: int) -> str:
    return f"hedge({index})"


def _s(symbol: str) -> str:
    return f"gammalooprs::{symbol}"


def _mink(index: str | int) -> str:
    return f"spenso::mink(4,{index})"


def _bis(index: str | int) -> str:
    if isinstance(index, int):
        return f"spenso::bis(4,{_h(index)})"
    return f"spenso::bis(4,{_s(index)})"


def _cof(hedge: int) -> str:
    return f"spenso::cof(3,{_h(hedge)})"


def _coad(hedge: int) -> str:
    return f"spenso::coad(8,{_h(hedge)})"


def _metric(left: str, right: str) -> str:
    return f"spenso::g({left},{right})"


def _gamma(left: str | int, right: str | int, lorentz_index: str) -> str:
    return f"spenso::gamma({_bis(left)},{_bis(right)},{_mink(lorentz_index)})"


def _p(ext_id: int, lorentz_index: str) -> str:
    return f"Q({ext_id},{_mink(lorentz_index)})"


def _q(edge_id: int, lorentz_index: str) -> str:
    return f"Q({edge_id},{_mink(lorentz_index)})"


def _ose(edge_id: int) -> str:
    return f"OSE({edge_id})"


def _signed_q(edge_id: int, lorentz_index: str, sign: int) -> str:
    if sign == 1:
        return _q(edge_id, lorentz_index)
    if sign == -1:
        return f"(-1*{_q(edge_id, lorentz_index)})"
    raise ValueError(f"Invalid momentum sign {sign}; expected +/-1.")


def _paper_kg1_template(light_edge_id: int) -> str:
    return "(" + _q(light_edge_id, "{index}") + "-" + _p(0, "{index}") + ")"


def _paper_kg2_template(light_edge_id: int) -> str:
    return "(-1*" + _q(light_edge_id, "{index}") + "-" + _p(1, "{index}") + ")"


def _component_p(ext_id: int, component: int) -> str:
    return f"Q({ext_id},spenso::cind({component}))"


def _component_q(edge_id: int, component: int) -> str:
    return f"Q({edge_id},spenso::cind({component}))"


def _minkowski_dot(left: list[str], right: list[str]) -> str:
    return (
        f"{left[0]}*{right[0]}"
        f"-{left[1]}*{right[1]}"
        f"-{left[2]}*{right[2]}"
        f"-{left[3]}*{right[3]}"
    )


def _external_dot(left_ext: int, right_ext: int) -> str:
    return _minkowski_dot(
        [_component_p(left_ext, i) for i in range(4)],
        [_component_p(right_ext, i) for i in range(4)],
    )


def _tensor_dot(left: str, right: str, *, prefix: str) -> str:
    mu = f"{prefix}_mu"
    nu = f"{prefix}_nu"
    return f"{_metric(_mink(mu), _mink(nu))}*{left.format(mu=mu)}*{right.format(nu=nu)}"


def _external_dot_tensor(left_ext: int, right_ext: int, *, prefix: str) -> str:
    return _tensor_dot(
        f"Q({left_ext},{_mink('{mu}')})",
        f"Q({right_ext},{_mink('{nu}')})",
        prefix=prefix,
    )


def _external_zeta_dot_tensor(external_id: int, *, prefix: str) -> str:
    mu = f"{prefix}_mu"
    nu = f"{prefix}_nu"
    return (
        f"{_metric(_mink(mu), _mink(nu))}"
        f"*Q({external_id},{_mink(mu)})"
        f"*(Q(0,{_mink(nu)})+Q(1,{_mink(nu)}))"
    )


def _external_xi_dot(left_ext: int, xi_parameter_names: tuple[str, str, str, str]) -> str:
    return _minkowski_dot(
        [_component_p(left_ext, i) for i in range(4)],
        list(xi_parameter_names),
    )


def _edge_xi_dot(edge_id: int, xi_parameter_names: tuple[str, str, str, str]) -> str:
    return _minkowski_dot(
        [_component_q(edge_id, i) for i in range(4)],
        list(xi_parameter_names),
    )


def _xi_square(xi_parameter_names: tuple[str, str, str, str]) -> str:
    return _minkowski_dot(list(xi_parameter_names), list(xi_parameter_names))


def _edge_external_dot(edge_id: int, external_id: int) -> str:
    return _minkowski_dot(
        [_component_q(edge_id, i) for i in range(4)],
        [_component_p(external_id, i) for i in range(4)],
    )


def _edge_external_dot_tensor(edge_id: int, external_id: int, *, prefix: str) -> str:
    return _tensor_dot(
        f"Q({edge_id},{_mink('{mu}')})",
        f"Q({external_id},{_mink('{nu}')})",
        prefix=prefix,
    )


def _edge_external_dot_spatial(
    edge_id: int,
    external_id: int,
    *,
    energy_sign: int,
) -> str:
    if energy_sign not in {-1, 1}:
        raise ValueError(f"Invalid edge energy sign {energy_sign}; expected +/-1.")
    edge_energy = _ose(edge_id) if energy_sign == 1 else f"(-1*{_ose(edge_id)})"
    return _minkowski_dot(
        [edge_energy] + [_component_q(edge_id, i) for i in range(1, 4)],
        [_component_p(external_id, i) for i in range(4)],
    )


def _edge_external_spatial_projection(edge_id: int, external_id: int) -> str:
    """Euclidean spatial projection used for finite routing-fraction factors.

    These factors are graph-global numerator factors.  They must not introduce
    loop-energy poles, so they intentionally use only cind(1..3) components.
    """
    return (
        f"{_component_q(edge_id, 1)}*{_component_p(external_id, 1)}"
        f"+{_component_q(edge_id, 2)}*{_component_p(external_id, 2)}"
        f"+{_component_q(edge_id, 3)}*{_component_p(external_id, 3)}"
    )


def _external_spatial_norm_squared(external_id: int) -> str:
    return (
        f"{_component_p(external_id, 1)}^2"
        f"+{_component_p(external_id, 2)}^2"
        f"+{_component_p(external_id, 3)}^2"
    )


def _edge_virtuality(edge_id: int) -> str:
    return (
        f"{_component_q(edge_id, 0)}^2"
        f"-{_component_q(edge_id, 1)}^2"
        f"-{_component_q(edge_id, 2)}^2"
        f"-{_component_q(edge_id, 3)}^2"
    )


def _edge_virtuality_tensor(edge_id: int, *, prefix: str) -> str:
    return _tensor_dot(
        f"Q({edge_id},{_mink('{mu}')})",
        f"Q({edge_id},{_mink('{nu}')})",
        prefix=prefix,
    )


def _auxiliary_denominator_spatial_proxy(light_edge_id: int, *, beam: int) -> str:
    if beam not in {1, 2}:
        raise ValueError(f"Unknown auxiliary denominator proxy beam id {beam}.")
    sign = "-1" if beam == 1 else "1"
    longitudinal_fraction = (
        f"({_edge_external_spatial_projection(light_edge_id, 0)})"
        f"*(({_external_spatial_norm_squared(0)})^(-1))"
    )
    return f"({sign})*2*({_external_dot(0, 1)})*({longitudinal_fraction})"


def _opposite_topology_auxiliary_damping(
    *,
    opposite_edge_id: int,
    light_edge_id: int,
    beam: int,
    prefix: str,
) -> str:
    auxiliary_proxy = _auxiliary_denominator_spatial_proxy(light_edge_id, beam=beam)
    if beam == 1:
        # For Delta_1, aux ~= -k_g2^2 in the p1-collinear region.
        denominator = f"(-1*({auxiliary_proxy}))"
    elif beam == 2:
        # For Delta_2, aux ~= +k_g1^2 in the p2-collinear region.
        denominator = auxiliary_proxy
    else:
        raise ValueError(f"Unknown auxiliary damping beam id {beam}.")
    return (
        f"({_edge_virtuality_tensor(opposite_edge_id, prefix=f'{prefix}_virt')})"
        f"*(({denominator})^(-1))"
    )


def _spatial_proxy_auxiliary_factor(
    light_edge_id: int,
    *,
    beam: int,
) -> str:
    """Leading-power fixed-zeta auxiliary denominator without loop energies."""
    return f"(({_auxiliary_denominator_spatial_proxy(light_edge_id, beam=beam)})^(-1))"


def _fake_xi_external_mass_expression() -> str:
    return "0"


def _auxiliary_mass_expression(
    *,
    use_parametric_xi: bool,
    xi_parameter_names: tuple[str, str, str, str],
    xi_external_id: int | None = None,
) -> str:
    if xi_external_id is not None:
        return f"(({_external_dot(xi_external_id, xi_external_id)})^(1/2))"
    return (
        f"(({_auxiliary_square(use_parametric_xi=use_parametric_xi, xi_parameter_names=xi_parameter_names)})"
        "^(1/2))"
    )


def _edge_by_id(graph: pydot.Dot, edge_id: int) -> pydot.Edge:
    matches = [edge for edge in graph.get_edges() if _edge_id(edge) == edge_id]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one edge {edge_id} in {graph_name(graph)}, "
            f"found {len(matches)}."
        )
    return matches[0]


def _next_internal_node_id(graph: pydot.Dot) -> str:
    node_ids = [int(name) for name in graph_internal_nodes(graph)]
    return str(max(node_ids, default=-1) + 1)


def _next_edge_id(graph: pydot.Dot) -> int:
    edge_ids = [_edge_id(edge) for edge in graph.get_edges()]
    return max(edge_ids, default=-1) + 1


def _used_hedge_ids(graph: pydot.Dot) -> set[int]:
    hedge_ids: set[int] = set()
    for edge in graph.get_edges():
        for endpoint in (edge.get_source(), edge.get_destination()):
            port = _endpoint_port(endpoint)
            if port is not None:
                hedge_ids.add(port)
    return hedge_ids


def _next_hedge_id(graph: pydot.Dot) -> int:
    hedge_ids = _used_hedge_ids(graph)
    return max(hedge_ids, default=-1) + 1


def _auxiliary_square(
    *,
    use_parametric_xi: bool,
    xi_parameter_names: tuple[str, str, str, str],
) -> str:
    if use_parametric_xi:
        return _xi_square(xi_parameter_names)
    return (
        f"({_external_dot(0, 0)}"
        f"+2*({_external_dot(0, 1)})"
        f"+{_external_dot(1, 1)})"
    )


def _auxiliary_dot_external(
    external_id: int,
    *,
    use_parametric_xi: bool,
    xi_parameter_names: tuple[str, str, str, str],
) -> str:
    if use_parametric_xi:
        return _external_xi_dot(external_id, xi_parameter_names)
    return f"({_external_dot(external_id, 0)}+{_external_dot(external_id, 1)})"


def _auxiliary_dot_edge(
    edge_id: int,
    *,
    use_parametric_xi: bool,
    xi_parameter_names: tuple[str, str, str, str],
) -> str:
    if use_parametric_xi:
        return _edge_xi_dot(edge_id, xi_parameter_names)
    return f"({_edge_external_dot(edge_id, 0)}+{_edge_external_dot(edge_id, 1)})"


def _auxiliary_dot_spatial_edge(
    edge_id: int,
    *,
    energy_sign: int,
    use_parametric_xi: bool,
    xi_parameter_names: tuple[str, str, str, str],
) -> str:
    if energy_sign not in {-1, 1}:
        raise ValueError(f"Invalid auxiliary edge energy sign {energy_sign}; expected +/-1.")

    edge_energy = _ose(edge_id) if energy_sign == 1 else f"(-1*{_ose(edge_id)})"
    edge_components = [edge_energy] + [_component_q(edge_id, i) for i in range(1, 4)]
    if use_parametric_xi:
        auxiliary_components = list(xi_parameter_names)
    else:
        auxiliary_components = [f"({_component_p(0, i)}+{_component_p(1, i)})" for i in range(4)]
    return _minkowski_dot(edge_components, auxiliary_components)


def _auxiliary_denominator(
    light_edge_id: int,
    *,
    energy_sign: int,
    use_parametric_xi: bool,
    xi_parameter_names: tuple[str, str, str, str],
) -> str:
    edge_aux_dot = _auxiliary_dot_spatial_edge(
        light_edge_id,
        energy_sign=energy_sign,
        use_parametric_xi=use_parametric_xi,
        xi_parameter_names=xi_parameter_names,
    )
    return f"(-2*({edge_aux_dot}))"


def _imaginary_factor(sign: int) -> str:
    if sign == 1:
        return f"1{chr(0x1D456)}"
    if sign == -1:
        return f"-1{chr(0x1D456)}"
    raise ValueError(f"Invalid imaginary factor sign {sign}; expected +/-1.")


def _counterterm_phase_factor(phase: str) -> str:
    normalized = phase.strip().lower().replace(" ", "")
    if normalized in {"", "1", "+1"}:
        return "1"
    if normalized == "-1":
        return "-1"
    if normalized in {"i", "+i", "1i", "+1i"}:
        return _imaginary_factor(1)
    if normalized in {"-i", "-1i"}:
        return _imaginary_factor(-1)
    raise ValueError(
        "qqbar_nX counterterm global phase must be one of 1, -1, i or -i."
    )


def _auxiliary_prefactor(
    *,
    beam: int,
    use_parametric_xi: bool,
    xi_external_id: int | None,
    xi_parameter_names: tuple[str, str, str, str],
    prefix: str,
) -> str:
    if beam not in {1, 2}:
        raise ValueError(f"Unknown auxiliary prefactor beam id {beam}.")
    external_id = 0 if beam == 1 else 1
    if xi_external_id is not None:
        numerator = _external_dot_tensor(
            external_id, xi_external_id, prefix=f"{prefix}_auxpref_xi"
        )
    elif use_parametric_xi:
        numerator = _auxiliary_dot_external(
            external_id,
            use_parametric_xi=use_parametric_xi,
            xi_parameter_names=xi_parameter_names,
        )
    else:
        numerator = _external_zeta_dot_tensor(external_id, prefix=f"{prefix}_auxpref")
    return f"({numerator})*(({_external_dot(0, 1)})^(-1))"


def _routing_fraction_factor(
    structure: LightLineStructure,
    *,
    beam: int,
    spatial_only: bool = False,
    prefix: str = "rf",
) -> str:
    """Convert the paper's gluon-collinear numerator to the pinned k1 routing."""
    if beam == 1:
        if spatial_only:
            numerator = _edge_external_spatial_projection(structure.light_edge_id, 1)
            denominator = _edge_external_spatial_projection(
                structure.p1.gluon_edge_id, 1
            )
        else:
            numerator = _edge_external_dot_tensor(
                structure.light_edge_id, 1, prefix=f"{prefix}_b1_num"
            )
            denominator = _edge_external_dot_spatial(
                structure.p1.gluon_edge_id,
                1,
                energy_sign=structure.p1.gluon_momentum_sign_into_loop,
            )
    elif beam == 2:
        if spatial_only:
            numerator = _edge_external_spatial_projection(structure.light_edge_id, 0)
            denominator = _edge_external_spatial_projection(
                structure.p2.gluon_edge_id, 0
            )
        else:
            numerator = _edge_external_dot_tensor(
                structure.light_edge_id, 0, prefix=f"{prefix}_b2_num"
            )
            denominator = _edge_external_dot_spatial(
                structure.p2.gluon_edge_id,
                0,
                energy_sign=structure.p2.gluon_momentum_sign_into_loop,
            )
    else:
        raise ValueError(f"Unknown routing-fraction beam id {beam}.")
    return f"({numerator})*(({denominator})^(-1))"


def _sanitize_symbol(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", value)


def _beam_projector(
    left_spin: str | int,
    right_spin: str | int,
    prefix: str,
    *,
    beam: int,
    use_parametric_xi: bool = False,
    xi_external_id: int | None = None,
    xi_parameter_names: tuple[str, str, str, str] = ("xi0", "xi1", "xi2", "xi3"),
) -> str:
    if beam not in {1, 2}:
        raise ValueError(f"Unknown beam projector id {beam}.")

    # The massless spin projector must act as the identity on both external
    # spinors: /p1 /p2 u(p1)/(2 p1.p2) = u(p1), and
    # vbar(p2) /p1 /p2/(2 p1.p2) = vbar(p2).  Reversing the order on the
    # anti-quark side would put /p2 directly next to vbar(p2) and annihilate
    # the p2 counterterm.
    first_external = 0
    second_external = 1

    mu1 = f"{prefix}_mu1"
    mu2 = f"{prefix}_mu2"
    spin = f"{prefix}_s"
    if xi_external_id is not None:
        reference_slash = f"{_gamma(spin, right_spin, mu2)}*{_p(xi_external_id, mu2)}"
        denominator = f"(2*({_external_dot(first_external, xi_external_id)}))"
    elif use_parametric_xi:
        reference_slash = _xi_slash(
            spin,
            right_spin,
            f"{prefix}_xi",
            xi_parameter_names,
        )
        denominator = f"(2*({_external_xi_dot(first_external, xi_parameter_names)}))"
    else:
        reference_slash = f"{_gamma(spin, right_spin, mu2)}*{_p(second_external, mu2)}"
        denominator = f"(2*({_external_dot(0, 1)}))"
    return (
        "("
        f"{_gamma(left_spin, spin, mu1)}*{_p(first_external, mu1)}"
        f"*{reference_slash}"
        f"*({denominator})^(-1)"
        ")"
    )


def _xi_slash(
    left_spin: str | int,
    right_spin: str | int,
    prefix: str,
    xi_parameter_names: tuple[str, str, str, str],
) -> str:
    terms = []
    for component, parameter_name in enumerate(xi_parameter_names):
        sign = "" if component == 0 else "-"
        terms.append(
            f"{sign}{parameter_name}"
            f"*{_gamma(left_spin, right_spin, f'{prefix}_{component}')}"
        )
    return f"({'+'.join(terms).replace('+-', '-')})"


def _spin_identity(left_spin: str | int, right_spin: str | int) -> str:
    return _metric(_bis(left_spin), _bis(right_spin))


def _projector_complement(
    left_spin: str | int,
    right_spin: str | int,
    prefix: str,
    *,
    beam: int,
    use_parametric_xi: bool = False,
    xi_external_id: int | None = None,
    xi_parameter_names: tuple[str, str, str, str] = ("xi0", "xi1", "xi2", "xi3"),
) -> str:
    return (
        "("
        f"{_spin_identity(left_spin, right_spin)}"
        f"-{_beam_projector(left_spin, right_spin, prefix, beam=beam, use_parametric_xi=use_parametric_xi, xi_external_id=xi_external_id, xi_parameter_names=xi_parameter_names)}"
        ")"
    )


def _p1_vertex_counterterm_num(
    *,
    external_hedge: int,
    internal_hedge: int,
    gluon_light_hedge: int,
    gluon_edge_id: int,
    gluon_momentum_sign: int,
    gluon_momentum_template: str | None = None,
    prefix: str,
    projector_mode: str,
    use_parametric_xi: bool,
    xi_external_id: int | None,
    xi_parameter_names: tuple[str, str, str, str],
) -> str:
    projector = _beam_projector(
        internal_hedge,
        external_hedge,
        f"{prefix}_p",
        beam=1,
        use_parametric_xi=use_parametric_xi,
        xi_external_id=xi_external_id,
        xi_parameter_names=xi_parameter_names,
    )

    def gluon_momentum(index: str) -> str:
        if gluon_momentum_template is not None:
            return gluon_momentum_template.format(index=index)
        return _signed_q(gluon_edge_id, index, gluon_momentum_sign)

    leading = (
        f"2*({gluon_momentum(_h(gluon_light_hedge))})"
        f"*{_spin_identity(internal_hedge, external_hedge)}"
    )
    if projector_mode == "leading":
        spin_part = f"({leading})"
    elif projector_mode == "full":
        comp = _projector_complement(
            f"{prefix}_a",
            f"{prefix}_b",
            f"{prefix}_pc",
            beam=1,
            use_parametric_xi=use_parametric_xi,
            xi_external_id=xi_external_id,
            xi_parameter_names=xi_parameter_names,
        )
        slash_mu = f"{prefix}_kg_mu"
        subleading = (
            f"{_gamma(internal_hedge, f'{prefix}_a', _h(gluon_light_hedge))}"
            f"*{comp}"
            f"*{_gamma(f'{prefix}_b', external_hedge, slash_mu)}"
            f"*({gluon_momentum(slash_mu)})"
        )
        spin_part = f"({leading}-{subleading})"
    else:
        raise ValueError(f"Unknown qqbar_nX counterterm projector mode '{projector_mode}'.")
    if gluon_momentum_template is None:
        # Convert the DOT edge orientation to the paper's fixed kg1 orientation.
        # The raw DOT may attach the gluon with either arrow relative to the
        # light-line bridge inside the same canonical group.
        orientation_factor = "1" if gluon_momentum_sign == -1 else "-1"
        spin_part = f"({orientation_factor})*({spin_part})"
    color_part = (
        f"spenso::t({_coad(gluon_light_hedge)},{_cof(external_hedge)},"
        f"spenso::dind({_cof(internal_hedge)}))"
    )
    return f"UFO::GC_11*{color_part}*{spin_part}"


def _p2_vertex_counterterm_num(
    *,
    external_hedge: int,
    internal_hedge: int,
    gluon_light_hedge: int,
    gluon_edge_id: int,
    gluon_momentum_sign: int,
    gluon_momentum_template: str | None = None,
    prefix: str,
    projector_mode: str,
    use_parametric_xi: bool,
    xi_external_id: int | None,
    xi_parameter_names: tuple[str, str, str, str],
) -> str:
    projector = _beam_projector(
        external_hedge,
        internal_hedge,
        f"{prefix}_p",
        beam=2,
        use_parametric_xi=use_parametric_xi,
        xi_external_id=xi_external_id,
        xi_parameter_names=xi_parameter_names,
    )

    def gluon_momentum(index: str) -> str:
        if gluon_momentum_template is not None:
            return gluon_momentum_template.format(index=index)
        return _signed_q(gluon_edge_id, index, gluon_momentum_sign)

    leading = (
        f"-2*({gluon_momentum(_h(gluon_light_hedge))})"
        f"*{_spin_identity(external_hedge, internal_hedge)}"
    )
    if projector_mode == "leading":
        spin_part = f"({leading})"
    elif projector_mode == "full":
        comp = _projector_complement(
            f"{prefix}_a",
            f"{prefix}_b",
            f"{prefix}_pc",
            beam=2,
            use_parametric_xi=use_parametric_xi,
            xi_external_id=xi_external_id,
            xi_parameter_names=xi_parameter_names,
        )
        slash_mu = f"{prefix}_kg_mu"
        subleading = (
            f"{_gamma(external_hedge, f'{prefix}_a', _h(gluon_light_hedge))}"
            f"*{comp}"
            f"*{_gamma(f'{prefix}_b', internal_hedge, slash_mu)}"
            f"*({gluon_momentum(slash_mu)})"
        )
        spin_part = f"({leading}-{subleading})"
    else:
        raise ValueError(f"Unknown qqbar_nX counterterm projector mode '{projector_mode}'.")
    if gluon_momentum_template is None:
        # Convert the DOT edge orientation to the paper's fixed kg2 orientation.
        # There is no extra 1/2 in Eq. (13); the finite collinear fraction is
        # carried by the shifted CT topology and the auxiliary prefactor.
        orientation_factor = "-1" if gluon_momentum_sign == -1 else "1"
        spin_part = f"({orientation_factor})*({spin_part})"
    color_part = (
        f"spenso::t({_coad(gluon_light_hedge)},{_cof(internal_hedge)},"
        f"spenso::dind({_cof(external_hedge)}))"
    )
    return f"UFO::GC_11*{color_part}*{spin_part}"


def _light_edge_identity_num(source_hedge: int, destination_hedge: int) -> str:
    return (
        f"{_metric(_cof(source_hedge), f'spenso::dind({_cof(destination_hedge)})')}"
        f"*{_metric(_bis(destination_hedge), _bis(source_hedge))}"
    )


@dataclass(frozen=True)
class LightVertexAttachment:
    vertex: str
    external_edge: pydot.Edge
    external_hedge: int
    gluon_edge: pydot.Edge
    gluon_light_hedge: int
    internal_light_hedge: int

    @property
    def gluon_edge_id(self) -> int:
        return _edge_id(self.gluon_edge)

    @property
    def gluon_momentum_sign_into_loop(self) -> int:
        source = endpoint_node(self.gluon_edge.get_source())
        destination = endpoint_node(self.gluon_edge.get_destination())
        if source == self.vertex:
            return 1
        if destination == self.vertex:
            return -1
        raise ValueError(
            f"Gluon edge {self.gluon_edge.to_string().strip()} is not attached to light vertex {self.vertex}."
        )


@dataclass(frozen=True)
class LightLineStructure:
    p1: LightVertexAttachment
    p2: LightVertexAttachment
    light_edge: pydot.Edge
    light_edge_source_hedge: int
    light_edge_destination_hedge: int

    @property
    def light_edge_id(self) -> int:
        return _edge_id(self.light_edge)


@dataclass
class CountertermGraph:
    graph: pydot.Dot
    original_graph: str
    counterterm: str
    group_id: str | None
    cancelled_edge_id: int
    collinear_gluon_edge_id: int


@dataclass
class CountertermReport:
    original_graphs: list[str] = field(default_factory=list)
    counterterms: list[CountertermGraph] = field(default_factory=list)
    projector_mode: str = "leading"
    denominator_strategy: str = "dummy"
    auxiliary_denominator_mode: str = "global"
    global_phase: str = "1"
    normalization_factor: str = "1"
    uv_inert_dod: int = -100
    use_parametric_xi: bool = False
    xi_parameter_names: tuple[str, str, str, str] = ("xi0", "xi1", "xi2", "xi3")
    xi_default_values: tuple[float, float, float, float] = (1000.0, 0.0, 0.0, 100.0)

    def manifest(self) -> dict[str, Any]:
        return {
            "projector_mode": self.projector_mode,
            "denominator_strategy": self.denominator_strategy,
            "auxiliary_denominator_mode": self.auxiliary_denominator_mode,
            "global_phase": self.global_phase,
            "normalization_factor": self.normalization_factor,
            "uv_inert_dod": self.uv_inert_dod,
            "use_parametric_xi": self.use_parametric_xi,
            "xi_parameter_names": list(self.xi_parameter_names)
            if self.use_parametric_xi
            else [],
            "xi_default_values": list(self.xi_default_values)
            if self.use_parametric_xi
            else [],
            "counts": {
                "original_graphs": len(self.original_graphs),
                "counterterm_graphs": len(self.counterterms),
                "total_graphs": len(self.original_graphs) + len(self.counterterms),
            },
            "counterterms": [
                {
                    "name": graph_name(item.graph),
                    "original_graph": item.original_graph,
                    "counterterm": item.counterterm,
                    "group_id": item.group_id,
                    "cancelled_gluon_edge_id": item.cancelled_edge_id,
                    "collinear_gluon_edge_id": item.collinear_gluon_edge_id,
                }
                for item in self.counterterms
            ],
        }


def _external_edge_for_particle(
    graph: pydot.Dot,
    *,
    vertex: str,
    particle: str,
) -> tuple[pydot.Edge, int] | None:
    for edge in graph_external_edges(graph):
        if edge_particle(edge) != particle:
            continue
        source = endpoint_node(edge.get_source())
        destination = endpoint_node(edge.get_destination())
        if source == vertex:
            hedge = _endpoint_port(edge.get_source())
        elif destination == vertex:
            hedge = _endpoint_port(edge.get_destination())
        else:
            continue
        if hedge is None:
            raise ValueError(
                f"External edge {edge.to_string().strip()} attached to {vertex} has no hedge."
            )
        return edge, hedge
    return None


def _gluon_attachment_for_vertex(
    graph: pydot.Dot,
    *,
    vertex: str,
) -> tuple[pydot.Edge, int]:
    candidates: list[tuple[pydot.Edge, int]] = []
    for edge in graph_internal_edges(graph):
        if edge_particle(edge) != "g":
            continue
        source, source_hedge, destination, destination_hedge = _edge_endpoints(edge)
        if source == vertex:
            candidates.append((edge, source_hedge))
        elif destination == vertex:
            candidates.append((edge, destination_hedge))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one internal gluon bridge attached to {graph_name(graph)}:{vertex}, "
            f"found {len(candidates)}."
        )
    return candidates[0]


def identify_light_line_structure(
    graph: pydot.Dot,
    *,
    light_vertex_id: str = "V_74",
) -> LightLineStructure:
    nodes = graph_internal_nodes(graph)
    light_vertices = [
        name for name, node in nodes.items() if node_int_id(node) == light_vertex_id
    ]
    if len(light_vertices) != 2:
        raise ValueError(
            f"Expected two {light_vertex_id} light vertices in {graph_name(graph)}, "
            f"found {len(light_vertices)}."
        )

    p1_vertex: str | None = None
    p1_external: tuple[pydot.Edge, int] | None = None
    p2_vertex: str | None = None
    p2_external: tuple[pydot.Edge, int] | None = None
    for vertex in light_vertices:
        d_external = _external_edge_for_particle(graph, vertex=vertex, particle="d")
        dbar_external = _external_edge_for_particle(graph, vertex=vertex, particle="d~")
        if d_external is not None:
            p1_vertex = vertex
            p1_external = d_external
        if dbar_external is not None:
            p2_vertex = vertex
            p2_external = dbar_external

    if p1_vertex is None or p1_external is None:
        raise ValueError(f"Could not identify the incoming d vertex in {graph_name(graph)}.")
    if p2_vertex is None or p2_external is None:
        raise ValueError(f"Could not identify the incoming d~ vertex in {graph_name(graph)}.")

    light_edges = [
        edge
        for edge in graph_internal_edges(graph)
        if edge_particle(edge) in {"d", "d~"}
    ]
    if len(light_edges) != 1:
        raise ValueError(
            f"Expected one internal light-quark edge in {graph_name(graph)}, found {len(light_edges)}."
        )
    light_edge = light_edges[0]
    source, source_hedge, destination, destination_hedge = _edge_endpoints(light_edge)
    if {source, destination} != {p1_vertex, p2_vertex}:
        raise ValueError(
            f"The internal light edge in {graph_name(graph)} does not connect the d and d~ vertices."
        )

    if source == p1_vertex:
        p1_internal_hedge = source_hedge
        p2_internal_hedge = destination_hedge
    else:
        p1_internal_hedge = destination_hedge
        p2_internal_hedge = source_hedge

    p1_gluon, p1_gluon_hedge = _gluon_attachment_for_vertex(graph, vertex=p1_vertex)
    p2_gluon, p2_gluon_hedge = _gluon_attachment_for_vertex(graph, vertex=p2_vertex)

    return LightLineStructure(
        p1=LightVertexAttachment(
            vertex=p1_vertex,
            external_edge=p1_external[0],
            external_hedge=p1_external[1],
            gluon_edge=p1_gluon,
            gluon_light_hedge=p1_gluon_hedge,
            internal_light_hedge=p1_internal_hedge,
        ),
        p2=LightVertexAttachment(
            vertex=p2_vertex,
            external_edge=p2_external[0],
            external_hedge=p2_external[1],
            gluon_edge=p2_gluon,
            gluon_light_hedge=p2_gluon_hedge,
            internal_light_hedge=p2_internal_hedge,
        ),
        light_edge=light_edge,
        light_edge_source_hedge=source_hedge,
        light_edge_destination_hedge=destination_hedge,
    )


def _set_node_counterterm_num(graph: pydot.Dot, vertex_name: str, num: str) -> None:
    nodes = graph_internal_nodes(graph)
    node = nodes[vertex_name]
    node.set("num", num)
    node.set("dod", "0")


def _set_edge_counterterm_num(edge: pydot.Edge, num: str) -> None:
    edge.set("num", num)


def _set_edge_denominator_power(edge: pydot.Edge, dod: int) -> None:
    edge.set("dod", str(dod))


def _replace_hedge_symbols(expression: str, replacements: dict[int, str]) -> str:
    if not replacements:
        return expression

    def replace(match: re.Match[str]) -> str:
        hedge_id = int(match.group(1))
        return replacements.get(hedge_id, match.group(0))

    return re.sub(r"\bhedge\((\d+)\)", replace, expression)


def _replace_edge_symbols(expression: str, replacements: dict[int, int]) -> str:
    if not replacements:
        return expression

    def replace_q_or_edge(match: re.Match[str]) -> str:
        function_name = match.group(1)
        edge_id = int(match.group(2))
        return f"{function_name}({replacements.get(edge_id, edge_id)},"

    def replace_ose(match: re.Match[str]) -> str:
        edge_id = int(match.group(1))
        return f"OSE({replacements.get(edge_id, edge_id)})"

    expression = re.sub(r"\b(Q|edge)\((\d+),", replace_q_or_edge, expression)
    return re.sub(r"\bOSE\((\d+)\)", replace_ose, expression)


def _replace_external_p_symbols(expression: str, replacements: dict[int, int]) -> str:
    if not replacements:
        return expression

    def replace(match: re.Match[str]) -> str:
        external_id = int(match.group(1))
        return f"P({replacements.get(external_id, external_id)},"

    return re.sub(r"\bP\((\d+),", replace, expression)


def _replace_node_symbols(expression: str, replacements: dict[int, int]) -> str:
    if not replacements:
        return expression

    def replace(match: re.Match[str]) -> str:
        node_id = int(match.group(1))
        return f"vertex({replacements.get(node_id, node_id)},"

    return re.sub(r"\bvertex\((\d+),", replace, expression)


def _rewrite_attribute_edge_symbols(
    attributes: dict[str, Any],
    replacements: dict[int, int],
) -> None:
    for key, value in list(attributes.items()):
        if isinstance(value, str):
            attributes[key] = _replace_edge_symbols(value, replacements)


def _rewrite_attribute_external_p_symbols(
    attributes: dict[str, Any],
    replacements: dict[int, int],
) -> None:
    for key, value in list(attributes.items()):
        if isinstance(value, str):
            attributes[key] = _replace_external_p_symbols(value, replacements)


def _rewrite_attribute_node_symbols(
    attributes: dict[str, Any],
    replacements: dict[int, int],
) -> None:
    for key, value in list(attributes.items()):
        if isinstance(value, str):
            attributes[key] = _replace_node_symbols(value, replacements)


def _rewrite_attribute_hedge_symbols(
    attributes: dict[str, Any],
    replacements: dict[int, int],
) -> None:
    if not replacements:
        return
    hedge_replacements = {
        old_hedge_id: _h(new_hedge_id)
        for old_hedge_id, new_hedge_id in replacements.items()
    }
    for key, value in list(attributes.items()):
        if isinstance(value, str):
            attributes[key] = _replace_hedge_symbols(value, hedge_replacements)


def _renumber_endpoint_hedge(endpoint: str, replacements: dict[int, int]) -> str:
    port = _endpoint_port(endpoint)
    if port is None:
        return endpoint
    return f"{endpoint_node(endpoint)}:{replacements.get(port, port)}"


def _renumber_endpoint_node(endpoint: str, replacements: dict[int, int]) -> str:
    node = endpoint_node(endpoint)
    if is_external_node(node):
        return endpoint
    try:
        old_node_id = int(node)
    except ValueError:
        return endpoint
    new_node_id = replacements.get(old_node_id, old_node_id)
    port = _endpoint_port(endpoint)
    return f"{new_node_id}:{port}" if port is not None else str(new_node_id)


def _fake_xi_edge_id_mapping(graph: pydot.Dot) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for edge in graph.get_edges():
        old_edge_id = _edge_id(edge)
        if old_edge_id <= 1:
            mapping[old_edge_id] = old_edge_id
        else:
            mapping[old_edge_id] = old_edge_id + 4
    return mapping


def _fake_xi_hedge_mapping(graph: pydot.Dot) -> dict[int, int]:
    hedge_ids: set[int] = set()
    for edge in graph.get_edges():
        for endpoint in (edge.get_source(), edge.get_destination()):
            port = _endpoint_port(endpoint)
            if port is not None:
                hedge_ids.add(port)

    mapping: dict[int, int] = {}
    for hedge_id in hedge_ids:
        if hedge_id <= 1:
            mapping[hedge_id] = hedge_id
        else:
            mapping[hedge_id] = hedge_id + 4
    return mapping


def _fake_xi_external_mapping() -> dict[int, int]:
    return {0: 0, 1: 1, 2: 6, 3: 7, 4: 8}


def _renumber_endpoint_external_node(endpoint: str, replacements: dict[int, int]) -> str:
    node = endpoint_node(endpoint)
    if not is_external_node(node):
        return endpoint
    try:
        old_external_id = int(node.removeprefix("ext"))
    except ValueError:
        return endpoint
    new_external_id = replacements.get(old_external_id, old_external_id)
    port = _endpoint_port(endpoint)
    return f"ext{new_external_id}:{port}" if port is not None else f"ext{new_external_id}"


def _update_edge_name(attributes: dict[str, Any], old_edge_id: int, new_edge_id: int) -> None:
    raw_name = attributes.get("name")
    if raw_name is not None and strip_quotes(raw_name) == f"e{old_edge_id}":
        attributes["name"] = f"e{new_edge_id}"


def _rewrite_fake_xi_graph_attributes(
    attributes: dict[str, Any],
    *,
    edge_mapping: dict[int, int],
    hedge_mapping: dict[int, int],
    external_mapping: dict[int, int],
) -> None:
    _rewrite_attribute_edge_symbols(attributes, edge_mapping)
    _rewrite_attribute_hedge_symbols(attributes, hedge_mapping)
    _rewrite_attribute_external_p_symbols(attributes, external_mapping)


def _add_fake_xi_external_nodes_and_edges(
    graph: pydot.Dot,
    *,
    incoming_vertex: str,
    outgoing_vertex: str,
    dummy: bool,
) -> None:
    common_attributes = {
        "dir": "none",
        "dod": "-2",
        "num": "1",
        "particle": "H",
        "mass": _fake_xi_external_mass_expression(),
    }
    if dummy:
        common_attributes["is_dummy"] = "true"

    for incoming_id, outgoing_id in FAKE_XI_PAIR_IDS.values():
        graph.add_node(pydot.Node(f"ext{incoming_id}", style="invis"))
        graph.add_node(pydot.Node(f"ext{outgoing_id}", style="invis"))
        graph.add_edge(
            pydot.Edge(
                f"ext{incoming_id}",
                f"{incoming_vertex}:{incoming_id}",
                id=str(incoming_id),
                name=f"e{incoming_id}",
                **common_attributes,
            )
        )
        graph.add_edge(
            pydot.Edge(
                f"{outgoing_vertex}:{outgoing_id}",
                f"ext{outgoing_id}",
                id=str(outgoing_id),
                name=f"e{outgoing_id}",
                **common_attributes,
            )
        )


def _delete_fake_xi_external_edges(
    graph: pydot.Dot, edge_ids: set[int] | None = None
) -> None:
    ids_to_delete = set(FAKE_XI_EXTERNAL_IDS if edge_ids is None else edge_ids)
    for edge in list(graph.get_edges()):
        if _edge_id(edge) in ids_to_delete:
            _delete_edge(graph, edge)


def _add_fake_xi_edge(
    graph: pydot.Dot,
    *,
    edge_id: int,
    source: str,
    destination: str,
    dummy: bool,
) -> None:
    attributes: dict[str, Any] = {
        "id": str(edge_id),
        "name": f"e{edge_id}",
        "particle": "H",
        "mass": _fake_xi_external_mass_expression(),
        "dod": "-2",
        "num": "1",
    }
    if dummy:
        attributes["is_dummy"] = "true"
    graph.add_edge(
        pydot.Edge(
            source,
            destination,
            **attributes,
        )
    )


def _split_light_bridge_with_exact_xi_auxiliary(
    graph: pydot.Dot,
    *,
    structure: LightLineStructure,
    counterterm: str,
    use_parametric_xi: bool,
    xi_parameter_names: tuple[str, str, str, str],
    denominator_strategy: str,
    prefix: str,
) -> int:
    """Add the paper auxiliary propagator as a shifted light-line segment.

    The exact Kermanshah denominator is represented by splitting the pinned
    massless light bridge into a massless segment carrying K(0) and a massive
    auxiliary segment.  The fake incoming xi edge enters at the far side of the
    auxiliary segment, and the fake outgoing xi edge leaves at the split node.
    With GammaLoop's incoming-external sign convention the auxiliary segment is
    routed as K(0)-P(xi), directly matching the paper shift once the active
    helper pair is sampled as the auxiliary xi momentum.
    """
    if counterterm not in {"isr_p1", "isr_p2"}:
        raise ValueError(f"Unknown qqbar_nX counterterm '{counterterm}'.")
    incoming_id, outgoing_id = FAKE_XI_PAIR_IDS[counterterm]

    light_edge = _edge_by_id(graph, structure.light_edge_id)
    if counterterm == "isr_p1":
        replaced_gluon = structure.p2.gluon_edge
        keep_replaced_gluon_vertex = structure.p2.vertex
    else:
        replaced_gluon = structure.p1.gluon_edge
        keep_replaced_gluon_vertex = structure.p1.vertex
    source, source_hedge, destination, destination_hedge = _edge_endpoints(light_edge)
    split_node = _next_internal_node_id(graph)
    light_split_hedge = _next_hedge_id(graph)
    auxiliary_split_hedge = light_split_hedge + 1
    auxiliary_edge_id = _next_edge_id(graph)

    light_attributes = dict(light_edge.get_attributes())
    _delete_edge(graph, light_edge)
    graph.add_node(pydot.Node(split_node, dod="0", num="1"))

    # Keep the original light-edge id and orientation on the massless segment
    # so lmb_id=0 continues to pin K(0) to the collinear light momentum.
    light_attributes["id"] = str(structure.light_edge_id)
    _update_edge_name(light_attributes, structure.light_edge_id, structure.light_edge_id)
    light_attributes.pop("mass", None)
    graph.add_edge(
        pydot.Edge(
            f"{source}:{source_hedge}",
            f"{split_node}:{light_split_hedge}",
            **light_attributes,
        )
    )

    graph.add_edge(
        pydot.Edge(
            f"{split_node}:{auxiliary_split_hedge}",
            f"{destination}:{destination_hedge}",
            id=str(auxiliary_edge_id),
            name=f"e{auxiliary_edge_id}",
            particle="g",
            dod="-2",
            num="1",
            mass=_auxiliary_mass_expression(
                use_parametric_xi=use_parametric_xi,
                xi_parameter_names=xi_parameter_names,
                xi_external_id=incoming_id,
            ),
        )
    )

    _delete_fake_xi_external_edges(graph, {incoming_id, outgoing_id})
    _add_fake_xi_edge(
        graph,
        edge_id=incoming_id,
        source=f"ext{incoming_id}",
        destination=f"{destination}:{incoming_id}",
        dummy=False,
    )
    _add_fake_xi_edge(
        graph,
        edge_id=outgoing_id,
        source=f"{split_node}:{outgoing_id}",
        destination=f"ext{outgoing_id}",
        dummy=False,
    )
    _remove_edge_denominator(
        graph,
        replaced_gluon,
        strategy=denominator_strategy,
        keep_vertex=keep_replaced_gluon_vertex,
        prefix=prefix,
    )
    return auxiliary_edge_id


def _attach_exact_xi_pair_around_edge(
    graph: pydot.Dot,
    *,
    edge_id: int,
    counterterm: str,
    use_parametric_xi: bool,
    xi_parameter_names: tuple[str, str, str, str],
    reverse: bool = False,
) -> None:
    """Route the paper auxiliary shift through fake external legs.

    GammaLoop ignores imported lmb_rep when a process is loaded from DOT.  The
    only robust way to change a propagator signature is therefore to change the
    graph connectivity.  The incoming fake xi leg is attached to the source side
    of the special edge and the outgoing fake xi leg to the destination side.
    The p1 and p2 collinear CTs use independent helper pairs:
    Q(2)=Q(4) for the p1 CT and Q(3)=Q(5) for the p2 CT.  The pair is attached
    on the two endpoints of the explicit Kermanshah auxiliary propagator, so
    setting the active helper pair directly to the reference vector xi shifts
    that propagator as k1 -> k1 - xi.  The inactive helper pair stays present
    but dummy.
    """
    if counterterm not in FAKE_XI_PAIR_IDS:
        raise ValueError(f"Unknown qqbar_nX counterterm '{counterterm}'.")
    incoming_id, outgoing_id = FAKE_XI_PAIR_IDS[counterterm]
    target_edge = _edge_by_id(graph, edge_id)
    source, _source_hedge, destination, _destination_hedge = _edge_endpoints(
        target_edge
    )
    incoming_vertex = destination if reverse else source
    outgoing_vertex = source if reverse else destination
    _delete_fake_xi_external_edges(graph, {incoming_id, outgoing_id})
    _add_fake_xi_edge(
        graph,
        edge_id=incoming_id,
        source=f"ext{incoming_id}",
        destination=f"{incoming_vertex}:{incoming_id}",
        dummy=False,
    )
    _add_fake_xi_edge(
        graph,
        edge_id=outgoing_id,
        source=f"{outgoing_vertex}:{outgoing_id}",
        destination=f"ext{outgoing_id}",
        dummy=False,
    )
    target_edge = _edge_by_id(graph, edge_id)
    target_edge.set(
        "mass",
        _auxiliary_mass_expression(
            use_parametric_xi=use_parametric_xi,
            xi_parameter_names=xi_parameter_names,
            xi_external_id=incoming_id,
        ),
    )
    _strip_autogenerated_attributes(target_edge.get_attributes())


def _with_fake_xi_externals(
    graph: pydot.Dot,
    *,
    incoming_vertex: str,
    outgoing_vertex: str,
    dummy: bool,
) -> tuple[pydot.Dot, dict[int, int]]:
    """Reserve external ids 2..5 for two fake xi in/out pairs.

    The physical Higgs external ids are shifted from 2,3,4 to 6,7,8 and all
    original non-initial edge and hedge ids are shifted by four.  This keeps
    external ids contiguous after adding the p1 helper pair Q(2)=Q(4) and the
    p2 helper pair Q(3)=Q(5), with the last physical Higgs Q(8) available as
    the dependent runtime momentum.
    """
    out = copy.deepcopy(graph)
    edge_mapping = _fake_xi_edge_id_mapping(out)
    hedge_mapping = _fake_xi_hedge_mapping(out)
    external_mapping = _fake_xi_external_mapping()

    _rewrite_fake_xi_graph_attributes(
        out.get_attributes(),
        edge_mapping=edge_mapping,
        hedge_mapping=hedge_mapping,
        external_mapping=external_mapping,
    )

    node_specs: list[tuple[str, dict[str, Any]]] = []
    for node in out.get_nodes():
        name = node_name(node)
        attributes = dict(node.get_attributes())
        _rewrite_fake_xi_graph_attributes(
            attributes,
            edge_mapping=edge_mapping,
            hedge_mapping=hedge_mapping,
            external_mapping=external_mapping,
        )
        if is_external_node(name):
            try:
                external_id = int(name.removeprefix("ext"))
                name = f"ext{external_mapping.get(external_id, external_id)}"
            except ValueError:
                pass
        node_specs.append((name, attributes))

    edge_specs: list[tuple[str, str, dict[str, Any]]] = []
    for edge in out.get_edges():
        old_edge_id = _edge_id(edge)
        new_edge_id = edge_mapping[old_edge_id]
        attributes = dict(edge.get_attributes())
        _rewrite_fake_xi_graph_attributes(
            attributes,
            edge_mapping=edge_mapping,
            hedge_mapping=hedge_mapping,
            external_mapping=external_mapping,
        )
        attributes["id"] = str(new_edge_id)
        _update_edge_name(attributes, old_edge_id, new_edge_id)
        source = _renumber_endpoint_external_node(edge.get_source(), external_mapping)
        destination = _renumber_endpoint_external_node(
            edge.get_destination(), external_mapping
        )
        source = _renumber_endpoint_hedge(source, hedge_mapping)
        destination = _renumber_endpoint_hedge(destination, hedge_mapping)
        edge_specs.append((source, destination, attributes))

    for edge in list(out.get_edges()):
        _delete_edge(out, edge)
    for node in list(out.get_nodes()):
        out.del_node(node.get_name())
    for name, attributes in node_specs:
        out.add_node(pydot.Node(name, **attributes))
    for source, destination, attributes in edge_specs:
        out.add_edge(pydot.Edge(source, destination, **attributes))

    _add_fake_xi_external_nodes_and_edges(
        out,
        incoming_vertex=incoming_vertex,
        outgoing_vertex=outgoing_vertex,
        dummy=dummy,
    )
    return out, edge_mapping


def _renumber_nodes_contiguously(graph: pydot.Dot) -> dict[int, int]:
    internal_nodes = graph_internal_nodes(graph)
    numeric_node_ids = sorted(int(name) for name in internal_nodes)
    mapping = {
        old_node_id: new_node_id
        for new_node_id, old_node_id in enumerate(numeric_node_ids)
    }
    if all(old_node_id == new_node_id for old_node_id, new_node_id in mapping.items()):
        return mapping

    _rewrite_attribute_node_symbols(graph.get_attributes(), mapping)
    node_specs: list[tuple[str, dict[str, Any]]] = []
    for name, node in sorted(internal_nodes.items(), key=lambda item: int(item[0])):
        attributes = dict(node.get_attributes())
        _rewrite_attribute_node_symbols(attributes, mapping)
        node_specs.append((str(mapping[int(name)]), attributes))

    edge_specs: list[tuple[str, str, dict[str, Any]]] = []
    for edge in graph.get_edges():
        attributes = dict(edge.get_attributes())
        _rewrite_attribute_node_symbols(attributes, mapping)
        edge_specs.append(
            (
                _renumber_endpoint_node(edge.get_source(), mapping),
                _renumber_endpoint_node(edge.get_destination(), mapping),
                attributes,
            )
        )

    for name in internal_nodes:
        graph.del_node(name)
    for edge in list(graph.get_edges()):
        _delete_edge(graph, edge)
    for name, attributes in node_specs:
        graph.add_node(pydot.Node(name, **attributes))
    for source, destination, attributes in edge_specs:
        graph.add_edge(pydot.Edge(source, destination, **attributes))
    return mapping


def _renumber_hedges_contiguously(graph: pydot.Dot) -> dict[int, int]:
    hedge_ids: list[int] = []
    for edge in graph.get_edges():
        for endpoint in (edge.get_source(), edge.get_destination()):
            port = _endpoint_port(endpoint)
            if port is not None:
                hedge_ids.append(port)
    mapping = {
        old_hedge_id: new_hedge_id
        for new_hedge_id, old_hedge_id in enumerate(sorted(set(hedge_ids)))
    }
    if all(
        old_hedge_id == new_hedge_id for old_hedge_id, new_hedge_id in mapping.items()
    ):
        return mapping

    _rewrite_attribute_hedge_symbols(graph.get_attributes(), mapping)
    for node in graph.get_nodes():
        _rewrite_attribute_hedge_symbols(node.get_attributes(), mapping)

    edge_specs: list[tuple[str, str, dict[str, Any]]] = []
    for edge in graph.get_edges():
        attributes = dict(edge.get_attributes())
        _rewrite_attribute_hedge_symbols(attributes, mapping)
        edge_specs.append(
            (
                _renumber_endpoint_hedge(edge.get_source(), mapping),
                _renumber_endpoint_hedge(edge.get_destination(), mapping),
                attributes,
            )
        )

    for edge in list(graph.get_edges()):
        _delete_edge(graph, edge)
    for source, destination, attributes in edge_specs:
        graph.add_edge(pydot.Edge(source, destination, **attributes))
    return mapping


def _renumber_edges_contiguously(graph: pydot.Dot) -> dict[int, int]:
    edges = sorted(graph.get_edges(), key=_edge_id)
    mapping = {_edge_id(edge): index for index, edge in enumerate(edges)}
    if all(old_edge_id == new_edge_id for old_edge_id, new_edge_id in mapping.items()):
        return mapping

    _rewrite_attribute_edge_symbols(graph.get_attributes(), mapping)
    for node in graph.get_nodes():
        _rewrite_attribute_edge_symbols(node.get_attributes(), mapping)
    for edge in edges:
        _rewrite_attribute_edge_symbols(edge.get_attributes(), mapping)
        edge.set("id", str(mapping[_edge_id(edge)]))
    return mapping


def _compact_edge_ids_preserving_existing_low_ids(graph: pydot.Dot) -> dict[int, int]:
    """Compact edge ids while keeping surviving ids stable whenever possible.

    GammaLoop requires edge ids to span 0..n_edges-1.  Contracting a propagator
    creates one gap; the naive sorted compaction shifts every later edge and can
    move the pinned LMB edges away from the corresponding master graph edges.
    Instead, keep all surviving ids already inside the valid range and move only
    overflow ids into the gaps.
    """
    edges = sorted(graph.get_edges(), key=_edge_id)
    n_edges = len(edges)
    used_targets = {
        _edge_id(edge)
        for edge in edges
        if _edge_id(edge) < n_edges
    }
    free_targets = [
        edge_id for edge_id in range(n_edges) if edge_id not in used_targets
    ]
    mapping: dict[int, int] = {}
    for edge in edges:
        old_edge_id = _edge_id(edge)
        if old_edge_id < n_edges:
            mapping[old_edge_id] = old_edge_id
        else:
            if not free_targets:
                raise ValueError(
                    f"Could not compact edge ids for {graph_name(graph)}: "
                    f"no target slot left for edge {old_edge_id}."
                )
            mapping[old_edge_id] = free_targets.pop(0)

    if all(old_edge_id == new_edge_id for old_edge_id, new_edge_id in mapping.items()):
        return mapping

    _rewrite_attribute_edge_symbols(graph.get_attributes(), mapping)
    for node in graph.get_nodes():
        _rewrite_attribute_edge_symbols(node.get_attributes(), mapping)
    for edge in edges:
        _rewrite_attribute_edge_symbols(edge.get_attributes(), mapping)
        edge.set("id", str(mapping[_edge_id(edge)]))
    return mapping


def _node_num(graph: pydot.Dot, node_name_value: str) -> str:
    node = graph_internal_nodes(graph)[node_name_value]
    return strip_quotes(node.get_attributes().get("num", "1"))


def _edge_num(edge: pydot.Edge) -> str:
    return strip_quotes(edge.get_attributes().get("num", "1"))


def _graph_num(graph: pydot.Dot) -> str:
    return strip_quotes(graph.get_attributes().get("num", "1"))


def _product_expression(parts: list[str]) -> str:
    nontrivial = [part for part in parts if part and part != "1"]
    if not nontrivial:
        return "1"
    return "*".join(f"({part})" for part in nontrivial)


def _collect_full_local_numerator(graph: pydot.Dot) -> str:
    parts = [_graph_num(graph)]
    parts.extend(_node_num(graph, name) for name in graph_internal_nodes(graph))
    # Counterterms reset every edge numerator after collecting the global one;
    # external edge numerators carry spinor phase factors in GammaLoop DOTs.
    parts.extend(_edge_num(edge) for edge in graph.get_edges())
    return _product_expression(parts)


def _set_all_local_numerators_to_one(graph: pydot.Dot) -> None:
    for node in graph_internal_nodes(graph).values():
        node.set("num", "1")
        _strip_autogenerated_attributes(node.get_attributes())
    for edge in graph.get_edges():
        edge.set("num", "1")
        _strip_autogenerated_attributes(edge.get_attributes())


def _strip_autogenerated_attributes(attributes: dict[str, Any]) -> None:
    for key in ("num_autogen", "dod_autogen", "name_autogen"):
        attributes.pop(key, None)


def _strip_graph_autogenerated_attributes(graph: pydot.Dot) -> None:
    _strip_autogenerated_attributes(graph.get_attributes())
    for node in graph.get_nodes():
        _strip_autogenerated_attributes(node.get_attributes())
    for edge in graph.get_edges():
        _strip_autogenerated_attributes(edge.get_attributes())


def _force_counterterm_uv_inert_dod(graph: pydot.Dot, dod: int) -> None:
    """Force the CT's local DOD far negative so GammaLoop sees no UV target."""
    for node in graph_internal_nodes(graph).values():
        node.set("dod", str(dod))
        _strip_autogenerated_attributes(node.get_attributes())
    for edge in graph.get_edges():
        edge.set("dod", str(dod))
        _strip_autogenerated_attributes(edge.get_attributes())


def _validate_counterterm_uv_inert_dod(graph: pydot.Dot, dod: int) -> None:
    expected = str(dod)
    failures: list[str] = []
    for node_id, node in graph_internal_nodes(graph).items():
        value = strip_quotes(node.get_attributes().get("dod", ""))
        if value != expected:
            failures.append(f"node {node_id}: dod={value!r}")
    for edge in graph.get_edges():
        value = strip_quotes(edge.get_attributes().get("dod", ""))
        if value != expected:
            failures.append(f"edge {_edge_id(edge)}: dod={value!r}")
    if failures:
        details = ", ".join(failures[:8])
        if len(failures) > 8:
            details += f", ... ({len(failures)} total)"
        raise ValueError(
            f"Counterterm graph {graph_name(graph)} did not retain UV-inert "
            f"DOD={expected}: {details}"
        )


def _remove_counterterm_vertex_rules(graph: pydot.Dot) -> None:
    for node in graph_internal_nodes(graph).values():
        _remove_node_int_id(node)


def _remove_node_int_id(node: pydot.Node) -> None:
    node.get_attributes().pop("int_id", None)


def _with_replaced_endpoint(endpoint: str, old_node: str, new_node: str) -> str:
    node = endpoint_node(endpoint)
    if node != old_node:
        return endpoint
    port = _endpoint_port(endpoint)
    return f"{new_node}:{port}" if port is not None else new_node


def _delete_edge(graph: pydot.Dot, edge: pydot.Edge) -> None:
    if not graph.del_edge(edge.get_source(), edge.get_destination()):
        raise ValueError(f"Could not delete edge {edge.to_string().strip()}.")


def _contract_edge_into_vertex(
    graph: pydot.Dot,
    edge: pydot.Edge,
    *,
    keep_vertex: str,
    prefix: str,
) -> None:
    source, source_hedge, destination, destination_hedge = _edge_endpoints(edge)
    if keep_vertex == source:
        remove_vertex = destination
    elif keep_vertex == destination:
        remove_vertex = source
    else:
        raise ValueError(
            f"Cannot contract edge {edge.to_string().strip()} into unrelated vertex {keep_vertex}."
        )

    nodes = graph_internal_nodes(graph)
    keep_node = nodes[keep_vertex]
    remove_node = nodes[remove_vertex]

    hedge_replacements = {
        source_hedge: f"{prefix}_contract_h{source_hedge}",
        destination_hedge: f"{prefix}_contract_h{destination_hedge}",
    }
    combined_num = "*".join(
        f"({_replace_hedge_symbols(part, hedge_replacements)})"
        for part in (
            _node_num(graph, keep_vertex),
            _edge_num(edge),
            _node_num(graph, remove_vertex),
        )
    )
    keep_node.set("num", combined_num)
    keep_node.set("dod", "0")
    _strip_autogenerated_attributes(keep_node.get_attributes())
    _remove_node_int_id(keep_node)

    _delete_edge(graph, edge)

    for candidate in list(graph.get_edges()):
        candidate_source = endpoint_node(candidate.get_source())
        candidate_destination = endpoint_node(candidate.get_destination())
        if remove_vertex not in {candidate_source, candidate_destination}:
            continue
        new_source = _with_replaced_endpoint(
            candidate.get_source(), remove_vertex, keep_vertex
        )
        new_destination = _with_replaced_endpoint(
            candidate.get_destination(), remove_vertex, keep_vertex
        )
        attributes = dict(candidate.get_attributes())
        _delete_edge(graph, candidate)
        graph.add_edge(pydot.Edge(new_source, new_destination, **attributes))

    graph.del_node(remove_node.get_name())


def _remove_edge_denominator(
    graph: pydot.Dot,
    edge: pydot.Edge,
    *,
    strategy: str,
    keep_vertex: str,
    prefix: str,
) -> None:
    if strategy == "contract":
        _contract_edge_into_vertex(
            graph,
            edge,
            keep_vertex=keep_vertex,
            prefix=prefix,
        )
        return
    if strategy == "contract_loop":
        source, _, destination, _ = _edge_endpoints(edge)
        if keep_vertex == source:
            loop_vertex = destination
        elif keep_vertex == destination:
            loop_vertex = source
        else:
            raise ValueError(
                f"Cannot contract edge {edge.to_string().strip()} away from unrelated vertex {keep_vertex}."
            )
        _contract_edge_into_vertex(
            graph,
            edge,
            keep_vertex=loop_vertex,
            prefix=prefix,
        )
        return
    if strategy == "dummy":
        # A dummy edge preserves the master topology and loop-momentum routing
        # while removing the edge from the active 3D energy map. GammaLoop's
        # CFF construction still emits the corresponding inverse OSE factor;
        # _make_counterterm_graph cancels that technical factor in the global
        # numerator after final edge renumbering.
        edge.set("is_dummy", "true")
        _strip_autogenerated_attributes(edge.get_attributes())
        return
    if strategy != "dod_zero":
        raise ValueError(f"Unknown qqbar_nX denominator_strategy '{strategy}'.")
    # GammaLoop's 3D denominator topology is built from non-dummy paired
    # internal edges.  Setting dod=0 is therefore not enough to remove this
    # propagator from the generated 3D expression, but marking it as a dummy
    # currently leaves a dangling inverse-energy factor in GammaLoop's CFF
    # production path.  Keep the DOT loadable and record the intended 4D
    # propagator power here until the denominator topology can be steered.
    _set_edge_denominator_power(edge, 0)


def _set_graph_additional_params(
    graph: pydot.Dot,
    xi_parameter_names: tuple[str, str, str, str],
) -> None:
    existing = graph.get_attributes().get("params")
    params: list[str] = []
    if existing is not None:
        params.extend(
            param
            for param in strip_quotes(existing).split(";")
            if param
        )
    for parameter_name in xi_parameter_names:
        if parameter_name not in params:
            params.append(parameter_name)
    graph.set("params", ";".join(params))


def _top_gluon_vertex_for_edge(graph: pydot.Dot, edge: pydot.Edge) -> str:
    nodes = graph_internal_nodes(graph)
    source = endpoint_node(edge.get_source())
    destination = endpoint_node(edge.get_destination())
    if source in nodes and node_int_id(nodes[source]) == "V_137":
        return source
    if destination in nodes and node_int_id(nodes[destination]) == "V_137":
        return destination
    raise ValueError(
        f"Could not identify the top-gluon endpoint of edge {_edge_id(edge)} "
        f"in {graph_name(graph)}."
    )


def _external_higgs_edge_id_for_vertex(graph: pydot.Dot, vertex: str) -> int | None:
    for edge in graph_external_edges(graph):
        if edge_particle(edge) == "h" and non_external_endpoint(edge) == vertex:
            return _edge_id(edge)
    return None


def _paper_heavy_basis_edge_id(graph: pydot.Dot) -> int:
    """Pick a stable heavy-loop basis edge shared by graphs in one topology group."""
    structure = identify_light_line_structure(graph)
    p1_top_vertex = _top_gluon_vertex_for_edge(graph, structure.p1.gluon_edge)
    p2_top_vertex = _top_gluon_vertex_for_edge(graph, structure.p2.gluon_edge)
    top_edges = [
        edge for edge in graph_internal_edges(graph) if edge_particle(edge) in {"t", "t~"}
    ]

    bridge_candidates = [
        edge
        for edge in top_edges
        if {
            endpoint_node(edge.get_source()),
            endpoint_node(edge.get_destination()),
        }
        == {p1_top_vertex, p2_top_vertex}
    ]
    if bridge_candidates:
        return _edge_id(min(bridge_candidates, key=_edge_id))

    p1_side_candidates: list[tuple[int, int, pydot.Edge]] = []
    for edge in top_edges:
        source = endpoint_node(edge.get_source())
        destination = endpoint_node(edge.get_destination())
        if p1_top_vertex not in {source, destination}:
            continue
        other_vertex = destination if source == p1_top_vertex else source
        higgs_edge_id = _external_higgs_edge_id_for_vertex(graph, other_vertex)
        if higgs_edge_id is None:
            continue
        p1_side_candidates.append((higgs_edge_id, _edge_id(edge), edge))
    if p1_side_candidates:
        return _edge_id(max(p1_side_candidates, key=lambda item: (item[0], -item[1]))[2])

    if not top_edges:
        raise ValueError(f"Could not find a heavy-loop edge in {graph_name(graph)}.")
    return _edge_id(min(top_edges, key=_edge_id))


def _pin_paper_loop_momentum_basis(
    graph: pydot.Dot,
    *,
    light_edge_id: int | None = None,
    heavy_edge_id: int | None = None,
) -> None:
    """Pin K(0) to the paper's massless quark-bridge momentum k1."""
    if light_edge_id is None:
        structure = identify_light_line_structure(graph)
        light_edge_id = structure.light_edge_id
    internal_edges = graph_internal_edges(graph)
    light_edges = [edge for edge in internal_edges if _edge_id(edge) == light_edge_id]
    if len(light_edges) != 1:
        raise ValueError(
            f"Could not find the light bridge edge {light_edge_id} to pin as "
            f"lmb_id=0 in {graph_name(graph)}."
        )
    light_edge = light_edges[0]
    if heavy_edge_id is None:
        heavy_edge_id = _paper_heavy_basis_edge_id(graph)
    heavy_loop_edges = [
        edge for edge in internal_edges if _edge_id(edge) == heavy_edge_id
    ]
    if len(heavy_loop_edges) != 1:
        raise ValueError(
            f"Could not find the heavy-loop edge {heavy_edge_id} to pin as "
            f"lmb_id=1 in {graph_name(graph)}."
        )

    heavy_loop_edge = heavy_loop_edges[0]
    for edge in internal_edges:
        edge.get_attributes().pop("lmb_id", None)
    light_edge.set("lmb_id", "0")
    heavy_loop_edge.set("lmb_id", "1")


def minimise_edge_attributes_for_import(graphs: list[pydot.Dot]) -> None:
    """Keep only DOT edge attributes that steer GammaLoop import.

    GammaLoop recomputes momentum signatures such as ``lmb_rep`` when the DOT is
    loaded.  The subtracted DOT should therefore only prescribe the graph
    topology, particle identities, edge ids, explicit LMB pins, and attributes
    that are genuinely part of the custom counterterm representation.
    """

    common_keep = {"id", "particle", "lmb_id", "mass", "is_dummy"}
    counterterm_keep = common_keep | {"num", "dod"}
    for graph in graphs:
        keep = (
            counterterm_keep
            if graph_name(graph).endswith("_ct")
            else common_keep
        )
        for edge in graph.get_edges():
            attributes = edge.get_attributes()
            for key in list(attributes):
                if key not in keep:
                    attributes.pop(key, None)


def _make_counterterm_graph(
    original: pydot.Dot,
    structure: LightLineStructure,
    *,
    heavy_edge_id: int,
    counterterm: str,
    projector_mode: str,
    denominator_strategy: str,
    auxiliary_denominator_mode: str,
    global_phase: str,
    normalization_factor: str,
    uv_inert_dod: int,
    use_parametric_xi: bool,
    xi_parameter_names: tuple[str, str, str, str],
) -> CountertermGraph:
    if counterterm not in {"isr_p1", "isr_p2"}:
        raise ValueError(f"Unknown qqbar_nX counterterm '{counterterm}'.")

    exact_xi_topology = auxiliary_denominator_mode == EXACT_XI_AUXILIARY_MODE
    graph = copy.deepcopy(original)
    original_name = graph_name(original)
    new_name = f"{original_name}_{counterterm}_ct"
    graph.set_name(new_name)
    graph.set("is_group_master", "false")
    if use_parametric_xi:
        _set_graph_additional_params(graph, xi_parameter_names)

    if auxiliary_denominator_mode == "opposite_topology" and use_parametric_xi:
        raise ValueError(
            "counterterms.auxiliary_denominator_mode='opposite_topology' is only "
            "valid for the fixed zeta=p1+p2 subtraction. Use 'global' for the "
            "parametric-xi diagnostic mode until an explicit xi denominator "
            "topology is implemented."
        )

    group_id = original.get_attributes().get("group_id")
    if group_id is not None:
        graph.set("group_id", strip_quotes(group_id))

    if exact_xi_topology:
        graph, fake_xi_edge_mapping = _with_fake_xi_externals(
            graph,
            incoming_vertex=structure.p1.vertex,
            outgoing_vertex=structure.p2.vertex,
            dummy=True,
        )
        heavy_edge_id = fake_xi_edge_mapping.get(heavy_edge_id, heavy_edge_id)

    prefix = _sanitize_symbol(new_name)
    copied_structure = identify_light_line_structure(graph)
    # In exact-xi topology, Q(2)=Q(4) and Q(3)=Q(5) are independent in/out
    # helper pairs for the p1 and p2 collinear CTs.  Both pairs are sampled
    # directly as the paper reference vector xi; the topology inserts an
    # explicit auxiliary propagator carrying k_1-xi, while the local vertex
    # current is kept as the paper's adjacent gluon momentum k_gi.
    xi_external_id = None
    _set_edge_counterterm_num(
        copied_structure.light_edge,
        _light_edge_identity_num(
            copied_structure.light_edge_source_hedge,
            copied_structure.light_edge_destination_hedge,
        ),
    )
    if counterterm == "isr_p1":
        if exact_xi_topology:
            xi_external_id = FAKE_XI_P1_IN_EXTERNAL_ID
        cancelled_edge_id = copied_structure.p2.gluon_edge_id
        aux_prefactor = _auxiliary_prefactor(
            beam=1,
            use_parametric_xi=use_parametric_xi,
            xi_external_id=xi_external_id,
            xi_parameter_names=xi_parameter_names,
            prefix=f"{prefix}_p1",
        )
        aux_denominator = _auxiliary_denominator(
            copied_structure.light_edge_id,
            energy_sign=1,
            use_parametric_xi=use_parametric_xi,
            xi_parameter_names=xi_parameter_names,
        )
        if auxiliary_denominator_mode == "global":
            routing_fraction = _routing_fraction_factor(
                copied_structure, beam=1, prefix=f"{prefix}_p1_rf"
            )
            aux_denominator_factor = f"(({aux_denominator})^(-1))"
            auxiliary_damping = "1"
            remove_cancelled_denominator = True
        elif auxiliary_denominator_mode == "opposite_topology":
            # In the p1-collinear limit with zeta=p1+p2, the paper's auxiliary
            # denominator is leading-power equal to minus the opposite shifted
            # light-line denominator.  Keep that denominator in the CT topology
            # so CFF sees the loop-energy pole instead of hiding it in num.
            # The finite routing fraction is written with spatial projections
            # only; it matches the exact paper projector in the collinear limit
            # without introducing loop-energy denominators in graph num.
            aux_denominator_factor = "-1"
            routing_fraction = _routing_fraction_factor(
                copied_structure, beam=1, spatial_only=True
            )
            auxiliary_damping = _opposite_topology_auxiliary_damping(
                opposite_edge_id=copied_structure.p2.gluon_edge_id,
                light_edge_id=copied_structure.light_edge_id,
                beam=1,
                prefix=f"{prefix}_p1_aux",
            )
            remove_cancelled_denominator = False
        elif exact_xi_topology:
            # The auxiliary denominator is now carried by the DOT topology
            # itself.  With lmb_id=0 pinned to the light bridge, the adjacent
            # gluon momentum entering the paper numerator is an affine
            # function of K(0) and the external beam momentum; no extra spatial
            # routing ratio belongs in graph num.
            aux_denominator_factor = "1"
            routing_fraction = "1"
            auxiliary_damping = "1"
            remove_cancelled_denominator = False
        else:
            # CFF applies on-shell substitutions before this graph-level
            # numerator is algebraically combined with topology denominators.
            # Keep the leading fixed-zeta auxiliary denominator as a purely
            # spatial factor and remove the opposite propagator from the CT
            # topology, avoiding a hidden Q^2/topology-denominator cancellation.
            aux_denominator_factor = _spatial_proxy_auxiliary_factor(
                copied_structure.light_edge_id,
                beam=1,
            )
            routing_fraction = _routing_fraction_factor(
                copied_structure, beam=1, spatial_only=True
            )
            auxiliary_damping = "1"
            remove_cancelled_denominator = True
        graph.set(
            "num",
            f"({_imaginary_factor(-1)})*({routing_fraction})*({aux_prefactor})*({aux_denominator_factor})*({auxiliary_damping})",
        )
        _set_node_counterterm_num(
            graph,
            copied_structure.p1.vertex,
            _p1_vertex_counterterm_num(
                external_hedge=copied_structure.p1.external_hedge,
                internal_hedge=copied_structure.p1.internal_light_hedge,
                gluon_light_hedge=copied_structure.p1.gluon_light_hedge,
                gluon_edge_id=copied_structure.p1.gluon_edge_id,
                gluon_momentum_sign=copied_structure.p1.gluon_momentum_sign_into_loop,
                gluon_momentum_template=(
                    _paper_kg1_template(copied_structure.light_edge_id)
                    if exact_xi_topology
                    else None
                ),
                prefix=prefix,
                projector_mode=projector_mode,
                use_parametric_xi=use_parametric_xi,
                xi_external_id=xi_external_id,
                xi_parameter_names=xi_parameter_names,
            ),
        )
        if remove_cancelled_denominator:
            _remove_edge_denominator(
                graph,
                copied_structure.p2.gluon_edge,
                strategy=denominator_strategy,
                keep_vertex=copied_structure.p2.vertex,
                prefix=prefix,
            )
        collinear_gluon_edge_id = copied_structure.p1.gluon_edge_id
    else:
        if exact_xi_topology:
            xi_external_id = FAKE_XI_P2_IN_EXTERNAL_ID
        cancelled_edge_id = copied_structure.p1.gluon_edge_id
        aux_prefactor = _auxiliary_prefactor(
            beam=2,
            use_parametric_xi=use_parametric_xi,
            xi_external_id=xi_external_id,
            xi_parameter_names=xi_parameter_names,
            prefix=f"{prefix}_p2",
        )
        aux_denominator = _auxiliary_denominator(
            copied_structure.light_edge_id,
            energy_sign=-1,
            use_parametric_xi=use_parametric_xi,
            xi_parameter_names=xi_parameter_names,
        )
        if auxiliary_denominator_mode == "global":
            routing_fraction = _routing_fraction_factor(
                copied_structure, beam=2, prefix=f"{prefix}_p2_rf"
            )
            aux_denominator_factor = f"(({aux_denominator})^(-1))"
            auxiliary_damping = "1"
            remove_cancelled_denominator = True
        elif auxiliary_denominator_mode == "opposite_topology":
            # For the p2-collinear limit the fixed-zeta auxiliary denominator
            # is leading-power equal to the opposite shifted light-line
            # denominator, so no extra sign is required.
            aux_denominator_factor = "1"
            routing_fraction = _routing_fraction_factor(
                copied_structure, beam=2, spatial_only=True
            )
            auxiliary_damping = _opposite_topology_auxiliary_damping(
                opposite_edge_id=copied_structure.p1.gluon_edge_id,
                light_edge_id=copied_structure.light_edge_id,
                beam=2,
                prefix=f"{prefix}_p2_aux",
            )
            remove_cancelled_denominator = False
        elif exact_xi_topology:
            # Same exact-topology convention as in the p1 CT: the denominator
            # sign is fixed by the routed auxiliary edge and the sampled fake
            # external momentum, while the adjacent gluon momentum in the
            # paper numerator is written as an affine function of K(0).
            aux_denominator_factor = "1"
            routing_fraction = "1"
            auxiliary_damping = "1"
            remove_cancelled_denominator = False
        else:
            aux_denominator_factor = _spatial_proxy_auxiliary_factor(
                copied_structure.light_edge_id,
                beam=2,
            )
            routing_fraction = _routing_fraction_factor(
                copied_structure, beam=2, spatial_only=True
            )
            auxiliary_damping = "1"
            remove_cancelled_denominator = True
        graph.set(
            "num",
            f"({_imaginary_factor(1)})*({routing_fraction})*({aux_prefactor})*({aux_denominator_factor})*({auxiliary_damping})",
        )
        _set_node_counterterm_num(
            graph,
            copied_structure.p2.vertex,
            _p2_vertex_counterterm_num(
                external_hedge=copied_structure.p2.external_hedge,
                internal_hedge=copied_structure.p2.internal_light_hedge,
                gluon_light_hedge=copied_structure.p2.gluon_light_hedge,
                gluon_edge_id=copied_structure.p2.gluon_edge_id,
                gluon_momentum_sign=copied_structure.p2.gluon_momentum_sign_into_loop,
                gluon_momentum_template=(
                    _paper_kg2_template(copied_structure.light_edge_id)
                    if exact_xi_topology
                    else None
                ),
                prefix=prefix,
                projector_mode=projector_mode,
                use_parametric_xi=use_parametric_xi,
                xi_external_id=xi_external_id,
                xi_parameter_names=xi_parameter_names,
            ),
        )
        if remove_cancelled_denominator:
            _remove_edge_denominator(
                graph,
                copied_structure.p1.gluon_edge,
                strategy=denominator_strategy,
                keep_vertex=copied_structure.p1.vertex,
                prefix=prefix,
            )
        collinear_gluon_edge_id = copied_structure.p2.gluon_edge_id

    if exact_xi_topology:
        auxiliary_edge_id = _split_light_bridge_with_exact_xi_auxiliary(
            graph,
            structure=copied_structure,
            counterterm=counterterm,
            use_parametric_xi=use_parametric_xi,
            xi_parameter_names=xi_parameter_names,
            denominator_strategy=denominator_strategy,
            prefix=prefix,
        )
        cancelled_edge_id = auxiliary_edge_id

    _renumber_nodes_contiguously(graph)
    _renumber_hedges_contiguously(graph)
    if auxiliary_denominator_mode in {
        "global",
        "spatial_proxy",
        EXACT_XI_AUXILIARY_MODE,
    } and denominator_strategy in {"contract", "contract_loop"}:
        edge_id_mapping = _compact_edge_ids_preserving_existing_low_ids(graph)
    else:
        edge_id_mapping = _renumber_edges_contiguously(graph)
    light_edge_id = edge_id_mapping.get(
        copied_structure.light_edge_id, copied_structure.light_edge_id
    )
    heavy_edge_id = edge_id_mapping.get(heavy_edge_id, heavy_edge_id)
    cancelled_edge_id = edge_id_mapping.get(cancelled_edge_id, cancelled_edge_id)
    collinear_gluon_edge_id = edge_id_mapping.get(
        collinear_gluon_edge_id, collinear_gluon_edge_id
    )
    _pin_paper_loop_momentum_basis(
        graph, light_edge_id=light_edge_id, heavy_edge_id=heavy_edge_id
    )
    full_counterterm_numerator = _collect_full_local_numerator(graph)
    if auxiliary_denominator_mode == "global" and denominator_strategy == "dummy":
        full_counterterm_numerator = _product_expression(
            [_ose(cancelled_edge_id), full_counterterm_numerator]
        )
    phase_factor = _counterterm_phase_factor(global_phase)
    if phase_factor != "1":
        full_counterterm_numerator = _product_expression(
            [phase_factor, full_counterterm_numerator]
        )
    if normalization_factor.strip() not in {"", "1", "+1"}:
        full_counterterm_numerator = _product_expression(
            [normalization_factor, full_counterterm_numerator]
        )
    graph.set("num", full_counterterm_numerator)
    _set_all_local_numerators_to_one(graph)
    _remove_counterterm_vertex_rules(graph)
    _force_counterterm_uv_inert_dod(graph, uv_inert_dod)
    _validate_counterterm_uv_inert_dod(graph, uv_inert_dod)
    _strip_graph_autogenerated_attributes(graph)

    counterterm_group_id = graph.get_attributes().get("group_id")
    return CountertermGraph(
        graph=graph,
        original_graph=original_name,
        counterterm=counterterm,
        group_id=strip_quotes(counterterm_group_id)
        if counterterm_group_id is not None
        else None,
        cancelled_edge_id=cancelled_edge_id,
        collinear_gluon_edge_id=collinear_gluon_edge_id,
    )


def _compact_group_ids(graphs: list[pydot.Dot], report: CountertermReport) -> None:
    old_group_ids: list[str] = []
    for graph in graphs:
        group_id = graph.get_attributes().get("group_id")
        if group_id is None:
            continue
        group_id = strip_quotes(group_id)
        if group_id not in old_group_ids:
            old_group_ids.append(group_id)

    def sort_key(value: str) -> tuple[int, int | str]:
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    mapping = {
        old_group_id: str(new_group_id)
        for new_group_id, old_group_id in enumerate(sorted(old_group_ids, key=sort_key))
    }
    if not mapping:
        return

    grouped_graphs: dict[str, list[pydot.Dot]] = {}
    for graph in graphs:
        group_id = graph.get_attributes().get("group_id")
        if group_id is None:
            continue
        new_group_id = mapping[strip_quotes(group_id)]
        graph.set("group_id", new_group_id)
        grouped_graphs.setdefault(new_group_id, []).append(graph)

    for counterterm in report.counterterms:
        if counterterm.group_id in mapping:
            counterterm.group_id = mapping[counterterm.group_id]

    for group in grouped_graphs.values():
        masters = [
            graph
            for graph in group
            if strip_quotes(graph.get_attributes().get("is_group_master", "false")) == "true"
        ]
        if len(masters) == 1:
            continue
        for index, graph in enumerate(group):
            graph.set("is_group_master", "true" if index == 0 else "false")


def build_isr_counterterm_graphs(
    graphs: list[pydot.Dot],
    *,
    projector_mode: str = "leading",
    denominator_strategy: str = "dummy",
    auxiliary_denominator_mode: str = "global",
    global_phase: str = "1",
    normalization_factor: str = "1",
    uv_inert_dod: int = -100,
    use_parametric_xi: bool = False,
    xi_parameter_names: tuple[str, str, str, str] = ("xi0", "xi1", "xi2", "xi3"),
    xi_default_values: tuple[float, float, float, float] = (1000.0, 0.0, 0.0, 100.0),
) -> tuple[list[pydot.Dot], CountertermReport]:
    if projector_mode not in {"leading", "full"}:
        raise ValueError(
            "qqbar_nX counterterm projector_mode must be either 'leading' or 'full'."
        )
    if denominator_strategy not in {"contract", "contract_loop", "dummy", "dod_zero"}:
        raise ValueError(
            "qqbar_nX counterterm denominator_strategy must be either "
            "'contract', 'contract_loop', 'dummy' or 'dod_zero'."
        )
    if auxiliary_denominator_mode not in {
        "global",
        "opposite_topology",
        "spatial_proxy",
        EXACT_XI_AUXILIARY_MODE,
    }:
        raise ValueError(
            "qqbar_nX counterterm auxiliary_denominator_mode must be either "
            "'global', 'opposite_topology', 'spatial_proxy' or "
            f"'{EXACT_XI_AUXILIARY_MODE}'."
        )
    _counterterm_phase_factor(global_phase)
    report = CountertermReport(
        original_graphs=[graph_name(graph) for graph in graphs],
        projector_mode=projector_mode,
        denominator_strategy=denominator_strategy,
        auxiliary_denominator_mode=auxiliary_denominator_mode,
        global_phase=global_phase,
        normalization_factor=normalization_factor,
        uv_inert_dod=uv_inert_dod,
        use_parametric_xi=use_parametric_xi,
        xi_parameter_names=xi_parameter_names,
        xi_default_values=xi_default_values,
    )
    out: list[pydot.Dot] = []
    for graph in graphs:
        original_copy = copy.deepcopy(graph)
        if use_parametric_xi:
            _set_graph_additional_params(original_copy, xi_parameter_names)
        structure = identify_light_line_structure(graph)
        heavy_edge_id = _paper_heavy_basis_edge_id(graph)
        if auxiliary_denominator_mode == EXACT_XI_AUXILIARY_MODE:
            original_copy, fake_xi_edge_mapping = _with_fake_xi_externals(
                original_copy,
                incoming_vertex=structure.p1.vertex,
                outgoing_vertex=structure.p1.vertex,
                dummy=True,
            )
            heavy_edge_id_for_original = fake_xi_edge_mapping.get(
                heavy_edge_id, heavy_edge_id
            )
        else:
            heavy_edge_id_for_original = heavy_edge_id
        _pin_paper_loop_momentum_basis(
            original_copy, heavy_edge_id=heavy_edge_id_for_original
        )
        _strip_graph_autogenerated_attributes(original_copy)
        out.append(original_copy)
        for counterterm in ("isr_p1", "isr_p2"):
            counterterm_graph = _make_counterterm_graph(
                graph,
                structure,
                heavy_edge_id=heavy_edge_id,
                counterterm=counterterm,
                projector_mode=projector_mode,
                denominator_strategy=denominator_strategy,
                auxiliary_denominator_mode=auxiliary_denominator_mode,
                global_phase=global_phase,
                normalization_factor=normalization_factor,
                uv_inert_dod=uv_inert_dod,
                use_parametric_xi=use_parametric_xi,
                xi_parameter_names=xi_parameter_names,
            )
            report.counterterms.append(counterterm_graph)
            out.append(counterterm_graph.graph)
    _compact_group_ids(out, report)
    return out, report
