from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pydot
from gammaloop import GammaLoopAPI
from symbolica import AtomType, E, Expression, Replacement, S
from symbolica.community.idenso import (
    simplify_color,
    simplify_gamma,
    simplify_metrics,
)
from symbolica.community.vakint import Vakint, VakintEvaluationMethod

BASE_DIR = Path(__file__).resolve().parent
PYGLOOP_DIR = BASE_DIR.parents[3]
RUNS_DIR = BASE_DIR / ".runs"
PUBLISHED_DOT_DIR = BASE_DIR / "generated_dot_files"
PUBLISHED_INTEGRAND_DIR = BASE_DIR / "generated_integrands"
INTEGRAND_FILENAME = "uv_integrated_counterterms_1L.json"
PUBLISHED_JSON_PATH = PUBLISHED_INTEGRAND_DIR / INTEGRAND_FILENAME
PUBLISH_LOCK_PATH = BASE_DIR / ".publish.lock"
GENERATE_TOML = PYGLOOP_DIR / "configs" / "DY" / "generate.toml"
BASE_NAME = "UVIntegratedCT"
PROCESS_SPECS = [
    {
        "process": "g > g",
        "integrand_name": "UVIntegratedCT_GluonSE",
        "dot_filename": "gluon_self_energies_1L.dot",
    },
    {
        "process": "d > d",
        "integrand_name": "UVIntegratedCT_DownSE",
        "dot_filename": "down_self_energies_1L.dot",
    },
    {
        "process": "d d~ > g",
        "integrand_name": "UVIntegratedCT_ddg_vertex",
        "dot_filename": "ddg_vertex_1L.dot",
    },
]
SP_MOM = S("sp_mom", is_linear=True, is_symmetric=True)
LAM = S("lam", is_scalar=True)
SP_UV = S("sp", is_linear=True, is_symmetric=True)


@dataclass(frozen=True)
class RunPaths:
    run_root: Path
    run_id: str
    dot_dir: Path
    integrand_dir: Path
    json_path: Path
    vakint_tmp_dir: Path


def _create_run_paths() -> RunPaths:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix="run_", dir=RUNS_DIR))
    run_id = run_root.name
    dot_dir = run_root / "generated_dot_files"
    integrand_dir = run_root / "generated_integrands"
    vakint_tmp_dir = run_root / "vakint_tmp"
    dot_dir.mkdir(parents=True, exist_ok=True)
    integrand_dir.mkdir(parents=True, exist_ok=True)
    vakint_tmp_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_root=run_root,
        run_id=run_id,
        dot_dir=dot_dir,
        integrand_dir=integrand_dir,
        json_path=integrand_dir / INTEGRAND_FILENAME,
        vakint_tmp_dir=vakint_tmp_dir,
    )


@contextmanager
def _publish_lock():
    PUBLISH_LOCK_PATH.touch(exist_ok=True)
    with PUBLISH_LOCK_PATH.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _replace_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    shutil.copyfile(src, tmp_dest)
    os.replace(tmp_dest, dest)


def _publish_outputs(run_paths: RunPaths) -> None:
    with _publish_lock():
        for process_spec in PROCESS_SPECS:
            _replace_file(
                _dot_path(run_paths, process_spec),
                PUBLISHED_DOT_DIR / process_spec["dot_filename"],
            )
        _replace_file(run_paths.json_path, PUBLISHED_JSON_PATH)


def _strip_quotes(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).replace('"', "")


def _base_node(endpoint: str) -> str:
    endpoint = _strip_quotes(endpoint)
    if ":" in endpoint:
        return endpoint.split(":", 1)[0]
    return endpoint


def _expr(expr: str | None) -> Expression:
    return E(_strip_quotes(expr), default_namespace="gammalooprs")


def _vakint_expr(expr: str) -> Expression:
    return E(expr, default_namespace="vakint")


def _strip_ansi(text: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\x1b" and i + 1 < len(text) and text[i + 1] == "[":
            i += 2
            while i < len(text) and text[i] != "m":
                i += 1
            if i < len(text):
                i += 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _canonicalize_symbolica_display_string(expr: str) -> str:
    return " ".join(_strip_ansi(expr).split())


def _strip_known_namespaces(expr: str) -> str:
    for prefix in (
        "gammalooprs::{}::",
        "python::{}::",
        "vakint::{}::",
        "vakint::{symmetric}::",
        "symbolica::{}::",
    ):
        expr = expr.replace(prefix, "")
    return expr


def _expr_to_string(expr: Expression) -> str:
    expr = _strip_namespaces_structurally(expr)
    return _strip_known_namespaces(
        _canonicalize_symbolica_display_string(expr.to_canonical_string())
    ).replace("+-", "-")


@lru_cache(maxsize=None)
def _get_vakint(temporary_directory: str) -> Vakint:
    return Vakint(
        integral_normalization_factor="MSbar",
        mu_r_sq_symbol=S("mursq"),
        number_of_terms_in_epsilon_expansion=2,
        evaluation_order=[
            VakintEvaluationMethod.new_matad_method(),
            VakintEvaluationMethod.new_fmft_method(),
        ],
        form_exe_path="form",
        python_exe_path="python3",
        temporary_directory=temporary_directory,
    )


def _strip_namespaces_structurally(expr: Expression) -> Expression:
    args__ = S("args__")
    atom_repls = []
    seen = set()

    for sym in expr.get_all_symbols():
        full = sym.get_name()
        if "::" not in full:
            continue
        if full.startswith("symbolica::"):
            continue
        if full in seen:
            continue
        seen.add(full)

        short = full.rsplit("::", 1)[-1]
        old = S(full)
        new = S(short)

        if short.endswith(("_", "__", "___")):
            continue

        expr = expr.replace(
            old(args__),
            new(args__),
            allow_new_wildcards_on_rhs=True,
        )
        atom_repls.append(Replacement(old, new))

    if atom_repls:
        expr = expr.replace_multiple(atom_repls)

    return expr


def _is_external_edge(edge: pydot.Edge) -> bool:
    return any(
        _base_node(endpoint).startswith("ext")
        for endpoint in [str(edge.get_source()), str(edge.get_destination())]
    )


def _edge_port(endpoint: str) -> str | None:
    endpoint = _strip_quotes(endpoint)
    if ":" not in endpoint:
        return None
    return endpoint.split(":", 1)[1]


def _edge_sort_key(edge_id: str) -> tuple[int, str]:
    if str(edge_id).isdigit():
        return int(edge_id), str(edge_id)
    return 10**9, str(edge_id)


def _external_edge_metadata(graph: pydot.Dot) -> list[dict[str, str]]:
    external_edges = []
    for edge in graph.get_edges():
        if not _is_external_edge(edge):
            continue

        endpoints = [str(edge.get_source()), str(edge.get_destination())]
        graph_endpoint = next(
            (
                endpoint
                for endpoint in endpoints
                if not _base_node(endpoint).startswith("ext")
            ),
            endpoints[0],
        )
        external_edges.append(
            {
                "id": _strip_quotes(edge.get("id")),
                "particle": _strip_quotes(edge.get("particle")),
                "port": _edge_port(graph_endpoint) or "",
                "lmb_rep": _strip_quotes(edge.get("lmb_rep")),
            }
        )

    return sorted(external_edges, key=lambda edge: _edge_sort_key(edge["id"]))


def _internal_edge_metadata(graph: pydot.Dot) -> list[dict[str, str]]:
    internal_edges = []
    for edge in graph.get_edges():
        if _is_external_edge(edge):
            continue

        internal_edges.append(
            {
                "id": _strip_quotes(edge.get("id")),
                "particle": _strip_quotes(edge.get("particle")),
                "source_port": _edge_port(str(edge.get_source())) or "",
                "destination_port": _edge_port(str(edge.get_destination())) or "",
                "lmb_rep": _strip_quotes(edge.get("lmb_rep")),
            }
        )

    return sorted(internal_edges, key=lambda edge: _edge_sort_key(edge["id"]))


def _mass_for_particle(particle: str) -> Expression:
    if particle in {"d", "d~", "g", "ghG", "ghG~"}:
        return E("0")
    if particle in {"t", "t~"}:
        return _expr("UFO::MT")
    raise ValueError(
        f"Unsupported particle in UV integrated counterterm graph: {particle}"
    )


def _product_of_numerators(graph: pydot.Dot) -> Expression:
    numerator = _expr(graph.get("num") or "1")
    numerator *= _expr(graph.get("overall_factor_evaluated") or "1")

    for node in graph.get_nodes():
        node_name = _strip_quotes(str(node.get_name()))
        if node_name in {"node", "edge", "graph"}:
            continue
        node_num = node.get("num")
        if node_num:
            numerator *= _expr(node_num)

    for edge in graph.get_edges():
        edge_num = edge.get("num")
        if edge_num:
            numerator *= _expr(edge_num)

    return numerator


def _simplify_numerator(numerator: Expression) -> Expression:
    simplified = simplify_metrics(simplify_gamma(simplify_color(numerator))).expand()
    simplified = _strip_namespaces_structurally(simplified)

    simplified = simplified.replace(
        E("Q(y_,mink(4,x_))") * E("Q(z_,mink(4,x_))"),
        E("sp(y_,z_)"),
        repeat=True,
    )
    simplified = simplified.replace(
        E("Qp(y_,mink(4,x_))") * E("Q(z_,mink(4,x_))"),
        E("spp(qp(y_),z_)"),
        repeat=True,
    )
    simplified = simplified.replace(
        E("Qp(y_,mink(4,x_))") * E("Qp(z_,mink(4,x_))"),
        E("spp(qp(y_),qp(z_))"),
        repeat=True,
    )

    return simplified.expand()


def _propagator_factor(edge: pydot.Edge) -> Expression:
    particle = _strip_quotes(edge.get("particle"))
    momentum = _expr(edge.get("lmb_rep"))
    mass = _mass_for_particle(particle)
    denominator = SP_MOM(momentum, momentum) - mass**2
    return E("1") / denominator


def _route_numerator(expr: Expression, graph: pydot.Dot) -> Expression:
    routed = _canonicalize_symbolica_display_string(str(expr))
    edge_momenta = {
        _strip_quotes(edge.get("id")): _strip_quotes(edge.get("lmb_rep"))
        for edge in graph.get_edges()
        if edge.get("lmb_rep") is not None
    }
    for match in list(expr.match(E("Q(edge_,mink(4,slot_))"))):
        edge_id = match[S("edge_")].to_canonical_string()
        lmb_rep = edge_momenta.get(edge_id)
        if lmb_rep is None:
            continue
        slot = _expr_to_string(match[S("slot_")])
        routed = routed.replace(
            f"Q({edge_id},mink(4,{slot}))",
            f"({lmb_rep.replace('a___', slot)})",
        )

    for match in list(expr.match(E("sp(i_,j_)"))):
        left_id = match[S("i_")].to_canonical_string()
        right_id = match[S("j_")].to_canonical_string()
        if left_id not in edge_momenta or right_id not in edge_momenta:
            continue
        routed = routed.replace(
            f"sp({left_id},{right_id})",
            f"sp_mom({edge_momenta[left_id]},{edge_momenta[right_id]})",
        )

    return _expr(routed).expand()


def _replace_sp_mom_with_linear_sp(expr: Expression) -> Expression:
    expr = _strip_namespaces_structurally(expr)
    return expr.replace(
        E("sp_mom(x___,y___)"),
        SP_UV(S("x___"), S("y___")),
        allow_new_wildcards_on_rhs=True,
    )


def _uv_propagator_factor(edge: pydot.Edge) -> Expression:
    particle = _strip_quotes(edge.get("particle"))
    momentum = _expr(_strip_quotes(edge.get("lmb_rep")))
    loop_part = momentum.replace(_expr("P(x_,a___)"), E("0")).expand()
    external_part = momentum.replace(_expr("K(0,a___)"), E("0")).expand()
    uv_momentum = (loop_part / LAM + external_part).expand()
    mass = _mass_for_particle(particle)
    uv_mass_sq = E("1") / LAM**2 * E("mUV") ** 2 + (mass**2 - E("mUV") ** 2)
    denominator = (SP_UV(uv_momentum, uv_momentum) - uv_mass_sq).expand()
    return (E("1") / denominator).expand()


def _uv_rescale_numerator(expr: Expression) -> Expression:
    expr = _strip_namespaces_structurally(expr)
    expr = expr.replace(
        E("sp_mom(x___,y___)"),
        SP_MOM(S("x___"), S("y___")),
        allow_new_wildcards_on_rhs=True,
    )
    expr = expr.replace(E("K(0,slot_)"), E("K(0,slot_)") / LAM)
    return expr.expand()


def _map_hedge_index(raw_index: str) -> str:
    return str(int(raw_index) + 11)


def _slot_index(slot: Expression) -> str:
    slot = _strip_namespaces_structurally(slot)

    hedge_match = next(iter(slot.match(E("hedge(i_)"))), None)
    if hedge_match is not None:
        return _map_hedge_index(hedge_match[S("i_")].to_canonical_string())

    edge_match = next(iter(slot.match(E("edge(i_,j_)"))), None)
    if edge_match is not None:
        left = int(edge_match[S("i_")].to_canonical_string())
        right = int(edge_match[S("j_")].to_canonical_string())
        return str(1_000_000 + 1_000 * left + right)

    if _is_integer_atom(slot):
        return slot.to_canonical_string()

    raise ValueError(f"Unsupported Lorentz slot: {slot}")


def _expanded_add_terms(expr: Expression) -> list[Expression]:
    expr = expr.expand()
    if bool(expr.is_type(AtomType.Add)):
        return list(expr)
    return [expr]


def _multiplicative_factors(expr: Expression) -> list[Expression]:
    if bool(expr.is_type(AtomType.Mul)):
        return list(expr)
    return [expr]


def _power_base_and_exponent(expr: Expression) -> tuple[Expression, int]:
    match = next(iter(expr.match(E("base_^power_"))), None)
    if match is None:
        return expr, 1
    return match[S("base_")], int(match[S("power_")].to_canonical_string())


def _same_expr(left: Expression, right: Expression) -> bool:
    return _expr_to_string(left.expand()) == _expr_to_string(right.expand())


def _is_integer_atom(expr: Expression) -> bool:
    return expr.to_canonical_string().isdigit()


def _loop_sq() -> Expression:
    return SP_UV(E("K(0,a___)"), E("K(0,a___)"))


def _uv_propagator_power(factor: Expression) -> tuple[int, Expression] | None:
    base, exponent = _power_base_and_exponent(factor)
    if exponent >= 0:
        return None

    base = _strip_namespaces_structurally(base)
    loop_sq = _loop_sq()
    power = -exponent
    if _same_expr(base, loop_sq - E("mUV") ** 2):
        return power, E("1")
    if _same_expr(base, E("mUV") ** 2 - loop_sq):
        return power, E(str((-1) ** power))
    return None


def _vakint_sp_mapping(
    base: Expression,
) -> tuple[Expression, Expression] | None:
    match = next(iter(base.match(E("sp(left_,right_)"))), None)
    if match is None:
        return None

    left = match[S("left_")]
    right = match[S("right_")]
    loop = E("K(0,a___)")
    loop_scaled = (loop / LAM).expand()

    if _same_expr(left, loop_scaled) and _same_expr(right, loop_scaled):
        return E("1") / LAM**2, _vakint_expr("k(1,900)*k(1,900)")
    if _same_expr(left, loop) and _same_expr(right, loop):
        return E("1"), _vakint_expr("k(1,900)*k(1,900)")

    for external_label in range(8):
        external = E(f"P({external_label},a___)")
        vakint_external = external_label + 1
        if (_same_expr(left, loop_scaled) and _same_expr(right, external)) or (
            _same_expr(left, external) and _same_expr(right, loop_scaled)
        ):
            return (
                E("1") / LAM,
                _vakint_expr(f"k(1,901)*p({vakint_external},901)"),
            )
        if (_same_expr(left, loop) and _same_expr(right, external)) or (
            _same_expr(left, external) and _same_expr(right, loop)
        ):
            return E("1"), _vakint_expr(f"k(1,901)*p({vakint_external},901)")

    for left_external_label in range(8):
        left_external = E(f"P({left_external_label},a___)")
        for right_external_label in range(8):
            right_external = E(f"P({right_external_label},a___)")
            if _same_expr(left, left_external) and _same_expr(right, right_external):
                return E("1"), _vakint_expr(
                    f"p({left_external_label + 1},902)*p({right_external_label + 1},902)"
                )
    return None


def _map_vakint_factor(factor: Expression) -> tuple[Expression, Expression]:
    base, exponent = _power_base_and_exponent(factor)
    base = _strip_namespaces_structurally(base)
    power = E(str(exponent))

    sp_mapping = _vakint_sp_mapping(base)
    if sp_mapping is not None:
        prefactor, mapped = sp_mapping
        return prefactor**power, mapped**power

    match = next(iter(base.match(E("K(0,slot_)"))), None)
    if match is not None:
        mapped = _vakint_expr(f"k(1,{_slot_index(match[S('slot_')])})")
        return E("1"), mapped**power

    match = next(iter(base.match(E("P(ext_,slot_)"))), None)
    if match is not None:
        ext = int(match[S("ext_")].to_canonical_string()) + 1
        mapped = _vakint_expr(f"p({ext},{_slot_index(match[S('slot_')])})")
        return E("1"), mapped**power

    match = next(iter(base.match(E("g(mink(4,left_),mink(4,right_))"))), None)
    if match is not None:
        left = _slot_index(match[S("left_")])
        right = _slot_index(match[S("right_")])
        mapped = _vakint_expr(f"g({left},{right})")
        return E("1"), mapped**power

    raise ValueError(f"Unsupported Vakint numerator factor: {factor}")


def _is_vakint_numerator_factor(factor: Expression) -> bool:
    base, _ = _power_base_and_exponent(factor)
    base = _strip_namespaces_structurally(base)
    return (
        next(iter(base.match(E("K(0,slot_)"))), None) is not None
        or next(iter(base.match(E("P(ext_,slot_)"))), None) is not None
        or next(iter(base.match(E("g(mink(4,left_),mink(4,right_))"))), None)
        is not None
        or next(iter(base.match(E("sp(left_,right_)"))), None) is not None
    )


def _uv_term_to_vakint(
    term: Expression,
) -> tuple[Expression, Expression, int]:
    term = _strip_namespaces_structurally(term.factor())
    prefactor = E("1")
    vakint_numerator = E("1")
    propagator_power = None

    for factor in _multiplicative_factors(term):
        uv_propagator = _uv_propagator_power(factor)
        if uv_propagator is not None:
            power, sign = uv_propagator
            propagator_power = power
            prefactor *= sign
            continue

        if _is_vakint_numerator_factor(factor):
            extra_prefactor, mapped_factor = _map_vakint_factor(factor)
            prefactor *= extra_prefactor
            vakint_numerator *= mapped_factor
            continue

        prefactor *= factor

    if propagator_power is None:
        raise ValueError(f"Could not identify UV propagator in term: {term}")

    return prefactor.expand(), vakint_numerator.expand(), propagator_power


def _vakint_input_expr(numerator: Expression, power: int) -> Expression:
    topo = _vakint_expr(f"topo(prop(1,edge(1,1),k(1),muvsq,{power}))")
    return (numerator * topo).expand()


def _vakint_input_string(numerator: Expression, power: int) -> str:
    return f"({_expr_to_string(numerator)})*topo(prop(1,edge(1,1),k(1),muvsq,{power}))"


def _vakint_input_parts(expr: Expression) -> tuple[Expression, list[int]]:
    numerator = E("1")
    powers: list[int] = []
    stripped = _strip_namespaces_structurally(expr)

    for factor in _multiplicative_factors(stripped):
        topo_match = next(iter(factor.match(E("topo(body_)"))), None)
        if topo_match is None:
            numerator *= factor
            continue

        body = topo_match[S("body_")]
        for topo_factor in _multiplicative_factors(body):
            prop_match = next(iter(topo_factor.match(E("prop(a_,b_,c_,d_,n_)"))), None)
            if prop_match is not None:
                powers.append(int(prop_match[S("n_")].to_canonical_string()))

    return numerator.expand(), powers


def _accumulate_counts(
    target: dict[int, int],
    expr: Expression,
    multiplicity: int,
) -> None:
    if not _is_integer_atom(expr):
        return
    index = int(expr.to_canonical_string())
    target[index] = target.get(index, 0) + multiplicity


def _index_signature(
    expr: Expression, include_color: bool
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    lorentz_counts: dict[int, int] = {}
    color_counts: dict[int, int] = {}
    stripped = _strip_namespaces_structurally(expr).expand()

    for factor in _multiplicative_factors(stripped):
        base, exponent = _power_base_and_exponent(factor)
        metric_match = next(iter(base.match(E("g(i_,j_)"))), None)
        if metric_match is not None:
            _accumulate_counts(lorentz_counts, metric_match[S("i_")], exponent)
            _accumulate_counts(lorentz_counts, metric_match[S("j_")], exponent)
            continue

        metric_match = next(
            iter(base.match(E("g(mink(4,left_),mink(4,right_))"))), None
        )
        if metric_match is not None:
            _accumulate_counts(
                lorentz_counts, E(_slot_index(metric_match[S("left_")])), exponent
            )
            _accumulate_counts(
                lorentz_counts, E(_slot_index(metric_match[S("right_")])), exponent
            )
            continue

        momentum_match = next(iter(base.match(E("k(1,i_)"))), None)
        if momentum_match is not None:
            _accumulate_counts(lorentz_counts, momentum_match[S("i_")], exponent)
            continue

        momentum_match = next(iter(base.match(E("p(ext_,i_)"))), None)
        if momentum_match is not None:
            _accumulate_counts(lorentz_counts, momentum_match[S("i_")], exponent)
            continue

        gamma_match = next(
            iter(base.match(E("gamma(left_,right_,mink(4,slot_))"))), None
        )
        if gamma_match is not None:
            _accumulate_counts(
                lorentz_counts, E(_slot_index(gamma_match[S("slot_")])), exponent
            )
            continue

        if not include_color:
            continue

        color_match = next(
            iter(base.match(E("g(coad(8,hedge(i_)),coad(8,hedge(j_)))"))), None
        )
        if color_match is not None:
            _accumulate_counts(color_counts, color_match[S("i_")], exponent)
            _accumulate_counts(color_counts, color_match[S("j_")], exponent)

    lorentz_free = tuple(
        sorted(
            index for index, count in lorentz_counts.items() for _ in range(count % 2)
        )
    )
    color_free = tuple(
        sorted(index for index, count in color_counts.items() for _ in range(count % 2))
    )
    return lorentz_free, color_free


def _term_dimension(expr: Expression) -> int:
    total = 0
    stripped = _strip_namespaces_structurally(expr).expand()

    for factor in _multiplicative_factors(stripped):
        base, exponent = _power_base_and_exponent(factor)
        if (
            next(iter(base.match(E("k(1,i_)"))), None) is not None
            or next(iter(base.match(E("p(ext_,i_)"))), None) is not None
        ):
            total += exponent
            continue
        if next(iter(base.match(E("sp(x_,y_)"))), None) is not None:
            total += 2 * exponent
            continue
        if _same_expr(base, E("MT")) or _same_expr(base, E("mUV")):
            total += exponent
            continue
        if _same_expr(base, E("muvsq")) or _same_expr(base, E("mursq")):
            total += 2 * exponent

    return total


def _vakint_input_term_signature(
    vakint_input: Expression,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], int]:
    numerator, powers = _vakint_input_parts(vakint_input)
    return _index_signature(numerator, include_color=False), _term_dimension(
        numerator
    ) - 2 * sum(powers)


def _mapped_vakint_term_signature(
    prefactor: Expression, vakint_input: Expression
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], int]:
    numerator, powers = _vakint_input_parts(vakint_input)
    combined = (prefactor * numerator).expand()
    return _index_signature(combined, include_color=True), _term_dimension(
        combined
    ) - 2 * sum(powers)


def _analytic_term_signature(
    term: Expression,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], int]:
    return _index_signature(term, include_color=True), _term_dimension(term)


def _check_homogeneity(
    graph_name: str,
    label: str,
    terms: list[Expression],
    signature_fn,
) -> None:
    if not terms:
        return

    reference_signature, reference_dimension = signature_fn(terms[0])
    mismatches = []
    for i, term in enumerate(terms[1:], start=1):
        signature, dimension = signature_fn(term)
        if signature != reference_signature or dimension != reference_dimension:
            mismatches.append((i, signature, dimension, _expr_to_string(term)))

    if mismatches:
        mismatch_lines = [
            f"term[{i}] signature={signature} dimension={dimension}: {term}"
            for i, signature, dimension, term in mismatches
        ]
        raise ValueError(
            f"{graph_name} {label} not homogeneous. "
            f"reference signature={reference_signature}, dimension={reference_dimension}. "
            + " | ".join(mismatch_lines)
        )


def _check_vakint_input_homogeneity(
    graph_name: str,
    vakint_prefactors: list[Expression],
    vakint_inputs: list[Expression],
) -> None:
    if not vakint_inputs:
        return

    # Raw Vakint inputs can mix scalar and tensor numerators; only the
    # prefactor-weighted terms must be homogeneous.
    reference_signature, reference_dimension = _mapped_vakint_term_signature(
        vakint_prefactors[0], vakint_inputs[0]
    )
    mapped_mismatches = []
    for i, (prefactor, term) in enumerate(
        zip(vakint_prefactors[1:], vakint_inputs[1:], strict=False), start=1
    ):
        signature, dimension = _mapped_vakint_term_signature(prefactor, term)
        if signature != reference_signature or dimension != reference_dimension:
            mapped_mismatches.append((
                i,
                signature,
                dimension,
                _expr_to_string(prefactor),
                _expr_to_string(term),
            ))

    if mapped_mismatches:
        mismatch_lines = [
            (
                f"term[{i}] signature={signature} dimension={dimension}: "
                f"prefactor={prefactor}, input={term}"
            )
            for i, signature, dimension, prefactor, term in mapped_mismatches
        ]
        raise ValueError(
            f"{graph_name} full mapped Vakint terms not homogeneous. "
            f"reference signature={reference_signature}, dimension={reference_dimension}. "
            + " | ".join(mismatch_lines)
        )


def _evaluate_vakint_terms(
    uv_counterterm: Expression, temporary_directory: str
) -> tuple[list[Expression], list[str], list[str], list[Expression], Expression, str]:
    vakint = _get_vakint(temporary_directory)
    uv_terms = _expanded_add_terms(
        _strip_namespaces_structurally(_replace_sp_mom_with_linear_sp(uv_counterterm))
    )

    vakint_prefactors: list[Expression] = []
    vakint_input_strings: list[str] = []
    vakint_analytic_terms: list[str] = []
    vakint_input_exprs: list[Expression] = []
    total = E("0")

    for term in uv_terms:
        if _expr_to_string(term) == "0":
            continue

        prefactor, vakint_numerator, power = _uv_term_to_vakint(term)
        vakint_input = _vakint_input_expr(vakint_numerator, power)
        vakint_prefactors.append(prefactor)
        vakint_input_exprs.append(vakint_input)
        vakint_input_strings.append(_vakint_input_string(vakint_numerator, power))

        canonical = vakint.to_canonical(vakint_input, short_form=True)
        tensor_reduced = vakint.tensor_reduce(canonical)
        evaluated = vakint.evaluate_integral(tensor_reduced)
        full_term = (prefactor * evaluated).expand()
        vakint_analytic_terms.append(_expr_to_string(full_term))
        total += full_term

    total = total.expand()
    return (
        vakint_prefactors,
        vakint_input_strings,
        vakint_analytic_terms,
        vakint_input_exprs,
        total,
        _expr_to_string(total),
    )


def _build_integrand(
    graph: pydot.Dot, temporary_directory: str
) -> dict[str, str | list[str]]:
    raw_numerator = _product_of_numerators(graph)
    simplified_numerator = _simplify_numerator(raw_numerator)
    routed_numerator = _route_numerator(simplified_numerator, graph)

    propagators = []
    propagator_factor = E("1")
    uv_propagator_factor = E("1")
    external_edges = _external_edge_metadata(graph)
    internal_edges = _internal_edge_metadata(graph)
    internal_particles = [edge["particle"] for edge in internal_edges]
    for edge_metadata in internal_edges:
        edge = next(
            edge
            for edge in graph.get_edges()
            if _strip_quotes(edge.get("id")) == edge_metadata["id"]
        )
        factor = _propagator_factor(edge)
        uv_factor = _uv_propagator_factor(edge)
        propagators.append(_canonicalize_symbolica_display_string(str(factor)))
        propagator_factor *= factor
        uv_propagator_factor *= uv_factor

    integrand = (routed_numerator * propagator_factor).expand()
    uv_rescaled_numerator = (
        _uv_rescale_numerator(routed_numerator)
        if len(internal_edges) > 2
        else routed_numerator
    )
    uv_rescaled_integrand = (
        E("1") / LAM**4 * uv_rescaled_numerator * uv_propagator_factor
    ).expand()
    uv_counterterm = (
        uv_rescaled_integrand.series(LAM, 0, 0)
        .to_expression()
        .replace(LAM, E("1"))
        .factor()
    )
    uv_counterterm_str = _canonicalize_symbolica_display_string(str(uv_counterterm))
    (
        vakint_prefactors,
        vakint_input_terms,
        vakint_analytic_terms,
        vakint_input_exprs,
        vakint_analytic_total_expr,
        vakint_analytic_total,
    ) = _evaluate_vakint_terms(uv_counterterm, temporary_directory)
    _check_vakint_input_homogeneity(
        _strip_quotes(graph.get_name()),
        vakint_prefactors,
        vakint_input_exprs,
    )
    _check_homogeneity(
        _strip_quotes(graph.get_name()),
        "Vakint analytic total",
        _expanded_add_terms(vakint_analytic_total_expr),
        _analytic_term_signature,
    )

    return {
        "graph_name": _strip_quotes(graph.get_name()),
        "projector": _strip_quotes(graph.get("projector")),
        "external_edges": external_edges,
        "external_particles": [edge["particle"] for edge in external_edges],
        "internal_edges": internal_edges,
        "internal_particles": sorted(set(internal_particles)),
        "internal_particle_sequence": internal_particles,
        "internal_particles_multiset": sorted(internal_particles),
        "raw_numerator": _canonicalize_symbolica_display_string(str(raw_numerator)),
        "simplified_numerator": _canonicalize_symbolica_display_string(
            str(simplified_numerator)
        ),
        "routed_numerator": _canonicalize_symbolica_display_string(
            str(routed_numerator)
        ),
        "propagators": propagators,
        "integrand": _canonicalize_symbolica_display_string(str(integrand)),
        "uv_rescaled_integrand": _canonicalize_symbolica_display_string(
            str(uv_rescaled_integrand)
        ),
        "uv_counterterm": uv_counterterm_str,
        "vakint_input_terms": vakint_input_terms,
        "vakint_analytic_terms": vakint_analytic_terms,
        "vakint_analytic_total": vakint_analytic_total,
    }


def _generate_command(process_spec: dict[str, str]) -> str:
    return (
        f"generate amp {process_spec['process']} | d d~ t t~ g ghG ghG~ [{{{{1}}}} QCD=1] "
        "--only-diagrams "
        "--numerator-grouping group_identical_graphs_up_to_scalar_rescaling "
        f"-p {BASE_NAME} -i {process_spec['integrand_name']}"
    )


def _dot_path(run_paths: RunPaths, process_spec: dict[str, str]) -> Path:
    return run_paths.dot_dir / process_spec["dot_filename"]


def _state_dir(run_paths: RunPaths, process_spec: dict[str, str]) -> Path:
    state_dir = run_paths.run_root / "state" / process_spec["integrand_name"]
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def _generate_graphs(
    process_spec: dict[str, str], run_paths: RunPaths
) -> list[pydot.Dot]:
    api = GammaLoopAPI(str(_state_dir(run_paths, process_spec)))
    api.run(f"set global file {GENERATE_TOML}")
    api.run("import model sm-default.json")
    api.run(_generate_command(process_spec))
    api.run("save state -o")

    dot_str = api.get_dot_files(
        process_id=None, integrand_name=process_spec["integrand_name"]
    )
    _dot_path(run_paths, process_spec).write_text(dot_str, encoding="utf-8")

    graphs = pydot.graph_from_dot_data(dot_str)
    return sorted(graphs, key=lambda graph: _strip_quotes(graph.get_name()))


def main() -> int:
    run_paths = _create_run_paths()
    results = []
    process_counts: list[tuple[str, int]] = []
    for process_spec in PROCESS_SPECS:
        graphs = _generate_graphs(process_spec, run_paths)
        process_counts.append((process_spec["process"], len(graphs)))
        for graph in graphs:
            result = _build_integrand(graph, str(run_paths.vakint_tmp_dir))
            result["process"] = process_spec["process"]
            results.append(result)
    run_paths.json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _publish_outputs(run_paths)

    print(f"Run ID: {run_paths.run_id}")
    for process, count in process_counts:
        print(f"Generated {count} graphs for {process}.")
    for result in results:
        print(
            f"{result['process']} :: {result['graph_name']}: {result['internal_particles']}"
        )
    print(PUBLISHED_JSON_PATH)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
