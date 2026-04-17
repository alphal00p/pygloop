from __future__ import annotations

import re
import shutil
import subprocess
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pydot
from gammaloop import GammaLoopAPI
from symbolica import E, Expression, S
from symbolica.community.idenso import (
    simplify_color,
    simplify_gamma,
    simplify_metrics,
)

BASE_DIR = Path(__file__).resolve().parent
PYGLOOP_DIR = BASE_DIR.parents[3]
STATE_DIR = BASE_DIR / "_gammaloop_state"
OUTPUT_DIR = BASE_DIR / "generated_dot_files"
DRAW_OUTPUT_DIR = BASE_DIR / "generated_drawings"
DOT_PATH = OUTPUT_DIR / "factorisation_test_ttbar_generated_graphs.dot"
DRAW_PDF_PATH = DRAW_OUTPUT_DIR / "factorisation_test_ttbar_generated_graphs.pdf"
GENERATE_TOML = PYGLOOP_DIR / "configs" / "DY" / "generate.toml"
BASE_NAME = "DY_factorisation_test"
INTEGRAND_NAME = "factorisation_test_ttbar_generated_graphs"
TARGET_COUPLING_POWERS = {"GC_1": 2, "GC_3": 2}
SP_MOM = S("sp_mom", is_linear=True, is_symmetric=True)
XI = S("ξ", is_scalar=True)
LAMBDA = S("λ", is_scalar=True)
X = S("x", is_scalar=True)
Y = S("y", is_scalar=True)
XBAR = S("xbar", is_scalar=True)
PSQ = S("psq", is_scalar=True)
PPERP = S("Pperp")
PBAR = S("Pminus")
# GENERATE_COMMAND = (
#    # "generate xs g g > t t~ | d d~ g t t~ [{{1}} QCD=1] "
#    "generate xs d g e- > e- d | d d~ g t t~ [{{0}} QCD=1 QED=1]"
#    "--only-diagrams "
#    "--numerator-grouping group_identical_graphs_up_to_scalar_rescaling "
#    "--symmetrize-left-right-states true "
#    "--symmetrize-initial-states true "
#    f"-p {BASE_NAME} -i {INTEGRAND_NAME} "
#    "--max-multiplicity-for-fast-cut-filter 99"
# )
# GENERATE_COMMAND = (
#    "generate xs d g e- > e- d | d d~ g a e- e+ "
#    "QED^2==4 QCD^2==2 [{{1}}] "
#    "--only-diagrams "
#    "--numerator-grouping group_identical_graphs_up_to_scalar_rescaling "
#    f"-p {BASE_NAME} -i {INTEGRAND_NAME} "
#    "--max-multiplicity-for-fast-cut-filter 99"
# )

GENERATE_COMMAND = (
    "generate xs d g e- > e- d | d d~ g a e- e+ "
    "QED^2==4 QCD^2==2 [{{1}} QCD=1] "
    "--n-cut-spectators 0 1 "
    "--only-diagrams "
    "--numerator-grouping group_identical_graphs_up_to_scalar_rescaling "
    # "--select-graphs GL97 "
    f"-p {BASE_NAME} -i {INTEGRAND_NAME} "
    "--max-multiplicity-for-fast-cut-filter 0"
)


def _strip_quotes(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).replace('"', "")


def _edge_particle(edge: pydot.Edge) -> str:
    return _strip_quotes(edge.get("particle"))


def _expr(expr: str) -> Expression:
    return E(expr.replace('"', ""), default_namespace="gammalooprs")


def edge_id_int(edge: pydot.Edge) -> int:
    return int(_strip_quotes(edge.get("id")))


def _parse_port(endpoint: str) -> int:
    return int(endpoint.rsplit(":", 1)[1])


def _patch_cut_edge_numerator(edge: pydot.Edge) -> str:
    if _edge_particle(edge) == "d":
        return (
            f"Q({edge_id_int(edge)},spenso::mink(4,mu))*"
            f"spenso::gamma(spenso::bis(4,hedge({_parse_port(edge.get_destination())})),"
            f"spenso::bis(4,hedge({_parse_port(edge.get_source())})),spenso::mink(4,mu))*"
            f"spenso::g(spenso::dind(spenso::cof(3,hedge({_parse_port(edge.get_destination())}))),"
            f"spenso::cof(3,hedge({_parse_port(edge.get_source())})))"
        )
    if _edge_particle(edge) == "d~":
        return (
            f"Q({edge_id_int(edge)},spenso::mink(4,rho))*"
            f"spenso::gamma(spenso::bis(4,hedge({_parse_port(edge.get_source())})),"
            f"spenso::bis(4,hedge({_parse_port(edge.get_destination())})),spenso::mink(4,rho))*"
            f"spenso::g(spenso::dind(spenso::cof(3,hedge({_parse_port(edge.get_source())}))),"
            f"spenso::cof(3,hedge({_parse_port(edge.get_destination())})))"
        )
    if _edge_particle(edge) == "e-":
        return (
            f"Q({edge_id_int(edge)},spenso::mink(4,mu))*"
            f"spenso::gamma(spenso::bis(4,hedge({_parse_port(edge.get_destination())})),"
            f"spenso::bis(4,hedge({_parse_port(edge.get_source())})),spenso::mink(4,mu))"
        )
    if _edge_particle(edge) == "e+":
        return (
            f"Q({edge_id_int(edge)},spenso::mink(4,rho))*"
            f"spenso::gamma(spenso::bis(4,hedge({_parse_port(edge.get_source())})),"
            f"spenso::bis(4,hedge({_parse_port(edge.get_destination())})),spenso::mink(4,rho))"
        )
    if _edge_particle(edge) == "g":
        return (
            f"-spenso::g(spenso::mink(4,hedge({_parse_port(edge.get_destination())})),"
            f"spenso::mink(4,hedge({_parse_port(edge.get_source())})))*"
            f"spenso::g(spenso::coad(8,hedge({_parse_port(edge.get_source())})),"
            f"spenso::coad(8,hedge({_parse_port(edge.get_destination())})))"
        )
    raise ValueError(f"Unsupported cut-edge particle '{_edge_particle(edge)}'.")


def _momentum_sort_key(momentum: str) -> tuple[int, int]:
    kind = 0 if momentum.startswith("K(") else 1
    match = re.search(r"\((\d+)\)", momentum)
    index = int(match.group(1)) if match is not None else -1
    return (kind, index)


def _propagator_from_lmb_rep(lmb_rep: str) -> str:
    momenta = re.findall(r"[KP]\(\d+,a___\)", lmb_rep)
    cleaned = sorted(
        {momentum.replace(",a___", "") for momentum in momenta},
        key=_momentum_sort_key,
    )
    if len(cleaned) == 0:
        raise ValueError(
            f"Could not parse propagator momentum from lmb_rep='{lmb_rep}'."
        )
    return f"1/sq4({'+'.join(cleaned)})"


def _momentum_from_lmb_rep(lmb_rep: str) -> Expression:
    return _expr(lmb_rep.replace(",a___", ""))


def _incoming_particles_from_generate_command(command: str) -> list[str]:
    match = re.search(r"\bgenerate\b\s+\w+\s+(.*?)\s*>\s*.*?\|", command)
    if match is None:
        raise ValueError(
            f"Could not parse incoming particles from command '{command}'."
        )
    return match.group(1).split()


def _edge_dod(edge: pydot.Edge) -> int:
    dod = _strip_quotes(edge.get("dod"))
    if not dod:
        raise ValueError(f"Edge '{edge}' is missing a dod attribute.")
    return int(dod)


def _external_momentum_label(edge: pydot.Edge) -> str | None:
    match = re.fullmatch(r"P\((\d+),a___\)", _strip_quotes(edge.get("lmb_rep")))
    if match is None:
        return None
    return f"P({match.group(1)})"


def _graph_initial_state_momentum_assignments(
    graph: pydot.Dot,
) -> list[tuple[str, str]]:
    assignments: list[tuple[str, str]] = []
    for edge in graph.get_edges():
        if abs(_edge_dod(edge)) != 2:
            continue
        momentum = _external_momentum_label(edge)
        if momentum is None:
            continue
        assignments.append((_edge_particle(edge), momentum))
    return sorted(assignments, key=lambda item: _momentum_sort_key(item[1]))


def _parton_momentum_relation(
    graphs: list[pydot.Dot], command: str
) -> tuple[list[tuple[str, str]], str]:
    incoming_particles = _incoming_particles_from_generate_command(command)
    if len(graphs) == 0:
        raise ValueError(
            "Cannot derive initial-state momenta from an empty graph list."
        )

    reference_assignments = _graph_initial_state_momentum_assignments(graphs[0])
    if len(reference_assignments) != len(incoming_particles):
        raise ValueError(
            "Could not match the graph's external P(i) assignments to the incoming process definition."
        )
    if [particle for particle, _ in reference_assignments] != incoming_particles:
        raise ValueError(
            "The graph's incoming external particles do not match the generate command."
        )

    for graph in graphs[1:]:
        assignments = _graph_initial_state_momentum_assignments(graph)
        if assignments != reference_assignments:
            raise ValueError(
                "Incoming external momentum assignments are not consistent across graphs."
            )

    incoming_partons = Counter(_incoming_parton_labels(command))
    parton_assignments: list[tuple[str, str]] = []
    for particle, momentum in reference_assignments:
        if incoming_partons[particle] == 0:
            continue
        parton_assignments.append((particle, momentum))
        incoming_partons[particle] -= 1

    missing_partons = list(incoming_partons.elements())
    if missing_partons:
        raise ValueError(
            "Failed to derive initial-state momenta for partons: "
            + ", ".join(missing_partons)
        )

    momentum_sum = " + ".join(momentum for _particle, momentum in parton_assignments)
    return parton_assignments, f"P = {momentum_sum}"


def _solved_parton_momentum_substitution(
    parton_assignments: list[tuple[str, str]],
) -> tuple[str, Expression, str]:
    if len(parton_assignments) == 0:
        raise ValueError(
            "Cannot solve momentum conservation without initial-state partons."
        )

    ordered_assignments = sorted(
        parton_assignments,
        key=lambda item: _momentum_sort_key(item[1]),
    )
    solved_momentum = ordered_assignments[0][1]
    substitution = _expr("P")
    for _particle, momentum in ordered_assignments[1:]:
        substitution -= _expr(momentum)
    return (
        solved_momentum,
        substitution,
        f"{solved_momentum} = {_display_expression(substitution)}",
    )


def _graph_integrand(graph: pydot.Dot) -> str:
    factors: list[str] = []
    overall_factor = graph.get("overall_factor_evaluated")
    if overall_factor:
        factors.append(overall_factor.replace('"', ""))

    graph_num = graph.get("num")
    if graph_num and graph_num.replace('"', "") != "1":
        factors.append(graph_num.replace('"', ""))

    for node in graph.get_nodes():
        if node.get_name() in {"edge", "node"}:
            continue
        node_num = node.get("num")
        if node_num:
            factors.append(node_num.replace('"', ""))

    for edge in graph.get_edges():
        if edge.get("is_cut") is not None:
            factors.append(_patch_cut_edge_numerator(edge))
            continue

        edge_num = edge.get("num")
        if edge_num:
            factors.append(edge_num.replace('"', ""))
        factors.append(_propagator_from_lmb_rep(edge.get("lmb_rep").replace('"', "")))

    return "\n  * ".join([f"({factor})" for factor in factors])


def _graph_simplified_numerator_expr(graph: pydot.Dot) -> Expression:
    numerator = E("1")
    overall_factor = _strip_quotes(graph.get("overall_factor_evaluated"))
    symmetry_factor = _expr(overall_factor) if overall_factor else E("1")

    graph_num = _strip_quotes(graph.get("num"))
    if graph_num and graph_num != "1":
        numerator *= _expr(graph_num)

    for node in graph.get_nodes():
        if node.get_name() in {"edge", "node"}:
            continue
        node_num = node.get("num")
        if node_num:
            numerator *= _expr(node_num)

    for edge in graph.get_edges():
        if edge.get("is_cut") is not None:
            numerator *= _expr(_patch_cut_edge_numerator(edge))
            continue

        edge_num = edge.get("num")
        if edge_num:
            numerator *= _expr(edge_num)

    simplified = E(
        str(simplify_metrics(simplify_gamma(simplify_color(numerator))).expand())
    )
    simplified = simplified.replace(
        E("Q(y_,mink(4,x_))") * E("Q(z_,mink(4,x_))"),
        E("sp(y_,z_)"),
        repeat=True,
    )
    return (symmetry_factor * simplified).expand()


def _graph_edge_momenta(graph: pydot.Dot) -> dict[int, Expression]:
    edge_momenta: dict[int, Expression] = {}
    for edge in graph.get_edges():
        lmb_rep = edge.get("lmb_rep")
        if lmb_rep is None:
            continue
        edge_momenta[edge_id_int(edge)] = _momentum_from_lmb_rep(_strip_quotes(lmb_rep))
    return edge_momenta


def _substitute_sp_momenta(expr: Expression, graph: pydot.Dot) -> Expression:
    edge_momenta = _graph_edge_momenta(graph)
    left_index = S("left_index_")
    right_index = S("right_index_")
    sp_pattern = E("sp(left_index_,right_index_)")

    def _replace_scalar_product(
        matches: dict[Expression, Expression],
    ) -> Expression:
        left_label = matches[left_index]
        right_label = matches[right_index]

        try:
            left_edge_id = int(str(left_label))
            right_edge_id = int(str(right_label))
        except ValueError:
            return SP_MOM(left_label, right_label)

        return SP_MOM(
            edge_momenta.get(left_edge_id, left_label),
            edge_momenta.get(right_edge_id, right_label),
        )

    return expr.replace(sp_pattern, _replace_scalar_product).expand()


def _graph_propagator_expression(graph: pydot.Dot) -> Expression:
    propagators = E("1")
    for edge in graph.get_edges():
        if edge.get("is_cut") is not None:
            continue
        propagators *= _expr(
            _propagator_from_lmb_rep(_strip_quotes(edge.get("lmb_rep")))
        )
    return propagators


def _initial_state_parton_propagator_strings(
    parton_assignments: list[tuple[str, str]],
) -> set[str]:
    return {
        str(_expr(f"1/sq4({momentum})").expand())
        for _particle, momentum in parton_assignments
    }


def _graph_propagator_expression_with_initial_state_substitution(
    graph: pydot.Dot,
    parton_assignments: list[tuple[str, str]],
) -> Expression:
    initial_state_propagators = _initial_state_parton_propagator_strings(
        parton_assignments
    )
    propagators = E("1")
    for edge in graph.get_edges():
        if edge.get("is_cut") is not None:
            continue
        propagator = _expr(_propagator_from_lmb_rep(_strip_quotes(edge.get("lmb_rep"))))
        if str(propagator.expand()) in initial_state_propagators:
            propagators *= 1 / (LAMBDA * PSQ)
            continue
        propagators *= propagator
    return propagators


def _graph_full_expression(
    graph: pydot.Dot,
    parton_assignments: list[tuple[str, str]],
) -> Expression:
    numerator = _substitute_sp_momenta(_graph_simplified_numerator_expr(graph), graph)
    return numerator * _graph_propagator_expression_with_initial_state_substitution(
        graph, parton_assignments
    )


def _graph_expression_with_momentum_conservation(
    graph: pydot.Dot,
    parton_assignments: list[tuple[str, str]],
) -> Expression:
    expr = _graph_full_expression(graph, parton_assignments)
    total_parton_momentum = _expr("0")
    for _particle, momentum in parton_assignments:
        total_parton_momentum += _expr(momentum)
    expr = expr.replace(total_parton_momentum, _expr("P"))

    solved_momentum, substitution, _display_substitution = (
        _solved_parton_momentum_substitution(parton_assignments)
    )
    return expr.replace(_expr(solved_momentum), substitution)


def _graph_expression_with_parton_parametrization(
    graph: pydot.Dot,
    parton_assignments: list[tuple[str, str]],
) -> Expression:
    pplus = _expr("Pplus")
    pminus = PBAR
    pperp = PPERP

    expr = _graph_expression_with_momentum_conservation(graph, parton_assignments)

    # expr = 1 / SP_MOM(_expr("P") - _expr("P(1)"), _expr("P") - _expr("P(1)"))
    # print(expr)
    expr = expr.replace(
        _expr("P(1)"), XI * pplus + LAMBDA ** E("1/2") * pperp + LAMBDA * XBAR * pminus
    )
    xp = S("xp", is_scalar=True)
    expr = expr.replace(_expr("P"), pplus + LAMBDA * xp * pminus)
    expr = expr.replace(_expr("P(2)"), pminus)
    expr = expr.replace(E("gammalooprs::sq4(x___)"), SP_MOM(S("x___"), S("x___")))
    expr = expr.replace(SP_MOM(_expr("P(2)"), _expr("P(2)")), E("0"))
    expr = expr.replace(SP_MOM(pplus, pplus), E("0"))
    expr = expr.replace(SP_MOM(pminus, pminus), E("0"))
    expr = expr.replace(SP_MOM(pplus, pperp), E("0"))
    expr = expr.replace(SP_MOM(pperp, pplus), E("0"))
    expr = expr.replace(SP_MOM(pminus, pperp), E("0"))
    expr = expr.replace(SP_MOM(pperp, pminus), E("0"))
    expr = expr.replace(SP_MOM(pperp, _expr("K(0)")), E("0"))
    # expr = expr.replace(SP_MOM(pplus, pminus), E("s/2"))
    # expr = expr.replace(SP_MOM(pminus, pplus), E("s/2"))
    expr = expr.replace(xp, PSQ / (2 * SP_MOM(pplus, pminus)))
    expr = expr.replace(XBAR, E("-pperpsq") / (2 * SP_MOM(pplus, pminus)) / XI)
    expr = expr.replace(SP_MOM(pplus, pminus), -E("qsq") / (2 * X * Y))
    expr = expr.replace(SP_MOM(pminus, pplus), -E("qsq") / (2 * X * Y))
    expr = expr.replace(SP_MOM(_expr("K(0)"), _expr("K(0)")), E("qsq"))
    expr = expr.replace(SP_MOM(pperp, pperp), E("pperpsq"))
    expr = expr.replace(SP_MOM(pplus, _expr("K(0)")), -E("qsq") / (2 * X))
    expr = expr.replace(SP_MOM(pminus, _expr("K(0)")), -E("qsq") / (2 * Y))
    expr = expr.replace(E("psq"), E("pperpsq") / (XI * (1 - XI)))
    print(expr)
    expr = (LAMBDA * expr).series(LAMBDA, 0, 0).to_expression()
    if expr.series(LAMBDA, 0, -1).to_expression().expand() != E("0"):
        print(expr.series(LAMBDA, 0, -1).to_expression().expand())
        # raise ValueError("problemo with power-counting")
    print(expr.expand())
    return expr.expand()


def _display_expression(expr: Expression) -> str:
    return str(expr).replace("sp_mom(", "sp(")


@lru_cache(maxsize=1)
def _sm_particle_colors() -> dict[str, int]:
    from ufo_model_loader.commands import load_model

    model, _ = load_model(
        "sm",
        None,
        simplify_model=True,
        wrap_indices_in_lorentz_structures=False,
    )

    particle_colors: dict[str, int] = {}
    for particle in model.particles:
        particle_colors[particle.name] = particle.color
        particle_colors[particle.antiname] = particle.color
    return particle_colors


def _incoming_parton_labels(command: str) -> list[str]:
    particle_colors = _sm_particle_colors()
    incoming_partons: list[str] = []
    for particle in _incoming_particles_from_generate_command(command):
        if particle not in particle_colors:
            raise ValueError(f"Unknown incoming particle '{particle}'.")
        if particle_colors[particle] != 1:
            incoming_partons.append(particle)
    return incoming_partons


@lru_cache(maxsize=1)
def _sm_vertex_couplings() -> dict[str, tuple[str, ...]]:
    from ufo_model_loader.commands import load_model

    model, _ = load_model(
        "sm",
        None,
        simplify_model=True,
        wrap_indices_in_lorentz_structures=False,
    )

    couplings_by_vertex: dict[str, tuple[str, ...]] = {}
    for vertex_rule in model.vertex_rules:
        coupling_names = sorted({
            coupling.name for coupling in vertex_rule.couplings.values()
        })
        couplings_by_vertex[vertex_rule.name] = tuple(coupling_names)
    return couplings_by_vertex


def _graph_coupling_counts(graph: pydot.Dot) -> Counter[str]:
    couplings_by_vertex = _sm_vertex_couplings()
    coupling_counts: Counter[str] = Counter()

    for node in graph.get_nodes():
        if node.get_name() in {"edge", "node"}:
            continue
        int_id = _strip_quotes(node.get("int_id"))
        if int_id not in couplings_by_vertex:
            raise ValueError(
                f"Unknown vertex rule '{int_id}' in graph '{graph.get_name()}'."
            )
        for coupling_name in couplings_by_vertex[int_id]:
            coupling_counts[coupling_name] += 1

    return coupling_counts


def _graph_has_target_couplings(graph: pydot.Dot) -> bool:
    coupling_counts = _graph_coupling_counts(graph)
    return all(
        coupling_counts.get(coupling_name, 0) == power
        for coupling_name, power in TARGET_COUPLING_POWERS.items()
    )


def _serialize_graphs(graphs: list[pydot.Dot]) -> str:
    return "\n\n".join(graph.to_string().rstrip() for graph in graphs) + "\n"


def _rebuild_state_from_filtered_dot(filtered_dot_path: Path) -> str:
    shutil.rmtree(STATE_DIR, ignore_errors=True)
    state_worker = GammaLoopAPI(str(STATE_DIR))
    state_worker.run(f"set global file {GENERATE_TOML}")
    state_worker.run("import model sm-default.json")
    state_worker.run(
        f"import graphs {filtered_dot_path.resolve()} -p {BASE_NAME} -i {INTEGRAND_NAME}"
    )
    state_worker.run("save state -o")
    state_worker.run("save dot")
    return state_worker.get_dot_files(process_id=None, integrand_name=INTEGRAND_NAME)


def _draw_graphs(dot_str: str) -> list[Path]:
    try:
        subprocess.run(
            ["just", "draw"],
            cwd=STATE_DIR,
            check=True,
        )
        shutil.copy2(STATE_DIR / "drawings.pdf", DRAW_PDF_PATH)
        return [DRAW_PDF_PATH]
    except subprocess.CalledProcessError:
        output_paths: list[Path] = []
        for graph in pydot.graph_from_dot_data(dot_str):
            output_path = DRAW_OUTPUT_DIR / f"{graph.get_name()}.pdf"
            graph.write_pdf(str(output_path))
            output_paths.append(output_path)
        return output_paths


def main() -> int:
    shutil.rmtree(STATE_DIR, ignore_errors=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    shutil.rmtree(DRAW_OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DRAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gl_worker = GammaLoopAPI(str(STATE_DIR))
    gl_worker.run(f"set global file {GENERATE_TOML}")
    gl_worker.run("import model sm-default.json")
    gl_worker.run(GENERATE_COMMAND)

    dot_str = gl_worker.get_dot_files(process_id=None, integrand_name=INTEGRAND_NAME)
    dot_str = dot_str.replace("UFO::Me", "0")
    generated_graphs = pydot.graph_from_dot_data(dot_str)
    filtered_graphs = [
        graph for graph in generated_graphs if _graph_has_target_couplings(graph)
    ]
    if len(filtered_graphs) == 0:
        raise ValueError(
            "No graphs matched the target GC_1^2 * GC_3^2 coupling pattern."
        )

    print(
        f"Kept {len(filtered_graphs)} / {len(generated_graphs)} graphs with coupling pattern "
        "GC_1^2 * GC_3^2."
    )

    parton_assignments, total_parton_momentum = _parton_momentum_relation(
        filtered_graphs, GENERATE_COMMAND
    )
    print("Initial-state parton momenta:")
    for particle, momentum in parton_assignments:
        print(f"  {particle}: {momentum}")
    print("Solved partonic momentum relation:")
    print(f"  {total_parton_momentum}")
    _solved_momentum, _substitution_expr, solved_substitution = (
        _solved_parton_momentum_substitution(parton_assignments)
    )
    print("Solved momentum-conservation substitution:")
    print(f"  {solved_substitution}")
    print()

    filtered_dot_str = _serialize_graphs(filtered_graphs)
    DOT_PATH.write_text(filtered_dot_str, encoding="utf-8")

    del gl_worker
    dot_str = _rebuild_state_from_filtered_dot(DOT_PATH).replace("UFO::Me", "0")
    DOT_PATH.write_text(dot_str, encoding="utf-8")
    # drawing_paths = _draw_graphs(dot_str)

    # print(DOT_PATH)
    # for drawing_path in drawing_paths:
    #    print(drawing_path)
    # print()
    sum = E("0")
    for graph in pydot.graph_from_dot_data(dot_str):
        if graph.get_name() in {"GL97", "GL98"}:
            continue
        print(f"{graph.get_name()} full expression:")
        gg = _graph_expression_with_parton_parametrization(graph, parton_assignments)
        sum = sum + gg
        print(gg)
    print("SUMMM")
    print(sum.expand())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
