import json
import os
import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from itertools import combinations, count, product

import pydot

from gammaloop import (  # isort: skip # type: ignore # noqa: F401
    GammaLoopAPI,
    LogLevel,
    evaluate_graph_overall_factor,
)
from sympy.assumptions.assume import false

try:
    from gammaloop import git_version  # isort: skip # type: ignore # noqa: F401
except ImportError:
    try:
        from gammaloop import __version__ as git_version  # isort: skip # type: ignore # noqa: F401
    except ImportError:
        git_version = "unknown"
from symbolica import AtomType, E, Expression, Replacement, S  # pyright: ignore
from symbolica.community.idenso import (  # pyright: ignore
    simplify_color,
    simplify_gamma,
    simplify_metrics,
)

from processes.dy.dy_evaluators import substitute_process_couplings
from processes.dy.dy_graph_utils import (
    _base_node,
    _node_key,
    _parse_port,
    _strip_quotes,
    boundary_edges,
    change_routing,
    get_components,
    get_LR_components,
    get_simple_cycles,
    select_gl059_lmb_choice,
    select_gl071_lmb_choice,
    select_gl081_lmb_choice,
    select_gl101_lmb_choice,
)
from processes.dy.dy_integrated_uv_ct import (
    attach_uv_routing_metadata,
    build_uv_subgraph_routing,
    construct_integrated_counter_term as construct_integrated_uv_counter_term,
    remap_uv_expression_to_production,
)
from processes.dy.dy_top_self_energy import (
    PROJECTED_OS_SPATIAL_PROBE_ID,
    resolve_top_self_energy_renormalisation,
)
from utils.cff import CFFStructure
from utils.utils import (
    EVALUATORS_FOLDER,
    PYGLOOP_FOLDER,
    ParamBuilder,
    PygloopEvaluator,
    expr_to_string,
    logger,
)

pjoin = os.path.join

gl_log_level = LogLevel.Off

debug = False


_LAURENT_CHUNK_SIZE = 128
_DY_COLLINEAR_X = S(
    "dy_collinear_x",
    is_scalar=True,
    is_positive=True,
)

THRESHOLD_H_FUNCTIONS = frozenset(
    {"gaussian", "inverse_square_damped"}
)


def resolve_threshold_h_function(name, threshold_h_function=None):
    """Resolve an optional threshold h-function choice without changing defaults."""
    if threshold_h_function is None:
        return "gaussian" if name == "DY" else "inverse_square_damped"
    resolved = str(threshold_h_function).strip()
    if resolved not in THRESHOLD_H_FUNCTIONS:
        choices = ", ".join(sorted(THRESHOLD_H_FUNCTIONS))
        raise ValueError(
            f"threshold_h_function must be one of {choices}; got {resolved!r}"
        )
    return resolved


def _projected_top_self_energy_tensor_coefficients(
    loop_momentum_sq: Expression,
) -> dict[str, Expression]:
    """Exact identity/spatial coefficients of the two-branch projector.

    The input is the squared spatial momentum of the massive cycle line at
    the fixed on-shell boundary momentum.  No angular averaging is performed;
    the spatial coefficient multiplies the open ``slash(k_spatial)`` basis.
    """

    radius = loop_momentum_sq ** E("1/2")
    top_energy = (loop_momentum_sq + E("MT") ** 2) ** E("1/2")
    causal_sum = top_energy + radius
    causal_denominator = causal_sum**2 - E("MT") ** 2
    uv_energy = (loop_momentum_sq + E("mUV") ** 2) ** E("1/2")
    return {
        "pole_identity": (
            -2
            * E("MT")
            * causal_sum
            / (top_energy * radius * causal_denominator)
            + E("MT") / (radius * causal_denominator)
        ),
        "pole_spatial": causal_sum
        / (top_energy * radius * causal_denominator),
        "uv_identity": -3 * E("MT") / (4 * uv_energy**3),
        "uv_spatial": E("1") / (2 * uv_energy**3),
    }


class _LaurentSeriesImplementationError(RuntimeError):
    """Raised when an exact chunked Laurent expansion cannot be established."""


@dataclass(frozen=True)
class _LaurentPathInfo:
    """Structural information about one chunked Laurent expansion."""

    decomposition_type: str
    common_valuation: int | None
    residual_valuations: tuple[tuple[int, int], ...]
    residual_count: int
    chunk_count: int

    # Short aliases keep diagnostic consumers readable while the more explicit
    # field names above document the record on their own.
    @property
    def decomposition_case(self) -> str:
        return self.decomposition_type

    @property
    def residual_terms(self) -> int:
        return self.residual_count

    @property
    def chunks(self) -> int:
        return self.chunk_count


@dataclass(frozen=True)
class _LaurentDecomposition:
    decomposition_type: str
    common: Expression
    additive_choices: tuple[tuple[Expression, ...], ...]
    residual_count: int
    common_factor_count: int

    def residual_terms(self):
        for choices in product(*self.additive_choices):
            yield _balanced_product(choices)


def _balanced(values, combine, identity):
    current = list(values)
    if not current:
        return identity
    while len(current) > 1:
        current = [
            current[index]
            if index + 1 == len(current)
            else combine(current[index], current[index + 1])
            for index in range(0, len(current), 2)
        ]
    return current[0]


def _balanced_sum(values):
    return _balanced(values, lambda left, right: left + right, E("0"))


def _balanced_product(values):
    return _balanced(values, lambda left, right: left * right, E("1"))


def _laurent_multiplicative_factors(expression: Expression) -> list[Expression]:
    if bool(expression.is_type(AtomType.Mul)):
        return list(expression)
    return [expression]


def _extract_laurent_common_factors(
    terms: list[Expression],
) -> tuple[Expression, tuple[Expression, ...], int]:
    """Extract the exact multiset intersection of immediate term factors."""
    if not terms:
        return E("1"), (), 0

    keyed_terms = []
    for term in terms:
        keyed_terms.append(
            [
                (factor.to_canonical_string(), factor)
                for factor in _laurent_multiplicative_factors(term)
            ]
        )

    common = Counter(key for key, _factor in keyed_terms[0])
    for factors in keyed_terms[1:]:
        common &= Counter(key for key, _factor in factors)

    representatives = {key: factor for key, factor in keyed_terms[0]}
    common_factors = [
        representatives[key]
        for key, multiplicity in common.items()
        for _ in range(multiplicity)
    ]
    residual_terms = []
    for factors in keyed_terms:
        remaining = common.copy()
        residual_factors = []
        for key, factor in factors:
            if remaining[key]:
                remaining[key] -= 1
            else:
                residual_factors.append(factor)
        residual_terms.append(_balanced_product(residual_factors))

    return (
        _balanced_product(common_factors),
        tuple(residual_terms),
        len(common_factors),
    )


def _decompose_laurent_expression(expression: Expression) -> _LaurentDecomposition:
    """Describe an exact additive decomposition without distributing it."""
    if bool(expression.is_type(AtomType.Add)):
        terms = list(expression)
        common, residual_terms, common_count = _extract_laurent_common_factors(
            terms
        )
        return _LaurentDecomposition(
            "top_sum_common_factor" if common_count else "top_sum_chunks",
            common,
            (residual_terms,),
            len(residual_terms),
            common_count,
        )

    if bool(expression.is_type(AtomType.Mul)):
        factors = list(expression)
        additive_factors = [
            factor for factor in factors if bool(factor.is_type(AtomType.Add))
        ]
        if additive_factors:
            common_factors = [
                factor for factor in factors if not bool(factor.is_type(AtomType.Add))
            ]
            residual_count = 1
            additive_choices = []
            for factor in additive_factors:
                choices = tuple(factor)
                residual_count *= len(choices)
                additive_choices.append(choices)
            return _LaurentDecomposition(
                "natural_product"
                if len(additive_factors) == 1
                else "multi_sum_product",
                _balanced_product(common_factors),
                tuple(additive_choices),
                residual_count,
                len(common_factors),
            )

    return _LaurentDecomposition(
        "single_chunk",
        E("1"),
        ((expression,),),
        1,
        0,
    )


def _reconstruct_laurent_decomposition(
    decomposition: _LaurentDecomposition,
) -> Expression:
    """Reconstruct a decomposition; intended for focused structural tests."""
    return decomposition.common * _balanced_sum(decomposition.residual_terms())


def _integral_trailing_exponent(series, *, context: str) -> int:
    try:
        trailing = series.get_trailing_exponent()
    except (TypeError, ValueError, RuntimeError) as error:
        raise _LaurentSeriesImplementationError(
            f"Symbolica could not establish the Laurent valuation for {context}: "
            f"{error}"
        ) from error
    if (
        not isinstance(trailing, tuple)
        or len(trailing) != 2
        or not isinstance(trailing[0], int)
        or not isinstance(trailing[1], int)
        or trailing[1] != 1
    ):
        raise _LaurentSeriesImplementationError(
            f"Expected an integral Laurent valuation for {context}, got "
            f"{trailing!r}."
        )
    return trailing[0]


def _integral_laurent_valuation(
    expression: Expression,
    variable: Expression,
    *,
    context: str,
) -> int:
    try:
        series = expression.series(variable, 0, 0)
    except (TypeError, ValueError, RuntimeError) as error:
        raise _LaurentSeriesImplementationError(
            f"Symbolica could not construct a Laurent series for {context}: "
            f"{error}"
        ) from error
    return _integral_trailing_exponent(series, context=context)


def _laurent_valuation_counts(
    terms,
    variable: Expression,
    *,
    context: str,
) -> Counter:
    valuations = Counter()
    for index, term in enumerate(terms):
        valuations[
            _integral_laurent_valuation(
                term,
                variable,
                context=f"{context} term {index}",
            )
        ] += 1
    return valuations


def _chunked_laurent_series(terms, variable: Expression, order: int, chunk_size: int):
    combined = None
    chunk = []
    chunk_count = 0
    for term in terms:
        chunk.append(term)
        if len(chunk) < chunk_size:
            continue
        try:
            part = _balanced_sum(chunk).series(variable, 0, order)
        except (TypeError, ValueError, RuntimeError) as error:
            raise _LaurentSeriesImplementationError(
                f"Symbolica could not expand Laurent chunk {chunk_count} "
                f"through order {order}: {error}"
            ) from error
        combined = part if combined is None else combined + part
        chunk_count += 1
        chunk = []
    if chunk:
        try:
            part = _balanced_sum(chunk).series(variable, 0, order)
        except (TypeError, ValueError, RuntimeError) as error:
            raise _LaurentSeriesImplementationError(
                f"Symbolica could not expand Laurent chunk {chunk_count} "
                f"through order {order}: {error}"
            ) from error
        combined = part if combined is None else combined + part
        chunk_count += 1
    if combined is None:
        raise _LaurentSeriesImplementationError(
            "An exact Laurent decomposition unexpectedly contained no terms."
        )
    return combined, chunk_count


def _factor_chunk_laurent_series(
    expression: Expression,
    variable: Expression,
    order: int,
    *,
    chunk_size: int = _LAURENT_CHUNK_SIZE,
    min_terms: int = 0,
) -> tuple[Expression, _LaurentPathInfo]:
    """Expand a Laurent expression through ``order`` using exact chunks.

    Factored residuals are used only after proving that every residual term
    has valuation zero.  Direct decompositions are linear sums of exact terms
    and therefore safely admit arbitrary integral Laurent valuations.
    """
    if chunk_size <= 0:
        raise ValueError("Laurent chunk_size must be positive.")
    if min_terms < 0:
        raise ValueError("Laurent min_terms must be non-negative.")

    decomposition = _decompose_laurent_expression(expression)
    use_factorization = (
        decomposition.common_factor_count > 0
        and decomposition.residual_count >= min_terms
    )

    if use_factorization:
        try:
            common_series = decomposition.common.series(variable, 0, order)
        except (TypeError, ValueError, RuntimeError) as error:
            raise _LaurentSeriesImplementationError(
                "Symbolica could not construct a Laurent series for the "
                f"{decomposition.decomposition_type} common factor: {error}"
            ) from error
        common_valuation = _integral_trailing_exponent(
            common_series,
            context=f"{decomposition.decomposition_type} common factor",
        )
        residual_valuations = _laurent_valuation_counts(
            decomposition.residual_terms(),
            variable,
            context=decomposition.decomposition_type,
        )
        if residual_valuations != Counter({0: decomposition.residual_count}):
            raise _LaurentSeriesImplementationError(
                f"The {decomposition.decomposition_type} decomposition requires "
                "every residual term to have valuation zero; got "
                f"{dict(sorted(residual_valuations.items()))}."
            )

        residual_series, chunk_count = _chunked_laurent_series(
            decomposition.residual_terms(),
            variable,
            order - common_valuation,
            chunk_size,
        )
        expanded = (common_series * residual_series).to_expression()
    else:
        common_valuation = None

        def exact_terms():
            for residual in decomposition.residual_terms():
                yield decomposition.common * residual

        residual_valuations = _laurent_valuation_counts(
            exact_terms(),
            variable,
            context=decomposition.decomposition_type,
        )
        expanded_series, chunk_count = _chunked_laurent_series(
            exact_terms(), variable, order, chunk_size
        )
        expanded = expanded_series.to_expression()

    return expanded, _LaurentPathInfo(
        decomposition_type=decomposition.decomposition_type,
        common_valuation=common_valuation,
        residual_valuations=tuple(sorted(residual_valuations.items())),
        residual_count=decomposition.residual_count,
        chunk_count=chunk_count,
    )


def Es(expr: str) -> Expression:
    return E(expr.replace('"', ""), default_namespace="gammalooprs")


def heaviside_theta(x):
    if x > 0:
        return 1
    else:
        return 0


# Little struct that makes it more manageable to deal with cut graphs


@dataclass(frozen=True)
class RaisedCutPair:
    cut_edge_id: str
    partner_edge_id: str
    routing_relation: str
    particle: str

    @property
    def routing_sign(self) -> int:
        if self.routing_relation == "same":
            return 1
        if self.routing_relation == "opp":
            return -1
        raise ValueError(
            f"Unsupported raised-cut routing relation: {self.routing_relation}"
        )


def _collinear_momentum_seed_edge(
    partition_side: list[pydot.Edge],
    raised_pairs: tuple[RaisedCutPair, ...],
) -> pydot.Edge:
    if not partition_side:
        raise ValueError("Cannot select a momentum seed from an empty partition.")

    edge_by_id = {
        _strip_quotes(str(edge.get_attributes()["id"])): edge
        for edge in partition_side
    }
    raised_cut_ids = {
        pair.cut_edge_id for pair in raised_pairs if pair.cut_edge_id in edge_by_id
    }
    if len(raised_cut_ids) > 1:
        raise ValueError(
            "A collinear partition contains multiple physical raised-cut "
            f"edges: {sorted(raised_cut_ids)}."
        )
    if raised_cut_ids:
        return edge_by_id[next(iter(raised_cut_ids))]
    return partition_side[0]


class routed_cut_graph(object):
    def __init__(self, graph, initial_cut, final_cut, partition):
        self.graph = graph
        self.initial_cut = initial_cut
        self.final_cut = final_cut
        self.partition = partition
        self.raised_cut_pairs: tuple[RaisedCutPair, ...] = ()
        self.raised_cut_detection_complete = False


def _cut_sign_from_attributes(attributes) -> int:
    nonzero_signs = set()
    for key in ("is_cut", "is_cut_DY"):
        value = _strip_quotes(str(attributes.get(key, "0")))
        try:
            sign = int(float(value))
        except ValueError:
            continue
        if sign != 0:
            nonzero_signs.add(sign)
    if len(nonzero_signs) > 1:
        raise ValueError(
            "Conflicting nonzero is_cut and is_cut_DY metadata: "
            + ", ".join(str(sign) for sign in sorted(nonzero_signs))
        )
    return next(iter(nonzero_signs), 0)


def _copy_raised_cut_annotations(source_cut_graph, target_cut_graph):
    pairs = tuple(getattr(source_cut_graph, "raised_cut_pairs", ()))
    source_edges = {
        _strip_quotes(str(edge.get_attributes()["id"])): edge
        for edge in source_cut_graph.graph.get_edges()
    }
    target_edges = {
        _strip_quotes(str(edge.get_attributes()["id"])): edge
        for edge in target_cut_graph.graph.get_edges()
    }
    surviving_pairs = tuple(
        pair
        for pair in pairs
        if pair.cut_edge_id in target_edges and pair.partner_edge_id in target_edges
    )
    promoted_ids = {pair.partner_edge_id for pair in surviving_pairs}
    for edge_id in promoted_ids:
        source_attributes = source_edges[edge_id].get_attributes()
        target_attributes = target_edges[edge_id].get_attributes()
        target_attributes["is_cut"] = str(
            _cut_sign_from_attributes(source_attributes)
        )
        target_attributes["is_cut_DY"] = str(
            _cut_sign_from_attributes(source_attributes)
        )

    target_cut_graph.raised_cut_pairs = surviving_pairs
    target_cut_graph.raised_cut_detection_complete = bool(
        getattr(source_cut_graph, "raised_cut_detection_complete", False)
    )
    return target_cut_graph


def _raised_gluon_projector_spec(
    pair: RaisedCutPair, edge_id: str, overall_sign: int
) -> tuple[str, str, int]:
    """Return denominator head, lifted-momentum edge, and coefficient."""
    edge_id = _strip_quotes(str(edge_id))
    if edge_id == pair.cut_edge_id:
        return "Q", pair.cut_edge_id, -1
    if edge_id != pair.partner_edge_id:
        raise ValueError(f"edge {edge_id} is not in its matched raised pair")

    projector_momentum_id = (
        pair.cut_edge_id
        if pair.routing_relation == "opp"
        else pair.partner_edge_id
    )
    projector_coefficient = -pair.routing_sign * overall_sign
    return "Qr", projector_momentum_id, projector_coefficient


def _raised_numerator_energy_signs(
    cut_energy_signs: dict[str, int],
    raised_pairs: tuple[RaisedCutPair, ...],
) -> dict[str, int]:
    """Validate and return the cut signs used by temporal numerators.

    Synthetic partner signs already encode the pair's routing relation, while
    the graph routing itself orients its spatial momentum. Folding the routing
    sign in here as well would cross its temporal momentum twice.
    """
    numerator_energy_signs = dict(cut_energy_signs)
    partner_relations: dict[str, str] = {}
    for pair in raised_pairs:
        partner_sign = cut_energy_signs.get(pair.partner_edge_id)
        if partner_sign is None:
            raise ValueError(
                "Raised-cut partner is missing nonzero cut metadata: "
                f"{pair.partner_edge_id}."
            )

        previous_relation = partner_relations.get(pair.partner_edge_id)
        if (
            previous_relation is not None
            and previous_relation != pair.routing_relation
        ):
            raise ValueError(
                "Conflicting raised-cut numerator orientations for edge "
                f"{pair.partner_edge_id}."
            )
        partner_relations[pair.partner_edge_id] = pair.routing_relation
    return numerator_energy_signs


def _finalise_cff_momentum_heads(expression: Expression) -> Expression:
    expression = expression.replace(E("Qr(x_,0)"), E("En(x_)"))
    forbidden_heads = sorted(
        {
            symbol.get_name().rsplit("::", 1)[-1]
            for symbol in expression.get_all_symbols()
        }
        & {"E", "Q", "Qp", "Qr", "p1sq", "p2sq", "same"}
    )
    if forbidden_heads:
        raise ValueError(
            "Temporary heads escaped CFF construction: "
            + ", ".join(forbidden_heads)
        )
    return expression


_GL085_CFF_BUCKET_A = "A"
_GL085_CFF_BUCKET_B = "B"
_GL085_CFF_BUCKETS = (_GL085_CFF_BUCKET_A, _GL085_CFF_BUCKET_B)
_GL085_CFF_PARTITION = "GL085_AB"
_GL085_CFF_CANONICAL_LMB = (2, 7)
_GL085_CFF_BUCKET_LMBS = {
    _GL085_CFF_BUCKET_A: (2, 4),
    _GL085_CFF_BUCKET_B: (5, 6),
}


@dataclass(frozen=True)
class _GL085CFFPartition:
    expressions: dict[int, dict[str, Expression]]
    inventory: dict[str, int]
    surface_ids: dict[str, int]


def _additive_terms(expression: Expression) -> list[Expression]:
    expanded = expression.expand()
    if expanded.is_type(AtomType.Add):
        return list(expanded)
    return [expanded]


def _expression_is_zero(expression: Expression) -> bool:
    return expression.expand().to_canonical_string() == "0"


def _same_expression_up_to_sign(left: Expression, right: Expression) -> bool:
    return _expression_is_zero(left - right) or _expression_is_zero(left + right)


def _materialise_causal_etas(
    expression: Expression,
    mapped_eta_expressions: dict[int, Expression],
) -> Expression:
    result = expression
    for eta_id, eta_expression in mapped_eta_expressions.items():
        result = result.replace(
            CFFStructure.SB["eta"](eta_id),
            -eta_expression,
        )
    return result


def _gl085_cff_surface_id(
    mapped_eta_expressions: dict[int, Expression],
    target: Expression,
) -> int:
    matching = [
        eta_id
        for eta_id, eta_expression in mapped_eta_expressions.items()
        if _same_expression_up_to_sign(eta_expression, target)
    ]
    if len(matching) != 1:
        raise ValueError(
            "GL085 CFF split expected exactly one causal surface matching "
            f"{target}, found {matching}."
        )
    return matching[0]


def _partition_gl085_cff_expressions(
    cff_structure: CFFStructure,
    mapped_eta_expressions: dict[int, Expression],
) -> _GL085CFFPartition:
    """Split the physical GL085 {2,5} CFF before numerator multiplication."""

    surface_targets = _gl085_causal_surfaces()
    surface_ids = {
        label: _gl085_cff_surface_id(mapped_eta_expressions, target)
        for label, target in surface_targets.items()
    }
    eta_a = CFFStructure.SB["eta"](surface_ids["A"])
    eta_b = CFFStructure.SB["eta"](surface_ids["B"])

    bucket_terms: dict[int, dict[str, list[Expression]]] = {
        cff_term.id: {bucket: [] for bucket in _GL085_CFF_BUCKETS}
        for cff_term in cff_structure.expressions
    }
    inventory: Counter[str] = Counter()
    overlapping: list[tuple[int, Expression, tuple[int, ...]]] = []
    x_ = CFFStructure.SB["x_"]
    eta_pattern = CFFStructure.SB["eta(x_)"]

    for cff_term in cff_structure.expressions:
        for atomic_term in _additive_terms(cff_term.expression):
            has_a = bool(atomic_term.contains(eta_a**-1))
            has_b = bool(atomic_term.contains(eta_b**-1))
            if has_a and has_b:
                eta_ids = tuple(
                    int(str(match[x_])) for match in atomic_term.match(eta_pattern)
                )
                overlapping.append((cff_term.id, atomic_term, eta_ids))
                inventory["both"] += 1
            elif has_a:
                bucket_terms[cff_term.id][_GL085_CFF_BUCKET_A].append(atomic_term)
                inventory["A_only"] += 1
            elif has_b:
                bucket_terms[cff_term.id][_GL085_CFF_BUCKET_B].append(atomic_term)
                inventory["B_only"] += 1
            else:
                # Neutral terms may be assigned to either class. Keep the
                # convention deterministic by placing all of them in A.
                bucket_terms[cff_term.id][_GL085_CFF_BUCKET_A].append(atomic_term)
                inventory["neutral"] += 1

    if len(overlapping) != 2 or len({row[0] for row in overlapping}) != 1:
        raise ValueError(
            "GL085 CFF split expected two overlapping causal families in one "
            f"orientation, found {[(row[0], row[2]) for row in overlapping]}."
        )

    overlap_orientation = overlapping[0][0]
    complementary_ids = []
    coefficients = []
    for _orientation_id, atomic_term, eta_ids in overlapping:
        other_ids = set(eta_ids) - {surface_ids["A"], surface_ids["B"]}
        if len(other_ids) != 1:
            raise ValueError(
                "GL085 overlapping CFF family did not contain exactly one "
                f"complementary surface: {eta_ids}."
            )
        complementary_id = other_ids.pop()
        complementary_ids.append(complementary_id)
        eta_other = CFFStructure.SB["eta"](complementary_id)
        coefficients.append((atomic_term * eta_a * eta_b * eta_other).cancel())

    if complementary_ids[0] == complementary_ids[1]:
        raise ValueError("GL085 overlapping CFF families use the same complement.")
    if not _expression_is_zero(coefficients[0] - coefficients[1]):
        raise ValueError(
            "GL085 overlapping CFF families have different coefficients and "
            "cannot use the causal partial-fraction identity."
        )

    c_id, d_id = complementary_ids
    eta_c = CFFStructure.SB["eta"](c_id)
    eta_d = CFFStructure.SB["eta"](d_id)
    coefficient = coefficients[0]
    mapped_a = mapped_eta_expressions[surface_ids["A"]]
    mapped_b = mapped_eta_expressions[surface_ids["B"]]
    mapped_c = mapped_eta_expressions[c_id]
    mapped_d = mapped_eta_expressions[d_id]
    if not _expression_is_zero(mapped_a + mapped_b - mapped_c - mapped_d):
        raise ValueError(
            "GL085 causal surfaces do not satisfy the required A+B=C+D relation."
        )

    rewritten_a = coefficient / (eta_a * eta_c * eta_d)
    rewritten_b = coefficient / (eta_b * eta_c * eta_d)
    original_overlap = sum((row[1] for row in overlapping), E("0"))
    materialised_residual = (
        _materialise_causal_etas(
            original_overlap - rewritten_a - rewritten_b,
            mapped_eta_expressions,
        )
        .together()
        .expand()
    )
    if not _expression_is_zero(materialised_residual):
        raise ValueError(
            "GL085 causal partial-fraction rewrite failed exact reconstruction."
        )

    bucket_terms[overlap_orientation][_GL085_CFF_BUCKET_A].append(rewritten_a)
    bucket_terms[overlap_orientation][_GL085_CFF_BUCKET_B].append(rewritten_b)
    partitioned = {
        orientation_id: {
            bucket: sum(terms, E("0"))
            for bucket, terms in expressions.items()
        }
        for orientation_id, expressions in bucket_terms.items()
    }

    original = sum(
        (cff_term.expression for cff_term in cff_structure.expressions),
        E("0"),
    )
    reconstructed = sum(
        (
            expression
            for expressions in partitioned.values()
            for expression in expressions.values()
        ),
        E("0"),
    )
    # Cancel the families copied verbatim while they are still expressed in
    # compact eta heads. Materialising all neutral families before this
    # cancellation causes an unnecessary large common-denominator problem.
    causal_residual = (original - reconstructed).expand()
    residual = (
        _materialise_causal_etas(causal_residual, mapped_eta_expressions)
        .together()
        .expand()
    )
    if not _expression_is_zero(residual):
        raise ValueError("GL085 CFF buckets do not reconstruct the original CFF.")

    inventory["A_rewritten"] = 1
    inventory["B_rewritten"] = 1
    inventory["A_total"] = inventory["neutral"] + inventory["A_only"] + 1
    inventory["B_total"] = inventory["B_only"] + 1
    return _GL085CFFPartition(
        expressions=partitioned,
        inventory=dict(inventory),
        surface_ids={
            "A": surface_ids["A"],
            "B": surface_ids["B"],
            "C": c_id,
            "D": d_id,
        },
    )


def _materialise_gl085_scalar_causal_buckets(
    partition: _GL085CFFPartition,
    mapped_eta_expressions: dict[int, Expression],
) -> dict[str, Expression]:
    """Return the numerator-independent scalar CFF carried by each bucket."""

    return {
        bucket: _materialise_causal_etas(
            sum(
                (
                    expressions[bucket]
                    for expressions in partition.expressions.values()
                ),
                E("0"),
            ),
            mapped_eta_expressions,
        )
        for bucket in _GL085_CFF_BUCKETS
    }


@dataclass(frozen=True)
class _GL085ReferenceCFFTerm:
    id: int
    expression: Expression


@cache
def _gl085_reference_scalar_causal_buckets() -> tuple[Expression, Expression]:
    """Topology-locked scalar CFF used to partition the crossed {2,5} cut.

    The crossed cut is tree-like after cutting and therefore has no loop CFF
    of its own.  These 20 atomic terms are the canonical GL085 physical
    {2,5} CFF.  Live physical-cut generation checks this template exactly, so
    a GammaLoop CFF change cannot silently leave the crossed split stale.
    """

    mapped_eta_expressions = {
        0: E("En(5)+En(6)+En(7)"),
        1: E("-En(0)+En(3)+En(5)+En(7)"),
        2: E("-En(0)+En(2)+En(4)+En(5)+En(7)"),
        3: E("En(2)+En(3)+En(4)"),
        4: E("En(0)+En(3)+En(6)"),
        5: E("-En(2)+En(3)+En(4)"),
        6: E("-En(2)+En(0)+En(4)+En(6)"),
        7: E("-En(0)+En(3)+En(6)"),
        8: E("-En(0)+En(2)+En(4)+En(6)"),
        9: E("-En(5)+En(0)+En(3)+En(7)"),
        10: E("-En(2)-En(5)+En(0)+En(4)+En(7)"),
        11: E("-En(5)+En(6)+En(7)"),
    }
    atomic_eta_ids = (
        ((0, 1, 2),),
        ((0, 2, 3), (0, 3, 4)),
        ((0, 1, 5), (0, 5, 6)),
        ((0, 4, 6),),
        ((1, 2, 7), (2, 7, 8)),
        ((2, 3, 8),),
        ((1, 5, 7),),
        ((3, 4, 9),),
        ((10, 5, 6),),
        ((10, 4, 6), (10, 4, 9)),
        ((11, 7, 8),),
        ((11, 3, 8), (11, 3, 9)),
        ((11, 5, 7), (10, 11, 5)),
        ((10, 11, 9),),
    )
    expressions = []
    for orientation_id, orientation_terms in enumerate(atomic_eta_ids):
        expression = E("0")
        for eta_ids in orientation_terms:
            atomic_term = E("1")
            for eta_id in eta_ids:
                atomic_term /= CFFStructure.SB["eta"](eta_id)
            expression += atomic_term
        expressions.append(_GL085ReferenceCFFTerm(orientation_id, expression))

    partition = _partition_gl085_cff_expressions(
        type("GL085ReferenceCFF", (), {"expressions": expressions})(),
        mapped_eta_expressions,
    )
    expected_inventory = {
        "neutral": 10,
        "A_only": 4,
        "B_only": 4,
        "both": 2,
        "A_rewritten": 1,
        "B_rewritten": 1,
        "A_total": 15,
        "B_total": 5,
    }
    if partition.inventory != expected_inventory:
        raise ValueError(
            "The topology-locked GL085 scalar CFF has an unexpected inventory."
        )
    buckets = _materialise_gl085_scalar_causal_buckets(
        partition,
        mapped_eta_expressions,
    )
    return tuple(buckets[bucket] for bucket in _GL085_CFF_BUCKETS)


def _gl085_reference_scalar_causal_bucket_map() -> dict[str, Expression]:
    expressions = _gl085_reference_scalar_causal_buckets()
    return dict(zip(_GL085_CFF_BUCKETS, expressions, strict=True))


def _gl085_crossed_25_residue(expression: Expression) -> Expression:
    """Take the GL085 D-surface residue without mixing CFF orientations."""

    delta = E("gl085_delta")
    # D = En(0)+En(4)+En(7)-En(2)-En(5).  The crossed {2,5}
    # cut enforces D=0, so solve D=delta for En(2).
    replacement = E(
        "En(0)+En(4)+En(7)-En(5)-gl085_delta"
    )
    return (
        expression.replace(E("En(2)"), replacement)
        .series(delta, 0, -1)
        .to_expression()
        .replace(delta, E("1"))
    )


@cache
def _gl085_reference_scalar_crossed_residues() -> tuple[Expression, Expression]:
    """Return the fixed-orientation A/B residues on the crossed D surface."""

    causal_buckets = _gl085_reference_scalar_causal_bucket_map()
    return tuple(
        _gl085_crossed_25_residue(causal_buckets[bucket])
        for bucket in _GL085_CFF_BUCKETS
    )


def _gl085_reference_scalar_crossed_residue_map() -> dict[str, Expression]:
    expressions = _gl085_reference_scalar_crossed_residues()
    return dict(zip(_GL085_CFF_BUCKETS, expressions, strict=True))


def _gl085_crossed_25_cut_exchange_factor() -> Expression:
    """Convert the physical-cut D residue to the crossed-cut measure."""

    return E("-2*En(1)/((En(4)+En(7))^2-En(1)^2)")


def _gl085_crossed_25_residue_buckets(
    physical_buckets: dict[str, Expression],
) -> dict[str, Expression]:
    """Build crossed buckets without combining fixed-orientation numerators."""

    if set(physical_buckets) != set(_GL085_CFF_BUCKETS):
        raise ValueError("Incomplete GL085 physical CFF buckets for D residue.")
    exchange_factor = _gl085_crossed_25_cut_exchange_factor()
    return {
        bucket: _gl085_crossed_25_residue(expression) * exchange_factor
        for bucket, expression in physical_buckets.items()
    }


def _is_gl085_physical_25_cut(cut_graph) -> bool:
    graph_name = _strip_quotes(
        str(cut_graph.graph.get("base_graph_name") or cut_graph.graph.get_name())
    )
    if graph_name != "GL085":
        return False
    initial_ids = {
        _strip_quotes(str(edge.get_attributes()["id"]))
        for edge in cut_graph.initial_cut
    }
    final_ids = {
        _strip_quotes(str(edge.get_attributes()["id"]))
        for edge in cut_graph.final_cut
    }
    return initial_ids == {"0", "1"} and final_ids == {"2", "5"}


def _is_gl085_anti_collinear_25_cut(cut_graph) -> bool:
    graph_name = _strip_quotes(
        str(cut_graph.graph.get("base_graph_name") or cut_graph.graph.get_name())
    )
    if graph_name != "GL085":
        return False
    initial_ids = {
        _strip_quotes(str(edge.get_attributes()["id"]))
        for edge in cut_graph.initial_cut
    }
    final_ids = {
        _strip_quotes(str(edge.get_attributes()["id"]))
        for edge in cut_graph.final_cut
    }
    return initial_ids == {"0", "4", "7"} and final_ids == {"2", "5"}


def _gl085_physical_25_counterpart(cut_graph) -> routed_cut_graph:
    """Recover the physical cut whose oriented D residues define this cut."""

    if not _is_gl085_anti_collinear_25_cut(cut_graph):
        raise ValueError("The GL085 counterpart builder received the wrong cut.")
    graph = deepcopy(cut_graph.graph)
    edge_by_id = {
        _strip_quotes(str(edge.get_attributes()["id"])): edge
        for edge in graph.get_edges()
    }
    if set(edge_by_id) != {str(edge_id) for edge_id in range(9)}:
        raise ValueError("The GL085 counterpart topology has unexpected edges.")

    physical_cut_signs = {"0": 1, "1": 1, "2": 1, "5": -1}
    for edge_id, edge in edge_by_id.items():
        attributes = edge.get_attributes()
        attributes["is_cut"] = physical_cut_signs.get(edge_id, 0)
        # normalise_graph will recreate this derived flag from is_cut.
        attributes.pop("is_cut_DY", None)

    return routed_cut_graph(
        graph,
        [edge_by_id["0"], edge_by_id["1"]],
        [edge_by_id["2"], edge_by_id["5"]],
        [[edge_by_id["0"]], [edge_by_id["1"]]],
    )


def _gl085_causal_surfaces() -> dict[str, Expression]:
    return {
        _GL085_CFF_BUCKET_A: E("En(3)+En(4)-En(2)"),
        _GL085_CFF_BUCKET_B: E("En(6)+En(7)-En(5)"),
    }


def _gl085_physical_family_lmb(cut_graph) -> tuple[int, int] | None:
    """Return the LMB shared by one GL085 causal cancellation family."""

    graph_name = _strip_quotes(
        str(cut_graph.graph.get("base_graph_name") or cut_graph.graph.get_name())
    )
    if graph_name != "GL085":
        return None
    initial_ids = {
        _strip_quotes(str(edge.get_attributes()["id"]))
        for edge in cut_graph.initial_cut
    }
    if initial_ids not in ({"0", "1"}, {"0", "4", "7"}):
        return None
    final_ids = {
        _strip_quotes(str(edge.get_attributes()["id"]))
        for edge in cut_graph.final_cut
    }
    if final_ids == {"3", "4", "5"}:
        return _GL085_CFF_BUCKET_LMBS[_GL085_CFF_BUCKET_A]
    if final_ids == {"2", "6", "7"}:
        return _GL085_CFF_BUCKET_LMBS[_GL085_CFF_BUCKET_B]
    return None


_GL085_NO_THRESHOLD_TOPOLOGY = {
    "0": ("2", "4", "g"),
    "1": ("3", "5", "g"),
    "2": ("0", "3", "t"),
    "3": ("4", "0", "t"),
    "4": ("0", "5", "g"),
    "5": ("2", "1", "t"),
    "6": ("1", "4", "t"),
    "7": ("1", "5", "g"),
    "8": ("3", "2", "t"),
}


def _assert_gl085_no_threshold_counterterm_topology(cut_graph) -> None:
    """Guard the audited zero-threshold GL085 {2,5} topology."""

    if not _is_gl085_physical_25_cut(cut_graph):
        raise ValueError("The GL085 no-threshold guard received the wrong cut.")
    actual = {}
    for edge in cut_graph.graph.get_edges():
        attributes = edge.get_attributes()
        edge_id = _strip_quotes(str(attributes["id"]))
        particle = _strip_quotes(str(attributes["particle"]))
        if particle in {"t", "t~"}:
            particle = "t"
        actual[edge_id] = (
            _base_node(edge.get_source()),
            _base_node(edge.get_destination()),
            particle,
        )
    if actual != _GL085_NO_THRESHOLD_TOPOLOGY:
        raise ValueError(
            "The GL085 {2,5} topology changed; re-audit threshold "
            f"counterterms before splitting its CFF. Found {actual}."
        )


def _is_zero_cut_value(value) -> bool:
    return _strip_quotes(str(value)) in ["0", "0.0"]


def _cut_edge_ids(cut_graph) -> set[str]:
    cut_ids = {
        _strip_quotes(str(edge.get_attributes()["id"]))
        for edge in list(cut_graph.initial_cut) + list(cut_graph.final_cut)
    }
    for edge in cut_graph.graph.get_edges():
        attrs = edge.get_attributes()
        if not (
            _is_zero_cut_value(attrs.get("is_cut", "0"))
            and _is_zero_cut_value(attrs.get("is_cut_DY", "0"))
        ):
            cut_ids.add(_strip_quotes(str(attrs["id"])))
    return cut_ids


def _post_cut_graph_has_loop(cut_graph) -> bool:
    cut_ids = _cut_edge_ids(cut_graph)
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node_a, node_b) -> bool:
        root_a = find(node_a)
        root_b = find(node_b)
        if root_a == root_b:
            return False
        parent[root_b] = root_a
        return True

    for edge in cut_graph.graph.get_edges():
        attrs = edge.get_attributes()
        if _strip_quotes(str(attrs.get("id", ""))) in cut_ids:
            continue
        source = _base_node(edge.get_source())
        destination = _base_node(edge.get_destination())
        if not union(source, destination):
            return True

    return False


# We will extract amplitude graphs from cut graphs. Amplitude graphs are graphs together with a list
# of replacements that allow to map edge ids in the amplitude graph back to those of the cut graph


class amplitude_graph(object):
    def __init__(self, graph, replacements):
        self.graph = graph
        self.replacements = replacements


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

        # Rename function heads, e.g. spenso::mink(...) -> mink(...)
        expr = expr.replace(
            old(args__),
            new(args__),
            allow_new_wildcards_on_rhs=True,
        )

        # Rename bare symbols/constants, e.g. UFO::MT -> MT
        atom_repls.append(Replacement(old, new))

    if atom_repls:
        expr = expr.replace_multiple(atom_repls)

    return expr


_EXTERNAL_GLUON_POLARISATION_PROTECTED_IDS_ATTR = (
    "external_gluon_polarisation_protected_energy_ids"
)
_EXTERNAL_GLUON_POLARISATION_P1_ID = "1001"
_EXTERNAL_GLUON_POLARISATION_P2_ID = "1002"
_EXTERNAL_GLUON_POLARISATION_BEAM_IDS = {
    _EXTERNAL_GLUON_POLARISATION_P1_ID: "1",
    _EXTERNAL_GLUON_POLARISATION_P2_ID: "2",
}


def _mul_factors(expr: Expression) -> list[Expression]:
    if bool(expr.is_type(AtomType.Mul)):
        return list(expr)
    return [expr]


class DYDotContractionError(ValueError):
    """Raised when a supported DY momentum contraction cannot be closed."""


def _dy_momentum_heads(expr: Expression) -> dict[str, list]:
    heads: dict[str, list] = {"Q": [], "Qp": []}
    names: dict[str, set[str]] = {"Q": set(), "Qp": set()}
    for symbol in expr.get_all_symbols():
        full_name = symbol.get_name()
        short_name = full_name.rsplit("::", 1)[-1]
        if short_name in names:
            names[short_name].add(full_name)
    for short_name, full_names in names.items():
        heads[short_name] = [S(name) for name in sorted(full_names)]
    return heads


def _dy_dot_patterns(expr: Expression) -> list[tuple[str, Expression]]:
    """Return every supported Q/Qp contraction pattern present in ``expr``."""
    heads = _dy_momentum_heads(expr)
    mink = S("spenso::mink")
    metric = S("spenso::g")
    dim_ = S("dy_dot_dim_")
    mu_ = S("dy_dot_mu_")
    nu_ = S("dy_dot_nu_")
    left_id_ = S("dy_dot_left_id_")
    right_id_ = S("dy_dot_right_id_")
    patterns: list[tuple[str, Expression]] = []

    for left_short, right_short in (("Q", "Q"), ("Qp", "Q"), ("Qp", "Qp")):
        for left in heads[left_short]:
            for right in heads[right_short]:
                label = f"{left_short.lower()}_{right_short.lower()}"
                patterns.append(
                    (
                        f"direct.{label}",
                        left(left_id_, mink(dim_, mu_))
                        * right(right_id_, mink(dim_, mu_)),
                    )
                )
                metric_factor = metric(mink(dim_, mu_), mink(dim_, nu_))
                patterns.append(
                    (
                        f"metric_forward.{label}",
                        metric_factor
                        * left(left_id_, mink(dim_, mu_))
                        * right(right_id_, mink(dim_, nu_)),
                    )
                )
                patterns.append(
                    (
                        f"metric_reverse.{label}",
                        metric_factor
                        * left(left_id_, mink(dim_, nu_))
                        * right(right_id_, mink(dim_, mu_)),
                    )
                )

    for short_name in ("Q", "Qp"):
        for head in heads[short_name]:
            patterns.append(
                (
                    f"self.{short_name.lower()}",
                    head(left_id_, mink(dim_, mu_)) ** 2,
                )
            )
    return patterns


_DY_DOT_LOCAL_COMPONENT_BUDGET = 16
# GL013's physical-polarisation/t-channel numerator has a largest local
# distributive fan-out of 432 and an aggregate fan-out of 5,028.  Keep the
# expansion bounded, but leave enough headroom for that production tensor
# network to close before the CFF boundary.
_DY_DOT_TERM_FANOUT_BUDGET = 512
_DY_DOT_EXPRESSION_FANOUT_BUDGET = 8192


def _additive_expansion_fanout(expr: Expression, limit: int) -> int:
    if not bool(expr.is_type(AtomType.Mul)):
        return 1
    fanout = 1
    for factor in expr:
        if not bool(factor.is_type(AtomType.Add)):
            continue
        fanout *= len(factor)
        if fanout > limit:
            return limit + 1
    return fanout


def _has_hidden_additive_momentum_factor(
    expr: Expression,
    *,
    expression_expansion_is_bounded: bool = False,
) -> bool:
    if not bool(expr.is_type(AtomType.Mul)):
        return False
    mink = S("spenso::mink")
    dim_ = S("dy_hidden_dot_dim_")
    slot_ = S("dy_hidden_dot_slot_")
    edge_ = S("dy_hidden_dot_edge_")
    heads = _dy_momentum_heads(expr)
    component_count = sum(
        len(list(expr.match(head(edge_, mink(dim_, slot_)))))
        for short_name in ("Q", "Qp")
        for head in heads[short_name]
    )
    if component_count == 0:
        return False

    additive_factors = [
        factor for factor in expr if bool(factor.is_type(AtomType.Add))
    ]
    if not additive_factors:
        return False

    # Expanding the full profiled tensor expressions is the dominant
    # historical generation cost.  Small closed tensor networks, on the
    # other hand, need distribution before metrics can see components hidden
    # behind a projector or a traced gamma sum.  Admit either a locally small
    # tensor term or an expression whose total distributive cost is bounded,
    # and always cap the fan-out of an individual term.
    expansion_fanout = _additive_expansion_fanout(
        expr, _DY_DOT_TERM_FANOUT_BUDGET
    )
    return (
        expansion_fanout <= _DY_DOT_TERM_FANOUT_BUDGET
        and (
            component_count <= _DY_DOT_LOCAL_COMPONENT_BUDGET
            or expression_expansion_is_bounded
        )
    )


_DY_DOT_VECTOR_SERIAL = count()


def _simplify_labelled_momentum_metrics(expr: Expression) -> Expression:
    """Let Idenso simplify metrics while preserving DY momentum labels."""
    mink = S("spenso::mink")
    dim_ = S("dy_metric_dim_")
    slot_ = S("dy_metric_slot_")
    edge_ = S("dy_metric_edge_")
    restorations = []

    for short_name, heads in _dy_momentum_heads(expr).items():
        for head in heads:
            matches = list(expr.match(head(edge_, mink(dim_, slot_))))
            edge_values = {}
            for match in matches:
                edge_value = match[edge_]
                edge_values.setdefault(edge_value.to_canonical_string(), edge_value)
            for edge_value in edge_values.values():
                placeholder = S(
                    "gammalooprs::dy_dot_vector_"
                    f"{next(_DY_DOT_VECTOR_SERIAL)}_{short_name}",
                    tags=["spenso::tensor", "spenso::rank1"],
                )
                expr = expr.replace(
                    head(edge_value, mink(dim_, slot_)),
                    placeholder(mink(dim_, slot_)),
                    repeat=True,
                    allow_new_wildcards_on_rhs=True,
                )
                restorations.append((placeholder, head, edge_value))

    expr = simplify_metrics(expr)
    for placeholder, head, edge_value in restorations:
        expr = expr.replace(
            placeholder(mink(dim_, slot_)),
            head(edge_value, mink(dim_, slot_)),
            repeat=True,
            allow_new_wildcards_on_rhs=True,
        )
        expr = expr.replace(
            placeholder(mink(dim_)),
            head(edge_value, mink(dim_)),
            repeat=True,
            allow_new_wildcards_on_rhs=True,
        )
    return expr


def to_dots_dy(expr: Expression) -> Expression:
    """Close DY Q/Qp Lorentz components as explicit Spenso dot products.

    DY momenta carry an edge label before their Lorentz index, so Idenso's
    generic ``to_dots`` cannot treat them as ordinary rank-one tensors.  This
    routine owns the narrowly controlled expansion needed to expose additive
    momentum factors and otherwise performs structural replacements only.
    """
    # Match the legacy production ordering: simplifying the tensor network
    # before custom replacements avoids expanding large already-contracted
    # gamma expressions.  Explicit metric rules below remain a fallback for
    # Q/Qp heads that Idenso intentionally treats as opaque.
    expr = _simplify_labelled_momentum_metrics(expr)

    heads = _dy_momentum_heads(expr)
    mink = S("spenso::mink")
    metric = S("spenso::g")
    dot = S("spenso::dot")
    dim_ = S("dy_dot_dim_")
    mu_ = S("dy_dot_mu_")
    nu_ = S("dy_dot_nu_")
    left_id_ = S("dy_dot_left_id_")
    right_id_ = S("dy_dot_right_id_")
    head_pairs = (("Q", "Q"), ("Qp", "Q"), ("Qp", "Qp"))

    # Idenso may close a tagged-vector network directly as g(Q(...),Qp(...)).
    # Convert that scalar form without reintroducing component indices.
    for left_short, right_short in head_pairs:
        for left in heads[left_short]:
            for right in heads[right_short]:
                expr = expr.replace(
                    metric(
                        left(left_id_, mink(dim_)),
                        right(right_id_, mink(dim_)),
                    ),
                    dot(mink(dim_), left(left_id_), right(right_id_)),
                    repeat=True,
                    allow_new_wildcards_on_rhs=True,
                )

    # Propagate Q/Qp components through arbitrary metric chains.  Idenso
    # deliberately treats these labelled momentum heads as opaque, so its
    # metric simplifier cannot close paths such as Q(mu) g(mu,rho)
    # g(rho,nu) Qp(nu) without these structural rules.
    metric_factor = metric(mink(dim_, mu_), mink(dim_, nu_))
    has_lorentz_metrics = any(
        symbol.get_name() == "spenso::g" for symbol in expr.get_all_symbols()
    ) and any(expr.match(metric_factor))
    if has_lorentz_metrics:
        for short_name in ("Q", "Qp"):
            for head in heads[short_name]:
                expr = expr.replace(
                    metric_factor * head(left_id_, mink(dim_, mu_)),
                    head(left_id_, mink(dim_, nu_)),
                    repeat=True,
                    allow_new_wildcards_on_rhs=True,
                )
                expr = expr.replace(
                    metric_factor * head(left_id_, mink(dim_, nu_)),
                    head(left_id_, mink(dim_, mu_)),
                    repeat=True,
                    allow_new_wildcards_on_rhs=True,
                )

    # Handle any direct metric-mediated pair that remains after propagation.
    if has_lorentz_metrics:
        for left_short, right_short in head_pairs:
            for left in heads[left_short]:
                for right in heads[right_short]:
                    replacement = dot(mink(dim_), left(left_id_), right(right_id_))
                    expr = expr.replace(
                        metric_factor
                        * left(left_id_, mink(dim_, mu_))
                        * right(right_id_, mink(dim_, nu_)),
                        replacement,
                        repeat=True,
                        allow_new_wildcards_on_rhs=True,
                    )
                    expr = expr.replace(
                        metric_factor
                        * left(left_id_, mink(dim_, nu_))
                        * right(right_id_, mink(dim_, mu_)),
                        replacement,
                        repeat=True,
                        allow_new_wildcards_on_rhs=True,
                    )

    # Symbolica canonicalises identical factors as a power, which requires a
    # rule separate from the ordinary two-factor contraction.
    for short_name in ("Q", "Qp"):
        for head in heads[short_name]:
            component = head(left_id_, mink(dim_, mu_))
            expr = expr.replace(
                component**2,
                dot(mink(dim_), head(left_id_), head(left_id_)),
                repeat=True,
                allow_new_wildcards_on_rhs=True,
            )

    for left_short, right_short in head_pairs:
        for left in heads[left_short]:
            for right in heads[right_short]:
                expr = expr.replace(
                    left(left_id_, mink(dim_, mu_))
                    * right(right_id_, mink(dim_, mu_)),
                    dot(mink(dim_), left(left_id_), right(right_id_)),
                    repeat=True,
                    allow_new_wildcards_on_rhs=True,
                )

    # Expand only if the structural no-expansion pass left a Q/Qp momentum
    # hidden behind an additive Lorentz factor.  This late fallback is vital:
    # ordinary graphs are fully contracted above and stay mute, while the
    # larger metric networks generated by closed fermion traces are exposed
    # one affected summand at a time.
    if bool(expr.is_type(AtomType.Add)):
        terms = list(expr)
        expression_fanout = 0
        for term in terms:
            expression_fanout += _additive_expansion_fanout(
                term, _DY_DOT_EXPRESSION_FANOUT_BUDGET
            )
            if expression_fanout > _DY_DOT_EXPRESSION_FANOUT_BUDGET:
                break
        expression_expansion_is_bounded = (
            expression_fanout <= _DY_DOT_EXPRESSION_FANOUT_BUDGET
        )
        hidden_terms = {
            index
            for index, term in enumerate(terms)
            if _has_hidden_additive_momentum_factor(
                term,
                expression_expansion_is_bounded=(
                    expression_expansion_is_bounded
                ),
            )
        }
        if hidden_terms:
            expanded = E("0")
            for index, term in enumerate(terms):
                expanded += term.expand() if index in hidden_terms else term
            return to_dots_dy(expanded)
    elif _has_hidden_additive_momentum_factor(
        expr, expression_expansion_is_bounded=True
    ):
        return to_dots_dy(expr.expand())

    unresolved = [
        label
        for label, pattern in _dy_dot_patterns(expr)
        if any(expr.match(pattern))
    ]
    if unresolved:
        preview = expr.to_canonical_string()[:1000]
        raise DYDotContractionError(
            "Eligible Q/Qp Minkowski contractions remain after DY dot "
            f"conversion ({sorted(set(unresolved))}): {preview}"
        )
    return expr


def _numerator_factor_kind(factor: Expression) -> str:
    text = factor.format_plain()
    colour_markers = (
        "spenso::coad(",
        "spenso::cof(",
        "spenso::dind(",
        "spenso::f(",
        "spenso::t(",
        "coad(",
        "cof(",
        "dind(",
    )
    kinematic_markers = (
        "spenso::mink(",
        "spenso::bis(",
        "spenso::gamma(",
        "mink(",
        "bis(",
        "gamma(",
        "Q(",
        "Qp(",
        "sp(",
        "spp(",
    )
    has_colour = any(marker in text for marker in colour_markers)
    has_kinematic = any(marker in text for marker in kinematic_markers)
    if has_kinematic:
        return "kinematic"
    if has_colour:
        return "colour"
    return "scalar"


def _product_factors(factors: list[Expression]) -> Expression:
    out = E("1")
    for factor in factors:
        out *= factor
    return out


def _id_sort_key(edge_id) -> tuple[int, object]:
    edge_id = _strip_quotes(str(edge_id))
    try:
        return (0, int(edge_id))
    except ValueError:
        return (1, edge_id)


def _basis_edge_id(edge_id):
    edge_id = _strip_quotes(str(edge_id))
    try:
        return int(edge_id)
    except ValueError:
        return edge_id


def _graph_loop_count(graph) -> int:
    loop_keys = {
        key
        for edge in graph.get_edges()
        for key in edge.get_attributes()
        if key.startswith("routing_k")
    }
    return len(loop_keys)


def _divergent_uncut_cycle_ids(cut_graph) -> list[tuple[tuple[str, ...], int]]:
    cut_ids = {
        _strip_quotes(str(edge.get_attributes()["id"]))
        for edge in cut_graph.initial_cut + cut_graph.final_cut
    }
    nodes_by_name = {
        _node_key(node.get_name()): node for node in cut_graph.graph.get_nodes()
    }
    divergent_cycles = []
    for cycle in get_simple_cycles(cut_graph.graph):
        cycle_ids = tuple(
            sorted(
                (
                    _strip_quotes(str(edge.get_attributes()["id"]))
                    for edge in cycle
                ),
                key=_id_sort_key,
            )
        )
        if set(cycle_ids) & cut_ids:
            continue

        dod = sum(
            int(_strip_quotes(str(edge.get_attributes()["dod"])))
            for edge in cycle
        )
        visited_nodes = {
            _node_key(endpoint)
            for edge in cycle
            for endpoint in (edge.get_source(), edge.get_destination())
        }
        for node_name in visited_nodes:
            node = nodes_by_name.get(node_name)
            if node is not None:
                dod += int(_strip_quotes(str(node.get_attributes()["dod"])))

        if dod + 4 >= 0:
            divergent_cycles.append((cycle_ids, dod + 4))

    return sorted(divergent_cycles, key=lambda item: item[0])


def _uv_basis_rejection_reason(
    cut_graph,
    candidate,
    n_loops: int,
    divergent_cycles: list[tuple[tuple[str, ...], int]],
) -> str | None:
    candidate_ids = tuple(_strip_quotes(str(edge_id)) for edge_id in candidate)
    if len(candidate_ids) != n_loops:
        return f"expected {n_loops} edges, got {len(candidate_ids)}"
    if len(set(candidate_ids)) != len(candidate_ids):
        return "contains repeated edge IDs"

    graph_edge_ids = {
        _strip_quotes(str(edge.get_attributes()["id"]))
        for edge in cut_graph.graph.get_edges()
    }
    missing = sorted(set(candidate_ids) - graph_edge_ids, key=_id_sort_key)
    if missing:
        return f"missing edge IDs {missing}"

    try:
        routed_graph = change_routing(deepcopy(cut_graph.graph), candidate_ids)
    except (KeyError, TypeError, ValueError) as error:
        return str(error)

    edges_by_id = {
        _strip_quotes(str(edge.get_attributes()["id"])): edge
        for edge in routed_graph.get_edges()
    }
    for cycle_ids, dod in divergent_cycles:
        common_indices = set(range(n_loops))
        for edge_id in cycle_ids:
            edge = edges_by_id.get(edge_id)
            if edge is None:
                return f"divergent cycle {list(cycle_ids)} is missing edge {edge_id}"
            attributes = edge.get_attributes()
            edge_indices = {
                index
                for index in range(n_loops)
                if _strip_quotes(
                    str(attributes.get(f"routing_k{index}", "0"))
                ).strip()
                not in {"0", "+0", "-0"}
            }
            common_indices &= edge_indices

        if len(common_indices) != 1:
            return (
                f"divergent cycle {list(cycle_ids)} (dod + 4 = {dod}) "
                "has common loop coordinates "
                f"{sorted(common_indices)}, expected exactly one"
            )

    return None


_AUTHORITATIVE_GG_LMB_CHOICES = {
    "GL018": (2, 8),
    "GL020": (2, 7),
    "GL021": (2, 7),
    "GL027": (2, 7),
    "GL033": (3, 4),
    "GL043": (2, 5),
    "GL045": (5, 6),
    "GL051": (2, 5),
    "GL053": (5, 6),
    "GL057": (3, 4),
    "GL061": (2, 6),
    "GL073": (2, 8),
    "GL075": (2, 8),
    "GL077": (5, 6),
    "GL079": (2, 7),
    "GL091": (2, 8),
    "GL093": (2, 5),
    "GL105": (3, 6),
    "GL115": (6, 8),
    "GL117": (3, 6),
}

_THRESHOLD_CT_DISABLED_GRAPH_CHANNELS = frozenset(
    {
        ("GL17", (1, -1)),
        ("GL17", (-1, 1)),
        ("GL093", (0, 0)),
    }
)


def _threshold_counterterms_disabled(graph, channel) -> bool:
    if channel is None:
        return False
    base_graph_name = _strip_quotes(str(graph.get("base_graph_name")))
    return (base_graph_name, tuple(channel)) in (
        _THRESHOLD_CT_DISABLED_GRAPH_CHANNELS
    )


def _select_uv_compatible_lmb_choice(cut_graph, preferred_basis):
    n_loops = _graph_loop_count(cut_graph.graph)
    if n_loops == 0:
        return list(preferred_basis)

    preferred_ids = tuple(
        _strip_quotes(str(edge_id)) for edge_id in preferred_basis
    )
    divergent_cycles = _divergent_uncut_cycle_ids(cut_graph)
    preferred_rejection = _uv_basis_rejection_reason(
        cut_graph,
        preferred_ids,
        n_loops,
        divergent_cycles,
    )
    if preferred_rejection is None:
        return list(preferred_basis)

    edge_ids = sorted(
        {
            _strip_quotes(str(edge.get_attributes()["id"]))
            for edge in cut_graph.graph.get_edges()
        },
        key=_id_sort_key,
    )
    cut_ids = {
        _strip_quotes(str(edge.get_attributes()["id"]))
        for edge in cut_graph.initial_cut + cut_graph.final_cut
    }
    preferred_set = set(preferred_ids)
    rejected = [(preferred_ids, preferred_rejection)]
    valid_candidates = []
    for candidate in combinations(edge_ids, n_loops):
        rejection = _uv_basis_rejection_reason(
            cut_graph,
            candidate,
            n_loops,
            divergent_cycles,
        )
        if rejection is not None:
            rejected.append((candidate, rejection))
            continue
        score = (
            len(set(candidate) & cut_ids),
            -len(set(candidate) & preferred_set),
            tuple(_id_sort_key(edge_id) for edge_id in candidate),
        )
        valid_candidates.append((score, candidate))

    if valid_candidates:
        _score, selected = min(valid_candidates, key=lambda item: item[0])
        return [_basis_edge_id(edge_id) for edge_id in selected]

    graph_name = _strip_quotes(
        str(
            cut_graph.graph.get("base_graph_name")
            or cut_graph.graph.get_name()
        )
    )
    rejected_reasons = "; ".join(
        f"{list(candidate)}: {reason}" for candidate, reason in rejected
    )
    raise ValueError(
        f"No UV-compatible loop-momentum basis for {graph_name}; "
        f"preferred basis {list(preferred_ids)} was rejected. "
        f"Candidate rejections: {rejected_reasons}"
    )


def _select_production_lmb_choice(cut_graph, preferred_basis, channel):
    base_graph_name = _strip_quotes(
        str(cut_graph.graph.get("base_graph_name"))
    )
    if channel == (0, 0) and base_graph_name in _AUTHORITATIVE_GG_LMB_CHOICES:
        return list(_AUTHORITATIVE_GG_LMB_CHOICES[base_graph_name])
    return list(preferred_basis)


def _cut_external_energy_ids(cut_graph) -> tuple[list[str], list[str]]:
    initial_ids = [
        _strip_quotes(str(edge.get_attributes()["id"]))
        for edge in cut_graph.initial_cut
    ]
    final_ids = [
        _strip_quotes(str(edge.get_attributes()["id"]))
        for edge in cut_graph.final_cut
    ]
    return initial_ids, final_ids


def _energy_conservation_solution(
    cut_graph, edge_to_sub: str, energy_head: str = "En"
) -> tuple[Expression, Expression]:
    edge_to_sub = _strip_quotes(str(edge_to_sub))
    initial_ids, final_ids = _cut_external_energy_ids(cut_graph)
    initial_set = set(initial_ids)
    final_set = set(final_ids)

    if edge_to_sub in initial_set and edge_to_sub in final_set:
        raise ValueError(
            f"Cannot solve energy conservation for cancelling cut edge {edge_to_sub}."
        )

    if edge_to_sub in final_set:
        replacement = E("0")
        for edge_id in sorted(initial_ids, key=_id_sort_key):
            replacement += E(f"{energy_head}({edge_id})")
        for edge_id in sorted(
            [edge_id for edge_id in final_ids if edge_id != edge_to_sub],
            key=_id_sort_key,
        ):
            replacement -= E(f"{energy_head}({edge_id})")
        return E(f"{energy_head}({edge_to_sub})"), replacement

    if edge_to_sub in initial_set:
        replacement = E("0")
        for edge_id in sorted(final_ids, key=_id_sort_key):
            replacement += E(f"{energy_head}({edge_id})")
        for edge_id in sorted(
            [edge_id for edge_id in initial_ids if edge_id != edge_to_sub],
            key=_id_sort_key,
        ):
            replacement -= E(f"{energy_head}({edge_id})")
        return E(f"{energy_head}({edge_to_sub})"), replacement

    raise ValueError(f"Cut edge {edge_to_sub} is not external to the cut graph.")


def _energy_conservation_replacement(
    cut_graph,
    energy_head: str = "En",
    protected_ids: set[str] | None = None,
) -> tuple[Expression, Expression] | None:
    initial_ids, final_ids = _cut_external_energy_ids(cut_graph)
    if not initial_ids or not final_ids:
        return None
    initial_set = set(initial_ids)
    candidates = [edge_id for edge_id in final_ids if edge_id not in initial_set]
    if protected_ids is not None:
        protected_ids = {_strip_quotes(str(edge_id)) for edge_id in protected_ids}
        candidates = [
            edge_id for edge_id in candidates if edge_id not in protected_ids
        ]
    if not candidates:
        return None

    graph_edge_by_id = {
        _strip_quotes(str(edge.get_attributes()["id"])): edge
        for edge in cut_graph.graph.get_edges()
    }

    def is_s_channel_candidate(edge_id: str) -> bool:
        edge = graph_edge_by_id.get(edge_id)
        if edge is None:
            return False
        attributes = edge.get_attributes()
        has_loop_routing = any(
            _strip_quotes(str(value)) != "0"
            for key, value in attributes.items()
            if key.startswith("routing_k")
        )
        has_p1 = _strip_quotes(
            str(attributes.get("routing_p1", "0"))
        ) != "0"
        has_p2 = _strip_quotes(
            str(attributes.get("routing_p2", "0"))
        ) != "0"
        # A change of loop basis can represent the same s-channel either as
        # the complete incoming momentum or as a pure loop momentum.  Mixed
        # loop/external routings belong to the complementary real-emission
        # edge and are a numerically poor energy-conservation pivot.
        return (
            not has_loop_routing and has_p1 and has_p2
        ) or (
            has_loop_routing and not has_p1 and not has_p2
        )

    s_channel_candidates = [
        edge_id for edge_id in candidates if is_s_channel_candidate(edge_id)
    ]
    edge_to_sub = (
        s_channel_candidates[0]
        if s_channel_candidates
        else candidates[0]
    )
    return _energy_conservation_solution(cut_graph, edge_to_sub, energy_head)


def _cleanup_final_state_raised_energies(
    expression: Expression,
    cut_graph,
) -> tuple[Expression, bool]:
    raised_pairs = tuple(getattr(cut_graph, "raised_cut_pairs", ()))
    final_ids = set(_cut_external_energy_ids(cut_graph)[1])
    final_pairs = [
        pair
        for pair in raised_pairs
        if pair.cut_edge_id in final_ids or pair.partner_edge_id in final_ids
    ]
    if not final_pairs:
        return expression, False
    if len(final_pairs) > 1:
        raise ValueError(
            "A single physical energy-conservation equation cannot "
            "simultaneously eliminate multiple final-state raised pairs."
        )

    pair = final_pairs[0]
    physical_final_members = [
        edge_id
        for edge_id in (pair.cut_edge_id, pair.partner_edge_id)
        if edge_id in final_ids
    ]
    if len(physical_final_members) != 1:
        raise ValueError(
            "A final-state raised pair must contain exactly one original "
            "final-cut edge."
        )

    physical_final_member = physical_final_members[0]
    _target, replacement = _energy_conservation_solution(
        cut_graph,
        physical_final_member,
    )
    for edge_id in (pair.cut_edge_id, pair.partner_edge_id):
        expression = expression.replace(E(f"En({edge_id})"), replacement)

    # Promoting the partner gives the pair two ordinary CFF cut-energy
    # factors. Energy conservation applies to the repeated propagator
    # dependence, but the original physical cut measure must remain
    # evaluated on its own on-shell energy. Restore that one factor after
    # replacing both pair energies; this is the edge-based successor to the
    # former numerator/denominator E-vs-En gate.
    expression *= replacement / E(f"En({physical_final_member})")
    return expression, True


_MASSLESS_PARTICLES = {"d", "d~", "g", "ghG", "ghG~", "a"}
_MASSLESS_NUMERATOR_CUT_PARTICLES = {
    "d",
    "d~",
    "g",
    "gh",
    "gh~",
    "ghG",
    "ghG~",
}


def _numerator_cut_mass_squared(particle: str) -> Expression:
    if particle in _MASSLESS_NUMERATOR_CUT_PARTICLES:
        return E("0")
    if particle in {"t", "t~"}:
        return E("m(t)^2")
    if particle == "a":
        return E("m(a)^2")
    raise ValueError(
        f"Unsupported on-shell numerator cut particle '{particle}'."
    )


def _routed_numerator_on_shell_replacements(
    cut_graph,
) -> tuple[tuple[Expression, Expression], ...]:
    replacements = [
        (E("sp(p(1),p(1))"), E("0")),
        (E("sp(p(2),p(2))"), E("0")),
    ]
    if cut_graph is None:
        return tuple(replacements)

    _, final_cut_ids = _cut_external_energy_ids(cut_graph)
    final_cut_id_set = set(final_cut_ids)
    # A final-state raised residue still differentiates through both members
    # of its repeated pair, so their numerator self-products stay explicit.
    preserved_raised_self_products = {
        edge_id
        for pair in getattr(cut_graph, "raised_cut_pairs", ())
        if (
            pair.cut_edge_id in final_cut_id_set
            or pair.partner_edge_id in final_cut_id_set
        )
        for edge_id in (pair.cut_edge_id, pair.partner_edge_id)
    }

    for edge in sorted(
        cut_graph.graph.get_edges(),
        key=lambda edge: _id_sort_key(edge.get_attributes()["id"]),
    ):
        attributes = edge.get_attributes()
        if _cut_sign_from_attributes(attributes) == 0:
            continue
        edge_id = _strip_quotes(str(attributes["id"]))
        if edge_id in preserved_raised_self_products:
            continue
        particle = _strip_quotes(str(attributes.get("particle", "")))
        replacements.append(
            (
                E(f"sp({edge_id},{edge_id})"),
                _numerator_cut_mass_squared(particle),
            )
        )
    return tuple(replacements)


def _normalise_routed_numerator_on_shell(
    expr: Expression,
    on_shell_replacements: tuple[tuple[Expression, Expression], ...],
) -> Expression:
    expr = expr.replace(
        E("Q(x_,mink(y_,z_))^2"),
        E("sp(x_,x_)"),
    )
    for pattern, replacement in on_shell_replacements:
        expr = expr.replace(pattern, replacement)
    return expr


def _massless_external_beam(
    edge_attributes, loop_count: int
) -> str | None:
    particle = _strip_quotes(str(edge_attributes.get("particle", "")))
    if particle not in _MASSLESS_PARTICLES:
        return None
    if any(
        _strip_quotes(str(edge_attributes.get(f"routing_k{i}", "0"))) != "0"
        for i in range(loop_count)
    ):
        return None
    has_p1 = _strip_quotes(str(edge_attributes.get("routing_p1", "0"))) != "0"
    has_p2 = _strip_quotes(str(edge_attributes.get("routing_p2", "0"))) != "0"
    if has_p1 == has_p2:
        return None
    return "p1" if has_p1 else "p2"


def _routing_sign_match(e: pydot.Edge, ep: pydot.Edge) -> str | None:
    first = e.get_attributes()
    second = ep.get_attributes()
    keys = sorted(
        key
        for key in set(first) | set(second)
        if key.startswith("routing_")
    )

    def value(attributes, key):
        return E(_strip_quotes(str(attributes.get(key, "0"))))

    if all(value(first, key) == value(second, key) for key in keys):
        return "same"
    if all(value(first, key) == -value(second, key) for key in keys):
        return "opp"
    return None


def _extract_common_colour_factors_from_sum(
    expr: Expression,
) -> tuple[list[Expression], Expression]:
    if not bool(expr.is_type(AtomType.Add)):
        return [], expr

    term_factors = [_mul_factors(term) for term in list(expr)]
    if not term_factors:
        return [], expr

    common_colour_factors = []
    first_term_colour_factors = [
        factor
        for factor in term_factors[0]
        if _numerator_factor_kind(factor) == "colour"
    ]

    for candidate in first_term_colour_factors:
        candidate_key = candidate.to_canonical_string()
        matching_indices = []
        for factors in term_factors:
            match_index = next(
                (
                    i
                    for i, factor in enumerate(factors)
                    if factor.to_canonical_string() == candidate_key
                ),
                None,
            )
            if match_index is None:
                matching_indices = []
                break
            matching_indices.append(match_index)
        if not matching_indices:
            continue

        common_colour_factors.append(candidate)
        for factors, match_index in zip(term_factors, matching_indices, strict=True):
            factors.pop(match_index)

    if not common_colour_factors:
        return [], expr

    reduced = E("0")
    for factors in term_factors:
        reduced += _product_factors(factors)
    return common_colour_factors, reduced


def _multiply_into_numerator_parts(
    parts: tuple[Expression, Expression, Expression], expr: Expression
) -> tuple[Expression, Expression, Expression]:
    scalar, colour, kinematic = parts
    for factor in _mul_factors(expr):
        extracted_colour, factor = _extract_common_colour_factors_from_sum(factor)
        for colour_factor in extracted_colour:
            colour *= colour_factor

        kind = _numerator_factor_kind(factor)
        if kind == "colour":
            colour *= factor
        elif kind == "kinematic":
            kinematic *= factor
        else:
            scalar *= factor
    return scalar, colour, kinematic


def _factorised_graph_numerator(
    graph,
) -> tuple[Expression, Expression, Expression]:
    parts = (E("1"), E("1"), E("1"))
    for node in graph.get_nodes():
        if node.get_name() not in ["edge", "node"]:
            node_numerator = node.get("num")
            if node_numerator:
                parts = _multiply_into_numerator_parts(parts, Es(node_numerator))
    for edge in graph.get_edges():
        edge_numerator = edge.get("num")
        if edge_numerator:
            parts = _multiply_into_numerator_parts(parts, Es(edge_numerator))
    return parts


def _four_gluon_vertex_terms(node) -> list[Expression] | None:
    if _strip_quotes(str(node.get("int_id"))) != "V_37":
        return None
    node_numerator = node.get("num")
    if not node_numerator:
        raise ValueError(
            f"Four-gluon vertex {node.get_name()} has no numerator rule."
        )
    rule = Es(node_numerator)
    if not bool(rule.is_type(AtomType.Add)):
        raise ValueError(
            f"Four-gluon vertex {node.get_name()} numerator is not an additive "
            "three-term rule."
        )
    terms = list(rule)
    if len(terms) != 3:
        raise ValueError(
            f"Four-gluon vertex {node.get_name()} has {len(terms)} numerator "
            "branches; expected 3."
        )
    return terms


def _factorised_graph_numerator_branches(
    graph,
) -> list[tuple[Expression, Expression, Expression]]:
    """Factor a graph numerator while preserving V_37 colour/kinematic pairs."""
    branches = [(E("1"), E("1"), E("1"))]
    for node in graph.get_nodes():
        if node.get_name() in ["edge", "node"]:
            continue
        four_gluon_terms = _four_gluon_vertex_terms(node)
        if four_gluon_terms is not None:
            branches = [
                _multiply_into_numerator_parts(parts, term)
                for parts in branches
                for term in four_gluon_terms
            ]
            continue
        node_numerator = node.get("num")
        if node_numerator:
            node_expr = Es(node_numerator)
            branches = [
                _multiply_into_numerator_parts(parts, node_expr)
                for parts in branches
            ]

    for edge in graph.get_edges():
        edge_numerator = edge.get("num")
        if edge_numerator:
            edge_expr = Es(edge_numerator)
            branches = [
                _multiply_into_numerator_parts(parts, edge_expr)
                for parts in branches
            ]
    return branches


def _normalise_dy_colour_invariants(expr: Expression) -> Expression:
    """Translate Symbolica 2.1 QCD invariants to the legacy DY basis.

    Symbolica 2.1 can leave quadratic Casimirs as ``ca``, ``cf``, ``CA``,
    ``CF``, or ``Nc``.  DY uses the fixed SU(3) model, so close those symbols
    at the colour boundary instead of leaking them into CFF expressions that
    do not carry colour parameters.

    The new colour simplifier exposes the fundamental cubic invariant for a
    handful of closed two-loop colour networks. For the fixed SU(3)
    representations used by the DY model,

        d_F^{abc} d_F^{abc} = 20/3 * TR^3.

    Keeping the explicit ``TR`` normalization reproduces the Symbolica 1.5
    result and lets the existing evaluator perform its usual substitution.
    """
    for prefix in ("", "spenso::"):
        for adjoint_casimir in ("ca", "CA", "Nc"):
            expr = expr.replace(
                E(f"{prefix}{adjoint_casimir}"), E("3"), repeat=True
            )
        for fundamental_casimir in ("cf", "CF"):
            expr = expr.replace(
                E(f"{prefix}{fundamental_casimir}"), E("4/3"), repeat=True
            )

    for prefix in ("", "spenso::"):
        expr = expr.replace(
            E(f"{prefix}d33({prefix}cof(3),{prefix}cof(3))"),
            E(f"20/3*{prefix}TR^3"),
            repeat=True,
        )
    return expr


def _prepare_dy_colour_for_simplification(expr: Expression) -> Expression:
    """Close elementary colour contractions before Idenso simplification.

    Symbolica 2.1 registers ``f`` as an antisymmetric tensor.  Letting the new
    metric simplifier choose the representative of a contracted adjoint index
    can consequently add a sign that was absent from the Symbolica 1.5 DY
    results.  Contracting propagator identities in the legacy direction, and
    reducing exact two-generator traces first, makes the remaining colour
    network independent of that representative choice.

    The input is one already-factorised colour branch, so this transformation
    never distributes products over sums.
    """
    if bool(expr.is_type(AtomType.Add)):
        out = E("0")
        for term in expr:
            out += _prepare_dy_colour_for_simplification(term)
        return out

    metric_replacements: list[tuple[Expression, Expression]] = []
    remaining_factors = []
    for factor in _mul_factors(expr):
        if not (
            bool(factor.is_type(AtomType.Fn))
            and factor.get_name() == "spenso::g"
            and len(factor) == 2
        ):
            remaining_factors.append(factor)
            continue

        left, right = factor[0], factor[1]
        if (
            left.get_name() == "spenso::coad"
            and right.get_name() == "spenso::coad"
        ):
            # Idenso 1.5 retained the first self-dual adjoint slot.
            metric_replacements.append((right, left))
            continue

        left_is_lower_fundamental = (
            left.get_name() == "spenso::dind"
            and len(left) == 1
            and left[0].get_name() == "spenso::cof"
        )
        right_is_lower_fundamental = (
            right.get_name() == "spenso::dind"
            and len(right) == 1
            and right[0].get_name() == "spenso::cof"
        )
        if left_is_lower_fundamental and right.get_name() == "spenso::cof":
            # Retain the upper fundamental slot, as in Idenso 1.5.
            metric_replacements.append((left[0], right))
            continue
        if right_is_lower_fundamental and left.get_name() == "spenso::cof":
            metric_replacements.append((right[0], left))
            continue

        remaining_factors.append(factor)

    expr = _product_factors(remaining_factors)
    # Fundamental identity factors can form paths spanning several
    # propagators.  Re-run the ordered substitutions until newly introduced
    # intermediate labels have also reached the retained endpoint.
    for _ in range(len(metric_replacements) + 1):
        before = expr.to_canonical_string()
        for old_index, retained_index in metric_replacements:
            expr = expr.replace(old_index, retained_index, repeat=True)
        if expr.to_canonical_string() == before:
            break
    else:
        raise ValueError("DY colour metric contractions did not converge.")

    def generator_data(factor: Expression):
        if not (
            bool(factor.is_type(AtomType.Fn))
            and factor.get_name() == "spenso::t"
            and len(factor) == 3
            and factor[1].get_name() == "spenso::cof"
            and factor[2].get_name() == "spenso::dind"
            and len(factor[2]) == 1
            and factor[2][0].get_name() == "spenso::cof"
        ):
            return None
        return factor[0], factor[1], factor[2][0]

    # Tr(T^a T^b) = TR delta^{ab}.  Apply the delta directly instead of
    # materialising a new metric that would re-enter the changed 2.1 path.
    while True:
        factors = _mul_factors(expr)
        trace_pair = None
        for left_index, left_factor in enumerate(factors):
            left_data = generator_data(left_factor)
            if left_data is None:
                continue
            left_adjoint, left_upper, left_lower = left_data
            for right_index in range(left_index + 1, len(factors)):
                right_data = generator_data(factors[right_index])
                if right_data is None:
                    continue
                right_adjoint, right_upper, right_lower = right_data
                if (
                    left_upper.to_canonical_string()
                    == right_lower.to_canonical_string()
                    and left_lower.to_canonical_string()
                    == right_upper.to_canonical_string()
                ):
                    trace_pair = (
                        left_index,
                        right_index,
                        left_adjoint,
                        right_adjoint,
                    )
                    break
            if trace_pair is not None:
                break

        if trace_pair is None:
            break

        left_index, right_index, left_adjoint, right_adjoint = trace_pair
        same_adjoint = (
            left_adjoint.to_canonical_string()
            == right_adjoint.to_canonical_string()
        )
        trace_factor = E("spenso::TR")
        if same_adjoint:
            # A closed, already-identified adjoint slot also sums over its
            # representation dimension (eight for the DY SU(3) model).
            trace_factor *= left_adjoint[0]

        expr = trace_factor * _product_factors(
            [
                factor
                for index, factor in enumerate(factors)
                if index not in (left_index, right_index)
            ]
        )
        if not same_adjoint:
            expr = expr.replace(right_adjoint, left_adjoint, repeat=True)

    # A repeated generator can remain inside a longer closed fundamental
    # chain, where ``simplify_color`` represents the chain as an unevaluated
    # trace.  Close an adjoint index that occurs on exactly two generators
    # with the SU(N) completeness relation
    #
    #   T^a_ij T^a_kl = TR (delta_il delta_kj - delta_ij delta_kl / N).
    #
    # Requiring exactly two occurrences makes this a closed-index reduction;
    # open adjoint indices and indices shared with structure constants are
    # deliberately left to Idenso.
    factors = _mul_factors(expr)
    generators = [
        (index, data)
        for index, factor in enumerate(factors)
        if (data := generator_data(factor)) is not None
    ]
    for pair_index, (left_index, left_data) in enumerate(generators):
        left_adjoint, left_upper, left_lower = left_data
        for right_index, right_data in generators[pair_index + 1 :]:
            right_adjoint, right_upper, right_lower = right_data
            if (
                left_adjoint.to_canonical_string()
                != right_adjoint.to_canonical_string()
                or sum(1 for _ in expr.match(left_adjoint)) != 2
                or left_upper[0].to_canonical_string()
                != right_upper[0].to_canonical_string()
            ):
                continue

            remainder = _product_factors(
                [
                    factor
                    for index, factor in enumerate(factors)
                    if index not in (left_index, right_index)
                ]
            )
            fundamental_slots = (
                left_upper,
                left_lower,
                right_upper,
                right_lower,
            )
            if (
                len(
                    {
                        slot.to_canonical_string()
                        for slot in fundamental_slots
                    }
                )
                != 4
                or any(
                    sum(1 for _ in remainder.match(slot)) != 1
                    for slot in fundamental_slots
                )
            ):
                # Empty subchains produce an additional closed delta loop.
                # Leave those cases to Idenso rather than dropping its
                # representation-dimension factor during direct substitution.
                continue

            crossed = remainder.replace_multiple(
                [
                    Replacement(left_lower, right_upper),
                    Replacement(right_lower, left_upper),
                ]
            )
            direct = remainder.replace_multiple(
                [
                    Replacement(left_lower, left_upper),
                    Replacement(right_lower, right_upper),
                ]
            )
            representation_dimension = left_upper[0]
            return E("spenso::TR") * (
                _prepare_dy_colour_for_simplification(crossed)
                - _prepare_dy_colour_for_simplification(direct)
                / representation_dimension
            )

    return expr


_UNRESOLVED_COLOUR_SYMBOLS = {
    "python::coad",
    "python::cof",
    "python::dind",
    "python::f",
    "python::k",
    "python::t",
    "spenso::coad",
    "spenso::cof",
    "spenso::dind",
    "spenso::f",
    "spenso::k",
    "spenso::t",
}


def _assert_no_unresolved_colour_heads(
    expr: Expression, *, context: str
) -> None:
    args__ = S("args__")
    unresolved = sorted(
        {
            symbol.get_name()
            for symbol in expr.get_all_symbols()
            if symbol.get_name() in _UNRESOLVED_COLOUR_SYMBOLS
            and any(expr.match(symbol(args__)))
        }
    )
    if unresolved:
        raise ValueError(
            "Numerator colour contraction is incomplete for "
            f"{context}: {', '.join(unresolved)}"
        )


def _cut_numerator_context(cut_graph: routed_cut_graph) -> str:
    graph_name = _strip_quotes(
        str(
            cut_graph.graph.get("base_graph_name")
            or cut_graph.graph.get_name()
        )
    )
    initial_ids = sorted(
        (
            _strip_quotes(str(edge.get_attributes()["id"]))
            for edge in cut_graph.initial_cut
        ),
        key=_id_sort_key,
    )
    final_ids = sorted(
        (
            _strip_quotes(str(edge.get_attributes()["id"]))
            for edge in cut_graph.final_cut
        ),
        key=_id_sort_key,
    )
    partition_ids = [
        sorted(
            (_strip_quotes(str(edge.get_attributes()["id"])) for edge in side),
            key=_id_sort_key,
        )
        for side in cut_graph.partition
    ]
    return (
        f"graph={graph_name}, cut_key=(initial={initial_ids}, "
        f"final={final_ids}, partition={partition_ids})"
    )


def _dots_to_dy_scalar_products(expr: Expression) -> Expression:
    expr = _strip_namespaces_structurally(expr)
    dot_replacements = [
        (
            E("spenso::dot(spenso::mink(4),Q(x_),Q(y_))"),
            E("sp(x_,y_)"),
        ),
        (
            E("spenso::dot(spenso::mink(4),Qp(x_),Q(y_))"),
            E("spp(qp(x_),y_)"),
        ),
        (
            E("spenso::dot(spenso::mink(4),Q(x_),Qp(y_))"),
            E("spp(qp(y_),x_)"),
        ),
        (
            E("spenso::dot(spenso::mink(4),Qp(x_),Qp(y_))"),
            E("spp(qp(x_),qp(y_))"),
        ),
        (
            E("dot(mink(4),Q(x_),Q(y_))"),
            E("sp(x_,y_)"),
        ),
        (
            E("dot(mink(4),Qp(x_),Q(y_))"),
            E("spp(qp(x_),y_)"),
        ),
        (
            E("dot(mink(4),Q(x_),Qp(y_))"),
            E("spp(qp(y_),x_)"),
        ),
        (
            E("dot(mink(4),Qp(x_),Qp(y_))"),
            E("spp(qp(x_),qp(y_))"),
        ),
    ]
    for pattern, replacement in dot_replacements:
        expr = expr.replace(pattern, replacement, repeat=True)
    return expr


class RoutedIntegrand(object):
    def __init__(
        self,
        integrand,
        cut_graph,
        replacements,
        emr_integrand,
        type,
        ir_limit,
        t_derivative=False,
    ):
        self.emr_integrand = emr_integrand
        self.integrand = integrand
        self.cut_graph = cut_graph
        self.replacements = replacements
        self.approximation_type = type
        self.ir_limit = ir_limit
        self.t_derivative = t_derivative
        # Set for independently routed pieces of a partitioned causal CFF.
        # Ordinary routed integrands deliberately retain ``None``.
        self.cff_bucket = None
        self.cff_lmb = None


# This class is responsible for generating the CFF representation of the cut graph


class EMRIntegrandConstructor(object):
    def __init__(self, params, name, L, state_name=None):
        self.L = L
        self.params = params
        self.name = name
        state_name = state_name if state_name is not None else name
        self.gl_worker = GammaLoopAPI(
            pjoin(PYGLOOP_FOLDER, "outputs", "gammaloop_states", state_name),
            # log_file_name=self.name,
            # log_level=gl_log_level,
        )
        # GAMMALOOP_STATE_FOLDER
        self.gl_worker.run("import model sm-default.json")
        self._protected_external_gluon_polarisation_energy_ids: set[str] = set()

    def _routing_sign_match(self, e: pydot.Edge, ep: pydot.Edge):
        return _routing_sign_match(e, ep)

    def identify_and_mark_raised_cuts(
        self, cut_graph: routed_cut_graph
    ) -> tuple[RaisedCutPair, ...]:
        existing_pairs = tuple(getattr(cut_graph, "raised_cut_pairs", ()))
        if existing_pairs:
            cut_graph.raised_cut_detection_complete = True
            return existing_pairs
        if getattr(cut_graph, "raised_cut_detection_complete", False):
            return ()

        edges = sorted(
            cut_graph.graph.get_edges(),
            key=lambda edge: _id_sort_key(edge.get_attributes()["id"]),
        )
        original_signs = {
            _strip_quotes(str(edge.get_attributes()["id"])): (
                _cut_sign_from_attributes(edge.get_attributes())
            )
            for edge in edges
        }
        proposals: dict[str, int] = {}
        pairs = []
        paired_edge_ids = set()

        for first, second in combinations(edges, 2):
            first_attributes = first.get_attributes()
            second_attributes = second.get_attributes()
            relation = self._routing_sign_match(first, second)
            if relation is None:
                continue
            if not any(
                _strip_quotes(
                    str(first_attributes.get(f"routing_k{i}", "0"))
                )
                != "0"
                for i in range(self.L)
            ):
                continue

            first_id = _strip_quotes(str(first_attributes["id"]))
            second_id = _strip_quotes(str(second_attributes["id"]))
            first_sign = original_signs[first_id]
            second_sign = original_signs[second_id]
            if (first_sign == 0) == (second_sign == 0):
                continue

            if first_sign != 0:
                cut_edge, partner_edge = first, second
                cut_id, partner_id, cut_sign = first_id, second_id, first_sign
            else:
                cut_edge, partner_edge = second, first
                cut_id, partner_id, cut_sign = second_id, first_id, second_sign

            partner_sign = cut_sign if relation == "same" else -cut_sign
            if partner_id in proposals:
                existing_sign = proposals[partner_id]
                raise ValueError(
                    "Ambiguous raised-cut promotion for edge "
                    f"{partner_id}: signs {existing_sign} and {partner_sign}."
                )
            reused_ids = paired_edge_ids & {cut_id, partner_id}
            if reused_ids:
                raise ValueError(
                    "A raised-cut edge participates in multiple pairs: "
                    + ", ".join(sorted(reused_ids, key=_id_sort_key))
                )
            proposals[partner_id] = partner_sign
            paired_edge_ids.update((cut_id, partner_id))
            pairs.append(
                RaisedCutPair(
                    cut_edge_id=cut_id,
                    partner_edge_id=partner_id,
                    routing_relation=relation,
                    particle=_strip_quotes(
                        str(cut_edge.get_attributes().get("particle", ""))
                    ),
                )
            )

        edge_by_id = {
            _strip_quotes(str(edge.get_attributes()["id"])): edge for edge in edges
        }
        for edge_id, sign in proposals.items():
            attributes = edge_by_id[edge_id].get_attributes()
            attributes["is_cut"] = str(sign)
            attributes["is_cut_DY"] = str(sign)

        pairs = tuple(
            sorted(
                pairs,
                key=lambda pair: (
                    _id_sort_key(pair.cut_edge_id),
                    _id_sort_key(pair.partner_edge_id),
                ),
            )
        )
        cut_graph.raised_cut_pairs = pairs
        cut_graph.raised_cut_detection_complete = True
        return pairs

    def _raised_t_channel_beam(self, cut_graph) -> str | None:
        edge_by_id = {
            _strip_quotes(str(edge.get_attributes()["id"])): edge
            for edge in cut_graph.graph.get_edges()
        }
        beams = set()
        for pair in getattr(cut_graph, "raised_cut_pairs", ()):
            attributes = edge_by_id[pair.cut_edge_id].get_attributes()
            has_p1 = _strip_quotes(
                str(attributes.get("routing_p1", "0"))
            ) != "0"
            has_p2 = _strip_quotes(
                str(attributes.get("routing_p2", "0"))
            ) != "0"
            if has_p1 == has_p2:
                continue
            beams.add("p1" if has_p1 else "p2")
        if len(beams) > 1:
            raise ValueError(
                "Raised t-channel pairs refer to multiple incoming beams."
            )
        return next(iter(beams), None)

    def _squared_tree_propagator_edge_ids(self, cut_graph) -> set[str]:
        if not getattr(cut_graph, "raised_cut_pairs", ()):
            return set()
        raised_beam = self._raised_t_channel_beam(cut_graph)
        if raised_beam is None:
            return set()

        partition = getattr(cut_graph, "partition", ())
        if (
            len(partition) != 2
            or len(partition[0]) == len(partition[1])
        ):
            return set()
        # The unequal partition selects the external virtuality to expand.
        # Its beam can be opposite to the repeated pair's routed beam after
        # a loop-basis change, as in the one-loop DY raised cut.
        collinear_beam = "p1" if len(partition[0]) > len(partition[1]) else "p2"

        candidates = []
        for edge in cut_graph.graph.get_edges():
            attributes = edge.get_attributes()
            if _cut_sign_from_attributes(attributes) != 0:
                continue
            edge_beam = _massless_external_beam(attributes, self.L)
            if edge_beam == collinear_beam:
                candidates.append(
                    _strip_quotes(str(attributes["id"]))
                )

        if len(candidates) != 1:
            raise ValueError(
                "Expected exactly one uncut massless "
                f"{collinear_beam} propagator to square for a raised "
                f"t-channel cut, found {len(candidates)}."
            )
        return {candidates[0]}

    def _cut_edge_ids_for_numerator_rewrite(self, graph, cut_graph=None) -> set[str]:
        cut_ids = set()
        synthetic_partner_ids = set()
        if cut_graph is not None:
            cut_ids.update(_cut_edge_ids(cut_graph))
            synthetic_partner_ids.update(
                pair.partner_edge_id
                for pair in getattr(cut_graph, "raised_cut_pairs", ())
            )
            cut_ids.difference_update(synthetic_partner_ids)

        for edge in graph.get_edges():
            attrs = edge.get_attributes()
            if not (
                _is_zero_cut_value(attrs.get("is_cut", "0"))
                and _is_zero_cut_value(attrs.get("is_cut_DY", "0"))
            ):
                edge_id = _strip_quotes(str(attrs["id"]))
                if edge_id not in synthetic_partner_ids:
                    cut_ids.add(edge_id)

        return cut_ids

    def _components_from_edges(self, edges, removed_edge_id: str | None = None):
        nodes = sorted(
            {
                _base_node(endpoint)
                for edge in edges
                if self._edge_local_id(edge) != removed_edge_id
                for endpoint in (edge.get_source(), edge.get_destination())
            }
        )
        if not nodes:
            return []

        adj = {node: set() for node in nodes}
        for edge in edges:
            if self._edge_local_id(edge) == removed_edge_id:
                continue
            source = _base_node(edge.get_source())
            destination = _base_node(edge.get_destination())
            adj.setdefault(source, set()).add(destination)
            adj.setdefault(destination, set()).add(source)

        components = []
        seen = set()
        for node in nodes:
            if node in seen:
                continue
            stack = [node]
            seen.add(node)
            component = {node}
            while stack:
                current = stack.pop()
                for neighbour in adj.get(current, ()):
                    if neighbour not in seen:
                        seen.add(neighbour)
                        component.add(neighbour)
                        stack.append(neighbour)
            components.append(component)
        return components

    def _dot_from_edge_subset(self, graph, edges):
        graph_type = graph.get_type() or "digraph"
        out = pydot.Dot(graph_type=graph_type)
        for key, value in graph.get_attributes().items():
            out.set(key, value)

        node_by_name = {
            _strip_quotes(str(node.get_name())): node for node in graph.get_nodes()
        }
        node_names = sorted(
            {
                _base_node(endpoint)
                for edge in edges
                for endpoint in (edge.get_source(), edge.get_destination())
            }
        )
        for node_name in node_names:
            node = node_by_name.get(node_name)
            if node is None:
                out.add_node(pydot.Node(node_name))
            else:
                out.add_node(deepcopy(node))

        for edge in sorted(edges, key=self._edge_sort_key):
            out.add_edge(deepcopy(edge))

        return out

    def _loop_core_for_numerator_rewrite(self, graph, cut_graph=None):
        cut_ids = self._cut_edge_ids_for_numerator_rewrite(graph, cut_graph)
        retained_edges = [
            edge
            for edge in graph.get_edges()
            if self._edge_local_id(edge) not in cut_ids
        ]
        if not retained_edges:
            return None

        working_edges = retained_edges
        while True:
            base_component_count = len(self._components_from_edges(working_edges))
            bridge_ids = set()
            for edge in working_edges:
                if self._edge_touches_ext(edge):
                    continue
                removed_component_count = len(
                    self._components_from_edges(
                        working_edges, self._edge_local_id(edge)
                    )
                )
                if removed_component_count > base_component_count:
                    bridge_ids.add(self._edge_local_id(edge))

            if not bridge_ids:
                break

            next_edges = [
                edge
                for edge in working_edges
                if self._edge_local_id(edge) not in bridge_ids
            ]
            if len(next_edges) == len(working_edges):
                break
            working_edges = next_edges

        if not working_edges:
            return None

        components = self._components_from_edges(working_edges)
        loop_count = 0
        for component in components:
            component_edges = [
                edge
                for edge in working_edges
                if _base_node(edge.get_source()) in component
                and _base_node(edge.get_destination()) in component
            ]
            non_ext_nodes = {
                node for node in component if not self._is_ext_node(node)
            }
            loop_count += max(0, len(component_edges) - len(non_ext_nodes) + 1)

        if loop_count <= 0:
            return None

        loop_graph = self._dot_from_edge_subset(graph, working_edges)
        loop_edge_ids = {self._edge_local_id(edge) for edge in working_edges}
        loop_nodes = {
            node
            for edge in working_edges
            for node in (_base_node(edge.get_source()), _base_node(edge.get_destination()))
            if not self._is_ext_node(node)
        }
        return loop_graph, loop_edge_ids, loop_nodes, loop_count

    def _node_boundary_edges(self, graph, node_name: str):
        return [
            edge
            for edge in graph.get_edges()
            if _base_node(edge.get_source()) == node_name
            or _base_node(edge.get_destination()) == node_name
        ]

    def _q_edge_ids_in_expr(self, expr: Expression) -> list[str]:
        for pattern, edge_key in [
            (Es("Q(edge_,slot_)"), S("gammalooprs::edge_")),
            (E("Q(edge_,slot_)"), S("edge_")),
        ]:
            matches = list(expr.match(pattern))
            if matches:
                return [
                    _strip_quotes(str(match[edge_key].to_canonical_string()))
                    for match in matches
                ]
        return []

    def _split_single_add_factor(self, expr: Expression):
        if bool(expr.is_type(AtomType.Add)):
            return E("1"), list(expr)
        if not bool(expr.is_type(AtomType.Mul)):
            return None

        factors = list(expr)
        additive = [
            (len(list(factor)), index, list(factor))
            for index, factor in enumerate(factors)
            if bool(factor.is_type(AtomType.Add))
        ]
        if not additive:
            return None

        _n_terms, add_index, terms = min(additive)
        rest = _product_factors(
            [factor for index, factor in enumerate(factors) if index != add_index]
        )
        return rest, terms

    def _three_gluon_vertex_choices(self, node):
        num = node.get("num")
        if not num:
            return None

        split = self._split_single_add_factor(Es(num))
        if split is None:
            return None

        rest, terms = split
        grouped: dict[str, Expression] = {}
        for term in terms:
            q_edge_ids = self._q_edge_ids_in_expr(term)
            if len(q_edge_ids) != 1:
                return None
            edge_id = q_edge_ids[0]
            grouped[edge_id] = grouped.get(edge_id, E("0")) + term

        if len(grouped) != 3:
            return None

        return [
            (edge_id, rest * grouped[edge_id])
            for edge_id in sorted(grouped, key=_id_sort_key)
        ]

    def _loop_three_gluon_vertices(self, graph, loop_nodes: set[str]):
        vertices = []
        for node in graph.get_nodes():
            node_name = _strip_quotes(str(node.get_name()))
            if node_name in ["node", "edge", "graph"] or node_name not in loop_nodes:
                continue

            boundary = self._node_boundary_edges(graph, node_name)
            if len(boundary) != 3:
                continue
            boundary_ids = {self._edge_local_id(edge) for edge in boundary}
            if any(
                _strip_quotes(str(edge.get_attributes().get("particle", ""))) != "g"
                for edge in boundary
            ):
                continue

            choices = self._three_gluon_vertex_choices(node)
            if choices is None:
                continue
            if {edge_id for edge_id, _expr in choices} != boundary_ids:
                continue

            vertices.append((node_name, choices))

        return vertices

    def _boundary_edges_for_cut(self, graph, cut_nodes: set[str]):
        graph_nodes = {
            _base_node(endpoint)
            for edge in graph.get_edges()
            for endpoint in (edge.get_source(), edge.get_destination())
        }
        if not cut_nodes.issubset(graph_nodes):
            raise ValueError("Momentum-conservation cut contains missing nodes.")

        return [
            edge
            for edge in graph.get_edges()
            if (_base_node(edge.get_source()) in cut_nodes)
            != (_base_node(edge.get_destination()) in cut_nodes)
        ]

    def _connected_subsets_containing_not(
        self, graph, nodes: set[str], include: set[str], exclude: set[str]
    ):
        if include & exclude:
            return []

        rest = sorted(nodes - include - exclude)
        out = []
        for size in range(len(rest) + 1):
            for combo in combinations(rest, size):
                subset = include | set(combo)
                if len(subset) <= 1:
                    out.append(subset)
                    continue

                allowed = set(subset)
                start = next(iter(allowed))
                seen = {start}
                stack = [start]
                while stack:
                    current = stack.pop()
                    for edge in graph.get_edges():
                        source = _base_node(edge.get_source())
                        destination = _base_node(edge.get_destination())
                        if source not in allowed or destination not in allowed:
                            continue
                        if source == current and destination not in seen:
                            seen.add(destination)
                            stack.append(destination)
                        elif destination == current and source not in seen:
                            seen.add(source)
                            stack.append(source)
                if seen == allowed:
                    out.append(subset)
        return out

    def _momentum_cut_replacements(
        self,
        loop_graph,
        conservation_graph,
        repeated_ids: set[str],
        single_ids: set[str],
    ) -> list[tuple[str, Expression, Expression]]:
        if not repeated_ids:
            return []

        edge_by_id = {
            self._edge_local_id(edge): edge for edge in loop_graph.get_edges()
        }
        missing = sorted(repeated_ids - set(edge_by_id), key=_id_sort_key)
        if missing:
            raise ValueError(
                "Cannot rewrite repeated loop momenta; missing loop-core edges "
                f"{missing}."
            )

        reduced_graph_edges_deg2 = [
            edge_by_id[edge_id] for edge_id in sorted(repeated_ids, key=_id_sort_key)
        ]
        reduced_graph_edges_deg1 = [
            edge_by_id[edge_id]
            for edge_id in sorted(single_ids & set(edge_by_id), key=_id_sort_key)
        ]

        reduced_vertices = {
            _base_node(edge.get_source()) for edge in reduced_graph_edges_deg2
        }.union(_base_node(edge.get_destination()) for edge in reduced_graph_edges_deg2)
        if len(reduced_graph_edges_deg2) - len(reduced_vertices) + 1 > 0:
            raise ValueError(
                "Cannot rewrite repeated loop momenta: repeated edges contain a loop."
            )

        nodes = {
            _strip_quotes(str(node.get_name()))
            for node in loop_graph.get_nodes()
            if not self._is_ext_node(_strip_quotes(str(node.get_name())))
            and _strip_quotes(str(node.get_name())) not in ["node", "edge", "graph"]
        }
        blocked_once = {self._edge_local_id(edge) for edge in reduced_graph_edges_deg1}
        possible_cuts: list[list[tuple[str, set[str]]]] = []
        remaining = list(reduced_graph_edges_deg2)

        while remaining:
            chosen_edge = remaining[-1]
            chosen_id = self._edge_local_id(chosen_edge)
            source = _base_node(chosen_edge.get_source())
            destination = _base_node(chosen_edge.get_destination())
            total_cuts = self._connected_subsets_containing_not(
                loop_graph, nodes, {source}, {destination}
            )

            current_repeated = {self._edge_local_id(edge) for edge in remaining}
            next_possible: list[list[tuple[str, set[str]]]] = []
            seed_cuts = possible_cuts or [[]]
            for previous_cuts in seed_cuts:
                previous_boundary_ids = {
                    self._edge_local_id(edge)
                    for _previous_id, previous_cut in previous_cuts
                    for edge in self._boundary_edges_for_cut(
                        conservation_graph, previous_cut
                    )
                    if not self._edge_touches_ext(edge)
                }

                for cut_nodes in total_cuts:
                    boundary_ids = {
                        self._edge_local_id(edge)
                        for edge in self._boundary_edges_for_cut(
                            conservation_graph, cut_nodes
                        )
                    }
                    reduced_boundary_ids = boundary_ids - {chosen_id}
                    if reduced_boundary_ids & (current_repeated | blocked_once):
                        continue
                    if reduced_boundary_ids & previous_boundary_ids:
                        continue
                    next_possible.append(previous_cuts + [(chosen_id, cut_nodes)])

            if not next_possible:
                raise ValueError(
                    "Cannot find momentum-conservation cuts for repeated loop "
                    f"momentum edge {chosen_id}."
                )

            possible_cuts = next_possible
            remaining = remaining[:-1]

        selected_cuts = possible_cuts[0]
        replacements = []
        for edge_id, cut_nodes in selected_cuts:
            target = edge_by_id[edge_id]
            boundary = [
                edge
                for edge in self._boundary_edges_for_cut(
                    conservation_graph, cut_nodes
                )
                if self._edge_local_id(edge) != edge_id
            ]
            if not boundary:
                raise ValueError(
                    f"Momentum-conservation cut for edge {edge_id} has empty boundary."
                )

            replacement = E("0")
            plain_replacement = E("0")
            for edge in sorted(boundary, key=self._edge_sort_key):
                sign = 1 if _base_node(edge.get_source()) in cut_nodes else -1
                replacement += sign * Es(
                    f"Q({self._edge_local_id(edge)},y___)"
                )
                plain_replacement += sign * E(
                    f"Q({self._edge_local_id(edge)},y___)"
                )

            if _base_node(target.get_source()) in cut_nodes:
                replacement = -replacement
                plain_replacement = -plain_replacement

            replacements.append((edge_id, Es(f"Q({edge_id},y___)"), replacement))
            replacements.append(
                (edge_id, E(f"Q({edge_id},y___)"), plain_replacement)
            )

        return replacements

    def _apply_q_replacements(
        self, expr: Expression, replacements: list[tuple[str, Expression, Expression]]
    ) -> Expression:
        out = expr
        for _edge_id, pattern, replacement in replacements:
            out = out.replace(pattern, replacement)
        return out

    def _dot_level_repeated_momentum_graphs(
        self, graph, cut_graph=None
    ) -> list | None:
        loop_core = self._loop_core_for_numerator_rewrite(graph, cut_graph)
        if loop_core is None:
            return None

        loop_graph, loop_edge_ids, loop_nodes, _loop_count = loop_core
        vertices = self._loop_three_gluon_vertices(graph, loop_nodes)
        if len(vertices) < 2:
            return None

        rewritten_graphs = []
        for product_choices in product(*(choices for _node, choices in vertices)):
            chosen_edge_ids = [edge_id for edge_id, _expr in product_choices]
            loop_choice_counts = Counter(
                edge_id for edge_id in chosen_edge_ids if edge_id in loop_edge_ids
            )
            repeated_ids = {
                edge_id for edge_id, count in loop_choice_counts.items() if count > 1
            }
            single_ids = {
                edge_id for edge_id, count in loop_choice_counts.items() if count == 1
            }

            replacements = self._momentum_cut_replacements(
                loop_graph,
                graph,
                repeated_ids,
                single_ids,
            )
            replacements_by_edge: dict[str, list[tuple[str, Expression, Expression]]] = {}
            for edge_id, pattern, replacement in replacements:
                replacements_by_edge.setdefault(edge_id, []).append(
                    (edge_id, pattern, replacement)
                )
            replacement_budget = {
                edge_id: count - 1
                for edge_id, count in loop_choice_counts.items()
                if count > 1
            }
            graph_copy = deepcopy(graph)
            node_by_name = {
                _strip_quotes(str(node.get_name())): node
                for node in graph_copy.get_nodes()
            }

            for (node_name, _choices), (_edge_id, choice_expr) in zip(
                vertices, product_choices, strict=True
            ):
                vertex_replacements = []
                if replacement_budget.get(_edge_id, 0) > 0:
                    vertex_replacements = replacements_by_edge.get(_edge_id, [])
                    replacement_budget[_edge_id] -= 1
                rewritten_expr = self._apply_q_replacements(
                    choice_expr, vertex_replacements
                )
                node_by_name[node_name].get_attributes()["num"] = expr_to_string(
                    rewritten_expr
                )

            rewritten_graphs.append(graph_copy)

        return rewritten_graphs

    def _factorised_prepared_numerator(
        self,
        numerator_graph,
        post_momentum_rewrite_factor,
        on_shell_replacements,
    ) -> Expression:
        out = E("0")
        for parts in _factorised_graph_numerator_branches(numerator_graph):
            scalar, colour, kinematic = _multiply_into_numerator_parts(
                parts,
                post_momentum_rewrite_factor,
            )

            colour = _normalise_dy_colour_invariants(
                simplify_color(_prepare_dy_colour_for_simplification(colour))
            )
            kinematic = to_dots_dy(simplify_gamma(kinematic))
            kinematic = _dots_to_dy_scalar_products(kinematic)
            kinematic = _normalise_routed_numerator_on_shell(
                kinematic,
                on_shell_replacements,
            )

            branch = scalar * colour * kinematic
            branch = _strip_namespaces_structurally(branch)
            out += substitute_process_couplings(branch, self.name, self.L)
        return out

    def factorised_numerator_branches(
        self,
        numerator_graph,
        post_momentum_rewrite_factor=E("1"),
    ) -> list[tuple[Expression, Expression, Expression]]:
        """Return scalar/colour/kinematic branches without simplifying them.

        Integrated UV tensor closure needs the same branch separation as the
        ordinary numerator path, but it has its own Lorentz and gamma closure.
        Exposing the already-audited splitter here avoids sending expanded,
        colourless kinematics through Idenso's colour simplifier while still
        preserving correlated four-gluon colour/kinematic branches.
        """

        return [
            _multiply_into_numerator_parts(
                parts,
                post_momentum_rewrite_factor,
            )
            for parts in _factorised_graph_numerator_branches(numerator_graph)
        ]

    # Get the numerator of the graph

    def get_numerator(
        self, graph, numerator_factorisation=None, cut_graph=None
    ) -> Expression:
        symmetry_factor = Es(graph.get("overall_factor_evaluated"))
        on_shell_replacements = _routed_numerator_on_shell_replacements(
            cut_graph
        )

        numerator_graph = graph
        post_momentum_rewrite_factor = E("1")
        self._protected_external_gluon_polarisation_energy_ids = set()
        if numerator_factorisation is not None:
            factorisation_result = numerator_factorisation(graph)
            protected_energy_ids = getattr(
                factorisation_result,
                "protected_energy_ids",
                set(),
            )
            if len(factorisation_result) == 2:
                numerator_graph, post_momentum_rewrite_factor = factorisation_result
            elif len(factorisation_result) == 3:
                (
                    numerator_graph,
                    post_momentum_rewrite_factor,
                    protected_energy_ids,
                ) = factorisation_result
            else:
                raise ValueError(
                    "numerator_factorisation must return graph/factor or "
                    "graph/factor/protected_energy_ids"
                )
            graph_protected_energy_ids = numerator_graph.get(
                _EXTERNAL_GLUON_POLARISATION_PROTECTED_IDS_ATTR
            )
            if graph_protected_energy_ids:
                protected_energy_ids = set(protected_energy_ids)
                protected_energy_ids.update(
                    edge_id
                    for edge_id in _strip_quotes(str(graph_protected_energy_ids)).split(
                        ","
                    )
                    if edge_id
                )
            self._protected_external_gluon_polarisation_energy_ids = {
                _strip_quotes(str(edge_id)) for edge_id in protected_energy_ids
            }

        rewritten_graphs = self._dot_level_repeated_momentum_graphs(
            numerator_graph, cut_graph
        )
        if rewritten_graphs is None:
            out = self._factorised_prepared_numerator(
                numerator_graph,
                post_momentum_rewrite_factor,
                on_shell_replacements,
            )
        else:
            out = E("0")
            for rewritten_graph in rewritten_graphs:
                out += self._factorised_prepared_numerator(
                    rewritten_graph,
                    post_momentum_rewrite_factor,
                    on_shell_replacements,
                )

        return symmetry_factor * out

    # Get cff of a graph; the dependence on subgraph_as_nodes and reversed_edge_flows_ids
    # is explicit but is not used for the rest of the code. All graphs are amplitude graphs now.

    def get_CFF(
        self, graph, subgraph_as_nodes, reversed_edge_flows_ids
    ) -> CFFStructure:
        graph_for_cff = deepcopy(graph)
        self.canonicalize_ports_for_cff(graph_for_cff)

        cff_structure = self.gl_worker.generate_cff_as_json_string(
            dot_string=graph_for_cff.to_string(),
            subgraph_nodes=subgraph_as_nodes,
            reverse_dangling=reversed_edge_flows_ids,
            orientation_pattern=None,
        )

        try:
            cff_structure = json.loads(cff_structure)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding CFF structure JSON: {e}") from e

        cff_structure = CFFStructure(cff_structure)

        return cff_structure

    # The following function canonicalises the nodes and the ports of a pydot graph so that they go from 0,...,N and 0,...,M.
    # This normalisation is needed for linnet. One could also remove the ports.

    def canonicalize_ports_for_cff(self, graph):
        port_re = re.compile(r"^([^:]+):(\d+)$")

        def parse_endpoint(endpoint):
            ep = _strip_quotes(str(endpoint))
            m = port_re.fullmatch(ep)
            if not m:
                return ep, None
            return m.group(1), int(m.group(2))

        # Build a deterministic node relabeling map: old label -> "0", "1", ..., "N".
        node_labels = []
        for n in graph.get_nodes():
            name = _strip_quotes(str(n.get_name()))
            if name in ["node", "edge", "graph"]:
                continue
            node_labels.append(name)

        for e in graph.get_edges():
            src_node, _ = parse_endpoint(e.get_source())
            dst_node, _ = parse_endpoint(e.get_destination())
            node_labels.append(src_node)
            node_labels.append(dst_node)
            e.get_attributes().pop("lmb_id", None)

        unique_labels = list(dict.fromkeys(node_labels))
        node_map = {old: str(i) for i, old in enumerate(unique_labels)}

        remapped_nodes = []
        for n in graph.get_nodes():
            name = _strip_quotes(str(n.get_name()))
            if name in ["node", "edge", "graph"]:
                continue
            attrs = deepcopy(n.get_attributes())
            remapped_nodes.append(pydot.Node(node_map[name], **attrs))

        remapped_edges = []
        next_port = 0
        for e in graph.get_edges():
            attrs = deepcopy(e.get_attributes())

            src_node, src_port = parse_endpoint(e.get_source())
            src_node = node_map[src_node]
            if src_port is None:
                new_src = src_node
            else:
                new_src = f"{src_node}:{next_port}"
                next_port += 1

            dst_node, dst_port = parse_endpoint(e.get_destination())
            dst_node = node_map[dst_node]
            if dst_port is None:
                new_dst = dst_node
            else:
                new_dst = f"{dst_node}:{next_port}"
                next_port += 1

            remapped_edges.append(pydot.Edge(new_src, new_dst, **attrs))

        graph.obj_dict["nodes"] = {}
        graph.obj_dict["edges"] = {}
        for n in remapped_nodes:
            graph.add_node(n)
        for e in remapped_edges:
            graph.add_edge(e)

    # The following function can be used to remove unwanted attributes from the graph.

    def normalise_graph(self, graph):

        for e in graph.get_edges():
            e_atts = e.get_attributes()
            if e_atts.get("is_cut", 0) != 0:
                e_atts["is_cut_DY"] = e_atts["is_cut"]
            e_atts.pop("is_cut", None)
            e_atts.pop("source", None)
            e_atts.pop("num", None)
            e_atts.pop("sink", None)
            e_atts.pop("is_dummy", None)
            e_atts.pop("dir_in_cycle", None)

    # The following function takes a cut graph and gives out the two amplitude graphs which, glued together
    # give back the origi   q   nal cut graph. In order to do so it has to check if edges of the cut_graph are contained
    # in one or the other graph, if they are "externals", or if they are spectators.
    # Edge ids are normalised to that they go from 0,...,M

    def get_amplitude_graphs(self, cut_graph):
        removed_edges = [
            edge
            for edge in cut_graph.graph.get_edges()
            if _cut_sign_from_attributes(edge.get_attributes()) != 0
        ]
        comps = get_components(
            cut_graph.graph,
            removed_edges,
        )

        new_graphs = [deepcopy(cut_graph.graph) for _ in comps]

        highest_ext = 0
        for v in cut_graph.graph.get_nodes():
            name = _strip_quotes(v.get_name())
            if name.startswith("ext"):
                suffix = name[3:]
                if suffix.isdigit():
                    highest_ext = max(highest_ext, int(suffix))

        # TODO: should check indexing logic, it seems a bit contrived.

        tot_e = 0
        replacements = [[] for _ in comps]

        for i in range(len(comps)):
            counter = 1
            for e in cut_graph.graph.get_edges():
                e_atts = e.get_attributes()
                src = e.get_source()
                dest = e.get_destination()
                src_key = _node_key(e.get_source())
                dest_key = _node_key(e.get_destination())
                if src_key not in comps[i] and dest_key not in comps[i]:
                    new_graphs[i].del_edge(src, dest, int(e_atts["id"]))
                elif src_key in comps[i] and dest_key not in comps[i]:
                    new_graphs[i].del_edge(src, dest, int(e_atts["id"]))
                    new_atts = deepcopy(e_atts)
                    new_atts["id"] = tot_e + counter - 1
                    new_graphs[i].add_edge(
                        pydot.Edge(src, f"ext{highest_ext + counter}", **new_atts)
                    )
                    new_graphs[i].add_node(
                        pydot.Node(f"ext{highest_ext + counter}", style="invis")
                    )
                    replacements[i].append([
                        tot_e + counter - 1,
                        e_atts["id"],
                    ])

                    counter += 1
                elif dest_key in comps[i] and src_key not in comps[i]:
                    new_graphs[i].del_edge(src, dest, int(e_atts["id"]))
                    new_atts = deepcopy(e_atts)
                    new_atts["id"] = tot_e + counter - 1
                    new_graphs[i].add_edge(
                        pydot.Edge(f"ext{highest_ext + counter}", dest, **new_atts)
                    )
                    new_graphs[i].add_node(
                        pydot.Node(f"ext{highest_ext + counter}", style="invis")
                    )
                    replacements[i].append([
                        tot_e + counter - 1,
                        e_atts["id"],
                    ])
                    counter += 1
                elif (
                    dest_key in comps[i]
                    and src_key in comps[i]
                    and _cut_sign_from_attributes(e_atts) != 0
                    and not (dest_key.startswith("ext") or src_key.startswith("ext"))
                ):
                    new_graphs[i].del_edge(src, dest, int(e_atts["id"]))
                    new_atts1 = deepcopy(e_atts)
                    new_atts1["id"] = tot_e + counter - 1
                    new_graphs[i].add_edge(
                        pydot.Edge(f"ext{highest_ext + counter}", dest, **new_atts1)
                    )
                    new_graphs[i].add_node(
                        pydot.Node(f"ext{highest_ext + counter}", style="invis")
                    )
                    replacements[i].append([
                        tot_e + counter - 1,
                        e_atts["id"],
                    ])
                    counter += 1
                    new_atts2 = deepcopy(e_atts)
                    new_atts2["id"] = tot_e + counter - 1
                    new_graphs[i].add_edge(
                        pydot.Edge(src, f"ext{highest_ext + counter}", **new_atts2)
                    )
                    new_graphs[i].add_node(
                        pydot.Node(f"ext{highest_ext + counter}", style="invis")
                    )
                    replacements[i].append([
                        tot_e + counter - 1,
                        e_atts["id"],
                    ])
                    counter += 1
                else:
                    new_graphs[i].del_edge(src, dest, int(e_atts["id"]))
                    new_atts1 = deepcopy(e_atts)
                    new_atts1["id"] = tot_e + counter - 1
                    new_graphs[i].add_edge(pydot.Edge(src, dest, **new_atts1))
                    replacements[i].append([tot_e + counter - 1, e_atts["id"]])
                    counter += 1

            edge_nodes = set()
            for e in new_graphs[i].get_edges():
                edge_nodes.add(_node_key(e.get_source()))
                edge_nodes.add(_node_key(e.get_destination()))
            for v in list(new_graphs[i].get_nodes()):
                if _node_key(v.get_name()) not in edge_nodes:
                    new_graphs[i].del_node(v)

        return [
            amplitude_graph(graph, graph_replacements)
            for graph, graph_replacements in zip(new_graphs, replacements)
        ]

    # This function takes an amplitude graph and makes the composition of its replacements with
    # those of old replacements. In other words, indexes are propagated through the two replacements
    # and the final result is a single replacement expressing this chain of replacements

    def update_substitutions(self, graph: amplitude_graph, old_replacements):
        old_map = {src: dst for src, dst in old_replacements}
        composed = []
        for src, mid in graph.replacements:
            composed.append([src, old_map.get(mid, mid)])
        graph.replacements = composed

    # This function finds s_channel propagators and iteratively divides the amplitude graph into
    # amplitude subgraphs by deleting the s_channel propagators (which must be bridges). This is
    # needed because in the rest-frame the cff representation of massless s-channel propagators
    # is ill-defined. The amplitude subgraphs obtained by this procedure will later be individually fed to
    # the cff generator and the s-channel propagators will be added back by and in non-partial fractioned
    # form.

    def split_s_channels(self, graph: amplitude_graph):

        # Find s-channel propagators

        s_channel_edges = []
        for e in graph.graph.get_edges():
            e_atts = e.get_attributes()
            loop_keys = [f"routing_k{i}" for i in range(0, self.L)]
            s_channel = (
                all(e_atts[key] == "0" for key in loop_keys)
                and e_atts["routing_p1"] != "0"
                and e_atts["routing_p2"] != "0"
            )
            if s_channel:
                s_channel_edges.append(e)

        # Goes through one s-channel propagator at a time and divides the in two more subgraph by cutting it.

        s_channel_edges_copy = deepcopy(s_channel_edges)
        s_split_graphs = [graph]
        while len(s_channel_edges) > 0:
            chosen_s_edge = s_channel_edges.pop()
            chosen_src = chosen_s_edge.get_source()
            chosen_dest = chosen_s_edge.get_destination()
            check = False
            for g in s_split_graphs:
                for e in g.graph.get_edges():
                    src = e.get_source()
                    dest = e.get_destination()
                    if src == chosen_src and dest == chosen_dest:
                        s_cut_graph = routed_cut_graph(g.graph, [e], [], [])
                        split_graphs = self.get_amplitude_graphs(s_cut_graph)
                        if len(split_graphs) != 2:
                            raise ValueError(
                                "An s-channel bridge must split an amplitude "
                                f"into 2 components, got {len(split_graphs)}."
                            )
                        s_split_graphs.remove(g)
                        for split_graph in split_graphs:
                            self.update_substitutions(
                                split_graph, g.replacements
                            )
                            s_split_graphs.append(split_graph)
                        check = True
                        break
                if check:
                    break

        return s_split_graphs, s_channel_edges_copy

    def _edge_local_id(self, edge) -> str:
        return _strip_quotes(str(edge.get_attributes()["id"]))

    def _replacement_map(self, graph: amplitude_graph) -> dict[str, object]:
        return {_strip_quotes(str(src)): dst for src, dst in graph.replacements}

    def _edge_sort_key(self, edge):
        edge_id = self._edge_local_id(edge)
        try:
            return (0, int(edge_id))
        except ValueError:
            return (1, edge_id)

    def _is_ext_node(self, node: str) -> bool:
        node = _strip_quotes(str(node))
        return node.startswith("ext") and node[3:].isdigit()

    def _edge_touches_ext(self, edge) -> bool:
        return self._is_ext_node(_base_node(edge.get_source())) or self._is_ext_node(
            _base_node(edge.get_destination())
        )

    def _edge_cut_sign(self, edge) -> int:
        return _cut_sign_from_attributes(edge.get_attributes())

    def _components_without_edge(self, graph, removed_edge_id: str | None = None):
        nodes = []
        for edge in graph.get_edges():
            nodes.append(_base_node(edge.get_source()))
            nodes.append(_base_node(edge.get_destination()))
        nodes = sorted(set(nodes))
        if not nodes:
            return []

        adj = {node: set() for node in nodes}
        for edge in graph.get_edges():
            if (
                removed_edge_id is not None
                and self._edge_local_id(edge) == removed_edge_id
            ):
                continue
            source = _base_node(edge.get_source())
            destination = _base_node(edge.get_destination())
            adj.setdefault(source, set()).add(destination)
            adj.setdefault(destination, set()).add(source)

        components = []
        seen = set()
        for node in nodes:
            if node in seen:
                continue
            stack = [node]
            seen.add(node)
            component = {node}
            while stack:
                current = stack.pop()
                for neighbour in adj.get(current, ()):
                    if neighbour not in seen:
                        seen.add(neighbour)
                        component.add(neighbour)
                        stack.append(neighbour)
            components.append(component)
        return components

    def _tree_edges(self, graph: amplitude_graph) -> list:
        tree_edges = []
        for edge in sorted(graph.graph.get_edges(), key=self._edge_sort_key):
            if self._edge_touches_ext(edge):
                continue
            components = self._components_without_edge(
                graph.graph, self._edge_local_id(edge)
            )
            if len(components) > 1:
                tree_edges.append(edge)
        return tree_edges

    def _component_graphs_without_tree_edges(
        self, graph: amplitude_graph, tree_edges: list
    ) -> list[amplitude_graph]:
        tree_ids = {self._edge_local_id(edge) for edge in tree_edges}
        retained_edges = [
            edge
            for edge in graph.graph.get_edges()
            if self._edge_local_id(edge) not in tree_ids
        ]
        if not retained_edges:
            return []

        nodes = sorted(
            {
                _base_node(endpoint)
                for edge in retained_edges
                for endpoint in (edge.get_source(), edge.get_destination())
            }
        )
        adj = {node: set() for node in nodes}
        for edge in retained_edges:
            source = _base_node(edge.get_source())
            destination = _base_node(edge.get_destination())
            adj.setdefault(source, set()).add(destination)
            adj.setdefault(destination, set()).add(source)

        components = []
        seen = set()
        for node in nodes:
            if node in seen:
                continue
            stack = [node]
            seen.add(node)
            component = {node}
            while stack:
                current = stack.pop()
                for neighbour in adj.get(current, ()):
                    if neighbour not in seen:
                        seen.add(neighbour)
                        component.add(neighbour)
                        stack.append(neighbour)
            components.append(component)

        node_by_name = {
            _strip_quotes(str(node.get_name())): node for node in graph.graph.get_nodes()
        }
        replacement_map = self._replacement_map(graph)
        highest_ext = 0
        for node in graph.graph.get_nodes():
            name = _strip_quotes(str(node.get_name()))
            if name.startswith("ext") and name[3:].isdigit():
                highest_ext = max(highest_ext, int(name[3:]))
        for edge in graph.graph.get_edges():
            for endpoint in (edge.get_source(), edge.get_destination()):
                node = _base_node(endpoint)
                if node.startswith("ext") and node[3:].isdigit():
                    highest_ext = max(highest_ext, int(node[3:]))

        loop_graphs = []
        for component in components:
            component_edges = [
                edge
                for edge in retained_edges
                if _base_node(edge.get_source()) in component
                and _base_node(edge.get_destination()) in component
            ]
            non_ext_nodes = {
                node for node in component if not self._is_ext_node(_base_node(node))
            }
            internal_edges = [
                edge for edge in component_edges if not self._edge_touches_ext(edge)
            ]
            if len(non_ext_nodes) <= 1 or len(internal_edges) < len(non_ext_nodes):
                continue

            boundary_tree_edges = []
            for edge in tree_edges:
                source = _base_node(edge.get_source())
                destination = _base_node(edge.get_destination())
                if (source in component) != (destination in component):
                    boundary_tree_edges.append(edge)

            graph_type = graph.graph.get_type() or "digraph"
            new_graph = pydot.Dot(graph_type=graph_type)
            for key, value in graph.graph.get_attributes().items():
                new_graph.set(key, value)
            for node_name in sorted(component):
                node = node_by_name.get(node_name)
                if node is None:
                    new_graph.add_node(pydot.Node(node_name))
                else:
                    new_graph.add_node(deepcopy(node))

            replacements = []
            ext_counter = 1
            for new_id, edge in enumerate(
                sorted(component_edges + boundary_tree_edges, key=self._edge_sort_key)
            ):
                old_id = self._edge_local_id(edge)
                attrs = deepcopy(edge.get_attributes())
                attrs["id"] = new_id
                source = edge.get_source()
                destination = edge.get_destination()
                source_node = _base_node(source)
                destination_node = _base_node(destination)
                if source_node not in component:
                    source = f"ext{highest_ext + ext_counter}"
                    new_graph.add_node(pydot.Node(source, style="invis"))
                    ext_counter += 1
                if destination_node not in component:
                    destination = f"ext{highest_ext + ext_counter}"
                    new_graph.add_node(pydot.Node(destination, style="invis"))
                    ext_counter += 1
                new_graph.add_edge(
                    pydot.Edge(source, destination, **attrs)
                )
                replacements.append([new_id, replacement_map[old_id]])

            loop_graphs.append(amplitude_graph(new_graph, replacements))
        return loop_graphs

    def _tree_boundary_energy_sum(
        self,
        graph: amplitude_graph,
        tree_edge,
        component: set[str],
        physical_cut_ids: set[str],
    ) -> tuple[Expression, int, set[str]]:
        replacement_map = self._replacement_map(graph)
        tree_id = self._edge_local_id(tree_edge)
        S = {node for node in component if not self._is_ext_node(node)}
        energy_sum = E("0")
        external_count = 0

        for edge in graph.graph.get_edges():
            edge_id = self._edge_local_id(edge)
            source = _base_node(edge.get_source())
            destination = _base_node(edge.get_destination())
            if (source in S) == (destination in S):
                continue
            if edge_id == tree_id:
                continue
            if not (self._is_ext_node(source) or self._is_ext_node(destination)):
                raise ValueError(
                    "Tree cut boundary contains a second internal edge "
                    f"{edge_id} while processing tree edge {tree_id}."
                )

            cut_sign = self._edge_cut_sign(edge)
            if cut_sign == 0:
                continue

            original_id = _strip_quotes(str(replacement_map[edge_id]))
            if original_id not in physical_cut_ids:
                continue
            if self._is_ext_node(source) and destination in S:
                energy_sum += cut_sign * E(f"En({original_id})")
                external_count += 1
            elif self._is_ext_node(destination) and source in S:
                energy_sum -= cut_sign * E(f"En({original_id})")
                external_count += 1

        return energy_sum, external_count, S

    def _tree_energy_evaluation(
        self,
        graph: amplitude_graph,
        tree_edge,
        physical_cut_ids: set[str],
    ) -> tuple[Expression, Expression, object]:
        tree_id = self._edge_local_id(tree_edge)
        components = self._components_without_edge(graph.graph, tree_id)
        source = _base_node(tree_edge.get_source())
        destination = _base_node(tree_edge.get_destination())
        ordered_components = sorted(
            components,
            key=lambda component: (
                0 if source in component else 1,
                0 if destination in component else 1,
                sorted(component),
            ),
        )

        candidates = [
            self._tree_boundary_energy_sum(
                graph,
                tree_edge,
                component,
                physical_cut_ids,
            )
            for component in ordered_components
        ]
        nonzero_candidates = [
            candidate
            for candidate in candidates
            if candidate[1] > 0
            and candidate[0].to_canonical_string() != "0"
        ]
        eligible_candidates = nonzero_candidates or [
            candidate for candidate in candidates if candidate[1] > 0
        ]
        if not eligible_candidates:
            raise ValueError(
                f"Could not determine external boundary for tree edge {tree_id}."
            )

        # Either side of a tree bridge is a valid momentum-conservation
        # surface, but promoted CFF cuts can leave one side with only a
        # partial physical boundary. Prefer the side carrying the most
        # original initial/final cut edges; keep component order as a stable
        # tie-breaker.
        energy_sum, _external_count, S = max(
            eligible_candidates,
            key=lambda candidate: candidate[1],
        )
        tree_sign = 1 if destination in S else -1
        replacement = -energy_sum if tree_sign == 1 else energy_sum
        original_id = self._replacement_map(graph)[tree_id]
        return energy_sum, replacement, original_id

    def _tree_energy_constraints(
        self, cut_graph, amplitude_graphs
    ) -> dict[str, tuple[Expression, Expression]]:
        constraints = {}
        physical_cut_ids = set().union(*_cut_external_energy_ids(cut_graph))
        protected_ids = {
            edge_id
            for pair in getattr(cut_graph, "raised_cut_pairs", ())
            for edge_id in (pair.cut_edge_id, pair.partner_edge_id)
        }
        external_energy_replacement = _energy_conservation_replacement(
            cut_graph, protected_ids=protected_ids
        )

        for graph in amplitude_graphs:
            for tree_edge in self._tree_edges(graph):
                energy_sum, energy_replacement, original_id = (
                    self._tree_energy_evaluation(
                        graph,
                        tree_edge,
                        physical_cut_ids,
                    )
                )
                original_id = _strip_quotes(str(original_id))

                denominator_energy_sum = energy_sum
                if external_energy_replacement is not None:
                    denominator_energy_sum = denominator_energy_sum.replace(
                        *external_energy_replacement
                    )

                if external_energy_replacement is not None:
                    energy_replacement = energy_replacement.replace(
                        *external_energy_replacement
                    )

                constraints[original_id] = (
                    denominator_energy_sum,
                    energy_replacement,
                )

        return constraints

    def _cff_scalar_product_energy_form(
        self,
        numerator: Expression,
        signable_edge_ids: set[str],
        cut_energy_signs: dict[str, int],
    ) -> Expression:
        numerator = numerator.replace(E("Q(x_,0)"), E("En(x_)"))
        numerator = numerator.replace(E("Qp(x_,0)"), E("En(x_)"))
        numerator = numerator.replace(
            E("spp(qp(x_),qp(y_))"),
            E("En(x_)*En(y_)-sp3(q(x_),q(y_))"),
        )
        numerator = numerator.replace(
            E("spp(qp(x_),y_)"), E("En(x_)*En(y_)-sp3(q(x_),q(y_))")
        )
        numerator = numerator.replace(
            E("spp(x_,qp(y_))"), E("En(x_)*En(y_)-sp3(q(x_),q(y_))")
        )
        numerator = numerator.replace(
            E("sp(x_,y_)"), E("En(x_)*En(y_)-sp3(q(x_),q(y_))")
        )

        for edge_id, cut_sign in sorted(
            cut_energy_signs.items(), key=lambda item: _id_sort_key(item[0])
        ):
            if cut_sign == -1:
                numerator = numerator.replace(E(f"En({edge_id})"), E(f"-En({edge_id})"))

        for edge_id in sorted(signable_edge_ids, key=lambda x: (len(x), x)):
            numerator = numerator.replace(
                E(f"En({edge_id})"), E(f"sigma({edge_id})*En({edge_id})")
            )

        for edge_id in sorted(
            self._protected_external_gluon_polarisation_energy_ids,
            key=_id_sort_key,
        ):
            beam_id = _EXTERNAL_GLUON_POLARISATION_BEAM_IDS.get(edge_id)
            if beam_id is None:
                numerator = numerator.replace(
                    E(f"En({edge_id})"),
                    E(f"(sp3(q({edge_id}),q({edge_id})))^(1/2)"),
                )
                continue
            numerator = numerator.replace(
                E(f"En({edge_id})"),
                E(f"(sp3(p({beam_id}),p({beam_id})))^(1/2)"),
            )
            numerator = numerator.replace(
                E(f"q({edge_id})"),
                E(f"p({beam_id})"),
            )

        numerator = numerator.replace(E("sigma(1000)"), E("1"))
        numerator = numerator.replace(E("sp3(q(1000), x___)"), E("0"))
        numerator = numerator.replace(E("sp3(x___, q(1000))"), E("0"))
        numerator = numerator.replace(E("En(1000)"), E("1"))
        return numerator

    def get_cff(
        self,
        cut_graph,
        amplitude_graphs,
        numerator,
        get_residues=False,
        cff_partition=None,
    ):
        split_gl085 = cff_partition == _GL085_CFF_PARTITION
        if cff_partition is not None and not split_gl085:
            raise ValueError(f"Unknown CFF partition request: {cff_partition}.")
        if split_gl085 and get_residues:
            raise ValueError("GL085 CFF splitting is not supported for residues.")
        if split_gl085 and not _is_gl085_physical_25_cut(cut_graph):
            raise ValueError(
                "The GL085 CFF split is restricted to physical initial {0,1}, "
                "final {2,5} cuts."
            )
        had_pairs = bool(getattr(cut_graph, "raised_cut_pairs", ()))
        detection_was_complete = bool(
            getattr(cut_graph, "raised_cut_detection_complete", False)
        )
        detected_pairs = self.identify_and_mark_raised_cuts(cut_graph)
        if detected_pairs and not (had_pairs or detection_was_complete):
            raise ValueError(
                "Raised cuts must be identified before constructing amplitude "
                "graphs for get_cff."
            )
        raised_energy_ids = {
            edge_id
            for pair in getattr(cut_graph, "raised_cut_pairs", ())
            for edge_id in (pair.cut_edge_id, pair.partner_edge_id)
        }
        external_energy_replacement = (
            _energy_conservation_replacement(
                cut_graph, protected_ids=raised_energy_ids
            )
            if raised_energy_ids
            else None
        )

        cut_g_edges = sorted(
            cut_graph.graph.get_edges(), key=lambda e: int(e.get_attributes()["id"])
        )
        cut_g_edge_by_id = {
            _strip_quotes(str(e.get_attributes()["id"])): e for e in cut_g_edges
        }

        tree_edges_by_graph = [
            (graph, self._tree_edges(graph)) for graph in amplitude_graphs
        ]
        squared_tree_ids = self._squared_tree_propagator_edge_ids(cut_graph)
        tree_original_ids = set()
        loop_graphs = []
        for graph, tree_edges in tree_edges_by_graph:
            for tree_edge in tree_edges:
                tree_original_ids.add(
                    _strip_quotes(
                        str(
                            self._replacement_map(graph)[
                                self._edge_local_id(tree_edge)
                            ]
                        )
                    )
                )
            loop_graphs.extend(
                self._component_graphs_without_tree_edges(graph, tree_edges)
            )

        signable_edge_ids = set()
        for loop_graph in loop_graphs:
            replacement_map = self._replacement_map(loop_graph)
            signable_edge_ids.update(
                _strip_quotes(str(replacement_map[self._edge_local_id(edge)]))
                for edge in loop_graph.graph.get_edges()
                if not self._edge_touches_ext(edge)
            )
        signable_edge_ids.difference_update(tree_original_ids)

        cut_energy_signs = {
            _strip_quotes(str(edge.get_attributes()["id"])): self._edge_cut_sign(edge)
            for edge in cut_graph.graph.get_edges()
            if self._edge_cut_sign(edge) != 0
        }
        numerator_energy_signs = _raised_numerator_energy_signs(
            cut_energy_signs,
            tuple(getattr(cut_graph, "raised_cut_pairs", ())),
        )

        tree_factor = E("1")
        delayed_tree_replacements = []
        tree_energy_constraints = self._tree_energy_constraints(
            cut_graph, amplitude_graphs
        )
        numerator = self._cff_scalar_product_energy_form(
            numerator, signable_edge_ids, numerator_energy_signs
        )
        external_cut_ids = set().union(*_cut_external_energy_ids(cut_graph))
        for graph, tree_edges in tree_edges_by_graph:
            for tree_edge in tree_edges:
                original_id = _strip_quotes(
                    str(
                        self._replacement_map(graph)[
                            self._edge_local_id(tree_edge)
                        ]
                    )
                )
                constraint = tree_energy_constraints[original_id]
                if isinstance(constraint, tuple):
                    denominator_energy_sum, energy_replacement = constraint
                else:
                    denominator_energy_sum = constraint
                    energy_replacement = constraint
                delayed_tree_replacements.append(
                    (original_id, denominator_energy_sum, energy_replacement)
                )
                tree_denominator = (
                    denominator_energy_sum**2 - E(f"En({original_id})") ** 2
                )
                if original_id in squared_tree_ids:
                    tree_factor *= (
                        E(f"En({original_id})")
                        / (2 * tree_denominator**2)
                    )
                else:
                    tree_factor *= E("1") / tree_denominator

        missing_squared_ids = squared_tree_ids - tree_original_ids
        if missing_squared_ids:
            raise ValueError(
                "Raised t-channel propagator squaring targets are not tree "
                "propagators: "
                + ", ".join(sorted(missing_squared_ids, key=_id_sort_key))
            )

        if split_gl085 and len(loop_graphs) != 1:
            raise ValueError(
                "The GL085 CFF split expects exactly one loop CFF component, "
                f"found {len(loop_graphs)}."
            )

        e_surfaces = set()
        bucket_names = _GL085_CFF_BUCKETS if split_gl085 else ("all",)
        previous_cffs = {bucket: numerator for bucket in bucket_names}
        cut_energy_reversal_ids = {
            _strip_quotes(str(edge.get_attributes()["id"]))
            for edge in cut_graph.graph.get_edges()
            if self._edge_cut_sign(edge) == -1
        }
        for n_graph, g in enumerate(loop_graphs):
            cff_g = self.get_CFF(g.graph, [], [])
            new_cffs = {bucket: E("0") for bucket in bucket_names}
            g_rep = g.replacements

            mapped_eta_expressions = {}
            for etas in cff_g.e_surfaces:
                eta = etas.expression
                for rep in g.replacements:
                    eta = eta.replace(
                        E(f"pygloop::E({rep[0]})"),
                        E(f"En({rep[1]})"),
                    )

                for edge_id in cut_energy_reversal_ids:
                    eta = eta.replace(E(f"En({edge_id})"), -E(f"En({edge_id})"))
                if external_energy_replacement is not None:
                    eta = eta.replace(*external_energy_replacement)
                mapped_eta_expressions[etas.id] = eta
                if get_residues:
                    e_surfaces.add(deepcopy(eta))

            cff_expression_partition = (
                _partition_gl085_cff_expressions(
                    cff_g,
                    mapped_eta_expressions,
                )
                if split_gl085
                else None
            )
            if cff_expression_partition is not None:
                live_scalar_buckets = _materialise_gl085_scalar_causal_buckets(
                    cff_expression_partition,
                    mapped_eta_expressions,
                )
                reference_scalar_buckets = (
                    _gl085_reference_scalar_causal_bucket_map()
                )
                mismatched_buckets = [
                    bucket
                    for bucket in _GL085_CFF_BUCKETS
                    if not _expression_is_zero(
                        live_scalar_buckets[bucket]
                        - reference_scalar_buckets[bucket]
                    )
                ]
                if mismatched_buckets:
                    raise ValueError(
                        "The live GL085 scalar CFF no longer matches its "
                        "topology-locked crossed-cut template in buckets: "
                        + ", ".join(mismatched_buckets)
                    )

            for cffterm in cff_g.expressions:
                causal_expressions = (
                    cff_expression_partition.expressions[cffterm.id]
                    if cff_expression_partition is not None
                    else {"all": cffterm.expression}
                )
                for bucket, causal_expression in causal_expressions.items():
                    if _expression_is_zero(causal_expression):
                        continue
                    cff_term = previous_cffs[bucket] * causal_expression

                    for o, i in zip(
                        cffterm.orientation, range(len(cffterm.orientation))
                    ):
                        id_in_original_graph = g_rep[i][1]
                        original_edge_id = _strip_quotes(str(id_in_original_graph))
                        original_edge = cut_g_edge_by_id.get(original_edge_id)
                        if original_edge is None:
                            raise ValueError(
                                f"Could not find original edge {original_edge_id} "
                                "in cut graph CFF lookup."
                            )
                        if o.is_reversed():
                            cff_term = cff_term.replace(
                                E(f"sigma({id_in_original_graph})"), E("-1")
                            )
                        if o.is_default():
                            cff_term = cff_term.replace(
                                E(f"sigma({id_in_original_graph})"), E("1")
                            )
                    for eta_id, eta in mapped_eta_expressions.items():
                        cff_term = cff_term.replace(
                            E(f"pygloop::η({eta_id})"), -eta
                        )
                    new_cffs[bucket] += cff_term

            previous_cffs = new_cffs

        for (
            original_id,
            denominator_energy_sum,
            energy_replacement,
        ) in delayed_tree_replacements:
            # A raised t-channel derivative used to act only on the
            # propagator-energy head: its numerator energy stayed on shell.
            # The direct-square replacement must preserve that distinction
            # by protecting the identified edge, rather than by reviving
            # separate E/En symbol families.
            if (
                original_id not in external_cut_ids
                and original_id not in squared_tree_ids
            ):
                for bucket in bucket_names:
                    previous_cffs[bucket] = previous_cffs[bucket].replace(
                        E(f"En({original_id})"), energy_replacement
                    )
            if get_residues:
                e_surfaces = {
                    eta.replace(
                        E(f"En({original_id})"), denominator_energy_sum
                    )
                    for eta in e_surfaces
                }

        # Multiplies by inverse cut energies for non-tree propagators.

        energies = E("1")
        for e in cut_graph.graph.get_edges():
            e_atts = e.get_attributes()
            edge_id = _strip_quotes(str(e_atts["id"]))

            if edge_id not in tree_original_ids:
                energies *= E("1") / E(f"2*En({e_atts['id']})")

        if external_energy_replacement is not None:
            tree_factor = tree_factor.replace(*external_energy_replacement)
            energies = energies.replace(*external_energy_replacement)

        raised_crossing_weight = 1
        for _pair in getattr(cut_graph, "raised_cut_pairs", ()):
            raised_crossing_weight *= -2
        common_scalar_factor = tree_factor * energies * E(
            str(raised_crossing_weight)
        )
        total_cffs = {
            bucket: previous_cff * common_scalar_factor
            for bucket, previous_cff in previous_cffs.items()
        }
        if get_residues:
            total_cff = total_cffs["all"]
            delta = E("δ")
            residues = []
            for eta in e_surfaces:
                eta_for_residue = eta.expand()
                energies = (
                    list(eta_for_residue)
                    if eta_for_residue.is_type(AtomType.Add)
                    else [eta_for_residue]
                )
                eN = eta_for_residue.replace(E("En(x___)"), E("1"))
                if len(energies) > 1 and eN < len(energies):
                    pivot = energies[0]
                    # if pivot.match(E("-E(x___)")) is not None:
                    if pivot.format_plain().lstrip().startswith("-"):
                        patt = pivot.replace(E("-En(x___)"), E("En(x___)"))
                        repl = sum(en for en in energies[1:]) - delta
                    else:
                        patt = pivot
                        repl = -sum(en for en in energies[1:]) + delta
                    res_i = deepcopy(total_cff)
                    res_i = (
                        res_i
                        .replace(patt, repl)
                        .series(delta, 0, -1)
                        .to_expression()
                        .replace(delta, E("1"))
                    )
                    res_i = _finalise_cff_momentum_heads(res_i)
                    residues.append((eta_for_residue, res_i))
                elif eN > len(energies):
                    raise ValueError("really weird stuff happening with e surfaces")
            return residues

        # Qr deliberately bypasses routing reversal, synthetic cut signs,
        # signable-loop orientation, protected-energy substitutions, and
        # residue pivots. It becomes En only at the completed-CFF boundary.
        total_cffs = {
            bucket: _finalise_cff_momentum_heads(total_cff)
            for bucket, total_cff in total_cffs.items()
        }

        # print(total_cff)

        return total_cffs if split_gl085 else total_cffs["all"]

    def get_integrand(
        self,
        cut_graph: routed_cut_graph,
        get_residues=False,
        numerator_factorisation=None,
        prepared_numerator=None,
        cff_partition=None,
    ):

        # Derives numerator, eliminates useless labels, get left and right graphs and further
        # splits them if they have s-channel propagators.

        print("got to num")
        self.identify_and_mark_raised_cuts(cut_graph)

        if prepared_numerator is not None:
            if numerator_factorisation is not None:
                raise ValueError(
                    "prepared_numerator and numerator_factorisation are mutually exclusive"
                )
            self._protected_external_gluon_polarisation_energy_ids = set()
            num = _normalise_routed_numerator_on_shell(
                prepared_numerator,
                _routed_numerator_on_shell_replacements(cut_graph),
            )
        else:
            num = self.get_numerator(
                cut_graph.graph,
                numerator_factorisation=numerator_factorisation,
                cut_graph=cut_graph,
            )

        _assert_no_unresolved_colour_heads(
            num,
            context=_cut_numerator_context(cut_graph),
        )

        # print(num.replace(E("sp(x_,y_)"),E("1")))

        print("and beyond num")

        # print("NUM before contraction:   " , num)

        self.normalise_graph(cut_graph.graph)

        amplitude_graphs = self.get_amplitude_graphs(cut_graph)

        ## DEBUG: set numerator to 1
        # num = E("1")
        # print("NUMERATORRRRRRRR")
        # print(num)

        print("got to cff construction")

        if cff_partition is None:
            # Preserve the historical four-argument call boundary for
            # downstream wrappers and lightweight test doubles.
            cut_graph_cff = self.get_cff(
                cut_graph,
                amplitude_graphs,
                num,
                get_residues,
            )
        else:
            cut_graph_cff = self.get_cff(
                cut_graph,
                amplitude_graphs,
                num,
                get_residues,
                cff_partition=cff_partition,
            )

        print("and beyond cff construction")

        return cut_graph_cff


class UltraVioletSubtraction(object):
    def __init__(
        self,
        emr_integrand,
        cut_graph,
        L,
        emr_processor=None,
        integrated_cut_graph=None,
        integrated_numerator_factorisation=None,
        disable_integrated_uv_cts=True,
        raw_emr_integrand=None,
        top_self_energy_os_subtraction=None,
        top_self_energy_renormalisation=None,
    ):
        self.cut_graph = cut_graph
        self.emr_integrand = emr_integrand
        self.raw_emr_integrand = raw_emr_integrand
        self.sp3D = S("sp3D", is_linear=True, is_symmetric=True)
        self.L = L
        self.emr_processor = emr_processor
        self.integrated_cut_graph = integrated_cut_graph
        self.integrated_numerator_factorisation = (
            integrated_numerator_factorisation
        )
        self.disable_integrated_uv_cts = bool(disable_integrated_uv_cts)
        self.top_self_energy_renormalisation = (
            resolve_top_self_energy_renormalisation(
                top_self_energy_renormalisation,
                top_self_energy_os_subtraction,
            )
        )
        # Keep the historical attribute faithful to its public meaning.  A
        # projected-os instance is active, but it did not originate from the
        # legacy boolean and must never be serialised as ordinary ``os``.
        self.top_self_energy_os_subtraction = (
            self.top_self_energy_renormalisation == "os"
        )
        if (
            self.top_self_energy_renormalisation != "no-os"
            and self.disable_integrated_uv_cts
        ):
            raise ValueError(
                "On-shell top self-energy subtraction requires integrated "
                "UV counterterms to be enabled."
            )

    @staticmethod
    def _edge_id(edge) -> str:
        return _strip_quotes(str(edge.get_attributes()["id"]))

    def _top_self_energy_context(self, cycle=(), boundary=()) -> str:
        graph_name = _strip_quotes(
            str(
                self.cut_graph.graph.get("base_graph_name")
                or self.cut_graph.graph.get_name()
            )
        )
        initial_ids = sorted(
            (self._edge_id(edge) for edge in self.cut_graph.initial_cut),
            key=_id_sort_key,
        )
        final_ids = sorted(
            (self._edge_id(edge) for edge in self.cut_graph.final_cut),
            key=_id_sort_key,
        )
        cycle_ids = sorted(
            (self._edge_id(edge) for edge in cycle),
            key=_id_sort_key,
        )
        boundary_ids = sorted(
            (self._edge_id(edge) for edge in boundary),
            key=_id_sort_key,
        )
        return (
            f"graph={graph_name}, initial_cut={initial_ids}, "
            f"final_cut={final_ids}, cycle={cycle_ids}, "
            f"boundary={boundary_ids}"
        )

    @staticmethod
    def _normalised_self_energy_particle(edge) -> str:
        particle = _strip_quotes(
            str(edge.get_attributes().get("particle", ""))
        )
        return "t" if particle in {"t", "t~"} else particle

    def _top_self_energy_boundary(self, cycle):
        particles = sorted(
            self._normalised_self_energy_particle(edge) for edge in cycle
        )
        if particles != ["g", "t"]:
            return None

        cycle_nodes = {
            _base_node(endpoint)
            for edge in cycle
            for endpoint in (edge.get_source(), edge.get_destination())
        }
        boundary = boundary_edges(self.cut_graph.graph, cycle_nodes)
        if len(boundary) != 2:
            return None
        if any(
            self._normalised_self_energy_particle(edge) != "t"
            for edge in boundary
        ):
            return None
        if _routing_sign_match(boundary[0], boundary[1]) is None:
            return None
        return tuple(boundary)

    def _select_top_self_energy_cycle(self, spinneys):
        if self.top_self_energy_renormalisation == "no-os":
            return None

        selected = []
        for cycle, dod in spinneys:
            boundary = self._top_self_energy_boundary(cycle)
            if boundary is not None:
                selected.append((cycle, dod, boundary))
        if len(selected) > 1:
            details = "; ".join(
                self._top_self_energy_context(cycle, boundary)
                for cycle, _dod, boundary in selected
            )
            raise ValueError(
                "Multiple eligible top self-energy cycles in one cut are "
                f"unsupported: {details}"
            )
        return selected[0] if selected else None

    def _rest_cycle_graph(self, cycle, boundary):
        rest_cut_graph = deepcopy(self.cut_graph)
        edge_by_id = {
            self._edge_id(edge): edge
            for edge in rest_cut_graph.graph.get_edges()
        }
        rest_cycle = tuple(edge_by_id[self._edge_id(edge)] for edge in cycle)
        context = self._top_self_energy_context(cycle, boundary)

        external_keys = sorted(
            {
                key
                for edge in rest_cut_graph.graph.get_edges()
                for key in edge.get_attributes()
                if key.startswith("routing_p")
            }
        )

        def is_pure_loop_coordinate(edge):
            attributes = edge.get_attributes()
            loop_coefficients = [
                Fraction(
                    _strip_quotes(
                        str(attributes.get(f"routing_k{index}", "0"))
                    )
                )
                for index in range(self.L)
            ]
            return (
                sum(coefficient != 0 for coefficient in loop_coefficients) == 1
                and any(abs(coefficient) == 1 for coefficient in loop_coefficients)
                and all(
                    Fraction(
                        _strip_quotes(str(attributes.get(key, "0")))
                    )
                    == 0
                    for key in external_keys
                )
            )

        preserved = [
            edge for edge in rest_cycle if is_pure_loop_coordinate(edge)
        ]
        if len(preserved) != 1:
            raise ValueError(
                "Expected exactly one pure UV loop-coordinate edge while "
                f"constructing the rest routing; found {len(preserved)}; "
                f"{context}"
            )
        preserved_edge = preserved[0]
        changed_edge = next(edge for edge in rest_cycle if edge is not preserved_edge)

        cycle_nodes = sorted(
            {
                _base_node(endpoint)
                for edge in rest_cycle
                for endpoint in (edge.get_source(), edge.get_destination())
            }
        )
        if len(cycle_nodes) != 2:
            raise ValueError(
                "A top self-energy rest routing requires a two-vertex cycle; "
                f"found nodes {cycle_nodes}; {context}"
            )
        reference_node = cycle_nodes[0]

        def incidence(edge):
            source = _base_node(edge.get_source())
            destination = _base_node(edge.get_destination())
            if source == reference_node and destination != reference_node:
                return Fraction(1)
            if destination == reference_node and source != reference_node:
                return Fraction(-1)
            raise ValueError(
                "Cycle incidence is ambiguous at reference node "
                f"{reference_node}; {context}"
            )

        opposite_factor = -incidence(preserved_edge) / incidence(changed_edge)
        routing_keys = [f"routing_k{index}" for index in range(self.L)]
        routing_keys.extend(external_keys)
        preserved_attributes = preserved_edge.get_attributes()
        changed_attributes = changed_edge.get_attributes()
        for key in routing_keys:
            coefficient = opposite_factor * Fraction(
                _strip_quotes(str(preserved_attributes.get(key, "0")))
            )
            changed_attributes[key] = (
                str(coefficient.numerator)
                if coefficient.denominator == 1
                else str(coefficient)
            )

        if _routing_sign_match(preserved_edge, changed_edge) != (
            "opp" if opposite_factor == -1 else "same"
        ):
            raise ValueError(
                "Failed to construct incidence-opposite rest routing; "
                f"{context}"
            )
        return rest_cut_graph, rest_cycle

    @staticmethod
    def _residual_overall_sign(residual: Expression) -> int | None:
        factored = residual.factor()
        if factored == E("0"):
            return None
        if bool(factored.is_type(AtomType.Mul)):
            factors = list(factored)
            if any(factor == E("-1") for factor in factors):
                return -1
        return 1

    def _replace_self_energy_causal_factors_at_rest(
        self,
        expression,
        cycle,
        boundary,
    ):
        cycle_ids = sorted(
            (self._edge_id(edge) for edge in cycle),
            key=_id_sort_key,
        )
        context = self._top_self_energy_context(cycle, boundary)
        if len(cycle_ids) != 2:
            raise ValueError(
                "A top self-energy causal replacement requires two cycle "
                f"edges; {context}"
            )
        first_energy, second_energy = (
            E(f"En({cycle_ids[0]})"),
            E(f"En({cycle_ids[1]})"),
        )
        denominator_base = S("top_se_denominator_base_")
        candidates = {}
        for match in expression.match(E("top_se_denominator_base_^-1")):
            base = match[denominator_base]
            residual = (base - first_energy - second_energy).expand()
            if residual.contains(first_energy) or residual.contains(second_energy):
                continue
            candidates.setdefault(
                base.to_canonical_string(),
                (base, residual),
            )
        if len(candidates) != 2:
            raise ValueError(
                "Expected exactly two unique reciprocal causal denominators "
                "containing both self-energy cycle energies with coefficient "
                f"+1; found {len(candidates)}; {context}"
            )

        candidate_values = list(candidates.values())
        if (
            candidate_values[0][1] + candidate_values[1][1]
        ).expand().to_canonical_string() != "0":
            raise ValueError(
                "Top self-energy causal residuals are not symbolic "
                f"opposites; residuals={[str(value[1]) for value in candidate_values]}; "
                f"{context}"
            )
        signs = [
            self._residual_overall_sign(residual)
            for _base, residual in candidate_values
        ]
        if None in signs or sorted(signs) != [-1, 1]:
            raise ValueError(
                "Could not inherit opposite signs from top self-energy causal "
                f"residuals; signs={signs}; {context}"
            )

        transformed = expression
        old_factors = []
        for (base, _residual), sign in zip(candidate_values, signs, strict=True):
            old_factor = base**E("-1")
            new_factor = (
                first_energy + second_energy + sign * E("MT")
            ) ** E("-1")
            old_factors.append(old_factor)
            transformed = transformed.replace(old_factor, new_factor)
        remaining = [factor for factor in old_factors if transformed.contains(factor)]
        if remaining:
            raise ValueError(
                "Failed to replace every old top self-energy causal factor; "
                f"remaining={[str(factor) for factor in remaining]}; {context}"
            )
        return transformed

    # Focuses on cycles, and not unions of cycles. Specialised to NLO
    def enumerate_spinneys(self):

        cycles = get_simple_cycles(self.cut_graph.graph)

        cut_edges = set(self.cut_graph.initial_cut).union(set(self.cut_graph.final_cut))
        divergent_cycles = []

        for cycle in cycles:
            cut_cycle_edges = set(cycle).intersection(cut_edges)
            if len(cut_cycle_edges) == 0:  ## FIX: SHOULD BE == 0
                dod = 0
                visited_nodes = set()

                for e in cycle:
                    e_atts = e.get_attributes()
                    dod += int(_strip_quotes(str(e_atts["dod"])))
                    visited_nodes.add(_node_key(e.get_source()))
                    visited_nodes.add(_node_key(e.get_destination()))

                for v in self.cut_graph.graph.get_nodes():
                    v_atts = v.get_attributes()
                    if _node_key(v.get_name()) in visited_nodes:
                        dod += int(_strip_quotes(str(v_atts["dod"])))

                if dod + 4 >= 0:
                    divergent_cycles.append([cycle, dod + 4])

        return divergent_cycles

    def replace_energies(self, integrand, graph):

        for e in graph.get_edges():
            e_atts = e.get_attributes()
            eid_raw = e_atts["id"]
            eid = _strip_quotes(eid_raw) if isinstance(eid_raw, str) else eid_raw
            target = E(f"En({eid})")
            particle = _strip_quotes(str(e_atts["particle"]))
            replacement = (
                self.sp3D(E(f"q({eid})"), E(f"q({eid})")) + E(f"m({eid})") ** 2
            ) ** E("1/2")
            integrand = integrand.replace(target, replacement)

        return integrand

    def route_integrand(self, integrand, graph):

        for e in graph.get_edges():
            e_atts = e.get_attributes()
            routing_items = E("0")
            for i in range(self.L + 1):
                key = f"routing_k{i}"
                if key in e_atts:
                    routing_items += E(f"{e_atts[key]}*k[{i}]")
            for i in range(0, 2):
                key = f"routing_p{i + 1}"
                if key in e_atts:
                    routing_items += E(f"{e_atts[key]}*p[{i + 1}]")
            integrand = integrand.replace(E(f"q({e_atts['id']})"), routing_items)

        integrand = integrand.replace(
            E("sp3(x___,y___)"), self.sp3D(S("x___"), S("y___"))
        )
        return integrand

    @staticmethod
    def _replace_edge_masses_with_particle_masses(integrand, graph):
        for edge in graph.get_edges():
            attributes = edge.get_attributes()
            particle = _strip_quotes(str(attributes["particle"]))
            mass = (
                E("0")
                if particle in ["d", "d~", "g", "ghG", "ghG~"]
                else E(f"m({particle})")
            )
            integrand = integrand.replace(
                E(f"m({attributes['id']})"),
                mass,
            )
        return integrand

    def _projected_top_self_energy_loop_momentum(
        self,
        cycle,
        boundary,
        uv_routing=None,
    ) -> Expression:
        """Return the fixed production UV coordinate used by both branches.

        The projector evaluates the two boundary-energy branches without a
        dummy-loop relabelling.  In particular, a massive propagator routed as
        ``-k(uv)+p_boundary`` at rest does *not* turn the open spatial basis
        into ``slash(-k(uv))``: the validated tensor is expressed in the
        unchanged positive UV coordinate.
        """

        external_keys = sorted(
            {
                key
                for edge in self.cut_graph.graph.get_edges()
                for key in edge.get_attributes()
                if key.startswith("routing_p")
            }
        )

        def routing_fraction(edge, key):
            return Fraction(
                _strip_quotes(str(edge.get_attributes().get(key, "0")))
            )

        def is_pure_loop_coordinate(edge):
            coefficients = [
                routing_fraction(edge, f"routing_k{index}")
                for index in range(self.L)
            ]
            return (
                sum(value != 0 for value in coefficients) == 1
                and any(abs(value) == 1 for value in coefficients)
                and all(routing_fraction(edge, key) == 0 for key in external_keys)
            )

        context = self._top_self_energy_context(cycle, boundary)
        massive_edges = [
            edge
            for edge in cycle
            if self._normalised_self_energy_particle(edge) == "t"
        ]
        if len(massive_edges) != 1:
            raise ValueError(
                "Expected one massive line in the projected top self-energy; "
                f"found {len(massive_edges)}; {context}"
            )

        common_loop_indices = set(range(self.L))
        for edge in cycle:
            common_loop_indices &= {
                loop_index
                for loop_index in range(self.L)
                if routing_fraction(edge, f"routing_k{loop_index}") != 0
            }
        if len(common_loop_indices) == 1:
            # The coordinate itself, not an individual edge's possibly
            # negative coefficient, defines the unrelabelled tensor basis.
            return E(f"k({next(iter(common_loop_indices))})")

        pure_edges = sorted(
            (edge for edge in cycle if is_pure_loop_coordinate(edge)),
            key=lambda edge: _id_sort_key(self._edge_id(edge)),
        )
        if pure_edges:
            reference_edge = pure_edges[0]
            loop_indices = [
                loop_index
                for loop_index in range(self.L)
                if routing_fraction(
                    reference_edge, f"routing_k{loop_index}"
                )
                != 0
            ]
            if len(loop_indices) == 1:
                return E(f"k({loop_indices[0]})")

        if uv_routing is not None:
            row = uv_routing.adapted_loop_matrix[uv_routing.uv_loop_index]
            nonzero_indices = [
                index for index, coefficient in enumerate(row)
                if Fraction(coefficient) != 0
            ]
            if len(nonzero_indices) == 1:
                return E(f"k({nonzero_indices[0]})")

        raise ValueError(
            "Could not identify one unrelabelled projected top self-energy "
            f"UV coordinate; {context}"
        )

    def _projected_top_self_energy_local_terms(
        self,
        cycle,
        dod,
        boundary,
        uv_routing,
    ) -> tuple[RoutedIntegrand, RoutedIntegrand]:
        """Embed the projected identity and spatial Dirac tensors in the host."""

        graph_edge_ids = {
            self._edge_id(edge) for edge in self.cut_graph.graph.get_edges()
        }
        if PROJECTED_OS_SPATIAL_PROBE_ID in graph_edge_ids:
            raise ValueError(
                "Projected top self-energy spatial probe collides with a "
                "production graph edge; "
                + self._top_self_energy_context(cycle, boundary)
            )

        identity_basis = self.construct_integrated_counter_term(
            cycle,
            dod,
            uv_routing,
            top_self_energy_renormalisation="projected-os",
            projected_os_basis="identity",
        )
        spatial_basis = self.construct_integrated_counter_term(
            cycle,
            dod,
            uv_routing,
            top_self_energy_renormalisation="projected-os",
            projected_os_basis="spatial",
        )
        if identity_basis is None or spatial_basis is None:
            raise ValueError(
                "Projected top self-energy could not construct both open "
                "Dirac host bases; "
                + self._top_self_energy_context(cycle, boundary)
            )

        loop_momentum = self._projected_top_self_energy_loop_momentum(
            cycle,
            boundary,
            uv_routing,
        )
        loop_sq = self.sp3D(loop_momentum, loop_momentum)
        normalising_tadpole = (
            E("mUV")
            / E("𝜋") ** 2
            / (loop_sq + E("mUV") ** 2) ** 2
        )

        spatial_host = spatial_basis.integrand.replace(
            E(f"En({PROJECTED_OS_SPATIAL_PROBE_ID})"), E("0"), repeat=True
        )
        spatial_host = spatial_host.replace(
            E(f"q({PROJECTED_OS_SPATIAL_PROBE_ID})"), loop_momentum, repeat=True
        )
        if spatial_host.contains(
            E(f"En({PROJECTED_OS_SPATIAL_PROBE_ID})")
        ) or spatial_host.contains(
            E(f"q({PROJECTED_OS_SPATIAL_PROBE_ID})")
        ):
            raise ValueError(
                "Projected top self-energy spatial slot was not fully routed; "
                + self._top_self_energy_context(cycle, boundary)
            )

        coefficients = _projected_top_self_energy_tensor_coefficients(loop_sq)

        def tensor_term(
            identity_coefficient: Expression,
            spatial_coefficient: Expression,
            approximation_type: str,
        ) -> RoutedIntegrand:
            term = deepcopy(identity_basis)
            term.integrand = (
                identity_basis.integrand
                * (-2 / E("𝜋"))
                * identity_coefficient
                / E("MT")
                / normalising_tadpole
                + spatial_host
                * (-2 / E("𝜋"))
                * spatial_coefficient
                / normalising_tadpole
            )
            term.emr_integrand = E("0")
            term.approximation_type = approximation_type
            term.ir_limit = approximation_type
            attach_uv_routing_metadata(
                term.cut_graph.graph,
                uv_routing,
                contracted=False,
            )
            return term

        return (
            tensor_term(
                -coefficients["pole_identity"],
                -coefficients["pole_spatial"],
                "os",
            ),
            tensor_term(
                coefficients["uv_identity"],
                coefficients["uv_spatial"],
                "uv_os",
            ),
        )

    # For now we construct the counter-term associated to a cycle. This ignores loop-induced.
    def construct_counter_term(
        self,
        cycle,
        dod,
        *,
        emr_integrand=None,
        cut_graph=None,
    ):

        lam = S("λ", is_scalar=True)
        mUV = E("mUV")
        source_integrand = (
            self.emr_integrand if emr_integrand is None else emr_integrand
        )
        source_cut_graph = self.cut_graph if cut_graph is None else cut_graph

        uv_routing, uv_graph = build_uv_subgraph_routing(
            source_cut_graph,
            cycle,
            self.L,
        )

        routed_integrand = self.replace_energies(source_integrand, uv_graph)
        routed_integrand = self.route_integrand(routed_integrand, uv_graph)

        parametrised_integrand = routed_integrand.replace(E("k(0)"), E("k(0)") / lam)

        for e in cycle:
            e_atts = e.get_attributes()
            # This mass substitution also acts on the numerator, which might be dangerous
            # On a second thought this might be needed to reproduce the 4D version of the UV ct
            parametrised_integrand = parametrised_integrand.replace(
                E(f"m({e_atts['id']})") ** 2,
                1 / lam**2 * mUV**2 + (E(f"m({e_atts['id']})") ** 2 - mUV**2),
            )

        parametrised_integrand = self._replace_edge_masses_with_particle_masses(
            parametrised_integrand,
            source_cut_graph.graph,
        )

        print("DOD" * 10)
        print(dod)
        dod = 0
        expanded_integrand = (
            (1 / lam**3 * parametrised_integrand)
            .series(lam, 0, dod)
            .to_expression()
            .replace(lam, E("1"))
        )

        expanded_integrand = remap_uv_expression_to_production(
            expanded_integrand,
            uv_routing,
        )
        return -expanded_integrand, uv_routing

    def construct_integrated_counter_term(
        self,
        cycle,
        dod,
        uv_routing,
        *,
        top_self_energy_os_subtraction=False,
        top_self_energy_renormalisation=None,
        projected_os_basis=None,
    ):
        if self.disable_integrated_uv_cts:
            return None

        return construct_integrated_uv_counter_term(
            self,
            cycle,
            dod,
            routed_cut_graph,
            RoutedIntegrand,
            _cleanup_final_state_raised_energies,
            uv_routing,
            external_numerator_factorisation=(
                self.integrated_numerator_factorisation
            ),
            top_self_energy_os_subtraction=(
                top_self_energy_os_subtraction
            ),
            top_self_energy_renormalisation=(
                top_self_energy_renormalisation
            ),
            projected_os_basis=projected_os_basis,
        )

    def construct_uv_counter_terms(self):
        spinneys = self.enumerate_spinneys()
        counterms = []
        selected = self._select_top_self_energy_cycle(spinneys)
        selected_cycle_ids = (
            {self._edge_id(edge) for edge in selected[0]}
            if selected is not None
            else set()
        )
        if selected is not None and self.raw_emr_integrand is None:
            raise ValueError(
                "On-shell top self-energy subtraction requires the raw "
                "pre-cleanup EMR expression; "
                + self._top_self_energy_context(selected[0], selected[2])
            )

        # check copies and deepcopies
        for cycle in spinneys:
            uv_ct, uv_routing = self.construct_counter_term(cycle[0], cycle[1])
            uv_cut_graph = deepcopy(self.cut_graph)
            attach_uv_routing_metadata(
                uv_cut_graph.graph,
                uv_routing,
                contracted=False,
            )
            routed_uv_ct = RoutedIntegrand(
                uv_ct, uv_cut_graph, [], self.emr_integrand, "uv", "uv"
            )
            counterms.append(routed_uv_ct)
            is_selected = (
                bool(selected_cycle_ids)
                and {self._edge_id(edge) for edge in cycle[0]}
                == selected_cycle_ids
            )
            if is_selected and self.top_self_energy_renormalisation == "projected-os":
                boundary = selected[2]
                partition = tuple(getattr(self.cut_graph, "partition", ()))
                if (
                    len(partition) != 2
                    or any(len(side) != 1 for side in partition)
                ):
                    raise ValueError(
                        "Projected top self-energy subtraction currently "
                        "supports only the ordinary one-edge-per-side virtual "
                        "partition; "
                        + self._top_self_energy_context(cycle[0], boundary)
                    )
                counterms.extend(
                    self._projected_top_self_energy_local_terms(
                        cycle[0],
                        cycle[1],
                        boundary,
                        uv_routing,
                    )
                )
            if is_selected and self.top_self_energy_renormalisation == "os":
                boundary = selected[2]
                partition = tuple(getattr(self.cut_graph, "partition", ()))
                if (
                    len(partition) != 2
                    or any(len(side) != 1 for side in partition)
                ):
                    raise ValueError(
                        "On-shell top self-energy subtraction currently "
                        "supports only the ordinary one-edge-per-side virtual "
                        "partition; "
                        + self._top_self_energy_context(cycle[0], boundary)
                    )
                rest_cut_graph, rest_cycle = self._rest_cycle_graph(
                    cycle[0],
                    boundary,
                )
                rest_emr = self._replace_self_energy_causal_factors_at_rest(
                    self.raw_emr_integrand,
                    cycle[0],
                    boundary,
                )
                _physical_cleaned, physical_is_final_raised = (
                    _cleanup_final_state_raised_energies(
                        self.raw_emr_integrand,
                        self.cut_graph,
                    )
                )
                rest_emr, rest_is_final_raised = (
                    _cleanup_final_state_raised_energies(
                        rest_emr,
                        rest_cut_graph,
                    )
                )
                if rest_is_final_raised != physical_is_final_raised:
                    raise ValueError(
                        "Physical and rest top self-energy expressions have "
                        "different final-state raised status; "
                        + self._top_self_energy_context(cycle[0], boundary)
                    )

                routed_rest = self.replace_energies(
                    rest_emr,
                    rest_cut_graph.graph,
                )
                routed_rest = self.route_integrand(
                    routed_rest,
                    rest_cut_graph.graph,
                )
                routed_rest = self._replace_edge_masses_with_particle_masses(
                    routed_rest,
                    rest_cut_graph.graph,
                )
                counterms.append(
                    RoutedIntegrand(
                        -routed_rest,
                        rest_cut_graph,
                        [],
                        rest_emr,
                        "os",
                        "os",
                    )
                )

                rest_uv_ct, rest_uv_routing = self.construct_counter_term(
                    rest_cycle,
                    cycle[1],
                    emr_integrand=rest_emr,
                    cut_graph=rest_cut_graph,
                )
                attach_uv_routing_metadata(
                    rest_cut_graph.graph,
                    rest_uv_routing,
                    contracted=False,
                )
                rest_uv_cut_graph = deepcopy(rest_cut_graph)
                attach_uv_routing_metadata(
                    rest_uv_cut_graph.graph,
                    rest_uv_routing,
                    contracted=False,
                )
                counterms.append(
                    RoutedIntegrand(
                        -rest_uv_ct,
                        rest_uv_cut_graph,
                        [],
                        rest_emr,
                        "uv_os",
                        "uv_os",
                    )
                )
            if not self.disable_integrated_uv_cts:
                if (
                    is_selected
                    and self.top_self_energy_renormalisation == "projected-os"
                ):
                    integrated_uv_ct = self.construct_integrated_counter_term(
                        cycle[0],
                        cycle[1],
                        uv_routing,
                        top_self_energy_renormalisation="projected-os",
                    )
                else:
                    integrated_uv_ct = self.construct_integrated_counter_term(
                        cycle[0],
                        cycle[1],
                        uv_routing,
                        top_self_energy_os_subtraction=is_selected,
                    )
                if integrated_uv_ct is not None:
                    counterms.append(integrated_uv_ct)
                elif is_selected:
                    raise ValueError(
                        "The selected top self-energy cycle did not produce "
                        "the required integrated UV difference; "
                        + self._top_self_energy_context(cycle[0], selected[2])
                    )

        return counterms


class ThresholdSubtractor(object):
    def __init__(
        self,
        routed_cut_graph,
        params,
        name,
        L,
        theta_support=True,
        numerator_factorisation=None,
        threshold_collinear_momentum=None,
        emr_state_name=None,
        threshold_h_function=None,
    ):
        self.routed_cut_graph = routed_cut_graph
        self.emr_processor = EMRIntegrandConstructor(
            params,
            name,
            L,
            state_name=emr_state_name,
        )
        self.sp3D = S("sp3D", is_linear=True, is_symmetric=True)
        self.residues = self.emr_processor.get_integrand(
            routed_cut_graph,
            True,
            numerator_factorisation=numerator_factorisation,
        )
        self.L = L
        self.theta_support = theta_support
        self.threshold_collinear_momentum = threshold_collinear_momentum
        self.name = name
        self.threshold_h_function = resolve_threshold_h_function(
            name, threshold_h_function
        )

    # Here we can do something that is sort of process-specific...
    def filter_e_surfaces(self):
        filtered_e_surf = []

        for esurf, residue in self.residues:
            esurf = esurf.expand()
            if not esurf.is_type(AtomType.Add):
                raise ValueError(
                    "problem: single energy e-surface in threshold approximator"
                )
            energy_ids = list(esurf)

            energy_ids = [
                en.replace(E("-En(x___)"), E("x___")).replace(
                    E("En(x___)"), E("x___")
                )
                for en in energy_ids
            ]

            positive_ids = []
            negative_ids = []
            for e in self.routed_cut_graph.graph.get_edges():
                e_atts = e.get_attributes()
                edge_id = _strip_quotes(str(e_atts["id"]))
                if int(edge_id) not in energy_ids:
                    continue
                if _cut_sign_from_attributes(e_atts) != 0:
                    positive_ids.append(edge_id)
                else:
                    negative_ids.append(edge_id)

            external_momentum = [0, 0]
            outgoing_mass = 0
            incoming_mass = 0
            for e in self.routed_cut_graph.graph.get_edges():
                e_atts = e.get_attributes()
                edge_id = _strip_quotes(str(e_atts["id"]))
                particle = _strip_quotes(str(e_atts["particle"]))
                if edge_id in positive_ids:
                    cut_sign = _cut_sign_from_attributes(e_atts)
                    external_momentum[0] += (
                        int(e_atts["routing_p1"]) * cut_sign
                    )
                    external_momentum[1] += (
                        int(e_atts["routing_p2"]) * cut_sign
                    )
                    if particle in ["t", "t~"]:
                        outgoing_mass += 1
                if edge_id in positive_ids:
                    if particle in ["t", "t~"]:
                        incoming_mass += 1

            if (
                external_momentum[0] * external_momentum[1] == 1
                and external_momentum[0] == external_momentum[1]
                and len(negative_ids) > 1
            ):
                if outgoing_mass == 1 and incoming_mass == 1:
                    continue
                filtered_e_surf.append((negative_ids, esurf, residue))

        return filtered_e_surf

    def replace_energies(self, integrand, cut_graph):

        for e in cut_graph.get_edges():
            e_atts = e.get_attributes()
            eid_raw = e_atts["id"]
            eid = _strip_quotes(eid_raw) if isinstance(eid_raw, str) else eid_raw
            target = E(f"En({eid})")
            particle = _strip_quotes(str(e_atts["particle"]))
            if particle not in ["d", "d~", "g", "ghG", "ghG~"]:
                replacement = (
                    self.sp3D(E(f"q({eid})"), E(f"q({eid})")) + E(f"m({particle})") ** 2
                ) ** E("1/2")
            else:
                # new stuff
                k_keys = [f"routing_k{i}" for i in range(0, self.L)]
                zero_routing = all(e_atts[kk] == "0" for kk in k_keys)
                mass = E("0")
                if (
                    zero_routing
                    and e_atts["routing_p1"] == "0"
                    and e_atts["routing_p2"] != "0"
                ):
                    mass = E("0")
                if (
                    zero_routing
                    and e_atts["routing_p1"] != "0"
                    and e_atts["routing_p2"] == "0"
                ):
                    mass = E("0")

                replacement = (self.sp3D(E(f"q({eid})"), E(f"q({eid})")) + mass) ** E(
                    "1/2"
                )

            integrand = integrand.replace(target, replacement)

        return integrand

    # Substituted the emr momenta by their linear decomposition in terms of
    # loop variables and external momenta (p1 and p2)

    def route_integrand(self, integrand, cut_graph):

        for e in cut_graph.get_edges():
            e_atts = e.get_attributes()
            routing_items = E("0")
            for i in range(self.L):
                key = f"routing_k{i}"
                if key in e_atts:
                    routing_items += E(f"{e_atts[key]}*k[{i}]")
            for i in range(2):
                key = f"routing_p{i + 1}"
                if key in e_atts:
                    routing_items += E(f"{e_atts[key]}*p[{i + 1}]")
            integrand = integrand.replace(E(f"q({e_atts['id']})"), routing_items)

        integrand = integrand.replace(
            E("sp3(x___,y___)"), self.sp3D(S("x___"), S("y___"))
        )
        return integrand

    def construct_threshold_counter_term(
        self, emr_residue_integrand, threshold_graph, threshold_ids
    ):

        collinear_momentum = E("0")

        for e in threshold_graph.graph.get_edges():
            e_atts = e.get_attributes()
            if e_atts["id"] == threshold_ids[0]:
                collinear_momentum = (
                    collinear_momentum
                    + E(f"{e_atts['routing_p1']}*p(1)")
                    + E(f"{e_atts['routing_p2']}*p(2)")
                )
                k_sign = E(f"{e_atts['routing_k0']}*k(0)")

        if self.threshold_collinear_momentum is not None:
            collinear_momentum = self.threshold_collinear_momentum

        # for ttbar: careful of k0 sign

        lmb_ids = threshold_ids[:-1] + [
            e.get_attributes()["id"] for e in threshold_graph.final_cut[:-1]
        ]

        threshold_graph_routed = change_routing(
            deepcopy(threshold_graph.graph), lmb_ids
        )

        sqrts = S("s") ** E("1/2")

        threshold_integrand = self.replace_energies(
            emr_residue_integrand, threshold_graph_routed
        )
        threshold_integrand = self.route_integrand(
            threshold_integrand, threshold_graph_routed
        )  # .replace(r, r.exp())

        shifts = []
        masses = []

        for id in threshold_ids:
            shift = E("0")
            for e in threshold_graph_routed.get_edges():
                e_atts = e.get_attributes()
                if e_atts["id"] == id:
                    routing_items = E("0")
                    for i in range(1, self.L + 1):
                        key = f"routing_k{i}"
                        if key in e_atts:
                            routing_items += E(f"{e_atts[key]}*k[{i}]")
                    for i in range(2):
                        key = f"routing_p{i + 1}"
                        if key in e_atts:
                            routing_items += E(f"{e_atts[key]}*p[{i + 1}]")
                    shift = routing_items
                    mass = E("0")
                    e_particle = _strip_quotes(str(e_atts["particle"]))
                    if e_particle in ["d", "d~", "g", "ghG", "ghG~"]:
                        mass = E("0")
                    else:
                        mass = E(f"m({e_particle})")
            shifts.append(shift)
            masses.append(mass)

        # Use private construction symbols: Symbolica interns names globally,
        # so a generic ``r`` created elsewhere without scalar attributes can
        # otherwise make this constructor order-dependent.
        r = S("__dy_threshold_r", is_scalar=True)
        rexp = S("__dy_threshold_rexp", is_scalar=True)
        khat = S("__dy_threshold_khat")
        patt = E("k(0)")
        rep = rexp * khat

        # dot products
        # kp1 = self.sp3D(E("k(0)"), shifts[0])
        # kp2 = self.sp3D(E("k(0)"), shifts[1])
        # kk = self.sp3D(E("k(0)"), E("k(0)"))
        kp1 = self.sp3D(khat, shifts[0])
        kp2 = self.sp3D(khat, shifts[1])
        kk = self.sp3D(khat, khat)
        p1p1 = self.sp3D(shifts[0], shifts[0])
        p2p2 = self.sp3D(shifts[1], shifts[1])
        m1sq = masses[0] ** 2
        m2sq = masses[1] ** 2

        A = 4 * (sqrts**2) * kk - kp1**2 + 2 * kp1 * kp2 - kp2**2

        B = (
            -2 * m1sq * kp1
            + 2 * m2sq * kp1
            + 2 * (sqrts**2) * kp1
            + 2 * m1sq * kp2
            - 2 * m2sq * kp2
            + 2 * (sqrts**2) * kp2
            - 2 * kp1 * p1p1
            + 2 * kp2 * p1p1
            + 2 * kp1 * p2p2
            - 2 * kp2 * p2p2
        )

        C = (
            -(m1sq**2)
            + 2 * m1sq * m2sq
            - m2sq**2
            + 2 * m1sq * (sqrts**2)
            + 2 * m2sq * (sqrts**2)
            - (sqrts**4)
            - 2 * m1sq * p1p1
            + 2 * m2sq * p1p1
            + 2 * (sqrts**2) * p1p1
            - (p1p1**2)
            + 2 * m1sq * p2p2
            - 2 * m2sq * p2p2
            + 2 * (sqrts**2) * p2p2
            + 2 * p1p1 * p2p2
            - (p2p2**2)
        )

        rstar = ((-B + (B**2 - 4 * A * C) ** E("1/2")) / (2 * A)).log()

        derivative = (2 * (2 * rstar).exp() * kk + (rstar).exp() * kp1) / (
            2.0
            * ((2 * rstar).exp() * kk + (rstar).exp() * kp1 + m1sq + p1p1) ** E("1/2")
        ) + (2 * (2 * rstar).exp() * kk + (rstar).exp() * kp2) / (
            2.0
            * ((2 * rstar).exp() * kk + (rstar).exp() * kp2 + m2sq + p2p2) ** E("1/2")
        )

        threshold_integrand = threshold_integrand.replace(patt, rep)

        threshold_integrand = (
            (threshold_integrand.replace(rexp, r.exp()).replace(r, rstar))
            / (r - rstar)
            / derivative
        )

        inv_knorm = S("__dy_threshold_inv_knorm", is_scalar=True)
        threshold_integrand = (
            threshold_integrand
            .replace(rexp, r.exp())
            .replace(r, (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")).log())
            .replace(khat, E("k(0)") * inv_knorm)
            .replace(inv_knorm, 1 / (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")))
        )

        threshold_integrand = threshold_integrand.replace(
            E("s"), 4 * (E("p(1,1)") ** 2 + E("p(1,2)") ** 2 + E("p(1,3)") ** 2)
        )

        rep_r = rexp * khat

        if self.theta_support == True:
            repl_x = self.sp3D(rep_r, collinear_momentum) / self.sp3D(
                collinear_momentum, collinear_momentum
            )

            x = _DY_COLLINEAR_X

            repl_kperp = -x * collinear_momentum + rep_r

            # NEW: CUT THRESHOLD CUTTING REGION BY 4

            theta1 = (
                E(
                    f"Θ(({self.sp3D(repl_kperp, repl_kperp)})-"
                    f"({x}*(1-{x}))*Lambdasq/16)"
                )
                .replace(x, repl_x)
                .replace(rexp, r.exp())
                .replace(r, (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")).log())
                .replace(khat, E("k(0)") * inv_knorm)
                .replace(inv_knorm, 1 / (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")))
            )
            theta2 = (
                # E(f"Θ(({self.sp3D(repl_kperp, repl_kperp)})-(x*(1-x))*Lambdasq)")
                E(
                    f"Θ(({self.sp3D(repl_kperp, repl_kperp)})-"
                    f"({x}*(1-{x}))*Lambdasq/16)"
                )
                .replace(x, repl_x)
                .replace(rexp, r.exp())
                .replace(r, 2 * rstar - r)
                .replace(r, (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")).log())
                .replace(khat, E("k(0)") * inv_knorm)
                .replace(inv_knorm, 1 / (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")))
                .replace(
                    E("s"), 4 * (E("p(1,1)") ** 2 + E("p(1,2)") ** 2 + E("p(1,3)") ** 2)
                )
            )

            print(theta1)
            print(theta2)
        else:
            theta1 = E("1")
            theta2 = E("1")

        if self.threshold_h_function == "gaussian":
            hr = (
                (-((r - rstar) ** 2))
                .exp()
                .replace(r, (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")).log())
                .replace(khat, E("k(0)") * inv_knorm)
                .replace(inv_knorm, 1 / (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")))
                .replace(
                    E("s"), 4 * (E("p(1,1)") ** 2 + E("p(1,2)") ** 2 + E("p(1,3)") ** 2)
                )
            )
        else:
            hr = (
                (-((r - rstar) ** 2) - 1 / r**2 + 1 / rstar**2)
                .exp()
                .replace(r, (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")).log())
                .replace(khat, E("k(0)") * inv_knorm)
                .replace(inv_knorm, 1 / (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")))
                .replace(
                    E("s"), 4 * (E("p(1,1)") ** 2 + E("p(1,2)") ** 2 + E("p(1,3)") ** 2)
                )
            )

        jacobian_correction = (
            (-3 * (r - rstar))
            .exp()
            .replace(r, (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")).log())
            .replace(khat, E("k(0)") * inv_knorm)
            .replace(inv_knorm, 1 / (self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")))
            .replace(
                E("s"), 4 * (E("p(1,1)") ** 2 + E("p(1,2)") ** 2 + E("p(1,3)") ** 2)
            )
        )

        threshold_integrand = (
            jacobian_correction * threshold_integrand * theta1 * theta2 * hr
        )

        # Go back to previous basis
        edge_by_id = {
            _strip_quotes(str(edge.get_attributes()["id"])): edge
            for edge in threshold_graph.graph.get_edges()
        }
        basis_replacements = []
        for j, id in enumerate(lmb_ids):
            basis_edge = edge_by_id.get(_strip_quotes(str(id)))
            if basis_edge is None:
                raise ValueError(f"Threshold basis edge {id} not found in cut graph")
            e_atts = basis_edge.get_attributes()
            routing_items = E("0")
            for i in range(self.L + 1):
                key = f"routing_k{i}"
                if key in e_atts:
                    routing_items += E(f"{e_atts[key]}*k[{i}]")
            for i in range(2):
                key = f"routing_p{i + 1}"
                if key in e_atts:
                    routing_items += E(f"{e_atts[key]}*p[{i + 1}]")
            basis_replacements.append((E(f"k({j})"), routing_items))

        tmp_symbols = [E(f"__tmp_th_{j}") for j in range(len(basis_replacements))]
        for (pattern, _), tmp_symbol in zip(basis_replacements, tmp_symbols):
            threshold_integrand = threshold_integrand.replace(pattern, tmp_symbol)
        for tmp_symbol, (_, replacement) in zip(tmp_symbols, basis_replacements):
            threshold_integrand = threshold_integrand.replace(tmp_symbol, replacement)

        return RoutedIntegrand(
            -threshold_integrand,
            threshold_graph,
            [],
            E("0"),
            "threshold",
            "threshold",
        )

    def construct_threshold_counter_terms(self):

        filtered_e_surfs = self.filter_e_surfaces()
        threshold_cts = []

        print("filtered e surfs")
        print(filtered_e_surfs)

        for thresh_ids, e_surf, residue in filtered_e_surfs:
            if len(thresh_ids) > 2:
                raise ValueError("not ready for two-loop threshold subtraction yet...")

            threshold_cts.append(
                self.construct_threshold_counter_term(
                    residue, deepcopy(self.routed_cut_graph), thresh_ids
                )
            )

        return threshold_cts


class Approximator(object):
    def __init__(self):
        self.sp3D = S("sp3D", is_linear=True, is_symmetric=True)

    def collinear_approximation(self, integrand, momentum, k_id, direction, order=-2):

        # kperp_sq = E(f"sp3D(k_perp({k_id[0]}),k_perp({k_id[0]}))")
        # coll_en = x * E("sp3D(p(1),p(1))^(1/2)") + lam**2 * kperp_sq / (
        #    2 * x * E("sp3D(p(1),p(1))^(1/2)")
        # )
        # a_coll_en = (1 - x) * E("sp3D(p(1),p(1))^(1/2)") + lam**2 * kperp_sq / (
        #    2 * (1 - x) * E("sp3D(p(1),p(1))^(1/2)")
        # )

        x = _DY_COLLINEAR_X
        lam = S("λ", is_scalar=True)

        # Let s*q(i) be the vector that should become collinear to p(1). s encodes the cut orientation. We
        # write s*q(i)=s*(q(i)-a*k(j))+a*s*k(j)= x*p(1)+lam*k_perp(j) and solve in k(j), giving
        # k(j)=a*s*x*p(1)+a*s*lam*k_perp(j)-a*(q(i)-a*k(j)). Now s=momentum[1] and a=k_id[1] and j=k_id[0].

        repl = k_id[1] * (
            momentum[1] * x * direction
            - (momentum[0] - k_id[1] * E(f"k({k_id[0]})"))
            + lam * E(f"k_perp({k_id[0]})")
        )

        integrand = integrand.replace(E(f"k({k_id[0]})"), repl)
        integrand = integrand.replace(
            self.sp3D(E(f"k_perp({k_id[0]})"), E("p(x_)")), E("0")
        )

        # Only consider the leading-virtuality contribution.  Keep the
        # factor/chunk path unconditional: unsupported Laurent structure must
        # fail explicitly instead of silently returning to whole-expression
        # expansion.
        integrand, _path_info = _factor_chunk_laurent_series(
            integrand,
            lam,
            order,
            chunk_size=_LAURENT_CHUNK_SIZE,
            min_terms=0,
        )
        integrand = integrand.replace(lam, 1)

        # Invert back the collinear parametrisation. Since s*q(i)= x*p(1)+lam*k_perp(j), we have
        # x=s*q(i).p(1)/p(1).p(1)

        repl_x = (
            momentum[1]
            * self.sp3D(momentum[0], direction)
            / self.sp3D(direction, direction)
        )

        repl_kperp = -momentum[1] * x * direction + momentum[0]

        # integrand = integrand.replace(
        #    self.sp3D(E(f"k_perp({k_id[0]})"), E(f"k_perp({k_id[0]})")),
        #    self.sp3D(repl_kperp, repl_kperp),
        # ).replace(x, repl_x)

        # new: normalisation for general ecm: THIS IS FOR RAISED PROPAGATORS ONLY
        # integrand = integrand * (self.sp3D(E("p(1)"), E("p(1)"))) ** E("1/2")

        integrand = integrand.replace(
            E(f"k_perp({k_id[0]})"),
            repl_kperp,
        ).replace(x, repl_x)

        return integrand, repl, repl_x, repl_kperp

    def soft_approximation(self, integrand, momentum, k_id):

        lam = S("λ", is_scalar=True)

        repl = k_id[1] * (-(momentum[0] - k_id[1] * E(f"k({k_id[0]})"))) + k_id[
            1
        ] * lam * E("qsoft")

        integrand = integrand.replace(E(f"k({k_id[0]})"), repl)

        integrand = integrand.series(lam, 0, -3).to_expression().replace(
            lam, 1
        )

        integrand = integrand.replace(E("qsoft"), momentum[0])

        return integrand, repl


class LoopIntegrandConstructor(object):
    def __init__(
        self,
        params,
        name,
        L,
        channel=None,
        disable_integrated_uv_cts=True,
        external_gluon_polarisation=False,
        emr_state_name=None,
        symmetrise_p1_p2=False,
        top_self_energy_os_subtraction=None,
        top_self_energy_renormalisation=None,
        threshold_h_function=None,
    ):
        self.L = L
        self.params = params
        self.name = name
        self.emr_processor = EMRIntegrandConstructor(
            params,
            name,
            L,
            state_name=emr_state_name,
        )
        self.sp3D = S("sp3D", is_linear=True, is_symmetric=True)
        self.approximator = Approximator()
        self.channel = channel
        self.external_gluon_polarisation = bool(external_gluon_polarisation)
        self.symmetrise_p1_p2 = bool(symmetrise_p1_p2)
        self.disable_integrated_uv_cts = bool(disable_integrated_uv_cts)
        self.top_self_energy_renormalisation = (
            resolve_top_self_energy_renormalisation(
                top_self_energy_renormalisation,
                top_self_energy_os_subtraction,
            )
        )
        self.top_self_energy_os_subtraction = (
            self.top_self_energy_renormalisation == "os"
        )
        self.threshold_h_function = resolve_threshold_h_function(
            name, threshold_h_function
        )
        self.emr_state_name = emr_state_name

    def _external_beam_edges(self, graph):
        candidates = {"p1": [], "p2": []}
        for edge in graph.get_edges():
            attributes = edge.get_attributes()
            loop_coefficients = [
                Fraction(
                    _strip_quotes(
                        str(attributes.get(f"routing_k{index}", "0"))
                    )
                )
                for index in range(self.L)
            ]
            if any(coefficient != 0 for coefficient in loop_coefficients):
                continue
            p1_coefficient = Fraction(
                _strip_quotes(str(attributes.get("routing_p1", "0")))
            )
            p2_coefficient = Fraction(
                _strip_quotes(str(attributes.get("routing_p2", "0")))
            )
            if p1_coefficient in {-1, 1} and p2_coefficient == 0:
                candidates["p1"].append((edge, int(p1_coefficient)))
            elif p2_coefficient in {-1, 1} and p1_coefficient == 0:
                candidates["p2"].append((edge, int(p2_coefficient)))

        if any(len(candidates[beam]) != 1 for beam in ("p1", "p2")):
            graph_name = _strip_quotes(
                str(graph.get("base_graph_name") or graph.get_name())
            )
            candidate_ids = {
                beam: [
                    _strip_quotes(str(edge.get_attributes().get("id", "")))
                    for edge, _sign in candidates[beam]
                ]
                for beam in ("p1", "p2")
            }
            raise ValueError(
                "An external-gluon projector requires exactly one pure ±p1 "
                "edge and one pure ±p2 "
                f"edge for {graph_name}; found {candidate_ids}."
            )

        return candidates["p1"][0], candidates["p2"][0]

    def _external_gluon_polarisation_edges(self, graph):
        if getattr(self, "symmetrise_p1_p2", False) and getattr(
            self, "external_gluon_polarisation", False
        ):
            edges_by_id = {"0": [], "1": []}
            for edge in graph.get_edges():
                edge_id = _strip_quotes(
                    str(edge.get_attributes().get("id", ""))
                )
                if edge_id in edges_by_id:
                    edges_by_id[edge_id].append(edge)

            if any(len(edges_by_id[edge_id]) != 1 for edge_id in ("0", "1")):
                graph_name = _strip_quotes(
                    str(graph.get("base_graph_name") or graph.get_name())
                )
                counts = {
                    edge_id: len(edges_by_id[edge_id]) for edge_id in ("0", "1")
                }
                raise ValueError(
                    "A p1/p2-symmetrised external-gluon projector requires "
                    f"exactly one edge 0 and one edge 1 for {graph_name}; "
                    f"found counts {counts}."
                )

            beam_edges = (edges_by_id["0"][0], edges_by_id["1"][0])
        else:
            (p1_edge, _p1_sign), (p2_edge, _p2_sign) = (
                self._external_beam_edges(graph)
            )
            beam_edges = (p1_edge, p2_edge)

        return [
            edge
            for edge in beam_edges
            if _strip_quotes(str(edge.get_attributes().get("particle", "")))
            == "g"
        ]

    @staticmethod
    def _store_external_gluon_beam_metadata(graph):
        graph.set(
            _EXTERNAL_GLUON_POLARISATION_PROTECTED_IDS_ATTR,
            ",".join(
                (
                    _EXTERNAL_GLUON_POLARISATION_P1_ID,
                    _EXTERNAL_GLUON_POLARISATION_P2_ID,
                )
            ),
        )

    def substitute_external_gluon_polarisation_sum(self, cut_graph):
        def _mink(port):
            return f"spenso::mink(4,hedge({_parse_port(port)}))"

        def _projector(mu, nu):
            p1_id = _EXTERNAL_GLUON_POLARISATION_P1_ID
            p2_id = _EXTERNAL_GLUON_POLARISATION_P2_ID
            denominator = f"sp({p1_id},{p2_id})"
            return (
                f"spenso::g({mu},{nu})"
                f"-Q({p1_id},{mu})*Q({p2_id},{nu})/{denominator}"
                f"-Q({p1_id},{nu})*Q({p2_id},{mu})/{denominator}"
            )

        target_edges = self._external_gluon_polarisation_edges(cut_graph.graph)
        if not target_edges:
            return cut_graph
        self._store_external_gluon_beam_metadata(cut_graph.graph)

        for edge in target_edges:
            edge_atts = edge.get_attributes()
            num = _strip_quotes(str(edge_atts.get("num", "")))
            if not num:
                raise ValueError(
                    f"Cannot replace missing numerator for cut gluon edge {edge_atts.get('id')}."
                )

            source_mu = _mink(edge.get_source())
            destination_mu = _mink(edge.get_destination())
            metric_candidates = [
                (
                    f"spenso::g({destination_mu},{source_mu})",
                    f"({_projector(destination_mu, source_mu)})",
                ),
                (
                    f"spenso::g({source_mu},{destination_mu})",
                    f"({_projector(source_mu, destination_mu)})",
                ),
            ]

            new_num = num
            for metric, projector in metric_candidates:
                if metric in new_num:
                    new_num = new_num.replace(metric, projector)
                    break

            if new_num == num:
                edge_id = _strip_quotes(str(edge_atts.get("id", "")))
                raise ValueError(
                    f"Could not find Lorentz metric in cut gluon edge {edge_id} numerator."
                )

            edge_atts["num"] = new_num

        return cut_graph

    def external_gluon_polarisation_numerator_factorisation(self, graph):
        numerator_graph = deepcopy(graph)

        def _mink(port):
            return f"spenso::mink(4,hedge({_parse_port(port)}))"

        def _projector(mu, nu):
            p1_id = _EXTERNAL_GLUON_POLARISATION_P1_ID
            p2_id = _EXTERNAL_GLUON_POLARISATION_P2_ID
            denominator = f"sp({p1_id},{p2_id})"
            return (
                f"spenso::g({mu},{nu})"
                f"-Q({p1_id},{mu})*Q({p2_id},{nu})/{denominator}"
                f"-Q({p1_id},{nu})*Q({p2_id},{mu})/{denominator}"
            )

        target_edges = self._external_gluon_polarisation_edges(numerator_graph)
        if not target_edges:
            return numerator_graph, E("1")
        self._store_external_gluon_beam_metadata(numerator_graph)

        post_momentum_rewrite_factor = E("1")
        for edge in target_edges:
            edge_atts = edge.get_attributes()
            num = _strip_quotes(str(edge_atts.get("num", "")))
            if not num:
                raise ValueError(
                    f"Cannot factor missing numerator for cut gluon edge {edge_atts.get('id')}."
                )

            source_mu = _mink(edge.get_source())
            destination_mu = _mink(edge.get_destination())
            metric_candidates = [
                (
                    f"spenso::g({destination_mu},{source_mu})",
                    f"({_projector(destination_mu, source_mu)})",
                ),
                (
                    f"spenso::g({source_mu},{destination_mu})",
                    f"({_projector(source_mu, destination_mu)})",
                ),
            ]

            for metric, projector in metric_candidates:
                if metric in num:
                    edge_atts["num"] = num.replace(metric, "1", 1)
                    post_momentum_rewrite_factor *= Es(projector)
                    break
            else:
                edge_id = _strip_quotes(str(edge_atts.get("id", "")))
                raise ValueError(
                    f"Could not find Lorentz metric in cut gluon edge {edge_id} numerator."
                )

        return numerator_graph, post_momentum_rewrite_factor

    # Replaces energies by their expression in terms of the emr momenta and particle masses

    def replace_energies(self, integrand, cut_graph):

        for e in cut_graph.graph.get_edges():
            e_atts = e.get_attributes()
            eid_raw = e_atts["id"]
            eid = _strip_quotes(eid_raw) if isinstance(eid_raw, str) else eid_raw
            target = E(f"En({eid})")
            particle = _strip_quotes(str(e_atts["particle"]))
            if particle not in ["d", "d~", "g", "ghG", "ghG~"]:
                replacement = (
                    self.sp3D(E(f"q({eid})"), E(f"q({eid})")) + E(f"m({particle})") ** 2
                ) ** E("1/2")
            else:
                replacement = self.sp3D(
                    E(f"q({eid})"), E(f"q({eid})")
                ) ** E(
                    "1/2"
                )

            integrand = integrand.replace(target, replacement)

        return integrand

    # Substituted the emr momenta by their linear decomposition in terms of
    # loop variables and external momenta (p1 and p2)

    def route_integrand(self, integrand, cut_graph):

        for e in cut_graph.graph.get_edges():
            e_atts = e.get_attributes()
            routing_items = E("0")
            for i in range(self.L):
                key = f"routing_k{i}"
                if key in e_atts:
                    routing_items += E(f"{e_atts[key]}*k({i})")
            for i in range(2):
                key = f"routing_p{i + 1}"
                if key in e_atts:
                    routing_items += E(f"{e_atts[key]}*p({i + 1})")
            integrand = integrand.replace(E(f"q({e_atts['id']})"), routing_items)

        integrand = integrand.replace(
            E("sp3(x___,y___)"), self.sp3D(S("x___"), S("y___"))
        )
        return integrand

    # Checks if two edges have the same routing, implying they form a raised propagator.

    def _routing_sign_match(self, e: pydot.Edge, ep: pydot.Edge):
        return _routing_sign_match(e, ep)

    def concretise_scalar_products(self, integrand):

        return integrand.replace(
            E("sp3D(w_(x_),z_(y_))"),
            E("w_(x_,1)*z_(y_,1)+w_(x_,2)*z_(y_,2)+w_(x_,3)*z_(y_,3)"),
        )

    # Approximates the integrand at leading virtuality. Raised t-channel
    # propagators have already been squared explicitly during CFF construction.

    def leading_virtuality_expansion(self, integrand, cut_graph, raised_cut):
        emr_integrand = deepcopy(integrand)
        partition = cut_graph.partition

        routed_integrands = []

        if len(partition[0]) == 1 and len(partition[1]) == 1:
            integrand = self.replace_energies(integrand, cut_graph)
            integrand = self.route_integrand(integrand, cut_graph)
            routed_integrand = RoutedIntegrand(
                integrand, cut_graph, [], emr_integrand, "PM", []
            )
            routed_integrands.append(routed_integrand)
            return routed_integrands

        elif len(partition[0]) > 1 and len(partition[1]) == 1:
            x = _DY_COLLINEAR_X
            lam = S("λ", is_scalar=True)
            momentum = E("0")

            coll_moms = []

            # Find collinear momenta and particles.
            seed_edge = _collinear_momentum_seed_edge(
                partition[0],
                tuple(getattr(cut_graph, "raised_cut_pairs", ())),
            )
            seed_id = _strip_quotes(str(seed_edge.get_attributes()["id"]))
            for ep in partition[0]:
                ep_atts = ep.get_attributes()
                id = ep_atts["id"]
                coll_moms.append(id)
                for e in cut_graph.graph.get_edges():
                    e_atts = e.get_attributes()
                    if e_atts["id"] == id:
                        k_keys = ["routing_k" + str(i) for i in range(0, self.L)]
                        loop_coeff = [E(e_atts[rout]) for rout in k_keys]
                        if _strip_quotes(str(id)) == seed_id:
                            k_id = next(
                                (i, c)
                                for i, c in enumerate(loop_coeff)
                                if str(c) != "0"
                            )
                            momentum = [
                                (
                                    sum(
                                        loop_coeff[i] * E(f"k({i})")
                                        for i in range(self.L)
                                    )
                                    + E(e_atts["routing_p1"]) * E("p(1)")
                                    + E(e_atts["routing_p2"]) * E("p(2)")
                                ),
                                e_atts["is_cut_DY"],
                            ]

            integrand = self.replace_energies(integrand, cut_graph)

            integrand = self.route_integrand(integrand, cut_graph)

            integrand, repl, repl_x, repl_kperp = (
                self.approximator.collinear_approximation(
                    integrand, momentum, k_id, E("p(1)")
                )
            )

            print("expanded expression is available")

            thetaLambdasq = E(
                f"Θ(Lambdasq-({self.sp3D(repl_kperp, repl_kperp)})/"
                f"({x}*(1-{x})))"
            ).replace(x, repl_x)

            integrand = (
                integrand * E(f"Θ({repl_x})") * E(f"Θ(1-{repl_x})") * thetaLambdasq
            )

            routed_integrand = RoutedIntegrand(
                integrand,
                cut_graph,
                [
                    E(f"k({k_id[0]})"),
                    repl.replace(x, repl_x).series(lam, 0, 0).to_expression(),
                ],
                emr_integrand,
                "collinear",
                [
                    E(f"k({k_id[0]})"),
                    repl.replace(x, repl_x).series(lam, 0, 0).to_expression(),
                ],
            )
            routed_integrands.append(routed_integrand)

            print("routed integrand constructed")

            return routed_integrands

        elif len(partition[0]) == 1 and len(partition[1]) > 1:
            x = _DY_COLLINEAR_X
            lam = S("λ", is_scalar=True)
            momentum = E("0")

            coll_moms = []

            # Find collinear momenta and particles.

            seed_edge = _collinear_momentum_seed_edge(
                partition[1],
                tuple(getattr(cut_graph, "raised_cut_pairs", ())),
            )
            seed_id = _strip_quotes(str(seed_edge.get_attributes()["id"]))
            for ep in partition[1]:
                ep_atts = ep.get_attributes()
                id = ep_atts["id"]
                coll_moms.append(id)
                for e in cut_graph.graph.get_edges():
                    e_atts = e.get_attributes()
                    if e_atts["id"] == id:
                        k_keys = ["routing_k" + str(i) for i in range(self.L)]
                        loop_coeff = [E(e_atts[rout]) for rout in k_keys]
                        if _strip_quotes(str(id)) == seed_id:
                            k_id = next(
                                (i, c)
                                for i, c in enumerate(loop_coeff)
                                if str(c) != "0"
                            )
                            momentum = [
                                (
                                    sum(
                                        loop_coeff[i] * E(f"k({i})")
                                        for i in range(self.L)
                                    )
                                    + E(e_atts["routing_p1"]) * E("p(1)")
                                    + E(e_atts["routing_p2"]) * E("p(2)")
                                ),
                                e_atts["is_cut_DY"],
                            ]

            integrand = self.replace_energies(integrand, cut_graph)

            integrand = self.route_integrand(integrand, cut_graph)

            integrand_old = deepcopy(integrand)

            integrand, repl, repl_x, repl_kperp = (
                self.approximator.collinear_approximation(
                    integrand, momentum, k_id, E("p(2)")
                )
            )

            thetaLambdasq = E(
                f"Θ(Lambdasq-({self.sp3D(repl_kperp, repl_kperp)})/"
                f"({x}*(1-{x})))"
            ).replace(x, repl_x)
            integrand = (
                integrand * E(f"Θ({repl_x})") * E(f"Θ(1-{repl_x})") * thetaLambdasq
            )

            # HACKED OLD
            routed_integrand = RoutedIntegrand(
                integrand,
                cut_graph,
                [
                    E(f"k({k_id[0]})"),
                    repl.replace(x, repl_x).series(lam, 0, 0).to_expression(),
                ],
                emr_integrand,
                "anti-collinear",
                [
                    E(f"k({k_id[0]})"),
                    repl.replace(x, repl_x).series(lam, 0, 0).to_expression(),
                ],
            )
            routed_integrands.append(routed_integrand)

            return routed_integrands

        elif len(partition[0]) > 1 and len(partition[1]) > 1:
            lam = S("λ", is_scalar=True)
            x = _DY_COLLINEAR_X

            soft_edge = list(set(partition[0]).intersection(partition[1]))
            hard_edge1 = [e for e in partition[0] if e not in soft_edge]
            hard_edge2 = [e for e in partition[1] if e not in soft_edge]

            if len(soft_edge) != 1 or len(hard_edge1) != 1 or len(hard_edge2) != 1:
                raise ValueError(
                    "Big problem with soft approximation!!! (or your desires are too demanding for this poor, little code)"
                )

            id = soft_edge[0].get_attributes()["id"]
            id_hard1 = hard_edge1[0].get_attributes()["id"]
            id_hard2 = hard_edge2[0].get_attributes()["id"]
            momentum = []
            hard_momentum1 = []
            hard_momentum2 = []
            for e in cut_graph.graph.get_edges():
                e_atts = e.get_attributes()
                if e_atts["id"] == id:
                    k_keys = ["routing_k" + str(i) for i in range(0, self.L)]
                    loop_coeff = [E(e_atts[rout]) for rout in k_keys]
                    k_id = next(
                        (i, c) for i, c in enumerate(loop_coeff) if str(c) != "0"
                    )
                    momentum = [
                        (
                            sum(loop_coeff[i] * E(f"k({i})") for i in range(0, self.L))
                            + E(e_atts["routing_p1"]) * E("p(1)")
                            + E(e_atts["routing_p2"]) * E("p(2)")
                        ),
                        e_atts["is_cut_DY"],
                    ]
                if e_atts["id"] == id_hard1:
                    k_keys = ["routing_k" + str(i) for i in range(0, self.L)]
                    loop_coeff = [E(e_atts[rout]) for rout in k_keys]
                    hard_momentum1 = [
                        (
                            sum(loop_coeff[i] * E(f"k({i})") for i in range(0, self.L))
                            + E(e_atts["routing_p1"]) * E("p(1)")
                            + E(e_atts["routing_p2"]) * E("p(2)")
                        ),
                        e_atts["is_cut_DY"],
                    ]
                if e_atts["id"] == id_hard2:
                    k_keys = ["routing_k" + str(i) for i in range(0, self.L)]
                    loop_coeff = [E(e_atts[rout]) for rout in k_keys]
                    hard_momentum2 = [
                        (
                            sum(loop_coeff[i] * E(f"k({i})") for i in range(0, self.L))
                            + E(e_atts["routing_p1"]) * E("p(1)")
                            + E(e_atts["routing_p2"]) * E("p(2)")
                        ),
                        e_atts["is_cut_DY"],
                    ]

            # integrand = E("(-(E(0)+E(2)+E(3))+E(0)+E(5))^-1*E(2)^(-2)")
            base_graph_name = _strip_quotes(str(cut_graph.graph.get("base_graph_name")))

            if base_graph_name in {"GL07", "GL09"} and self.channel in {
                (1, -1),
                (-1, 1),
            }:
                integrand = integrand.replace(
                    E("En(5)"), self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")
                )
                integrand = integrand.replace(
                    E("En(4)"), self.sp3D(E("k(0)"), E("k(0)")) ** E("1/2")
                )

            # print("hacked integrand")
            # print(integrand)

            # print("soft emr integrand")
            # print(integrand)

            integrand = self.replace_energies(integrand, cut_graph)
            integrand = self.route_integrand(integrand, cut_graph)

            # print("routed emr integrand")
            # print(integrand)

            # Factor of 1/s for soft and soft-collinear for virtual DY diagram

            factor = 1

            # if self.name=="tt~":
            # integrand = integrand.replace(E("p(2)"), -E("p(1)"))
            # momentum[0] = momentum[0].replace(E("p(2)"), -E("p(1)"))

            print("input-" * 10)
            print(f"hard particles are {id_hard1} and {id_hard2}")
            print(deepcopy(momentum))
            print(deepcopy(k_id))

            soft_integrand, repl_s = self.approximator.soft_approximation(
                deepcopy(integrand), deepcopy(momentum), deepcopy(k_id)
            )

            # print("soft approximation")
            # print(soft_integrand)

            soft_collinear_integrand1, repl1, repl1_x, repl_kperp1 = (
                self.approximator.collinear_approximation(
                    deepcopy(soft_integrand),
                    momentum,
                    k_id,
                    E("p(1)"),
                )
            )

            soft_collinear_integrand2, repl2, repl2_x, repl_kperp2 = (
                self.approximator.collinear_approximation(
                    deepcopy(soft_integrand),
                    momentum,
                    k_id,
                    E("p(2)"),
                )
            )

            if len(cut_graph.final_cut) <= 2:
                soft_collinear_integrand1 = -soft_collinear_integrand1
                soft_collinear_integrand2 = -soft_collinear_integrand2
                soft_integrand = -soft_integrand

            # integrand = (
            #    soft_integrand + soft_collinear_integrand1 + soft_collinear_integrand2
            # )  # * E(f"Θ({repl_x})") * E(f"Θ(1-{repl_x})")

            # need to construct propagators
            propsoft1 = 2 * (
                (self.sp3D(momentum[0], momentum[0])) ** E("1/2")
                * (self.sp3D(E("p(1)"), E("p(1)"))) ** E("1/2")
                - momentum[1] * self.sp3D(momentum[0], E("p(1)"))
            )
            propsoft2 = 2 * (
                (self.sp3D(momentum[0], momentum[0])) ** E("1/2")
                * (self.sp3D(E("p(2)"), E("p(2)"))) ** E("1/2")
                - momentum[1] * self.sp3D(momentum[0], E("p(2)"))
            )

            thetaSoft = E(f"Θ(Lambdasq-{propsoft1})") * E(f"Θ(Lambdasq-{propsoft2})")  #

            routed_integrand_soft = RoutedIntegrand(
                -factor * soft_integrand * thetaSoft,
                cut_graph,
                [
                    E(f"k({k_id[0]})"),
                    repl_s.series(lam, 0, 0).to_expression(),
                ],
                emr_integrand,
                "soft",
                [
                    E(f"k({k_id[0]})"),
                    repl_s.series(lam, 0, 0).to_expression(),
                ],
            )
            routed_integrands.append(routed_integrand_soft)

            thetacollinear1 = (
                E(
                    f"Θ(Lambdasq-({self.sp3D(repl_kperp1, repl_kperp1)})/({x}))"
                ).replace(
                    x, repl1_x
                )
                * E(
                    f"Θ(Lambdasq-4*{self.sp3D(E('p(1)'), E('p(1)'))}*{x})"
                ).replace(
                    x, repl1_x
                )
                * E(f"Θ({repl1_x})")
                # * E(f"Θ(1-{repl1_x})")
            )

            routed_integrand_collinear1 = RoutedIntegrand(
                factor * soft_collinear_integrand1 * thetacollinear1,
                cut_graph,
                [
                    E(f"k({k_id[0]})"),
                    repl_s.series(lam, 0, 0).to_expression(),
                    # repl1.replace(x, repl1_x).series(lam, 0, 0).to_expression(),
                ],
                emr_integrand,
                "soft-collinear",
                [
                    E(f"k({k_id[0]})"),
                    repl1.replace(x, repl1_x).series(lam, 0, 0).to_expression(),
                ],
            )
            routed_integrands.append(routed_integrand_collinear1)

            thetacollinear2 = (
                E(
                    f"Θ(Lambdasq-({self.sp3D(repl_kperp2, repl_kperp2)})/({x}))"
                ).replace(
                    x, repl2_x
                )
                * E(
                    f"Θ(Lambdasq-4*{self.sp3D(E('p(1)'), E('p(1)'))}*({x}))"
                ).replace(
                    x, repl2_x
                )
                * E(f"Θ({repl2_x})")
                # * E(f"Θ(1-{repl2_x})")
            )

            routed_integrand_collinear2 = RoutedIntegrand(
                factor * soft_collinear_integrand2 * thetacollinear2,
                cut_graph,
                [
                    E(f"k({k_id[0]})"),
                    repl_s.series(lam, 0, 0).to_expression(),
                    # repl2.replace(x, repl2_x).series(lam, 0, 0).to_expression(),
                ],
                emr_integrand,
                "soft-anti-collinear",
                [
                    E(f"k({k_id[0]})"),
                    repl2.replace(x, repl2_x).series(lam, 0, 0).to_expression(),
                ],
            )
            routed_integrands.append(routed_integrand_collinear2)

            return routed_integrands

        else:
            raise ValueError("Big problem if you get here :( ")

    def eliminate_raised_cuts(self, emr_representation, cut_graph):
        raised_pairs = tuple(getattr(cut_graph, "raised_cut_pairs", ()))
        emr_representation, is_final_raised = (
            _cleanup_final_state_raised_energies(
                emr_representation,
                cut_graph,
            )
        )
        return emr_representation, raised_pairs, is_final_raised

    def modify_t_channel_gluon_numerator(self, cut_graph):

        raised_t_channel_gluon_candidates = []
        g_edges = cut_graph.graph.get_edges()
        for e, i in zip(g_edges, range(len(g_edges))):
            e_atts = e.get_attributes()
            k_keys = ["routing_k" + str(i) for i in range(0, self.L)]
            p_keys = ["routing_p1", "routing_p2"]
            for ep, j in zip(g_edges, range(len(g_edges))):
                ep_atts = ep.get_attributes()
                relation = self._routing_sign_match(e, ep)
                if (
                    j > i
                    and relation is not None
                    and _strip_quotes(str(e_atts.get("particle"))) == "g"
                    and _strip_quotes(str(ep_atts.get("particle"))) == "g"
                    and not all(e_atts.get(key) == "0" for key in k_keys)
                    and not all(e_atts.get(key) != "0" for key in p_keys)
                ):
                    raised_t_channel_gluon_candidates.append((e, ep, relation))

        if len(raised_t_channel_gluon_candidates) > 1:
            raise ValueError(
                "Multiple repeated t-channel gluon candidates were found."
            )

        if raised_t_channel_gluon_candidates:
            e1, e2, _repeated_relation = raised_t_channel_gluon_candidates[0]
            k_keys = [f"routing_k{i}" for i in range(0, self.L)]

            def _is_pure_external_edge(edge):
                edge_atts = edge.get_attributes()
                has_p1 = _strip_quotes(str(edge_atts.get("routing_p1", "0"))) != "0"
                has_p2 = _strip_quotes(str(edge_atts.get("routing_p2", "0"))) != "0"
                has_single_external_momentum = has_p1 != has_p2
                has_no_loop_momentum = all(
                    _strip_quotes(str(edge_atts.get(key, "0"))) == "0" for key in k_keys
                )
                return has_single_external_momentum and has_no_loop_momentum

            def _pure_external_beam(edge):
                if not _is_pure_external_edge(edge):
                    return None
                attributes = edge.get_attributes()
                has_p1 = _strip_quotes(
                    str(attributes.get("routing_p1", "0"))
                ) != "0"
                return "p1" if has_p1 else "p2"

            def _incident_external_beams(candidate_edge):
                candidate_id = _strip_quotes(
                    str(candidate_edge.get_attributes()["id"])
                )
                beams = set()
                for node_name in (
                    _base_node(candidate_edge.get_source()),
                    _base_node(candidate_edge.get_destination()),
                ):
                    for incident_edge in boundary_edges(
                        cut_graph.graph, {node_name}
                    ):
                        incident_id = _strip_quotes(
                            str(incident_edge.get_attributes()["id"])
                        )
                        if incident_id == candidate_id:
                            continue
                        beam = _pure_external_beam(incident_edge)
                        if beam is not None:
                            beams.add(beam)
                return beams

            common_external_beams = _incident_external_beams(e1).intersection(
                _incident_external_beams(e2)
            )
            partition = getattr(cut_graph, "partition", ())
            if len(partition) == 2 and len(partition[0]) != len(partition[1]):
                projector_beam = (
                    "p1" if len(partition[0]) > len(partition[1]) else "p2"
                )
                if projector_beam not in common_external_beams:
                    raise ValueError(
                        f"partition selects {projector_beam}, but repeated gluons "
                        f"share external beams {sorted(common_external_beams)}"
                    )
            elif len(common_external_beams) == 1:
                projector_beam = next(iter(common_external_beams))
            elif common_external_beams == {"p1", "p2"}:
                (p1_edge, _p1_sign), (p2_edge, _p2_sign) = (
                    self._external_beam_edges(cut_graph.graph)
                )
                p1_is_gluon = _strip_quotes(
                    str(p1_edge.get_attributes().get("particle", ""))
                ) == "g"
                p2_is_gluon = _strip_quotes(
                    str(p2_edge.get_attributes().get("particle", ""))
                ) == "g"
                if p1_is_gluon != p2_is_gluon:
                    projector_beam = "p2" if p1_is_gluon else "p1"
                elif p1_is_gluon and self.symmetrise_p1_p2:
                    projector_beam = "p2"
                else:
                    raise ValueError(
                        "could not select a unique t-channel projector beam "
                        "for an equal partition"
                    )
            else:
                raise ValueError(
                    "repeated gluons do not share a unique external beam: "
                    f"{sorted(common_external_beams)}"
                )

            def _external_vertex_with_orientation(candidate_edge, orientation):
                candidate_nodes = [
                    _base_node(candidate_edge.get_source()),
                    _base_node(candidate_edge.get_destination()),
                ]
                matching_nodes = set()

                for node_name in candidate_nodes:
                    for incident_edge in boundary_edges(cut_graph.graph, {node_name}):
                        incident_atts = incident_edge.get_attributes()
                        if _strip_quotes(str(incident_atts["id"])) == _strip_quotes(
                            str(candidate_edge.get_attributes()["id"])
                        ):
                            continue
                        if not _is_pure_external_edge(incident_edge):
                            continue
                        if _pure_external_beam(incident_edge) != projector_beam:
                            continue

                        incident_node = (
                            _base_node(incident_edge.get_destination())
                            if orientation == "injecting"
                            else _base_node(incident_edge.get_source())
                        )
                        if incident_node == node_name:
                            matching_nodes.add(node_name)

                if len(matching_nodes) > 1:
                    edge_id = _strip_quotes(
                        str(candidate_edge.get_attributes()["id"])
                    )
                    raise ValueError(
                        f"raised gluon edge {edge_id} has multiple {orientation} "
                        f"external endpoints: {sorted(matching_nodes)}"
                    )
                return next(iter(matching_nodes), None)

            e1_injecting_vertex = _external_vertex_with_orientation(e1, "injecting")
            e2_injecting_vertex = _external_vertex_with_orientation(e2, "injecting")
            e1_has_injecting_external = e1_injecting_vertex is not None
            e2_has_injecting_external = e2_injecting_vertex is not None
            if e1_has_injecting_external == e2_has_injecting_external:
                raise ValueError(
                    "could not uniquely order raised gluon edges by external injection"
                )
            if not e1_has_injecting_external:
                e1, e2 = e2, e1
                e1_injecting_vertex = e2_injecting_vertex

            e1_atts = e1.get_attributes()
            e2_atts = e2.get_attributes()

            e2_departing_vertex = _external_vertex_with_orientation(e2, "departing")
            if e2_departing_vertex is None:
                raise ValueError(
                    "the second raised gluon edge has no departing pure external edge"
                )

            e1_s = e1.get_source()
            e1_d = e1.get_destination()
            e2_s = e2.get_source()
            e2_d = e2.get_destination()

            # The temporal vector Q(1000) must sit on the beam-adjacent side
            # of each projector.  Routing coefficients are not a reliable
            # proxy for that topology after p1/p2 symmetrisation (notably for
            # GL047 and GL073), so orient the projector directly from the
            # injecting/departing pure-external edge.
            e1_external_is_source = (
                e1_injecting_vertex == _base_node(e1_s)
            )
            e2_external_is_source = (
                e2_departing_vertex == _base_node(e2_s)
            )
            vertices1 = [e1_s, e1_d] if e1_external_is_source else [e1_d, e1_s]
            vertices2 = [e2_s, e2_d] if e2_external_is_source else [e2_d, e2_s]
            overall_sign1 = 1 if e1_external_is_source else -1
            overall_sign2 = -1 if e2_external_is_source else 1

            def _incident_energy_denominator(edge, vertex, overall_sign):
                edge_id = _strip_quotes(
                    str(edge.get_attributes()["id"])
                )
                terms = []
                for incident_edge in boundary_edges(
                    cut_graph.graph, {_base_node(vertex)}
                ):
                    incident_attributes = incident_edge.get_attributes()
                    incident_id = _strip_quotes(
                        str(incident_attributes["id"])
                    )
                    if incident_id == edge_id:
                        continue
                    incident_sign = (
                        overall_sign
                        if _base_node(vertex)
                        == _base_node(incident_edge.get_source())
                        else -overall_sign
                    )
                    terms.append(f"({incident_sign})*Q({incident_id},0)")

                if not terms:
                    raise ValueError(
                        "the raised-gluon projector has no incident energy "
                        f"denominator at {_base_node(vertex)}"
                    )
                return "(" + "+".join(terms) + ")"

            incident_denominator1 = _incident_energy_denominator(
                e1, vertices1[0], overall_sign1
            )
            incident_denominator2 = _incident_energy_denominator(
                e2, vertices2[0], overall_sign2
            )

            # Eliminate the repeated t-channel momentum at every adjacent
            # three-gluon current.  Rewriting only the Qp endpoint leaves the
            # Q(1000) endpoint cut-dependent: in the gg channel its explicit
            # Q(t-channel) terms have an uncancelled leading collinear pole.
            # The two substitutions are the same exact vertex momentum-
            # conservation identity and are harmless when the other endpoint
            # is a quark current.
            target_node_to_edge_ids = {}
            for vertex in vertices1:
                target_node_to_edge_ids.setdefault(_base_node(vertex), []).append(
                    e1_atts["id"]
                )
            for vertex in vertices2:
                target_node_to_edge_ids.setdefault(_base_node(vertex), []).append(
                    e2_atts["id"]
                )
            for node in cut_graph.graph.get_nodes():
                node_name = _strip_quotes(str(node.get_name()))
                node_int_id = _strip_quotes(str(node.get_attributes().get("int_id")))
                if node_int_id != "V_36" or node_name not in target_node_to_edge_ids:
                    continue

                node_num = node.get("num")
                if not node_num:
                    continue

                print("substituting momentum conservation condition")

                incident_edges = boundary_edges(cut_graph.graph, {node_name})
                num = Es(node_num)

                for target_edge_id in target_node_to_edge_ids[node_name]:
                    target_edge = next(
                        (
                            e
                            for e in incident_edges
                            if e.get_attributes()["id"] == target_edge_id
                        ),
                        None,
                    )
                    if target_edge is None:
                        raise ValueError(
                            f"target edge {target_edge_id} not incident to {node_name}"
                        )

                    edge_id_pattern = Es(f"Q({target_edge_id},y___)")
                    edge_id_replace = Es("0")
                    for incident_edge in incident_edges:
                        incident_edge_id = incident_edge.get_attributes()["id"]
                        if incident_edge_id == target_edge_id:
                            continue
                        edge_sign = (
                            1
                            if _base_node(incident_edge.get_source()) == node_name
                            else -1
                        )
                        edge_id_replace += edge_sign * Es(f"Q({incident_edge_id},y___)")
                    if _base_node(target_edge.get_source()) == node_name:
                        edge_id_replace = -edge_id_replace

                    num = num.replace(edge_id_pattern, edge_id_replace)

                node.get_attributes()["num"] = expr_to_string(num)

            # Finally construct the counter-terms
            #

            repeated_ids = {
                _strip_quotes(str(e1_atts["id"])),
                _strip_quotes(str(e2_atts["id"])),
            }
            matching_raised_pairs = [
                pair
                for pair in getattr(cut_graph, "raised_cut_pairs", ())
                if {pair.cut_edge_id, pair.partner_edge_id} == repeated_ids
            ]
            if len(matching_raised_pairs) > 1:
                raise ValueError(
                    "Multiple raised-cut metadata pairs match the repeated "
                    "t-channel gluons."
                )
            raised_pair = (
                matching_raised_pairs[0] if matching_raised_pairs else None
            )
            repeated_pair_has_cut = (
                _cut_sign_from_attributes(e1_atts) != 0
                or _cut_sign_from_attributes(e2_atts) != 0
            )

            def _single_energy_raised_gluon_numerator(
                edge,
                vertices,
                overall_sign,
                incident_denominator,
                noncut_projector_sign,
            ):
                edge_atts = edge.get_attributes()
                edge_id = _strip_quotes(str(edge_atts["id"]))
                if raised_pair is not None:
                    (
                        denominator_head,
                        projector_momentum_id,
                        projector_coefficient,
                    ) = _raised_gluon_projector_spec(
                        raised_pair, edge_id, overall_sign
                    )
                    projector_denominator = (
                        f"{denominator_head}({edge_atts['id']},0)"
                    )
                elif repeated_pair_has_cut:
                    projector_momentum_id = edge_id
                    projector_coefficient = f"-({overall_sign})"
                    projector_denominator = f"Q({edge_atts['id']},0)"
                else:
                    projector_momentum_id = edge_id
                    projector_coefficient = f"+({noncut_projector_sign})"
                    projector_denominator = incident_denominator
                return (
                    f"-1𝑖*(spenso::g(spenso::coad(8,hedge({_parse_port(edge.get_destination())})),spenso::coad(8,hedge({_parse_port(edge.get_source())})))*spenso::g(spenso::mink(4,hedge({_parse_port(edge.get_destination())})),spenso::mink(4,hedge({_parse_port(edge.get_source())})))+({projector_coefficient})*1/{projector_denominator}*Qp({projector_momentum_id},spenso::mink(4,hedge({_parse_port(vertices[1])})))*Q(1000,spenso::mink(4,hedge({_parse_port(vertices[0])})))*spenso::g(spenso::coad(8,hedge({_parse_port(edge.get_destination())})),spenso::coad(8,hedge({_parse_port(edge.get_source())}))))"
                )

            for e in cut_graph.graph.get_edges():
                e_atts = e.get_attributes()
                if e_atts["id"] == e1_atts["id"]:
                    e_atts["num"] = _single_energy_raised_gluon_numerator(
                        e,
                        vertices1,
                        overall_sign1,
                        incident_denominator1,
                        1,
                    )

                if e_atts["id"] == e2_atts["id"]:
                    e_atts["num"] = _single_energy_raised_gluon_numerator(
                        e,
                        vertices2,
                        overall_sign2,
                        incident_denominator2,
                        -1,
                    )

        return cut_graph

    def _gl085_oriented_25_bucket_expressions(self, physical_cut_graph):
        """Generate and cache the physical buckets and their crossed residues."""

        if not _is_gl085_physical_25_cut(physical_cut_graph):
            raise ValueError("GL085 oriented buckets require its physical cut.")
        _assert_gl085_no_threshold_counterterm_topology(physical_cut_graph)
        cached = getattr(
            self,
            "_gl085_physical_25_bucket_expressions",
            None,
        )
        if cached is not None:
            return cached

        canonical_cut_graph = deepcopy(physical_cut_graph)
        canonical_cut_graph.graph = change_routing(
            canonical_cut_graph.graph,
            _GL085_CFF_CANONICAL_LMB,
        )
        self.emr_processor.identify_and_mark_raised_cuts(canonical_cut_graph)
        canonical_cut_graph = self.modify_t_channel_gluon_numerator(
            canonical_cut_graph
        )

        numerator_factorisation = None
        if self.external_gluon_polarisation:
            numerator_factorisation = (
                self.external_gluon_polarisation_numerator_factorisation
            )
        physical_buckets = self.emr_processor.get_integrand(
            canonical_cut_graph,
            numerator_factorisation=numerator_factorisation,
            cff_partition=_GL085_CFF_PARTITION,
        )
        if set(physical_buckets) != set(_GL085_CFF_BUCKETS):
            raise ValueError(
                "GL085 CFF construction returned unexpected buckets: "
                f"{sorted(physical_buckets)}."
            )
        self._gl085_physical_25_bucket_expressions = physical_buckets
        self._gl085_crossed_25_bucket_expressions = (
            _gl085_crossed_25_residue_buckets(physical_buckets)
        )
        return physical_buckets

    def _get_gl085_split_integrand(self, cut_graph):
        """Build the two physical GL085 {2,5} CFF buckets from one numerator."""

        if self.channel != (0, 0) or not _is_gl085_physical_25_cut(cut_graph):
            raise ValueError(
                "The GL085 split pipeline was called outside its gg physical "
                "initial {0,1}, final {2,5} target."
            )

        source_cut_graph = deepcopy(cut_graph)
        numerator_factorisation = None
        if self.external_gluon_polarisation:
            numerator_factorisation = (
                self.external_gluon_polarisation_numerator_factorisation
            )
        bucket_expressions = self._gl085_oriented_25_bucket_expressions(
            source_cut_graph
        )

        routed_buckets = []
        for bucket in _GL085_CFF_BUCKETS:
            lmb_choice = _GL085_CFF_BUCKET_LMBS[bucket]
            bucket_cut_graph = deepcopy(source_cut_graph)
            bucket_orig_cut_graph = deepcopy(source_cut_graph)
            bucket_cut_graph.graph = change_routing(
                bucket_cut_graph.graph,
                lmb_choice,
            )
            bucket_orig_cut_graph.graph = change_routing(
                bucket_orig_cut_graph.graph,
                lmb_choice,
            )

            self.emr_processor.identify_and_mark_raised_cuts(bucket_cut_graph)
            bucket_cut_graph = self.modify_t_channel_gluon_numerator(
                bucket_cut_graph
            )
            _copy_raised_cut_annotations(
                bucket_cut_graph,
                bucket_orig_cut_graph,
            )

            loop_integrand, raised_cut, is_final_raised = (
                self.eliminate_raised_cuts(
                    bucket_expressions[bucket],
                    bucket_cut_graph,
                )
            )

            uv_approximator = UltraVioletSubtraction(
                loop_integrand,
                deepcopy(bucket_cut_graph),
                self.L,
                self.emr_processor,
                deepcopy(bucket_orig_cut_graph),
                integrated_numerator_factorisation=numerator_factorisation,
                disable_integrated_uv_cts=self.disable_integrated_uv_cts,
                raw_emr_integrand=bucket_expressions[bucket],
                top_self_energy_os_subtraction=(
                    None
                    if self.top_self_energy_renormalisation == "projected-os"
                    else self.top_self_energy_os_subtraction
                ),
                top_self_energy_renormalisation=(
                    self.top_self_energy_renormalisation
                ),
            )
            uv_ct = uv_approximator.construct_uv_counter_terms()

            # This exact topology was audited with the ordinary residue
            # pipeline and has no selected threshold surface.  Re-running
            # that empty probe for both buckets regenerates the full numerator
            # twice, so retain a structural guard instead.
            _assert_gl085_no_threshold_counterterm_topology(
                bucket_orig_cut_graph
            )
            threshold_cts = []

            if uv_ct or threshold_cts:
                raise ValueError(
                    "The GL085 CFF split cannot duplicate counterterms; "
                    f"bucket {bucket} unexpectedly produced {len(uv_ct)} UV "
                    f"and {len(threshold_cts)} threshold terms."
                )

            bucket_integrands = self.leading_virtuality_expansion(
                loop_integrand,
                bucket_cut_graph,
                raised_cut,
            )
            if is_final_raised:
                for routed_integrand in bucket_integrands:
                    routed_integrand.t_derivative = is_final_raised

            for routed_integrand in bucket_integrands:
                routed_integrand.cff_bucket = bucket
                routed_integrand.cff_lmb = tuple(lmb_choice)
            routed_buckets.extend(bucket_integrands)

        return routed_buckets

    def _get_gl085_anti_collinear_25_split_integrand(self, cut_graph):
        """Split the GL085 {2,5} anti-collinear partner by causal surface."""

        if self.channel != (0, 0) or not _is_gl085_anti_collinear_25_cut(
            cut_graph
        ):
            raise ValueError(
                "The GL085 anti-collinear split pipeline received the wrong cut."
            )

        source_cut_graph = deepcopy(cut_graph)
        if _post_cut_graph_has_loop(source_cut_graph):
            raise ValueError(
                "The GL085 crossed {2,5} residue split unexpectedly retained "
                "a post-cut loop."
            )
        # The crossed graph is tree-like, so its ordinary CFF path collapses
        # all loop orientations under one numerator.  Recover the physical
        # CFF instead and take D residues before those numerators are combined.
        physical_counterpart = _gl085_physical_25_counterpart(source_cut_graph)
        self._gl085_oriented_25_bucket_expressions(physical_counterpart)
        bucket_expressions = self._gl085_crossed_25_bucket_expressions

        routed_buckets = []
        for bucket in _GL085_CFF_BUCKETS:
            lmb_choice = _GL085_CFF_BUCKET_LMBS[bucket]
            bucket_cut_graph = deepcopy(source_cut_graph)
            bucket_cut_graph.graph = change_routing(
                bucket_cut_graph.graph,
                lmb_choice,
            )
            self.emr_processor.identify_and_mark_raised_cuts(bucket_cut_graph)
            bucket_cut_graph = self.modify_t_channel_gluon_numerator(
                bucket_cut_graph
            )
            # The canonical graph was normalised during CFF construction;
            # these independently routed copies must expose the same
            # is_cut_DY metadata before the anti-collinear expansion.
            self.emr_processor.normalise_graph(bucket_cut_graph.graph)

            bucket_loop_integrand, bucket_raised_cut, bucket_is_final_raised = (
                self.eliminate_raised_cuts(
                    bucket_expressions[bucket],
                    bucket_cut_graph,
                )
            )
            bucket_integrands = self.leading_virtuality_expansion(
                bucket_loop_integrand,
                bucket_cut_graph,
                bucket_raised_cut,
            )
            if bucket_is_final_raised:
                for routed_integrand in bucket_integrands:
                    routed_integrand.t_derivative = bucket_is_final_raised
            for routed_integrand in bucket_integrands:
                routed_integrand.cff_bucket = bucket
                routed_integrand.cff_lmb = tuple(lmb_choice)
            routed_buckets.extend(bucket_integrands)

        return routed_buckets

    # Derive cff, set the lmb so that the loop momentum coincides with the photon (for DY), and
    # derive the approximated representation.

    def get_integrand(self, cut_graph):

        if self.channel == (0, 0) and _is_gl085_physical_25_cut(cut_graph):
            return self._get_gl085_split_integrand(cut_graph)
        if self.channel == (0, 0) and _is_gl085_anti_collinear_25_cut(cut_graph):
            return self._get_gl085_anti_collinear_25_split_integrand(cut_graph)

        # FIX: cut graph logic and overwriting
        orig_cut_graph = deepcopy(cut_graph)
        skip_threshold_cts = _threshold_counterterms_disabled(
            cut_graph.graph, self.channel
        )
        threshold_collinear_momentum = None
        gluonic_t_channel = True

        # emr_integrand_tmp = self.emr_processor.get_integrand(orig_cut_graph)

        if len(cut_graph.final_cut) > 1 and self.name == "DY":
            lmb_choice = []
            theta_flag = True
            for e in cut_graph.final_cut:
                e_atts = e.get_attributes()
                if _strip_quotes(str(e_atts["particle"])) == "a":
                    lmb_choice.append(e_atts["id"])
            cut_graph.graph = change_routing(cut_graph.graph, lmb_choice)
            orig_cut_graph.graph = change_routing(orig_cut_graph.graph, lmb_choice)

        if len(cut_graph.final_cut) == 1 and self.name == "DY":
            theta_flag = True
            lmb_choice = []
            for e in cut_graph.graph.get_edges():
                e_atts = e.get_attributes()
                if _strip_quotes(str(e_atts["particle"])) == "g":
                    lmb_choice.append(e_atts["id"])
            cut_graph.graph = change_routing(cut_graph.graph, lmb_choice)
            orig_cut_graph.graph = change_routing(orig_cut_graph.graph, lmb_choice)

        if self.name == "tt~":
            lmb_choice = []

            if self.L == 1:
                theta_flag = True
                lmb_choice = [3]

            if self.L == 2:
                ## only for q g channel
                if self.channel == (1, 0) or self.channel == (0, 1):
                    theta_flag = True
                    lmb_choice = [
                        (
                            e.get_attributes()["id"],
                            _strip_quotes(str(e.get_attributes()["particle"])),
                        )
                        for e in cut_graph.final_cut
                    ]

                    new_lmb_choice = []
                    for id, part in lmb_choice:
                        if part in ["d", "d~", "g"]:
                            new_lmb_choice.append(id)

                    massives = []
                    for id, part in lmb_choice:
                        if part not in ["d", "d~", "g"]:
                            massives.append(id)

                    new_lmb_choice.extend(sorted(massives))

                    lmb_choice = new_lmb_choice[:-1]

                if self.channel == (1, -1) or self.channel == (-1, 1):
                    base_graph_name = _strip_quotes(
                        str(cut_graph.graph.get("base_graph_name"))
                    )

                    if base_graph_name in [
                        "GL00",
                        "GL01",
                        "GL02",
                        "GL03",
                        "GL05",
                        "GL13",
                        "GL18",
                    ]:
                        theta_flag = False
                        lmb_choice = [6, 3]

                    if base_graph_name in [
                        "GL17",
                    ]:
                        theta_flag = False
                        lmb_choice = [3, 6]

                    if base_graph_name in ["GL04", "GL15"]:
                        theta_flag = False
                        lmb_choice = [8, 3]

                    if base_graph_name in ["GL06", "GL08", "GL11"]:
                        theta_flag = False
                        lmb_choice = [7, 3]

                    if base_graph_name in ["GL07"]:
                        theta_flag = False
                        gluonic_t_channel = False
                        lmb_choice = [7, 3]

                    if base_graph_name in ["GL09"]:
                        theta_flag = True
                        lmb_choice = [8, 3]

                    if base_graph_name == "GL10":
                        lmb_choice = []
                        theta_flag = False
                        t_count = 0
                        d_count = 0
                        for e in cut_graph.graph.get_edges():
                            e_atts = e.get_attributes()
                            e_part = _strip_quotes(str(e_atts["particle"]))
                            is_cut = _strip_quotes(str(e_atts.get("is_cut", "")))

                            if (e_part == "t" or e_part == "t~") and t_count == 0:
                                lmb_choice.append(e_atts["id"])
                                t_count += 1
                            elif (
                                (e_part == "d" or e_part == "d~")
                                and d_count == 0
                                and is_cut == "0"
                            ):
                                lmb_choice.append(e_atts["id"])
                                d_count += 1

                if self.channel == (0, 0):
                    base_graph_name = _strip_quotes(
                        str(cut_graph.graph.get("base_graph_name"))
                    )

                    if base_graph_name in [
                        "GL000",
                        "GL002",
                        "GL004",
                        "GL006",
                        "GL008",
                        "GL011",
                        "GL012",
                        "GL013",
                        "GL014",
                        "GL015",
                        "GL016",
                        "GL020",
                        "GL021",
                        "GL023",
                        "GL024",
                        "GL026",
                        "GL027",
                        "GL035",
                        "GL039",
                        "GL041",
                        "GL047",
                        "GL079",
                        "GL081",
                        "GL085",
                        "GL087",
                        "GL097",
                    ]:
                        theta_flag = False
                        lmb_choice = [2, 7]

                    if base_graph_name == "GL081":
                        lmb_choice = select_gl081_lmb_choice(cut_graph, lmb_choice)

                    if base_graph_name == "GL087":
                        lmb_choice = [2, 7]

                    if base_graph_name == "GL035":
                        lmb_choice = [3, 5]

                    if base_graph_name in [
                        "GL017",
                        "GL019",
                        "GL022",
                        "GL029",
                        "GL031",
                        "GL033",
                        "GL043",
                        "GL045",
                        "GL051",
                        "GL053",
                        "GL055",
                        "GL057",
                        "GL059",
                        "GL061",
                        "GL063",
                        "GL094",
                        "GL096",
                        "GL101",
                        "GL113",
                    ]:
                        theta_flag = False
                        lmb_choice = [2, 6]
                    if base_graph_name == "GL101":
                        lmb_choice = select_gl101_lmb_choice(cut_graph, lmb_choice)[0]
                    if base_graph_name in ["GL033", "GL057"]:
                        lmb_choice = [3, 4]

                    if base_graph_name == "GL059":
                        theta_flag = True
                        lmb_choice, collinear_sign = select_gl059_lmb_choice(
                            cut_graph
                        )
                        if collinear_sign == -1:
                            threshold_collinear_momentum = -E("p(1)")
                        else:
                            threshold_collinear_momentum = E("p(1)")

                    if base_graph_name in ["GL065"]:
                        theta_flag = False
                        lmb_choice = [2, 5]

                    if base_graph_name in [
                        "GL010",
                        "GL018",
                        "GL067",
                        "GL073",
                        "GL075",
                        "GL083",
                        "GL099",
                    ]:
                        theta_flag = False
                        lmb_choice = [2, 8]

                        if base_graph_name == "GL018":
                            theta_flag = True
                    if base_graph_name == "GL093":
                        theta_flag = False
                        lmb_choice = [2, 5]
                    if base_graph_name == "GL071":
                        theta_flag = True
                        lmb_choice, collinear_sign = select_gl071_lmb_choice(
                            cut_graph
                        )
                        if collinear_sign == -1:
                            threshold_collinear_momentum = -E("p(1)")
                        else:
                            threshold_collinear_momentum = E("p(1)")
                    if base_graph_name == "GL077":
                        theta_flag = False
                        lmb_choice = [5, 6]

                    if base_graph_name in ["GL091"]:
                        theta_flag = False
                        lmb_choice = [5, 8]

                    if base_graph_name == "GL105":
                        theta_flag = False
                        lmb_choice = [3, 6]

                    if base_graph_name == "GL107":
                        theta_flag = False
                        lmb_choice = [4, 5]

                    if base_graph_name in ["GL109", "GL111"]:
                        theta_flag = False
                        lmb_choice = [4, 6]

                    if base_graph_name == "GL115":
                        theta_flag = True
                        lmb_choice = [6, 8]

                    if base_graph_name == "GL117":
                        theta_flag = False
                        lmb_choice = [3, 6]

                    if base_graph_name == "GL119":
                        theta_flag = False
                        lmb_choice = [8, 6]

                    if base_graph_name == "GL123":
                        theta_flag = False
                        lmb_choice = [2, 6]

                    gl085_family_lmb = _gl085_physical_family_lmb(cut_graph)
                    if gl085_family_lmb is not None:
                        theta_flag = False
                        lmb_choice = list(gl085_family_lmb)

            print(lmb_choice)
            # lmb_choice = [7, 2]
            # lmb_choice = [2, 7]

            lmb_choice = _select_production_lmb_choice(
                cut_graph, lmb_choice, self.channel
            )
            cut_graph.graph = change_routing(cut_graph.graph, lmb_choice)
            orig_cut_graph.graph = change_routing(orig_cut_graph.graph, lmb_choice)

        print("heyyy")
        self.emr_processor.identify_and_mark_raised_cuts(cut_graph)
        if gluonic_t_channel:
            cut_graph = self.modify_t_channel_gluon_numerator(cut_graph)
        _copy_raised_cut_annotations(cut_graph, orig_cut_graph)

        numerator_factorisation = None
        if self.external_gluon_polarisation:
            numerator_factorisation = (
                self.external_gluon_polarisation_numerator_factorisation
            )

        print(cut_graph.graph)
        emr_integrand = self.emr_processor.get_integrand(
            cut_graph,
            numerator_factorisation=numerator_factorisation,
        )

        # print(emr_integrand)

        loop_integrand, raised_cut, is_final_raised = self.eliminate_raised_cuts(
            emr_integrand, cut_graph
        )

        uv_approximator = UltraVioletSubtraction(
            loop_integrand,
            deepcopy(cut_graph),
            self.L,
            self.emr_processor,
            deepcopy(orig_cut_graph),
            integrated_numerator_factorisation=numerator_factorisation,
            disable_integrated_uv_cts=self.disable_integrated_uv_cts,
            raw_emr_integrand=emr_integrand,
            top_self_energy_os_subtraction=(
                None
                if self.top_self_energy_renormalisation == "projected-os"
                else self.top_self_energy_os_subtraction
            ),
            top_self_energy_renormalisation=(
                self.top_self_energy_renormalisation
            ),
        )
        uv_ct = uv_approximator.construct_uv_counter_terms()

        threshold_cts = []
        if not skip_threshold_cts and _post_cut_graph_has_loop(orig_cut_graph):
            threshold_approximator = ThresholdSubtractor(
                deepcopy(orig_cut_graph),
                self.params,
                self.name,
                self.L,
                theta_flag,
                numerator_factorisation=numerator_factorisation,
                threshold_collinear_momentum=threshold_collinear_momentum,
                emr_state_name=self.emr_state_name,
                threshold_h_function=getattr(
                    self, "threshold_h_function", None
                ),
            )
            threshold_cts = threshold_approximator.construct_threshold_counter_terms()
        # print("this emr")
        # print(emr_integrand)

        loop_integrand = self.leading_virtuality_expansion(
            loop_integrand, cut_graph, raised_cut
        )

        print(len(loop_integrand))

        print("*** " * 10)
        # for lp in loop_integrand:
        #    print(lp.integrand)

        if is_final_raised:
            for ct in uv_ct:
                ct.t_derivative = is_final_raised
            for lp in loop_integrand:
                lp.t_derivative = is_final_raised
            for ct in threshold_cts:
                ct.t_derivative = is_final_raised

        loop_integrand = (
            loop_integrand + threshold_cts + uv_ct  #   # + soft_t_channel_gluon_cts
        )

        print("returning final integrand")

        return loop_integrand
