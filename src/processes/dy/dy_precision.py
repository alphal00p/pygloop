"""Sampling-independent precision escalation for DY and ttbar evaluators."""
from dataclasses import dataclass


DEFAULT_PRECISION_LADDER = (16, 32)


def validate_precision_ladder(levels):
    levels = tuple(levels)
    if (not levels or levels[0] != 16
            or any(type(level) is not int for level in levels)
            or any(b <= a for a, b in zip(levels, levels[1:]))):
        raise ValueError("precision_ladder must start at 16 and contain strictly increasing integer decimal precisions; zero is the failure outcome, not a precision")
    return levels


@dataclass(frozen=True)
class PrecisionResult:
    value: object
    accepted_precision: int | None
    failures: tuple


def run_precision_ladder(levels, evaluate, *, start=0, count=None):
    """Retry numerical failures only; the caller zeros an exhausted sample.

    ``start=1`` lets the ordinary evaluator reuse its already-completed native
    evaluation. Paired packets use the same controller starting at level zero.
    Programming/configuration exceptions deliberately propagate.
    """
    failures = []
    for digits in validate_precision_ladder(levels)[start:]:
        if count is not None:
            count(digits, "attempt")
        try:
            value = evaluate(digits)
        except ArithmeticError as exc:
            failures.append((digits, str(exc)))
            if count is not None:
                count(digits, "failed")
        else:
            if count is not None:
                count(digits, "accepted")
            return PrecisionResult(value, digits, tuple(failures))
    return PrecisionResult(None, None, tuple(failures))
