"""Generation-time modes for massive-top self-energy renormalisation."""

from __future__ import annotations

from typing import Final


TOP_SELF_ENERGY_RENORMALISATION_MODES: Final[tuple[str, ...]] = (
    "no-os",
    "os",
    "projected-os",
)
PROJECTED_OS_CONSTRUCTION_SCHEMA_VERSION: Final[int] = 1
# Private momentum label used while contracting the open spatial Dirac tensor
# with its host graph. It must not alias either physical beam-projector label
# (1001/1002) or a production graph edge: an alias changes a tensor linear in
# the projected momentum into a spurious higher-degree polynomial.
PROJECTED_OS_SPATIAL_PROBE_ID: Final[str] = "2147483646"


def resolve_top_self_energy_renormalisation(
    mode: str | None = None,
    legacy_os_subtraction: bool | None = None,
    *,
    legacy_is_explicit: bool | None = None,
) -> str:
    """Resolve the enum while retaining the historical boolean interface.

    ``legacy_is_explicit`` is useful at parser/card boundaries, where an
    omitted boolean can be distinguished from an explicitly supplied false
    value.  Direct callers get strict conflict checking whenever they pass a
    non-``None`` legacy value.
    """

    resolved_mode = None if mode is None else str(mode).strip()
    if (
        resolved_mode is not None
        and resolved_mode not in TOP_SELF_ENERGY_RENORMALISATION_MODES
    ):
        choices = ", ".join(TOP_SELF_ENERGY_RENORMALISATION_MODES)
        raise ValueError(
            "top_self_energy_renormalisation must be one of "
            f"{choices}; got {resolved_mode!r}"
        )

    if legacy_os_subtraction is not None and type(legacy_os_subtraction) is not bool:
        raise ValueError(
            "top_self_energy_os_subtraction must be a boolean when supplied"
        )
    if legacy_is_explicit is None:
        legacy_is_explicit = legacy_os_subtraction is not None

    legacy_mode = "os" if legacy_os_subtraction else "no-os"
    if resolved_mode is None:
        return legacy_mode
    if legacy_is_explicit and resolved_mode != legacy_mode:
        raise ValueError(
            "Conflicting top self-energy renormalisation settings: "
            f"top_self_energy_renormalisation={resolved_mode!r} but "
            "top_self_energy_os_subtraction="
            f"{legacy_os_subtraction!r} selects {legacy_mode!r}."
        )
    return resolved_mode


def projected_os_schema_for_mode(mode: str) -> int:
    """Return the construction schema carried in generation fingerprints."""

    resolved = resolve_top_self_energy_renormalisation(mode)
    return (
        PROJECTED_OS_CONSTRUCTION_SCHEMA_VERSION
        if resolved == "projected-os"
        else 0
    )


__all__ = [
    "PROJECTED_OS_CONSTRUCTION_SCHEMA_VERSION",
    "PROJECTED_OS_SPATIAL_PROBE_ID",
    "TOP_SELF_ENERGY_RENORMALISATION_MODES",
    "projected_os_schema_for_mode",
    "resolve_top_self_energy_renormalisation",
]
