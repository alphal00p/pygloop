"""Physical/graph-channel Symbolica integration for DY processes.

The default path samples the Cartesian sum of all ``(physical, graph)`` leaves.
When a physical-channel probability floor is requested, an outer native
Symbolica discrete integrator is mixed with an exact per-channel floor and one
native graph/continuous integrator is used inside each physical channel.  In
both modes, only the selected expensive contribution is evaluated.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from symbolica import NumericalIntegrator, Probe, Sample

from processes.dy.dy import (
    DY,
    DY_GROUPED_SOFT_PAIR_SELECTOR,
    DY_STABILITY_FAILURE_COUNTERS,
)
from utils.utils import IntegrationResult, logger, pygloopException


PHYSICAL_CHANNEL_PARTONS: dict[str, frozenset[tuple[int, int]]] = {
    "qqbar": frozenset({(1, -1), (-1, 1)}),
    "qg": frozenset({(1, 0), (0, 1)}),
    "gg": frozenset({(0, 0)}),
}

_PhysicalRow = tuple[int, list[float], int]
_EvaluatedRow = tuple[int, float, tuple[float, ...]]

_LIVE_DIAGNOSTIC_COUNTERS = (
    "large_weight_zeroed_count",
    "large_weight_retry_count",
    "large_weight_salvaged_count",
    "large_weight_unstable_count",
    "stability_float_pair_accepted_count",
    "stability_hp_retry_count",
    "stability_hp_accepted_count",
    "stability_hp_disagreement_count",
    "stability_hp_nonfinite_count",
    "stability_hp_error_count",
    "unstable_count",
    "unstable_retry_count",
    "unstable_salvaged_count",
    "t_solver_float_failure_sample_count",
    "t_solver_hp_retry_count",
    "t_solver_hp_salvaged_count",
    "t_solver_hp_unresolved_count",
    "soft_mirror_pair_count",
    "soft_mirror_large_trigger_count",
    "soft_mirror_hp_orbit_retry_count",
    "soft_mirror_hp_orbit_salvaged_count",
    "soft_mirror_hp_orbit_failure_count",
    "nan_weight_count",
    "nan_weight_rotated_count",
)


def _bundle_symmetrisation_provenance(
    process: DY,
) -> tuple[tuple[str, object], ...]:
    metadata = getattr(
        getattr(process, "compiled_bundle", None), "bundle_metadata", {}
    )
    if not isinstance(metadata, Mapping):
        return ()
    records: list[tuple[str, object]] = []
    generation = metadata.get("dy_generation_settings")
    if isinstance(generation, Mapping):
        key = "common.initial_state.symmetrise_p1_p2"
        if key in generation:
            records.append((f"dy_generation_settings.{key}", generation[key]))
    campaign = metadata.get("pp_ttbar_campaign_generation")
    if isinstance(campaign, Mapping) and "symmetrise_p1_p2" in campaign:
        records.append((
            "pp_ttbar_campaign_generation.symmetrise_p1_p2",
            campaign["symmetrise_p1_p2"],
        ))
    return tuple(records)


@dataclass
class DYPhysicalChannel:
    """One physical branch of the joint integration."""

    label: str
    process: DY
    integrand_implementation: Mapping[str, Any]
    graph_channel_indices: Sequence[int] | None = None
    source_card: str | None = None
    graph_channel_names: tuple[str, ...] = field(init=False)
    sampling_factor: float = field(init=False)

    def __post_init__(self) -> None:
        self.label = str(self.label).strip().lower()
        if self.label not in PHYSICAL_CHANNEL_PARTONS:
            known = ", ".join(sorted(PHYSICAL_CHANNEL_PARTONS))
            raise pygloopException(
                f"Unknown DY physical channel {self.label!r}; expected {known}."
            )
        partons = tuple(int(value) for value in self.process.dy_channel)
        if partons not in PHYSICAL_CHANNEL_PARTONS[self.label]:
            raise pygloopException(
                f"Physical-channel label {self.label!r} does not match "
                f"DY channel {partons}."
            )
        if (
            self.process.process_name.lower() == "tt~"
            and self.process.n_loops == 2
        ):
            if not getattr(self.process, "symmetrise_p1_p2", False):
                raise pygloopException(
                    "Two-loop ttbar physical-channel integration requires "
                    "symmetrise_p1_p2=True for "
                    f"{self.label}."
                )
            for source, value in _bundle_symmetrisation_provenance(self.process):
                if value is not True:
                    raise pygloopException(
                        f"Physical channel {self.label} uses a bundle whose "
                        f"provenance records {source}={value!r}; regenerate it "
                        "with p1<->p2 symmetrisation."
                    )
        if isinstance(self.process, DY) and self.process.integrate_beams:
            if self.process.dy_pdf_set is None:
                raise pygloopException(
                    "Hadronic physical-channel multi-channeling requires PDFs "
                    f"for {self.label}."
                )
            # A physical branch represents the complete one-light-flavour pp
            # luminosity family, not one ordered cut used to generate its hard
            # kernel. The provider is shared by hard and auxiliary terms.
            self.process.dy_pdf_luminosity_family = self.label
            self.process._dy_pdf_provider = None

        implementation = dict(
            self.process._normalize_integrand_implementation(
                dict(self.integrand_implementation)
            )
        )
        if implementation.get("integrand_type") != "zenos":
            raise pygloopException(
                "Physical-channel multi-channeling requires ZenoS evaluators."
            )
        self.integrand_implementation = implementation

        all_names = self.process.graph_channel_names(implementation)
        if not all_names:
            raise pygloopException(
                f"Physical channel {self.label} has no compiled graph channels."
            )
        if self.graph_channel_indices is None:
            configured_indices = self.process._integration_graph_channel_indices(
                implementation, None
            )
            indices = tuple(
                range(len(all_names))
                if configured_indices is None
                else configured_indices
            )
        else:
            indices = tuple(int(index) for index in self.graph_channel_indices)
        if not indices or len(set(indices)) != len(indices):
            raise pygloopException(
                f"Physical channel {self.label} requires unique graph channels."
            )
        for index in indices:
            if index == DY_GROUPED_SOFT_PAIR_SELECTOR:
                grouped_pair = self.process._grouped_soft_pair(implementation)
                if grouped_pair is None:
                    raise pygloopException(
                        f"Physical channel {self.label} selects a grouped "
                        "soft pair without a card mapping."
                    )
                continue
            if index < 0 or index >= len(all_names):
                raise pygloopException(
                    f"Graph channel {index} is out of range for physical channel "
                    f"{self.label} ({len(all_names)} channels)."
                )
        self.graph_channel_indices = indices
        self.graph_channel_names = tuple(
            self.process._integration_graph_channel_name(implementation, index)
            if index == DY_GROUPED_SOFT_PAIR_SELECTOR
            else all_names[index]
            for index in indices
        )
        graph_weights = implementation.get("dy_graph_weights")
        if graph_weights is not None:
            if len(graph_weights) != len(all_names):
                raise pygloopException(
                    f"Physical channel {self.label} supplies {len(graph_weights)} "
                    f"graph weights for {len(all_names)} bundle channels."
                )
            if any(not math.isfinite(float(weight)) for weight in graph_weights):
                raise pygloopException(
                    f"Physical channel {self.label} graph weights must be finite."
                )

        self.sampling_factor = 1.0
        if (
            self.process.dy_msbar_scheme_counterterm
            and self.process.process_name.lower() == "tt~"
            and self.label in {"qqbar", "qg", "gg"}
        ):
            # Apply each channel's established forward-to-cross-section factor
            # while sampling so the outer discrete layer optimises the actual
            # pp hard sum.  The crossed qg forward construction has the opposite
            # orientation sign from qqbar and gg.  Auxiliary Born convolutions
            # have their own, independent normalisation.
            self.sampling_factor = 0.5 if self.label == "qg" else -0.5

    @property
    def dimension(self) -> int:
        return self.process.integration_dimension(self.integrand_implementation)


@dataclass
class _WeightedMoments:
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def start_batch(self, n_samples: int) -> None:
        self.n += int(n_samples)

    def add_selected(self, value: float) -> None:
        self.total += value
        self.total_sq += value * value

    def estimate(self) -> tuple[float, float]:
        if self.n == 0:
            return 0.0, 0.0
        average = self.total / self.n
        if self.n == 1:
            return average, 0.0
        centered = max(0.0, self.total_sq - self.total * self.total / self.n)
        error = math.sqrt(centered / (self.n * (self.n - 1)))
        return average, error


@dataclass
class _HierarchicalBatch:
    outer_samples: list[Sample]
    outer_global_indices: list[int]
    inner_samples: list[Sample]
    sample_paths: list[tuple[int, int]]
    positions_by_channel: tuple[list[int], ...]
    raw_values: list[float]
    bin_values: list[tuple[float, ...]]
    diagnostics: IntegrationResult
    adaptive_probabilities: tuple[float, ...]
    effective_probabilities: tuple[float, ...]
    floor_counts: tuple[int, ...]
    adaptive_counts: tuple[int, ...]


class DYPhysicalChannelIntegrationResult(IntegrationResult):
    """Integration result with physical- and graph-channel observers."""

    def __init__(self) -> None:
        super().__init__(0.0, 0.0)
        self.hard_central_value = 0.0
        self.hard_error = 0.0
        self.physical_channel_estimates: dict[str, dict[str, Any]] = {}
        self.hard_channel_covariance: dict[str, dict[str, float]] = {}
        self.physical_channel_covariance: dict[str, dict[str, float]] = {}
        self.graph_channel_estimates: dict[str, dict[str, dict[str, Any]]] = {}
        self.worker_allocation: dict[str, int] = {}
        self.evaluation_backend = "saved-jit"
        self.batch_size = 256
        self.outer_discrete_channel_count = 0
        self.joint_discrete_channel_count = 0
        self.q_bin_edges: tuple[float, ...] = ()
        self.q_bin_estimates: list[dict[str, Any]] = []
        self.warmup_iterations = 0
        self.warmup_samples = 0
        self.production_samples = 0
        self.production_iterations = 0
        self.target_relative_precision: float | None = None
        self.achieved_relative_precision: float | None = None
        self.precision_target_met: bool | None = None
        self.precision_stop_reason = "fixed_iterations"
        self.precision_history: list[dict[str, Any]] = []
        self.preview_auxiliary_central_value = 0.0
        self.preview_auxiliary_error = 0.0
        self.iteration_diagnostics: list[dict[str, Any]] = []
        self.physical_channel_min_fraction = 0.0
        self.physical_channel_floor_samples: dict[str, int] = {}
        self.physical_channel_adaptive_samples: dict[str, int] = {}
        self.physical_channel_effective_probabilities: dict[str, float] = {}
        self.physical_channel_probability_history: list[dict[str, Any]] = []
        self.sampling_topology = "joint-leaf"

    def str_report(self, target: float | None = None) -> str:
        report = super().str_report(target)
        if not self.physical_channel_estimates:
            return report
        allocation = ", ".join(
            f"{label}={workers}"
            for label, workers in self.worker_allocation.items()
        )
        lines = [
            report,
            "Physical-channel sampling:",
            f"  backend={self.evaluation_backend}, batch={self.batch_size}, "
            f"warmup={self.warmup_samples}, production={self.production_samples}, "
            f"topology={self.sampling_topology}, "
            f"joint leaves={self.joint_discrete_channel_count}",
            f"  physical floor={self.physical_channel_min_fraction:.6f}",
            f"  persistent workers: {allocation or 'in-process'}",
        ]
        if self.target_relative_precision is not None:
            achieved = self.achieved_relative_precision
            achieved_text = (
                f"{achieved:.6%}" if achieved is not None else "unavailable"
            )
            lines.append(
                "  precision target="
                f"{self.target_relative_precision:.6%}, achieved={achieved_text}, "
                f"met={self.precision_target_met}, "
                f"stop={self.precision_stop_reason}, "
                f"iterations={self.production_iterations}"
            )
        lines.append("Physical-channel observer estimates:")
        for label, row in self.physical_channel_estimates.items():
            allocation = ""
            if "floor_samples" in row:
                allocation = (
                    f"; floor={row['floor_samples']}, "
                    f"adaptive={row['adaptive_samples']}"
                )
            lines.append(
                f"  {label:<8}: {row['central_value']:+.16e} +/- "
                f"{row['error']:.4e} [{row['selected_samples']} selected"
                f"{allocation}]"
            )
        channel_sum = sum(
            float(row["central_value"])
            for row in self.physical_channel_estimates.values()
        )
        lines.append(f"  {'sum':<8}: {channel_sum:+.16e}")
        lines.append(f"  {'total':<8}: {self.central_value:+.16e}")
        if self.physical_channel_covariance:
            lines.append("Physical-channel correlation matrix:")
            for first_label, covariance_row in (
                self.physical_channel_covariance.items()
            ):
                correlations = []
                first_variance = covariance_row[first_label]
                for second_label, covariance in covariance_row.items():
                    second_variance = self.physical_channel_covariance[
                        second_label
                    ][second_label]
                    denominator = math.sqrt(
                        max(0.0, first_variance * second_variance)
                    )
                    correlation = covariance / denominator if denominator else 0.0
                    correlations.append(f"{second_label}={correlation:+.6f}")
                lines.append(f"  {first_label:<8}: " + ", ".join(correlations))
        if self.q_bin_estimates:
            lines.append("Q-bin observer estimates:")
            for row in self.q_bin_estimates:
                lines.append(
                    f"  [{row['q_min']:.8g}, {row['q_max']:.8g}]: "
                    f"{row['central_value']:+.16e} +/- {row['error']:.4e}"
                )
        return "\n".join(lines)


_WORKER_PROCESS: DY | None = None
_WORKER_IMPLEMENTATION: dict[str, Any] | None = None
_WORKER_PARAMETERISATION = "spherical"
_WORKER_PHASE = "real"
_WORKER_GRAPH_INDICES: tuple[int, ...] = ()
_WORKER_BACKEND = "saved-jit"
_WORKER_BATCH_SIZE = 256
_WORKER_SAMPLING_FACTOR = 1.0
_WORKER_Q_BINS: tuple[float, ...] = ()


_DIAGNOSTIC_SOFT_PAIR_SELECTOR = -2
_DIAGNOSTIC_SOFT_PAIR_ENV = "PYGLOOP_DIAGNOSTIC_LOCAL_SOFT_PAIR_SUM"
_DIAGNOSTIC_SOFT_PAIR_TRANSFORM_ENV = (
    "PYGLOOP_DIAGNOSTIC_LOCAL_SOFT_PAIR_TRANSFORM"
)


def _diagnostic_soft_pair_coordinates(
    coordinates: Sequence[float], selected_graph: bool | int
) -> list[float]:
    transformed = list(coordinates)
    if selected_graph != 1:
        return transformed
    mode = os.environ.get(_DIAGNOSTIC_SOFT_PAIR_TRANSFORM_ENV, "same")
    if mode in {"polar", "invert", "polar-beams", "invert-beams"}:
        transformed[2] = 1.0 - transformed[2]
    if mode in {"invert", "invert-beams"}:
        transformed[1] = (transformed[1] + 0.5) % 1.0
    if mode in {"polar-beams", "invert-beams"}:
        transformed[-2], transformed[-1] = transformed[-1], transformed[-2]
    return transformed


def _physical_worker_init(
    process_builder_inputs: Mapping[str, Any],
    implementation: Mapping[str, Any],
    parameterisation: str,
    phase: str,
    graph_channel_indices: Sequence[int],
    backend: str,
    batch_size: int,
    sampling_factor: float,
    q_bins: Sequence[float],
) -> None:
    global _WORKER_PROCESS
    global _WORKER_IMPLEMENTATION
    global _WORKER_PARAMETERISATION
    global _WORKER_PHASE
    global _WORKER_GRAPH_INDICES
    global _WORKER_BACKEND
    global _WORKER_BATCH_SIZE
    global _WORKER_SAMPLING_FACTOR
    global _WORKER_Q_BINS

    _WORKER_PROCESS = DY(
        **dict(process_builder_inputs),
        clean=False,
        logger_level=50,
        skip_gl_worker_init=True,
    )
    _WORKER_IMPLEMENTATION = dict(implementation)
    _WORKER_PARAMETERISATION = parameterisation
    _WORKER_PHASE = phase
    _WORKER_GRAPH_INDICES = tuple(int(index) for index in graph_channel_indices)
    _WORKER_BACKEND = backend
    _WORKER_BATCH_SIZE = int(batch_size)
    _WORKER_SAMPLING_FACTOR = float(sampling_factor)
    _WORKER_Q_BINS = tuple(float(edge) for edge in q_bins)


def _q_bin_physical_z_intervals(
    process: DY,
    implementation: Mapping[str, Any],
    coordinates: Sequence[float],
    q_bins: Sequence[float],
) -> tuple[tuple[float, float] | None, ...]:
    if not q_bins:
        return ()
    if str(implementation.get("dy_beam_parameterisation", "x1_x2")) != "x1_x2":
        raise pygloopException(
            "DY Q-bin batching requires the x1_x2 beam parameterisation."
        )
    beam_offset = 3 * int(process.n_loops) + int(
        process.sampled_uses_z(dict(implementation))
    )
    if len(coordinates) < beam_offset + 2:
        raise pygloopException(
            "DY Q-bin sample is missing the two beam-fraction coordinates."
        )
    x1 = float(coordinates[beam_offset])
    x2 = float(coordinates[beam_offset + 1])
    shat = x1 * x2 * float(process.e_cm) ** 2
    if shat <= 0.0:
        return tuple(None for _edge in q_bins[:-1])
    base_interval = process.dy_physical_z_interval(x1, x2)
    if base_interval is None:
        return tuple(None for _edge in q_bins[:-1])
    base_min, base_max = base_interval
    intervals: list[tuple[float, float] | None] = []
    for q_min, q_max in zip(q_bins, q_bins[1:]):
        z_min = max(base_min, float(q_min) ** 2 / shat)
        z_max = min(base_max, float(q_max) ** 2 / shat)
        intervals.append((z_min, z_max) if z_min < z_max else None)
    return tuple(intervals)


def _evaluate_rows(
    process: DY,
    implementation: Mapping[str, Any],
    parameterisation: str,
    phase: str,
    graph_channel_indices: Sequence[int],
    backend: str,
    batch_size: int,
    sampling_factor: float,
    q_bins: Sequence[float],
    rows: Sequence[_PhysicalRow],
) -> tuple[list[_EvaluatedRow], IntegrationResult, int]:
    started = time.monotonic()
    policy = getattr(process, "_dy_precision_ladder", None)
    if callable(policy):
        policy(implementation)  # Initialize dynamic level counters before the snapshot.
    counters_before = {
        name: value
        for name, value in vars(process).items()
        if name.endswith("_count") and type(value) is int
    }
    sums_before = {
        name: float(getattr(process, name, 0.0))
        for name in (
            "large_weight_zeroed_signed_sum",
            "large_weight_zeroed_abs_sum",
        )
    }
    t_solver_histograms_before = {
        name: dict(getattr(process, name, {}))
        for name in (
            "t_solver_failure_surfaces",
            "t_solver_failure_terms",
        )
    }
    buckets: dict[int, list[tuple[int, list[float]]]] = {}
    for global_index, coordinates, local_graph_index in rows:
        if local_graph_index < 0 or local_graph_index >= len(graph_channel_indices):
            raise pygloopException(
                f"Joint Symbolica graph index {local_graph_index} is out of range."
            )
        buckets.setdefault(local_graph_index, []).append(
            (int(global_index), list(coordinates))
        )

    ordered_values: list[_EvaluatedRow] = []
    n_q_bins = max(0, len(q_bins) - 1)
    for local_graph_index, bucket in buckets.items():
        graph_index = int(graph_channel_indices[local_graph_index])
        graph_selector: bool | int = graph_index
        if (
            graph_index == -1
            and os.environ.get("PYGLOOP_DIAGNOSTIC_LOCAL_GRAPH_SUM") == "1"
        ):
            graph_selector = False
        sample_batch_size = max(1, batch_size // max(1, n_q_bins))
        for start in range(0, len(bucket), sample_batch_size):
            block = bucket[start : start + sample_batch_size]
            lane_coordinates: list[list[float]] = []
            lane_implementations: list[dict[str, Any]] = []
            lane_share_keys: list[int] = []
            lane_destinations: list[tuple[int, int]] = []
            binned_values = [
                [0.0 for _bin_index in range(n_q_bins)] for _row in block
            ]
            if n_q_bins:
                for block_index, (global_index, coordinates) in enumerate(block):
                    intervals = _q_bin_physical_z_intervals(
                        process, implementation, coordinates, q_bins
                    )
                    for bin_index, interval in enumerate(intervals):
                        if interval is None:
                            continue
                        lane_implementation = dict(implementation)
                        (
                            lane_implementation["dy_physical_z_min"],
                            lane_implementation["dy_physical_z_max"],
                        ) = interval
                        lane_coordinates.append(coordinates)
                        lane_implementations.append(lane_implementation)
                        lane_share_keys.append(global_index)
                        lane_destinations.append((block_index, bin_index))
            else:
                lane_coordinates = [row[1] for row in block]
                lane_implementations = [dict(implementation) for _row in block]
                lane_share_keys = [row[0] for row in block]
                lane_destinations = [
                    (block_index, -1) for block_index, _row in enumerate(block)
                ]

            if not lane_coordinates:
                for (global_index, _coordinates), bin_values in zip(
                    block, binned_values, strict=True
                ):
                    ordered_values.append(
                        (global_index, 0.0, tuple(bin_values))
                    )
                continue
            graph_selectors = (graph_selector,)
            grouped_pair: tuple[int, int, str] | None = None
            if graph_index == DY_GROUPED_SOFT_PAIR_SELECTOR:
                grouped_pair = process._grouped_soft_pair(implementation)
                if grouped_pair is not None:
                    graph_selectors = grouped_pair[:2]
                elif os.environ.get(_DIAGNOSTIC_SOFT_PAIR_ENV) == "1":
                    # Retain compatibility with historical diagnostic cards.
                    graph_selectors = (0, 1)
                else:
                    raise pygloopException(
                        "A grouped DY soft-pair channel was selected without "
                        "a card mapping."
                    )
            block_values = [0.0 for _row in lane_coordinates]
            block_diagnostics: list[dict[str, int]] = [
                {} for _row in lane_coordinates
            ]
            for selected_graph in graph_selectors:
                if grouped_pair is None:
                    selected_coordinates = [
                        _diagnostic_soft_pair_coordinates(row, selected_graph)
                        for row in lane_coordinates
                    ]
                else:
                    selected_coordinates = [
                        process._grouped_soft_pair_coordinates(
                            row,
                            transformed_member=(
                                selected_graph == grouped_pair[1]
                            ),
                        )
                        for row in lane_coordinates
                    ]
                selected_values, selected_diagnostics = (
                    process.integrand_xspace_batch(
                        selected_coordinates,
                        parameterisation,
                        lane_implementations,
                        phase,
                        [selected_graph for _row in lane_coordinates],
                        integrand_backend=backend,
                        kinematic_share_keys=lane_share_keys,
                    )
                )
                for lane, value in enumerate(selected_values):
                    block_values[lane] += float(value)
                for lane, diagnostic in enumerate(selected_diagnostics):
                    for name, count in diagnostic.items():
                        block_diagnostics[lane][name] = (
                            block_diagnostics[lane].get(name, 0) + int(count)
                        )
            scalar_values = [0.0 for _row in block]
            for (block_index, bin_index), value, _diagnostics in zip(
                lane_destinations,
                block_values,
                block_diagnostics,
                strict=True,
            ):
                if len(graph_selectors) > 1 and any(
                    _diagnostics.get(name, 0) for name in DY_STABILITY_FAILURE_COUNTERS
                ):
                    # One failed member zeros the full correlated sample.
                    value = 0.0
                scaled_value = float(value) * sampling_factor
                scalar_values[block_index] += scaled_value
                if bin_index >= 0:
                    binned_values[block_index][bin_index] = scaled_value
            for (global_index, _coordinates), value, bin_values in zip(
                block, scalar_values, binned_values, strict=True
            ):
                ordered_values.append((global_index, value, tuple(bin_values)))

    result = IntegrationResult(
        0.0,
        0.0,
        n_samples=len(rows),
        elapsed_time=time.monotonic() - started,
    )
    counter_aliases = {
        "rotation_unstable_count": "unstable_count",
        "rotation_hp_retry_count": "unstable_retry_count",
        "rotation_hp_salvaged_count": "unstable_salvaged_count",
        "large_weight_hp_retry_count": "large_weight_retry_count",
        "large_weight_hp_salvaged_count": "large_weight_salvaged_count",
    }
    for name, before in counters_before.items():
        result_name = counter_aliases.get(name, name)
        if hasattr(result, result_name) or name.startswith("stability_precision_"):
            setattr(result, result_name, int(getattr(process, name)) - before)
    for name, before in sums_before.items():
        setattr(result, name, float(getattr(process, name, 0.0)) - before)
    for name, before in t_solver_histograms_before.items():
        current = getattr(process, name, {})
        setattr(
            result,
            name,
            {
                label: int(count) - int(before.get(label, 0))
                for label, count in current.items()
                if int(count) != int(before.get(label, 0))
            },
        )
    for name in (
        "max_wgt",
        "max_wgt_point",
        "max_wgt_jacobian",
        "max_wgt_momentum_point",
        "max_stable_wgt",
        "max_stable_wgt_point",
        "max_stable_wgt_jacobian",
        "max_stable_wgt_momentum_point",
        "max_preclip_wgt",
        "max_preclip_wgt_point",
        "max_preclip_wgt_momentum_point",
    ):
        if hasattr(process, name):
            setattr(result, name, getattr(process, name))
    return ordered_values, result, os.getpid()


def _physical_worker_task(
    rows: Sequence[_PhysicalRow],
) -> tuple[list[_EvaluatedRow], IntegrationResult, int]:
    if _WORKER_PROCESS is None or _WORKER_IMPLEMENTATION is None:
        raise pygloopException("Physical-channel worker was not initialized.")
    return _evaluate_rows(
        _WORKER_PROCESS,
        _WORKER_IMPLEMENTATION,
        _WORKER_PARAMETERISATION,
        _WORKER_PHASE,
        _WORKER_GRAPH_INDICES,
        _WORKER_BACKEND,
        _WORKER_BATCH_SIZE,
        _WORKER_SAMPLING_FACTOR,
        _WORKER_Q_BINS,
        rows,
    )


def _allocate_channel_workers(
    channels: Sequence[DYPhysicalChannel], n_cores: int
) -> dict[str, int]:
    if n_cores <= 1:
        return {channel.label: 0 for channel in channels}
    if n_cores < len(channels):
        raise pygloopException(
            "Physical-channel persistent workers require either one core or at "
            "least one core per physical channel."
        )
    allocation = {channel.label: 1 for channel in channels}
    weights = {
        channel.label: max(1, len(channel.graph_channel_names))
        for channel in channels
    }
    for _ in range(n_cores - len(channels)):
        selected = max(
            channels,
            key=lambda channel: weights[channel.label]
            / allocation[channel.label],
        )
        allocation[selected.label] += 1
    return allocation


class _PersistentChannelExecutors:
    def __init__(
        self,
        channels: Sequence[DYPhysicalChannel],
        *,
        parameterisation: str,
        phase: str,
        n_cores: int,
        backend: str,
        batch_size: int,
        q_bins: Sequence[float] = (),
        start_method: str = "spawn",
    ) -> None:
        self.channels = tuple(channels)
        self.parameterisation = parameterisation
        self.phase = phase
        self.backend = backend
        self.batch_size = batch_size
        self.q_bins = tuple(float(edge) for edge in q_bins)
        self.start_method = start_method
        self.allocation = _allocate_channel_workers(self.channels, n_cores)
        self.pools: dict[str, Any] = {}
        self.worker_pids: dict[str, set[int]] = {
            channel.label: set() for channel in channels
        }

    def __enter__(self) -> _PersistentChannelExecutors:
        if not any(self.allocation.values()):
            return self
        context = multiprocessing.get_context(self.start_method)
        try:
            for channel in self.channels:
                n_workers = self.allocation[channel.label]
                self.pools[channel.label] = context.Pool(
                    processes=n_workers,
                    initializer=_physical_worker_init,
                    initargs=(
                        channel.process.builder_inputs(),
                        dict(channel.integrand_implementation),
                        self.parameterisation,
                        self.phase,
                        tuple(channel.graph_channel_indices or ()),
                        self.backend,
                        self.batch_size,
                        channel.sampling_factor,
                        self.q_bins,
                    ),
                )
        except BaseException:
            self.close(terminate=True)
            raise
        return self

    def close(self, *, terminate: bool = False) -> None:
        for pool in self.pools.values():
            if terminate:
                pool.terminate()
            else:
                pool.close()
        for pool in self.pools.values():
            pool.join()
        self.pools.clear()

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        self.close(terminate=exc_type is not None)

    def evaluate(
        self,
        rows_by_channel: Mapping[str, Sequence[_PhysicalRow]],
    ) -> tuple[list[_EvaluatedRow], IntegrationResult]:
        all_values: list[_EvaluatedRow] = []
        diagnostics = IntegrationResult(0.0, 0.0)
        pending: list[tuple[str, Any, bool]] = []
        for channel in self.channels:
            rows = list(rows_by_channel.get(channel.label, ()))
            if not rows:
                continue
            n_workers = self.allocation[channel.label]
            if n_workers == 0:
                results = [
                    _evaluate_rows(
                        channel.process,
                        channel.integrand_implementation,
                        self.parameterisation,
                        self.phase,
                        tuple(channel.graph_channel_indices or ()),
                        self.backend,
                        self.batch_size,
                        channel.sampling_factor,
                        self.q_bins,
                        rows,
                    )
                ]
                pending.append((channel.label, results, False))
            else:
                chunk_size = max(
                    self.batch_size,
                    math.ceil(len(rows) / max(1, 4 * n_workers)),
                )
                tasks = [
                    rows[start : start + chunk_size]
                    for start in range(0, len(rows), chunk_size)
                ]
                pending.append(
                    (
                        channel.label,
                        self.pools[channel.label].map_async(
                            _physical_worker_task, tasks
                        ),
                        True,
                    )
                )
        for label, submitted, is_async in pending:
            results = submitted.get() if is_async else submitted
            for values, task_diagnostics, worker_pid in results:
                all_values.extend(values)
                diagnostics.combine_with(task_diagnostics)
                self.worker_pids[label].add(worker_pid)
        return all_values, diagnostics


class DYPhysicalChannelIntegrator:
    """Integrate a pp sum with joint physical and graph importance sampling."""

    def __init__(
        self,
        channels: Sequence[DYPhysicalChannel],
        *,
        parameterisation: str,
        phase: str = "real",
        n_cores: int = 1,
        evaluation_backend: str = "saved-jit",
        batch_size: int = 256,
        q_bins: Sequence[float] | None = None,
        min_channel_fraction: float = 0.0,
        live_iteration_diagnostics: bool = False,
    ) -> None:
        self.channels = tuple(channels)
        self.parameterisation = str(parameterisation)
        self.phase = str(phase)
        self.n_cores = int(n_cores)
        self.evaluation_backend = str(evaluation_backend)
        self.batch_size = int(batch_size)
        self.q_bins = tuple(float(edge) for edge in (q_bins or ()))
        self.min_channel_fraction = float(min_channel_fraction)
        self.live_iteration_diagnostics = bool(live_iteration_diagnostics)
        self._validate()
        self.joint_channel_paths = tuple(
            (physical_index, graph_index)
            for physical_index, channel in enumerate(self.channels)
            for graph_index in range(len(channel.graph_channel_names))
        )

    def _validate(self) -> None:
        if len(self.channels) < 2:
            raise pygloopException(
                "Physical-channel multi-channeling requires at least two channels."
            )
        labels = [channel.label for channel in self.channels]
        if len(set(labels)) != len(labels):
            raise pygloopException("Physical-channel labels must be unique.")
        if self.n_cores < 1:
            raise pygloopException("Physical-channel core count must be positive.")
        if self.batch_size < 1:
            raise pygloopException("Physical-channel batch size must be positive.")
        if self.evaluation_backend not in {"compiled", "saved-jit"}:
            raise pygloopException(
                "Physical-channel backend must be 'compiled' or 'saved-jit'."
            )
        if (
            not math.isfinite(self.min_channel_fraction)
            or self.min_channel_fraction < 0.0
            or self.min_channel_fraction * len(self.channels) > 1.0 + 1.0e-15
        ):
            raise pygloopException(
                "Physical-channel minimum fraction must be finite, nonnegative, "
                "and no larger than 1 / number of channels."
            )
        if self.q_bins:
            if len(self.q_bins) < 2 or any(
                not (math.isfinite(lower) and math.isfinite(upper))
                or lower < 0.0
                or lower >= upper
                for lower, upper in zip(self.q_bins, self.q_bins[1:])
            ):
                raise pygloopException(
                    "Physical-channel Q bins require at least two strictly "
                    "increasing nonnegative edges."
                )
            if len(self.q_bins) - 1 > self.batch_size:
                raise pygloopException(
                    "The number of physical-channel Q bins cannot exceed the "
                    "matrix batch size."
                )
            for channel in self.channels:
                if (
                    channel.process.process_name.lower() != "dy"
                    or not channel.process.integrate_beams
                    or str(
                        channel.integrand_implementation.get(
                            "dy_beam_parameterisation", "x1_x2"
                        )
                    )
                    != "x1_x2"
                ):
                    raise pygloopException(
                        "Physical-channel Q bins require beam-convoluted "
                        "Drell-Yan with the x1_x2 parameterisation."
                    )
                if (
                    channel.process.dy_msbar_scheme_counterterm
                    and self.q_bins[0] <= 0.0
                ):
                    raise pygloopException(
                        "Drell-Yan scheme counterterms require a positive "
                        "first Q-bin edge."
                    )
        dimensions = {channel.dimension for channel in self.channels}
        if len(dimensions) != 1:
            raise pygloopException(
                "Physical channels must use the same continuous dimension; got "
                + ", ".join(
                    f"{channel.label}={channel.dimension}"
                    for channel in self.channels
                )
            )

        reference = self.channels[0].process
        compatibility_attributes = (
            "process_name",
            "n_loops",
            "final_state",
            "m_top",
            "m_higgs",
            "helicities",
            "e_cm",
            "integrate_beams",
            "dy_z_bin",
            "dy_q_min",
            "dy_q_max",
            "dy_lambda_sq",
            "dy_mur_sq",
            "dy_observable_muv",
            "dy_pdf_set",
            "dy_pdf_member",
            "dy_muf_sq",
            "dy_physical_normalisation",
            "dy_integrated_leptonic_phase_space",
            "dy_coupling_normalisation_applied",
            "dy_coupling_normalisation_factor",
            "dy_coupling_normalisation_convention",
            "symmetrise_p1_p2",
        )
        for channel in self.channels[1:]:
            differences = [
                name
                for name in compatibility_attributes
                if getattr(reference, name, None)
                != getattr(channel.process, name, None)
            ]
            if channel.integrand_implementation.get(
                "dy_beam_parameterisation", "x1_x2"
            ) != self.channels[0].integrand_implementation.get(
                "dy_beam_parameterisation", "x1_x2"
            ):
                differences.append("dy_beam_parameterisation")
            if differences:
                raise pygloopException(
                    f"Physical channel {channel.label} is incompatible with "
                    f"{self.channels[0].label}: {', '.join(differences)}."
                )

    def build_integrator(self) -> NumericalIntegrator:
        # Symbolica currently leaves inner discrete probabilities uniform when
        # a discrete integrator is nested below another discrete integrator.
        # A single native discrete layer over joint leaves learns p(a, g),
        # which simultaneously learns the physical marginal p(a) and graph
        # conditional p(g | a).  Keep Symbolica's variance-sensitive default
        # (train_on_avg=False) and its native probability-ratio safeguard.
        return NumericalIntegrator.discrete(
            [
                NumericalIntegrator.continuous(
                    self.channels[physical_index].dimension
                )
                for physical_index, _graph_index in self.joint_channel_paths
            ],
            train_on_avg=False,
        )

    def build_hierarchical_integrators(
        self,
    ) -> tuple[NumericalIntegrator, tuple[NumericalIntegrator, ...]]:
        # Symbolica currently cannot train a discrete grid whose bins have no
        # subgrid.  A one-dimensional, one-bin terminal is exactly uniform and
        # therefore contributes unit weight while permitting native discrete
        # adaptation at the physical layer.
        outer = NumericalIntegrator.discrete(
            [
                NumericalIntegrator.continuous(1, n_bins=1)
                for _channel in self.channels
            ],
            train_on_avg=False,
        )
        inner = tuple(
            NumericalIntegrator.discrete(
                [
                    NumericalIntegrator.continuous(channel.dimension)
                    for _graph_name in channel.graph_channel_names
                ],
                train_on_avg=False,
            )
            for channel in self.channels
        )
        return outer, inner

    def _outer_channel_probabilities(
        self,
        integrator: NumericalIntegrator,
    ) -> tuple[float, ...]:
        inverse_probabilities = tuple(
            float(integrator.probe(Probe.discrete([channel_index])))
            for channel_index in range(len(self.channels))
        )
        if any(
            not math.isfinite(weight) or weight <= 0.0
            for weight in inverse_probabilities
        ):
            raise pygloopException(
                "Symbolica returned an invalid physical-channel probability."
            )
        probabilities = [1.0 / weight for weight in inverse_probabilities]
        normalisation = math.fsum(probabilities)
        if not math.isfinite(normalisation) or normalisation <= 0.0:
            raise pygloopException(
                "Symbolica physical-channel probabilities cannot be normalised."
            )
        return tuple(probability / normalisation for probability in probabilities)

    @staticmethod
    def _attach_coupling_normalisation_metadata(
        process: Any, result: IntegrationResult
    ) -> IntegrationResult:
        attach = getattr(
            process, "_attach_dy_coupling_normalisation_metadata", None
        )
        if callable(attach):
            return attach(result)
        result.dy_coupling_normalisation_applied = bool(
            getattr(process, "dy_coupling_normalisation_applied", False)
        )
        result.dy_coupling_normalisation_factor = float(
            getattr(process, "dy_coupling_normalisation_factor", 1.0)
        )
        result.dy_coupling_normalisation_convention = str(
            getattr(process, "dy_coupling_normalisation_convention", "none")
        )
        result.dy_coupling_normalisation_includes_unit_conversion = False
        return result

    def _correct_channel_result(
        self,
        channel: DYPhysicalChannel,
        hard_central: float,
        hard_error: float,
        *,
        seed: int,
        q_bin: tuple[float, float] | None = None,
        auxiliary_workers: int = 1,
    ) -> IntegrationResult:
        channel_result = IntegrationResult(
            hard_central,
            hard_error,
        )
        if not channel.process.dy_msbar_scheme_counterterm:
            return self._attach_coupling_normalisation_metadata(
                channel.process, channel_result
            )

        raw_result = IntegrationResult(
            hard_central / channel.sampling_factor,
            hard_error / abs(channel.sampling_factor),
        )
        if q_bin is not None:
            old_q_min = channel.process.dy_q_min
            old_q_max = channel.process.dy_q_max
            q_min = max(
                q_bin[0],
                old_q_min if old_q_min is not None else q_bin[0],
            )
            q_max = min(
                q_bin[1],
                old_q_max if old_q_max is not None else q_bin[1],
            )
            if q_min >= q_max:
                return self._attach_coupling_normalisation_metadata(
                    channel.process, channel_result
                )
            channel.process.dy_q_min = q_min
            channel.process.dy_q_max = q_max
        try:
            if channel.process._dy_ttbar_gg_scheme_mode:
                if auxiliary_workers == 1:
                    return channel.process._apply_ttbar_gg_scheme(
                        raw_result,
                        seed,
                        self.parameterisation,
                        self.phase,
                    )
                return channel.process._apply_ttbar_gg_scheme(
                    raw_result,
                    seed,
                    self.parameterisation,
                    self.phase,
                    auxiliary_workers,
                )
            if channel.process._dy_ttbar_qqbar_scheme_mode:
                if auxiliary_workers == 1:
                    return channel.process._apply_ttbar_qqbar_scheme_and_decoupling(
                        raw_result,
                        seed,
                        self.parameterisation,
                        self.phase,
                    )
                return channel.process._apply_ttbar_qqbar_scheme_and_decoupling(
                    raw_result,
                    seed,
                    self.parameterisation,
                    self.phase,
                    auxiliary_workers,
                )
            if auxiliary_workers == 1:
                return channel.process._add_dy_msbar_scheme_counterterm(
                    raw_result,
                    seed,
                    self.parameterisation,
                    self.phase,
                )
            return channel.process._add_dy_msbar_scheme_counterterm(
                raw_result,
                seed,
                self.parameterisation,
                self.phase,
                auxiliary_workers,
            )
        finally:
            if q_bin is not None:
                channel.process.dy_q_min = old_q_min
                channel.process.dy_q_max = old_q_max

    @staticmethod
    def _auxiliary_error(channel_result: IntegrationResult) -> float:
        return float(
            getattr(
                channel_result,
                "dy_auxiliary_error",
                getattr(channel_result, "dy_scheme_counterterm_error", 0.0),
            )
        )

    def _preview_auxiliary_channels(
        self, *, seed: int
    ) -> dict[str, dict[str, float]]:
        """Evaluate fixed corrections and hard factors before production.

        The normal finalisation path is deliberately retained after production
        to preserve full diagnostics and independently verify this preview.
        """

        preview: dict[str, dict[str, float]] = {}
        for channel_index, channel in enumerate(self.channels):
            correction_seed = int(seed) + 1009 * (channel_index + 1)
            channel_central = 0.0
            channel_error_sq = 0.0
            hard_factor: float | None = None
            if self.q_bins:
                for bin_index, (q_min, q_max) in enumerate(
                    zip(self.q_bins, self.q_bins[1:], strict=True)
                ):
                    bin_result = self._correct_channel_result(
                        channel,
                        0.0,
                        0.0,
                        seed=correction_seed + 104729 * (bin_index + 1),
                        q_bin=(q_min, q_max),
                        auxiliary_workers=self.n_cores,
                    )
                    bin_hard_factor = float(
                        getattr(bin_result, "dy_hard_factor", 1.0)
                    )
                    if hard_factor is None:
                        hard_factor = bin_hard_factor
                    elif hard_factor != bin_hard_factor:
                        raise pygloopException(
                            f"Physical hard factor changed across Q bins for "
                            f"{channel.label}."
                        )
                    channel_central += bin_result.central_value
                    channel_error_sq += self._auxiliary_error(bin_result) ** 2
            else:
                channel_result = self._correct_channel_result(
                    channel,
                    0.0,
                    0.0,
                    seed=correction_seed,
                    auxiliary_workers=self.n_cores,
                )
                channel_central = channel_result.central_value
                channel_error_sq = self._auxiliary_error(channel_result) ** 2
                hard_factor = float(
                    getattr(channel_result, "dy_hard_factor", 1.0)
                )
            preview[channel.label] = {
                "central_value": channel_central,
                "error": math.sqrt(channel_error_sq),
                "hard_factor": 1.0 if hard_factor is None else hard_factor,
            }
        return preview

    def _preview_auxiliary_total(self, *, seed: int) -> tuple[float, float]:
        preview = self._preview_auxiliary_channels(seed=seed)
        return (
            math.fsum(row["central_value"] for row in preview.values()),
            math.sqrt(
                math.fsum(row["error"] ** 2 for row in preview.values())
            ),
        )

    def _physical_iteration_estimate(
        self,
        channel_moments: Mapping[str, _WeightedMoments],
        auxiliary_preview: Mapping[str, Mapping[str, float]],
    ) -> dict[str, Any]:
        covariance = self._channel_covariance(channel_moments)
        factors = {
            channel.label: float(
                auxiliary_preview.get(channel.label, {}).get("hard_factor", 1.0)
            )
            for channel in self.channels
        }
        hard_variance = math.fsum(
            factors[first] * factors[second] * value
            for first, row in covariance.items()
            for second, value in row.items()
        )
        channels: dict[str, dict[str, float]] = {}
        for channel in self.channels:
            label = channel.label
            raw_central, raw_error = channel_moments[label].estimate()
            auxiliary = auxiliary_preview.get(label, {})
            auxiliary_central = float(auxiliary.get("central_value", 0.0))
            auxiliary_error = float(auxiliary.get("error", 0.0))
            physical_hard_central = factors[label] * raw_central
            channels[label] = {
                "raw_hard_central_value": raw_central,
                "raw_hard_error": raw_error,
                "hard_factor": factors[label],
                "hard_central_value": physical_hard_central,
                "hard_error": abs(factors[label]) * raw_error,
                "auxiliary_central_value": auxiliary_central,
                "auxiliary_error": auxiliary_error,
                "central_value": physical_hard_central + auxiliary_central,
                "error": math.hypot(
                    abs(factors[label]) * raw_error, auxiliary_error
                ),
            }
        hard_central = math.fsum(
            row["hard_central_value"] for row in channels.values()
        )
        auxiliary_central = math.fsum(
            row["auxiliary_central_value"] for row in channels.values()
        )
        hard_error = math.sqrt(max(0.0, hard_variance))
        auxiliary_error = math.sqrt(
            math.fsum(
                row["auxiliary_error"] ** 2 for row in channels.values()
            )
        )
        full_central = hard_central + auxiliary_central
        full_error = math.hypot(hard_error, auxiliary_error)
        return {
            "hard_central_value": hard_central,
            "hard_error": hard_error,
            "auxiliary_central_value": auxiliary_central,
            "auxiliary_error": auxiliary_error,
            "central_value": full_central,
            "error": full_error,
            "relative_error": (
                abs(full_error / full_central) if full_central else None
            ),
            "channels": channels,
        }

    def _record_live_iteration_diagnostics(
        self,
        result: DYPhysicalChannelIntegrationResult,
        *,
        iteration: int,
        batch_diagnostics: IntegrationResult,
        estimate: Mapping[str, Any],
        selected_counts: Mapping[str, int],
        effective_probabilities: Mapping[str, float],
    ) -> None:
        cumulative = {
            name: int(getattr(result, name, 0))
            for name in (*_LIVE_DIAGNOSTIC_COUNTERS, *[k for k in vars(result) if k.startswith("stability_precision_") and k.endswith("_count")])
        }
        delta = {
            name: int(getattr(batch_diagnostics, name, 0))
            for name in (*_LIVE_DIAGNOSTIC_COUNTERS, *[k for k in vars(batch_diagnostics) if k.startswith("stability_precision_") and k.endswith("_count")])
        }
        denominator = int(result.n_samples)
        payload = {
            "schema_version": 1,
            "precision_ladders": {ch.label: ch.integrand_implementation.get("dy_precision_ladder") for ch in self.channels},
            "iteration": int(iteration),
            "production_samples": int(result.production_samples),
            "evaluated_samples": denominator,
            "estimate": dict(estimate),
            "sampling": {
                "selected_samples": {
                    label: int(count) for label, count in selected_counts.items()
                },
                "effective_probabilities": {
                    label: float(probability)
                    for label, probability in effective_probabilities.items()
                },
            },
            "counters": {
                "iteration": delta,
                "cumulative": cumulative,
                "cumulative_clipped_fraction": (
                    cumulative["large_weight_zeroed_count"] / denominator
                    if denominator
                    else 0.0
                ),
            },
        }
        result.iteration_diagnostics.append(payload)
        logger.info(
            "DY_PHYSICAL_CHANNEL_ITERATION %s",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )

    @staticmethod
    def _relative_precision(central_value: float, error: float) -> float:
        if (
            not math.isfinite(central_value)
            or not math.isfinite(error)
            or central_value == 0.0
            or error < 0.0
        ):
            return math.inf
        return abs(error / central_value)

    def _record_precision_checkpoint(
        self,
        result: DYPhysicalChannelIntegrationResult,
        *,
        iteration: int,
        auxiliary_central: float,
        auxiliary_error: float,
        physical_estimate: Mapping[str, Any] | None = None,
    ) -> bool:
        if physical_estimate is None:
            hard_central = result.central_value
            hard_error = result.error
            full_central = hard_central + auxiliary_central
            full_error = math.hypot(hard_error, auxiliary_error)
        else:
            hard_central = float(physical_estimate["hard_central_value"])
            hard_error = float(physical_estimate["hard_error"])
            full_central = float(physical_estimate["central_value"])
            full_error = float(physical_estimate["error"])
        relative_precision = self._relative_precision(full_central, full_error)
        result.production_iterations = int(iteration)
        result.achieved_relative_precision = relative_precision
        result.precision_history.append({
            "iteration": int(iteration),
            "production_samples": result.production_samples,
            "hard_central_value": hard_central,
            "hard_error": hard_error,
            "full_central_value": full_central,
            "full_error": full_error,
            "relative_precision": relative_precision,
        })
        logger.info(
            "Physical-channel corrected PP precision after iteration %d: "
            "%+.16e +/- %.4e (relative %.6f%%; target %.6f%%)",
            iteration,
            full_central,
            full_error,
            100.0 * relative_precision,
            100.0 * float(result.target_relative_precision),
        )
        if relative_precision <= float(result.target_relative_precision):
            result.precision_target_met = True
            result.precision_stop_reason = "relative_precision"
            return True
        return False

    def _finalise_precision_metadata(
        self,
        result: DYPhysicalChannelIntegrationResult,
    ) -> None:
        result.achieved_relative_precision = self._relative_precision(
            result.central_value, result.error
        )
        if result.target_relative_precision is None:
            result.precision_target_met = None
            result.precision_stop_reason = "fixed_iterations"
            return
        result.precision_target_met = (
            result.achieved_relative_precision
            <= result.target_relative_precision
        )
        if result.precision_target_met:
            result.precision_stop_reason = "relative_precision"
        elif result.precision_stop_reason == "relative_precision":
            result.precision_stop_reason = "final_verification_failed"
        else:
            result.precision_stop_reason = "max_iterations"

    @staticmethod
    def _channel_covariance(
        channel_moments: Mapping[str, _WeightedMoments],
    ) -> dict[str, dict[str, float]]:
        """Covariance of the one-hot physical-channel mean estimators."""

        labels = tuple(channel_moments)
        sample_counts = {moments.n for moments in channel_moments.values()}
        if len(sample_counts) != 1:
            raise pygloopException(
                "Physical-channel observers have inconsistent sample counts."
            )
        n_samples = sample_counts.pop() if sample_counts else 0
        if n_samples <= 1:
            return {
                first: {second: 0.0 for second in labels}
                for first in labels
            }

        denominator = n_samples * (n_samples - 1)
        covariance: dict[str, dict[str, float]] = {}
        for first in labels:
            first_moments = channel_moments[first]
            covariance[first] = {}
            for second in labels:
                second_moments = channel_moments[second]
                cross_sum = (
                    first_moments.total_sq if first == second else 0.0
                )
                centered_cross_sum = cross_sum - (
                    first_moments.total
                    * second_moments.total
                    / n_samples
                )
                covariance[first][second] = centered_cross_sum / denominator
        return covariance

    def _apply_auxiliary_corrections(
        self,
        result: DYPhysicalChannelIntegrationResult,
        channel_moments: Mapping[str, _WeightedMoments],
        channel_bin_moments: Mapping[str, Sequence[_WeightedMoments]],
        q_bin_moments: Sequence[_WeightedMoments],
        *,
        seed: int,
    ) -> None:
        result.hard_central_value = result.central_value
        result.hard_error = result.error
        result.hard_channel_covariance = self._channel_covariance(
            channel_moments
        )
        auxiliary_error_sq = 0.0
        bin_auxiliary_central = [0.0 for _moment in q_bin_moments]
        bin_auxiliary_error_sq = [0.0 for _moment in q_bin_moments]
        for channel_index, channel in enumerate(self.channels):
            hard_central, hard_error = channel_moments[channel.label].estimate()
            correction_seed = int(seed) + 1009 * (channel_index + 1)
            channel_bins: list[dict[str, Any]] = []
            if self.q_bins:
                channel_auxiliary_central = 0.0
                channel_auxiliary_error_sq = 0.0
                for bin_index, moments in enumerate(
                    channel_bin_moments[channel.label]
                ):
                    bin_hard_central, bin_hard_error = moments.estimate()
                    bin_result = self._correct_channel_result(
                        channel,
                        bin_hard_central,
                        bin_hard_error,
                        seed=correction_seed + 104729 * (bin_index + 1),
                        q_bin=(
                            self.q_bins[bin_index],
                            self.q_bins[bin_index + 1],
                        ),
                        auxiliary_workers=self.n_cores,
                    )
                    bin_auxiliary = (
                        bin_result.central_value - bin_hard_central
                    )
                    bin_auxiliary_error = self._auxiliary_error(bin_result)
                    channel_auxiliary_central += bin_auxiliary
                    channel_auxiliary_error_sq += bin_auxiliary_error**2
                    bin_auxiliary_central[bin_index] += bin_auxiliary
                    bin_auxiliary_error_sq[bin_index] += bin_auxiliary_error**2
                    channel_bins.append({
                        "q_min": self.q_bins[bin_index],
                        "q_max": self.q_bins[bin_index + 1],
                        "central_value": bin_result.central_value,
                        "error": bin_result.error,
                        "hard_central_value": bin_hard_central,
                        "hard_error": bin_hard_error,
                        "auxiliary": bin_result,
                    })
                channel_result = IntegrationResult(
                    hard_central + channel_auxiliary_central,
                    math.hypot(
                        hard_error, math.sqrt(channel_auxiliary_error_sq)
                    ),
                    n_samples=result.production_samples,
                )
                auxiliary_central = channel_auxiliary_central
                auxiliary_error = math.sqrt(channel_auxiliary_error_sq)
            else:
                channel_result = self._correct_channel_result(
                    channel,
                    hard_central,
                    hard_error,
                    seed=correction_seed,
                    auxiliary_workers=self.n_cores,
                )
                channel_result.n_samples = result.production_samples
                auxiliary_central = channel_result.central_value - hard_central
                auxiliary_error = self._auxiliary_error(channel_result)

            result.central_value += auxiliary_central
            auxiliary_error_sq += auxiliary_error * auxiliary_error

            result.physical_channel_estimates[channel.label] = {
                "central_value": channel_result.central_value,
                "error": channel_result.error,
                "hard_central_value": hard_central,
                "hard_error": hard_error,
                "selected_samples": channel_moments[channel.label].n,
                "sampling_factor": channel.sampling_factor,
                "pdf_luminosity_family": getattr(
                    channel.process, "dy_pdf_luminosity_family", None
                ),
                "auxiliary_error": auxiliary_error,
                "source_card": channel.source_card,
                "auxiliary": channel_result,
                "q_bins": channel_bins,
            }
        result.physical_channel_covariance = {
            first: dict(row)
            for first, row in result.hard_channel_covariance.items()
        }
        for channel in self.channels:
            result.physical_channel_covariance[channel.label][channel.label] += (
                float(
                    result.physical_channel_estimates[channel.label][
                        "auxiliary_error"
                    ]
                )
                ** 2
            )
        result.error = math.hypot(result.hard_error, math.sqrt(auxiliary_error_sq))
        for bin_index, moments in enumerate(q_bin_moments):
            hard_central, hard_error = moments.estimate()
            auxiliary_error = math.sqrt(bin_auxiliary_error_sq[bin_index])
            result.q_bin_estimates.append({
                "q_min": self.q_bins[bin_index],
                "q_max": self.q_bins[bin_index + 1],
                "central_value": hard_central
                + bin_auxiliary_central[bin_index],
                "error": math.hypot(hard_error, auxiliary_error),
                "hard_central_value": hard_central,
                "hard_error": hard_error,
                "auxiliary_central_value": bin_auxiliary_central[bin_index],
                "auxiliary_error": auxiliary_error,
            })

    def _integrate_hierarchical(
        self,
        *,
        n_iterations: int,
        points_per_iteration: int,
        seed: int,
        warmup_iterations: int,
        target_relative_precision: float | None,
    ) -> DYPhysicalChannelIntegrationResult:
        n_channels = len(self.channels)
        requested_floor_points = self.min_channel_fraction * points_per_iteration
        floor_points = int(
            math.floor(math.nextafter(requested_floor_points, math.inf))
        )
        if floor_points < 1:
            raise pygloopException(
                "The physical-channel minimum fraction must reserve at least "
                "one point per channel in every iteration."
            )
        total_floor_points = n_channels * floor_points
        if total_floor_points > points_per_iteration:
            raise pygloopException(
                "The rounded physical-channel floors exceed the points per "
                "iteration."
            )
        adaptive_points = points_per_iteration - total_floor_points
        floor_fraction = floor_points / points_per_iteration
        adaptive_fraction = adaptive_points / points_per_iteration

        outer_integrator, inner_integrators = (
            self.build_hierarchical_integrators()
        )
        outer_rng = outer_integrator.rng(seed=int(seed), stream_id=0)
        inner_rngs = tuple(
            integrator.rng(seed=int(seed), stream_id=channel_index + 1)
            for channel_index, integrator in enumerate(inner_integrators)
        )

        result = DYPhysicalChannelIntegrationResult()
        result.precision_ladders = {ch.label: ch.integrand_implementation.get("dy_precision_ladder") for ch in self.channels}
        result.precision_failure_action = "zero_complete_sample"
        result.evaluation_backend = self.evaluation_backend
        result.batch_size = self.batch_size
        result.outer_discrete_channel_count = n_channels
        result.joint_discrete_channel_count = len(self.joint_channel_paths)
        result.q_bin_edges = self.q_bins
        result.warmup_iterations = int(warmup_iterations)
        result.target_relative_precision = target_relative_precision
        result.physical_channel_min_fraction = self.min_channel_fraction
        result.sampling_topology = "physical-floor-mixture"
        result.physical_channel_floor_samples = {
            channel.label: 0 for channel in self.channels
        }
        result.physical_channel_adaptive_samples = {
            channel.label: 0 for channel in self.channels
        }

        total_moments = _WeightedMoments()
        channel_moments = {
            channel.label: _WeightedMoments() for channel in self.channels
        }
        graph_moments = {
            channel.label: {
                graph_name: _WeightedMoments()
                for graph_name in channel.graph_channel_names
            }
            for channel in self.channels
        }
        q_bin_moments = [
            _WeightedMoments() for _edge in self.q_bins[:-1]
        ]
        channel_bin_moments = {
            channel.label: [
                _WeightedMoments() for _edge in self.q_bins[:-1]
            ]
            for channel in self.channels
        }
        selected_counts = {channel.label: 0 for channel in self.channels}
        graph_selected_counts = {
            channel.label: {
                graph_name: 0 for graph_name in channel.graph_channel_names
            }
            for channel in self.channels
        }
        effective_probability_sums = {
            channel.label: 0.0 for channel in self.channels
        }

        auxiliary_preview_channels: dict[str, dict[str, float]] = {}
        auxiliary_preview = (0.0, 0.0)
        if (
            target_relative_precision is not None
            or self.live_iteration_diagnostics
        ):
            logger.info(
                "Evaluating auxiliary PP contribution once for the cumulative "
                "physical iteration report."
            )
            auxiliary_preview_channels = self._preview_auxiliary_channels(
                seed=int(seed)
            )
            auxiliary_preview = (
                math.fsum(
                    row["central_value"]
                    for row in auxiliary_preview_channels.values()
                ),
                math.sqrt(
                    math.fsum(
                        row["error"] ** 2
                        for row in auxiliary_preview_channels.values()
                    )
                ),
            )
            (
                result.preview_auxiliary_central_value,
                result.preview_auxiliary_error,
            ) = auxiliary_preview
            logger.info(
                "Auxiliary PP precision preview: %+.16e +/- %.4e",
                auxiliary_preview[0],
                auxiliary_preview[1],
            )

        with _PersistentChannelExecutors(
            self.channels,
            parameterisation=self.parameterisation,
            phase=self.phase,
            n_cores=self.n_cores,
            backend=self.evaluation_backend,
            batch_size=self.batch_size,
            q_bins=self.q_bins,
        ) as executors:
            result.worker_allocation = dict(executors.allocation)

            def evaluate_batch(
                active_outer: NumericalIntegrator,
                active_inner: tuple[NumericalIntegrator, ...],
                active_outer_rng: Any,
                active_inner_rngs: tuple[Any, ...],
                *,
                retain_outer_samples: bool,
            ) -> _HierarchicalBatch:
                adaptive_probabilities = self._outer_channel_probabilities(
                    active_outer
                )
                sampled_outer = (
                    list(active_outer.sample(adaptive_points, active_outer_rng))
                    if adaptive_points
                    else []
                )
                adaptive_counts = [0 for _channel in self.channels]
                outer_global_indices: list[int] = []
                positions_by_channel: list[list[int]] = [
                    [] for _channel in self.channels
                ]
                next_global_index = 0
                for physical_index in range(n_channels):
                    positions_by_channel[physical_index].extend(
                        range(
                            next_global_index,
                            next_global_index + floor_points,
                        )
                    )
                    next_global_index += floor_points
                for sample in sampled_outer:
                    if len(sample.d) != 1:
                        raise pygloopException(
                            "The outer Symbolica sample must contain exactly "
                            "one physical-channel index."
                        )
                    physical_index = int(sample.d[0])
                    if physical_index < 0 or physical_index >= n_channels:
                        raise pygloopException(
                            f"Physical channel index {physical_index} is out of range."
                        )
                    if retain_outer_samples:
                        outer_global_indices.append(next_global_index)
                    positions_by_channel[physical_index].append(next_global_index)
                    next_global_index += 1
                    adaptive_counts[physical_index] += 1
                if next_global_index != points_per_iteration:
                    raise pygloopException(
                        "Physical-channel mixture produced the wrong sample count."
                    )

                effective_probabilities = tuple(
                    floor_fraction
                    + adaptive_fraction * adaptive_probability
                    for adaptive_probability in adaptive_probabilities
                )
                if any(
                    not math.isfinite(probability) or probability <= 0.0
                    for probability in effective_probabilities
                ) or not math.isclose(
                    math.fsum(effective_probabilities),
                    1.0,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ):
                    raise pygloopException(
                        "The physical-channel mixture probabilities are invalid."
                    )

                pending_inner_samples: list[Sample | None] = [
                    None for _point in range(points_per_iteration)
                ]
                pending_paths: list[tuple[int, int] | None] = [
                    None for _point in range(points_per_iteration)
                ]
                rows_by_channel: dict[str, list[_PhysicalRow]] = {
                    channel.label: [] for channel in self.channels
                }
                for physical_index, positions in enumerate(positions_by_channel):
                    samples = list(
                        active_inner[physical_index].sample(
                            len(positions), active_inner_rngs[physical_index]
                        )
                    )
                    channel = self.channels[physical_index]
                    for global_index, sample in zip(
                        positions, samples, strict=True
                    ):
                        if len(sample.d) != 1:
                            raise pygloopException(
                                "An inner Symbolica sample must contain exactly "
                                "one graph-channel index."
                            )
                        graph_index = int(sample.d[0])
                        if graph_index < 0 or graph_index >= len(
                            channel.graph_channel_names
                        ):
                            raise pygloopException(
                                f"Graph channel index {graph_index} is out of "
                                f"range for {channel.label}."
                            )
                        pending_inner_samples[global_index] = sample
                        pending_paths[global_index] = (
                            physical_index,
                            graph_index,
                        )
                        rows_by_channel[channel.label].append(
                            (global_index, list(sample.c), graph_index)
                        )
                if any(sample is None for sample in pending_inner_samples) or any(
                    path is None for path in pending_paths
                ):
                    raise pygloopException(
                        "Hierarchical Symbolica sampling left an incomplete path."
                    )
                inner_samples = [
                    sample
                    for sample in pending_inner_samples
                    if sample is not None
                ]
                sample_paths = [
                    path for path in pending_paths if path is not None
                ]
                outer_samples = sampled_outer if retain_outer_samples else []
                if not retain_outer_samples:
                    del sampled_outer

                indexed_values, diagnostics = executors.evaluate(rows_by_channel)
                values: list[float | None] = [None] * points_per_iteration
                bin_values: list[tuple[float, ...] | None] = [
                    None
                ] * points_per_iteration
                for global_index, value, bins in indexed_values:
                    if values[global_index] is not None:
                        raise pygloopException(
                            f"Physical-channel sample {global_index} was "
                            "evaluated twice."
                        )
                    if len(bins) != len(q_bin_moments):
                        raise pygloopException(
                            "Physical-channel Q-bin evaluation returned the wrong "
                            "number of bins."
                        )
                    values[global_index] = value
                    bin_values[global_index] = bins
                if any(value is None for value in values) or any(
                    bins is None for bins in bin_values
                ):
                    raise pygloopException(
                        "Physical-channel evaluation left samples unevaluated."
                    )
                return _HierarchicalBatch(
                    outer_samples=outer_samples,
                    outer_global_indices=outer_global_indices,
                    inner_samples=inner_samples,
                    sample_paths=sample_paths,
                    positions_by_channel=tuple(positions_by_channel),
                    raw_values=[
                        float(value) for value in values if value is not None
                    ],
                    bin_values=[
                        bins for bins in bin_values if bins is not None
                    ],
                    diagnostics=diagnostics,
                    adaptive_probabilities=adaptive_probabilities,
                    effective_probabilities=effective_probabilities,
                    floor_counts=tuple(floor_points for _channel in self.channels),
                    adaptive_counts=tuple(adaptive_counts),
                )

            def train_batch(
                batch: _HierarchicalBatch,
                active_outer: NumericalIntegrator,
                active_inner: tuple[NumericalIntegrator, ...],
            ) -> None:
                for physical_index, integrator in enumerate(active_inner):
                    positions = batch.positions_by_channel[physical_index]
                    integrator.add_training_samples(
                        [batch.inner_samples[index] for index in positions],
                        [batch.raw_values[index] for index in positions],
                    )
                    integrator.update(
                        continuous_learning_rate=1.0,
                        discrete_learning_rate=1.0,
                    )
                if batch.outer_samples:
                    outer_values = [
                        batch.raw_values[global_index]
                        * DY._symbolica_sample_weight(
                            batch.inner_samples[global_index]
                        )
                        for global_index in batch.outer_global_indices
                    ]
                    active_outer.add_training_samples(
                        batch.outer_samples, outer_values
                    )
                    active_outer.update(
                        continuous_learning_rate=1.0,
                        discrete_learning_rate=1.0,
                    )

            def record_probability_history(
                stage: str,
                iteration: int,
                batch: _HierarchicalBatch,
            ) -> None:
                result.physical_channel_probability_history.append({
                    "stage": stage,
                    "iteration": iteration,
                    "floor_fraction": floor_fraction,
                    "adaptive_fraction": adaptive_fraction,
                    "adaptive_probabilities": {
                        channel.label: batch.adaptive_probabilities[index]
                        for index, channel in enumerate(self.channels)
                    },
                    "effective_probabilities": {
                        channel.label: batch.effective_probabilities[index]
                        for index, channel in enumerate(self.channels)
                    },
                    "floor_samples": {
                        channel.label: batch.floor_counts[index]
                        for index, channel in enumerate(self.channels)
                    },
                    "adaptive_samples": {
                        channel.label: batch.adaptive_counts[index]
                        for index, channel in enumerate(self.channels)
                    },
                })

            for iteration in range(warmup_iterations):
                logger.info(
                    "Hierarchical physical-channel Symbolica warmup: iteration "
                    "%d/%d with %d points (%.3f floor per channel)",
                    iteration + 1,
                    warmup_iterations,
                    points_per_iteration,
                    floor_fraction,
                )
                batch = evaluate_batch(
                    outer_integrator,
                    inner_integrators,
                    outer_rng,
                    inner_rngs,
                    retain_outer_samples=True,
                )
                train_batch(batch, outer_integrator, inner_integrators)
                record_probability_history("warmup", iteration + 1, batch)
                result.combine_with(batch.diagnostics)
                result.warmup_samples += len(batch.inner_samples)

            frozen_production = warmup_iterations > 0
            if frozen_production:
                production_outer = NumericalIntegrator.import_grid(
                    outer_integrator.export_grid(False)
                )
                production_inner = tuple(
                    NumericalIntegrator.import_grid(integrator.export_grid(False))
                    for integrator in inner_integrators
                )
                production_outer_rng = production_outer.rng(
                    seed=int(seed), stream_id=n_channels + 1
                )
                production_inner_rngs = tuple(
                    integrator.rng(
                        seed=int(seed), stream_id=n_channels + 2 + channel_index
                    )
                    for channel_index, integrator in enumerate(production_inner)
                )
            else:
                production_outer = outer_integrator
                production_inner = inner_integrators
                production_outer_rng = outer_rng
                production_inner_rngs = inner_rngs

            for iteration in range(n_iterations):
                logger.info(
                    "Hierarchical physical-channel Symbolica production: "
                    "iteration %d/%d with %d points (%.3f floor per channel)",
                    iteration + 1,
                    n_iterations,
                    points_per_iteration,
                    floor_fraction,
                )
                batch = evaluate_batch(
                    production_outer,
                    production_inner,
                    production_outer_rng,
                    production_inner_rngs,
                    retain_outer_samples=not frozen_production,
                )
                if not frozen_production:
                    train_batch(batch, production_outer, production_inner)
                record_probability_history("production", iteration + 1, batch)
                result.combine_with(batch.diagnostics)
                result.production_samples += len(batch.inner_samples)

                total_moments.start_batch(len(batch.inner_samples))
                for moment in channel_moments.values():
                    moment.start_batch(len(batch.inner_samples))
                for rows in graph_moments.values():
                    for moment in rows.values():
                        moment.start_batch(len(batch.inner_samples))
                for moment in q_bin_moments:
                    moment.start_batch(len(batch.inner_samples))
                for rows in channel_bin_moments.values():
                    for moment in rows:
                        moment.start_batch(len(batch.inner_samples))

                for physical_index, channel in enumerate(self.channels):
                    label = channel.label
                    result.physical_channel_floor_samples[label] += (
                        batch.floor_counts[physical_index]
                    )
                    result.physical_channel_adaptive_samples[label] += (
                        batch.adaptive_counts[physical_index]
                    )
                    effective_probability_sums[label] += (
                        batch.effective_probabilities[physical_index]
                        * len(batch.inner_samples)
                    )
                for sample, value, bins, (physical_index, graph_index) in zip(
                    batch.inner_samples,
                    batch.raw_values,
                    batch.bin_values,
                    batch.sample_paths,
                    strict=True,
                ):
                    channel = self.channels[physical_index]
                    sample_weight = (
                        DY._symbolica_sample_weight(sample)
                        / batch.effective_probabilities[physical_index]
                    )
                    weighted_value = value * sample_weight
                    total_moments.add_selected(weighted_value)
                    channel_moments[channel.label].add_selected(weighted_value)
                    graph_name = channel.graph_channel_names[graph_index]
                    graph_moments[channel.label][graph_name].add_selected(
                        weighted_value
                    )
                    graph_selected_counts[channel.label][graph_name] += 1
                    for bin_index, bin_value in enumerate(bins):
                        weighted_bin_value = bin_value * sample_weight
                        q_bin_moments[bin_index].add_selected(weighted_bin_value)
                        channel_bin_moments[channel.label][
                            bin_index
                        ].add_selected(weighted_bin_value)
                    selected_counts[channel.label] += 1

                result.central_value, result.error = total_moments.estimate()
                logger.info(
                    "Physical-channel hard total after iteration %d: "
                    "%+.16e +/- %.4e",
                    iteration + 1,
                    result.central_value,
                    result.error,
                )
                result.production_iterations = iteration + 1
                physical_estimate = None
                if (
                    target_relative_precision is not None
                    or self.live_iteration_diagnostics
                ):
                    physical_estimate = self._physical_iteration_estimate(
                        channel_moments, auxiliary_preview_channels
                    )
                if self.live_iteration_diagnostics:
                    self._record_live_iteration_diagnostics(
                        result,
                        iteration=iteration + 1,
                        batch_diagnostics=batch.diagnostics,
                        estimate=physical_estimate or {},
                        selected_counts=selected_counts,
                        effective_probabilities={
                            channel.label: batch.effective_probabilities[index]
                            for index, channel in enumerate(self.channels)
                        },
                    )
                if (
                    target_relative_precision is not None
                    and self._record_precision_checkpoint(
                        result,
                        iteration=iteration + 1,
                        auxiliary_central=auxiliary_preview[0],
                        auxiliary_error=auxiliary_preview[1],
                        physical_estimate=physical_estimate,
                    )
                ):
                    break

            result.worker_pids = {
                label: tuple(sorted(pids))
                for label, pids in executors.worker_pids.items()
            }

        result.physical_channel_effective_probabilities = {
            channel.label: (
                effective_probability_sums[channel.label]
                / result.production_samples
            )
            for channel in self.channels
        }
        self._apply_auxiliary_corrections(
            result,
            channel_moments,
            channel_bin_moments,
            q_bin_moments,
            seed=int(seed),
        )
        self._finalise_precision_metadata(result)
        self._attach_coupling_normalisation_metadata(
            self.channels[0].process, result
        )
        for channel in self.channels:
            label = channel.label
            result.physical_channel_estimates[label]["selected_samples"] = (
                selected_counts[label]
            )
            result.physical_channel_estimates[label]["floor_samples"] = (
                result.physical_channel_floor_samples[label]
            )
            result.physical_channel_estimates[label]["adaptive_samples"] = (
                result.physical_channel_adaptive_samples[label]
            )
            result.physical_channel_estimates[label][
                "effective_sampling_probability"
            ] = result.physical_channel_effective_probabilities[label]
            result.graph_channel_estimates[label] = {}
            for graph_name, moments in graph_moments[label].items():
                central, error = moments.estimate()
                result.graph_channel_estimates[label][graph_name] = {
                    "central_value": central,
                    "error": error,
                    "selected_samples": graph_selected_counts[label][graph_name],
                }
        return result

    def integrate(
        self,
        *,
        n_iterations: int,
        points_per_iteration: int,
        seed: int,
        target: float | None = None,
        warmup_iterations: int = 0,
        target_relative_precision: float | None = None,
    ) -> DYPhysicalChannelIntegrationResult:
        if (
            n_iterations < 1
            or points_per_iteration < 1
            or warmup_iterations < 0
        ):
            raise pygloopException(
                "Physical-channel production iterations and points must be "
                "positive, and warmup iterations must be nonnegative."
            )
        if target_relative_precision is not None and (
            not math.isfinite(target_relative_precision)
            or target_relative_precision <= 0.0
        ):
            raise pygloopException(
                "Physical-channel target relative precision must be finite "
                "and positive."
            )
        if self.min_channel_fraction > 0.0:
            return self._integrate_hierarchical(
                n_iterations=n_iterations,
                points_per_iteration=points_per_iteration,
                seed=seed,
                warmup_iterations=warmup_iterations,
                target_relative_precision=target_relative_precision,
            )
        integrator = self.build_integrator()
        rng = integrator.rng(seed=int(seed), stream_id=0)
        result = DYPhysicalChannelIntegrationResult()
        result.precision_ladders = {ch.label: ch.integrand_implementation.get("dy_precision_ladder") for ch in self.channels}
        result.precision_failure_action = "zero_complete_sample"
        result.evaluation_backend = self.evaluation_backend
        result.batch_size = self.batch_size
        result.outer_discrete_channel_count = len(self.channels)
        result.joint_discrete_channel_count = len(self.joint_channel_paths)
        result.q_bin_edges = self.q_bins
        result.warmup_iterations = int(warmup_iterations)
        result.target_relative_precision = target_relative_precision

        total_moments = _WeightedMoments()
        channel_moments = {
            channel.label: _WeightedMoments() for channel in self.channels
        }
        graph_moments = {
            channel.label: {
                graph_name: _WeightedMoments()
                for graph_name in channel.graph_channel_names
            }
            for channel in self.channels
        }
        q_bin_moments = [
            _WeightedMoments() for _edge in self.q_bins[:-1]
        ]
        channel_bin_moments = {
            channel.label: [
                _WeightedMoments() for _edge in self.q_bins[:-1]
            ]
            for channel in self.channels
        }
        selected_counts = {channel.label: 0 for channel in self.channels}
        graph_selected_counts = {
            channel.label: {
                graph_name: 0 for graph_name in channel.graph_channel_names
            }
            for channel in self.channels
        }

        auxiliary_preview_channels: dict[str, dict[str, float]] = {}
        auxiliary_preview = (0.0, 0.0)
        if (
            target_relative_precision is not None
            or self.live_iteration_diagnostics
        ):
            logger.info(
                "Evaluating auxiliary PP contribution once for the cumulative "
                "physical iteration report."
            )
            auxiliary_preview_channels = self._preview_auxiliary_channels(
                seed=int(seed)
            )
            auxiliary_preview = (
                math.fsum(
                    row["central_value"]
                    for row in auxiliary_preview_channels.values()
                ),
                math.sqrt(
                    math.fsum(
                        row["error"] ** 2
                        for row in auxiliary_preview_channels.values()
                    )
                ),
            )
            (
                result.preview_auxiliary_central_value,
                result.preview_auxiliary_error,
            ) = auxiliary_preview
            logger.info(
                "Auxiliary PP precision preview: %+.16e +/- %.4e",
                auxiliary_preview[0],
                auxiliary_preview[1],
            )

        with _PersistentChannelExecutors(
            self.channels,
            parameterisation=self.parameterisation,
            phase=self.phase,
            n_cores=self.n_cores,
            backend=self.evaluation_backend,
            batch_size=self.batch_size,
            q_bins=self.q_bins,
        ) as executors:
            result.worker_allocation = dict(executors.allocation)

            def evaluate_samples(
                active_integrator: NumericalIntegrator,
                active_rng: Any,
            ) -> tuple[
                list[Sample],
                list[tuple[int, int]],
                list[float],
                list[tuple[float, ...]],
                IntegrationResult,
            ]:
                samples = list(
                    active_integrator.sample(points_per_iteration, active_rng)
                )
                rows_by_channel: dict[str, list[_PhysicalRow]] = {
                    channel.label: [] for channel in self.channels
                }
                sample_paths: list[tuple[int, int]] = []
                for sample_index, sample in enumerate(samples):
                    if len(sample.d) != 1:
                        raise pygloopException(
                            "Joint Symbolica sample must contain exactly one "
                            "discrete leaf index."
                        )
                    joint_index = int(sample.d[0])
                    if joint_index < 0 or joint_index >= len(
                        self.joint_channel_paths
                    ):
                        raise pygloopException(
                            f"Joint channel index {joint_index} is out of range."
                        )
                    physical_index, graph_index = self.joint_channel_paths[
                        joint_index
                    ]
                    channel = self.channels[physical_index]
                    sample_paths.append((physical_index, graph_index))
                    rows_by_channel[channel.label].append(
                        (sample_index, list(sample.c), graph_index)
                    )

                indexed_values, diagnostics = executors.evaluate(rows_by_channel)
                values: list[float | None] = [None] * len(samples)
                bin_values: list[tuple[float, ...] | None] = [None] * len(samples)
                for sample_index, value, bins in indexed_values:
                    if values[sample_index] is not None:
                        raise pygloopException(
                            f"Physical-channel sample {sample_index} was evaluated twice."
                        )
                    if len(bins) != len(q_bin_moments):
                        raise pygloopException(
                            "Physical-channel Q-bin evaluation returned the wrong "
                            "number of bins."
                        )
                    values[sample_index] = value
                    bin_values[sample_index] = bins
                if any(value is None for value in values) or any(
                    bins is None for bins in bin_values
                ):
                    raise pygloopException(
                        "Physical-channel evaluation left samples unevaluated."
                    )
                return (
                    samples,
                    sample_paths,
                    [float(value) for value in values if value is not None],
                    [bins for bins in bin_values if bins is not None],
                    diagnostics,
                )

            for iteration in range(warmup_iterations):
                logger.info(
                    "Physical-channel Symbolica warmup: iteration %d/%d with "
                    "%d points",
                    iteration + 1,
                    warmup_iterations,
                    points_per_iteration,
                )
                (
                    samples,
                    _sample_paths,
                    raw_values,
                    _bin_values,
                    diagnostics,
                ) = evaluate_samples(integrator, rng)
                integrator.add_training_samples(samples, raw_values)
                integrator.update(
                    continuous_learning_rate=1.0,
                    discrete_learning_rate=1.0,
                )
                result.combine_with(diagnostics)
                result.warmup_samples += len(samples)

            frozen_production = warmup_iterations > 0
            if frozen_production:
                production_integrator = NumericalIntegrator.import_grid(
                    integrator.export_grid(False)
                )
                production_rng = production_integrator.rng(
                    seed=int(seed), stream_id=1
                )
            else:
                production_integrator = integrator
                production_rng = rng

            for iteration in range(n_iterations):
                logger.info(
                    "Physical-channel Symbolica production: iteration %d/%d "
                    "with %d points",
                    iteration + 1,
                    n_iterations,
                    points_per_iteration,
                )
                (
                    samples,
                    sample_paths,
                    raw_values,
                    sampled_bin_values,
                    diagnostics,
                ) = evaluate_samples(production_integrator, production_rng)
                if not frozen_production:
                    production_integrator.add_training_samples(
                        samples, raw_values
                    )
                    average, error, _chi_sq = production_integrator.update(
                        continuous_learning_rate=1.0,
                        discrete_learning_rate=1.0,
                    )
                    result.central_value = float(average)
                    result.error = float(error)
                result.combine_with(diagnostics)
                result.production_samples += len(samples)

                total_moments.start_batch(len(samples))
                for moment in channel_moments.values():
                    moment.start_batch(len(samples))
                for rows in graph_moments.values():
                    for moment in rows.values():
                        moment.start_batch(len(samples))
                for moment in q_bin_moments:
                    moment.start_batch(len(samples))
                for rows in channel_bin_moments.values():
                    for moment in rows:
                        moment.start_batch(len(samples))
                for sample, value, bins, (physical_index, graph_index) in zip(
                    samples,
                    raw_values,
                    sampled_bin_values,
                    sample_paths,
                    strict=True,
                ):
                    channel = self.channels[physical_index]
                    sample_weight = DY._symbolica_sample_weight(sample)
                    weighted_value = value * sample_weight
                    total_moments.add_selected(weighted_value)
                    channel_moments[channel.label].add_selected(weighted_value)
                    graph_name = channel.graph_channel_names[graph_index]
                    graph_moments[channel.label][graph_name].add_selected(
                        weighted_value
                    )
                    graph_selected_counts[channel.label][graph_name] += 1
                    for bin_index, bin_value in enumerate(bins):
                        weighted_bin_value = bin_value * sample_weight
                        q_bin_moments[bin_index].add_selected(weighted_bin_value)
                        channel_bin_moments[channel.label][
                            bin_index
                        ].add_selected(weighted_bin_value)
                    selected_counts[channel.label] += 1

                # Use one common importance-sampling estimator for the total and
                # all observers.  This keeps physical/bin decompositions locally
                # coherent and retains cancellations in the total variance even
                # while Symbolica adapts the grid between batches.
                result.central_value, result.error = total_moments.estimate()

                logger.info(
                    "Physical-channel hard total after iteration %d: "
                    "%+.16e +/- %.4e",
                    iteration + 1,
                    result.central_value,
                    result.error,
                )
                result.production_iterations = iteration + 1
                physical_estimate = None
                if (
                    target_relative_precision is not None
                    or self.live_iteration_diagnostics
                ):
                    physical_estimate = self._physical_iteration_estimate(
                        channel_moments, auxiliary_preview_channels
                    )
                if self.live_iteration_diagnostics:
                    self._record_live_iteration_diagnostics(
                        result,
                        iteration=iteration + 1,
                        batch_diagnostics=diagnostics,
                        estimate=physical_estimate or {},
                        selected_counts=selected_counts,
                        effective_probabilities={
                            label: count / result.production_samples
                            for label, count in selected_counts.items()
                        },
                    )
                if (
                    target_relative_precision is not None
                    and self._record_precision_checkpoint(
                        result,
                        iteration=iteration + 1,
                        auxiliary_central=auxiliary_preview[0],
                        auxiliary_error=auxiliary_preview[1],
                        physical_estimate=physical_estimate,
                    )
                ):
                    break

            result.worker_pids = {
                label: tuple(sorted(pids))
                for label, pids in executors.worker_pids.items()
            }

        self._apply_auxiliary_corrections(
            result,
            channel_moments,
            channel_bin_moments,
            q_bin_moments,
            seed=int(seed),
        )
        self._finalise_precision_metadata(result)
        self._attach_coupling_normalisation_metadata(
            self.channels[0].process, result
        )
        for channel in self.channels:
            result.physical_channel_estimates[channel.label][
                "selected_samples"
            ] = selected_counts[channel.label]
            result.graph_channel_estimates[channel.label] = {}
            for graph_name, moments in graph_moments[channel.label].items():
                central, error = moments.estimate()
                result.graph_channel_estimates[channel.label][graph_name] = {
                    "central_value": central,
                    "error": error,
                    "selected_samples": graph_selected_counts[channel.label][
                        graph_name
                    ],
                }
        return result


__all__ = [
    "DYPhysicalChannel",
    "DYPhysicalChannelIntegrationResult",
    "DYPhysicalChannelIntegrator",
    "PHYSICAL_CHANNEL_PARTONS",
]
