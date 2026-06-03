#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from processes.qqbar_nX.qqbar_nX import qqbar_nX  # noqa: E402
from processes.qqbar_nX.qqbar_nX_graphs import graph_name  # noqa: E402
from utils.vectors import LorentzVector  # noqa: E402


DEFAULT_BASE_POINT = [
    2.8571429093025008e-01,
    3.3393164400466513e-01,
    4.9999993610427912e-01,
    6.5006336725817859e-01,
    1.2047725413691221e-01,
    6.8364571096398241e-01,
]


def _default_ps_point() -> list[LorentzVector]:
    return [
        LorentzVector(500.0, 0.0, 0.0, 500.0),
        LorentzVector(500.0, 0.0, 0.0, -500.0),
        LorentzVector(
            438.5555662246945,
            155.3322001835378,
            348.0160396513587,
            -177.3773615718412,
        ),
        LorentzVector(
            356.3696374921922,
            -16.80238900851100,
            -318.7291102436005,
            97.48719163688098,
        ),
        LorentzVector(
            205.0747962831133,
            -138.5298111750267,
            -29.28692940775817,
            79.89016993496030,
        ),
    ]


def _complex_from_api_value(value: Any) -> complex:
    if isinstance(value, complex):
        return value
    if hasattr(value, "re") and hasattr(value, "im"):
        return complex(float(value.re), float(value.im))
    return complex(value)


def _double_precision_runtime_fragment(helicities: list[int]) -> str:
    return (
        "[kinematics.externals.data]\n"
        f"helicities = [{', '.join(str(item) for item in helicities)}]\n"
        "\n"
        "[sampling]\n"
        'graphs = "monte_carlo"\n'
        'orientations = "summed"\n'
        "lmb_multichanneling = false\n"
        'lmb_channels = "summed"\n'
        'coordinate_system = "spherical"\n'
        'mapping = "linear"\n'
        "b = 1.0\n"
        "\n"
        "[stability]\n"
        "rotation_axis = [{ type = \"none\" }]\n"
        "check_on_norm = true\n"
        "escalate_if_exact_zero = false\n"
        "\n"
        "[[stability.levels]]\n"
        'precision = "Double"\n'
        "required_precision_for_re = 1e-10\n"
        "required_precision_for_im = 1e-10\n"
        "escalate_for_large_weight_threshold = -1.0\n"
    )


def _member_values_from_sample(
    sample: Any,
    *,
    graph_id_to_name: dict[int, str],
    ordered_graph_names: list[str],
) -> dict[str, complex]:
    values: dict[str, complex] = defaultdict(complex)
    for event_group in sample.event_groups:
        for event in event_group.events:
            graph_id = int(event.cut_info.graph_id)
            name = graph_id_to_name.get(graph_id, f"graph#{graph_id}")
            values[name] += _complex_from_api_value(event.weight)
    return {name: values.get(name, 0.0j) for name in ordered_graph_names}


def _evaluate_scan(
    process: qqbar_nX,
    api: Any,
    *,
    integrand_name: str,
    graph_id_to_name: dict[int, str],
    graph_names: list[str],
    points: np.ndarray,
    group_id: int,
    batch_size: int,
    use_arb_prec: bool,
) -> dict[str, list[complex]]:
    series: dict[str, list[complex]] = {name: [] for name in graph_names}
    series["total"] = []
    discrete_dims = np.array([[group_id]] * min(batch_size, len(points)), dtype=np.uintp)
    for start in range(0, len(points), batch_size):
        chunk = points[start : start + batch_size]
        if len(discrete_dims) != len(chunk):
            discrete_dims = np.array([[group_id]] * len(chunk), dtype=np.uintp)
        batch = api.evaluate_samples(
            chunk,
            integrand_name=integrand_name,
            use_arb_prec=use_arb_prec,
            minimal_output=True,
            return_events=True,
            momentum_space=False,
            discrete_dims=discrete_dims,
        )
        for sample in batch.samples:
            members = _member_values_from_sample(
                sample,
                graph_id_to_name=graph_id_to_name,
                ordered_graph_names=graph_names,
            )
            total = sum(members.values(), 0.0j)
            for name in graph_names:
                series[name].append(members[name])
            series["total"].append(total)
    return series


def _plot_series(
    *,
    x_values: np.ndarray,
    series: dict[str, list[complex]],
    title: str,
    x_label: str,
    output_path: str,
    vertical_x: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    names = [name for name in series if name != "total"] + ["total"]
    cmap = plt.get_cmap("tab10")
    for index, name in enumerate(names):
        values = np.array(series[name], dtype=complex)
        y_values = np.abs(values)
        y_values[y_values == 0.0] = np.nan
        positive = np.real(values) >= 0.0
        color = "black" if name == "total" else cmap(index % 10)
        width = 2.6 if name == "total" else 1.5
        alpha = 1.0 if name == "total" else 0.82
        ax.plot(
            x_values,
            np.where(positive, y_values, np.nan),
            color=color,
            linestyle="-",
            linewidth=width,
            alpha=alpha,
            label=f"{name} Re>=0",
        )
        ax.plot(
            x_values,
            np.where(~positive, y_values, np.nan),
            color=color,
            linestyle="--",
            linewidth=width,
            alpha=alpha,
            label=f"{name} Re<0",
        )
    if vertical_x is not None and math.isfinite(vertical_x):
        ax.axvline(vertical_x, color="0.35", linestyle=":", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("|event weight|")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize="small", ncols=2)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _fit_power_law(x3_values: np.ndarray, values: list[complex], *, side: str) -> float:
    if side == "left":
        mask = (x3_values < 0.5) & ((0.5 - x3_values) < 1.0e-3)
        delta = 0.5 - x3_values[mask]
    elif side == "right":
        mask = (x3_values > 0.5) & ((x3_values - 0.5) < 1.0e-3)
        delta = x3_values[mask] - 0.5
    else:
        raise ValueError(side)
    weights = np.abs(np.array(values, dtype=complex)[mask])
    valid = np.isfinite(weights) & (weights > 0.0) & np.isfinite(delta) & (delta > 0.0)
    if np.count_nonzero(valid) < 4:
        return float("nan")
    # |w| ~ delta^{-p} => log |w| = -p log(delta) + const.
    slope, _intercept = np.polyfit(np.log(delta[valid]), np.log(weights[valid]), 1)
    return float(-slope)


def _points_from_x3(x3_values: np.ndarray, base_point: list[float]) -> np.ndarray:
    points = np.tile(np.array(base_point, dtype=float), (len(x3_values), 1))
    points[:, 2] = x3_values
    return points


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan the qqbar_nX x-space theta coordinate x[2] using GammaLoop's "
            "rich event output and plot individual graph/group weights."
        )
    )
    parser.add_argument(
        "--config",
        default=str(SRC_DIR / "processes/qqbar_nX/config_no_thresholds.toml"),
    )
    parser.add_argument("--m-top", type=float, default=1000.0)
    parser.add_argument("--m-higgs", type=float, default=125.0)
    parser.add_argument("--group-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-log", type=int, default=120)
    parser.add_argument("--n-linear", type=int, default=260)
    parser.add_argument("--arb", action="store_true", default=False)
    parser.add_argument("--clean", action="store_true", default=False)
    parser.add_argument(
        "--base-point",
        type=float,
        nargs=6,
        default=DEFAULT_BASE_POINT,
        help="Six-dimensional integration-space point; x[2] is overwritten.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SRC_DIR.parent / "outputs/dot_files/qqbar_nX/xspace_scans"),
    )
    args = parser.parse_args()

    process = qqbar_nX(
        args.m_top,
        args.m_higgs,
        _default_ps_point(),
        [1, -1, 0, 0, 0],
        2,
        # Recycle the selected Feyngen DOT for fast plotting.  The --clean flag
        # below is applied only to the generated GammaLoop state.
        clean=False,
        qqbar_nx_config_path=args.config,
        skip_gl_worker_init=True,
    )
    manifest = process.build_ir_subtracted_graphs(
        import_graphs=False,
        allow_cached_selected_dot=True,
    )
    process.clean = bool(args.clean)
    integrand_name = process.get_subtracted_integrand_name()
    process_name = process.get_subtracted_process_name()
    api = process._ensure_api_test_state(  # noqa: SLF001
        manifest=manifest,
        integrand_name=integrand_name,
        generate_command_block="generate_subtracted_integrand",
    )
    api.run(f"set process -p {process_name} -i {integrand_name} defaults")
    api.run(
        f"set process -p {process_name} -i {integrand_name} string "
        f"'\n{_double_precision_runtime_fragment(process._helicities_for_current_externals())}'"  # noqa: SLF001
    )

    graph_id_to_name = process._graph_id_name_map_from_integrand_info(  # noqa: SLF001
        api, integrand_name=integrand_name
    )
    grouped_graphs = process._graph_members_by_group_from_dot(  # noqa: SLF001
        manifest["subtracted_dot_path"]
    )
    if args.group_id not in grouped_graphs:
        raise SystemExit(f"group_id {args.group_id} not found in subtracted DOT.")
    graph_names = [graph_name(graph) for graph in grouped_graphs[args.group_id]]

    output_dir = Path(args.output_dir)
    target_x3 = float(args.base_point[2])
    target_log_x = -math.log(max(0.5 - target_x3, np.finfo(float).tiny))

    deltas = np.geomspace(0.5, 1.0e-13, args.n_log)
    x3_log = 0.5 - deltas
    log_points = _points_from_x3(x3_log, args.base_point)
    log_series = _evaluate_scan(
        process,
        api,
        integrand_name=integrand_name,
        graph_id_to_name=graph_id_to_name,
        graph_names=graph_names,
        points=log_points,
        group_id=args.group_id,
        batch_size=args.batch_size,
        use_arb_prec=args.arb,
    )
    log_x_axis = -np.log(0.5 - x3_log)
    log_plot_path = output_dir / "qqbar_nX_x3_theta_half_logapproach_group0.png"
    _plot_series(
        x_values=log_x_axis,
        series=log_series,
        title="qqbar_nX group 0 x[2] -> 0.5 from below",
        x_label="-log(0.5 - x[2])",
        output_path=str(log_plot_path),
        vertical_x=target_log_x,
    )

    linear = np.linspace(1.0e-8, 1.0 - 1.0e-8, args.n_linear)
    left = 0.5 - np.geomspace(1.0e-1, 1.0e-12, args.n_log // 2)
    right = 0.5 + np.geomspace(1.0e-12, 1.0e-1, args.n_log // 2)
    x3_direct = np.unique(
        np.concatenate(
            [
                linear,
                left[(left > 0.0) & (left < 0.5)],
                right[(right > 0.5) & (right < 1.0)],
                np.array([target_x3]),
            ]
        )
    )
    x3_direct = x3_direct[np.abs(x3_direct - 0.5) > 1.0e-15]
    direct_points = _points_from_x3(x3_direct, args.base_point)
    direct_series = _evaluate_scan(
        process,
        api,
        integrand_name=integrand_name,
        graph_id_to_name=graph_id_to_name,
        graph_names=graph_names,
        points=direct_points,
        group_id=args.group_id,
        batch_size=args.batch_size,
        use_arb_prec=args.arb,
    )
    direct_plot_path = output_dir / "qqbar_nX_x3_theta_half_direct_group0.png"
    _plot_series(
        x_values=x3_direct,
        series=direct_series,
        title="qqbar_nX group 0 scan in the theta coordinate x[2]",
        x_label="x[2]",
        output_path=str(direct_plot_path),
        vertical_x=0.5,
    )

    powers = {
        name: {
            "left": _fit_power_law(x3_direct, values, side="left"),
            "right": _fit_power_law(x3_direct, values, side="right"),
        }
        for name, values in direct_series.items()
    }
    output = {
        "base_point": args.base_point,
        "group_id": args.group_id,
        "config": os.path.abspath(args.config),
        "m_top": args.m_top,
        "integrand_name": integrand_name,
        "graph_names": graph_names,
        "log_scan": {
            "x3": x3_log.tolist(),
            "x_axis": log_x_axis.tolist(),
            "series": {
                name: [{"re": value.real, "im": value.imag} for value in values]
                for name, values in log_series.items()
            },
        },
        "direct_scan": {
            "x3": x3_direct.tolist(),
            "series": {
                name: [{"re": value.real, "im": value.imag} for value in values]
                for name, values in direct_series.items()
            },
        },
        "power_fits_near_half": powers,
        "plots": {
            "log_approach": str(log_plot_path),
            "direct": str(direct_plot_path),
        },
    }
    json_path = output_dir / "qqbar_nX_x3_theta_half_scan_group0.json"
    json_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"plots": output["plots"], "power_fits_near_half": powers}, indent=2))


if __name__ == "__main__":
    main()
