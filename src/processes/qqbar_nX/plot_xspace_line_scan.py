#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from gammaloop import GammaLoopAPI  # type: ignore

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from processes.qqbar_nX.qqbar_nX import qqbar_nX  # noqa: E402
from processes.qqbar_nX.qqbar_nX_graphs import (  # noqa: E402
    dot_graphs_to_string,
    graph_name,
    parse_dot_graphs,
    strip_quotes,
)
from utils.vectors import LorentzVector  # noqa: E402


DEFAULT_BASE_POINT = [
    3.1390634220043045e-01,
    6.1945042268059569e-01,
    9.0716850422920681e-01,
    7.1401078676936824e-01,
    4.9683778437982462e-01,
    7.8198648800276493e-02,
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


def _runtime_fragment(helicities: list[int], *, lmb_multichanneling: bool) -> str:
    lmb_channels = "summed" if lmb_multichanneling else "summed"
    return (
        "[kinematics.externals.data]\n"
        f"helicities = [{', '.join(str(item) for item in helicities)}]\n"
        "\n"
        "[sampling]\n"
        'graphs = "monte_carlo"\n'
        'orientations = "summed"\n'
        f"lmb_multichanneling = {str(lmb_multichanneling).lower()}\n"
        f'lmb_channels = "{lmb_channels}"\n'
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
    for start in range(0, len(points), batch_size):
        chunk = points[start : start + batch_size]
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


def _line_bounds(point: np.ndarray, direction: np.ndarray) -> tuple[float, float]:
    low = -math.inf
    high = math.inf
    for x_i, v_i in zip(point, direction, strict=True):
        if abs(v_i) < 1.0e-15:
            if not (0.0 <= x_i <= 1.0):
                raise ValueError("base point is outside the unit cube")
            continue
        left = (0.0 - x_i) / v_i
        right = (1.0 - x_i) / v_i
        low = max(low, min(left, right))
        high = min(high, max(left, right))
    if not (low <= 0.0 <= high):
        raise ValueError("line does not cross the base point inside the unit cube")
    return low, high


def _plot_series(
    *,
    t_values: np.ndarray,
    series: dict[str, list[complex]],
    title: str,
    output_path: Path,
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
        width = 2.8 if name == "total" else 1.5
        alpha = 1.0 if name == "total" else 0.82
        ax.plot(
            t_values,
            np.where(positive, y_values, np.nan),
            color=color,
            linestyle="-",
            linewidth=width,
            alpha=alpha,
            label=f"{name} Re>=0",
        )
        ax.plot(
            t_values,
            np.where(~positive, y_values, np.nan),
            color=color,
            linestyle="--",
            linewidth=width,
            alpha=alpha,
            label=f"{name} Re<0",
        )
    ax.axvline(0.0, color="0.35", linestyle=":", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("line parameter t, with x(t)=x0+t v")
    ax.set_ylabel("|event weight|")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize="small", ncols=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _direction_from_name(name: str, *, seed: int) -> np.ndarray:
    if name == "e0":
        direction = np.zeros(6)
        direction[0] = 1.0
        return direction
    if name == "e3":
        direction = np.zeros(6)
        direction[3] = 1.0
        return direction
    if name == "random":
        rng = np.random.default_rng(seed)
        direction = rng.normal(size=6)
        return direction / np.linalg.norm(direction)
    raise ValueError(name)


def _state_has_integrand(state_folder: str, integrand_name: str) -> bool:
    process_folder = Path(state_folder) / "processes"
    if not process_folder.is_dir():
        return False
    return any(
        path.parent.name == integrand_name
        for path in process_folder.rglob("amp.bin")
    )


def _prepare_api_state(
    *,
    manifest: dict[str, Any],
    integrand_name: str,
    process_name: str,
    clean: bool,
    thresholds: bool,
) -> GammaLoopAPI:
    state_folder = manifest.get("standalone_state_folder")
    run_card_path = manifest.get("standalone_run_card_path")
    if not state_folder or not run_card_path:
        raise RuntimeError("Missing standalone state folder or run card path in manifest.")
    if clean and Path(state_folder).is_dir():
        import shutil

        shutil.rmtree(state_folder)
    load_block = "load_with_thresholds" if thresholds else "load"
    generate_block = "generate_with_thresholds" if thresholds else "generate"

    def generate(api: GammaLoopAPI) -> None:
        if not thresholds:
            api.run(f"run {generate_block}")
            return
        # Keep threshold subtraction enabled, but use GammaLoop's default LMB
        # heuristics.  Forcing all heuristic pruning off makes the threshold
        # generation path prohibitively heavy for this diagnostic.
        for command in [
            "set global kv global.generation.override_lmb_heuristics=false",
            "set global kv global.generation.uv.subtract_uv=false",
            "set global kv global.generation.uv.generate_integrated=false",
            "set global kv global.generation.uv.local_uv_cts_from_expanded_4d_integrands=false",
            "set global kv global.generation.threshold_subtraction.enable_thresholds=true",
            "set global kv global.generation.threshold_subtraction.check_esurface_at_generation=false",
            "set global kv global.generation.threshold_subtraction.assume_positive_external_energies=false",
            f"generate existing -p {process_name} -i {integrand_name}",
        ]:
            api.run(command)

    if not Path(state_folder).is_dir():
        api = GammaLoopAPI(state_folder, boot_commands_path=run_card_path, clean_state=True)
        api.run(f"run {load_block}")
        generate(api)
        api.run("save state -o true")
        return api
    api = GammaLoopAPI(state_folder, clean_state=False)
    if thresholds or not _state_has_integrand(state_folder, integrand_name):
        api.run(f"run {load_block}")
        generate(api)
        api.run("save state -o true")
    return api


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan qqbar_nX rich event weights along 6D x-space lines."
    )
    parser.add_argument(
        "--config",
        default=str(SRC_DIR / "processes/qqbar_nX/config_no_thresholds.toml"),
    )
    parser.add_argument("--m-top", type=float, default=1000.0)
    parser.add_argument("--m-higgs", type=float, default=125.0)
    parser.add_argument("--group-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-points", type=int, default=301)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--arb", action="store_true", default=False)
    parser.add_argument("--clean", action="store_true", default=False)
    parser.add_argument("--thresholds", action="store_true", default=False)
    parser.add_argument("--lmb-multichanneling", action="store_true", default=False)
    parser.add_argument(
        "--subset-dot-to-group",
        action="store_true",
        default=False,
        help="Import only the requested group_id graphs in the diagnostic state.",
    )
    parser.add_argument(
        "--base-point",
        type=float,
        nargs=6,
        default=DEFAULT_BASE_POINT,
        help="Six-dimensional x-space point through which each line passes.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SRC_DIR.parent / "outputs/dot_files/qqbar_nX/xspace_line_scans"),
    )
    args = parser.parse_args()

    process = qqbar_nX(
        args.m_top,
        args.m_higgs,
        _default_ps_point(),
        [1, -1, 0, 0, 0],
        2,
        clean=False,
        qqbar_nx_config_path=args.config,
        skip_gl_worker_init=True,
    )
    manifest = process.build_ir_subtracted_graphs(
        import_graphs=False,
        allow_cached_selected_dot=True,
    )
    if args.subset_dot_to_group:
        full_dot_path = Path(manifest["subtracted_dot_path"])
        graphs = [
            graph
            for graph in parse_dot_graphs(full_dot_path.read_text(encoding="utf-8"))
            if strip_quotes(graph.get_attributes().get("group_id", "")) == str(args.group_id)
        ]
        if not graphs:
            raise SystemExit(f"group_id {args.group_id} not found in {full_dot_path}.")
        subset_name = f"{process.get_subtracted_integrand_name()}_group{args.group_id}_only"
        subset_dot_path = Path(process.dot_folder) / f"{subset_name}.dot"
        subset_dot_path.write_text(dot_graphs_to_string(graphs), encoding="utf-8")
        run_card_path = process.write_gammaloop_run_card(
            str(subset_dot_path), integrand_name=subset_name
        )
        manifest = {
            **manifest,
            "subtracted_dot_path": str(subset_dot_path),
            "standalone_run_card_path": run_card_path,
            "standalone_state_folder": process.get_standalone_state_folder(subset_name),
        }
    process.clean = bool(args.clean)

    api_integrand_name = "subtracted"
    api_process_name = "qqbar_hhh"
    api = _prepare_api_state(
        manifest=manifest,
        integrand_name=api_integrand_name,
        process_name=api_process_name,
        clean=bool(args.clean),
        thresholds=bool(args.thresholds),
    )
    api.run(f"set process -p {api_process_name} -i {api_integrand_name} defaults")
    api.run(
        f"set process -p {api_process_name} -i {api_integrand_name} kv "
        f"subtraction.disable_threshold_subtraction={str((not args.thresholds)).lower()}"
    )
    api.run(
        f"set process -p {api_process_name} -i {api_integrand_name} string "
        f"'\n{_runtime_fragment(process._helicities_for_current_externals(), lmb_multichanneling=args.lmb_multichanneling)}'"  # noqa: SLF001
    )

    graph_id_to_name = process._graph_id_name_map_from_integrand_info(  # noqa: SLF001
        api, integrand_name=api_integrand_name
    )
    grouped_graphs = process._graph_members_by_group_from_dot(  # noqa: SLF001
        manifest["subtracted_dot_path"]
    )
    if args.group_id not in grouped_graphs:
        raise SystemExit(f"group_id {args.group_id} not found in subtracted DOT.")
    graph_names = [graph_name(graph) for graph in grouped_graphs[args.group_id]]

    base_point = np.array(args.base_point, dtype=float)
    output_dir = Path(args.output_dir)
    outputs: dict[str, Any] = {
        "base_point": base_point.tolist(),
        "group_id": args.group_id,
        "graph_names": graph_names,
        "lmb_multichanneling": args.lmb_multichanneling,
        "plots": {},
        "line_bounds": {},
        "values_at_t0": {},
    }
    for direction_name in ("e0", "e3", "random"):
        direction = _direction_from_name(direction_name, seed=args.seed)
        t_min, t_max = _line_bounds(base_point, direction)
        t_values = np.linspace(t_min, t_max, args.n_points)
        if not np.any(np.isclose(t_values, 0.0)):
            t_values = np.sort(np.unique(np.concatenate([t_values, np.array([0.0])])))
        points = base_point[None, :] + t_values[:, None] * direction[None, :]
        points = np.clip(points, 0.0, 1.0)
        series = _evaluate_scan(
            api,
            integrand_name=api_integrand_name,
            graph_id_to_name=graph_id_to_name,
            graph_names=graph_names,
            points=points,
            group_id=args.group_id,
            batch_size=args.batch_size,
            use_arb_prec=args.arb,
        )
        plot_path = output_dir / f"qqbar_nX_group{args.group_id}_line_{direction_name}.png"
        _plot_series(
            t_values=t_values,
            series=series,
            title=(
                f"qqbar_nX group {args.group_id} line scan {direction_name} "
                f"through max-weight point"
            ),
            output_path=plot_path,
        )
        zero_index = int(np.argmin(np.abs(t_values)))
        outputs["plots"][direction_name] = str(plot_path)
        outputs["line_bounds"][direction_name] = {
            "direction": direction.tolist(),
            "t_min": float(t_min),
            "t_max": float(t_max),
        }
        outputs["values_at_t0"][direction_name] = {
            name: {
                "re": float(series[name][zero_index].real),
                "im": float(series[name][zero_index].imag),
                "abs": float(abs(series[name][zero_index])),
            }
            for name in series
        }

    json_path = output_dir / f"qqbar_nX_group{args.group_id}_line_scan.json"
    json_path.write_text(json.dumps(outputs, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"plots": outputs["plots"], "values_at_t0": outputs["values_at_t0"]}, indent=2))


if __name__ == "__main__":
    main()
