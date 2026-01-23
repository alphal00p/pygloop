import copy
import multiprocessing
from typing import Any

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import progressbar

from utils.utils import logger, pygloopException
from utils.vectors import Vector


def _plot_worker_init(base, config: dict[str, Any]) -> None:
    proc = multiprocessing.current_process()
    proc._plot_worker = copy.deepcopy(base)  # type: ignore[attr-defined]
    proc._plot_config = config  # type: ignore[attr-defined]


def _plot_worker(task: tuple[int, int, float, float]) -> tuple[int, int, float]:
    proc = multiprocessing.current_process()
    worker = getattr(proc, "_plot_worker", None)
    config = getattr(proc, "_plot_config", None)
    if worker is None or config is None:
        raise pygloopException("Plot worker is not initialized.")
    i, j, x_val, y_val = task
    xs = [0.0, 0.0, 0.0]
    xs[config["fixed_x"]] = config["fixed_value"]
    xs[config["xs0"]] = x_val
    xs[config["xs1"]] = y_val
    if config["x_space"]:
        val = worker.integrand_xspace(
            xs,
            config["parameterisation"],
            config["integrand_implementation"],
            config["phase"],
            config["multi_channeling"],
        )
    else:
        wgt = worker.integrand([Vector(xs[0], xs[1], xs[2])], config["integrand_implementation"])
        match config["phase"]:
            case "real":
                val = wgt.real
            case "imag":
                val = wgt.imag
            case _:
                val = abs(wgt)
    return i, j, val


def plot_integrand(process_instance, **opts):
    """Generic integrand plotting helper shared across processes."""
    fixed_x = None
    for i_x in range(3):
        if i_x not in opts["xs"]:
            fixed_x = i_x
            break
    if fixed_x is None:
        raise pygloopException("At least one x must be fixed (0,1 or 2).")
    n_bins = opts["mesh_size"]
    offset = 1e-6
    x = np.linspace(opts["range"][0] + offset, opts["range"][1] - offset, n_bins)
    y = np.linspace(opts["range"][0] + offset, opts["range"][1] - offset, n_bins)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros((n_bins, n_bins))
    xs = [0.0] * 3
    xs[fixed_x] = opts["fixed_x"]
    nb_cores = max(1, int(opts.get("nb_cores", 1)))
    total = n_bins * n_bins
    logger.info(f"Evaluating function on grid for plotting over {nb_cores} cores...")

    def sequential_plotting():
        for idx in progressbar.progressbar(range(total), max_value=total):
            i, j = divmod(idx, n_bins)
            xs[opts["xs"][0]] = X[i, j]
            xs[opts["xs"][1]] = Y[i, j]
            if opts["x_space"]:
                Z[i, j] = process_instance.integrand_xspace(  # type: ignore
                    xs,
                    opts["parameterisation"],
                    opts["integrand_implementation"],
                    opts.get("phase", "real"),
                    opts["multi_channeling"],
                )
            else:
                wgt = process_instance.integrand([Vector(xs[0], xs[1], xs[2])], opts["integrand_implementation"])
                match opts.get("phase", None):
                    case "real":
                        Z[i, j] = wgt.real  # type: ignore
                    case "imag":
                        Z[i, j] = wgt.imag  # type: ignore
                    case _:
                        Z[i, j] = abs(wgt)  # type: ignore

    if nb_cores == 1:
        sequential_plotting()
    else:
        config = {
            "fixed_x": fixed_x,
            "fixed_value": opts["fixed_x"],
            "xs0": opts["xs"][0],
            "xs1": opts["xs"][1],
            "x_space": opts["x_space"],
            "parameterisation": opts["parameterisation"],
            "integrand_implementation": opts["integrand_implementation"],
            "phase": opts.get("phase", "real") if opts["x_space"] else opts.get("phase", None),
            "multi_channeling": opts["multi_channeling"],
        }
        try:
            ctx = multiprocessing.get_context("fork")
            chunk_size = max(1, total // (nb_cores * 4))
            tasks = ((i, j, float(X[i, j]), float(Y[i, j])) for i in range(n_bins) for j in range(n_bins))
            with ctx.Pool(processes=nb_cores, initializer=_plot_worker_init, initargs=(process_instance, config)) as pool:
                for i, j, val in progressbar.progressbar(  # type: ignore
                    pool.imap_unordered(_plot_worker, tasks, chunksize=chunk_size),
                    max_value=total,
                ):
                    Z[i, j] = val
        except ValueError:
            logger.warning("Multiprocessing start method does not support forking; running sequentially.")
            sequential_plotting()
    logger.info("Done")

    with np.errstate(divide="ignore"):
        log_Z = np.log10(np.abs(Z))
        log_Z[log_Z == -np.inf] = 0

    if opts["x_space"]:
        xs_labels = ["x0", "x1", "x2"]
    else:
        xs_labels = ["kx", "ky", "kz"]
    xs_labels[fixed_x] = str(opts["fixed_x"])

    if not opts["3D"]:
        plt.figure(figsize=(8, 6))
        plt.imshow(
            log_Z,
            origin="lower",
            extent=[
                opts["range"][0],
                opts["range"][1],
                opts["range"][0],
                opts["range"][1],
            ],  # type: ignore
            cmap="viridis",
        )
        plt.colorbar(label=f"log10(I({','.join(xs_labels)}))")
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="viridis")
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_zlabel(f"log10(I({','.join(xs_labels)}))")

    plt.xlabel(f"{xs_labels[opts['xs'][0]]}")
    plt.ylabel(f"{xs_labels[opts['xs'][1]]}")
    plt.title(f"log10(I({','.join(xs_labels)}))")
    plt.show()
