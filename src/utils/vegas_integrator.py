import logging
import multiprocessing
import time
from typing import Any, Callable

import vegas  # type: ignore

from utils.utils import IntegrationResult, chunks


def vegas_worker(
    process_cls,
    process_builder_inputs: tuple[Any],
    id: int,
    all_xs: list[list[float]],
    call_args: list[Any],
) -> tuple[int, list[float], IntegrationResult]:
    """Worker that evaluates a batch of points for the VEGAS integrator."""
    res = IntegrationResult(0.0, 0.0)
    t_start = time.time()
    all_weights = []
    process = process_cls(*process_builder_inputs, clean=False, logger_level=logging.CRITICAL)  # type: ignore
    for xs in all_xs:
        weight = process.integrand_xspace(xs, *call_args)
        all_weights.append(weight)
        if res.max_wgt is None or abs(weight) > abs(res.max_wgt):
            res.max_wgt = weight
            res.max_wgt_point = xs
        res.central_value += weight
        res.error += weight**2
        res.n_samples += 1
    res.elapsed_time += time.time() - t_start

    return (id, all_weights, res)


def vegas_functor(process, res: IntegrationResult, n_cores: int, call_args: list[Any]) -> Callable[[list[list[float]]], list[float]]:
    """Create a VEGAS-compatible integrand that can fan out across processes."""
    process_cls = process.__class__

    @vegas.batchintegrand
    def f(all_xs):
        all_weights = []
        if n_cores > 1:
            all_args = [
                (process_cls, process.builder_inputs(), i_chunk, all_xs_split, call_args)
                for i_chunk, all_xs_split in enumerate(chunks(all_xs, len(all_xs) // n_cores + 1))
            ]
            with multiprocessing.Pool(processes=n_cores) as pool:
                all_results = pool.starmap(vegas_worker, all_args)
            for _id, wgts, this_result in sorted(all_results, key=lambda x: x[0]):
                all_weights.extend(wgts)
                res.combine_with(this_result)
            return all_weights
        else:
            _id, wgts, this_result = vegas_worker(process_cls, process.builder_inputs(), 0, all_xs, call_args)
            all_weights.extend(wgts)
            res.combine_with(this_result)
        return all_weights

    return f


def vegas_integrator(
    process_instance,
    parameterisation: str,
    integrand_implementation: str,
    _target,
    **opts,
) -> IntegrationResult:
    """Run VEGAS integration for a process instance."""
    integration_result = IntegrationResult(0.0, 0.0)

    integrator = vegas.Integrator(3 * [[0, 1]])

    local_worker = vegas_functor(
        process_instance,
        integration_result,
        opts["n_cores"],
        [
            parameterisation,
            integrand_implementation,
            opts.get("phase", "real"),
            opts["multi_channeling"],
        ],
    )
    integrator(
        local_worker,
        nitn=opts["n_iterations"],
        neval=opts["points_per_iteration"],
        analyzer=vegas.reporter(),
    )
    result = integrator(
        local_worker,
        nitn=opts["n_iterations"],
        neval=opts["points_per_iteration"],
        analyzer=vegas.reporter(),
    )

    integration_result.central_value = result.mean
    integration_result.error = result.sdev
    return integration_result
