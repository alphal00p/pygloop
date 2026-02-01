import logging
import multiprocessing
import random
import time
from typing import Any

from utils.utils import IntegrationResult, logger


def naive_worker(process_cls, builder_inputs: tuple[Any], n_points: int, call_args: list[Any]) -> IntegrationResult:
    """Worker that evaluates the integrand for a chunk of random points."""
    process_instance = process_cls(*builder_inputs, clean=False, logger_level=logging.CRITICAL)  # type: ignore
    this_result = IntegrationResult(0.0, 0.0)
    t_start = time.time()
    for _ in range(n_points):
        xs = [random.random() for _ in range(process_instance.n_loops * 3)]
        weight = process_instance.integrand_xspace(xs, *call_args)
        if this_result.max_wgt is None or abs(weight) > abs(this_result.max_wgt):
            this_result.max_wgt = weight
            this_result.max_wgt_point = xs
        this_result.central_value += weight
        this_result.error += weight**2
        this_result.n_samples += 1
    this_result.elapsed_time += time.time() - t_start

    return this_result


def naive_integrator(
    process_instance,
    parameterisation: str,
    integrand_implementation: dict[str, Any],
    target,
    **opts,
) -> IntegrationResult:
    """Run the naive Monte Carlo integrator for a process instance."""
    integration_result = IntegrationResult(0.0, 0.0)

    function_call_args = [parameterisation, integrand_implementation, opts["phase"], opts["multi_channeling"]]
    for i_iter in range(opts["n_iterations"]):
        logger.info(f"Naive integration: starting iteration {i_iter + 1}/{opts['n_iterations']} using {opts['points_per_iteration']} points ...")
        if opts["n_cores"] > 1:
            n_points_per_core = opts["points_per_iteration"] // opts["n_cores"]
            all_args = [
                (process_instance.__class__, process_instance.builder_inputs(), n_points_per_core, function_call_args),
            ] * (opts["n_cores"] - 1)
            all_args.append(
                (
                    process_instance.__class__,
                    process_instance.builder_inputs(),
                    opts["points_per_iteration"] - sum(a[2] for a in all_args),
                    function_call_args,
                )
            )
            with multiprocessing.Pool(processes=opts["n_cores"]) as pool:
                all_results = pool.starmap(naive_worker, all_args)

            for result in all_results:
                integration_result.combine_with(result)
        else:
            integration_result.combine_with(
                naive_worker(
                    process_instance.__class__,
                    process_instance.builder_inputs(),
                    opts["points_per_iteration"],
                    function_call_args,
                )
            )

        processed_result = IntegrationResult(integration_result.central_value, integration_result.error)
        processed_result.max_wgt = integration_result.max_wgt
        processed_result.max_wgt_point = integration_result.max_wgt_point
        processed_result.n_samples = integration_result.n_samples
        processed_result.elapsed_time = integration_result.elapsed_time
        processed_result.normalize()
        logger.info(f"... result after this iteration:\n{processed_result.str_report(target)}")

    integration_result.normalize()

    return integration_result
