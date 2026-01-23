import logging
import multiprocessing
import time
from typing import Any

from symbolica import NumericalIntegrator, Sample

from utils.utils import IntegrationResult, SymbolicaSample, chunks, logger


def symbolica_worker(
    process_cls,
    process_builder_inputs: tuple[Any],
    id: int,
    multi_channeling: bool,
    all_xs: list[SymbolicaSample],
    call_args: list[Any],
) -> tuple[int, list[float], IntegrationResult]:
    """Worker that evaluates Symbolica samples for integration."""
    res = IntegrationResult(0.0, 0.0)
    t_start = time.time()
    all_weights = []
    process = process_cls(*process_builder_inputs, clean=False, logger_level=logging.CRITICAL)  # type: ignore
    for xs in all_xs:
        if not multi_channeling:
            weight = process.integrand_xspace(xs.c, *(call_args + [False]))
        else:
            weight = process.integrand_xspace(xs.c, *(call_args + [xs.d[0]]))
        all_weights.append(weight)
        if res.max_wgt is None or abs(weight) > abs(res.max_wgt):
            res.max_wgt = weight
            if not multi_channeling:
                res.max_wgt_point = xs.c
            else:
                res.max_wgt_point = xs.d + xs.c
        res.central_value += weight
        res.error += weight**2
        res.n_samples += 1
    res.elapsed_time += time.time() - t_start

    return (id, all_weights, res)


def symbolica_integrand_function(
    process,
    res: IntegrationResult,
    n_cores: int,
    multi_channeling: bool,
    call_args: list[Any],
    samples: list[Sample],
) -> list[float]:
    """Fan out Symbolica sampling across processes and gather weights."""
    process_cls = process.__class__
    all_weights = []
    if n_cores > 1:
        all_args = [
            (
                process_cls,
                process.builder_inputs(),
                i_chunk,
                multi_channeling,
                [SymbolicaSample(s) for s in all_xs_split],
                call_args,
            )
            for i_chunk, all_xs_split in enumerate(chunks(samples, len(samples) // n_cores + 1))
        ]
        with multiprocessing.Pool(processes=n_cores) as pool:
            all_results = pool.starmap(symbolica_worker, all_args)
        for _id, wgts, this_result in sorted(all_results, key=lambda x: x[0]):
            all_weights.extend(wgts)
            res.combine_with(this_result)
        return all_weights
    else:
        _id, wgts, this_result = symbolica_worker(
            process_cls,
            process.builder_inputs(),
            0,
            multi_channeling,
            [SymbolicaSample(s) for s in samples],
            call_args,
        )
        all_weights.extend(wgts)
        res.combine_with(this_result)
    return all_weights


def symbolica_integrator(
    process_instance,
    parameterisation: str,
    integrand_implementation: str,
    target,
    **opts,
) -> IntegrationResult:
    """Run Symbolica integration for a process instance."""
    integration_result = IntegrationResult(0.0, 0.0)

    if opts["multi_channeling"]:
        integrator = NumericalIntegrator.discrete(
            [
                NumericalIntegrator.continuous(3),
                NumericalIntegrator.continuous(3),
                NumericalIntegrator.continuous(3),
                NumericalIntegrator.continuous(3),
                NumericalIntegrator.continuous(3),
            ]
        )
    else:
        integrator = NumericalIntegrator.continuous(3)

    rng = integrator.rng(seed=opts["seed"], stream_id=0)

    for i_iter in range(opts["n_iterations"]):
        logger.info(f"Symbolica integration: starting iteration {i_iter + 1}/{opts['n_iterations']} using {opts['points_per_iteration']} points ...")
        samples = integrator.sample(opts["points_per_iteration"], rng)
        res = symbolica_integrand_function(
            process_instance,
            integration_result,
            opts["n_cores"],
            opts["multi_channeling"],
            [parameterisation, integrand_implementation, opts.get("phase", "real")],
            samples,
        )
        integrator.add_training_samples(samples, res)

        avg, err, _chi_sq = integrator.update(continuous_learning_rate=1.5, discrete_learning_rate=1.5)  # type: ignore
        integration_result.central_value = avg
        integration_result.error = err
        logger.info(f"... result after this iteration:\n{integration_result.str_report(target)}")

    return integration_result
