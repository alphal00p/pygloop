"""Opt-in paired-angle randomized-QMC integration for one-loop hadronic DY.

A packet covers both hemispheres probabilistically and the entire polar interval
by disjoint central/negative/positive maps. The latter two share a quantile, so
their leading soft terms cancel *before* conversion to float or a stability
decision. No radial cutoff, clipping, integrand subtraction or target-dependent
sampling is used. All errors are between independent scrambled Sobol replicas.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from decimal import Decimal as D, localcontext
import json
import hashlib
import math
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
from scipy.stats import qmc

from processes.dy.dy import DY
from processes.dy.dy_precision import run_precision_ladder
from processes.dy.dy_histogram import (
    CutHistogram, histogram_estimate, auxiliary_target_met, precision_target_met,
)
from processes.dy.dy_paired_settings import PairedSobolSettings
from processes.dy.dy_physical_channels import DYPhysicalChannelIntegrationResult
from processes.dy.dy_runtime_parameters import dy_coupling_normalisation_factor
from processes.dy.dy_stability import decimal_pi, decimal_values_agree
from utils.utils import IntegrationResult, pygloopException

LABELS = ("qqbar", "qg")
PB = 389379660.0
NBINS = 32


def dec(value):
    return value if isinstance(value, D) else D.from_float(float(value))


class PacketUnstable(ArithmeticError):
    """Numerical failure; unlike programming/configuration errors, may be zeroed."""


class PairedGeometry:
    def __init__(self, energy, q_min, lambda_sq, digits=110, *, z_max=1):
        self.energy = D(str(energy))
        self.q_min = D(str(q_min))
        self.lambda_sq = D(str(lambda_sq))
        self.z_max = D(str(z_max))
        self.digits = digits
        if not 0 < self.q_min < self.energy or not 0 < self.lambda_sq < 8*self.q_min**2:
            raise ValueError("paired map requires 0 < Qmin < Ecm and 0 < lambda_sq < 8 Qmin^2")
        if not 0 < self.z_max <= 1:
            raise ValueError("paired map requires 0 < z_max <= 1")
        with localcontext() as context:
            context.prec = digits + 15
            self.pi = decimal_pi(digits + 5)
            self.s = self.energy**2
            # Q^2 = z*x1*x2*S and z <= z_max. Outside this triangle the
            # hard bin integrand is identically zero, not a soft cutoff.
            self.tau_min = (self.q_min / self.energy)**2 / self.z_max
            if self.tau_min >= 1:
                raise ValueError("paired bin has no beam phase space: Qmin^2 >= S*z_max")
            self.log_tau = self.tau_min.ln()

    def radial(self, unit, radial_upper=None):
        """Full support, including r=0 as a boundary, without a soft cutoff."""
        with localcontext() as context:
            context.prec = self.digits
            u = dec(unit)
            if radial_upper is not None:
                limit = D(str(radial_upper))
                return limit*u, limit
            if u < D(".2"):
                return D("1e-4")*u/D(".2"), D("1e-4")/D(".2")
            if u < D(".6"):
                r = D("1e-4")*(D(100).ln()*(u-D(".2"))/D(".4")).exp()
                return r, r*D(100).ln()/D(".4")
            return D(".01")+D(".99")*(u-D(".6"))/D(".4"), D(".99")/D(".4")

    def outer(self, row, radial_upper=None):
        with localcontext() as context:
            context.prec = self.digits
            r, jr = self.radial(row[0], radial_upper)
            u, v = dec(row[4]), dec(row[5])**3
            x1 = ((1-u)*self.log_tau).exp()
            x2 = ((1-v)*u*self.log_tau).exp()
            jb = x1*x2*self.log_tau**2*u*3*dec(row[5])**2
            return [r, dec(row[1]), dec(row[2]), dec(row[3]), x1, x2], jr*jb

    def entries(self, point, row, label):
        if label == "qg":
            return [("ordinary", point, D(1))]
        with localcontext() as context:
            context.prec = self.digits
            r, _, _, xh, x1, x2 = point
            shat = self.s*x1*x2
            b = self.energy*((1-xh)/xh).sqrt()/(self.pi*shat.sqrt())
            a0 = 8*shat/self.lambda_sq*b
            cap = D(".5")/r
            low = max(D(0), a0-2*b)
            sign = 1 if row[6] > .5 else -1
            u = dec(row[7])
            entries = []

            def append(name, a, derivative):
                p = list(point)
                p[2] = D(".5")+sign*r*a
                entries.append((name, p, 2*r*derivative))

            central = min(cap, low)
            if central > 0:
                append("central", central*dec(row[2]), central)
            high = min(a0, cap)
            if high > low:
                norm = ((high+b)/(low+b)).ln()
                a = (low+b)*(norm*u).exp()-b
                append("negative", a, (a+b)*norm)
            if cap > a0:
                def atanh(x):
                    return ((1+x)/(1-x)).ln()/2
                w = atanh(b/a0)
                delta = w-atanh(b/cap)
                e = (2*(w-delta*u)).exp()
                a = b*(e+1)/(e-1)
                append("positive", a, (a*a-b*b)*delta/b)
            return entries


class PacketEvaluator:
    def __init__(self, channels, settings):
        self.settings = settings
        self.channels = {channel.label: channel for channel in channels}
        self.impl, self.geometry, self.runtime, self.precision_ladders = {}, {}, {}, {}
        self.counts = Counter()
        self.factor_cache = {}
        for label, channel in self.channels.items():
            process = channel.process
            implementation = dict(channel.integrand_implementation)
            levels = process._dy_precision_ladder(implementation)
            for digits in levels[1:]:
                process.compiled_bundle.require_fallback_supported(digits)
            self.precision_ladders[label] = levels
            self.impl[label] = implementation
            runtime = process.compiled_bundle.resolve_runtime_parameters(
                implementation.get("dy_runtime_parameters") or {})
            self.runtime[label] = runtime
            self.geometry[label] = PairedGeometry(
                process.e_cm, process.dy_q_min, runtime["lambda_sq"],
                max(110, levels[-1]+30),
                z_max=(process.dy_z_bin or (0., 1.))[1])

    def beam_factor(self, label, point, digits):
        key = (label, point[4], point[5], digits)
        if key not in self.factor_cache:
            process = self.channels[label].process
            runtime = self.runtime[label]
            pi = decimal_pi(digits+20)
            coupling = dy_coupling_normalisation_factor(
                runtime["alpha_s"], runtime["alpha_ew_inverse"], pi)
            spin_colour = D(1)/D(36 if label == "qqbar" else 96)
            self.factor_cache[key] = (
                dec(process.dy_pdf_luminosity(float(point[4]), float(point[5])))
                * spin_colour/(2*pi)**2/(24*pi*pi)*coupling)
        return self.factor_cache[key]

    def packet_at_precision(self, label, entries, jout, digits):
        """Collect complete rotation orbits, then test the summed packet.

        Native batch and HP evaluators share the existing kernels, mirrors,
        root handling and tolerances. They do not retry or zero an angular
        partner independently. Geometry is retained from the original sample,
        not upcast from a failed native evaluation.
        """
        process = self.channels[label].process
        nb = max(0, len(self.settings.z_histogram_edges)-1)
        rotations = process._rotation_check_count(self.impl[label])
        records = []
        for _, point, jacobian in entries:
            interval = process.dy_physical_z_interval(float(point[4]), float(point[5]))
            if interval is None:
                continue
            self.counts["native_point_requests" if digits == 16 else f"hp{digits}_point_requests"] += 1
            for graph in range(3):
                p = list(point)
                if label == "qqbar" and graph == 2:
                    p[2] = 1-p[2]
                    p[1] = (p[1]+D(".5")) % 1
                impl = dict(self.impl[label], dy_physical_z_min=interval[0],
                            dy_physical_z_max=interval[1], _dy_packet_orbits=[])
                if nb:
                    impl["_dy_histogram"] = CutHistogram(self.settings.z_histogram_edges)
                records.append((graph, p, jacobian, impl))
        rtol, atol = process._stability_tolerances(
            self.impl[label], int(self.impl[label]["dy_rotation_check_digits"]))
        if digits == 16:
            process.integrand_xspace_batch(
                [list(map(float, p)) for _, p, _, _ in records], "spherical",
                [impl for _, _, _, impl in records], "real",
                [graph for graph, _, _, _ in records], integrand_backend="saved-jit")
        else:
            for graph, p, _, impl in records:
                result = process._evaluate_stability_hp_pair(
                    p, "spherical", impl, "real", graph, True, True,
                    process._rotation_descriptors_from_xs(p, rotations),
                    digits, rtol, atol, process._soft_mirror_routing(impl, graph))
                if result[2] is not None or result[0] is None:
                    raise PacketUnstable(f"HP{digits} graph {graph}: {result[2:]}")
        graphs = [[D(0)]*3 for _ in range(rotations+1)]
        bins = [[[D(0)]*nb for _ in range(3)] for _ in range(rotations+1)]
        for graph, _, jacobian, impl in records:
            orbits = impl["_dy_packet_orbits"]
            if len(orbits) != 1:
                raise RuntimeError("Packet evaluator did not return exactly one stability orbit")
            orbit = orbits[0]
            for rotation, value in enumerate(orbit["values"]):
                graphs[rotation][graph] += dec(value)*jacobian
                if nb:
                    for b, v in enumerate(orbit["bins"][rotation]):
                        bins[rotation][graph][b] += dec(v)*jacobian
        totals = [sum(values, D(0)) for values in graphs]
        bin_totals = [[sum(values, D(0)) for values in zip(*row, strict=True)] for row in bins]
        checks = [totals] + [[row[b] for row in bin_totals] for b in range(nb)]
        for values in checks:
            if any(not value.is_finite() for value in values):
                raise PacketUnstable(f"Non-finite packet at {digits} digits")
            for rotated in values[1:]:
                if not decimal_values_agree(
                    values[0], rotated, relative_tolerance=D(str(rtol)),
                    absolute_tolerance=D(str(atol)), decimal_digit_precision=digits):
                    raise PacketUnstable(f"Packet rotation disagrees at {digits} digits")
        if digits == 16 and entries:
            thresholds = process._rescale_dy_beam_stability_thresholds(
                *process._stability_thresholds(self.impl[label]),
                process.dy_pdf_luminosity(float(entries[0][1][4]), float(entries[0][1][5])))
            if any(abs(values[0]) > D(str(threshold))
                   for values in checks for threshold in thresholds if threshold is not None):
                raise PacketUnstable("Native packet exceeds the precision trigger")
        factor = (self.beam_factor(label, entries[0][1], digits) if entries else D(0))
        factor *= (-1 if label == "qqbar" else 1)*jout
        if nb:
            self.histogram_graphs = [[value*factor for value in row] for row in bins[0]]
        values = [value*factor for value in graphs[0]]
        observers = [*values, sum(values, D(0))]
        if nb:
            observers.extend(v for row in self.histogram_graphs for v in row)
            observers.extend(sum(row, D(0)) for row in zip(*self.histogram_graphs, strict=True))
        if any(not math.isfinite(float(value)) for value in observers):
            raise PacketUnstable("Packet overflows the integration accumulator")
        return values

    def evaluate(self, row, label, jproposal=1., radial_upper=None):
        self.factor_cache.clear()
        nb = max(0, len(self.settings.z_histogram_edges)-1)
        self.histogram_graphs = [[D(0)]*nb for _ in range(3)] if nb else None
        self.histogram_values = [D(0)]*nb
        levels = self.precision_ladders[label]
        with localcontext() as context:
            # Extra accumulation digits preserve cancellations, but do not
            # cause ANY evaluator request outside the configured ladder.
            context.prec = max(100, levels[-1]+40)
            point, jout = self.geometry[label].outer(row, radial_upper)
            jout *= dec(jproposal)
            entries = self.geometry[label].entries(point, row, label)
            def count(digits, event):
                self.counts[f"precision_{digits}_{event}_packets"] += 1
                self.counts[f"{label}_precision_{digits}_{event}_packets"] += 1
                if digits != 16 and event == "attempt":
                    self.counts["precision_retry_attempts"] += 1
            retry = run_precision_ladder(
                levels, lambda digits: self.packet_at_precision(label, entries, jout, digits),
                count=count)
            failure = None
            if retry.accepted_precision is None:
                self.counts["whole_packet_zeroed"] += 1
                self.counts[f"{label}_whole_packet_zeroed"] += 1
                values = [D(0)]*3
                if nb:
                    self.histogram_graphs = [[D(0)]*nb for _ in range(3)]
                failure = {"attempts": retry.failures, "zeroed": retry.failures[-1][1]}
            else:
                values = retry.value
            if retry.failures:
                self.counts["native_hp_retries"] += int(len(levels) > 1)
            if nb:
                self.histogram_values = [sum(values, D(0)) for values in zip(*self.histogram_graphs, strict=True)]
            return [float(v) for v in values], float(sum(values, D(0))), float(point[0]), failure

def transform(row, grid, label):
    out = np.asarray(row, dtype=float).copy()
    jacobian = 1.
    if grid is not None:
        for dim, edges in enumerate(grid[label]):
            if dim == 6:
                continue
            nb = len(edges)-1
            value = float(row[dim])*nb
            k = min(int(value), nb-1)
            width = edges[k+1]-edges[k]
            if width <= 0:
                raise ValueError("paired grid must have strictly positive widths")
            out[dim] = edges[k]+width*(value-k)
            jacobian *= nb*width
    return out, jacobian


def batch(evaluator, job):
    evaluator.counts.clear()
    start, cpu = time.monotonic(), time.process_time()
    sampler = qmc.Sobol(9, scramble=True, seed=job["seed"], bits=30)
    if job["start"]:
        sampler.fast_forward(job["start"])
    units = (sampler.random(job["n"])*2**30+.5)/2**30
    sums = np.zeros((2, 3), dtype=np.longdouble)
    totals = np.zeros(2, dtype=np.longdouble)
    squares = np.zeros(2, dtype=np.longdouble)
    hist = np.zeros((2, 8, NBINS))
    n_z_bins = max(0, len(evaluator.settings.z_histogram_edges)-1)
    bin_sums = np.zeros((2, 3, n_z_bins), dtype=np.longdouble)
    bin_totals = np.zeros((2, n_z_bins), dtype=np.longdouble)
    bin_hits = np.zeros((2, n_z_bins), dtype=np.int64)
    max_histogram_closure = 0.
    cost = np.zeros(2)
    largest, audits, failures = [], [], []
    for index, row in enumerate(units):
        ch = int(row[8] >= job["qqprob"])
        label = LABELS[ch]
        prob = job["qqprob"] if ch == 0 else 1-job["qqprob"]
        mapped, jac = transform(row, job.get("grid"), label)
        before = time.process_time()
        graphs, value, radius, failure = evaluator.evaluate(
            mapped, label, jac/prob, job.get("radial_upper"))
        cost[ch] += time.process_time()-before
        sums[ch] += graphs
        totals[ch] += value
        squares[ch] += value*value
        packet_bins = None
        if n_z_bins:
            packet_bins = np.asarray(evaluator.histogram_graphs, dtype=np.longdouble)
            packet_totals = np.asarray(evaluator.histogram_values, dtype=np.longdouble)
            bin_sums[ch] += packet_bins
            bin_totals[ch] += packet_totals
            # A packet may have several nonzero bins even if its total is zero.
            bin_hits[ch] += np.any(packet_bins != 0, axis=0)
            max_histogram_closure = max(max_histogram_closure, abs(float(packet_totals.sum())-value))
        evaluator.counts["packets"] += 1
        evaluator.counts[label+"_packets"] += 1
        if failure and len(failures) < 8:
            failures.append({"index": job["start"]+index, "label": label, **failure})
        if job.get("training"):
            for dim in range(8):
                hist[ch, dim, min(NBINS-1, int(mapped[dim]*NBINS))] += abs(value)
        record = dict(index=job["start"]+index, label=label, row=mapped.tolist(),
                      jproposal=jac/prob, value=value, graphs=graphs, r=radius,
                      replica=job["replica"], seed=job["seed"])
        if n_z_bins:
            record["histogram"] = packet_totals.astype(float).tolist()
        largest.append(record)
        if len(largest) > 12:
            largest = sorted(largest, key=lambda r: abs(r["value"]), reverse=True)[:4]
    largest.sort(key=lambda r: abs(r["value"]), reverse=True)
    # No automatic audit at an unlisted precision. The shared ladder above
    # is the entire production precision policy.
    return dict(n=job["n"], replica=job["replica"], start=job["start"],
                sum=np.asarray(sums, dtype=float).tolist(),
                total_sum=np.asarray(totals, dtype=float).tolist(),
                sum2=np.asarray(squares, dtype=float).tolist(),
                hist=hist.tolist() if job.get("training") else None,
                z_histogram_edges=list(evaluator.settings.z_histogram_edges),
                bin_sum=bin_sums.astype(float).tolist() if n_z_bins else None,
                bin_total_sum=bin_totals.astype(float).tolist() if n_z_bins else None,
                bin_hits=bin_hits.tolist() if n_z_bins else None,
                max_histogram_closure=max_histogram_closure,
                counts=dict(evaluator.counts), cost=cost.tolist(),
                audits=audits, failures=failures, largest=largest[:2],
                cpu_s=time.process_time()-cpu, wall_s=time.monotonic()-start)


def adapt(rows, oldprob):
    hist = np.sum([row["hist"] for row in rows], axis=0)
    n = sum(row["n"] for row in rows)
    ss = np.sum([row["sum2"] for row in rows], axis=0)
    costs = np.sum([row["cost"] for row in rows], axis=0)
    counts = np.sum([[row["counts"].get(label+"_packets", 0) for label in LABELS]
                     for row in rows], axis=0)
    second = np.asarray([oldprob, 1-oldprob])*ss/n
    percost = costs/np.maximum(counts, 1)
    score = np.sqrt(np.maximum(second, 1e-100)/np.maximum(percost, 1e-6))
    probability = float(np.clip(score[0]/sum(score), .075, .925))
    grid = {}
    for ch, label in enumerate(LABELS):
        grid[label] = []
        for dim in range(8):
            h = hist[ch, dim]
            edges = np.linspace(0, 1, NBINS+1)
            if dim not in (1, 6) and not (ch == 1 and dim == 7) and h.sum():
                smooth = (np.r_[h[0], h[:-1]]+2*h+np.r_[h[1:], h[-1]])/4
                density = .8*smooth/smooth.sum()+.2/NBINS
                cdf = np.r_[0., np.cumsum(density)]
                cdf[-1] = 1.
                edges = np.interp(np.linspace(0, 1, NBINS+1), cdf, edges)
                edges[0], edges[-1] = 0., 1.
            grid[label].append(edges.tolist())
    return grid, probability, dict(qq_probability=probability,
                                  selected=counts.tolist(),
                                  cpu_s_per_selected_packet=percost.tolist())


def replica_estimate(rows, auxiliary=None):
    """Keep qqbar/qg covariance; never infer QMC error from individual weights."""
    replicas = {}
    for row in rows:
        n, sums, graphs = replicas.setdefault(row["replica"], [
            0, np.zeros(2, dtype=np.longdouble), np.zeros((2, 3), dtype=np.longdouble)])
        replicas[row["replica"]][0] = n+row["n"]
        sums += row["total_sum"]
        graphs += row["sum"]
    if len(replicas) < 2 or len({v[0] for v in replicas.values()}) != 1:
        raise ValueError("estimates require at least two equal-sized complete replicas")
    vals = np.array([v[1]/v[0] for _, v in sorted(replicas.items())], dtype=float)
    graphs = np.array([v[2]/v[0] for _, v in sorted(replicas.items())], dtype=float)
    count = len(vals)
    means = vals.mean(axis=0)
    cov = np.cov(vals.T, ddof=1)/count
    hard_reps = vals.sum(axis=1)
    hard, error = float(hard_reps.mean()), float(hard_reps.std(ddof=1)/math.sqrt(count))
    aux = auxiliary or {}
    aux_value = sum(r["value"] for r in aux.values())
    aux_error = math.sqrt(sum(r["error"]**2 for r in aux.values()))
    result = dict(n=sum(v[0] for v in replicas.values()), replicas=count,
                  packets_per_replica=next(iter(replicas.values()))[0],
                  hard=hard, hard_error=error, scheme=aux_value, scheme_error=aux_error,
                  central_value=hard+aux_value, error=math.hypot(error, aux_error),
                  hard_replica_values=hard_reps.tolist(), hard_covariance=cov.tolist(),
                  channels={})
    result["relative_error"] = result["error"]/abs(result["central_value"]) if result["central_value"] else None
    for ch, label in enumerate(LABELS):
        a = aux.get(label, {"value": 0., "error": 0.})
        result["channels"][label] = dict(
            raw_hard=float(means[ch])*(-1 if ch == 0 else 1), hard=float(means[ch]),
            hard_error=float(math.sqrt(max(0., cov[ch, ch]))),
            scheme=a["value"], scheme_error=a["error"],
            central_value=float(means[ch])+a["value"],
            error=float(math.sqrt(max(0., cov[ch, ch])+a["error"]**2)),
            graphs_matched=graphs[:, ch].mean(axis=0).tolist(),
            graphs_errors=(graphs[:, ch].std(axis=0, ddof=1)/math.sqrt(count)).tolist())
    if rows[0].get("z_histogram_edges"):
        result["z_histogram"] = histogram_estimate(rows, auxiliary, result)
    return result


def numerical_summary(rows):
    counts = Counter()
    for row in rows:
        counts.update(row["counts"])
    for name in ("whole_packet_zeroed", "primary_hp_failures", "audit_failures",
                 "native_failed_graphs", "native_failures_salvaged_hp", "clipped_packets"):
        counts.setdefault(name, 0)
    audits = [a for row in rows for a in row["audits"]]
    differences = [row["audits"][0]["difference"] for row in rows
                   if row["audits"] and not row["audits"][0]["failure"]]
    bin_differences = [a["histogram_difference"] for a in audits
                       if not a["failure"] and "histogram_difference" in a]
    return dict(counts=dict(counts), max_audit_difference=max(
        (abs(a["difference"]) for a in audits if not a["failure"]), default=0.),
        max_bin_audit_difference=np.max(np.abs(bin_differences), axis=0).tolist() if bin_differences else None,
        audit_failures=sum(bool(a["failure"]) for a in audits),
        audit_difference_mean=float(np.mean(differences)) if differences else None,
        audit_difference_error=float(np.std(differences, ddof=1)/math.sqrt(len(differences)))
        if len(differences) > 1 else None,
        cpu_s=sum(row["cpu_s"] for row in rows),
        channel_cpu_s=np.sum([row["cost"] for row in rows], axis=0).tolist())


_EVALUATOR = None


def complete_replica_prefix(rows, replicas):
    """Exclude unfinished/straggling suffixes from provisional estimates."""
    by_replica = {replica: {} for replica in range(replicas)}
    for row in rows:
        by_replica[row["replica"]][row["start"]] = row
    prefix, result = 0, []
    while all(prefix in by_replica[replica] for replica in range(replicas)):
        block = [by_replica[replica][prefix] for replica in range(replicas)]
        if len({row["n"] for row in block}) != 1:
            raise ValueError("replica blocks have mismatched lengths")
        result.extend(block)
        prefix += block[0]["n"]
    return result


def _worker_init(specs, settings):
    from processes.dy.dy_physical_channels import DYPhysicalChannel
    global _EVALUATOR
    channels = []
    for label, builder, impl in specs:
        process = DY(**builder, clean=False, logger_level=50, skip_gl_worker_init=True)
        channels.append(DYPhysicalChannel(label, process, impl))
    _EVALUATOR = PacketEvaluator(channels, settings)


def _worker_job(job):
    if job["type"] == "batch":
        return batch(_EVALUATOR, job)
    if job["type"] == "aux":
        process = _EVALUATOR.channels[job["label"]].process
        start = time.monotonic()
        old_bin, old_power = process.dy_z_bin, process.dy_scheme_counterterm_sobol_power
        try:
            process.dy_scheme_counterterm_sobol_power = job["power"]
            if "z_bin" in job:
                process.dy_z_bin = tuple(job["z_bin"])
            result = process._add_dy_msbar_scheme_counterterm(
                IntegrationResult(0., 0.), job["seed"], "spherical", "real", workers=1)
        finally:
            process.dy_z_bin, process.dy_scheme_counterterm_sobol_power = old_bin, old_power
        return dict(label=job["label"], value=result.dy_auxiliary_central_value,
                    error=result.dy_auxiliary_error, points=result.dy_auxiliary_n_samples,
                    power=job["power"], bin_index=job.get("bin_index"), seed=job["seed"], wall_s=time.monotonic()-start,
                    details={k: v for k, v in vars(result).items() if k.startswith("dy_")})
    raise ValueError("unknown paired worker job")


class DYPairedSobolIntegrator:
    def __init__(self, channels, *, settings, n_cores=1, parameterisation="spherical", phase="real"):
        self.channels = sorted(channels, key=lambda ch: LABELS.index(ch.label) if ch.label in LABELS else 9)
        self.settings = settings.validate()
        self.n_cores = n_cores
        if parameterisation != "spherical" or phase != "real":
            raise pygloopException("dy-paired-sobol supports spherical/real integration only")
        if n_cores < 1 or n_cores > len(os.sched_getaffinity(0)):
            raise pygloopException("paired worker count exceeds available CPU affinity")
        self.validate_channels()
        self.directory = Path(settings.output_directory) if settings.output_directory else None
        self.state = dict(mode="dy-paired-sobol", units="GeV^-2", settings=asdict(settings),
                          workers=n_cores, pid=os.getpid(), status="created", learning=[],
                          checkpoints=[], auxiliary={}, cpu_affinity=sorted(os.sched_getaffinity(0)))

    def validate_channels(self):
        if [ch.label for ch in self.channels] != list(LABELS):
            raise pygloopException("dy-paired-sobol requires exactly qqbar and qg channels")
        reference = None
        for channel in self.channels:
            p, impl = channel.process, channel.integrand_implementation
            if self.settings.z_histogram_edges and p.dy_z_bin not in (None, (0., 1.), [0., 1.]):
                raise pygloopException("output histogram requires inclusive channel cards; keep individual-bin mode separate")
            if (p.process_name.lower() != "dy" or p.n_loops != 1 or not p.integrate_beams
                or not p.dy_physical_normalisation or not p.dy_integrated_leptonic_phase_space
                or not p.dy_coupling_normalisation_applied or not p.dy_msbar_scheme_counterterm
                or p.dy_pdf_set is None):
                raise pygloopException("paired mode requires one-loop hadronic DY with physical/leptonic normalization, PDFs and scheme conversion")
            if (impl.get("dy_large_weight_clip") is not None or p.dy_scheme_counterterm_clip is not None):
                raise pygloopException("paired mode requires unclipped hard and auxiliary cards")
            if (impl.get("dy_beam_parameterisation", "x1_x2") != "x1_x2"
                or impl.get("dy_integrated_uv_ct_filter", "all") != "all"
                or impl.get("dy_graph_weights") is not None
                or p._rotation_check_count(impl) < 1
                or impl.get("dy_evaluation_mode", "compiled") == "arb"):
                raise pygloopException("paired mode requires all unweighted graphs, x1_x2 beams, rotations and a native-first stability ladder")
            for digits in p._dy_precision_ladder(impl)[1:]:
                p.compiled_bundle.require_fallback_supported(digits)
            names = p.graph_channel_names(impl)
            if len(names) != 3:
                raise pygloopException("paired mode requires the three-graph DY bundle")
            if channel.label == "qqbar":
                pair = p._grouped_soft_pair(impl)
                if pair != (1, 2, "invert") or set(channel.graph_channel_indices) != {0, -2}:
                    raise pygloopException("paired qqbar requires graph_1/graph_2 grouped with invert, and graph_0")
                if any(p._soft_mirror_routing(impl, graph) is None for graph in (1, 2)):
                    raise pygloopException("paired qqbar requires both soft mirrors")
            elif tuple(channel.graph_channel_indices) != (0, 1, 2):
                raise pygloopException("paired qg requires all three graphs")
            runtime = p.compiled_bundle.resolve_runtime_parameters(impl.get("dy_runtime_parameters") or {})
            if (impl.get("dy_runtime_parameters") or {}).get("pi") is not None:
                raise pygloopException("paired mode uses mathematical pi, not a user pi override")
            signature = (p.e_cm, p.dy_q_min, p.dy_q_max, p.dy_z_bin,
                         p.dy_pdf_set, p.dy_pdf_member, p.dy_muf_sq,
                         tuple(sorted(runtime.items())), p._dy_precision_ladder(impl))
            if reference is not None and signature != reference:
                raise pygloopException("paired channel cards must share energy, cuts, PDFs and runtime scales/couplings")
            reference = signature
            PairedGeometry(p.e_cm, p.dy_q_min, runtime["lambda_sq"],
                           z_max=(p.dy_z_bin or (0., 1.))[1])
        forbidden = [key for key in os.environ if key.startswith("PYGLOOP_DIAGNOSTIC_")
                     and os.environ[key]]
        if forbidden:
            raise pygloopException("unset diagnostic environment overrides for paired production: "+", ".join(forbidden))

    def save(self, event):
        self.state["event"] = event
        self.state["updated_unix"] = time.time()
        if self.directory:
            temp = self.directory/"status.json.tmp"
            temp.write_text(json.dumps(self.state, indent=2, allow_nan=False), encoding="utf-8")
            temp.replace(self.directory/"status.json")
        brief = {key: self.state[key] for key in ("event", "status", "production_packets") if key in self.state}
        if self.state["checkpoints"]:
            last = self.state["checkpoints"][-1]
            brief.update(central_pb=last["central_value"]*PB, error_pb=last["error"]*PB)
        print("DY_PAIRED "+json.dumps(brief, allow_nan=False), flush=True)

    def jobs(self, seed, n, *, start=0, grid=None, probability=.75, training=False, radial_upper=None):
        chunk = min(128 if training or radial_upper is not None else self.settings.audit_block_size, n)
        return [dict(type="batch", seed=seed+rep, replica=rep, start=offset,
                     n=min(chunk, start+n-offset), grid=grid, qqprob=probability,
                     training=training, radial_upper=radial_upper)
                for offset in range(start, start+n, chunk)
                for rep in range(self.settings.replicas)]

    def run_jobs(self, pool, jobs, stage):
        self.state["status"] = stage
        self.state["stage_jobs_done"] = 0
        self.state["stage_jobs_total"] = len(jobs)
        self.save("stage_start")
        rows, last = [], time.monotonic()
        futures = {pool.submit(_worker_job, job): job for job in jobs}
        for future in as_completed(futures):
            rows.append(future.result())
            if time.monotonic()-last > 30:
                self.state["stage_jobs_done"] = len(rows)
                self.state["stage_jobs_total"] = len(jobs)
                if stage == "production":
                    prefix = complete_replica_prefix(self.production_rows+rows, self.settings.replicas)
                    if prefix:
                        self.state["partial"] = replica_estimate(prefix, self.state["auxiliary"])
                        self.state["partial_diagnostics"] = numerical_summary(prefix)
                    self.state["production_packets"] = sum(row["n"] for row in self.production_rows+rows)
                self.save("progress")
                last = time.monotonic()
        return sorted(rows, key=lambda r: (r.get("replica", -1), r.get("start", 0)))

    def integrate(self, *, seed=1337):
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=False)
            if getattr(self, "effective_card_text", None):
                (self.directory/"effective.toml").write_text(self.effective_card_text, encoding="utf-8")
            self.state["source_sha256"] = {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("dy_paired_sobol.py", "dy_paired_settings.py", "dy_precision.py", "dy.py", "dy_stability.py", "dy_evaluators.py", "dy_histogram.py")}
            for channel in self.channels:
                if channel.source_card:
                    (self.directory/(channel.label+".source.toml")).write_text(
                        Path(channel.source_card).read_text(), encoding="utf-8")
                    from processes.dy.dy_card import load_dy_card
                    effective = load_dy_card(channel.source_card).merge(
                        "integrate", {"dy_precision_ladder": channel.process._dy_precision_ladder(channel.integrand_implementation)},
                        explicit_destinations=("dy_precision_ladder",))
                    effective = effective.with_runtime_parameters(channel.integrand_implementation.get("dy_runtime_parameters") or {})
                    (self.directory/(channel.label+".effective.toml")).write_text(effective.to_toml(), encoding="utf-8")
        started = time.monotonic()
        # Separate, recorded scramble families for learning, production and aux.
        settings = self.settings
        self.state["seed_schedule"] = dict(
            base=seed, training=[seed+i*settings.replicas for i in range(len(settings.learning_points))],
            production=[seed+100000+i for i in range(settings.replicas)],
            auxiliary={label: seed+200000+i*100 for i, label in enumerate(LABELS)})
        self.state["precision_ladder"] = list(self.channels[0].process._dy_precision_ladder(self.channels[0].integrand_implementation))
        self.state["precision_failure_action"] = "zero_complete_sample"
        self.state["automatic_hp_audits"] = False
        specs = [(ch.label, ch.process.builder_inputs(), dict(ch.integrand_implementation))
                 for ch in self.channels]
        production, grid, probability = [], None, .75
        self.production_rows = production
        try:
            with ProcessPoolExecutor(max_workers=self.n_cores,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_worker_init, initargs=(specs, settings)) as pool:
                def auxiliary_jobs(powers):
                    if settings.z_histogram_edges:
                        return [dict(type="aux", label=label, power=powers[label][b],
                                     z_bin=[low, high], bin_index=b,
                                     seed=seed+400000+i*10000+b*100)
                                for i, label in enumerate(LABELS)
                                for b, (low, high) in enumerate(zip(settings.z_histogram_edges, settings.z_histogram_edges[1:]))]
                    return [dict(type="aux", label=label, power=power,
                                 seed=seed+200000+i*100) for i, (label, power) in enumerate(zip(LABELS, powers))]

                def combine_auxiliary(rows):
                    if not settings.z_histogram_edges:
                        return {r["label"]: r for r in rows}
                    output = {}
                    for label in LABELS:
                        bins = sorted([r for r in rows if r["label"] == label], key=lambda r: r["bin_index"])
                        output[label] = dict(value=math.fsum(r["value"] for r in bins),
                                             error=math.sqrt(math.fsum(r["error"]**2 for r in bins)),
                                             points=sum(r["points"] for r in bins), bins=bins)
                    return output

                powers = ([ch.process.dy_scheme_counterterm_sobol_power for ch in self.channels]
                          if not settings.z_histogram_edges else
                          {ch.label: [ch.process.dy_scheme_counterterm_sobol_power]*(len(settings.z_histogram_edges)-1) for ch in self.channels})
                self.state["auxiliary_seed_jobs"] = auxiliary_jobs(powers)
                auxiliary = combine_auxiliary(self.run_jobs(pool, auxiliary_jobs(powers), "auxiliary"))
                self.state["auxiliary"] = auxiliary
                for iteration, n in enumerate(settings.learning_points):
                    rows = self.run_jobs(pool, self.jobs(
                        seed+iteration*settings.replicas, n, grid=grid, probability=probability, training=True),
                        f"learning_{iteration+1}")
                    grid, probability, detail = adapt(rows, probability)
                    self.state["learning"].append(dict(
                        packets=settings.replicas*n, adaptation=detail, diagnostics=numerical_summary(rows)))
                self.state.update(frozen_grid=grid, qq_probability=probability, grid_frozen=True,
                                  learning_excluded=True, learning_packets=settings.replicas*sum(settings.learning_points))
                self.save("grid_frozen")
                previous = None
                current = settings.production_points//2
                start = 0
                while True:
                    rows = self.run_jobs(pool, self.jobs(
                        seed+100000, current-start, start=start, grid=grid, probability=probability), "production")
                    production.extend(rows)
                    estimate = replica_estimate(production, auxiliary)
                    while not auxiliary_target_met(estimate, settings.auxiliary_error_fraction):
                        if settings.z_histogram_edges:
                            jobs = []
                            for job in auxiliary_jobs(powers):
                                label, b = job["label"], job["bin_index"]
                                bin_estimate = estimate["z_histogram"]["bins"][b]
                                # No observed hard support is not evidence for a zero error.
                                # Defer refinement of such bins until they have hard samples.
                                if (bin_estimate["hard_error"] > 0 and job["power"] < 24
                                    and auxiliary[label]["bins"][b]["error"] > settings.auxiliary_error_fraction*bin_estimate["hard_error"]/math.sqrt(2)):
                                    powers[label][b] += 1
                                    jobs.append(dict(job, power=powers[label][b]))
                            if not jobs:
                                break
                            updates = self.run_jobs(pool, jobs, "auxiliary_refinement")
                            existing = {(label, row["bin_index"]): row for label, data in auxiliary.items() for row in data["bins"]}
                            existing.update({(row["label"], row["bin_index"]): row for row in updates})
                            auxiliary = combine_auxiliary(list(existing.values()))
                        else:
                            if max(powers) >= 24:
                                break
                            powers = [power+1 for power in powers]
                            auxiliary = combine_auxiliary(self.run_jobs(pool, auxiliary_jobs(powers), "auxiliary_refinement"))
                        self.state["auxiliary"] = auxiliary
                        estimate = replica_estimate(production, auxiliary)
                    estimate["auxiliary_target_met"] = auxiliary_target_met(estimate, settings.auxiliary_error_fraction)
                    if settings.z_histogram_edges:
                        for bin_row in estimate["z_histogram"]["bins"]:
                            bin_row["auxiliary_target_met"] = bin_row["hard_error"] > 0 and bin_row["scheme_error"] <= settings.auxiliary_error_fraction*bin_row["hard_error"]
                    consistency = None
                    if previous is not None:
                        fresh = replica_estimate(rows)
                        differences = np.array(fresh["hard_replica_values"])-previous["hard_replica_values"]
                        shift = float(differences.mean())
                        shift_error = float(differences.std(ddof=1)/math.sqrt(settings.replicas))
                        consistency = dict(shift=shift, error=shift_error,
                                           consistent=abs(shift) <= 3*shift_error)
                        if settings.z_histogram_edges:
                            differences = (np.array(fresh["z_histogram"]["hard_replica_values"])
                                           - previous["z_histogram"]["hard_replica_values"])
                            shifts = differences.mean(axis=0)
                            errors = differences.std(axis=0, ddof=1)/math.sqrt(settings.replicas)
                            consistency["bins"] = [dict(shift=float(s), error=float(e), consistent=bool(abs(s) <= 3*e))
                                                   for s, e in zip(shifts, errors, strict=True)]
                            consistency["consistent"] &= all(b["consistent"] for b in consistency["bins"])
                    estimate["new_half_comparison"] = consistency
                    self.state["checkpoints"].append(estimate)
                    self.state["production_packets"] = estimate["n"]
                    self.state["diagnostics"] = numerical_summary(production)
                    diagnostics = self.state["diagnostics"]
                    estimate["numerical_checks_passed"] = (
                        diagnostics["counts"]["whole_packet_zeroed"] == 0)
                    self.state["largest_packets"] = sorted(
                        [r for row in production for r in row["largest"]],
                        key=lambda r: abs(r["value"]), reverse=True)[:8]
                    self.state["wall_s"] = time.monotonic()-started
                    met = precision_target_met(estimate, settings.target_relative_error) and estimate["auxiliary_target_met"]
                    previous_met = previous is not None and precision_target_met(previous, settings.target_relative_error)
                    confirmed = met and previous_met and consistency["consistent"]
                    self.save("checkpoint")
                    if current >= settings.production_points and (confirmed or current >= settings.max_points):
                        self.state["precision_target_met"] = bool(confirmed)
                        self.state["stop_reason"] = "precision_and_doubling_confirmation" if confirmed else "max_points"
                        break
                    previous = estimate
                    start, current = current, min(2*current, settings.max_points)
            self.state.update(status="completed", wall_s=time.monotonic()-started)
            result = self.build_result(production)
            self.state["integration_result"] = vars(result)
            self.save("completed")
            return result
        except BaseException:
            self.state["status"] = "interrupted_or_error"
            self.save("not_completed")
            raise

    def build_result(self, rows):
        estimate = self.state["checkpoints"][-1]
        result = DYPhysicalChannelIntegrationResult()
        result.central_value, result.error = estimate["central_value"], estimate["error"]
        result.hard_central_value, result.hard_error = estimate["hard"], estimate["hard_error"]
        result.production_samples = estimate["n"]
        result.n_samples = estimate["n"]
        result.elapsed_time = self.state["diagnostics"]["cpu_s"]
        result.warmup_samples = self.state["learning_packets"]
        result.warmup_iterations = len(self.settings.learning_points)
        result.production_iterations = len(self.state["checkpoints"])
        result.evaluation_backend = "shared-card-precision-ladder"
        result.precision_ladder = self.state["precision_ladder"]
        result.sampling_topology = "paired-angular-frozen-sobol"
        result.worker_allocation = {"shared": self.n_cores}
        result.outer_discrete_channel_count = 2
        result.joint_discrete_channel_count = 6
        result.batch_size = self.settings.audit_block_size
        result.target_relative_precision = self.settings.target_relative_error
        result.achieved_relative_precision = estimate["relative_error"]
        result.precision_target_met = self.state["precision_target_met"]
        result.precision_stop_reason = self.state["stop_reason"]
        result.precision_history = self.state["checkpoints"]
        result.preview_auxiliary_central_value = estimate["scheme"]
        result.preview_auxiliary_error = estimate["scheme_error"]
        counts = self.state["diagnostics"]["counts"]
        result.unstable_count = counts.get("whole_packet_zeroed", 0)
        result.stability_hp_retry_count = counts.get("native_hp_retries", 0)
        for i, label in enumerate(LABELS):
            row = dict(estimate["channels"][label])
            row["selected_samples"] = counts.get(label+"_packets", 0)
            result.physical_channel_estimates[label] = row
            result.hard_channel_covariance[label] = dict(zip(LABELS, estimate["hard_covariance"][i]))
            result.physical_channel_covariance[label] = {
                other: value+(row["scheme_error"]**2 if other == label else 0.)
                for other, value in result.hard_channel_covariance[label].items()}
            result.graph_channel_estimates[label] = {
                f"graph_{graph}": dict(central_value=value, error=row["graphs_errors"][graph],
                                      raw_central_value=value*(-1 if label == "qqbar" else 1),
                                      selected_samples=row["selected_samples"])
                for graph, value in enumerate(row["graphs_matched"])}
        result.paired_sobol_diagnostics = self.state["diagnostics"]
        if "z_histogram" in estimate:
            result.z_histogram = estimate["z_histogram"]
            result.inclusive_relative_precision = estimate["relative_error"]
            result.achieved_relative_precision = estimate["z_histogram"]["max_relative_error"]
            result.precision_target_scope = "all_histogram_bins"
        return result
