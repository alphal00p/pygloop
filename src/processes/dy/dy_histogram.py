"""Cut-level output observers. These never define a sampling probability."""
from bisect import bisect_right
from decimal import Decimal
import math

import numpy as np


def validate_edges(edges):
    edges = tuple(float(x) for x in edges)
    if (len(edges) < 2 or edges[0] != 0. or edges[-1] != 1.
            or any(not math.isfinite(x) for x in edges)
            or any(b <= a for a, b in zip(edges, edges[1:]))):
        raise ValueError("z histogram edges must increase strictly from 0 to 1")
    return edges


class CutHistogram:
    """One evaluation's raw terms, or one accepted orbit's weighted bins.

    Interior bins are [low, high); the final bin includes z=1. Only outer
    endpoints are snapped, using the same 32-double-ULP allowance as DY cuts.
    Accumulation retains the evaluator's numeric type (including Decimal).
    """
    def __init__(self, edges):
        self.edges = tuple(Decimal(str(x)) for x in validate_edges(edges))
        self.bins = [0] * (len(self.edges)-1)
        self.accepted = None
        self.stage = None

    def fresh(self):
        return CutHistogram(self.edges)

    def add(self, z, value):
        z = z if isinstance(z, Decimal) else Decimal.from_float(float(z))
        if not z.is_finite():
            raise ArithmeticError("nonfinite physical z in histogram")
        tolerance = Decimal.from_float(32*math.ulp(1.))
        if z < 0 and abs(z) <= tolerance:
            z = Decimal(0)
        if z > 1 and z-1 <= tolerance:
            z = Decimal(1)
        if z < 0 or z > 1:
            raise ArithmeticError("accepted cut lies outside histogram support")
        index = min(bisect_right(self.edges, z)-1, len(self.bins)-1)
        self.bins[index] += value

    def accept(self, values, stage="prebeam"):
        self.accepted = list(values)
        self.stage = stage

    def reset(self):
        self.accepted = None
        self.stage = None


def histogram_estimate(rows, auxiliary, inclusive):
    """Covariance of independent scrambled-replica means, not of weights.

    Auxiliary bin/channel jobs use independent scramble families; their
    covariance is diagonal. Hard bins/channels retain their full covariance.
    """
    edges = rows[0]["z_histogram_edges"]
    nb = len(edges)-1
    replicas = {}
    for row in rows:
        item = replicas.setdefault(row["replica"], [0, np.zeros((2, 3, nb), dtype=np.longdouble), np.zeros((2, nb), dtype=int), np.zeros((2, nb), dtype=np.longdouble)])
        item[0] += row["n"]
        item[1] += row["bin_sum"]
        item[2] += row["bin_hits"]
        item[3] += row["bin_total_sum"]
    items = [v for _, v in sorted(replicas.items())]
    graphs = np.array([v[1]/v[0] for v in items], dtype=float)
    # Graphs must already have been summed in the packet's Decimal context.
    # Reconstructing this from rounded graph observers can lose soft cancellations.
    channels = np.array([v[3]/v[0] for v in items], dtype=float)
    totals = channels.sum(axis=1)
    hits = np.array([v[2] for v in items])
    nrep = len(items)
    hard_cov = np.atleast_2d(np.cov(totals, rowvar=False, ddof=1))/nrep
    channel_cov = np.cov(channels.reshape(nrep, -1), rowvar=False, ddof=1)/nrep
    scheme = np.zeros((2, nb))
    scheme_variance = np.zeros((2, nb))
    labels = ("qqbar", "qg")
    for ch, label in enumerate(labels):
        for b, row in enumerate((auxiliary or {}).get(label, {}).get("bins", [])):
            scheme[ch, b], scheme_variance[ch, b] = row["value"], row["error"]**2
    total_cov = hard_cov + np.diag(scheme_variance.sum(axis=0))
    total_channel_cov = channel_cov + np.diag(scheme_variance.reshape(-1))
    bins = []
    for b in range(nb):
        hard = float(totals[:, b].mean())
        scheme_value = float(scheme[:, b].sum())
        value = hard+scheme_value
        error = math.sqrt(max(0., total_cov[b, b]))
        hard_error = math.sqrt(max(0., hard_cov[b, b]))
        active = int(np.count_nonzero(hits[:, :, b].sum(axis=1)))
        supported = active == nrep and hits[:, :, b].sum() >= 2*nrep and hard_error > 0.
        entry = dict(low=edges[b], high=edges[b+1], hard=hard, hard_error=hard_error,
                     scheme=scheme_value, scheme_error=math.sqrt(float(scheme_variance[:, b].sum())),
                     central_value=value, error=error,
                     relative_error=error/abs(value) if value and supported else None,
                     observed_nonzero_packets=int(hits[:, :, b].sum()), active_replicas=active,
                     sampling_status="estimated" if supported else "undersampled", channels={})
        for ch, label in enumerate(labels):
            index = ch*nb+b
            sign = -1 if ch == 0 else 1
            hard_ch = float(channels[:, ch, b].mean())
            graph_means = graphs[:, ch, :, b].mean(axis=0)
            graph_errors = graphs[:, ch, :, b].std(axis=0, ddof=1)/math.sqrt(nrep)
            entry["channels"][label] = dict(
                raw_hard=sign*hard_ch, hard=hard_ch,
                hard_error=math.sqrt(max(0., channel_cov[index, index])),
                scheme=float(scheme[ch, b]), scheme_error=math.sqrt(float(scheme_variance[ch, b])),
                central_value=hard_ch+float(scheme[ch, b]),
                error=math.sqrt(max(0., total_channel_cov[index, index])),
                observed_nonzero_packets=int(hits[:, ch, b].sum()),
                graphs_matched=graph_means.tolist(), graphs_raw=(sign*graph_means).tolist(),
                graphs_errors=graph_errors.tolist())
        bins.append(entry)
    histogram_sum = math.fsum(b["central_value"] for b in bins)
    histogram_error = math.sqrt(max(0., float(total_cov.sum())))
    return dict(edges=edges, bins=bins, hard_covariance=hard_cov.tolist(),
                max_relative_error=max(b["relative_error"] for b in bins) if all(b["relative_error"] is not None for b in bins) else None,
                covariance=total_cov.tolist(), channel_bin_order=[f"{label}:{b}" for label in labels for b in range(nb)],
                hard_channel_bin_covariance=channel_cov.tolist(), channel_bin_covariance=total_channel_cov.tolist(),
                hard_replica_values=totals.tolist(),
                auxiliary_independent_bins=True,
                sum=histogram_sum, sum_error=histogram_error,
                inclusive_value_difference=histogram_sum-inclusive["central_value"],
                inclusive_error_difference=histogram_error-inclusive["error"],
                max_packet_closure=max(row["max_histogram_closure"] for row in rows))


def auxiliary_target_met(estimate, fraction):
    if "z_histogram" in estimate:
        return all(row["hard_error"] > 0 and row["scheme_error"] <= fraction*row["hard_error"]
                   for row in estimate["z_histogram"]["bins"])
    return estimate["scheme_error"] <= fraction*estimate["hard_error"]


def precision_target_met(estimate, target):
    if "z_histogram" in estimate:
        return all(row["relative_error"] is not None and row["relative_error"] <= target
                   for row in estimate["z_histogram"]["bins"])
    return estimate["relative_error"] is not None and estimate["relative_error"] <= target
