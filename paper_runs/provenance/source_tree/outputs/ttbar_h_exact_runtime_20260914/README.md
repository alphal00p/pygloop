# Exact-runtime h normalization campaign

All five bundles regenerated successfully. The original-point, ordinary-point,
and smoke validations passed. The single approved 1500 GeV 1B production run
started at 2026-09-14 21:08:06 UTC, with controller PID 442239 and owned worker
process-group ID 442244. Follow `runs/production/state.json` and `latest.json`
for its current state.

## Normalization and evaluator contract

For positive t,

    integral exp(-t^2 - 1/t^2) dt from 0 to infinity = sqrt(pi) exp(-2) / 2
    H = 2 exp(2) / sqrt(pi)

`dy_runtime_parameters.h_normalisation(digits)` evaluates this exact expression
directly with Symbolica, at the requested decimal precision plus 20 guard
digits. The result is cached separately by precision. Only the native path
converts it to float; HP obtains a fresh Decimal value from the exact formula.

Generated terms contain the shared input `dy_h_normalisation`, not a folded
floating coefficient or a large rational approximation. Registry version 2
records the extra derived input. It is not a user-overridable physics parameter.
Version 1 bundles keep their original ABI and remain loadable, but do not gain
the correction without regeneration.

This avoids two independently established defects in the installed Symbolica:

- Saved large negative rational coefficients can lose their sign.
- Saved HP evaluators with embedded built-in pi fail at precision changes.

Every generated saved evaluator now passes an exact instruction/constant
save-load integrity comparison before being written. No library patch or
diagnostic coefficient correction is applied.

## Validation

- `normalization_audit.json`: independent 260-digit quadrature and formula check.
- `focused_initial.log`: 113 focused checks passed, 3 skipped.
- Six additional run-observer checks passed, including physical importance
  weights and capture of rejected or clipped HP points.
- `focused_final.log`: preproduction combined suite, 119 passed and 3 skipped.
- `focused_handoff.log`: combined suite including replay-queue classification,
  129 passed and 3 skipped. The queue's 10 additional checks distinguish stable
  clipping, successful rescue, inconclusive replay, and a confirmed discrepancy.
- `validate_replay.py`: replay the original 2000 GeV GL105 and GL021 samples,
  check signed prefactors and graph/cut inventory, then precision and radial scans.
- `validate_ordinary.py`: ordinary points in every production graph channel and
  both Born bundles, comparing production, HP, and original bundles.
  All 92 graph-channel points and both Born checks passed.
- The user explicitly waived `tests/test_dy_graph_cut_sum_references.py`.
  It is not run, and its reference fixture is not changed.

Prior failed large-rational artifacts remain under `../ttbar_h_rational_20260914`.
Original run and replay artifacts remain under
`../ttbar_cm_e_surface_campaign_20260909`. The old 2000 GeV job finished; the old
1500 GeV controller and its worker process group were stopped earlier.

## Approved production

The original exceptional points now have the following fully weighted values,
using their original importance densities (not densities from a new run):

| Source | Original weight | Corrected weight at 192 digits |
| --- | ---: | ---: |
| GL105, iteration 59 | +0.06724012999420013 | -1.2216095771986297e-9 |
| GL021, iteration 328 | -0.07871950882250185 | +4.6196741023326685e-8 |

Both pass the normal saved-JIT/scalar pipeline at its configured precision,
without clipping or rejection. Forced 64/80/128/192-digit evaluations agree at
the precision of the returned float. Radial factors 0.1, 1, and 10 remain bounded.
The saved overall coefficients are exactly (-9/32, -1/2) for GL105 and
(9/32, -1/2) for GL021, multiplying the shared normalization input.

Thus the old exceptional weights were artifacts of broken cancellation, not
the true weights of these samples. This does not by itself certify every
sample in a new large run.

The smoke run completed 600k warmup and 400k retained samples, without clipping
or unresolved HP failures. Its large-weight alerts were replayed without
clipping through 192 digits (`probe_smoke_iteration1.log` and
`probe_smoke_iteration2.log`). GL093, GL047, and GL073 all reproduced the captured
values and agreed at every tested precision. The largest GL047 sample is a
localized peak: its three terms are approximately +8.825e-15, +8.825e-15, and
-1.448e-14, with a net +3.171e-15. This is modest cancellation, not a catastrophic
UV subtraction; its value falls sharply under the radial scan. At 400k samples
it contributes 12.18% of cumulative sum(w^2), but the variance still decreased
between the two smoke iterations. These observations do not justify stopping
for a stability-pipeline defect.

After original-point validation, ordinary checks, and a smoke run pass:

- 1500 GeV, exactly 500 x 2M production samples; 3 x 2M warmup samples excluded.
- Seed 909151001, at most 248 cores, existing physics and PDF settings preserved.
- muUV=750 GeV, mt=173 GeV, PDF4LHC21_40 member 0.
- No queued second replica and no automatic restart.

`monitor_run.py` captures the 64 largest fully importance-weighted contributions
in each iteration. For channel c, it uses

    w = F * physical_sampling_factor * sample.weights[0] / probability(c)

The first inner weight already contains graph and continuous importance
sampling; multiplying the remaining entries again would be incorrect.

A run-local live-report adapter prevents a second application of the physical
channel factors. It changes reporting only, not sampled integrands, the
estimator, adaptation, or cuts. Captured raw moments are checked against the
reported mean and hard error at every production iteration.

Alerts flag a variance rise of at least 25%, a single weight contributing at
least 5% of cumulative sum(w^2), unresolved HP failures, or clipping. HP failures
and preclip coordinates are also recorded in workers. `probe_points.py` can
replay captured points without clipping through 192 digits and scan the radial
tail. A serious confirmed stability/HP defect requires stopping both controller
and owned workers, preserving evidence, and reporting before any further fix.

## Ongoing diagnostics

`monitoring_findings.md` records the first two substantial variance jumps,
their correct centered variance attribution, the term/support explanation,
and an independently checked 28-digit graph-wrapper limitation that did not
cause either spike. The new weights are precision-stable peaks of the current
generated integrand, not repeats of the original normalization failure.

`watch_replays.py` is following the single approved run. It automatically
replays alarm points and every captured HP/clipping diagnostic, and requests
the approved stop only for a reproduced accepted value that disagrees with a
converged higher-precision ladder. It verifies process identities first and
cannot launch another run. Its current coverage and clipping counterfactual
are in `replay_queue/progress.json`; its own status is in
`replay_queue/watcher_state.json`. The production estimator and configured
clipping remain untouched. A counterfactual restoration is not the run result.

No matching full-channel 1500 GeV MG5 reference has yet been identified. The
available qqbar-only 1500 GeV result is not a valid comparison for this total.
