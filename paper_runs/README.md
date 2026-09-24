# Paper runs

This archive preserves the original user-confirmed selection: six September 17
Drell–Yan histogram runs, eight September 1–2 single partonic ttbar runs, the four
September 9 400 GeV gg replicas, and four historical hadronic ttbar runs. It also
contains the new 1000, 1500 and 2000 GeV G-hybrid campaigns, with four independent
50M production replicas each. There are 34 individual run records and 17 reported
results. Historical runs are retained separately, not replaced or recombined.

Original repository: `/home/zeno/DY/pygloop_6c15087`.
Archive created: 2026-09-21T21:23:17.647636+00:00.
G-hybrid metadata added: 2026-09-24.

## Results

These are saved finite **pure NLO contributions**, including the run-configured
scheme conversion and decoupling. LO is not added. [results.json](results.json)
retains uncertainties and full-precision values in both GeV^-2 and pb.
The table lists central values only.

| Calculation | sqrt(s) [GeV] | Channel | Runs | Production points | Central value [pb] |
|---|---:|---|---:|---:|---:|
| Drell–Yan hadronic | 1000 | inclusive | 6 | 25,165,824 | 5.89298295787e-06 |
| ttbar partonic | 400 | qqbar | 1 | 100,000,000 | -3.36796605293 |
| ttbar partonic | 400 | qg | 1 | 100,000,000 | -0.157786426518 |
| ttbar partonic | 400 | gg | 4 | 400,000,000 | 3.48754009398 |
| ttbar partonic | 1000 | qqbar | 1 | 100,000,000 | -1.48592265574 |
| ttbar partonic | 1000 | qg | 1 | 100,000,000 | 2.62500330803 |
| ttbar partonic | 1000 | gg | 1 | 100,000,000 | 9.00162549823 |
| ttbar partonic | 2000 | qqbar | 1 | 100,000,000 | 0.00659125302647 |
| ttbar partonic | 2000 | qg | 1 | 100,000,000 | 5.53673392157 |
| ttbar partonic | 2000 | gg | 1 | 100,000,000 | 23.6548336809 |
| ttbar hadronic | 400 | inclusive | 1 | 100,000,000 | 1.05529412904e-09 |
| ttbar hadronic | 1000 | inclusive | 1 | 1,000,000,000 | 0.00440750603537 |
| ttbar hadronic | 1500 | inclusive | 1 | 1,000,000,000 | 0.0600265217283 |
| ttbar hadronic | 2000 | inclusive | 1 | 1,000,000,000 | 0.252112191823 |
| ttbar hadronic, G-hybrid | 1000 | inclusive | 4 | 200,000,000 | 0.00440726814979 |
| ttbar hadronic, G-hybrid | 1500 | inclusive | 4 | 200,000,000 | 0.0605739270296 |
| ttbar hadronic, G-hybrid | 2000 | inclusive | 4 | 200,000,000 | 0.253986045234 |

The six DY runs contain 6 x 4,194,304 = 25,165,824 production packets.
Each outer run contains 16 internal scrambled Sobol replicas. Their inclusive
totals are combined; separate low-bin refinements do not enter. DY is at
sqrt(s)=1000 GeV, Q_min=300 GeV and full z interval [0,1]. Histogram bins share
packets, so bin sample counts are not additive.

Historical replica totals use inverse-variance weights from saved final uncertainties.
The four gg replicas reproduce their original saved combined result.
Weights, formulas and consistency statistics are in
[drell_yan/combined.json](drell_yan/combined.json) and
[ttbar_partonic/s400/gg/combined.json](ttbar_partonic/s400/gg/combined.json).

## Historical hadronic selection and sample counts

- 400 GeV: original September 1 campaign 100M result in
  results/production_validation/production/ttbar/hadronic/s400/combined.json.
- 1000 GeV: CM E-surface result completed September 10, seed 909101000;
  1B production and 30M warmup samples.
- 1500 GeV: exact-runtime h result completed September 15, seed 909151001;
  1B production and 6M warmup samples.
- 2000 GeV: the run completed September 8, seed 906102000; 1B production
  and 6M warmup samples, with final value 0.2521121918234394 pb. The initially
  selected September 14 restart has been superseded after the user identified
  the expected value. It is retained only under provenance/superseded.

The 1B result counters include warmup: 1,030,000,000 at 1000 GeV and
1,006,000,000 at 1500/2000 GeV. Normalized metadata separates these counts.
The historical selection contains 4,325,165,824 production and 42,688,128
warmup/learning points. Including the three new G-hybrid campaigns, the archive
contains 4,925,165,824 production and 78,688,128 warmup/learning points, before
separately recorded auxiliary work.

## New 200M G-hybrid hadronic campaigns

The new campaigns are under
[1000 GeV](ttbar_hadronic/s1000/g_hybrid_200m/metadata.json),
[1500 GeV](ttbar_hadronic/s1500/g_hybrid_200m/metadata.json) and
[2000 GeV](ttbar_hadronic/s2000/g_hybrid_200m/metadata.json).
Each contains campaign metadata, four replica metadata files, the unmodified
final result, run log, input provenance and numerical-validation metadata.

Each replica uses three 1M learning iterations followed by 50M production trials
with adaptation frozen: 12M learning plus 200M production per energy. A trial
includes its complete angular/mirror packet and any precision retries; it is not
one elementary evaluator call. The shared pool used 128 workers within the
one-third-machine resource cap. Sampling was G-hybrid in memory, with no
production-source changes, no clipping and the 16 -> 32 -> 60 -> zero policy.
The exact runtime h normalisation is 2*exp(2)/sqrt(pi).

The saved estimator pools production first and second moments across all four
replicas; it is not an inverse-variance recombination. The common auxiliary
estimate is added once, and its uncertainty is counted once. Its final error is
at most 0.1 times the hard error. Scheme and decoupling terms share auxiliary
samples; their saved combined uncertainty retains that covariance. Hard graph
results are retained without assigning channel-level auxiliary corrections to
individual graphs. Channel errors are marginal errors under joint sampling and
must not be added in quadrature to reconstruct the total hard error.

Metadata records raw, scheme-adjusted, scheme-converted and final matched values
in pb and GeV^-2, channel and graph breakdowns, sample counts, seeds, physics,
MG5 benchmarks and abs(MG5-pygloop)/sqrt(sigma_MG5^2+sigma_pygloop^2).
Evaluation CPU time per production trial is distinguished from summed block
wall time and full campaign wall time; no per-graph timings are assigned.
Physics uses mt=173 GeV, alpha_s=0.118, PDF4LHC21_40 member 0,
mu_R=mu_F=91.118 GeV and mu_UV=sqrt(s)/2. Source cards retain their original
defaults; effective overrides and the actual controller schedule are recorded.
The 1500 GeV effective settings are reconstructed from the source cards and
verified campaign overrides, rather than presented as a saved as-run card.

The 1000 GeV campaign has one unresolved gg GL014 replay disagreement among 92
retained-weight checks. The production value was about 58,800 times its agreeing
80/128-digit evaluations. Correcting only that point would shift the full 200M
mean by +4.45707729603e-6 pb (0.151 of its reported error); no correction was
applied. This is recorded in
[numerical_validation.json](ttbar_hadronic/s1000/g_hybrid_200m/numerical_validation.json).
The 1500 GeV run zeroed one packet at the highest precision, as configured.
The 1500/2000 GeV retained-weight checks had no replay disagreements. These
checks are not exhaustive checks of all production points.

## Files and provenance

- drell_yan/replica_00 through replica_05: final status, saved source/effective
  cards, log and normalized metadata.
- ttbar_partonic/s<energy>/<channel>: final result, log and metadata; the 400 GeV
  gg folder contains four individual replicas.
- ttbar_hadronic/s<energy>: combined-channel result, log and metadata, with
  saved comparisons or final state/iteration records where available.
- ttbar_hadronic/s<energy>/g_hybrid_200m: the three new campaigns, each with
  replica_01 through replica_04 and a saved combined result.
- provenance/source_tree: unmodified cards and configuration dependencies,
  campaign manifests, benchmark/validation records, bundle metadata and
  historical provenance archives, preserving original source paths.
- manifest.json: source-to-archive mappings, run index, byte sizes, SHA256
  digests and archive-assembly Git revision.
- SHA256SUMS: checksums of every other file in this archive.

The final 1500 GeV state and result establish completion. Its earlier manifest
and README still describe a running job and are preserved as historical records.
Source paths inside copied files are unchanged. Generated executable evaluators,
frozen sampling grids and individual production checkpoints remain referenced by
their source paths; this is a metadata archive. New saved final results include
their retained large-weight replay records. Each source_archive_index.json is
the original campaign inventory, not a claim that every checkpoint was copied.

Run-recorded card hashes and bundle fingerprints are verified when available.
All partonic bundle fingerprints and source-card hashes have also been checked
against their preserved launch logs. The 400 GeV hadronic result supplies a
hash-matching parent card; bundle verification status is recorded explicitly.
The three original 2000 GeV child cards were recovered by restoring their original
bundle names; each recovered file exactly matches the SHA256 recorded in the
selected result. See provenance/s2000_selection_correction.json.
Recorded source-code hashes are preserved where available.
The archive-assembly Git revision is not assigned to historical runs.

Original final estimators and numerical diagnostics are retained. Offline
restoration or replay estimates are not substituted.

## Partonic units audit

The original partonic final values use GeV^-2. The first response and the table
above use pb, with 1 GeV^-2 = 389379660 pb. Both representations are retained in
results.json. Source values, logged final totals, and component values are
recorded in provenance/partonic_value_audit.json.
