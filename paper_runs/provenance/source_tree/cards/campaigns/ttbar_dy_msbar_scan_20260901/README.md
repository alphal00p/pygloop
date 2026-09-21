# Reproducible ttbar and Drell--Yan MSbar campaign

This campaign prepares:

- partonic `qqbar`, `qg`, and `gg` contributions to `PP -> tt~` at
  `sqrt(s) = 400, 1000, 2000 GeV`;
- the corresponding hadronic three-channel sums, sampled with Symbolica's
  native physical-channel and graph-channel discrete integrators;
- hadronic Drell--Yan at `sqrt(s) = 1000, 2000 GeV`, in 10 equal `z` bins
  covering `[0,1]`, plus one inclusive closure card at each energy.

Every channel card returns the complete finite MSbar quantity. The ttbar
cards enable the in-code finite scheme-changing counterterms. Heavy-top
decoupling is disabled for the native `n_l+1` partonic Czakon--Mitov
comparison and enabled in the hadronic children compared with decoupled MG5.
The hadronic `qg` decoupling coefficient is recorded explicitly as zero at NLO
because that channel starts at NLO. Drell--Yan uses its in-code finite scheme
conversion; heavy-top decoupling is not applicable there.

## Locked physics and sampling choices

- `m_top = 173 GeV`, `alpha_s = 0.118`, `alpha_ew_inverse = 132.507`;
- `lambda_sq = mur_sq = muf_sq = 91.118^2 = 8302.489924 GeV^2`;
- `muv = sqrt(s)/2` for ttbar and `muv = sqrt(s)` for Drell--Yan;
- `PDF4LHC21_40`, member 0;
- `Q_min = 300 GeV` for Drell--Yan;
- `p1 <-> p2` symmetrisation in every generated channel;
- projected-OS top self-energies for ttbar `qqbar` and `gg`, and `no-os` for
  `qg` where no top self-energy subtraction is present;
- the validated soft mirrors `GL07:7, GL09:8` for ttbar `qqbar` and
  `GL015:7, GL018:8, GL059:8, GL071:7, GL075:8, GL099:8, GL115:8` for `gg`;
- the routed `GL1:4, GL2:5` soft mirrors and native
  `graph_1+graph_2[invert]` grouping for Drell--Yan `qqbar`, used consistently
  in smoke and production cards;
- `beta_y` beam sampling for hadronic ttbar and `x1_x2` for Drell--Yan;
- saved-JIT matrix batches of 256 with persistent workers, continuous
  adaptation, and a 7.5% minimum physical-channel sampling fraction;
- 64 cores in Drell--Yan cards; the older ttbar cards retain all 384 visible
  CPUs as their configurable default.

The later experimental GL059/GL071 ratio-centred map is deliberately absent:
its matched pilots were slower and less precise than the established map.

## One-command stages

Run every stage with the requested interpreter from the repository root:

```bash
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/run_campaign.py validate
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/run_campaign.py generate
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/run_campaign.py smoke --scope all
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/run_campaign.py calibrate
```

`generate` creates seven provenance-stamped bundles: three two-loop ttbar hard
bundles, the two one-loop ttbar Born bundles used by the finite terms, and the
two one-loop Drell--Yan bundles. It retains an existing bundle unless `--force`
is supplied.

`smoke` runs 2 x 1M unclipped points for each partonic and hadronic ttbar
channel. Drell--Yan uses one 1M-point smoke for each bin/channel pair and each
inclusive pair, with the validated 64-bin continuous grid, 0.5 density floor,
and soft-radius HP threshold of 0.001. There is no separate learning,
frozen-grid, or production phase in these scale-calibration smokes. It is
resumable and keeps an existing result only when its card hash and sample count
match the current campaign. HP failures and clipped samples are retained as
reported diagnostics rather than result-rejection conditions. The retained
partonic calibration cards include the component-resolved decoupling auxiliary
solely so the already validated smoke artifacts remain reusable; direct-CM
production cards disable that component.
Scopes
`ttbar_partonic`, `ttbar_hadronic`, and `dy_hadronic` may be run separately.

For Drell--Yan, `calibrate` defines each bin/channel hard scale as
`max(abs(raw central), raw error)`, sets the high-precision trigger to
`1e4 * scale`, and sets clipping to `1e7 * scale`. The auxiliary scheme
integral receives its own `1e7 * auxiliary_scale` clip in the same units. The
2000 GeV endpoint production card applies a recorded additional factor of 100
to both clips, while retaining the original precision threshold. Its current
independent production run uses the manifest-recorded seed `924009101`.
Use `calibrate --scope dy_hadronic` to consume the complete current DY smoke
matrix without requiring unrelated ttbar smokes.

Combined Drell--Yan production cards enable structured live diagnostics.  Each
completed iteration emits a `DY_PHYSICAL_CHANNEL_ITERATION` JSON record with
the corrected estimate, sampling allocation, and per-iteration and cumulative
stability counters, including clipped points and higher-precision failures.

Use `status` at any point:

```bash
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/run_campaign.py status
```

## Running production cards directly

After calibration, no command-line physics override is needed. The campaign
runner supplies the validated continuous-grid settings for Drell--Yan;
examples:

```bash
# Full partonic qg MSbar coefficient at 1 TeV
.venv_final/bin/python3.13 src/pygloop.py \
  --dy-card cards/campaigns/ttbar_dy_msbar_scan_20260901/production/ttbar/partonic/s1000/qg.toml \
  integrate

# Full hadronic pp -> tt~ sum at 2 TeV
.venv_final/bin/python3.13 src/pygloop.py \
  --dy-card cards/campaigns/ttbar_dy_msbar_scan_20260901/production/ttbar/hadronic/s2000/combined.toml \
  integrate

# First Drell--Yan z bin with its validated continuous-grid defaults
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/run_campaign.py \
  production --scope dy_bins \
  --card production/dy/s2000/z00_10/combined.toml
```

The 10 binned combined cards are under
`production/dy/s{1000,2000}/zXX_YY/`. At each energy their sum should be
compared with the corresponding `inclusive/combined.toml` as a bin-closure
check before comparing the same observable definition with MG5.
Each combined DY card pins the SHA256 of both child cards in its header, so a
child sampling or finite-part change also invalidates the parent result hash.

The resumable production wrapper applies the validation budgets used for this
campaign: 100M points for every partonic and hadronic ttbar card, 10M for each
nonzero Drell--Yan bin, and 40M for the endpoint/zero bin (`z90_100`) and each
inclusive Drell--Yan card.

```bash
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/run_campaign.py production --scope ttbar_partonic
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/run_campaign.py production --scope ttbar_hadronic
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/run_campaign.py production --scope dy_bins
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/run_campaign.py production --scope dy_inclusive
```

Use `--card MANIFEST_RELATIVE_CARD --cores N` to run one card with a chosen
worker count. For an individual card, `--iterations N` may raise the locked
minimum iteration count when more statistics are needed. A completed result
is retained only when its card hash, point budget, execution overrides, and
numerical diagnostics match the campaign.

For the endpoint-first Drell--Yan gate, calibrate the two `z90_100` children,
run only their combined parent, and require at most 1% Pygloop uncertainty
before comparing with MG5:

```bash
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/run_campaign.py smoke --scope dy_hadronic --card smoke/dy/s2000/z90_100/qqbar.toml --cores 38
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/run_campaign.py calibrate --scope dy_hadronic --card smoke/dy/s2000/z90_100/qqbar.toml
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/run_campaign.py smoke --scope dy_hadronic --card smoke/dy/s2000/z90_100/qg.toml --cores 38
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/run_campaign.py calibrate --scope dy_hadronic --card smoke/dy/s2000/z90_100/qg.toml
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/run_campaign.py production --scope dy_bins --card production/dy/s2000/z90_100/combined.toml
```

Any endpoint comparison must use an MG5 benchmark with the same 300 GeV cut
and `[0.9,1]` bin. The older `compare_dy_endpoint_mg5.py` helper targets the
superseded 20-bin campaign and is therefore not invoked for this matrix.

The cards use a 32-digit first HP fallback. If that orbit still disagrees,
the stability pipeline retries only that sample at 48 digits and retains a
64-digit final safety net. The result records the extra calls and successful
rescues as `stability_hp_escalation_count` and
`stability_hp_escalation_accepted_count`. Representative failures from all
three gg energies can be replayed without integration:

```bash
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/validate_hp_rescue_points.py
```

Partonic ttbar results can be compared directly, without a decoupling shift,
with the `NLOXSectionQQ`,
`NLOXSectionGQ`, and `NLOXSectionGG` entries in the checked-in Czakon--Mitov
source at
`outputs/qqbar_ttbar_first_principles_normalisation_audit_20260810/czakon_mitov_0811.4119_source/NLOXSections.m`.
The generated result is already the full MSbar value in the convention of its
card; do not add a second finite term or decoupling correction in analysis.

Build the exact native Czakon--Mitov reference and compare the nine partonic
results with:

```bash
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/build_czakon_mitov_references.py
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/compare_partonic_czakon_mitov.py
```

The historical full-MG5 preparation and comparison helpers below still target
the original 20-bin, 2 TeV DY matrix; they are retained for provenance and are
not used with the new 10-bin cards. Any new MG5 comparison must use both the
same energy and `Q_min = 300 GeV`, with histogram edges matching the ten bins.
The historical helpers prepare run and launch cards without mutating either
MG5 process directory:

```bash
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/prepare_mg5_benchmarks.py \
  --cores 384 --ttbar-process /path/to/PP_TTBAR_NF1_COMBINED \
  --dy-process /path/to/PP_DY_Z20
```

For each entry in `outputs/ttbar_dy_msbar_scan_20260901/benchmarks/mg5/manifest.json`,
run `PROCESS/bin/aMCatNLO LAUNCH_CARD`. The launch card is an MG5 command
file; do not pipe it to `generate_events`, whose stdin is reserved for the
interactive launch prompts. Once all four runs finish,
collect and compare them directly:

```bash
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/collect_mg5_benchmarks.py
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/compare_hadronic_mg5.py
```

Pygloop DY integration results already include the fixed
`-e^4 Q_d^2 g_s^2` coupling/sign normalisation from the card's runtime
`alpha_s` and `alpha_ew_inverse`. They remain in GeV^-2, so a direct MG5
comparison applies only `GeV^-2 -> pb`. No finite terms, decoupling shifts,
fitted constants, or reference shifts are added in analysis.
This normalization is automatic for `process_name = "dy"`; there is no card
switch to enable or disable it, and it is not applied to ttbar.

The Drell--Yan collector uses MG5's `z z-weighted NLO ...` histograms and
sums all ordered initial states. This is the histogram convention matching the
native Pygloop `z`-coefficient observable; the unweighted `z pure NLO ...`
histograms describe a different observable and are intentionally excluded.

The validated `Q_min = 300 GeV` MG5 runs at 1000 and 2000 GeV contain the
complete 20-bin histogram even though they were originally launched for the
endpoint check. Collect adjacent pairs into the current ten-bin campaign and
compare either just the endpoint gate or the completed campaign with:

```bash
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/collect_dy_qmin300_mg5.py
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/compare_dy_qmin300_mg5.py endpoints
.venv_final/bin/python3.13 cards/campaigns/ttbar_dy_msbar_scan_20260901/compare_dy_qmin300_mg5.py complete
```

The collector checks the MG5 analysis, process, card, banner, and raw-HwU
hashes and records that HwU does not expose covariance between adjacent bins.
The comparison requires every total bin and inclusive result, as well as each
independent ten-bin closure, to lie within three combined standard deviations.
The complete comparison also writes `dy_qmin300_s1000_results.json` and
`dy_qmin300_s2000_results.json` under the comparison output directory. Each
contains the total and physical-channel results, MG5 pulls, numerical counters,
card settings and hashes, and hashed paths to every retained raw result and log.

## Provenance and outputs

`manifest.json` contains every concrete card hash, bundle name, scale, bin
edge, and calibration status. `calibration_thresholds.json` is created from
the smoke results. Logs and machine-readable results live under
`outputs/ttbar_dy_msbar_scan_20260901/` and mirror the card directory layout.

The Symbolica multicore license and the existing LHAPDF Python/library paths
are installed locally in `.venv_final`; the key itself is intentionally not
copied into cards, manifests, logs, or this runbook.
