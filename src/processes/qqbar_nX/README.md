# qqbar_nX IR-subtracted generation

This process driver builds the current `d d~ -> h h h` two-loop amplitude
topology with a massive top pentagon, two gluons attached to the initial-state
quark line, and local ISR-collinear counterterms in the same GammaLoop graph
groups as their original graph.

The checked-in process configs cover the four currently supported generation
setups directly:

- `config.toml`: final-state symmetrized, threshold-enabled.
- `config_no_thresholds.toml`: final-state symmetrized, no-threshold.
- `config_nosym.toml`: non-symmetrized final states, threshold-enabled.
- `config_no_thresholds_nosym.toml`: non-symmetrized final states, no-threshold.

The default configuration in `config.toml` is the final-state-symmetrized,
exact-xi topology mode with GammaLoop threshold subtraction enabled:

- `generation.symmetrize_final_states = true`
- `counterterms.auxiliary_denominator_mode = "exact_xi_topology"`
- `standalone_run_card.enable_threshold_subtraction = true`
- `tests.collinear_precision = "ArbPrec"`
- `tests.collinear_fraction_x = 0.3`

Use one of the no-threshold configs for the fast heavy-top validation setup.
The threshold flag only changes GammaLoop generation/runtime settings; the
subtracted DOT construction is independent of it.

The exact-xi mode adds two dummy in/out helper pairs to every graph, so the
GammaLoop externals are ordered as
`Q(0) Q(1) Q(2) Q(4) -> Q(3) Q(5) Q(6) Q(7) Q(8)`. The physical process is
still `Q(0) Q(1) -> Q(6) Q(7) Q(8)`, with `Q(8)` dependent. The first
collinear counterterm uses only the helper pair `Q(2)=Q(4)=xi`; the second uses
only `Q(3)=Q(5)=xi`. The originals carry all four helper externals as dummy
edges.

## Generate the subtracted DOT

Run from the repository root with the pygloop virtual environment active:

```bash
source .venv/bin/activate
```

Final-state-symmetrized, threshold-enabled physical-top setup (`m_t = 173 GeV`):

```bash
python src/pygloop.py \
  --process qqbar_nX \
  --qqbar-nx-config src/processes/qqbar_nX/config.toml \
  --m_top 173 \
  --clean \
  build-qqbar-nx-ir
```

This writes:

```text
outputs/dot_files/qqbar_nX/qqbar_nX_d_dbar_h_h_h_2L_top_pentagon_isr_subtracted.dot
outputs/dot_files/qqbar_nX/qqbar_nX_d_dbar_h_h_h_2L_top_pentagon_isr_subtracted.toml
```

Final-state-symmetrized, no-threshold setup (`m_t = 1000 GeV`):

```bash
python src/pygloop.py \
  --process qqbar_nX \
  --qqbar-nx-config src/processes/qqbar_nX/config_no_thresholds.toml \
  --m_top 1000 \
  --clean \
  build-qqbar-nx-ir
```

The DOT file is identical between these threshold and no-threshold configs when
the same graph-selection/symmetrization options are used; only the generated
run-card defaults and GammaLoop generation/runtime threshold settings differ.

Non-symmetrized, threshold-enabled physical-top setup (`m_t = 173 GeV`):

```bash
python src/pygloop.py \
  --process qqbar_nX \
  --qqbar-nx-config src/processes/qqbar_nX/config_nosym.toml \
  --m_top 173 \
  --clean \
  build-qqbar-nx-ir
```

Non-symmetrized, no-threshold setup (`m_t = 1000 GeV`):

```bash
python src/pygloop.py \
  --process qqbar_nX \
  --qqbar-nx-config src/processes/qqbar_nX/config_no_thresholds_nosym.toml \
  --m_top 1000 \
  --clean \
  build-qqbar-nx-ir
```

This writes the non-symmetrized DOT/TOML pair with the
`_top_pentagon_isr_subtracted_nosym` suffix.

## Standalone GammaLoop steering

The generated TOML deliberately has `commands = []`: loading it only registers
settings and command blocks. Nothing is imported or generated until a block is
run explicitly.

The universal checked-in card is
`src/processes/qqbar_nX/qqbar_nX_standalone.toml`. It is not emitted from
pygloop and can be used after the expected DOT exists at
`outputs/dot_files/qqbar_nX/qqbar_nX_d_dbar_h_h_h_2L_top_pentagon_isr_subtracted.dot`.
It contains two explicit no-placeholder demo blocks:

- `demo`: no-threshold, `m_t = 1000 GeV`.
- `demo_with_thresholds`: threshold-enabled, `m_t = 173 GeV`.

Run the no-threshold demo:

```bash
GL=/Users/vjhirsch/Documents/Work/gammaloop_dev/target/dev-optim/gammaloop
CARD=src/processes/qqbar_nX/qqbar_nX_standalone.toml
STATE=outputs/gammaloop_states/qqbar_nX_standalone_demo

$GL --clean-state "$CARD"
$GL -o -s "$STATE" run demo
```

Run the threshold-enabled demo:

```bash
$GL --clean-state "$CARD"
$GL -o -s "$STATE" run demo_with_thresholds
```

Open the live GammaLoop CLI on that state:

```bash
$GL -o -s "$STATE"
```

The private blocks are reusable templates and accept `-D` overrides for the
process name, integrand name, DOT path, model masses, threshold flags, and
integration workspace. Multiple blocks can be run in a single command with one
shared context, which is the form used by `demo` and `demo_with_thresholds`:

```bash
PROC=qqbar_nX_d_dbar_h_h_h_2L_xi_ext
INT=qqbar_nX_d_dbar_h_h_h_2L_top_pentagon_isr_subtracted
DOT=$PWD/outputs/dot_files/qqbar_nX/${INT}.dot

$GL -o -s "$STATE" run \
  _generate_subtracted_integrand \
  _inspect_collinear_p1_group0_arb \
  _inspect_collinear_p2_group0_arb \
  _low_stat_integrate_pm \
  -D process_name=$PROC \
  -D integrand_name=$INT \
  -D dot_path=$DOT \
  -D m_top=1000.0 \
  -D m_higgs=125.0 \
  -D ymt=1000.0 \
  -D enable_thresholds=false \
  -D check_esurface_at_generation=false \
  -D assume_positive_external_energies=false \
  -D disable_threshold_subtraction=true \
  -D workspace_path=$PWD/outputs/dot_files/qqbar_nX/${INT}_demo_pm
```

For the threshold version, switch only the model and threshold defines:

```bash
$GL -o -s "$STATE" run \
  _generate_subtracted_integrand \
  _inspect_collinear_p1_group0_arb \
  _inspect_collinear_p2_group0_arb \
  _low_stat_integrate_pm \
  -D process_name=$PROC \
  -D integrand_name=$INT \
  -D dot_path=$DOT \
  -D m_top=173.0 \
  -D m_higgs=125.0 \
  -D ymt=173.0 \
  -D enable_thresholds=true \
  -D check_esurface_at_generation=false \
  -D assume_positive_external_energies=false \
  -D disable_threshold_subtraction=false \
  -D workspace_path=$PWD/outputs/dot_files/qqbar_nX/${INT}_demo_threshold_pm
```

For the non-symmetrized file, point `INT` and `DOT` at the `_nosym` suffix. The
same `_inspect_collinear_*` and `_low_stat_integrate_*` private blocks accept
the same `-D` set, plus `-D workspace_path=...` for integration.

The pygloop-emitted card in `outputs/dot_files/qqbar_nX/*.toml` contains the
same private template blocks and the same `demo` / `demo_with_thresholds`
convenience blocks, but the checked-in card above is the stable steering entry
point.

## pygloop tests

The process implements the top-level `test`/`test_process` subcommand with
process-specific modes.

Direct 4D Feynman-integrand collinear approach. This does not generate the 3D
GammaLoop representation. Use `--clean` when an older saved GammaLoop state is
present, because older state-local `global_settings.toml` files may contain
settings no longer accepted by the current GammaLoop wheel:

```bash
python src/pygloop.py \
  --process qqbar_nX \
  --qqbar-nx-config src/processes/qqbar_nX/config_no_thresholds.toml \
  --m_top 1000 \
  --clean \
  test --mode 4d
```

GammaLoop 3D/CFF local collinear approach:

```bash
python src/pygloop.py \
  --process qqbar_nX \
  --qqbar-nx-config src/processes/qqbar_nX/config_no_thresholds.toml \
  --m_top 1000 \
  test --mode gammaloop
```

Custom pygloop CFF meta-expression parity check against GammaLoop inspect:

```bash
python src/pygloop.py \
  --process qqbar_nX \
  --qqbar-nx-config src/processes/qqbar_nX/config_no_thresholds.toml \
  --m_top 1000 \
  test --mode pygloop-cff
```

The 4D and GammaLoop tests print colored PrettyTable summaries for p1 and p2
approaches, each group id, each graph contribution, the group sum, and the
ratio `abs(sum(graphs)) / sum(abs(graphs))`. The custom CFF test currently
validates the no-threshold CFF algebra at double precision. The public
GammaLoop Python/CLI inspect path returns an f64-projected result even when
`-f`/`use_arb_prec` is requested, so exact ArbPrec parity of the custom
Symbolica eager evaluator cannot yet be compared directly through that API.
At 32 decimal digits, where Symbolica uses its fast double-like path, the
single-orientation relative difference is about `1e-6`; at higher precision
the custom ArbPrec value follows a different branch from the f64-projected
GammaLoop reference.
