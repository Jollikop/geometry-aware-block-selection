# Snelson

Reference experiment of the project: the figure that shows what the four
methods do to a 200-point 1-D fit, where the whole picture fits on one axis.

```bash
python -m sgpr.snelson.run_snelson
```

## Protocol (hard-coded in the script)

- `M = 5` inducing points, sampled as random training points (`seed=72`),
  trainable `Z` for every sparse/structured method.
- SE kernel and noise at GPflow default initialisation.
- `N_BLOCKS = 10`, hence `block_size = 20`.
- 60 L-BFGS iterations in total; SBT uses the front-loaded schedule
  `[2, 3, 5, 20, 30]`, so five structures overall: the initial selection serves
  phase one and four refreshes happen before phases 2-5.
- Methods: `SGPR`, `T-SGPR`, `RBT-SGPR` (fixed random blocks) and `SBT-SGPR`.
- `FullResidualSGPR` provides the dense residual `E-SGPR` line in the loss
  panel. A separately fitted exact GPR provides the black `True GP` reference.

## Implementation

Structure selection is `common.partitions.sbt_terms_host`; the predictive
correction is `common.predictors.predictive_variance`, dispatched by the
method's entry in `COLLAPSED_METHOD_PREDICTOR` — the same registry and the same
function the benchmark headline uses. The script itself contains only the
protocol, the training loop and the outputs, so the figure cannot illustrate a
method the table does not measure.

SBT trains with a value-and-gradient compiled once
(`make_compiled_value_and_grad`) and reused across all phases: a refresh only
assigns the padded Variables inside `StructuredSGPR`, so there is no TensorFlow
retracing and no per-phase recompilation.

Snelson is a fit diagnostic over all 200 observations, so it reports no
held-out NLL. Its bands are bound-matched: SGPR draws the sparse predictive,
and every method that models a residual draws the correction over its own
complete fitted cover — there is no truncation:

```text
d_** - d_*f (D_ff + sigma^2 I)^-1 d_f*
```

The Gaussian NLL then uses that variance through the usual
`0.5 * (log(2*pi*var) + residual^2/var)`.

## Outputs

- `Report/figures/snelson_sgpr_t_rbt_sbt.pdf` — loss/N (log scale) and fits
- `snelson_summary_sgpr_t_rbt_sbt.csv` — summary with cost and anchors
- `snelson_table_sgpr_t_bt_sbt.tex` — LaTeX table
- `Report/Results/snelson_sgpr_t_rbt_sbt_manifest.json` — data SHA-256, Git
  state, runtime versions, protocol and the plotted objective values
