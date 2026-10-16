# Residual illustration

The motivation figure: "residual structure for the Snelson training set".

The paper's Snelson figure is a historical, same-fit illustration from commit
`ddaa90b0b` (31 July 2026), using seed 7, five inducing points, blocks of 40,
120 steps, and the then-current Optax L-BFGS implementation. The current
`sgpr/jax_collapsed.py` routes `lbfgs` to SciPy L-BFGS-B; rerunning the current
script with the *same seed* changes the fitted inducing points and hence the
selected partition and log-determinants. Therefore a typography-only revision
of the paper figure must use the historical fit, not overwrite it with a new
optimization run. The paper figure labels are "Random blocks" and "Selected
blocks"; the stars denote inducing points.

The script trains `SBT-SGPR` on Snelson, freezes the learned parameters, and
evaluates three residual corrections **on the same residual matrix**, which is
what isolates block selection from optimiser differences. Public figure and
table labels:

- `Exact` — dense residual correction at the fitted sparse parameters
- `SBT` — inducing-anchor selected-block correction
- `BT` — random equal-size block correction

Selection is `common.partitions.sbt_blocks_host`; packing is
`common.terms.pack_terms`; the initial point is `sgpr.jax_collapsed.init_params`,
the same one the benchmark starts from. Initial inducing points are random
training points at the configured seed.

By default the script uses the L-BFGS path shared with the collapsed benchmark.
To reproduce the older Adam version:

```bash
RESIDUAL_OPTIMIZER=adam RESIDUAL_STEPS=500 python -m sgpr.residual_illustration.run_residual_structure
```

```bash
python -m sgpr.residual_illustration.run_residual_structure
python -m sgpr.residual_illustration.run_residual_structure --split-mode boundary --block-size 40
python -m sgpr.residual_illustration.run_residual_structure --split-mode center --block-size 40
python -m sgpr.residual_illustration.run_residual_structure --n-blocks 5
```

Outputs:

- `residual_structure_snelson.png` and `Report/figures/residual_structure_snelson.pdf`
- `snelson_residual_results.csv`
- `snelson_residual_table.tex`
