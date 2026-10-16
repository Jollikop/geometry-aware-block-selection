# Migration to the two-stack layout

The refactor is delivered as a tarball plus `apply_refactor.sh` because the Git
repository root sits above the folder shared with this session, so the branch
has to be created from your terminal.

## What the script does

1. `git checkout -b refactor/two-stacks`
2. extracts the new tree into `Simulations/`
3. `git rm -r` the retired paths listed below
4. installs the package in editable mode and runs the submission checks

Nothing outside `Simulations/` is touched. `Report/`, `References/` and
`Simulations/dataset/` are left exactly as they are.

## Retired paths

| Removed | Where it went |
|---|---|
| `Simulations/libraries/provenance.py` | `common/provenance.py` |
| `Simulations/libraries/plot_style.py` | `common/plot_style.py` |
| `Simulations/libraries/jax_data.py` | `common/data_split.py` |
| `Simulations/libraries/data.py` | merged into `common/data_split.py` |
| `Simulations/libraries/jax_nonstationary.py` | `common/kernels_nonstationary.py` |
| `Simulations/libraries/jax_structures.py` | split into `common/kernels.py` + `common/partitions.py` |
| `Simulations/libraries/structures.py` | merged into `common/partitions.py` |
| `Simulations/libraries/SGPR.py` | `sgpr/gpflow_bounds.py` |
| `Simulations/libraries/training.py` | `sgpr/gpflow_training.py` |
| `Simulations/libraries/jax_models.py` | `sgpr/jax_collapsed.py` |
| `Simulations/libraries/jax_sgpr_bench.py` | `sgpr/benchmark.py` |
| `Simulations/libraries/jax_svgp.py` | `svgp/engine.py` |
| `Simulations/SGPR/run_sgpr_jax.py` | `sgpr/run_benchmark.py` |
| `Simulations/SGPR/datasets_sgpr.py` | `common/datasets_uci.py` |
| `Simulations/SGPR/make_main_table.py` | `sgpr/make_main_table.py` |
| `Simulations/SGPR/plot_predictive_nll_ablation.py` | `sgpr/plot_predictive_nll_ablation.py` |
| `Simulations/SNELSON/` | `sgpr/snelson/` |
| `Simulations/Residual_illustration/` | `sgpr/residual_illustration/` |
| `Simulations/SVGP_Benchmark_Modern/run_svgp_modern_jax2.py` | `svgp/run_benchmark.py` |
| `Simulations/SVGP_Benchmark_Modern/datasets_modern.py` | split into `common/datasets.py` + `svgp/plans.py` |
| `Simulations/SVGP_Benchmark_Modern/prepare_electricity.py` | `common/prepare_electricity.py` |
| `Simulations/SVGP_Benchmark_Modern/make_k_variant.py` | **deleted** — the top-k predictor was removed; every method now reports the estimator its own bound models |
| `Simulations/SVGP_Benchmark_Modern/bt_local_predictor.py` | **deleted** — superseded by the predictor registry in `common/predictors.py` |
| `Simulations/SVGP_Benchmark_Modern/{paper_status,freeze_campaign,check_*,screen_leakage,compare_*,sweep_block_size}.py` | `svgp/analysis/` |
| `Simulations/SVGP_Benchmark_Modern/plot_*.py` | `svgp/` |
| `Simulations/SVGP_Benchmark_Modern/make_top8_figures.sh` | **deleted** with the top-k predictor |
| `Simulations/SVGP_Benchmark_Modern/paper_top8_prediction.sbatch` | **deleted** with the top-k predictor |
| `Simulations/tests/test_jax_data.py` | `tests/test_data_split.py` |

`Simulations/archive/` is untouched.

## One directory you must remove by hand

This session created an empty probe directory it is not allowed to delete:

```bash
rmdir Simulations/_probe
```

## Manual rollback

```bash
git checkout main
git branch -D refactor/two-stacks
```
