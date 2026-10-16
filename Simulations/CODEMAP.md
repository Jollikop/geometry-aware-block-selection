# Code map

- `common/`: datasets, kernels, partitions, predictors, provenance, result IO.
- `sgpr/jax_collapsed.py`: collapsed objectives and predictive components.
- `sgpr/benchmark.py`: one fit and its saved artifacts.
- `sgpr/run_benchmark.py`: official grid and cache driver.
- `sgpr/snelson/`: one-dimensional reference figure.
- `sgpr/residual_illustration/`: same-parameter partition heatmaps.
- `svgp/engine.py`: variational objectives and Adam/NG training.
- `analysis/validate_final_suite.py`: release validation and freezing.
- `submit_sgpr_campaign.sh`: collapsed-only dependency chain.
- `submit_final_campaign.sh`: complete SGPR+SVGP dependency chain.
