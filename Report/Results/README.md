# Report/Results

Compact outputs behind the paper's tables and figures. Each `*_manifest.json`
records the inputs it was built from (with sha256), the source commit where
applicable, and the outputs it certifies. Absolute paths of the machine that
built them were replaced by `<root>/`.

| File | Content |
|---|---|
| `sgpr_rbf_v23_manifest.json` | RBF Kin40k SGPR release (v23): runs, transplant fits, checks, source commit |
| `kin40k_sgpr_main_summary.csv`, `kin40k_sgpr_timing_summary.csv`, `kin40k_sgpr_compact_timing_manifest.json` | SGPR comparison and timing summaries |
| `kin40k_sgpr_transplant_*_summary.csv` | sparse refinement of fitted donors, per kernel |
| `sgpr_frozen_manifest.json` | frozen SGPR campaign: code commit and result-file hashes |
| `snelson_*` | Snelson reference fit: manifests and summary |
| `residual_structure_kin40k_*` | residual-structure diagnostics on Kin40k |
| `geometric_recovery_main.json` | configurations and certificate margins of the geometric-recovery figure |
| `hard_pic_full_v1/` | one-block (PIC) complete predictor: SVGP summary over the full grid, SGPR and Snelson records, prediction certificate, and the hashes of the paper figures and tables built from them |
| `hard_pic_dense_diagnostic/` | paired one-block versus dense-residual NLL comparison (appendix) |
| `svgp_frozen_campaign_manifest.json`, `svgp_frozen_campaign_summary.csv` | frozen SVGP training campaign: sparse-transplant metrics, objectives, inputs and hashes |
| `svgp_focused_final_manifest.json`, `svgp_focused_final_summary.csv` | earlier summary of the RBF and amplitude-RBF page (complete curves superseded by `hard_pic_full_v1/`) |
| `aggregates/` | per-cell SVGP aggregates (full donors, 300-epoch sparse refinement, Gibbs anchors) |
| `epochs_rbf_pol_M128_summary.csv` | epoch diagnostic |
| `partition_controls_summary.csv` | prediction-partition control of Table 3: per configuration, mean and standard error over ten paired seeds of the (S)BT-SVGP minus (R)BT-SVGP test NLL for the current, same-selected and same-random prediction partitions |
| `partition_controls/` | certificates and seed summaries of the controls: `E1/`, `E2/` (640 predictions with frozen donors, none rejected) and `E4/` (dense cutoff sweep of Appendix D.4, 1400 evaluations, none rejected) |
| `block_diag_ablation/` | split versus shared batch on RBF Pol M=128: certificate (50 trainings, 900 block-diagonal predictions) and aggregate behind Figures 5 and 6 |
| `inducing_*_summary.csv`, `predictive_nll_ablation_summary.csv` | supporting diagnostics |
