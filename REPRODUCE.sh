#!/usr/bin/env bash
# One entry point for every experiment in the paper.
#
#   bash REPRODUCE.sh list            # stages, in dependency order
#   bash REPRODUCE.sh <stage> [args]  # launch one stage; extra args go to its launcher
#   DRY_RUN=1 bash REPRODUCE.sh <stage>   # print the commands without running them
#
# Each stage calls the launcher documented in README.md ("Campaigns" and
# "Tables and figures"). Launchers check their own environment variables
# (AIPY, result roots, dataset directory, Slurm partition) and stop with a
# message if one is missing; nothing here cancels or overwrites earlier runs.
# The GPU stages assume a Slurm cluster; the CPU stage "check" does not.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM="$ROOT/Simulations"
NOTES="$ROOT/Report/notes"
PY="${AIPY:-python3}"

run() {
  printf '+ %s\n' "$*"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then "$@"; fi
}

stage="${1:-list}"
shift || true

case "$stage" in
  list)
    cat <<'EOF'
check               CPU unit and protocol tests (minutes)
sgpr                RBF Kin40k SGPR comparison and sparse refinement (Table 1, Table 2)
svgp-donors         four-kernel SVGP donor grid (Figure 4, Appendix E)
svgp-resume         resumed training-only SVGP grid, including 3Droad
gibbs-anchors       Gibbs (S)BT-SVGP donors with inducing anchors
gibbs-predictions   predictions of the Gibbs anchor donors (needs an output root argument)
transplant300       300-epoch sparse SVGP refinement of every donor (dashed curves)
gibbs-transplant    300-epoch sparse refinement of the Gibbs anchor donors
block-diag-plan     plan the block-diagonal predictions (Snelson, Kin40k SGPR, SVGP grid)
block-diag-run      evaluate the planned block-diagonal predictions (Slurm array)
block-diag-certify  certify the block-diagonal predictions
snelson             Snelson fit and block-diagonal prediction (Figure 3)
dense-plan          plan the dense-residual reference (Appendix D.4)
dense-run           evaluate the dense-residual reference (Slurm)
split-ablation      split versus shared batch on RBF Pol, M = 128 (Figures 5 and 6; Slurm array, 5 GPUs)
split-certify       certify its 50 trainings and 900 block-diagonal predictions
partition-controls  plan the E1/E2 prediction-partition controls (Table 3; pass plan options)
partition-controls-run run a planned control (pass worker options)
partition-controls-certify verify and summarize a planned control
dense-cutoff-sweep  plan, evaluate or certify the dense cutoff sensitivity (E4, Appendix D.4)
figures             rebuild every table and figure from the certified outputs
EOF
    ;;
  check)             run bash "$SIM/run_submission_checks.sh" "$@" ;;
  sgpr)              run bash "$SIM/submit_sgpr_full_campaign.sh" "$@" ;;
  svgp-donors)       run bash "$SIM/submit_five_worker_campaign.sh" "$@" ;;
  svgp-resume)       run bash "$SIM/submit_svgp_all_campaign.sh" "$@" ;;
  gibbs-anchors)     run bash "$SIM/submit_gibbs_anchor_campaign.sh" "$@" ;;
  gibbs-predictions) run bash "$SIM/submit_gibbs_anchor_predictions.sh" "$@" ;;
  transplant300)     run bash "$SIM/submit_svgp_transplant_continue.sh" "$@" ;;
  gibbs-transplant)  run bash "$SIM/submit_gibbs_anchor_transplant300.sh" "$@" ;;
  block-diag-plan)   run "$PY" "$SIM/analysis/hard_block_prediction_campaign.py" plan "$@" ;;
  block-diag-run)    run sbatch "$@" "$SIM/analysis/hard_block_prediction_campaign.sbatch" ;;
  block-diag-certify) run sbatch "$@" "$SIM/analysis/hard_block_prediction_certify.sbatch" ;;
  snelson)           run "$PY" "$SIM/analysis/hard_snelson_prediction.py" "$@" ;;
  dense-plan)        run "$PY" "$SIM/analysis/real_predictor_campaign.py" plan "$@" ;;
  dense-run)         run sbatch "$@" "$SIM/analysis/real_predictor_main.sbatch" ;;
  split-ablation)
    # Needs AIPY, SOURCE_SIM, ABLATION_ROOT and SCALABLE_GP_DATASET_DIR;
    # see Simulations/analysis/README_batch_ablation_block_diag.md.
    run sbatch --array=0-4%5 "$@" \
      --export=ALL,AIPY,SOURCE_SIM,ABLATION_ROOT,SCALABLE_GP_DATASET_DIR,TOOLS_DIR="$SIM/analysis" \
      "$SIM/analysis/batch_ablation_block_diag.sbatch" ;;
  split-certify)
    run "$PY" "$SIM/analysis/batch_ablation_block_diag.py" certify \
      --source-sim "${SOURCE_SIM:?set SOURCE_SIM}" \
      --output "${ABLATION_ROOT:?set ABLATION_ROOT}" --tools "$SIM/analysis" "$@" ;;
  partition-controls) run "$PY" "$SIM/analysis/partition_controls.py" plan "$@" ;;
  partition-controls-run) run "$PY" "$SIM/analysis/partition_controls.py" worker "$@" ;;
  partition-controls-certify) run "$PY" "$SIM/analysis/partition_controls.py" certify "$@" ;;
  dense-cutoff-sweep) run "$PY" "$SIM/analysis/dense_cutoff_sweep.py" "$@" ;;
  figures)
    # Order matters: aggregates and tables first, then labels and layout.
    run "$PY" "$NOTES/build_hard_pic_paper_assets.py"
    run "$PY" "$NOTES/render_block_diag_paper_figures.py"
    run "$PY" "$NOTES/build_hard_pic_dense_diagnostic.py"
    run "$PY" "$NOTES/build_focused_svgp_page.py"
    run "$PY" "$NOTES/reflow_snelson_training.py"
    run "$PY" "$NOTES/reflow_snelson_prediction.py"
    if [[ -n "${ABLATION_ROOT:-}" ]]; then
      run "$PY" "$NOTES/build_block_diag_ablation.py" --input "$ABLATION_ROOT"
    fi
    run "$PY" "$NOTES/relabel_figures.py"
    ;;
  *)
    echo "unknown stage: $stage (try: bash REPRODUCE.sh list)" >&2
    exit 2
    ;;
esac
