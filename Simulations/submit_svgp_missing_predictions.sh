#!/usr/bin/env bash
# Submit the already-audited immutable donor prediction plan once.
set -euo pipefail
source "$HOME/aistats2026_svgp_all_training_jobs.env"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AIPY="${AIPY:?set AIPY to the python of the release environment}"
export SCALABLE_GP_DATASET_DIR="${SCALABLE_GP_DATASET_DIR:-$HERE/dataset}"
export PRED_TOOLS="${PRED_TOOLS:-$HERE}"
export PRED_OUTPUT="${FINAL_ROOT}_donor_predictions_missing_v1"
[[ -f "$PRED_OUTPUT/prediction_plan.json" && -x "$AIPY" ]]
ENV_FILE="$HOME/aistats2026_svgp_missing_prediction_jobs.env"
if [[ -e "$ENV_FILE" ]]; then
  echo "Existing dispatch environment found: $ENV_FILE; inspect before resubmitting." >&2
  exit 2
fi
mkdir "$PRED_OUTPUT/dispatch"
PRED_JOB=$(sbatch --parsable --array=0-4%5 --partition=NA100q --nodelist=node01 \
  --export=ALL --output="$PRED_OUTPUT/slurm/missing_%A_%a.out" \
  "$PRED_TOOLS/svgp_missing_predictions.sbatch")
printf '%s\n' "$PRED_JOB" > "$PRED_OUTPUT/dispatch/job_id.txt"
{
  printf 'export FINAL_ROOT=%q\n' "$FINAL_ROOT"
  printf 'export PRED_OUTPUT=%q\n' "$PRED_OUTPUT"
  printf 'export PRED_JOB=%q\n' "$PRED_JOB"
  printf 'export AIPY=%q\n' "$AIPY"
} > "$ENV_FILE"
printf 'PRED_JOB=%s\nPRED_OUTPUT=%s\nENV_FILE=%s\n' "$PRED_JOB" "$PRED_OUTPUT" "$ENV_FILE"
