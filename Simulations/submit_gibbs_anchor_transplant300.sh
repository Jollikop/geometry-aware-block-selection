#!/usr/bin/env bash
# Five finite, dependent stages; every result stays outside the donor freeze.
set -euo pipefail

: "${AIPY:?export the pinned Python interpreter}"
: "${TOOLS_ROOT:?export the dedicated numerical-tools Simulations directory}"
: "${ANCHOR_ROOT:?export the certified Gibbs-anchor donor root}"
: "${SCALABLE_GP_DATASET_DIR:?export the frozen dataset directory}"
: "${CAMPAIGN_ROOT:?export a NEW output directory}"

TOOLS_ROOT="$(realpath "$TOOLS_ROOT")"
ANCHOR_ROOT="$(realpath "$ANCHOR_ROOT")"
SCALABLE_GP_DATASET_DIR="$(realpath "$SCALABLE_GP_DATASET_DIR")"
PARENT_ROOT="$CAMPAIGN_ROOT/parent150"
CONT_ROOT="$CAMPAIGN_ROOT/continued300"
PRED_ROOT="$CAMPAIGN_ROOT/predictions"

for required in \
  "$AIPY" \
  "$TOOLS_ROOT/svgp/run_benchmark.py" \
  "$TOOLS_ROOT/analysis/continue_svgp_transplant.py" \
  "$TOOLS_ROOT/analysis/gibbs_anchor_transplant_scope.py" \
  "$TOOLS_ROOT/analysis/evaluate_svgp_frozen.py" \
  "$TOOLS_ROOT/gibbs_anchor_transplant_gpu.sbatch" \
  "$TOOLS_ROOT/gibbs_anchor_transplant_cpu.sbatch" \
  "$ANCHOR_ROOT/EXPECTED_HEAD"; do
  test -f "$required" || { echo "missing $required" >&2; exit 2; }
done
test "$(<"$ANCHOR_ROOT/EXPECTED_HEAD")" = \
  518db753c8a18e5a3d6349da9289568df36cb13e || {
    echo 'wrong donor numerical reference' >&2; exit 2;
  }
test ! -e "$CAMPAIGN_ROOT" || {
  echo "output already exists: $CAMPAIGN_ROOT" >&2; exit 2;
}

mkdir -p "$PARENT_ROOT/svgp/extended/runs" "$PRED_ROOT" "$CAMPAIGN_ROOT/dispatch"
cp "$ANCHOR_ROOT/EXPECTED_HEAD" "$PARENT_ROOT/EXPECTED_HEAD"
ln -s "$ANCHOR_ROOT/svgp/extended/runs/gibbs" \
  "$PARENT_ROOT/svgp/extended/runs/gibbs"

export AIPY TOOLS_ROOT ANCHOR_ROOT SCALABLE_GP_DATASET_DIR
export PARENT_ROOT CONT_ROOT PRED_ROOT
gpu=(--parsable --array=0-4 --partition=PA100q,NA10040q
     --nodelist=node02,node03,node13 --export=ALL)
cpu=(--parsable --partition=PA100q --nodelist=node02 --export=ALL)
gpu_script="$TOOLS_ROOT/gibbs_anchor_transplant_gpu.sbatch"
cpu_script="$TOOLS_ROOT/gibbs_anchor_transplant_cpu.sbatch"
stage_log="$CAMPAIGN_ROOT/dispatch"

parent_job="$(STAGE=parent150 sbatch "${gpu[@]}" \
  --job-name=gibbs_anchor_150 \
  --output="$stage_log/parent_%A_%a.out" --error="$stage_log/parent_%A_%a.err" \
  "$gpu_script")"
plan_job="$(STAGE=plan sbatch "${cpu[@]}" \
  --dependency="afterok:$parent_job" --job-name=gibbs_anchor_plan \
  --output="$stage_log/plan_%j.out" --error="$stage_log/plan_%j.err" \
  "$cpu_script")"
continue_job="$(STAGE=continue300 sbatch "${gpu[@]}" \
  --dependency="afterok:$plan_job" --job-name=gibbs_anchor_300 \
  --output="$stage_log/continue_%A_%a.out" --error="$stage_log/continue_%A_%a.err" \
  "$gpu_script")"
prepare_job="$(STAGE=prepare-prediction sbatch "${cpu[@]}" \
  --dependency="afterok:$continue_job" --job-name=gibbs_anchor_prep \
  --output="$stage_log/prepare_%j.out" --error="$stage_log/prepare_%j.err" \
  "$cpu_script")"
predict_job="$(STAGE=predict sbatch "${gpu[@]}" \
  --dependency="afterok:$prepare_job" --job-name=gibbs_anchor_pred \
  --output="$stage_log/predict_%A_%a.out" --error="$stage_log/predict_%A_%a.err" \
  "$gpu_script")"

printf 'CAMPAIGN_ROOT=%q\nANCHOR_ROOT=%q\nTOOLS_ROOT=%q\n' \
  "$CAMPAIGN_ROOT" "$ANCHOR_ROOT" "$TOOLS_ROOT" \
  > "$stage_log/jobs.env"
printf 'PARENT_JOB=%q\nPLAN_JOB=%q\nCONTINUE_JOB=%q\nPREPARE_JOB=%q\nPREDICT_JOB=%q\n' \
  "$parent_job" "$plan_job" "$continue_job" "$prepare_job" "$predict_job" \
  >> "$stage_log/jobs.env"
printf 'parent=%s plan=%s continuation=%s preparation=%s prediction=%s\n' \
  "$parent_job" "$plan_job" "$continue_job" "$prepare_job" "$predict_job"
printf 'Watch: bash %q %q\n' \
  "$TOOLS_ROOT/watch_gibbs_anchor_transplant300.sh" "$CAMPAIGN_ROOT"
