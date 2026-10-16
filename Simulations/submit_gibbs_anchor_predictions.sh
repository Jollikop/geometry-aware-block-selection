#!/usr/bin/env bash
# Dispatch the five disjoint shards to the A100 allocations probed on 2026-09-26.
set -euo pipefail
OUTPUT=${1:?prediction output root}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
test -f "$OUTPUT/prediction_plan.sha256"
test -f "$OUTPUT/tools/gibbs_anchor_predictions.py"
cp "$HERE/gibbs_anchor_prediction_worker.sbatch" "$OUTPUT/tools/"
cp "$HERE/watch_gibbs_anchor_predictions.sh" "$OUTPUT/tools/"
# Refuse duplicate submissions, including after a partial submission failure.
mkdir "$OUTPUT/dispatch/submission_started"
JOB=$(sbatch --parsable --partition=NA10040q --nodelist=node13 --array=0-2%3 \
  --output="$OUTPUT/slurm/predict_%A_%a.out" \
  "$OUTPUT/tools/gibbs_anchor_prediction_worker.sbatch" "$OUTPUT")
printf '%s\n' "${JOB%%;*}" >> "$OUTPUT/dispatch/jobs.txt"
JOB=$(sbatch --parsable --partition=PA100q --nodelist=node03 --array=3-4%2 \
  --output="$OUTPUT/slurm/predict_%A_%a.out" \
  "$OUTPUT/tools/gibbs_anchor_prediction_worker.sbatch" "$OUTPUT")
printf '%s\n' "${JOB%%;*}" >> "$OUTPUT/dispatch/jobs.txt"
printf 'Prediction output: %s\n' "$OUTPUT"
cat "$OUTPUT/dispatch/jobs.txt"
printf 'watch -n 15 bash %q %q\n' "$OUTPUT/tools/watch_gibbs_anchor_predictions.sh" "$OUTPUT"
