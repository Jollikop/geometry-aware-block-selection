#!/usr/bin/env bash
set -euo pipefail
OUTPUT=${1:?pass the exact prediction output directory}
AIPY="${AIPY:?set AIPY to the python of the release environment}"
SLURM_BIN=/cm/local/apps/slurm/current/bin
export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/etc/slurm/slurm.conf}
date
"$AIPY" "$OUTPUT/tools/gibbs_anchor_predictions.py" status --output "$OUTPUT"
if [[ -f "$OUTPUT/dispatch/jobs.txt" ]]; then
  JOBS=$(paste -sd, "$OUTPUT/dispatch/jobs.txt")
  printf '\n----- JOBS -----\n'
  "$SLURM_BIN/squeue" -j "$JOBS" -o '%.18i %.22j %.10T %.10M %.24R' || true
  "$SLURM_BIN/sacct" -j "$JOBS" -X -n -o JobID,State,Elapsed,ExitCode || true
fi
printf '\n----- LATEST ACTIVITY -----\n'
shopt -s nullglob
for LOG in "$OUTPUT"/slurm/predict_*.out; do
  printf '\n%s\n' "${LOG##*/}"
  tail -n 3 "$LOG"
done
