#!/usr/bin/env bash
set -euo pipefail
source "${CONT_ENV_FILE:-$HOME/aistats2026_transplant300_latest.env}"
JAX_PLATFORMS=cpu "$AIPY" "$CONT_TOOLS/analysis/continue_svgp_transplant.py" status --output "$CONT_ROOT"
printf '\n----- QUEUE -----\n'
squeue -j "$CONT_JOB" -o '%.18i %.18j %.9T %.10M %.22R' || true
printf '\n----- ACCOUNTING -----\n'
sacct -X -j "$CONT_JOB" -o JobID,State,Elapsed,ExitCode || true
printf '\n----- LATEST ACTIVITY -----\n'
shopt -s nullglob
for LOG in "$CONT_ROOT"/slurm/continue_"${CONT_JOB}"_*.out; do
  printf '%s\n' "${LOG##*/}"
  tail -n 3 "$LOG"
done
