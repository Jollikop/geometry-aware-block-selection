#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?campaign root required}
AIPY=$(<"$ROOT/dispatch/python.txt")
JOB=$(<"$ROOT/dispatch/job_id.txt")
"$AIPY" "$ROOT/tools/gibbs_anchor_campaign.py" status --root "$ROOT"
printf '\n----- SLURM -----\n'
squeue -j "$JOB" -o '%.18i %.20j %.10T %.10M %.24R' || true
sacct -X -j "$JOB" -o JobID,State,Elapsed,ExitCode -n || true
printf '\n----- LATEST ACTIVITY -----\n'
shopt -s nullglob
for log in "$ROOT"/slurm/gibbs_anchor_"$JOB"_*.out; do
  printf '\n%s\n' "${log##*/}"
  tail -n 3 "$log"
done
