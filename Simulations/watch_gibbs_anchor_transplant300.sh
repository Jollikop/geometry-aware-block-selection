#!/usr/bin/env bash
set -euo pipefail
: "${1:?usage: watch_gibbs_anchor_transplant300.sh CAMPAIGN_ROOT}"
campaign="$1"
jobs="$campaign/dispatch/jobs.env"
test -f "$jobs" || { echo "missing $jobs"; exit 2; }
# shellcheck source=/dev/null
source "$jobs"
if test -f "$campaign/dispatch/cert_job.env"; then
  # shellcheck source=/dev/null
  source "$campaign/dispatch/cert_job.env"
fi

count_files() {
  local directory="$1" name="$2"
  if test -d "$directory"; then
    find "$directory" -name "$name" -type f | wc -l | tr -d ' '
  else
    printf '0'
  fi
}

printf '\n===== GIBBS ANCHOR → SVGP TRANSPLANT | %s =====\n' "$(date)"
printf 'Donor anchor certificati: 320/320 (read-only)\n'
printf 'Transplant 150 epoche: %s/320\n' \
  "$(count_files "$campaign/parent150/svgp/transplant/runs/gibbs" training.json)"
for epoch in 200 250 300; do
  printf 'Transplant %d epoche: %s/320\n' "$epoch" \
    "$(count_files "$campaign/continued300/svgp/transplant/runs/gibbs" "epoch$epoch.json")"
done
printf 'Continuation complete: %s/320\n' \
  "$(count_files "$campaign/continued300/svgp/transplant/runs/gibbs" COMPLETE.json)"
printf 'Prediction SVGP separate: %s/320\n' \
  "$(count_files "$campaign/predictions/svgp/transplant/runs/gibbs" prediction.json)"
if test -f "$campaign/GIBBS_ANCHOR_TRANSPLANT_CERTIFICATE.json"; then
  printf 'Certification: PASSED (320/320)\n'
else
  printf 'Certification: pending\n'
fi

printf '\n----- JOBS -----\n'
all_jobs="$PARENT_JOB,$PLAN_JOB,$CONTINUE_JOB,$PREPARE_JOB,$PREDICT_JOB"
if test -n "${CERT_JOB:-}"; then all_jobs="$all_jobs,$CERT_JOB"; fi
squeue -j "$all_jobs" \
  -o '%.18i %.22j %.9T %.10M %.22R' || true
printf '\n----- ACCOUNTING -----\n'
sacct -X -j "$all_jobs" \
  -o JobID,JobName%22,State,Elapsed,ExitCode | tail -n 25 || true
printf '\n----- LATEST ACTIVITY -----\n'
for stage in parent continue predict; do
  latest="$(find "$campaign/dispatch" -name "${stage}_*.out" -type f \
    -exec stat -c '%Y %n' {} + 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if test -n "$latest"; then
    printf '%s\n' "$(basename "$latest")"
    tail -n 3 "$latest"
  fi
done
printf '\nNo stage silently requeues; a failed dependency prevents later stages.\n'
