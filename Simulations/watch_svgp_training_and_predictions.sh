#!/usr/bin/env bash
# Read-only combined dashboard; retain the user's existing training dashboard.
set -uo pipefail
ENV_FILE=${SVGP_TRAINING_ENV_FILE:-"$HOME/aistats2026_svgp_all_training_jobs.env"}
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing training environment: $ENV_FILE" >&2
  exit 2
fi
source "$ENV_FILE"
# Non-interactive SSH sessions may not have loaded the Slurm module.
if ! command -v squeue >/dev/null 2>&1 && type module >/dev/null 2>&1; then
  module load slurm
fi
: "${FINAL_ROOT:?FINAL_ROOT missing from training environment}"
JOB=${1:-1036173}
[[ "$JOB" =~ ^[0-9]+$ ]] || exit 2
HELPER_ROOT="${FINAL_ROOT}_transplant_live_eval_${JOB}"

if [[ -f "$HOME/watch_svgp_training_only.sh" ]]; then
  bash "$HOME/watch_svgp_training_only.sh"
else
  echo "Original training watch missing: $HOME/watch_svgp_training_only.sh"
fi

printf '\n----- TRANSPLANT PREDICTION HELPERS (OUTPUT SEPARATO) -----\n'
total=0
if [[ -d "$HELPER_ROOT/svgp/transplant/runs" ]]; then
  total=$(find "$HELPER_ROOT/svgp/transplant/runs" -type f -name prediction.json -print 2>/dev/null | wc -l | tr -d ' ')
fi
printf 'TOTALE prediction salvate: %s/5120 (conteggio file, non certificazione finale)\n' "$total"
for worker in 2 3; do
  seed_a=$worker
  seed_b=$((worker + 4))
  count=0
  if [[ -d "$HELPER_ROOT/svgp/transplant/runs" ]]; then
    count=$(find "$HELPER_ROOT/svgp/transplant/runs" -type f \
      \( -path "*/seed$seed_a/prediction.json" -o -path "*/seed$seed_b/prediction.json" \) \
      -print 2>/dev/null | wc -l | tr -d ' ')
  fi
  printf 'helper %s (seed %s,%s): saved=%s/1024\n' "$worker" "$seed_a" "$seed_b" "$count"
done
printf 'Queste prediction NON aggiornano i contatori del freeze originale.\n'

printf '\n----- STEP ATTIVI: TRAINING + PREDICTION -----\n'
squeue --steps -j "$JOB" -o '%.20i %.14j %.10M %.20N' 2>/dev/null || true

QUEUE_FILE="$HELPER_ROOT/wave_dispatch_$JOB/queued_jobs.txt"
if [[ -f "$QUEUE_FILE" ]]; then
  ids=$(awk -F';' '/^[0-9]+(;.*)?$/ {print $1}' "$QUEUE_FILE" | paste -sd, -)
  if [[ -n "$ids" ]]; then
    printf '\n----- TRE JOB PREDICTION: STATO E DIPENDENZA -----\n'
    squeue -j "$ids" -o '%.18i %.22j %.10T %.12M %.24R' 2>/dev/null || true
    sacct -X -j "$ids" -n -o JobID,State,Elapsed,ExitCode 2>/dev/null || true
  fi
fi

printf '\n----- ULTIMA ATTIVITA PREDICTION -----\n'
logs=()
patterns=("helper_${JOB}_2_*.out" "helper_${JOB}_3_*.out")
if [[ -d "$HELPER_ROOT/wave_dispatch_$JOB" ]]; then
  patterns=("remaining_${JOB}_0_*.out" "remaining_${JOB}_1_*.out" "finish_0_*.out" "finish_1_*.out" "finish_2_*.out")
fi
for pattern in "${patterns[@]}"; do
  # Patterns above are fixed internally; filenames contain no user shell input.
  log=$(find "$HELPER_ROOT/slurm" -maxdepth 1 -name "$pattern" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d ' ' -f2-)
  if [[ -n "$log" ]]; then
    logs+=("$log")
    printf '%s\n' "$(basename "$log")"
    tail -n 3 "$log" | sed 's/^/  /'
  else
    printf '%s: log non ancora presente\n' "$pattern"
  fi
done

printf '\n----- ERRORI PREDICTION (CODA DEI LOG) -----\n'
errors=$(
  for log in "${logs[@]}"; do
    tail -n 200 "$log" | grep -E 'Traceback|Error:|FAILED|Stopping after|CANCELLED|TIME LIMIT' || true
  done
)
if [[ -n "$errors" ]]; then
  printf '%s\n' "$errors" | tail -n 6
else
  echo 'none'
fi
launcher_log="$HELPER_ROOT/helpers.log"
if [[ -f "$HELPER_ROOT/wave_dispatch_$JOB/active_launcher.log" ]]; then
  launcher_log="$HELPER_ROOT/wave_dispatch_$JOB/active_launcher.log"
fi
if [[ -f "$launcher_log" ]]; then
  printf '\n----- STATO LAUNCHER HELPERS -----\n'
  tail -n 3 "$launcher_log"
fi
