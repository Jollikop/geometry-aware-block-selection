#!/usr/bin/env bash

set -u
ENV_FILE="${SGPR_FULL_ENV_FILE:-$HOME/aistats2026_sgpr_full_v23_jobs.env}"
[[ -f "$ENV_FILE" ]] || { echo "missing $ENV_FILE" >&2; exit 2; }
source "$ENV_FILE"

bar() {
  local done=$1 total=$2 width=24 filled empty
  (( done > total )) && done=$total
  filled=$((done * width / total)); empty=$((width - filled))
  printf '['; printf '%*s' "$filled" '' | tr ' ' '#'
  printf '%*s' "$empty" '' | tr ' ' '.'
  printf '] %4d/%-4d' "$done" "$total"
}

count_paths() {
  find "$1" -type f -name "$2" 2>/dev/null | wc -l | tr -d ' '
}

read -r valid invalid transplant_valid transplant_invalid < <(
  PYTHONPATH="$PROJECT_ROOT/Simulations" "$AIPY" - "$FINAL_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
valid = invalid = tv = ti = 0
for path in (root / "runs").rglob("metrics.json"):
    try:
        row = json.loads(path.read_text())
        ok = (row.get("implementation") == "sgpr-collapsed-sbt-jax-v23"
              and row.get("convergence_passed") is True
              and (path.parent / "params.npz").is_file())
    except Exception:
        ok = False
    valid += int(ok); invalid += int(not ok)
for path in (root / "transplant").rglob("metrics.json"):
    try:
        payload = json.loads(path.read_text()); rows = payload.get("rows", [])
        ok = (len(rows) == 4 and all(
            r.get("transplant_convergence_reached") is True for r in rows))
    except Exception:
        ok = False
    tv += int(ok); ti += int(not ok)
print(valid, invalid, tv, ti)
PY
)

echo "===== $(basename "$FINAL_ROOT") | $(date '+%F %T') ====="
echo
echo "----- RISULTATI CERTIFICATI -----"
for m in 128 256 512; do
  main=$(find "$FINAL_ROOT/runs" -path "*/M$m/*/metrics.json" 2>/dev/null | wc -l | tr -d ' ')
  params=$(find "$FINAL_ROOT/runs" -path "*/M$m/*/params.npz" 2>/dev/null | wc -l | tr -d ' ')
  transplant=$(count_paths "$FINAL_ROOT/transplant/M$m" metrics.json)
  printf 'M=%-3s main files  ' "$m"; bar "$main" 120; echo
  printf '      params      '; bar "$params" 120; echo
  printf '      transplant  '; bar "$transplant" 20; echo
done
printf 'main validi: %s/360; invalidi presenti: %s\n' "$valid" "$invalid"
printf 'transplant validi: %s/60; invalidi presenti: %s\n' \
  "$transplant_valid" "$transplant_invalid"

echo
echo "----- WORKERS -----"
sacct -j "$SGPR_JOB" -X -n -o JobID,State,Elapsed,NodeList 2>/dev/null | \
  sed '/^[[:space:]]*$/d'

echo
echo "----- QUEUE -----"
squeue -j "$SGPR_JOB" -o "%.18i %.18j %.9T %.10M %.24R" 2>/dev/null

echo
echo "----- ULTIMA ATTIVITA -----"
find "$FINAL_ROOT/slurm" -type f -name '*.out' -print0 2>/dev/null |
  xargs -0 ls -1t 2>/dev/null | head -n 5 |
  while read -r log; do
    printf '%-32s ' "$(basename "$log")"
    tail -n 120 "$log" |
      grep -E 'CONVERGENCE|FAILED CONVERGENCE|rmse=|COMPLETE|INCOMPLETE' |
      tail -n 1
  done

echo
echo "----- ERRORI CORRENTI -----"
grep -hE 'Traceback|Out of memory|no usable GPU|giving up|INCOMPLETE' \
  "$FINAL_ROOT"/slurm/sgpr_full_"$SGPR_JOB"_*.out 2>/dev/null | tail -n 16 || true

echo
if (( valid == 360 && invalid == 0 && transplant_valid == 60 && transplant_invalid == 0 )); then
  echo "RELEASE_READY=yes"
else
  echo "RELEASE_READY=no"
fi
