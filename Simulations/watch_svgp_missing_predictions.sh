#!/usr/bin/env bash
# Read-only status for the new donor predictions, plus completed transplants.
set -uo pipefail
source "$HOME/aistats2026_svgp_missing_prediction_jobs.env"
if ! command -v squeue >/dev/null 2>&1 && type module >/dev/null 2>&1; then
  module load slurm
fi
printf '\n===== SVGP PREDICTION MANCANTI | %s =====\n' "$(date)"
PYTHONDONTWRITEBYTECODE=1 "$AIPY" - "$PRED_OUTPUT" "$FINAL_ROOT" <<'PY'
import json, sys
from pathlib import Path
out, root = map(Path, sys.argv[1:])
plan = json.loads((out/'prediction_plan.json').read_text())
for stage in ('stationary','extended'):
    pending = [r for r in plan['pending'] if r['stage']==stage]
    saved = sum((out/r['run']/'prediction.json').is_file() for r in pending)
    preserved = plan['counts'][stage]['preserved']
    print(f'{stage:12s} predicted={preserved+saved}/2560 | preservate={preserved} nuove={saved}/{len(pending)}')
transplant = Path(str(root)+'_transplant_live_eval_1036173')/'svgp/transplant/runs'
print(f'transplant   predicted={sum(1 for _ in transplant.glob("**/prediction.json"))}/5120')
for path in sorted((out/'state').glob('worker_*.json')):
    row=json.loads(path.read_text())
    print(f"worker {row['worker']}: {row['status']} {row['completed']}/{row['assigned']} failed={row['failed']}")
print('Conteggi originali + risultati separati; non e una certificazione del freeze.')
PY
printf '\n----- CODA -----\n'
squeue -j "$PRED_JOB" -o '%.18i %.20j %.10T %.12M %.24R' 2>/dev/null || true
sacct -X -j "$PRED_JOB" -n -o JobID,State,Elapsed,ExitCode 2>/dev/null || true
printf '\n----- ULTIMA ATTIVITA -----\n'
shopt -s nullglob
for log in "$PRED_OUTPUT"/slurm/missing_"$PRED_JOB"_*.out; do
  printf '%s\n' "$(basename "$log")"
  tail -n 3 "$log" | cut -c1-260
done
