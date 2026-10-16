#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?usage: watch_dense_cutoff_sweep.sh OUTPUT_ROOT [ARRAY_JOB [CERT_JOB]]}"
ARRAY_JOB="${2:-}"
CERT_JOB="${3:-}"
PYTHON_BIN="${AIPY:-python3}"
"$PYTHON_BIN" - "$ROOT" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

root = Path(sys.argv[1])
plan = json.loads((root / "plan.json").read_text())
paths = [root / row["output_relative"] / "cutoff_sweep.json" for row in plan["runs"]]
saved = [p for p in paths if p.is_file()]
variants = rejected = 0
for path in saved:
    payload = json.loads(path.read_text())
    variants += len(payload["results"])
    rejected += sum(not row["valid"] for row in payload["results"])
filled = int(24 * len(saved) / max(len(paths), 1))
print(f"===== E4 DENSE CUTOFF | {datetime.now():%Y-%m-%d %H:%M:%S} =====")
print(f"References [{'#' * filled}{'.' * (24-filled)}] {len(saved)}/{len(paths)}")
print(f"Cutoff variants: {variants}/{len(paths) * len(plan['scales'])}; rejected variants: {rejected}")
print("File counts only; final certification below.")
print("\n----- WORKERS -----")
for worker in range(5):
    state_path = root / "state" / f"worker{worker}.json"
    if not state_path.is_file():
        print(f"worker {worker}: waiting")
        continue
    state = json.loads(state_path.read_text())
    print(f"worker {worker}: {state['processed']}/{state['assigned']} failed={len(state['failures'])} complete={state['complete']}")
certificate = root / "DENSE_CUTOFF_CERTIFICATE.json"
if certificate.is_file():
    cert = json.loads(certificate.read_text())
    print(f"\nCertificate: complete={cert['complete']} scale10_reproduced={cert['scale10_all_reproduced']} rejected_variants={cert['rejected_count']}")
else:
    print("\nCertificate: pending")
print("\n----- LATEST ACTIVITY -----")
for log in sorted((root / "slurm").glob("*.out")):
    print(log.name)
    lines = log.read_text(errors="replace").splitlines()
    print("\n".join(lines[-3:]) or "  waiting for output")
PY
if ! command -v squeue >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
    set +u
    source /etc/profile.d/modules.sh
    module load slurm
    set -u
fi
if [[ -n "$ARRAY_JOB" ]] && command -v squeue >/dev/null 2>&1; then
    printf '\n----- QUEUE -----\n'
    JOBS="$ARRAY_JOB"
    [[ -z "$CERT_JOB" ]] || JOBS="$JOBS,$CERT_JOB"
    squeue -j "$JOBS" -o '%.18i %.22j %.9T %.10M %.24R'
fi
