#!/usr/bin/env bash
set -euo pipefail
: "${CONTROL_OUTPUTS:?space-separated new E1/E2/E3 output roots}"
: "${AIPY:?set Python}"
"$AIPY" - $CONTROL_OUTPUTS <<'PY'
import json
from pathlib import Path
import sys
from datetime import datetime
print(datetime.now().isoformat(timespec="seconds"))
for name in sys.argv[1:]:
    root = Path(name)
    plan = json.loads((root / "plan.json").read_text())
    records = list((root / "hard").rglob("prediction.json"))
    valid = sum(json.loads(p.read_text())["valid"] for p in records)
    print(f"{plan['experiment']} {plan['scope']}: saved={len(records)}/{len(plan['runs'])} valid={valid} rejected={len(records)-valid}")
    for path in sorted((root / "state").glob("worker*.json")):
        state = json.loads(path.read_text())
        print(f"  {path.stem}: {state['processed']}/{state['assigned']} failed={len(state['failures'])} complete={state['complete']}")
    cert = root / "PARTITION_CONTROL_CERTIFICATE.json"
    if cert.is_file():
        print("  certificate complete=", json.loads(cert.read_text())["complete"])
PY
squeue -u "$USER" -o '%.18i %.22j %.9T %.10M %.24R'
