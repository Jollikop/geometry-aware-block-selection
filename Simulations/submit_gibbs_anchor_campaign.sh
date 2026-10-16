#!/usr/bin/env bash
# A fresh dispatch only; never cancels jobs or edits the original experiment.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo "usage: $0 PREPARED_ROOT PARTITION NODE|-" >&2
  exit 2
fi
ROOT=$(cd "$1" && pwd)
PARTITION=$2
NODE=$3
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ -f "$ROOT/campaign.json" ]]
AIPY=${AIPY:?set AIPY to the pinned Python 3.12 executable}
[[ -x "$AIPY" ]]
"$AIPY" "$ROOT/tools/gibbs_anchor_campaign.py" status --root "$ROOT"
# An atomic directory claim stops an accidental double submission. In case of
# sbatch failure, preserve this marker and inspect it instead of auto-retrying.
mkdir "$ROOT/dispatch"
cp "$TOOLS/gibbs_anchor_worker.sbatch" "$ROOT/tools/"
cp "$TOOLS/watch_gibbs_anchor_campaign.sh" "$ROOT/tools/"
args=(--parsable --array=0-4%5 --partition="$PARTITION" --export=ALL \
      --output="$ROOT/slurm/gibbs_anchor_%A_%a.out")
if [[ "$NODE" != - ]]; then
  # A single node avoids interpreting a comma-separated nodelist as a request
  # for all listed nodes. Use '-' to let this partition place the five tasks.
  [[ "$NODE" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Use one node or '-'" >&2; exit 2; }
  args+=(--nodelist="$NODE")
fi
JOB=$(sbatch "${args[@]}" "$ROOT/tools/gibbs_anchor_worker.sbatch" "$ROOT" "$AIPY")
JOB=${JOB%%;*}
printf '%s\n' "$JOB" > "$ROOT/dispatch/job_id.txt"
printf '%s\n' "$AIPY" > "$ROOT/dispatch/python.txt"
printf 'JOB=%s ROOT=%s\n' "$JOB" "$ROOT"
printf 'watch -n 15 bash %q %q\n' "$ROOT/tools/watch_gibbs_anchor_campaign.sh" "$ROOT"
