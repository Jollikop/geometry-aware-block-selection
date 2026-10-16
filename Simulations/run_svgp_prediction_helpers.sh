#!/usr/bin/env bash
# Opportunistic, disjoint prediction steps inside a still-running FOUR-worker
# bundle. Does not redistribute training, modify the allocation or touch inputs.
set -euo pipefail
if [[ $# != 4 ]]; then
  echo "usage: $0 JOB_ID SOURCE_PROJECT_ROOT FROZEN_ROOT OUTPUT_ROOT" >&2
  exit 2
fi
JOB=$1
SOURCE=$2
FROZEN=$3
OUTPUT=$4
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${AIPY:?export AIPY}"
: "${SCALABLE_GP_DATASET_DIR:?export SCALABLE_GP_DATASET_DIR}"
[[ "$JOB" =~ ^[0-9]+$ && -x "$AIPY" ]]
[[ "$(squeue -j "$JOB" -h -o '%T')" == RUNNING ]] || {
  echo "Parent job is not running; use a separate prediction job after training." >&2
  exit 2
}
# The original four-way seed schedule is 0,4,8 / 1,5,9 / 2,6 / 3,7.
# Confirm its exact completion markers before reading either finished shard.
STATE="$FROZEN/state/svgp-resume-$JOB"
for worker in 2 3; do
  [[ -f "$STATE/worker.$worker.complete" && -f "$STATE/transplant.$worker.done" ]] || {
    echo "Worker $worker transplant completion is not confirmed; not launching." >&2
    exit 2
  }
done
if [[ -e "$STATE/worker.4.complete" || -e "$STATE/donors.4.ready" ]]; then
  echo "Five-way sharding detected; this helper requires the four-worker schedule." >&2
  exit 2
fi
"$AIPY" - "$FROZEN" "$OUTPUT" <<'PY'
from pathlib import Path
import sys
a,b=(Path(p).resolve() for p in sys.argv[1:])
if a==b or a in b.parents or b in a.parents:
    raise SystemExit('Prediction output must be disjoint from the training root')
PY
mkdir -p "$OUTPUT/slurm"
# Hold a local/cluster-wide advisory lock for this output session. No training
# artifacts or existing evaluation files are deleted on duplicate launches.
exec 9>"$OUTPUT/.helpers.lock"
flock -n 9 || { echo "Helpers already running for this output." >&2; exit 2; }
pids=()
for worker in 2 3; do
  echo "Starting prediction helper $worker: seeds $worker $((worker + 4))"
  # No --overlap, no manual CUDA_VISIBLE_DEVICES, no new GPU allocation.
  # If CPU/memory/GPU reservations cannot be satisfied, fail rather than share.
  srun --jobid="$JOB" --immediate=20 --exclusive --exact \
    --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=64G \
    --output="$OUTPUT/slurm/helper_${JOB}_${worker}_%s.out" \
    env JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
      PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
      "SCALABLE_GP_DATASET_DIR=$SCALABLE_GP_DATASET_DIR" \
      "$AIPY" "$HERE/analysis/evaluate_svgp_frozen.py" \
      --source-sim "$SOURCE/Simulations" --root "$FROZEN" --output "$OUTPUT" \
      --stage transplant --seeds "$worker" "$((worker + 4))" \
      --measurement-context concurrent-training \
      --require-single-gpu --reps 10 --reference-points 8 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
echo "Helpers ended (status=$status). Saved predictions remain in $OUTPUT."
echo "Parent completion can terminate helpers; incomplete files are not accepted as results."
exit "$status"
