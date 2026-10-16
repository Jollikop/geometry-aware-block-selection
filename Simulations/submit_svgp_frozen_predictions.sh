#!/usr/bin/env bash
# Separate read-only evaluation: one GPU for homogeneous inference timings.
set -euo pipefail
if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 SOURCE_PROJECT_ROOT FROZEN_ROOT OUTPUT_ROOT PARTITION NODE STAGE[transplant]" >&2
  exit 2
fi
SOURCE_PROJECT_ROOT=$1
FROZEN_ROOT=$2
OUTPUT_ROOT=$3
PARTITION=$4
NODE=$5
STAGE=${6:-transplant}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AIPY=${AIPY:?export AIPY as an absolute Python interpreter path}
SCALABLE_GP_DATASET_DIR=${SCALABLE_GP_DATASET_DIR:?export frozen dataset cache path}
[[ -x "$AIPY" && -f "$SOURCE_PROJECT_ROOT/Simulations/svgp/engine.py" ]]
[[ -f "$FROZEN_ROOT/EXPECTED_HEAD" ]]
case "$STAGE" in stationary|extended|transplant) ;; *) exit 2 ;; esac
# Resolve symlinks too; no eval logs/metrics may be written inside the freeze.
"$AIPY" - "$FROZEN_ROOT" "$OUTPUT_ROOT" <<'PY'
from pathlib import Path
import sys
a,b=map(lambda x:Path(x).resolve(),sys.argv[1:])
if a==b or a in b.parents or b in a.parents:
    raise SystemExit('Evaluation output must be disjoint from the frozen root')
PY
mkdir -p "$OUTPUT_ROOT/slurm"
printf -v command '%q ' env JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  "SCALABLE_GP_DATASET_DIR=$SCALABLE_GP_DATASET_DIR" \
  "$AIPY" "$HERE/analysis/evaluate_svgp_frozen.py" \
  --source-sim "$SOURCE_PROJECT_ROOT/Simulations" \
  --root "$FROZEN_ROOT" --output "$OUTPUT_ROOT" --stage "$STAGE" \
  --measurement-context "${PRED_MEASUREMENT_CONTEXT:-unspecified}" \
  --test-chunk 256 --train-chunk 512 --block-batch 4 --factor-cache-mib 128 \
  --reps 10 --reference-points 8
printf -v preflight '%q ' env JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$AIPY" -c 'import jax; import jax.numpy as jnp; d=jax.devices(); print(d, flush=True); assert len(d)==1 and d[0].platform=="gpu"; jax.block_until_ready(jnp.ones(8)+1)'
job=$(sbatch --parsable --job-name="svgp_pred_$STAGE" \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=64G \
  --partition="$PARTITION" --nodelist="$NODE" --time=6-23:59:00 --no-requeue \
  --output="$OUTPUT_ROOT/slurm/predict_%j.out" --wrap="$preflight && exec $command")
printf 'Prediction job: %s\nLog: %s/slurm/predict_%s.out\n' "$job" "$OUTPUT_ROOT" "$job"
printf 'Frozen input (unchanged): %s\nSeparate evaluation: %s\n' "$FROZEN_ROOT" "$OUTPUT_ROOT"
