#!/usr/bin/env bash
# Exploratory SBT-guided SGPR experiment: one seed per GPU task.
# Required: SGPR_GUIDED_OUT and a GPU-capable project environment.
#SBATCH --job-name=sbt_guided_sgpr
#SBATCH --array=0-4%5
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_sbt_guided_sgpr_%A_%a.out

set -euo pipefail

SGPR_DIR="${SGPR_BENCHMARK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SIM_ROOT="${SIM_ROOT:-$(cd "$SGPR_DIR/.." && pwd)}"
ROOT="${PROJECT_ROOT:-$(cd "$SIM_ROOT/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$ROOT/../.." && pwd)}"
VENV_ACTIVATE="${PROJECT_VENV_ACTIVATE:-$WORKSPACE_ROOT/.venv/bin/activate}"
OUT_DIR="${SGPR_GUIDED_OUT:?set SGPR_GUIDED_OUT to a separate exploratory directory}"

test -f "$VENV_ACTIVATE" || {
  echo "Virtual environment not found: $VENV_ACTIVATE" >&2
  exit 1
}
source "$VENV_ACTIVATE"

export PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
unset JAX_PLATFORMS || true

CURRENT_HEAD=$(git -C "$ROOT" rev-parse HEAD)
if [[ -n "${EXPECTED_HEAD:-}" && "$CURRENT_HEAD" != "$EXPECTED_HEAD" ]]; then
  echo "Expected commit $EXPECTED_HEAD, found $CURRENT_HEAD" >&2
  exit 1
fi

python - <<'PY'
import jax

devices = jax.devices()
print("JAX devices:", devices)
if not any(device.platform == "gpu" for device in devices):
    raise SystemExit("JAX is not using the Slurm GPU")
PY

python -u "$SGPR_DIR/analysis/explore_sbt_guided_sgpr.py" \
  --dataset "${SGPR_GUIDED_DATASET:-kin40k}" \
  --kernel "${SGPR_GUIDED_KERNEL:-rbf}" \
  --m "${SGPR_GUIDED_M:-256}" \
  --seeds "$SLURM_ARRAY_TASK_ID" \
  --n-train 5000 \
  --n-test "${SGPR_GUIDED_N_TEST:-2000}" \
  --block-size "${SGPR_GUIDED_BLOCK_SIZE:-500}" \
  --sbt-schedule "${SGPR_GUIDED_SBT_SCHEDULE:-15,15,20,25,25}" \
  --target-steps "${SGPR_GUIDED_TARGET_STEPS:-100}" \
  --out-dir "$OUT_DIR" \
  --no-summary
