#!/usr/bin/env bash
#SBATCH --job-name=sgpr_theta
#SBATCH --array=0-4%5
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_sgpr_theta_%A_%a.out

set -euo pipefail

SGPR_DIR="${SGPR_BENCHMARK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SIM_ROOT="${SIM_ROOT:-$(cd "$SGPR_DIR/.." && pwd)}"
ROOT="${PROJECT_ROOT:-$(cd "$SIM_ROOT/.." && pwd)}"
VENV_ACTIVATE="${PROJECT_VENV_ACTIVATE:?set PROJECT_VENV_ACTIVATE}"
OUT_DIR="${SGPR_THETA_OUT:?set SGPR_THETA_OUT}"

source "$VENV_ACTIVATE"
export PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
unset JAX_PLATFORMS || true

python - <<'PY'
import jax
d = jax.devices()
print("JAX devices:", d)
assert len(d) == 1 and d[0].platform == "gpu"
PY

python -u "$SGPR_DIR/analysis/explore_fixed_theta_bounds.py" \
  --dataset "${SGPR_THETA_DATASET:-kin40k}" \
  --kernel "${SGPR_THETA_KERNEL:-rbf}" \
  --m "${SGPR_THETA_M:-256}" \
  --seeds "$SLURM_ARRAY_TASK_ID" \
  --n-train 5000 \
  --n-test "${SGPR_THETA_N_TEST:-2000}" \
  --block-size "${SGPR_THETA_BLOCK_SIZE:-500}" \
  --steps "${SGPR_THETA_STEPS:-1000}" \
  --refresh-schedule "${SGPR_THETA_SCHEDULE:-15,15,20,25,29925}" \
  --out-dir "$OUT_DIR" \
  --no-summary
