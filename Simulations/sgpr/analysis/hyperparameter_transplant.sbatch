#!/usr/bin/env bash
#SBATCH --job-name=sgpr_transplant
#SBATCH --array=0-4%3
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail

SGPR_DIR="${SGPR_BENCHMARK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SIM_ROOT="${SIM_ROOT:-$(cd "$SGPR_DIR/.." && pwd)}"
source "${PROJECT_VENV_ACTIVATE:?set PROJECT_VENV_ACTIVATE}"

export PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
unset JAX_PLATFORMS || true

python - <<'PY'
import jax
devices = jax.devices()
print("JAX devices:", devices)
assert len(devices) == 1 and devices[0].platform == "gpu"
PY

python -u "$SGPR_DIR/analysis/explore_hyperparameter_transplant.py" \
  --dataset "${SGPR_TRANSPLANT_DATASET:-kin40k}" \
  --kernel "${SGPR_TRANSPLANT_KERNEL:-rbf}" \
  --m "${SGPR_TRANSPLANT_M:-256}" \
  --seeds "$SLURM_ARRAY_TASK_ID" \
  --n-train "${SGPR_TRANSPLANT_N_TRAIN:-5000}" \
  --n-test "${SGPR_TRANSPLANT_N_TEST:-2000}" \
  --block-size "${SGPR_TRANSPLANT_BLOCK_SIZE:-500}" \
  --donor-steps "${SGPR_TRANSPLANT_DONOR_STEPS:-30000}" \
  --donor-schedule "${SGPR_TRANSPLANT_DONOR_SCHEDULE:-15,15,20,25,29925}" \
  --transplant-steps "${SGPR_TRANSPLANT_TARGET_STEPS:-10000}" \
  --out-dir "${SGPR_TRANSPLANT_OUT:?set SGPR_TRANSPLANT_OUT}" \
  --no-summary
