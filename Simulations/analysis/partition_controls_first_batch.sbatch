#!/usr/bin/env bash
# E1(a) and E2(a), one process per GPU, no training and no writes to inputs.
#SBATCH --job-name=partctrl_E1E2
#SBATCH --array=0-4%5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail
: "${AIPY:?set pinned Python}"
: "${TOOLS_DIR:?set evaluator snapshot}"
: "${E1_ROOT:?set new E1 output root}"
: "${E2_ROOT:?set new E2 output root}"
: "${SCALABLE_GP_DATASET_DIR:?set frozen data cache}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
status=0
for root in "$E1_ROOT" "$E2_ROOT"; do
  "$AIPY" -u "$TOOLS_DIR/partition_controls.py" worker --output "$root" \
    --worker "$SLURM_ARRAY_TASK_ID" --workers 5 || status=1
done
exit "$status"
