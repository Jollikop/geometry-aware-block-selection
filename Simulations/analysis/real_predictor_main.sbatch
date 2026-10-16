#!/usr/bin/env bash
#SBATCH --job-name=real_predictor
#SBATCH --array=0-4%5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

source /etc/profile.d/modules.sh
set -euo pipefail
module load slurm/slurm/24.11

: "${REAL_PREDICTOR_ROOT:?set REAL_PREDICTOR_ROOT}"
: "${REAL_PREDICTOR_SCRIPT:?set REAL_PREDICTOR_SCRIPT}"
: "${AIPY:?set AIPY to the pinned Python 3.12 environment}"
: "${SCALABLE_GP_DATASET_DIR:?set SCALABLE_GP_DATASET_DIR}"

export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

"$AIPY" -u "$REAL_PREDICTOR_SCRIPT" worker \
  --worker "$SLURM_ARRAY_TASK_ID" --workers 5 \
  --output "$REAL_PREDICTOR_ROOT"
