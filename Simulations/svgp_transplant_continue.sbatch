#!/usr/bin/env bash
#SBATCH --job-name=transplant300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${CONT_TOOLS:?}" "${CONT_ROOT:?}" "${AIPY:?}" "${SCALABLE_GP_DATASET_DIR:?}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH="$CONT_TOOLS"
unset XLA_PYTHON_CLIENT_MEM_FRACTION
exec "$AIPY" -u "$CONT_TOOLS/analysis/continue_svgp_transplant.py" worker \
  --output "$CONT_ROOT" --worker "$SLURM_ARRAY_TASK_ID"
