#!/usr/bin/env bash
#SBATCH --job-name=dense_cutoff
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
: "${CUTOFF_ROOT:?set new cutoff output root}"
: "${SCALABLE_GP_DATASET_DIR:?set frozen dataset cache}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
"$AIPY" -u "$TOOLS_DIR/dense_cutoff_sweep.py" worker --output "$CUTOFF_ROOT" \
  --worker "$SLURM_ARRAY_TASK_ID" --workers 5 --test-chunk 32
