#!/usr/bin/env bash
#SBATCH --job-name=gibbs_anchor_pred
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
set -euo pipefail
OUTPUT=${1:?prediction output root}
AIPY="${AIPY:?set AIPY to the python of the release environment}"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export SCALABLE_GP_DATASET_DIR="${SCALABLE_GP_DATASET_DIR:?set the dataset cache directory}"
exec "$AIPY" -u "$OUTPUT/tools/gibbs_anchor_predictions.py" worker --output "$OUTPUT" --worker "${SLURM_ARRAY_TASK_ID:?}"
