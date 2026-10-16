#!/usr/bin/env bash
#SBATCH --job-name=pol_blockdiag
#SBATCH --array=0-4%5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail
: "${AIPY:?export a Python environment with the pinned numerical dependencies}"
: "${SOURCE_SIM:?set the clean, commit-pinned Simulations checkout}"
: "${ABLATION_ROOT:?set a new output root outside the source checkout}"
: "${TOOLS_DIR:?set the directory containing the analysis scripts}"

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$SOURCE_SIM:$TOOLS_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$AIPY" -u "$TOOLS_DIR/batch_ablation_block_diag.py" worker \
  --source-sim "$SOURCE_SIM" \
  --output "$ABLATION_ROOT" \
  --tools "$TOOLS_DIR" \
  --worker "$SLURM_ARRAY_TASK_ID"
