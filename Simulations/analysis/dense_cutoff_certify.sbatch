#!/usr/bin/env bash
#SBATCH --job-name=cutoff_certify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail
: "${AIPY:?set pinned Python}"
: "${TOOLS_DIR:?set evaluator snapshot}"
: "${CUTOFF_ROOT:?set separate output root}"
export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
"$AIPY" "$TOOLS_DIR/dense_cutoff_sweep.py" certify --output "$CUTOFF_ROOT"
