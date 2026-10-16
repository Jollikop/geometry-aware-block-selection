#!/usr/bin/env bash
# Aggregate a completed SBT-guided SGPR exploratory array.
#SBATCH --job-name=sbt_guided_plot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_sbt_guided_plot_%j.out

set -euo pipefail

SGPR_DIR="${SGPR_BENCHMARK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SIM_ROOT="${SIM_ROOT:-$(cd "$SGPR_DIR/.." && pwd)}"
ROOT="${PROJECT_ROOT:-$(cd "$SIM_ROOT/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$ROOT/../.." && pwd)}"
VENV_ACTIVATE="${PROJECT_VENV_ACTIVATE:-$WORKSPACE_ROOT/.venv/bin/activate}"
OUT_DIR="${SGPR_GUIDED_OUT:?set SGPR_GUIDED_OUT to the exploratory directory}"

source "$VENV_ACTIVATE"
export PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg

python -u "$SGPR_DIR/analysis/explore_sbt_guided_sgpr.py" \
  --out-dir "$OUT_DIR" \
  --summarize-only
