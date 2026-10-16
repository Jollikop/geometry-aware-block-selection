#!/usr/bin/env bash
#SBATCH --job-name=sgpr_theta_plot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail
source "${PROJECT_VENV_ACTIVATE:?set PROJECT_VENV_ACTIVATE}"
export PYTHONPATH="${SIM_ROOT:?set SIM_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export JAX_PLATFORMS=cpu
python -u "${SGPR_BENCHMARK_DIR:?set SGPR_BENCHMARK_DIR}/analysis/explore_fixed_theta_bounds.py" \
  --out-dir "${SGPR_THETA_OUT:?set SGPR_THETA_OUT}" \
  --summarize-only
