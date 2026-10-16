#!/usr/bin/env bash
# Validate, render and freeze a completed final suite.
#SBATCH --job-name=gp_finalize
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_gp_finalize_%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT}"
FINAL_ROOT="${FINAL_ROOT:?set FINAL_ROOT}"
VENV_ACTIVATE="${PROJECT_VENV_ACTIVATE:?set PROJECT_VENV_ACTIVATE}"
SIM_ROOT="$PROJECT_ROOT/Simulations"
export PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$FINAL_ROOT/.matplotlib"
mkdir -p "$MPLCONFIGDIR"
source "$VENV_ACTIVATE"

python "$SIM_ROOT/analysis/validate_final_suite.py" --root "$FINAL_ROOT"
python "$SIM_ROOT/analysis/make_final_figures.py" --root "$FINAL_ROOT"

# The collapsed transplant table is the paper's second kin40k table. Its runs
# are validated above, but only this step turns them into a readable table,
# and make_final_figures' make_transplant is the variational one.
for kernel in rbf gibbs matern52 amplitude-rbf; do
  python "$SIM_ROOT/sgpr/analysis/explore_hyperparameter_transplant.py" \
    --out-dir "$FINAL_ROOT/sgpr/transplant/$kernel" \
    --summarize-only
done

python "$SIM_ROOT/analysis/validate_final_suite.py" \
  --root "$FINAL_ROOT" --freeze

echo "FINAL SUITE FROZEN: $FINAL_ROOT"
