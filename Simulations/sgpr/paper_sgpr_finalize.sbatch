#!/usr/bin/env bash
# Render and freeze a completed collapsed-only paper campaign.
#SBATCH --job-name=sgpr_freeze
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT}"
FINAL_ROOT="${FINAL_ROOT:?set FINAL_ROOT}"
VENV_ACTIVATE="${PROJECT_VENV_ACTIVATE:?set PROJECT_VENV_ACTIVATE}"
SIM_ROOT="$PROJECT_ROOT/Simulations"
SGPR="$SIM_ROOT/sgpr"
export PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$FINAL_ROOT/.matplotlib"
mkdir -p "$MPLCONFIGDIR" "$FINAL_ROOT/figures" "$FINAL_ROOT/sgpr/tables"
source "$VENV_ACTIVATE"

python "$SIM_ROOT/analysis/validate_final_suite.py" \
  --root "$FINAL_ROOT" --scope sgpr

python "$SGPR/make_main_table.py" \
  --runs-dir "$FINAL_ROOT/sgpr/runs" \
  --kernels rbf gibbs matern52 amplitude-rbf \
  --m 256 \
  --csv-out "$FINAL_ROOT/sgpr/tables/sgpr_kin40k_main.csv" \
  --tex-out "$FINAL_ROOT/sgpr/tables/sgpr_kin40k_main.tex"

for kernel in rbf gibbs matern52 amplitude-rbf; do
  python "$SGPR/analysis/explore_hyperparameter_transplant.py" \
    --out-dir "$FINAL_ROOT/sgpr/transplant/$kernel" \
    --summarize-only
done

python "$SIM_ROOT/analysis/validate_final_suite.py" \
  --root "$FINAL_ROOT" --scope sgpr --freeze

echo "COLLAPSED SGPR SUITE FROZEN: $FINAL_ROOT"
