#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"

RBF_PARTITION="${1:-RTXA6Kq}"
RBF_NODE="${2:-node16}"
GIBBS_PARTITION="${3:-RTXA6Kq}"
GIBBS_NODE="${4:-node08}"
MAX_PARALLEL="${5:-3}"

SGPR_DIR="${SGPR_BENCHMARK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SIM_ROOT="${SIM_ROOT:-$(cd "$SGPR_DIR/.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SIM_ROOT/.." && pwd)}"
ACTIVATE="${PROJECT_VENV_ACTIVATE:-$WORKSPACE_ROOT/.venv/bin/activate}"
STAMP=$(date +%Y%m%d_%H%M%S)
RESULT_ROOT="${SGPR_TRANSPLANT_RESULT_ROOT:-$HOME/experiment_results/sgpr_hyperparameter_transplant_$STAMP}"
RBF_OUT="$RESULT_ROOT/rbf"
GIBBS_OUT="$RESULT_ROOT/gibbs"
SLURM_LOGS="$RESULT_ROOT/slurm"
mkdir -p "$RBF_OUT" "$GIBBS_OUT" "$SLURM_LOGS"

test -f "$SIM_ROOT/dataset/kin40k.npz"
test -f "$SIM_ROOT/dataset/kin40k.npz.provenance.json"
test -f "$SGPR_DIR/analysis/explore_hyperparameter_transplant.py"
test -f "$ACTIVATE"

COMMON="PROJECT_ROOT=$PROJECT_ROOT,SIM_ROOT=$SIM_ROOT,SGPR_BENCHMARK_DIR=$SGPR_DIR,PROJECT_VENV_ACTIVATE=$ACTIVATE"

RBF_JOB=$(sbatch --parsable \
  --partition="$RBF_PARTITION" --nodelist="$RBF_NODE" \
  --array="0-4%$MAX_PARALLEL" --job-name=sgpr_transplant_rbf \
  --output="$SLURM_LOGS/rbf_%A_%a.out" \
  --export="ALL,$COMMON,SGPR_TRANSPLANT_OUT=$RBF_OUT,SGPR_TRANSPLANT_KERNEL=rbf" \
  "$SGPR_DIR/analysis/hyperparameter_transplant.sbatch")

GIBBS_JOB=$(sbatch --parsable \
  --partition="$GIBBS_PARTITION" --nodelist="$GIBBS_NODE" \
  --array="0-4%$MAX_PARALLEL" --job-name=sgpr_transplant_gibbs \
  --output="$SLURM_LOGS/gibbs_%A_%a.out" \
  --export="ALL,$COMMON,SGPR_TRANSPLANT_OUT=$GIBBS_OUT,SGPR_TRANSPLANT_KERNEL=gibbs" \
  "$SGPR_DIR/analysis/hyperparameter_transplant.sbatch")

RBF_PLOT_JOB=$(sbatch --parsable \
  --dependency="afterok:$RBF_JOB" \
  --partition="$RBF_PARTITION" --nodelist="$RBF_NODE" \
  --output="$SLURM_LOGS/rbf_plot_%j.out" \
  --export="ALL,$COMMON,SGPR_TRANSPLANT_OUT=$RBF_OUT" \
  "$SGPR_DIR/analysis/hyperparameter_transplant_summary.sbatch")

GIBBS_PLOT_JOB=$(sbatch --parsable \
  --dependency="afterok:$GIBBS_JOB" \
  --partition="$GIBBS_PARTITION" --nodelist="$GIBBS_NODE" \
  --output="$SLURM_LOGS/gibbs_plot_%j.out" \
  --export="ALL,$COMMON,SGPR_TRANSPLANT_OUT=$GIBBS_OUT" \
  "$SGPR_DIR/analysis/hyperparameter_transplant_summary.sbatch")

printf 'export RESULT_ROOT=%q\nexport RBF_OUT=%q\nexport GIBBS_OUT=%q\nexport SLURM_LOGS=%q\nexport RBF_JOB=%q\nexport GIBBS_JOB=%q\nexport RBF_PLOT_JOB=%q\nexport GIBBS_PLOT_JOB=%q\nexport WATCH_SCRIPT=%q\n' \
  "$RESULT_ROOT" "$RBF_OUT" "$GIBBS_OUT" "$SLURM_LOGS" \
  "$RBF_JOB" "$GIBBS_JOB" "$RBF_PLOT_JOB" "$GIBBS_PLOT_JOB" \
  "$SGPR_DIR/analysis/watch_hyperparameter_transplant.sh" \
  > "$HOME/sgpr_hyperparameter_transplant_jobs.env"

printf 'RBF_JOB=%s\nGIBBS_JOB=%s\nRBF_PLOT_JOB=%s\nGIBBS_PLOT_JOB=%s\nRESULT_ROOT=%s\n' \
  "$RBF_JOB" "$GIBBS_JOB" "$RBF_PLOT_JOB" "$GIBBS_PLOT_JOB" "$RESULT_ROOT"
