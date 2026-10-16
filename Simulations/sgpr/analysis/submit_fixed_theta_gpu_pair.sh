#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"

PARTITION="${1:-PA10080q}"
NODE="${2:-node04}"
MAX_PARALLEL="${3:-4}"
GIBBS_PARTITION="${4:-$PARTITION}"
GIBBS_NODE="${5:-$NODE}"

SGPR_DIR="${SGPR_BENCHMARK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SIM_ROOT="${SIM_ROOT:-$(cd "$SGPR_DIR/.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SIM_ROOT/.." && pwd)}"
ACTIVATE="${PROJECT_VENV_ACTIVATE:-$WORKSPACE_ROOT/.venv/bin/activate}"
STAMP=$(date +%Y%m%d_%H%M%S)
RESULT_ROOT="${SGPR_THETA_RESULT_ROOT:-$HOME/experiment_results/sgpr_fixed_theta_gpu_$STAMP}"
RBF_OUT="$RESULT_ROOT/rbf"
GIBBS_OUT="$RESULT_ROOT/gibbs"
SLURM_LOGS="$RESULT_ROOT/slurm"

mkdir -p "$RBF_OUT" "$GIBBS_OUT" "$SLURM_LOGS"
test -f "$SIM_ROOT/dataset/kin40k.npz"
test -f "$SIM_ROOT/dataset/kin40k.npz.provenance.json"
test -f "$SGPR_DIR/analysis/explore_fixed_theta_bounds.py"
test -f "$ACTIVATE"

COMMON="PROJECT_ROOT=$PROJECT_ROOT,SIM_ROOT=$SIM_ROOT,SGPR_BENCHMARK_DIR=$SGPR_DIR,PROJECT_VENV_ACTIVATE=$ACTIVATE"

RBF_JOB=$(sbatch --parsable \
  --partition="$PARTITION" --nodelist="$NODE" \
  --job-name=sgpr_theta_rbf --array="0-4%$MAX_PARALLEL" \
  --output="$SLURM_LOGS/rbf_%A_%a.out" \
  --export="ALL,$COMMON,SGPR_THETA_OUT=$RBF_OUT,SGPR_THETA_KERNEL=rbf" \
  "$SGPR_DIR/analysis/fixed_theta_bounds.sbatch")

GIBBS_JOB=$(sbatch --parsable \
  --partition="$GIBBS_PARTITION" --nodelist="$GIBBS_NODE" \
  --job-name=sgpr_theta_gibbs --array="0-4%$MAX_PARALLEL" \
  --output="$SLURM_LOGS/gibbs_%A_%a.out" \
  --export="ALL,$COMMON,SGPR_THETA_OUT=$GIBBS_OUT,SGPR_THETA_KERNEL=gibbs" \
  "$SGPR_DIR/analysis/fixed_theta_bounds.sbatch")

RBF_PLOT_JOB=$(sbatch --parsable \
  --dependency="afterok:$RBF_JOB" \
  --partition="$PARTITION" --nodelist="$NODE" \
  --job-name=sgpr_theta_plot_rbf \
  --output="$SLURM_LOGS/rbf_plot_%j.out" \
  --export="ALL,$COMMON,SGPR_THETA_OUT=$RBF_OUT" \
  "$SGPR_DIR/analysis/fixed_theta_bounds_summary.sbatch")

GIBBS_PLOT_JOB=$(sbatch --parsable \
  --dependency="afterok:$GIBBS_JOB" \
  --partition="$GIBBS_PARTITION" --nodelist="$GIBBS_NODE" \
  --job-name=sgpr_theta_plot_gibbs \
  --output="$SLURM_LOGS/gibbs_plot_%j.out" \
  --export="ALL,$COMMON,SGPR_THETA_OUT=$GIBBS_OUT" \
  "$SGPR_DIR/analysis/fixed_theta_bounds_summary.sbatch")

printf 'export RESULT_ROOT=%q\nexport RBF_OUT=%q\nexport GIBBS_OUT=%q\nexport SLURM_LOGS=%q\nexport RBF_JOB=%q\nexport GIBBS_JOB=%q\nexport RBF_PLOT_JOB=%q\nexport GIBBS_PLOT_JOB=%q\nexport WATCH_SCRIPT=%q\n' \
  "$RESULT_ROOT" "$RBF_OUT" "$GIBBS_OUT" "$SLURM_LOGS" \
  "$RBF_JOB" "$GIBBS_JOB" "$RBF_PLOT_JOB" "$GIBBS_PLOT_JOB" \
  "$SGPR_DIR/analysis/watch_fixed_theta_gpu.sh" \
  > "$HOME/sgpr_fixed_theta_gpu_jobs.env"

printf 'RBF_JOB=%s\nGIBBS_JOB=%s\nRBF_PLOT_JOB=%s\nGIBBS_PLOT_JOB=%s\nRESULT_ROOT=%s\n' \
  "$RBF_JOB" "$GIBBS_JOB" "$RBF_PLOT_JOB" "$GIBBS_PLOT_JOB" \
  "$RESULT_ROOT"
