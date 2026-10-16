#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"

[[ $# -ge 2 && $# -le 3 ]] || {
  echo "Usage: $0 PARTITION NODE [OUTPUT_DIR]" >&2
  exit 2
}
PARTITION=$1
NODE=$2
SGPR_DIR="${SGPR_BENCHMARK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SIM_ROOT="${SIM_ROOT:-$(cd "$SGPR_DIR/.." && pwd)}"
ROOT="${PROJECT_ROOT:-$(cd "$SIM_ROOT/.." && pwd)}"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="${3:-$HOME/experiment_results/sgpr_fixed_theta_${STAMP}}"
SLURM="$OUT/slurm"
mkdir -p "$SLURM"

export PROJECT_ROOT="$ROOT" SIM_ROOT SGPR_BENCHMARK_DIR="$SGPR_DIR"
export SGPR_THETA_OUT="$OUT"
export PROJECT_VENV_ACTIVATE="${PROJECT_VENV_ACTIVATE:-$WORKSPACE_ROOT/.venv/bin/activate}"

RUN_JOB=$(sbatch --parsable --partition="$PARTITION" --nodelist="$NODE" \
  --array=0-4%5 --output="$SLURM/run_%A_%a.out" --export=ALL \
  "$SGPR_DIR/analysis/fixed_theta_bounds.sbatch")
PLOT_JOB=$(sbatch --parsable --dependency="afterok:$RUN_JOB" \
  --partition="$PARTITION" --nodelist="$NODE" \
  --output="$SLURM/plot_%j.out" --export=ALL \
  "$SGPR_DIR/analysis/fixed_theta_bounds_summary.sbatch")

printf 'export RUN_JOB=%q\nexport PLOT_JOB=%q\nexport SGPR_THETA_OUT=%q\nexport SGPR_THETA_SLURM=%q\n' \
  "$RUN_JOB" "$PLOT_JOB" "$OUT" "$SLURM" > "$HOME/sgpr_fixed_theta_jobs.env"
printf 'RUN_JOB=%s\nPLOT_JOB=%s\nSGPR_THETA_OUT=%s\n' "$RUN_JOB" "$PLOT_JOB" "$OUT"
