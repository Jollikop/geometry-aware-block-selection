#!/usr/bin/env bash
# Submit the complete paper campaign with at most five concurrent GPUs.

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 PARTITION NODE [DATASET_CACHE]" >&2
  exit 2
fi
PARTITION=$1
NODE=$2

SIM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SIM_ROOT/.." && pwd)
SGPR="$SIM_ROOT/sgpr"
SVGP="$SIM_ROOT/svgp"
VENV_ACTIVATE="${PROJECT_VENV_ACTIVATE:-$WORKSPACE_ROOT/.venv/bin/activate}"
DATASET_CACHE="${3:-${SCALABLE_GP_DATASET_DIR:-$SIM_ROOT/dataset}}"
EXPECTED_HEAD=$(git -C "$PROJECT_ROOT" rev-parse HEAD)

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "missing environment: $VENV_ACTIVATE" >&2
  exit 2
fi
if [[ ! -d "$DATASET_CACHE" ]]; then
  echo "missing dataset cache: $DATASET_CACHE" >&2
  exit 2
fi
# One owner for "is the experiment source clean": the same tree, with the
# same flags, that common.provenance.code_state() records in every
# manifest. A launcher checking a narrower set can pass a run that the
# validator then rejects as dirty -- which cost one full campaign.
if [[ -n "$(git -C "$SIM_ROOT" status --porcelain --untracked-files=all -- .)" ]]; then
  echo "official experiment source is dirty; commit it before submission" >&2
  exit 2
fi

STAMP=$(date +%Y%m%d_%H%M%S)
FINAL_ROOT="${FINAL_ROOT:-$HOME/experiment_results/aistats2026_final_${EXPECTED_HEAD:0:12}_$STAMP}"
mkdir -p \
  "$FINAL_ROOT/slurm" \
  "$FINAL_ROOT/sgpr/runs" "$FINAL_ROOT/sgpr/logs" \
  "$FINAL_ROOT/sgpr/transplant" "$FINAL_ROOT/sgpr/snelson" \
  "$FINAL_ROOT/sgpr/residual_illustration" \
  "$FINAL_ROOT/svgp/stationary/runs" "$FINAL_ROOT/svgp/stationary/logs" \
  "$FINAL_ROOT/svgp/extended/runs" "$FINAL_ROOT/svgp/extended/logs" \
  "$FINAL_ROOT/svgp/transplant/runs" "$FINAL_ROOT/svgp/transplant/logs" \
  "$FINAL_ROOT/svgp/convergence/runs" "$FINAL_ROOT/svgp/convergence/logs"

COMMON_EXPORT="ALL,PROJECT_ROOT=$PROJECT_ROOT,PROJECT_VENV_ACTIVATE=$VENV_ACTIVATE,SCALABLE_GP_DATASET_DIR=$DATASET_CACHE,EXPECTED_HEAD=$EXPECTED_HEAD"

SGPR_JOB=$(sbatch --parsable \
  --partition="$PARTITION" --nodelist="$NODE" --array=0-39%5 \
  --output="$FINAL_ROOT/slurm/sgpr_main_%A_%a.out" \
  --export="$COMMON_EXPORT,SGPR_BENCHMARK_DIR=$SGPR,RUNS_DIR=$FINAL_ROOT/sgpr/runs,LOGS_DIR=$FINAL_ROOT/sgpr/logs" \
  "$SGPR/paper_sgpr_main.sbatch")

RESIDUAL_JOB=$(sbatch --parsable --dependency="afterok:$SGPR_JOB" \
  --partition="$PARTITION" --nodelist="$NODE" \
  --output="$FINAL_ROOT/slurm/sgpr_residual_%j.out" \
  --export="$COMMON_EXPORT,RESIDUAL_DIR=$SGPR/residual_illustration,SGPR_DONOR_RUNS=$FINAL_ROOT/sgpr/runs,RESIDUAL_OUT=$FINAL_ROOT/sgpr/residual_illustration" \
  "$SGPR/residual_illustration/paper_residual_kin40k.sbatch")

SNELSON_JOB=$(sbatch --parsable --dependency="afterok:$RESIDUAL_JOB" \
  --partition="$PARTITION" --nodelist="$NODE" \
  --output="$FINAL_ROOT/slurm/sgpr_snelson_%j.out" \
  --export="$COMMON_EXPORT,SNELSON_DIR=$SGPR/snelson,SNELSON_OUT=$FINAL_ROOT/sgpr/snelson" \
  "$SGPR/snelson/paper_snelson.sbatch")

SGPR_TRANSPLANT_JOB=$(sbatch --parsable --dependency="afterok:$SNELSON_JOB" \
  --partition="$PARTITION" --nodelist="$NODE" --array=0-39%5 \
  --output="$FINAL_ROOT/slurm/sgpr_transplant_%A_%a.out" \
  --export="$COMMON_EXPORT,SGPR_BENCHMARK_DIR=$SGPR,SGPR_DONOR_RUNS=$FINAL_ROOT/sgpr/runs,SGPR_TRANSPLANT_OUT=$FINAL_ROOT/sgpr/transplant" \
  "$SGPR/paper_sgpr_transplant.sbatch")

STATIONARY_JOB=$(sbatch --parsable --dependency="afterok:$SGPR_TRANSPLANT_JOB" \
  --partition="$PARTITION" --nodelist="$NODE" --array=0-4%5 \
  --output="$FINAL_ROOT/slurm/svgp_stationary_%A_%a.out" \
  --export="$COMMON_EXPORT,SVGP_BENCHMARK_DIR=$SVGP,RUN_TAG=aistats2026_svgp_stationary_v12,RUNS_DIR=$FINAL_ROOT/svgp/stationary/runs,LOGS_DIR=$FINAL_ROOT/svgp/stationary/logs,NSHARDS=5" \
  "$SVGP/paper_grid_stationary.sbatch")

EXTENDED_JOB=$(sbatch --parsable --dependency="afterok:$STATIONARY_JOB" \
  --partition="$PARTITION" --nodelist="$NODE" --array=0-4%5 \
  --output="$FINAL_ROOT/slurm/svgp_extended_%A_%a.out" \
  --export="$COMMON_EXPORT,SVGP_BENCHMARK_DIR=$SVGP,RUN_TAG=aistats2026_svgp_extended_v12,RUNS_DIR=$FINAL_ROOT/svgp/extended/runs,LOGS_DIR=$FINAL_ROOT/svgp/extended/logs,NSHARDS=5" \
  "$SVGP/paper_grid_gibbs.sbatch")

TRANSPLANT_JOB=$(sbatch --parsable --dependency="afterok:$EXTENDED_JOB" \
  --partition="$PARTITION" --nodelist="$NODE" --array=0-4%5 \
  --output="$FINAL_ROOT/slurm/svgp_transplant_%A_%a.out" \
  --export="$COMMON_EXPORT,SVGP_BENCHMARK_DIR=$SVGP,RUN_TAG=aistats2026_svgp_sparse_transplant_v1,RUNS_DIR=$FINAL_ROOT/svgp/transplant/runs,LOGS_DIR=$FINAL_ROOT/svgp/transplant/logs,SOURCE_STATIONARY_RUNS=$FINAL_ROOT/svgp/stationary/runs,SOURCE_EXTENDED_RUNS=$FINAL_ROOT/svgp/extended/runs,NSHARDS=5" \
  "$SVGP/paper_sparse_transplant.sbatch")

CONVERGENCE_JOB=$(sbatch --parsable --dependency="afterok:$TRANSPLANT_JOB" \
  --partition="$PARTITION" --nodelist="$NODE" --array=0-3%4 \
  --output="$FINAL_ROOT/slurm/svgp_convergence_%A_%a.out" \
  --export="$COMMON_EXPORT,SVGP_BENCHMARK_DIR=$SVGP,RUN_TAG=aistats2026_svgp_convergence_v1,RUNS_DIR=$FINAL_ROOT/svgp/convergence/runs,LOGS_DIR=$FINAL_ROOT/svgp/convergence/logs" \
  "$SVGP/paper_convergence.sbatch")

FINALIZE_JOB=$(sbatch --parsable --dependency="afterok:$CONVERGENCE_JOB" \
  --partition="$PARTITION" --nodelist="$NODE" \
  --output="$FINAL_ROOT/slurm/finalize_%j.out" \
  --export="$COMMON_EXPORT,FINAL_ROOT=$FINAL_ROOT" \
  "$SIM_ROOT/paper_finalize.sbatch")

ENV_FILE="$HOME/aistats2026_final_jobs.env"
{
  printf 'export FINAL_ROOT=%q\n' "$FINAL_ROOT"
  printf 'export PROJECT_ROOT=%q\n' "$PROJECT_ROOT"
  printf 'export SGPR_JOB=%q\n' "$SGPR_JOB"
  printf 'export RESIDUAL_JOB=%q\n' "$RESIDUAL_JOB"
  printf 'export SNELSON_JOB=%q\n' "$SNELSON_JOB"
  printf 'export SGPR_TRANSPLANT_JOB=%q\n' "$SGPR_TRANSPLANT_JOB"
  printf 'export STATIONARY_JOB=%q\n' "$STATIONARY_JOB"
  printf 'export EXTENDED_JOB=%q\n' "$EXTENDED_JOB"
  printf 'export TRANSPLANT_JOB=%q\n' "$TRANSPLANT_JOB"
  printf 'export CONVERGENCE_JOB=%q\n' "$CONVERGENCE_JOB"
  printf 'export FINALIZE_JOB=%q\n' "$FINALIZE_JOB"
} > "$ENV_FILE"

printf 'FINAL_ROOT=%s\n' "$FINAL_ROOT"
printf 'jobs: sgpr=%s residual=%s snelson=%s sgpr-transplant=%s stationary=%s extended=%s transplant=%s convergence=%s finalize=%s\n' \
  "$SGPR_JOB" "$RESIDUAL_JOB" "$SNELSON_JOB" "$SGPR_TRANSPLANT_JOB" "$STATIONARY_JOB" "$EXTENDED_JOB" \
  "$TRANSPLANT_JOB" "$CONVERGENCE_JOB" "$FINALIZE_JOB"
printf 'watch: watch -n 10 bash %q\n' "$SIM_ROOT/watch_final_campaign.sh"
