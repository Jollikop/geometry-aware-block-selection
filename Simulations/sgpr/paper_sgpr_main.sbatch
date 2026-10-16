#!/usr/bin/env bash
# Official Kin40k collapsed grid: four kernels, seven methods, ten seeds.
#
# One array task per (kernel, seed): 4 x 10 = 40 tasks, seven methods each.
# Existing results are reused only when their complete configuration matches.
#
#   sbatch --partition=PA10080q --nodelist=node04 \
#     --export=ALL,EXPECTED_HEAD="$(git rev-parse HEAD)" \
#     paper_sgpr_main.sbatch
#
#SBATCH --job-name=sgpr_main
#SBATCH --array=0-39%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_sgpr_main_%A_%a.out

set -euo pipefail

HERE="${SGPR_BENCHMARK_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
SIM_ROOT=${SIM_ROOT:-$(cd "$HERE/.." && pwd)}
# Works on a plain checkout as well as after `pip install -e Simulations/`.
export PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$HERE/run_benchmark.py" ]]; then
  echo "Submit from Simulations/sgpr or set SGPR_BENCHMARK_DIR." >&2
  exit 1
fi
ROOT="${PROJECT_ROOT:-$(cd "$HERE/../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$ROOT/../.." && pwd)}"
VENV_ACTIVATE="${PROJECT_VENV_ACTIVATE:-$WORKSPACE_ROOT/.venv/bin/activate}"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Virtual environment not found: $VENV_ACTIVATE" >&2
  exit 1
fi
source "$VENV_ACTIVATE"
cd "$HERE"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
unset JAX_PLATFORMS || true

CURRENT_HEAD="$(git -C "$ROOT" rev-parse HEAD)"
if [[ -n "${EXPECTED_HEAD:-}" && "$CURRENT_HEAD" != "$EXPECTED_HEAD" ]]; then
  echo "Expected commit $EXPECTED_HEAD, found $CURRENT_HEAD." >&2
  exit 1
fi

# One owner for "is the experiment source clean": the same tree, with the
# same flags, that common.provenance.code_state() records in every
# manifest. A launcher checking a narrower set can pass a run that the
# validator then rejects as dirty -- which cost one full campaign.
if [[ -n "$(git -C "$SIM_ROOT" status --porcelain --untracked-files=all -- .)" ]]; then
  echo "Experiment-source changes detected; refusing an official run." >&2
  exit 1
fi

source "$SIM_ROOT/gpu_preflight.sh"
require_gpu pymetis

# Array order matches the table's row order, so one kernel can be launched at
# a time and the blocks read left to right: 0-9 RBF, 10-19 Gibbs,
# 20-29 Matern-5/2, 30-39 amplitude-RBF. All four run under one protocol so
# the rows stay comparable.
KERNELS=(rbf gibbs matern52 amplitude-rbf)
TASK="${SLURM_ARRAY_TASK_ID:?submit this file as an array}"
KERNEL="${KERNELS[$((TASK / 10))]}"
SEED="$((TASK % 10))"
M="${SGPR_M:-256}"
if [[ "$M" != "128" && "$M" != "256" && "$M" != "512" ]]; then
  echo "SGPR_M must be one of 128, 256, 512; found $M" >&2
  exit 2
fi
RUNS_DIR="${RUNS_DIR:-$HERE/runs_v23}"
LOGS_DIR="${LOGS_DIR:-$HERE/logs_v23}"

echo "===== kernel=$KERNEL M=$M seed=$SEED ====="

# --save-block-contributions is on only for seed 0: the appendix figure needs
# the per-block decomposition of a few runs, and the [n_test, n_blocks] matrix
# is the one artefact here big enough to be worth gating.
EXTRA=()
if [[ "$SEED" == "0" && "${SGPR_SAVE_BLOCK_CONTRIBUTIONS:-1}" == "1" ]]; then
  EXTRA+=(--save-block-contributions)
fi

python -u run_benchmark.py \
  --datasets kin40k \
  --kernels "$KERNEL" \
  --methods Exact-GP E-SGPR METIS-SGPR SGPR T-SGPR RBT-SGPR SBT-SGPR \
  --m "$M" \
  --seeds "$SEED" \
  --n-train 5000 \
  --n-test 2000 \
  --block-size 500 \
  --optimizer lbfgs \
  --refresh-schedule 15,15,20,25,99925 \
  --require-convergence \
  --full-predictor-max-n 20000 \
  --logdet-max-n 5000 \
  --save-params \
  --runs-dir "$RUNS_DIR" \
  --logs-dir "$LOGS_DIR" \
  --no-live \
  --no-tables \
  "${EXTRA[@]}"

echo "kernel=$KERNEL M=$M seed=$SEED COMPLETE"
