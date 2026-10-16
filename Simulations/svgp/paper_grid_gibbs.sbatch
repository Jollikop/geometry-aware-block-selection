#!/usr/bin/env bash
# Extended-kernel campaign: amplitude-RBF and Gibbs, 10 seeds, 500 epochs.
#
# 2 kernels x 8 datasets x 4 budgets x 4 methods x 10 seeds = 2560 results.
#
# The Gibbs kernel uses trainable input-dependent lengthscales
#   ell_d(x) = ell_d * (1 + sigmoid(w_d . x)),
# from a cold start. W is trainable from epoch zero, no RBF checkpoint is read,
# and no SWA post-processing changes the final iterate. SBT refreshes throughout
# training so its saved final partition matches the fitted Gibbs parameters.
#
#   sbatch --partition=... --nodelist=... --array=0-4%5 \
#     --export=ALL,NSHARDS=5,EXPECTED_HEAD="$(git rev-parse HEAD)" \
#     paper_grid_gibbs.sbatch
#SBATCH --job-name=svgp_gibbs
#SBATCH --array=0-4%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=slurm_svgp_gibbs_%A_%a.out

set -euo pipefail

HERE="${SVGP_BENCHMARK_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
SIM_ROOT=${SIM_ROOT:-$(cd "$HERE/.." && pwd)}
# Works on a plain checkout as well as after `pip install -e Simulations/`.
export PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$HERE/run_benchmark.py" ]]; then
  echo "Submit from Simulations/svgp or set SVGP_BENCHMARK_DIR." >&2
  exit 1
fi
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$HERE/../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$PROJECT_ROOT/../.." && pwd)}"
VENV_ACTIVATE="${PROJECT_VENV_ACTIVATE:-$WORKSPACE_ROOT/.venv/bin/activate}"
RUNNER="$HERE/run_benchmark.py"
RUN_TAG="${RUN_TAG:-aistats2026_svgp_extended_v12}"
NSHARDS="${NSHARDS:-5}"
SHARD="${SLURM_ARRAY_TASK_ID:?submit this file as an array}"
METHODS=(SVGP T-SVGP RBT-SVGP SBT-SVGP)
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"

# SEEDS defines the campaign. The submitters use --export=ALL, so an exported
# SEEDS would silently shrink an official run and fail only at validation.
if [[ "${ALLOW_PARTIAL_CAMPAIGN:-0}" != "1" ]]; then
  SEEDS="0 1 2 3 4 5 6 7 8 9"
fi
read -ra SEED_LIST <<< "$SEEDS"
if [[ -f "$VENV_ACTIVATE" ]]; then
  source "$VENV_ACTIVATE"
else
  echo "No supported virtual environment found: $VENV_ACTIVATE" >&2
  exit 1
fi

cd "$HERE"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
unset JAX_PLATFORMS || true

CURRENT_HEAD="$(git rev-parse HEAD)"
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

python -c '
import jax
devices = jax.devices()
print("JAX devices:", devices)
if not any(device.platform == "gpu" for device in devices):
    raise SystemExit("JAX is not using the Slurm GPU")
'

COMMON=(
  --plan grid-exploratory
  --run-tag "$RUN_TAG"
  --methods "${METHODS[@]}"
  --seeds "${SEED_LIST[@]}"
  --all-data
  --epochs 500
  --block 500
  --ll-batch-size 1000
  --blocks-per-step 2
  --refresh-every 40
  --refresh-until 500
  --common-ll-batches
  --batch-coupling split
  --penalty-sampling uniform
  --optimizer adam
  --lr 0.005
  --z-lr 0.005
  --lr-profile fixed
  --lr-schedule constant
  --q-optimizer natural
  --natural-lr 0.05
  --sbt-geometry anchor
  --normalize-objective
  --warp-gate-fraction 0.0
  --swa-fraction 0.0
  --save-params
  --full-predictor-max-n 50000
  --runs-dir "${RUNS_DIR:-$HERE/runs_v12/$RUN_TAG}"
  --logs-dir "${LOGS_DIR:-$HERE/logs_v12/$RUN_TAG}"
  --num-shards "$NSHARDS"
  --shard-id "$SHARD"
  --no-live
)

echo "===== gibbs shard=$SHARD/$NSHARDS seeds='$SEEDS' cold-start ====="
python -u "$RUNNER" "${COMMON[@]}"

echo "shard=$SHARD COMPLETE"
