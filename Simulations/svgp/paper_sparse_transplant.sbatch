#!/usr/bin/env bash
# Frozen-theta sparse distillation from each completed SVGP-family donor.
#
# Per kernel/dataset/M/seed, the four donor checkpoints initialize a standard
# sparse SVGP target. Epochs 1--10 update q(u) only; epochs 11--150 update Z
# and q(u), while theta remains frozen throughout.
#SBATCH --job-name=svgp_transplant
#SBATCH --array=0-4%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --output=slurm_svgp_transplant_%A_%a.out

set -euo pipefail

HERE="${SVGP_BENCHMARK_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
SIM_ROOT=${SIM_ROOT:-$(cd "$HERE/.." && pwd)}
export PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$HERE/../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$PROJECT_ROOT/../.." && pwd)}"
VENV_ACTIVATE="${PROJECT_VENV_ACTIVATE:-$WORKSPACE_ROOT/.venv/bin/activate}"
RUNNER="$HERE/run_benchmark.py"
RUN_TAG="${RUN_TAG:-aistats2026_svgp_sparse_transplant_v1}"
NSHARDS="${NSHARDS:-5}"
SHARD="${SLURM_ARRAY_TASK_ID:?submit this file as an array}"
SOURCE_STATIONARY_RUNS="${SOURCE_STATIONARY_RUNS:?missing stationary source tree}"
SOURCE_EXTENDED_RUNS="${SOURCE_EXTENDED_RUNS:?missing extended source tree}"
RUNS_DIR="${RUNS_DIR:-$HERE/runs_v12/$RUN_TAG}"
LOGS_DIR="${LOGS_DIR:-$HERE/logs_v12/$RUN_TAG}"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Virtual environment not found: $VENV_ACTIVATE" >&2
  exit 1
fi
source "$VENV_ACTIVATE"
cd "$HERE"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
unset JAX_PLATFORMS || true

CURRENT_HEAD="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
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
  --plan custom
  --run-tag "$RUN_TAG"
  --datasets pol superconductivity kin40k protein elevators traffic electricity 3droad
  --methods SVGP T-SVGP RBT-SVGP SBT-SVGP
  --m 64 128 256 512
  --seeds 0 1 2 3 4 5 6 7 8 9
  --all-data
  --epochs 150
  --phases "10:theta,z 140:theta"
  --metrics-at-epochs 0 10 25 50 100 150
  --block 500
  --ll-batch-size 1000
  --blocks-per-step 2
  --refresh-every 151
  --refresh-until 0
  --common-ll-batches
  --batch-coupling split
  --penalty-sampling uniform
  --optimizer adam
  --lr 0.001
  --z-lr 0.001
  --lr-profile fixed
  --lr-schedule constant
  --q-optimizer natural
  --natural-lr 0.05
  --normalize-objective
  --transplant-to-svgp
  --save-params
  --record-history
  --full-predictor-max-n 50000
  --runs-dir "$RUNS_DIR"
  --logs-dir "$LOGS_DIR"
  --num-shards "$NSHARDS"
  --shard-id "$SHARD"
  --no-live
)

echo "===== stationary donors shard=$SHARD/$NSHARDS ====="
python -u "$RUNNER" "${COMMON[@]}" \
  --kernels rbf matern52 \
  --init-from-runs-dir "$SOURCE_STATIONARY_RUNS"

echo "===== extended donors shard=$SHARD/$NSHARDS ====="
python -u "$RUNNER" "${COMMON[@]}" \
  --kernels amplitude-rbf gibbs \
  --init-from-runs-dir "$SOURCE_EXTENDED_RUNS"

echo "shard=$SHARD COMPLETE"
