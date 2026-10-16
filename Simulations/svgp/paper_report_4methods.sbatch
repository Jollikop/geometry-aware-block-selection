#!/usr/bin/env bash
# Four-method, eight-pair paper campaign.
#
# Re-submit the same command after interruption: only exact v11 fingerprints
# are skipped.
#
# The partition and node are intentionally selected at submission time:
#   sbatch --partition=... --nodelist=... paper_report_4methods.sbatch
#SBATCH --job-name=svgp_report4
#SBATCH --array=0-4%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6-23:59:00
#SBATCH --requeue
#SBATCH --output=slurm_svgp_report4_%A_%a.out

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
RUN_TAG="${RUN_TAG:-paper_official_v1}"
NSHARDS="${NSHARDS:-5}"
SHARD="${SLURM_ARRAY_TASK_ID:?submit this file as an array}"
METHODS=(SVGP T-SVGP RBT-SVGP SBT-SVGP)

if [[ -f "$VENV_ACTIVATE" ]]; then
  source "$VENV_ACTIVATE"
else
  echo "No supported virtual environment found." >&2
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

if [[ "${MODE:-full}" == "replacement_smoke" ]]; then
  SMOKE_COMMON=(
    --plan custom
    --methods "${METHODS[@]}"
    --m 32
    --seeds 0
    --n-train 1500
    --n-test 500
    --epochs 30
    --block 128
    --ll-batch-size 128
    --blocks-per-step 1
    --refresh-every 10
    --refresh-until 20
    --importance-refresh-every 10
    --importance-epsilon 0.05
    --common-ll-batches
    --batch-coupling split
    --penalty-sampling gain
    --optimizer adam
    --lr 0.005
    --z-lr 0.005
    --lr-profile fixed
    --lr-schedule constant
    --q-optimizer natural
    --natural-lr 0.05
    --sbt-geometry anchor
    --normalize-objective
    --record-history
    --save-params
    --run-tag replacement_smoke
    --runs-dir "/tmp/svgp_replacement_smoke_${SLURM_JOB_ID}/runs"
    --logs-dir "/tmp/svgp_replacement_smoke_${SLURM_JOB_ID}/logs"
    --no-live
  )
  python -u "$RUNNER" \
    --kernels rbf --datasets electricity "${SMOKE_COMMON[@]}"
  python -u "$RUNNER" \
    --kernels matern32 --datasets elevators "${SMOKE_COMMON[@]}"
  exit
fi

if [[ "${MODE:-full}" == "smoke" ]]; then
  python -u "$RUNNER" \
    --smoke \
    --kernels rbf \
    --methods "${METHODS[@]}" \
    --optimizer adam \
    --lr 0.005 \
    --z-lr 0.005 \
    --lr-profile fixed \
    --lr-schedule constant \
    --q-optimizer natural \
    --natural-lr 0.05 \
    --sbt-geometry anchor \
    --normalize-objective \
    --record-history \
    --save-params \
    --run-tag smoke_report4 \
    --runs-dir "/tmp/svgp_smoke_${SLURM_JOB_ID}/runs" \
    --logs-dir "/tmp/svgp_smoke_${SLURM_JOB_ID}/logs" \
    --no-live
  exit
fi

COMMON=(
  --run-tag "$RUN_TAG"
  --methods "${METHODS[@]}"
  --seeds 0 1 2 3 4 5 6 7 8 9
  --all-data
  --epochs 350
  --block 500
  --ll-batch-size 1000
  --blocks-per-step 2
  --refresh-every 40
  --refresh-until -1
  --importance-refresh-every 10
  --importance-epsilon 0.05
  --common-ll-batches
  --batch-coupling split
  --penalty-sampling gain
  --optimizer adam
  --lr 0.005
  --z-lr 0.005
  --lr-profile fixed
  --lr-schedule constant
  --q-optimizer natural
  --natural-lr 0.05
  --sbt-geometry anchor
  --normalize-objective
  --save-params
  --warp-gate-fraction 0.0
  --full-predictor-max-n 5000
  --runs-dir "$HERE/runs_v12/$RUN_TAG"
  --logs-dir "$HERE/logs_v12/$RUN_TAG"
  --num-shards "$NSHARDS"
  --shard-id "$SHARD"
  --no-live
)

echo "===== report shard=$SHARD/$NSHARDS ====="
python -u "$RUNNER" --plan report "${COMMON[@]}"

echo "shard=$SHARD COMPLETE"
