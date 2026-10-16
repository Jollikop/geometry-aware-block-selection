#!/usr/bin/env bash
# Controlled RBF-pol-M128 epoch and batch-separation diagnostic.
#
# Submit as a five-task array (four concurrent GPUs are sufficient):
#   sbatch --array=0-4%4 --gres=gpu:1 batch_separation_ablation.sbatch
#
# Tasks 0--3 run the four paper methods with recorded histories. Task 4 adds
# SBT-SVGP with a shared local batch. Both SBT arms use uniform structural
# sampling so their difference isolates batch separation.
#SBATCH --job-name=svgp_batch_ablation
#SBATCH --array=0-4%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_svgp_batch_ablation_%A_%a.out

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
if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Virtual environment not found: $VENV_ACTIVATE" >&2
  exit 1
fi
source "$VENV_ACTIVATE"
cd "$HERE"

if [[ -n "${EXPECTED_HEAD:-}" ]] &&
   [[ "$(git rev-parse HEAD)" != "$EXPECTED_HEAD" ]]; then
  echo "Expected commit $EXPECTED_HEAD, found $(git rev-parse HEAD)." >&2
  exit 2
fi

# One owner for "is the experiment source clean": the same tree, with the
# same flags, that common.provenance.code_state() records in every
# manifest. A launcher checking a narrower set can pass a run that the
# validator then rejects as dirty -- which cost one full campaign.
if [[ -n "$(git -C "$SIM_ROOT" status --porcelain --untracked-files=all -- .)" ]]; then
  echo "Experiment-source changes detected; refusing an official run." >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID:?submit as array 0-4}" in
  0) METHOD=SVGP;     COUPLING=split ;;
  1) METHOD=T-SVGP;   COUPLING=split ;;
  2) METHOD=RBT-SVGP;  COUPLING=split ;;
  3) METHOD=SBT-SVGP; COUPLING=split ;;
  4) METHOD=SBT-SVGP; COUPLING=coupled ;;
  *) echo "Expected array task 0, 1, 2, 3, or 4." >&2; exit 2 ;;
esac

RUN_TAG="pol_m128_epochs_${COUPLING}_v1"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
unset JAX_PLATFORMS || true

python -c '
import jax
devices = jax.devices()
print("JAX devices:", devices)
if not any(device.platform == "gpu" for device in devices):
    raise SystemExit("JAX is not using the Slurm GPU")
'

python -u run_benchmark.py \
  --plan custom \
  --datasets pol \
  --kernels rbf \
  --methods "$METHOD" \
  --m 128 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --all-data \
  --epochs 350 \
  --block 500 \
  --ll-batch-size 1000 \
  --blocks-per-step 2 \
  --refresh-every 40 \
  --refresh-until 200 \
  --importance-refresh-every 10 \
  --importance-epsilon 0.05 \
  --common-ll-batches \
  --batch-coupling "$COUPLING" \
  --penalty-sampling uniform \
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
  --run-tag "$RUN_TAG" \
  --runs-dir "$HERE/runs_batch_ablation/$COUPLING" \
  --logs-dir "$HERE/logs_batch_ablation/$COUPLING" \
  --no-live
