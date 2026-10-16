#!/usr/bin/env bash
# SBT-SGPR prediction ablation at fixed trained parameters.
# The runner records no-cross, local training-block, dense Eq. (12), and
# true-GP diagnostics from each fitted model.
#SBATCH --job-name=sgpr_nll_ablation
#SBATCH --array=0-9%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_sgpr_nll_ablation_%A_%a.out

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
if [[ -f "$VENV_ACTIVATE" ]]; then
  source "$VENV_ACTIVATE"
else
  echo "Virtual environment not found: $VENV_ACTIVATE" >&2
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

python -u run_benchmark.py \
  --datasets kin40k \
  --kernels rbf \
  --methods SBT-SGPR \
  --m 128 \
  --seeds "$SLURM_ARRAY_TASK_ID" \
  --n-train 1500 \
  --n-test 500 \
  --block-size 150 \
  --optimizer lbfgs \
  --refresh-schedule 15,15,20,25,29925 \
  --require-convergence \
  --full-predictor-max-n 1500 \
  --no-logdet-diagnostics \
  --no-tables
