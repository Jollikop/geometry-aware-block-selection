#!/usr/bin/env bash
# SBT structure-refresh experiments. Results are always written to a separate
# tree so the frozen gain and uniform-cycle campaigns remain immutable.
#
# MODE=rbf-ablation:
#   2 datasets x 2 budgets x 4 seeds = 16 runs. This isolates late structure
#   refreshes on the two datasets that motivated the diagnostic.
#
# MODE=gibbs-smoke:
#   One Gibbs run used to verify the GPU and refresh count.
#
# MODE=gibbs:
#   8 datasets x 4 budgets x 10 seeds = 320 SBT runs. Likelihood batches and
#   structural blocks follow separate reshuffled uniform cycles. Structures
#   refresh throughout cold-start Gibbs training.
#SBATCH --job-name=sbt_refresh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --requeue

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=${SIM_ROOT:-$(cd "$HERE/.." && pwd)}
export PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
MODE=${MODE:-gibbs}
NSHARDS=${NSHARDS:-5}
SHARD=${SLURM_ARRAY_TASK_ID:?Submit this launcher as an array}

case "$MODE" in
  rbf-ablation)
    RUN_TAG=${RUN_TAG:-sbt_rbf_late_refresh_ablation_v1}
    RUNS_DIR=${RUNS_DIR:-$HERE/runs_structure_refresh/$RUN_TAG}
    LOGS_DIR=${LOGS_DIR:-$HERE/logs_structure_refresh/$RUN_TAG}
    PLAN=custom
    KERNELS=(rbf)
    DATASETS=(3droad elevators)
    M_VALUES=(256 512)
    SEEDS=(0 1 2 3)
    KERNEL_ARGS=(--sbt-geometry anchor)
    ;;
  gibbs-smoke)
    RUN_TAG=${RUN_TAG:-sbt_gibbs_refresh_smoke_v1}
    RUNS_DIR=${RUNS_DIR:-$HERE/runs_structure_refresh/$RUN_TAG}
    LOGS_DIR=${LOGS_DIR:-$HERE/logs_structure_refresh/$RUN_TAG}
    PLAN=custom
    KERNELS=(gibbs)
    DATASETS=(kin40k)
    M_VALUES=(64)
    SEEDS=(0)
    KERNEL_ARGS=(--warp-gate-fraction 0.0 --swa-fraction 0.0
                 --sbt-geometry anchor)
    ;;
  gibbs)
    RUN_TAG=${RUN_TAG:-sbt_gibbs_structure_refresh_v1}
    RUNS_DIR=${RUNS_DIR:-$HERE/runs_structure_refresh/$RUN_TAG}
    LOGS_DIR=${LOGS_DIR:-$HERE/logs_structure_refresh/$RUN_TAG}
    PLAN=grid-exploratory
    KERNELS=(gibbs)
    DATASETS=(
      kin40k protein 3droad superconductivity
      pol electricity elevators traffic
    )
    M_VALUES=(64 128 256 512)
    SEEDS=(0 1 2 3 4 5 6 7 8 9)
    KERNEL_ARGS=(--warp-gate-fraction 0.0 --swa-fraction 0.0
                 --sbt-geometry anchor)
    ;;
  *)
    echo "Unknown MODE=$MODE (expected rbf-ablation, gibbs-smoke, or gibbs)" >&2
    exit 2
    ;;
esac

ACTIVATE=${PROJECT_VENV_ACTIVATE:-${AIACTIVATE:-}}
if [[ -z "$ACTIVATE" || ! -f "$ACTIVATE" ]]; then
  ACTIVATE="$WORKSPACE_ROOT/.venv/bin/activate"
fi
if [[ ! -f "$ACTIVATE" ]]; then
  echo "Release virtual environment not found: $ACTIVATE" >&2
  exit 1
fi
source "$ACTIVATE"

cd "$HERE"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
unset JAX_PLATFORMS || true

CURRENT_HEAD=$(git rev-parse HEAD)
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

echo "mode=$MODE shard=$SHARD/$NSHARDS run_tag=$RUN_TAG"
echo "runs_dir=$RUNS_DIR"

python -u "$HERE/run_benchmark.py" \
  --plan "$PLAN" \
  --run-tag "$RUN_TAG" \
  --methods SBT-SVGP \
  --kernels "${KERNELS[@]}" \
  --datasets "${DATASETS[@]}" \
  --m "${M_VALUES[@]}" \
  --seeds "${SEEDS[@]}" \
  --all-data \
  --epochs 350 \
  --block 500 \
  --ll-batch-size 1000 \
  --blocks-per-step 2 \
  --refresh-every 40 \
  --refresh-until 350 \
  --importance-refresh-every 10 \
  --importance-epsilon 0.05 \
  --common-ll-batches \
  --batch-coupling split \
  --penalty-sampling uniform \
  --optimizer adam \
  --lr 0.005 \
  --z-lr 0.005 \
  --lr-profile fixed \
  --lr-schedule constant \
  --q-optimizer natural \
  --natural-lr 0.05 \
  --normalize-objective \
  --save-params \
  --record-history \
  --full-predictor-max-n 5000 \
  --runs-dir "$RUNS_DIR" \
  --logs-dir "$LOGS_DIR" \
  --num-shards "$NSHARDS" \
  --shard-id "$SHARD" \
  --no-live \
  "${KERNEL_ARGS[@]}"

if [[ "$MODE" == "gibbs-smoke" ]]; then
  python - "$RUNS_DIR" <<'PY'
import json
import sys
from pathlib import Path

paths = list(Path(sys.argv[1]).rglob("metrics.json"))
if len(paths) != 1:
    raise SystemExit(f"smoke expected one result, found {len(paths)}")
metrics = json.loads(paths[0].read_text())
if int(metrics.get("refreshes", -1)) != 8:
    raise SystemExit(
        f"smoke expected 8 structure refreshes, found "
        f"{metrics.get('refreshes')}")
if int(metrics.get("post_swa_structure_rebuilds", -1)) != 0:
    raise SystemExit("cold-start smoke unexpectedly used SWA post-processing")
if metrics["config"].get("penalty_sampling") != "uniform":
    raise SystemExit("smoke did not record uniform block cycling")
if int(metrics["config"].get("refresh_until", -1)) != 350:
    raise SystemExit("smoke did not record refresh_until=350")
print("SMOKE VERIFIED: uniform cycles and eight structure refreshes")
PY
fi

echo "mode=$MODE shard=$SHARD COMPLETE"
