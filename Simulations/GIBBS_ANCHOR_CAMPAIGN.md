# Paired Gibbs inducing-anchor ablation

Scope: **320 SBT-SVGP runs**, 8 datasets (including 3droad), M=64/128/256/512,
10 seeds. Five independent, nonoverlapping workers, 64 runs per GPU.
No other method, transplant, prediction, old result, or paper figure is changed.

The baseline has 320/320 Gibbs SBT records using `residual-probe`, not inducing
anchors. The 320 copied RBF parameter hashes must match the initializers in
those baseline records. Train Gibbs for 300 additional epochs, refresh at 40
and 80, initial and final structure construction as in the old code. Use
`--sbt-geometry anchor` explicitly; `auto` would select residual-probe again.
The learned Gibbs kernel (including its warp) defines the inducing-anchor
distances; this is not frozen RBF geometry.

The numerical code is the clean historical execution commit
`b4a2ebcad2dba7b2e712b212a749460c5d3e54bf`, whose common/ and svgp/ numerical
modules match reference `518db753c8a18e5a3d6349da9289568df36cb13e`. The current
working tree's modified engine is deliberately not used. New configuration
fingerprints change geometry, run tag, and one-run dispatch plan. A guard checks
the entire resolved configuration **before training**. Exact GPU-bitwise
reproduction across hardware is not promised.

## Transfer from Mac

From the project directory, after cluster connectivity has returned:

```bash
ssh gpu-lab 'mkdir -p "$HOME/gibbs-anchor-v1"'
rsync -av \
  Simulations/gibbs_anchor_campaign.py \
  Simulations/gibbs_anchor_worker.sbatch \
  Simulations/submit_gibbs_anchor_campaign.sh \
  Simulations/watch_gibbs_anchor_campaign.sh \
  gpu-lab:gibbs-anchor-v1/
```

## Prepare on cluster (does not submit anything)

Run in Bash. If the output directory already exists, the tool refuses to mix or
overwrite it. Preparation checks all inputs before creating the new campaign.

```bash
cd "$PROJECT_ROOT"
REPO_TOP=$(git rev-parse --show-toplevel)
SOURCE_TOP="$HOME/worktrees/gibbs-anchor-b4a2ebcad2db"
if [[ ! -d "$SOURCE_TOP" ]]; then
  mkdir -p "$HOME/worktrees"
  git -C "$REPO_TOP" worktree add --detach "$SOURCE_TOP" \
    b4a2ebcad2dba7b2e712b212a749460c5d3e54bf
fi
SOURCE_PROJECT="$SOURCE_TOP/Scalable_GP/Correlation_Block_SGPR"
BASELINE=$HOME/experiment_results/aistats2026_five_v18_518db753c8a1_20260828_012117
export AIPY="$PROJECT_ROOT/.venv/bin/python"
GIBBS_ANCHOR_ROOT="$HOME/experiment_results/gibbs_anchor_warm300_$(date +%Y%m%d_%H%M%S)"

"$AIPY" "$HOME/gibbs-anchor-v1/gibbs_anchor_campaign.py" prepare \
  --source "$SOURCE_PROJECT" --baseline "$BASELINE" \
  --datasets "$PROJECT_ROOT/Simulations/dataset" \
  --python "$AIPY" --root "$GIBBS_ANCHOR_ROOT"
```

## Submit after GPU availability has been checked

Check current jobs and `freep --all` first. Select one verified partition and,
optionally, one node. Do not assume a historical probe still proves availability.
An array requests up to five single-GPU tasks. Each worker checks the GPU Slurm
actually assigned with a CUDA operation; it never overrides the device mapping.
Insufficient resources may leave tasks pending; unusable CUDA fails explicitly,
without requeue loops. Do not overlap another array on the same output shards.

```bash
squeue -u "$USER" -o '%.18i %.24j %.9T %.10M %.20R'
freep --all
# Set these from the current probe, not old results:
read -r -p 'Verified partition: ' ANCHOR_PARTITION
read -r -p 'Single node, or - for any node in that partition: ' ANCHOR_NODE
bash "$HOME/gibbs-anchor-v1/submit_gibbs_anchor_campaign.sh" \
  "$GIBBS_ANCHOR_ROOT" "$ANCHOR_PARTITION" "$ANCHOR_NODE"

watch -n 15 bash "$GIBBS_ANCHOR_ROOT/tools/watch_gibbs_anchor_campaign.sh" \
  "$GIBBS_ANCHOR_ROOT"
```

Preparation alone is not a submission. A returned Slurm job ID is not proof
that five GPUs are running: verify the queue. After reboot, pass the printed
absolute campaign root to the watch; it does not depend on old environment files.

## Audit after completion

```bash
"$AIPY" "$GIBBS_ANCHOR_ROOT/tools/gibbs_anchor_campaign.py" audit \
  --root "$GIBBS_ANCHOR_ROOT"
```

This hashes training parameters, curves and saved final blocks and verifies
configuration, input hashes and finite arrays. It is a **training-artifact
audit**, not a convergence proof or final paper certification. RMSE/NLL must
subsequently be evaluated from these saved parameters/blocks into a separate
prediction root. The old residual-probe results remain an ablation baseline;
never relabel them as anchor or silently replace published aggregates.
