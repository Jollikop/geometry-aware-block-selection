#!/usr/bin/env bash
#SBATCH --job-name=partctrl_certify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail
: "${AIPY:?set pinned Python}"
: "${TOOLS_DIR:?set evaluator snapshot}"
: "${CERT_ROOTS:?colon-separated completed control roots}"
export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
IFS=':' read -r -a roots <<< "$CERT_ROOTS"
for root in "${roots[@]}"; do
  "$AIPY" "$TOOLS_DIR/partition_controls.py" certify --output "$root"
done
