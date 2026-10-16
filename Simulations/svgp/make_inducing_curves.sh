#!/usr/bin/env bash
# Build reviewer-ready inducing-point curves from complete metrics.json files.
#
# With no arguments, every available kernel/dataset pair is emitted in groups
# of COLUMNS_PER_FIGURE (four by default). Pass exact kernel:dataset pairs to
# select and order a paper figure:
#   bash make_inducing_curves.sh \
#     rbf:kin40k matern52:protein gibbs:3droad periodic-angular:co2
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SIM_ROOT=${SIM_ROOT:-$(cd "$HERE/.." && pwd)}
# Works on a plain checkout as well as after `pip install -e Simulations/`.
export PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$HERE"

PY="${PYTHON:-python3}"
RUN_TAG="${RUN_TAG:-paper_official_v1}"
RUNS_DIR="${RUNS_DIR:-$HERE/runs_v12/$RUN_TAG}"
OUT_DIR="${OUT_DIR:-$HERE/figures_v10/$RUN_TAG/inducing_curves}"
FIGURE_PREFIX="${FIGURE_PREFIX:-inducing_curves}"
REPORT_TARGET="${REPORT_TARGET:-}"
COLUMNS_PER_FIGURE="${COLUMNS_PER_FIGURE:-4}"
METHODS=(SVGP T-SVGP RBT-SVGP SBT-SVGP)

if ! [[ "$COLUMNS_PER_FIGURE" =~ ^[1-9][0-9]*$ ]]; then
  echo "COLUMNS_PER_FIGURE must be a positive integer." >&2
  exit 2
fi

PAIR_ARGS=()
if (( $# > 0 )); then
  PAIR_ARGS=(--pairs "$@")
fi

"$PY" "$HERE/plot_inducing_curves.py" \
  --runs-dir "$RUNS_DIR" \
  --run-tag "$RUN_TAG" \
  --plan report \
  --methods "${METHODS[@]}" \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --metrics rmse nll last_epoch_stochastic_objective_estimate_per_datum \
  --error sem \
  --require-complete-seeds \
  --columns-per-figure "$COLUMNS_PER_FIGURE" \
  --figure-prefix "$FIGURE_PREFIX" \
  --formats pdf png \
  --out-dir "$OUT_DIR" \
  "${PAIR_ARGS[@]}"

if [[ -n "$REPORT_TARGET" ]]; then
  if (( $# == 0 || $# > COLUMNS_PER_FIGURE )); then
    echo "REPORT_TARGET requires one figure (1-${COLUMNS_PER_FIGURE} explicit pairs)." >&2
    exit 2
  fi
  mkdir -p "$(dirname "$REPORT_TARGET")"
  cp "$OUT_DIR/${FIGURE_PREFIX}.pdf" "$REPORT_TARGET"
  REPORT_RESULTS_DIR="${REPORT_RESULTS_DIR:-$(dirname "$(dirname "$REPORT_TARGET")")/Results}"
  mkdir -p "$REPORT_RESULTS_DIR"
  cp "$OUT_DIR/${FIGURE_PREFIX}_summary.csv" "$REPORT_RESULTS_DIR/"
  cp "$OUT_DIR/${FIGURE_PREFIX}_manifest.json" "$REPORT_RESULTS_DIR/"
  echo "installed $REPORT_TARGET"
  echo "installed release summary and manifest in $REPORT_RESULTS_DIR"
fi
