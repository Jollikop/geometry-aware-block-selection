#!/usr/bin/env bash
# Regenerate the release Snelson figure, table and manifest.
#
# Every method draws the predictive covariance its own bound models: the sparse
# predictive for SGPR, and the bound-matched residual predictor for T-SGPR,
# RBT-SGPR, SBT-SGPR and E-SGPR. There
# is no predictor flag to pass -- the choice is a property of the method, and it
# is declared once in common/predictors.py.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=${SIM_ROOT:-$(cd "$HERE/../.." && pwd)}
# Works on a plain checkout as well as after `pip install -e Simulations/`.
export PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Two different roots. REPO_ROOT is this paper's checkout and owns Report/;
# PROJECT_ROOT is the workspace above it that holds the shared environment.
REPO_ROOT=${REPO_ROOT:-$(cd "$SIM_ROOT/.." && pwd)}
# Ask git rather than counting "..": the workspace sits three levels above
# Simulations, and getting that count wrong points at a directory with no
# environment in it.
PROJECT_ROOT=${PROJECT_ROOT:-$(
  git -C "$SIM_ROOT" rev-parse --show-toplevel 2>/dev/null \
    || (cd "$SIM_ROOT/../../.." && pwd)
)}
AIPY=${AIPY:-$PROJECT_ROOT/.venv-aistats/bin/python}
if [[ ! -x "$AIPY" ]]; then
  echo "Interpreter not found: $AIPY" >&2
  echo >&2
  echo "Set AIPY to a Python 3.12 environment holding gpflow AND jax." >&2
  echo "The figure draws its bands with the shared residual predictor, so it" >&2
  echo "needs both halves of the stack. To build one from the pinned profile:" >&2
  echo >&2
  echo "  /opt/homebrew/opt/python@3.12/bin/python3.12 -m venv \\" >&2
  echo "      \"$PROJECT_ROOT/.venv-aistats\"" >&2
  echo "  \"$PROJECT_ROOT/.venv-aistats/bin/pip\" install -r \\" >&2
  echo "      \"$SIM_ROOT/requirements.txt\"" >&2
  exit 1
fi

# The figure has drawn its bands with the shared residual predictor since the
# bound-matched change, so this environment needs JAX as well as GPflow. The
# release profile says so too; without it a manifest would pass validation for
# a figure the environment could not have produced.
"$AIPY" - <<'PYCHECK'
import gpflow, jax
print(f"gpflow {gpflow.__version__}, jax {jax.__version__}")
PYCHECK

cd "$HERE"
EXTRA=()
if [[ -n "${SNELSON_OUT:-}" ]]; then
  EXTRA+=(--out-dir "$SNELSON_OUT")
fi
"$AIPY" run_snelson.py "${EXTRA[@]}" "$@"

# The three artifacts do not share a tree: the figure and the manifest are
# paper outputs under Report/, the table stays beside this script.
CHECK_ROOT="${SNELSON_OUT:-}"
if [[ -n "$CHECK_ROOT" ]]; then
  test -s "$CHECK_ROOT/snelson_sgpr_t_rbt_sbt.pdf"
  test -s "$CHECK_ROOT/snelson_summary_sgpr_t_rbt_sbt.csv"
  test -s "$CHECK_ROOT/snelson_sgpr_t_rbt_sbt_manifest.json"
else
  test -s "$REPO_ROOT/Report/figures/snelson_sgpr_t_rbt_sbt.pdf"
  test -s "$HERE/snelson_summary_sgpr_t_rbt_sbt.csv"
  test -s "$REPO_ROOT/Report/Results/snelson_sgpr_t_rbt_sbt_manifest.json"
fi
echo "Snelson figure, table and manifest regenerated."
