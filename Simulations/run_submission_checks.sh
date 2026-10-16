#!/usr/bin/env bash
# Everything a reviewer can run without a GPU or a cluster allocation.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/sbt_submission_pycache"
# Works on a plain checkout as well as an installed one.
# The parent directory makes the Simulations.* imports used by the analysis
# tools resolve as well.
export PYTHONPATH="$HERE:$(dirname "$HERE")${PYTHONPATH:+:$PYTHONPATH}"

echo "== tests =="
"$PYTHON_BIN" -m unittest discover -s tests -v

echo "== every module compiles =="
"$PYTHON_BIN" -m py_compile \
  environment_manifest.py \
  common/*.py \
  sgpr/*.py \
  sgpr/snelson/*.py \
  sgpr/residual_illustration/*.py \
  sgpr/analysis/*.py \
  svgp/*.py \
  svgp/analysis/*.py \
  analysis/*.py

echo "== launchers parse =="
# Includes the repository-root drivers: the finalize job and the two campaign
# submitters were outside every check until a dangling continuation in one of
# them made it exit 127 after submitting.
# bash -n reads only its first file argument, so each launcher is parsed on
# its own.
for launcher in $(ls *.sbatch *.sh \
             sgpr/*.sbatch sgpr/*/*.sbatch sgpr/*/*.sh \
             svgp/*.sbatch svgp/*.sh 2>/dev/null); do
  bash -n "$launcher"
done

echo "== campaign plans expand =="
for plan in report grid grid-exploratory; do
  "$PYTHON_BIN" -m svgp.plans --show-plan "$plan" >/dev/null
done

if "$PYTHON_BIN" -c 'import jax' >/dev/null 2>&1; then
  echo "== numerical gain decomposition =="
  "$PYTHON_BIN" -m svgp.analysis.check_gain_decomposition

  # The reported-predictor contract: a config built from the runner's own
  # defaults must satisfy the validator that guards the campaign. These drifted
  # apart once already and nothing noticed until a campaign reported
  # complete=0.
  echo "== validators accept what the runners emit =="
  "$PYTHON_BIN" - <<'PYCHECK'
from common import predictors as pr
from sgpr import benchmark as sgpr_bench
from svgp import engine as svgp_engine

assert sgpr_bench.METHOD_PREDICTOR is pr.COLLAPSED_METHOD_PREDICTOR
assert svgp_engine.METHOD_PREDICTOR is pr.VARIATIONAL_METHOD_PREDICTOR
assert set(sgpr_bench.METHODS) == set(pr.COLLAPSED_METHOD_PREDICTOR)
assert set(svgp_engine.METHODS) == set(pr.VARIATIONAL_METHOD_PREDICTOR)
print("method/predictor registries agree across both stacks")
PYCHECK
else
  echo "JAX unavailable: skipped numerical checks."
fi

if command -v latexmk >/dev/null 2>&1 && [[ -f "$HERE/../Report/main.tex" ]]; then
  echo "== paper builds =="
  (
    cd "$HERE/../Report"
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

    # Undefined citations and references are always errors: they reach the PDF
    # as [?] and no reviewer should ever see one.
    if grep -En \
      'Citation .* undefined|Reference .* undefined|There were undefined (references|citations)' \
      main.log; then
      echo "LaTeX reference diagnostics failed." >&2
      exit 1
    fi

    # Overfull boxes are judged by size rather than presence. The AISTATS title
    # block overfills by about 5pt on its own: that comes from aistats2026.sty,
    # which a submission does not get to patch, and it is invisible in print.
    # Anything large enough to actually show still fails.
    limit=${OVERFULL_LIMIT_PT:-10}
    if LIMIT="$limit" perl -ne '
         if (/Overfull \\[hv]box \(([\d.]+)pt too wide/ and $1 >= $ENV{LIMIT}) {
           print; $bad = 1;
         }
         END { exit($bad ? 1 : 0) }
       ' main.log; then
      echo "no overfull box at or above ${limit}pt"
    else
      echo "LaTeX layout: overfull box(es) at or above ${limit}pt (above)." >&2
      exit 1
    fi
  )
else
  echo "latexmk or Report/main.tex unavailable: skipped paper build."
fi

echo "All submission checks passed."
