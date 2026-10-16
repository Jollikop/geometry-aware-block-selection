#!/usr/bin/env bash
# What in this tree would make an official run uncertifiable, and what is safe
# to reclaim.
#
# The question is not "which directories look old". It is the one the launchers
# and common.provenance.code_state() both ask: does
#   git status --porcelain --untracked-files=all -- .
# report anything under Simulations? Every entry it prints is enough to stamp
# code_source_dirty=true on a finished campaign and make the validator throw it
# away. A version-matching cleanup cannot answer that -- it misses whatever the
# pattern did not anticipate, which is how 39 leftover figures under a
# directory from a retired layout cost 280 completed runs.
#
#   bash audit_trees.sh              report only, touches nothing
#   bash audit_trees.sh --quarantine move the flagged paths aside, never delete
#
# Quarantine moves into ~/stale_trees/<timestamp>/ keeping relative paths, so
# anything moved by mistake is one mv away from coming back. Deleting is your
# call, made after looking at what is in there.
set -uo pipefail
cd "$(dirname "$0")" || exit 1

# This audit decides what is dead from what git tracks. Without a work tree
# git ls-files answers "nothing" for every path, every source directory looks
# dead, and --quarantine would move the codebase. Refuse instead.
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "not inside a git work tree: cannot tell source from output here" >&2
  exit 2
fi
TRACKED=$(git ls-files -- . 2>/dev/null | wc -l | tr -d ' ')
if [[ "$TRACKED" -lt 20 ]]; then
  echo "git tracks only $TRACKED files under $(pwd); refusing to call anything dead" >&2
  exit 2
fi

MODE="${1:-report}"
STAMP=$(date +%Y%m%d_%H%M%S)
QUARANTINE="${STALE_TREES_DIR:-$HOME/stale_trees}/$STAMP"
FLAGGED=()

hsize() { du -sh "$1" 2>/dev/null | cut -f1; }

echo "===== SOURCE-CLEAN VERDICT ====="
DIRTY=$(git status --porcelain --untracked-files=all -- . 2>/dev/null)
if [[ -z "$DIRTY" ]]; then
  echo "  clean: an official run started now would record code_source_dirty=false"
else
  n=$(printf '%s\n' "$DIRTY" | wc -l | tr -d ' ')
  echo "  DIRTY: $n entries. Any campaign started now is invalid on arrival."
  PREFIX=$(git rev-parse --show-prefix 2>/dev/null)
  printf '%s\n' "$DIRTY" | awk -v p="$PREFIX" '{s=$NF; sub("^" p, "", s); print s}' \
    | cut -d/ -f1 | sort | uniq -c | sort -rn \
    | while read -r count top; do printf '    %4s under %s\n' "$count" "$top"; done
fi

echo
echo "===== DEAD TREES ====="
echo "  (top-level directories holding no file git tracks)"
found_dead=0
for d in */; do
  d="${d%/}"
  [[ "$d" == "dataset" || "$d" == "__pycache__" ]] && continue
  tracked=$(git ls-files -- "$d" 2>/dev/null | head -1)
  if [[ -z "$tracked" ]]; then
    printf '    %-34s %6s   last touched %s\n' \
      "$d" "$(hsize "$d")" "$(date -r "$d" '+%Y-%m-%d' 2>/dev/null || echo '?')"
    FLAGGED+=("$d"); found_dead=1
  fi
done
[[ "$found_dead" == 0 ]] && echo "    none"

echo
echo "===== OUTPUT TREES ====="
echo "  (runs_/logs_/figures_/tables_ anywhere; the live ones are named by the runners)"
LIVE=$( { grep -hoE 'runs_v[0-9]+' sgpr/run_benchmark.py svgp/run_benchmark.py 2>/dev/null; \
          grep -hoE 'runs_gain_v[0-9]+' svgp/analysis/paper_status.py 2>/dev/null; } \
        | sort -u | paste -sd'|' - )
LIVE="${LIVE:-__none__}"
while IFS= read -r d; do
  [[ -z "$d" ]] && continue
  base=$(basename "$d")
  if [[ "$base" =~ ^($LIVE)$ ]]; then
    printf '    %-34s %6s   LIVE, keep\n' "$d" "$(hsize "$d")"
  else
    printf '    %-34s %6s   retired\n' "$d" "$(hsize "$d")"
    FLAGGED+=("$d")
  fi
done < <(find . -maxdepth 3 -type d \( -name 'runs_*' -o -name 'logs_*' \
           -o -name 'figures_*' -o -name 'tables_*' \) 2>/dev/null | sed 's|^\./||')

echo
echo "===== CAMPAIGN ROOTS ====="
ROOTS="${EXPERIMENT_RESULTS_DIR:-$HOME/experiment_results}"
if [[ -d "$ROOTS" ]]; then
  for r in "$ROOTS"/*/; do
    [[ -d "$r" ]] || continue
    runs=$(find "$r" -name metrics.json 2>/dev/null | wc -l | tr -d ' ')
    frozen=$([[ -f "$r/FROZEN_MANIFEST.json" ]] && echo frozen || echo "not frozen")
    printf '    %-46s %6s  %4s runs  %s\n' \
      "$(basename "$r")" "$(hsize "$r")" "$runs" "$frozen"
  done
else
  echo "    no $ROOTS"
fi
echo "  Campaign roots are never flagged here: only you know which one the"
echo "  paper cites. Frozen ones are the keepers."

# Quarantining a directory takes its children with it, so a nested flagged
# path would be moved twice and the second mv would fail on a path that is
# already gone. Keep only the outermost of each chain.
if [[ ${#FLAGGED[@]} -gt 0 ]]; then
  mapfile -t FLAGGED < <(printf '%s\n' "${FLAGGED[@]}" | sed 's|^\./||' | sort -u \
    | awk '{ keep = 1
             for (i = 1; i <= n; i++) if (index($0, kept[i] "/") == 1) { keep = 0; break }
             if (keep) { kept[++n] = $0; print } }')
fi

echo
echo "===== TOTAL FLAGGED ====="
if [[ ${#FLAGGED[@]} -eq 0 ]]; then
  echo "  nothing to reclaim"
  exit 0
fi
du -ch "${FLAGGED[@]}" 2>/dev/null | tail -1 | sed 's/^/  /'
if [[ "$MODE" != "--quarantine" ]]; then
  echo "  re-run with --quarantine to move these into $QUARANTINE"
  exit 0
fi
for path in "${FLAGGED[@]}"; do
  dest="$QUARANTINE/$(dirname "$path")"
  mkdir -p "$dest" && mv "$path" "$dest/" && echo "  moved $path"
done
echo "  quarantined under $QUARANTINE"
echo "  verify, then remove it yourself when you are sure."
