#!/usr/bin/env bash
# Isolated 3droad continuation. Default to one active GPU because this row is
# substantially slower and more memory hungry than the remaining datasets.

set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "usage: $0 SOURCE_PROJECT_ROOT FINAL_ROOT PARTITION NODELIST|- [CONCURRENCY] [array|bundle]" >&2
  exit 2
fi
if [[ $# -eq 4 ]]; then
  set -- "$@" 1 array
fi
exec bash "$HERE/submit_svgp_scope_campaign.sh" 3droad "$@"
