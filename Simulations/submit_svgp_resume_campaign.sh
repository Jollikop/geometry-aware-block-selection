#!/usr/bin/env bash
# Resume every dataset except 3droad in the preserved campaign root.

set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "usage: $0 SOURCE_PROJECT_ROOT FINAL_ROOT PARTITION NODELIST|- [CONCURRENCY] [array|bundle]" >&2
  exit 2
fi
exec bash "$HERE/submit_svgp_scope_campaign.sh" non3droad "$@"
