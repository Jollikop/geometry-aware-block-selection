#!/usr/bin/env bash
# Run an array campaign without ever leaving work parked in the queue.
#
# Slurm's %N throttle limits how many array tasks RUN; every other task still
# sits in the queue as PENDING. On a shared cluster forty parked tasks read as
# hoarding even though they hold no GPU. This driver submits one task at a
# time and only when an earlier one has left the queue, so the user never has
# more than MAX_IN_FLIGHT jobs in existence -- running plus pending.
#
# The cost is a scheduling gap: a slot stays idle for up to POLL_SECONDS
# between one task finishing and the next being submitted. That is the price
# of not queueing, and it is why the array throttle exists in the first place.
#
#   MAX_IN_FLIGHT=5 bash submit_capped.sh <job-name> <first> <last> \
#       <sbatch-script> [extra sbatch args ...]
#
# Example:
#   MAX_IN_FLIGHT=5 bash submit_capped.sh sgpr_transplant 0 39 \
#       "$SGPR/paper_sgpr_transplant.sbatch" \
#       --partition=NA10040q --nodelist=node13 --export="$COMMON"
set -uo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 JOB_NAME FIRST LAST SBATCH_SCRIPT [sbatch args...]" >&2
  exit 2
fi
JOB_NAME=$1; FIRST=$2; LAST=$3; SCRIPT=$4; shift 4
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-5}"
POLL_SECONDS="${POLL_SECONDS:-20}"

[[ -f "$SCRIPT" ]] || { echo "no such sbatch script: $SCRIPT" >&2; exit 2; }

# Count this campaign's jobs only, so an unrelated job of yours elsewhere
# neither blocks this driver nor is counted against its budget.
in_flight() {
  squeue -u "$USER" -h -n "$JOB_NAME" -t PD,R 2>/dev/null | wc -l | tr -d ' '
}

total=$(( LAST - FIRST + 1 ))
echo "$JOB_NAME: $total tasks, at most $MAX_IN_FLIGHT in the queue at a time"

for task in $(seq "$FIRST" "$LAST"); do
  while [[ "$(in_flight)" -ge "$MAX_IN_FLIGHT" ]]; do sleep "$POLL_SECONDS"; done
  if jid=$(sbatch --parsable --array="$task" "$@" "$SCRIPT" 2>&1); then
    echo "$(date '+%H:%M:%S') submitted task $task as $jid"
  else
    echo "$(date '+%H:%M:%S') submit FAILED for task $task: $jid" >&2
  fi
  sleep 1                      # let squeue see it before the next count
done

while [[ "$(in_flight)" -gt 0 ]]; do sleep "$POLL_SECONDS"; done
echo "$(date '+%H:%M:%S') $JOB_NAME: all $total tasks left the queue"
