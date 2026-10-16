#!/usr/bin/env bash
# Shared GPU preflight for every job that needs a device.
#
# Slurm's gres accounting and the node's real device occupancy can disagree:
# a task is handed a GPU another process still holds a context on, and JAX dies
# in the first seconds with
#   cuDevicePrimaryCtxRetain: CUDA_ERROR_DEVICE_UNAVAILABLE
# Failing there costs the whole array task, and with an afterok chain it costs
# every downstream job too.
#
# A retry inside one allocation cannot help when the assigned device is the bad
# one -- CUDA_VISIBLE_DEVICES is fixed for the life of the allocation. So the
# task retries briefly for a genuinely transient collision, and otherwise
# requeues itself to be given a different device.
#
# Source it, then call: require_gpu [extra python import ...]

# Seconds between in-allocation retries, and how many to make.
GPU_RETRY_SLEEP="${GPU_RETRY_SLEEP:-20}"
GPU_RETRY_ATTEMPTS="${GPU_RETRY_ATTEMPTS:-3}"
# Requeues before giving up, so a permanently sick node cannot loop forever.
GPU_MAX_REQUEUE="${GPU_MAX_REQUEUE:-6}"

_gpu_probe() {
  local extra="$1"
  python - "$extra" <<'PYCHECK'
import sys

extra = sys.argv[1] if len(sys.argv) > 1 else ""
for name in (n for n in extra.split(",") if n):
    __import__(name)
    print(f"{name}: ok")

import jax

devices = jax.devices()
print("JAX devices:", devices)
if not any(device.platform == "gpu" for device in devices):
    raise SystemExit("JAX is not using the Slurm GPU")
PYCHECK
}

require_gpu() {
  local extra="${1:-}"
  local attempt
  for attempt in $(seq 1 "$GPU_RETRY_ATTEMPTS"); do
    if _gpu_probe "$extra"; then
      return 0
    fi
    echo "GPU not usable (attempt $attempt/$GPU_RETRY_ATTEMPTS on" \
         "${SLURMD_NODENAME:-unknown}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})" >&2
    sleep "$GPU_RETRY_SLEEP"
  done

  local restarts="${SLURM_RESTART_COUNT:-0}"
  if [[ -n "${SLURM_JOB_ID:-}" && "$restarts" -lt "$GPU_MAX_REQUEUE" ]]; then
    echo "requeueing $SLURM_JOB_ID for a different device (restart $restarts)" >&2
    if scontrol requeue "$SLURM_JOB_ID"; then
      # Slurm kills the task during this window. Never exit 0 from here: a
      # zero exit would mark the task COMPLETED with nothing written, and the
      # afterok chain would run on a result that does not exist.
      sleep 120
    fi
    exit 1
  fi
  echo "no usable GPU after $restarts requeues; giving up" >&2
  exit 1
}
