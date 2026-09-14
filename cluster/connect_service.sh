#!/bin/bash
# Keep one localhost port connected to a ready Slurm service.

set -euo pipefail

SERVICE=${1:-}
REMOTE=gururaj@hpc.bits-hyderabad.ac.in
REMOTE_CLI=/home/gururaj/bin/ontomem
REMOTE_PYTHON=/scratch/gururaj/Sriniketh/envs/fern/bin/python
REMOTE_BRIDGE=/scratch/gururaj/Sriniketh/ontomem-inference/stdio_bridge.py
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

case "$SERVICE" in
  gemma|qwen) LOCAL_PORT=${LOCAL_LLM_PORT:-${LOCAL_QWEN_PORT:-18000}} ;;
  embed) LOCAL_PORT=${LOCAL_EMBED_PORT:-18001} ;;
  *)
    printf 'Usage: %s gemma|qwen|embed\n' "$0" >&2
    exit 2
    ;;
esac

manifest=$(ssh "$REMOTE" "$REMOTE_CLI info '$SERVICE'")
value() {
  printf '%s\n' "$manifest" | awk -F= -v key="$1" '$1 == key {print $2}'
}

job_id=$(value JOB_ID)
node=$(value NODE)
port=$(value PORT)
ready=$(value READY)

if [[ -z "$job_id" || -z "$node" || -z "$port" || "$ready" != "true" ]]; then
  printf 'Invalid %s service information from the cluster.\n' "$SERVICE" >&2
  exit 1
fi

exec python3 "$SCRIPT_DIR/stdio_bridge.py" listen-one \
  --remote "$REMOTE" \
  --job-id "$job_id" \
  --node "$node" \
  --destination-port "$port" \
  --local-port "$LOCAL_PORT" \
  --remote-python "$REMOTE_PYTHON" \
  --remote-bridge "$REMOTE_BRIDGE"
