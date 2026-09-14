#!/bin/bash
# Open local forwards through SSH and an existing Slurm allocation.

set -euo pipefail

REMOTE=gururaj@hpc.bits-hyderabad.ac.in
MANIFEST=/scratch/gururaj/Sriniketh/ontomem-inference/connection.env
LOCAL_LLM_PORT=${LOCAL_LLM_PORT:-18000}
LOCAL_EMBED_PORT=${LOCAL_EMBED_PORT:-18001}
REMOTE_PYTHON=/scratch/gururaj/Sriniketh/envs/fern/bin/python
REMOTE_BRIDGE=/scratch/gururaj/Sriniketh/ontomem-inference/stdio_bridge.py
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

manifest=$(ssh "$REMOTE" "test -f '$MANIFEST' && cat '$MANIFEST'")

value() {
  printf '%s\n' "$manifest" | awk -F= -v key="$1" '$1 == key {print $2}'
}

job_id=$(value JOB_ID)
llm_node=$(value LLM_NODE)
embed_node=$(value EMBED_NODE)
llm_port=$(value LLM_PORT)
embed_port=$(value EMBED_PORT)

if [[ -z "$job_id" || -z "$llm_node" || -z "$embed_node" || -z "$llm_port" || -z "$embed_port" ]]; then
  printf 'Invalid or incomplete connection manifest: %s\n' "$MANIFEST" >&2
  exit 1
fi

printf 'Forwarding job %s: localhost:%s -> LLM %s:%s, localhost:%s -> embeddings %s:%s\n' \
  "$job_id" "$LOCAL_LLM_PORT" "$llm_node" "$llm_port" \
  "$LOCAL_EMBED_PORT" "$embed_node" "$embed_port"

# The cluster disables OpenSSH direct-tcpip forwarding. The Python bridge
# carries each local connection over SSH stdio into a tiny overlapping Slurm
# step on the already allocated compute node.
exec python3 "$SCRIPT_DIR/stdio_bridge.py" listen \
  --remote "$REMOTE" \
  --job-id "$job_id" \
  --llm-node "$llm_node" \
  --embed-node "$embed_node" \
  --llm-port "$llm_port" \
  --embed-port "$embed_port" \
  --local-llm-port "$LOCAL_LLM_PORT" \
  --local-embed-port "$LOCAL_EMBED_PORT" \
  --remote-python "$REMOTE_PYTHON" \
  --remote-bridge "$REMOTE_BRIDGE"
