#!/bin/bash
# Open local forwards to the generation and embedding services.

set -euo pipefail

REMOTE=gururaj@hpc.bits-hyderabad.ac.in
MANIFEST=/scratch/gururaj/Sriniketh/ontomem-inference/connection.env
LOCAL_LLM_PORT=${LOCAL_LLM_PORT:-18000}
LOCAL_EMBED_PORT=${LOCAL_EMBED_PORT:-18001}

manifest=$(ssh "$REMOTE" "test -f '$MANIFEST' && cat '$MANIFEST'")

value() {
  printf '%s\n' "$manifest" | awk -F= -v key="$1" '$1 == key {print $2}'
}

job_id=$(value JOB_ID)
node=$(value NODE)
llm_port=$(value LLM_PORT)
embed_port=$(value EMBED_PORT)

if [[ -z "$job_id" || -z "$node" || -z "$llm_port" || -z "$embed_port" ]]; then
  printf 'Invalid or incomplete connection manifest: %s\n' "$MANIFEST" >&2
  exit 1
fi

printf 'Forwarding job %s on %s: localhost:%s -> LLM %s, localhost:%s -> embeddings %s\n' \
  "$job_id" "$node" "$LOCAL_LLM_PORT" "$llm_port" "$LOCAL_EMBED_PORT" "$embed_port"

exec ssh "$REMOTE" \
  -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L "$LOCAL_LLM_PORT:$node:$llm_port" \
  -L "$LOCAL_EMBED_PORT:$node:$embed_port"
