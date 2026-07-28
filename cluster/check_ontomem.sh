#!/bin/bash
# Show Slurm state and, when forwarded, query both model endpoints.

set -euo pipefail

REMOTE=gururaj@hpc.bits-hyderabad.ac.in
JOB_ID=${ONTOMEM_JOB_ID:-261844}
LOCAL_LLM_PORT=${LOCAL_LLM_PORT:-18000}
LOCAL_EMBED_PORT=${LOCAL_EMBED_PORT:-18001}
API_KEY=${OPENAI_API_KEY:-ontomem-cluster}

ssh "$REMOTE" "squeue -j '$JOB_ID' -o '%.18i %.12P %.24j %.8T %.10M %.19S %R'"

if curl -fsS --max-time 2 "http://127.0.0.1:$LOCAL_LLM_PORT/v1/models" \
  -H "Authorization: Bearer $API_KEY" >/dev/null 2>&1; then
  printf 'generation endpoint: ready\n'
else
  printf 'generation endpoint: not reachable through localhost:%s\n' "$LOCAL_LLM_PORT"
fi

if curl -fsS --max-time 2 "http://127.0.0.1:$LOCAL_EMBED_PORT/v1/models" \
  -H "Authorization: Bearer $API_KEY" >/dev/null 2>&1; then
  printf 'embedding endpoint: ready\n'
else
  printf 'embedding endpoint: not reachable through localhost:%s\n' "$LOCAL_EMBED_PORT"
fi
