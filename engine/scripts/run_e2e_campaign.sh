#!/bin/bash
# Run a resumable synthetic-life campaign after both endpoints are healthy.

set -euo pipefail

ENGINE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CAMPAIGN=${E2E_CAMPAIGN:-"$ENGINE_ROOT/e2e/synthetic_life_v1.json"}
OUTPUT=${E2E_OUTPUT:-"$ENGINE_ROOT/e2e_runs/synthetic_life_v1"}
LIMIT=${E2E_LIMIT-3}

: "${OPENAI_API_KEY:?source ../cluster/cluster_env.sh first}"
: "${OPENAI_BASE_URL:?source ../cluster/cluster_env.sh first}"
: "${OPENAI_MODEL:?source ../cluster/cluster_env.sh first}"
: "${OPENAI_EMBED_BASE_URL:?source ../cluster/cluster_env.sh first}"
: "${OPENAI_EMBED_MODEL:?source ../cluster/cluster_env.sh first}"

curl -fsS "$OPENAI_BASE_URL/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY" >/dev/null
curl -fsS "$OPENAI_EMBED_BASE_URL/models" \
  -H "Authorization: Bearer ${OPENAI_EMBED_API_KEY:-$OPENAI_API_KEY}" >/dev/null

args=(
  --campaign "$CAMPAIGN"
  --output "$OUTPUT"
  --audit-content
)
if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi

cd "$ENGINE_ROOT"
PYTHONPATH=src .venv/bin/python scripts/e2e_campaign.py "${args[@]}"
