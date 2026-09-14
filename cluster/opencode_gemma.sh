#!/bin/bash
# Start the cluster-backed ontomem stack, then open the plugin-capable OpenCode
# checkout with the H100 Gemma 4 31B reasoning model.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MEMORY_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
ENGINE_DIR="$MEMORY_ROOT/engine"
OPENCODE_DIR="$MEMORY_ROOT/opencode"
PROJECT_DIR=$(pwd -P)
MODEL=ontomem-h100/ontomem-llm
ONTOMEM_HOST=${ONTOMEM_HOST:-127.0.0.1}
ONTOMEM_PORT=${ONTOMEM_PORT:-8765}
ONTOMEM_URL=${ONTOMEM_URL:-http://${ONTOMEM_HOST}:${ONTOMEM_PORT}}
ONTOMEM_DIR=${ONTOMEM_DIR:-$HOME/.ontomem}
ONTOMEM_LOG="$ONTOMEM_DIR/service.log"

if ! command -v bun >/dev/null 2>&1; then
  printf 'bun is not installed or not on PATH.\n' >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  printf 'uv is not installed or not on PATH.\n' >&2
  exit 1
fi

if [ ! -f "$OPENCODE_DIR/.opencode/tui.json" ] ||
  [ ! -f "$OPENCODE_DIR/.opencode/plugins/memory-viewer.tsx" ]; then
  printf 'The local OpenCode memory-viewer plugin is missing.\n' >&2
  exit 1
fi

is_engine_up() {
  curl -fsS -o /dev/null --max-time 1 -X POST "$ONTOMEM_URL/health" \
    -H 'Content-Type: application/json' -d '{}' 2>/dev/null
}

# Memory reads need the embedder, while Gemma handles extraction, resolution,
# and the OpenCode conversation itself.
"$SCRIPT_DIR/ontomem_bridge.sh" start all

mkdir -p "$ONTOMEM_DIR"
if is_engine_up; then
  printf 'gemma-code: memory engine already running at %s\n' "$ONTOMEM_URL" >&2
else
  printf 'gemma-code: starting memory engine (log: %s)\n' "$ONTOMEM_LOG" >&2
  (
    cd "$ENGINE_DIR"
    OPENAI_API_KEY=ontomem-cluster \
      OPENAI_BASE_URL=http://127.0.0.1:18000/v1 \
      OPENAI_MODEL=ontomem-llm \
      OPENAI_EMBED_API_KEY=ontomem-cluster \
      OPENAI_EMBED_BASE_URL=http://127.0.0.1:18001/v1 \
      OPENAI_EMBED_MODEL=ontomem-embed \
      ONTOMEM_DIR="$ONTOMEM_DIR" \
      ONTOMEM_HOST="$ONTOMEM_HOST" \
      ONTOMEM_PORT="$ONTOMEM_PORT" \
      PYTHONPATH=src \
      nohup uv run python -m ontomem.service >>"$ONTOMEM_LOG" 2>&1 &
  )

  for _ in $(seq 1 40); do
    if is_engine_up; then
      break
    fi
    sleep 0.25
  done

  if ! is_engine_up; then
    printf 'Memory engine failed to start. Check %s\n' "$ONTOMEM_LOG" >&2
    exit 1
  fi
fi

# The released OpenCode binary does not contain this checkout's TUI-plugin
# runtime. Point the local build at its memory configs while keeping the
# user's current directory as the actual coding workspace.
export ONTOMEM_URL
export OPENCODE_CONFIG="$OPENCODE_DIR/.opencode/opencode.jsonc"
export OPENCODE_TUI_CONFIG="$OPENCODE_DIR/.opencode/tui.json"

cd "$OPENCODE_DIR"
exec bun run dev -- "$PROJECT_DIR" --model "$MODEL" "$@"
