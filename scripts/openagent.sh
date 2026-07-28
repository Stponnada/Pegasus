#!/usr/bin/env bash
# OpenAgent: ensure the ontomem memory engine is running, then launch opencode
# (with the ontomem plugin already registered in opencode/.opencode/opencode.jsonc)
# in the current terminal. Meant to be run via the `OpenAgent` shell alias.
set -euo pipefail

MEMORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_DIR="$MEMORY_ROOT/engine"
OPENCODE_DIR="$MEMORY_ROOT/opencode"

ONTOMEM_HOST="${ONTOMEM_HOST:-127.0.0.1}"
ONTOMEM_PORT="${ONTOMEM_PORT:-8765}"
export ONTOMEM_DIR="${ONTOMEM_DIR:-$HOME/.ontomem}"
ONTOMEM_LOG="$ONTOMEM_DIR/service.log"

mkdir -p "$ONTOMEM_DIR"

is_engine_up() {
  curl -s -o /dev/null -m 1 -X POST "http://${ONTOMEM_HOST}:${ONTOMEM_PORT}/health" -d '{}'
}

if is_engine_up; then
  echo "OpenAgent: memory engine already running on ${ONTOMEM_HOST}:${ONTOMEM_PORT}" >&2
else
  echo "OpenAgent: starting memory engine on ${ONTOMEM_HOST}:${ONTOMEM_PORT} (log: $ONTOMEM_LOG)..." >&2
  if [ -f "$ENGINE_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENGINE_DIR/.env"
    set +a
  fi

  (
    cd "$ENGINE_DIR"
    PYTHONPATH=src ONTOMEM_HOST="$ONTOMEM_HOST" ONTOMEM_PORT="$ONTOMEM_PORT" \
      nohup uv run python -m ontomem.service >>"$ONTOMEM_LOG" 2>&1 &
  )

  for _ in $(seq 1 30); do
    if is_engine_up; then
      break
    fi
    sleep 0.3
  done

  if ! is_engine_up; then
    echo "OpenAgent: memory engine failed to start — check $ONTOMEM_LOG" >&2
    exit 1
  fi
  echo "OpenAgent: memory engine ready" >&2
fi

cd "$OPENCODE_DIR"
exec bun run dev -- "$@"
