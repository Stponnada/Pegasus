#!/bin/bash
# Start, stop, or inspect persistent Mac-side service bridges.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ACTION=${1:-status}
SERVICE=${2:-all}

validate_service() {
  case "$1" in
    gemma|qwen|embed) ;;
    *)
      printf 'Expected gemma, qwen, or embed, got: %s\n' "$1" >&2
      exit 2
      ;;
  esac
}

local_port() {
  case "$1" in
    gemma|qwen) printf '%s\n' "${LOCAL_LLM_PORT:-${LOCAL_QWEN_PORT:-18000}}" ;;
    embed) printf '%s\n' "${LOCAL_EMBED_PORT:-18001}" ;;
  esac
}

pid_file() {
  printf '/tmp/ontomem-%s-bridge.pid\n' "$1"
}

log_file() {
  printf '/tmp/ontomem-%s-bridge.log\n' "$1"
}

health() {
  local service=$1
  local port endpoint
  port=$(local_port "$service")
  case "$service" in
    gemma|qwen) endpoint="http://127.0.0.1:$port/v1/models" ;;
    embed) endpoint="http://127.0.0.1:$port/v1/models" ;;
  esac
  curl -fsS --max-time 30 "$endpoint" \
    -H 'Authorization: Bearer ontomem-cluster' >/dev/null 2>&1
}

managed_pid() {
  local service=$1
  local file pid command port
  file=$(pid_file "$service")
  [[ -f "$file" ]] || return 1
  pid=$(sed -n '1p' "$file")
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  command=$(ps -p "$pid" -o command= 2>/dev/null || true)
  port=$(local_port "$service")
  [[ "$command" == *"stdio_bridge.py listen-one"* ]] || return 1
  [[ "$command" == *"--local-port $port"* ]] || return 1
  printf '%s\n' "$pid"
}

start_service() {
  local service=$1
  local port pid file log attempt
  validate_service "$service"
  port=$(local_port "$service")

  if pid=$(managed_pid "$service" 2>/dev/null); then
    if health "$service"; then
      printf '%s bridge is ready on localhost:%s.\n' "$service" "$port"
      return
    fi
    kill "$pid"
    for _ in 1 2 3 4 5; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid"
    printf 'Retired stale %s bridge process %s.\n' "$service" "$pid"
  fi

  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    printf 'localhost:%s is occupied by a different process.\n' "$port" >&2
    printf 'Stop that listener, then retry: ontomem-bridge start %s\n' "$service" >&2
    return 1
  fi
  file=$(pid_file "$service")
  log=$(log_file "$service")
  nohup "$SCRIPT_DIR/connect_service.sh" "$service" >"$log" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$file"
  printf 'Started %s bridge process %s.\n' "$service" "$pid"

  for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
    if health "$service"; then
      printf '%s bridge is ready on localhost:%s.\n' "$service" "$port"
      return
    fi
    sleep 1
  done

  printf '%s bridge did not become healthy. See %s\n' \
    "$service" "$(log_file "$service")" >&2
  return 1
}

stop_service() {
  local service=$1
  local pid file
  validate_service "$service"
  file=$(pid_file "$service")
  if pid=$(managed_pid "$service" 2>/dev/null); then
    kill "$pid"
    for _ in 1 2 3 4 5; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid"
    printf 'Stopped %s bridge process %s.\n' "$service" "$pid"
  else
    printf '%s bridge is not managed or not running.\n' "$service"
  fi
  : > "$file"
}

status_service() {
  local service=$1
  local port pid=-
  validate_service "$service"
  port=$(local_port "$service")
  pid=$(managed_pid "$service" 2>/dev/null || printf '%s' '-')
  if health "$service"; then
    printf '%-6s ready=yes port=%s pid=%s\n' "$service" "$port" "$pid"
  else
    printf '%-6s ready=no  port=%s pid=%s\n' "$service" "$port" "$pid"
  fi
}

run_for_selection() {
  local function=$1
  case "$SERVICE" in
    gemma|qwen|embed) "$function" "$SERVICE" ;;
    all)
      "$function" gemma
      "$function" embed
      ;;
    *)
      printf 'Expected gemma, qwen, embed, or all, got: %s\n' "$SERVICE" >&2
      exit 2
      ;;
  esac
}

case "$ACTION" in
  start) run_for_selection start_service ;;
  stop) run_for_selection stop_service ;;
  status) run_for_selection status_service ;;
  *)
    printf 'Usage: %s start|stop|status [gemma|qwen|embed|all]\n' "$0" >&2
    exit 2
    ;;
esac
