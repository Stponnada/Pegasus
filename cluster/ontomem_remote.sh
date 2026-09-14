#!/bin/bash
# One-command lifecycle manager for Gemma, legacy Qwen, and the embedder.

set -euo pipefail

ROOT=/scratch/gururaj/Sriniketh
RUN_ROOT="$ROOT/ontomem-inference"
STATE_ROOT="$RUN_ROOT/state"
POLL_SECONDS=${ONTOMEM_POLL_SECONDS:-10}

mkdir -p "$RUN_ROOT/logs" "$STATE_ROOT"

usage() {
  cat <<'EOF'
Usage:
  ontomem gemma             Start/reuse Gemma 4 31B and wait until ready
  ontomem qwen              Start/reuse legacy Qwen and wait until ready
  ontomem embed             Start/reuse the embedder and wait until ready
  ontomem all               Start/reuse Gemma and the embedder
  ontomem status [service]  Show Slurm and readiness state
  ontomem wait <service>    Wait for an existing job to become ready
  ontomem info <service>    Print the ready service manifest (for scripts)
  ontomem logs <service>    Follow the service log
  ontomem stop <service>    Cancel gemma, qwen, embed, or all

Services: gemma, qwen, embed
EOF
}

validate_service() {
  case "${1:-}" in
    gemma|qwen|embed) ;;
    *)
      printf 'Expected service gemma, qwen, or embed, got: %s\n' "${1:-<empty>}" >&2
      exit 2
      ;;
  esac
}

launcher() {
  printf '%s/serve_%s.sbatch\n' "$RUN_ROOT" "$1"
}

record_file() {
  printf '%s/%s.job\n' "$STATE_ROOT" "$1"
}

manifest_file() {
  printf '%s/%s.env\n' "$RUN_ROOT" "$1"
}

job_name() {
  printf 'ontomem-%s\n' "$1"
}

log_file() {
  printf '%s/logs/%s-%s.log\n' "$RUN_ROOT" "$(job_name "$1")" "$2"
}

read_job_id() {
  local file job_id next_file next_job
  file=$(record_file "$1")
  job_id=$([[ -f "$file" ]] && sed -n '1p' "$file")
  if [[ -n "$job_id" ]] && [[ -n "$(job_state "$job_id")" ]]; then
    printf '%s\n' "$job_id"
    return
  fi

  next_file="$STATE_ROOT/$1.next.job"
  next_job=$([[ -f "$next_file" ]] && sed -n '1p' "$next_file")
  if [[ -n "$next_job" ]] && [[ -n "$(job_state "$next_job")" ]]; then
    printf '%s\n' "$next_job" > "$file"
    rm -f "$next_file"
    printf '%s\n' "$next_job"
    return
  fi

  [[ -n "$job_id" ]] && printf '%s\n' "$job_id"
}

job_state() {
  squeue -h -j "$1" -o '%T' 2>/dev/null | sed -n '1p'
}

job_summary() {
  squeue -h -j "$1" -o '%T %R' 2>/dev/null | sed -n '1p'
}

is_ready() {
  local service=$1
  local job_id=$2
  local state manifest manifest_job log
  state=$(job_state "$job_id")
  [[ "$state" == "RUNNING" ]] || return 1
  manifest=$(manifest_file "$service")
  [[ -f "$manifest" ]] || return 1
  manifest_job=$(sed -n 's/^JOB_ID=//p' "$manifest")
  [[ "$manifest_job" == "$job_id" ]] || return 1
  log=$(sed -n 's/^LOG_FILE=//p' "$manifest")
  [[ -n "$log" ]] || log=$(log_file "$service" "$job_id")
  [[ -f "$log" ]] || return 1
  grep -q 'Application startup complete' "$log"
}

ensure_service() {
  local service=$1
  local job_id state launch
  validate_service "$service"
  job_id=$(read_job_id "$service" || true)
  if [[ -n "$job_id" ]]; then
    state=$(job_state "$job_id")
    if [[ -n "$state" ]]; then
      printf '%s already has job %s (%s).\n' "$service" "$job_id" "$state"
      return
    fi
  fi

  launch=$(launcher "$service")
  if [[ ! -f "$launch" ]]; then
    printf 'Missing launcher: %s\n' "$launch" >&2
    exit 1
  fi
  job_id=$(sbatch --parsable "$launch")
  printf '%s\n' "$job_id" > "$(record_file "$service")"
  printf 'Submitted %s as job %s.\n' "$service" "$job_id"
}

wait_service() {
  local service=$1
  local job_id state summary previous=
  validate_service "$service"
  job_id=$(read_job_id "$service" || true)
  if [[ -z "$job_id" ]]; then
    printf 'No recorded %s job. Run: ontomem %s\n' "$service" "$service" >&2
    return 1
  fi

  printf 'Waiting for %s job %s' "$service" "$job_id"
  while true; do
    if is_ready "$service" "$job_id"; then
      printf '\n%s is ready.\n' "$service"
      print_endpoint "$service"
      return
    fi

    state=$(job_state "$job_id")
    if [[ -z "$state" ]]; then
      state=$(sacct -n -X -j "$job_id" -o State 2>/dev/null | awk 'NF {print $1; exit}')
      printf '\n%s job %s ended with state %s.\n' \
        "$service" "$job_id" "${state:-UNKNOWN}" >&2
      printf 'Inspect: %s\n' "$(log_file "$service" "$job_id")" >&2
      return 1
    fi
    summary=$(job_summary "$job_id")
    if [[ "$summary" != "$previous" ]]; then
      printf '\n  %s\n' "$summary"
      previous=$summary
    else
      printf '.'
    fi
    sleep "$POLL_SECONDS"
  done
}

print_endpoint() {
  local service=$1
  local manifest
  manifest=$(manifest_file "$service")
  if [[ -f "$manifest" ]]; then
    local node port model
    node=$(sed -n 's/^NODE=//p' "$manifest")
    port=$(sed -n 's/^PORT=//p' "$manifest")
    model=$(sed -n 's/^MODEL=//p' "$manifest")
    printf '  model: %s\n  cluster endpoint: %s:%s\n' "$model" "$node" "$port"
  fi
}

status_service() {
  local service=$1
  local job_id state ready=no
  validate_service "$service"
  job_id=$(read_job_id "$service" || true)
  if [[ -z "$job_id" ]]; then
    printf '%-6s not submitted\n' "$service"
    return
  fi
  state=$(job_state "$job_id")
  [[ -n "$state" ]] || state=$(sacct -n -X -j "$job_id" -o State 2>/dev/null | awk 'NF {print $1; exit}')
  is_ready "$service" "$job_id" && ready=yes
  printf '%-6s job=%-8s state=%-12s ready=%s\n' \
    "$service" "$job_id" "${state:-UNKNOWN}" "$ready"
}

info_service() {
  local service=$1
  local job_id manifest
  validate_service "$service"
  job_id=$(read_job_id "$service" || true)
  if [[ -z "$job_id" ]] || ! is_ready "$service" "$job_id"; then
    printf '%s is not ready. On the cluster run: ontomem %s\n' \
      "$service" "$service" >&2
    return 1
  fi
  manifest=$(manifest_file "$service")
  cat "$manifest"
  printf 'READY=true\n'
}

logs_service() {
  local service=$1
  local job_id log manifest manifest_job
  validate_service "$service"
  job_id=$(read_job_id "$service" || true)
  [[ -n "$job_id" ]] || {
    printf 'No recorded %s job.\n' "$service" >&2
    return 1
  }
  manifest=$(manifest_file "$service")
  manifest_job=
  if [[ -f "$manifest" ]]; then
    manifest_job=$(sed -n 's/^JOB_ID=//p' "$manifest")
  fi
  if [[ "$manifest_job" == "$job_id" ]]; then
    log=$(sed -n 's/^LOG_FILE=//p' "$manifest")
  fi
  [[ -n "${log:-}" ]] || log=$(log_file "$service" "$job_id")
  touch "$log"
  exec tail -n 80 -f "$log"
}

stop_service() {
  local service=$1
  local job_id state manifest manifest_job job_mode
  validate_service "$service"
  job_id=$(read_job_id "$service" || true)
  if [[ -z "$job_id" ]]; then
    printf '%s has no recorded job.\n' "$service"
    return
  fi
  state=$(job_state "$job_id")
  if [[ -z "$state" ]]; then
    printf '%s job %s is already inactive.\n' "$service" "$job_id"
    return
  fi
  manifest=$(manifest_file "$service")
  if [[ -f "$manifest" ]]; then
    manifest_job=$(sed -n 's/^JOB_ID=//p' "$manifest")
    job_mode=$(sed -n 's/^JOB_MODE=//p' "$manifest")
    if [[ "$manifest_job" == "$job_id" && "$job_mode" == "shared" ]]; then
      printf '%s job %s is a shared migration allocation; refusing to cancel it through this service command.\n' \
        "$service" "$job_id" >&2
      printf 'Let it expire, or cancel the shared Slurm job explicitly with scancel %s.\n' \
        "$job_id" >&2
      return 1
    fi
  fi
  scancel "$job_id"
  printf 'Cancelled %s job %s.\n' "$service" "$job_id"
}

command=${1:-}
case "$command" in
  gemma|qwen|embed)
    ensure_service "$command"
    wait_service "$command"
    ;;
  all)
    ensure_service gemma
    ensure_service embed
    wait_service gemma
    wait_service embed
    ;;
  status)
    if [[ -n "${2:-}" ]]; then
      status_service "$2"
    else
      status_service gemma
      status_service embed
      status_service qwen
    fi
    ;;
  wait)
    [[ -n "${2:-}" ]] || { usage; exit 2; }
    wait_service "$2"
    ;;
  info)
    [[ -n "${2:-}" ]] || { usage; exit 2; }
    info_service "$2"
    ;;
  logs)
    [[ -n "${2:-}" ]] || { usage; exit 2; }
    logs_service "$2"
    ;;
  stop)
    case "${2:-}" in
      gemma|qwen|embed) stop_service "$2" ;;
      all)
        stop_service gemma
        stop_service embed
        ;;
      *) usage; exit 2 ;;
    esac
    ;;
  help|-h|--help|"")
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
