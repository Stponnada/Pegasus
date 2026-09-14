#!/bin/bash
#SBATCH --job-name=ontomem-gemma
#SBATCH --partition=gpu_h100_4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gururaj/Sriniketh/ontomem-inference/logs/%x-%j.log
#SBATCH --error=/scratch/gururaj/Sriniketh/ontomem-inference/logs/%x-%j.log

set -euo pipefail

SCRATCH_ROOT=/scratch/gururaj/Sriniketh
RUN_ROOT="$SCRATCH_ROOT/ontomem-inference"
CONTAINER="$SCRATCH_ROOT/containers/vllm-gemma4.sif"
MODEL="$SCRATCH_ROOT/models/google--gemma-4-31B-it"
CHAT_TEMPLATE="$RUN_ROOT/tool_chat_template_gemma4.jinja"
NODE=$(hostname -s)
PORT=$(( 20000 + (SLURM_JOB_ID % 10000) * 2 ))
API_KEY=ontomem-cluster
LOG_FILE="$RUN_ROOT/logs/ontomem-gemma-$SLURM_JOB_ID.log"
MANIFEST="$RUN_ROOT/gemma.env"
MANIFEST_TMP="$RUN_ROOT/gemma.env.$SLURM_JOB_ID.tmp"

export HF_HOME="$SCRATCH_ROOT/hf_cache"
export TOKENIZERS_PARALLELISM=false
export APPTAINER_CACHEDIR="$SCRATCH_ROOT/apptainer-cache"
export APPTAINER_TMPDIR="$SCRATCH_ROOT/apptainer-tmp"

runtime_ready() {
  local required
  for required in "$CONTAINER" "$MODEL/config.json" "$MODEL/model-00001-of-00002.safetensors" \
    "$MODEL/model-00002-of-00002.safetensors" "$CHAT_TEMPLATE"; do
    [[ -s "$required" ]] || return 1
  done
}

# This job may be queued while the checkpoint/container is still downloading.
# Keeping the wait inside the allocation avoids giving up a scarce H100 during
# the Qwen-to-Gemma handoff. Fail after one hour rather than wait forever.
for attempt in $(seq 1 360); do
  runtime_ready && break
  if (( attempt % 6 == 1 )); then
    printf 'Waiting for the Gemma checkpoint, container, and chat template (%s/360).\n' "$attempt"
  fi
  sleep 10
done

if ! runtime_ready; then
  printf 'Gemma runtime files were not ready after one hour.\n' >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/logs"
printf 'JOB_ID=%s\nSERVICE=gemma\nNODE=%s\nPORT=%s\nAPI_KEY=%s\nMODEL=%s\nLOG_FILE=%s\nTOOL_CALLS=true\nTHINKING=true\nJOB_MODE=dedicated\n' \
  "$SLURM_JOB_ID" "$NODE" "$PORT" "$API_KEY" "ontomem-llm" "$LOG_FILE" \
  > "$MANIFEST_TMP"
mv "$MANIFEST_TMP" "$MANIFEST"

exec apptainer exec --nv --bind "$SCRATCH_ROOT:$SCRATCH_ROOT" "$CONTAINER" \
  /usr/local/bin/vllm serve "$MODEL" \
  --served-model-name ontomem-llm \
  --runner generate \
  --host 0.0.0.0 \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.92 \
  --kv-cache-dtype fp8 \
  --language-model-only \
  --limit-mm-per-prompt '{"image": 0}' \
  --enforce-eager \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --chat-template "$CHAT_TEMPLATE" \
  --default-chat-template-kwargs '{"enable_thinking": true}'
