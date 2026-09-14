#!/bin/bash
# A100 variant of serve_gemma.sbatch. The only real difference: no
# --kv-cache-dtype fp8. That flag defaults to the Hopper-native fp8e4nv
# format, which vLLM/Triton cannot compile on Ampere (A100) -- confirmed via
# a crashed run: "ValueError: type fp8e4nv not supported in this
# architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')". Rather
# than force one of those older software-emulated formats, this just omits
# the flag so the KV cache uses the model's own dtype (bfloat16) --
# slower/more memory than fp8 would be on H100, but correct on A100.
#SBATCH --job-name=ontomem-gemma
#SBATCH --partition=gpu_a100_8
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
  --max-model-len 65536 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.92 \
  --language-model-only \
  --limit-mm-per-prompt '{"image": 0}' \
  --enforce-eager \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --chat-template "$CHAT_TEMPLATE" \
  --default-chat-template-kwargs '{"enable_thinking": true}'
