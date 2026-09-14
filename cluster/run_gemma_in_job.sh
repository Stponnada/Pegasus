#!/bin/bash
# Run Gemma inside an existing Slurm allocation without cancelling that job.
# This is used only for a zero-allocation-loss migration; normal launches use
# serve_gemma.sbatch.

set -euo pipefail

JOB_ID=${1:?Usage: run_gemma_in_job.sh JOB_ID PORT}
PORT=${2:?Usage: run_gemma_in_job.sh JOB_ID PORT}
SCRATCH_ROOT=/scratch/gururaj/Sriniketh
RUN_ROOT="$SCRATCH_ROOT/ontomem-inference"
CONTAINER="$SCRATCH_ROOT/containers/vllm-gemma4.sif"
MODEL="$SCRATCH_ROOT/models/google--gemma-4-31B-it"
CHAT_TEMPLATE="$RUN_ROOT/tool_chat_template_gemma4.jinja"
API_KEY=ontomem-cluster

export HF_HOME="$SCRATCH_ROOT/hf_cache"
export TOKENIZERS_PARALLELISM=false
export APPTAINER_CACHEDIR="$SCRATCH_ROOT/apptainer-cache"
export APPTAINER_TMPDIR="$SCRATCH_ROOT/apptainer-tmp"

exec srun --jobid="$JOB_ID" --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=96G --gpus-per-task=1 --gpu-bind=map_gpu:0 \
  apptainer exec --nv --bind "$SCRATCH_ROOT:$SCRATCH_ROOT" "$CONTAINER" \
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
