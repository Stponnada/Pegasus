#!/bin/bash
#SBATCH --job-name=ontomem-inference
#SBATCH --partition=gpu_a100_8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gururaj/Sriniketh/ontomem-inference/logs/%x-%j.out
#SBATCH --error=/scratch/gururaj/Sriniketh/ontomem-inference/logs/%x-%j.err

set -euo pipefail

SCRATCH_ROOT=/scratch/gururaj/Sriniketh
RUN_ROOT="$SCRATCH_ROOT/ontomem-inference"
VLLM="$SCRATCH_ROOT/envs/fern/bin/vllm"
LLM_MODEL="$SCRATCH_ROOT/models/Qwen2.5-32B-Instruct"
EMBED_MODEL="$SCRATCH_ROOT/hf_cache/hub/models--sentence-transformers--all-roberta-large-v1/snapshots/cf74d8acd4f198de950bf004b262e6accfed5d2c"

export HF_HOME="$SCRATCH_ROOT/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

# Derive ports from the job id so two experiments can share a compute node.
PORT_OFFSET=$(( (SLURM_JOB_ID % 10000) * 2 ))
LLM_PORT=$(( 20000 + PORT_OFFSET ))
EMBED_PORT=$(( LLM_PORT + 1 ))
NODE_NAME=$(hostname -s)
API_KEY=ontomem-cluster

mkdir -p "$RUN_ROOT/logs"
printf 'JOB_ID=%s\nNODE=%s\nLLM_PORT=%s\nEMBED_PORT=%s\nAPI_KEY=%s\n' \
  "$SLURM_JOB_ID" "$NODE_NAME" "$LLM_PORT" "$EMBED_PORT" "$API_KEY" \
  > "$RUN_ROOT/connection.env"

CUDA_VISIBLE_DEVICES=0 "$VLLM" serve "$LLM_MODEL" \
  --served-model-name ontomem-llm \
  --runner generate \
  --host 0.0.0.0 \
  --port "$LLM_PORT" \
  --api-key "$API_KEY" \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --disable-log-requests \
  > "$RUN_ROOT/logs/llm-$SLURM_JOB_ID.log" 2>&1 &
LLM_PID=$!

CUDA_VISIBLE_DEVICES=1 "$VLLM" serve "$EMBED_MODEL" \
  --served-model-name ontomem-embed \
  --runner pooling \
  --host 0.0.0.0 \
  --port "$EMBED_PORT" \
  --api-key "$API_KEY" \
  --max-model-len 512 \
  --gpu-memory-utilization 0.20 \
  --disable-log-requests \
  > "$RUN_ROOT/logs/embed-$SLURM_JOB_ID.log" 2>&1 &
EMBED_PID=$!

cleanup() {
  kill "$LLM_PID" "$EMBED_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait -n "$LLM_PID" "$EMBED_PID"
