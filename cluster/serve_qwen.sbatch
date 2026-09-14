#!/bin/bash
#SBATCH --job-name=ontomem-qwen
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
VLLM="$SCRATCH_ROOT/envs/fern/bin/vllm"
MODEL="$SCRATCH_ROOT/models/Qwen2.5-32B-Instruct"
NODE=$(hostname -s)
PORT=$(( 20000 + (SLURM_JOB_ID % 10000) * 2 ))
API_KEY=ontomem-cluster
LOG_FILE="$RUN_ROOT/logs/ontomem-qwen-$SLURM_JOB_ID.log"
MANIFEST="$RUN_ROOT/qwen.env"
MANIFEST_TMP="$RUN_ROOT/qwen.env.$SLURM_JOB_ID.tmp"

export HF_HOME="$SCRATCH_ROOT/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_ROOT/logs"
printf 'JOB_ID=%s\nSERVICE=qwen\nNODE=%s\nPORT=%s\nAPI_KEY=%s\nMODEL=%s\nLOG_FILE=%s\nTOOL_CALLS=true\nJOB_MODE=dedicated\n' \
  "$SLURM_JOB_ID" "$NODE" "$PORT" "$API_KEY" "ontomem-llm" "$LOG_FILE" \
  > "$MANIFEST_TMP"
mv "$MANIFEST_TMP" "$MANIFEST"

exec "$VLLM" serve "$MODEL" \
  --served-model-name ontomem-llm \
  --runner generate \
  --host 0.0.0.0 \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.95 \
  --enforce-eager \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
