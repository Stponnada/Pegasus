#!/bin/bash
#SBATCH --job-name=ontomem-embed
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
MODEL="$SCRATCH_ROOT/hf_cache/hub/models--sentence-transformers--all-roberta-large-v1/snapshots/cf74d8acd4f198de950bf004b262e6accfed5d2c"
NODE=$(hostname -s)
PORT=$(( 20001 + (SLURM_JOB_ID % 10000) * 2 ))
API_KEY=ontomem-cluster
LOG_FILE="$RUN_ROOT/logs/ontomem-embed-$SLURM_JOB_ID.log"
MANIFEST="$RUN_ROOT/embed.env"
MANIFEST_TMP="$RUN_ROOT/embed.env.$SLURM_JOB_ID.tmp"

export HF_HOME="$SCRATCH_ROOT/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_ROOT/logs"
printf 'JOB_ID=%s\nSERVICE=embed\nNODE=%s\nPORT=%s\nAPI_KEY=%s\nMODEL=%s\nLOG_FILE=%s\nJOB_MODE=dedicated\n' \
  "$SLURM_JOB_ID" "$NODE" "$PORT" "$API_KEY" "ontomem-embed" "$LOG_FILE" \
  > "$MANIFEST_TMP"
mv "$MANIFEST_TMP" "$MANIFEST"

exec "$VLLM" serve "$MODEL" \
  --served-model-name ontomem-embed \
  --runner pooling \
  --host 0.0.0.0 \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --max-model-len 256 \
  --gpu-memory-utilization 0.20 \
  --enforce-eager
