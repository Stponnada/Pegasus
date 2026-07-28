# Cluster Inference

`serve_ontomem.sbatch` starts two OpenAI-compatible vLLM services in one Slurm
allocation:

- Qwen2.5-32B-Instruct for extraction, merge disambiguation, and supersession.
- `all-roberta-large-v1` for dense embeddings.

Submit from the cluster after copying the script:

```bash
mkdir -p /scratch/gururaj/Sriniketh/ontomem-inference/logs
sbatch serve_ontomem.sbatch
```

Once the job is running, read
`/scratch/gururaj/Sriniketh/ontomem-inference/connection.env`. It records the
allocated node and per-job ports. Forward both services to the local machine:

```bash
ssh -N \
  -L 18000:${NODE}:${LLM_PORT} \
  -L 18001:${NODE}:${EMBED_PORT} \
  gururaj@hpc.bits-hyderabad.ac.in
```

The local engine then uses `http://127.0.0.1:18000/v1` for generation and
`http://127.0.0.1:18001/v1` for embeddings. Do not expose compute-node ports to
the public internet.

```bash
export OPENAI_API_KEY=ontomem-cluster
export OPENAI_BASE_URL=http://127.0.0.1:18000/v1
export OPENAI_MODEL=ontomem-llm
export OPENAI_EMBED_API_KEY=ontomem-cluster
export OPENAI_EMBED_BASE_URL=http://127.0.0.1:18001/v1
export OPENAI_EMBED_MODEL=ontomem-embed
```

## Prepared workflow

Use three terminals from the repository root:

```bash
# Terminal 1: keep the tunnel open
./cluster/tunnel_ontomem.sh

# Terminal 2: configure and check both endpoints
source cluster/cluster_env.sh
./cluster/check_ontomem.sh

# Terminal 3: run a three-conversation pilot
source cluster/cluster_env.sh
engine/scripts/run_e2e_campaign.sh
```

The pilot is resumable under `engine/e2e_runs/synthetic_life_v1/`. After
inspecting it, set `E2E_LIMIT=` to continue through all 20 sessions.
