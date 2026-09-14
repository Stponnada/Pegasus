# Cluster Inference

Gemma 4 and the embedder are exposed as ordinary OpenAI-compatible APIs.
Slurm allocation, ports, service manifests, and Mac-to-cluster bridging are
handled by the scripts in this directory.

## Fast Path

On the cluster, start or reuse both services and wait for readiness:

```bash
Cluster
ontomem all
```

Then run this from any project directory on the Mac:

```bash
gemma-code
```

`gemma-code` connects both localhost bridges, starts or reuses the Pegasus
memory engine, and opens the plugin-capable local OpenCode build with
`ontomem-h100/ontomem-llm` selected. The launch directory remains the active
coding workspace. Run `/memory` inside OpenCode to open the graph viewer.

The old `qwen-code` command is retained as a compatibility alias for
`gemma-code`; it no longer selects Qwen.

## Services and Operations

The cluster command is installed at `/home/gururaj/bin/ontomem`:

```bash
ontomem gemma          # generation only
ontomem embed          # embeddings only
ontomem all            # Gemma and embedder
ontomem status
ontomem info gemma
ontomem logs gemma
ontomem wait gemma
ontomem stop all
```

Starting a service is idempotent: a pending or running job is reused. Pressing
Ctrl-C stops only the readiness wait, not the Slurm job. The legacy
`ontomem qwen` command remains available solely as a fallback.

On the Mac, bridge commands are also idempotent:

```bash
ontomem-bridge start all
ontomem-bridge status all
ontomem-bridge stop all
```

The local endpoints are:

- Chat: `http://127.0.0.1:18000/v1`, model `ontomem-llm`
- Embeddings: `http://127.0.0.1:18001/v1`, model `ontomem-embed`
- Bearer token: `ontomem-cluster`

## Gemma Runtime

`serve_gemma.sbatch` serves the dense `google/gemma-4-31B-it` checkpoint from
an isolated official vLLM Gemma 4 container. Thinking is enabled by default.
The Gemma 4 reasoning and tool-call parsers, official tool chat template,
automatic tool choice, eager execution, prefix caching, FP8 KV cache, and a
16K operational context limit are enabled. Multimodal towers are disabled
because this stack needs text and tools only.

The model config must continue to report `enable_moe_block: false`, 60 layers,
and a hidden size of 5376. Those values distinguish the 31B dense checkpoint
from the 26B-A4B mixture-of-experts model.

`serve_gemma.sbatch` and `serve_embed.sbatch` request one H100 and four CPUs
each so Slurm can schedule them independently. The embedder retains its actual
256-token input limit.

Each service publishes its current node and private port in:

```text
/scratch/gururaj/Sriniketh/ontomem-inference/gemma.env
/scratch/gururaj/Sriniketh/ontomem-inference/embed.env
```

Normal SSH TCP forwarding is unavailable on the cluster.
`stdio_bridge.py` transports each localhost connection through SSH standard
I/O and a small overlapping Slurm step.

## Manual Dogfood Campaign

The inference launchers do not run the campaign automatically:

```bash
source cluster/cluster_env.sh
engine/scripts/run_e2e_campaign.sh
```
