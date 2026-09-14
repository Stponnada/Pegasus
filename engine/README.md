# ontomem — ontology-based long-term memory engine (Phase 1)

A knowledge-graph long-term memory for LLMs: recall is selective, context-
triggered, and proportional to connection strength rather than recency of an
explicit write. Implements spec **v0.2.1** (`../Documentation/`). Standalone,
host-agnostic Python engine; integrates with opencode via a thin plugin adapter.

## Layout

```
src/ontomem/
  model.py            Node / Edge / Episode dataclasses (edge-only strength/decay)
  store.py            in-memory pointer-graph, atomic JSON snapshot, neighbourhood BFS
  decay.py            exponential decay + Hebbian reinforcement math
  journal.py          append-only JSONL audit trail
  embeddings.py       Embedder interface, HashingEmbedder (offline) + GeminiEmbedder, numpy ANN
  extractor_prompt.py the §7 extractor prompt
  extractor.py        Stage 1: parse/salvage/normalise + Gemini call
  merge.py            Stage 2a: deterministic candidate generation (5 rules)
  merge_llm.py        Stage 2b: LLM disambiguation + confidence banding
  consolidate.py      Stage 0 context assembly, merge planning, deterministic Stage 3 write
  retriever.py        Read: noun-extract -> seed -> spreading activation -> tiered injection
  engine.py           Engine facade: read / write / retrieve_memory / decay + persistence
  service.py          stdlib HTTP service exposing the plain-data contract
adapters/opencode/    thin opencode plugin adapter (TS)
tests/                171 tests (167 hermetic + 4 live integration)
```

## Develop

```bash
uv venv && uv pip install -e ".[dev,llm]"

uv run pytest                 # hermetic suite (no network) — the default loop guard
uv run pytest -m integration  # live Gemini tests (needs GEMINI_API_KEY in .env)
```

Test discipline: every unit has a test written with it; the full hermetic suite
stays green at every step; LLM-dependent code is covered by opt-in integration
tests. Deterministic logic is asserted against hand-checked values.

## Run the service

```bash
GEMINI_API_KEY=... ONTOMEM_DIR=~/.ontomem uv run python -m ontomem.service
# POST /read /write /retrieve_memory /decay /health  (plain JSON)
```

With no embedding backend configured (no Gemini key, no `OPENAI_EMBED_*`),
the service defaults to `LocalEmbedder` (`fastembed`, fully offline) if the
`local` extra is installed, falling back further to the lexical
`HashingEmbedder` only if it isn't.

The process self-exits after `ONTOMEM_IDLE_SHUTDOWN_MINUTES` (default 60,
`0` disables it) with no requests — meant for a detached-spawned deployment
(e.g. the opencode plugin) that shouldn't run forever on an idle machine.
Set it to `0` for an always-on deployment (e.g. the cluster dogfooding setup,
which also drives `/decay` on its own schedule).

See `adapters/opencode/README.md` to wire it into opencode.

Any provider implementing the OpenAI Chat Completions and Embeddings protocols
can be selected through environment variables; no provider-specific code is
required:

```bash
OPENAI_API_KEY=... \
OPENAI_BASE_URL=https://provider.example/v1 \
OPENAI_MODEL=provider-chat-model \
OPENAI_EMBED_MODEL=provider-embedding-model \
uv run python -m ontomem.service
```

Set `OPENAI_EMBED_BASE_URL` and `OPENAI_EMBED_API_KEY` only when embeddings use
a different server. A key cannot identify a provider's URL or model, so those
values must be configured explicitly for non-OpenAI endpoints.

## The contract (host-agnostic, plain data)

```
read(message)                 -> { memory_block, context_block, text, trace }
write(conversation)           -> write stats
retrieve_memory(name, depth)  -> { found, node, relationships, episodes }
decay()                       -> { edges_decayed, edges_dormant }
```

The engine never imports host types; all opencode coupling lives in the adapter.

## Status vs. spec build order (§10)

All six layers implemented and tested: Extractor, Store, Merge (2a+2b),
Retriever, Harness (read traces + journal), Decay schedule. Tunables in
`RetrievalConfig` are provisional (spec §9.3) — to be calibrated against ≥50 real
conversations via the harness logs.
