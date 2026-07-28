# Memory-graph live demo

A visual, zero-setup demonstration of the Read pipeline: a pre-seeded knowledge
graph, animated spreading-activation, and a mock "context window" that fills with
`[MEMORY]` lines and `[CONTEXT]` snippets **as each edge is crossed**.

## Run it

```bash
cd engine
PYTHONPATH=src uv run python demo/server.py
# open http://127.0.0.1:8800
```

No API key needed — it uses the offline `HashingEmbedder` (lexical, deterministic).
To seed with real semantic embeddings instead:

```bash
ONTOMEM_DEMO_GEMINI=1 PYTHONPATH=src uv run python demo/server.py   # needs GEMINI_API_KEY(S) in engine/.env
```

## What you're looking at

- **The graph** (left): the pre-seeded "Marcus" world — three loosely linked
  clusters (work · Roman history · personal). `Rome` is a deliberate *bridge*
  (`Marcus -PLANS_TO_VISIT-> Rome -PART_OF-> Roman Empire`) so a trip query
  spreads two hops into the history thread.
- **Run a query**: seeds light up gold, then activation spreads edge-by-edge.
  - **red-bordered** nodes have fired (activation ≥ the primary threshold);
  - **green** edges were injected into context; **amber** edges were traversed
    but filtered out (below threshold / budget) — the selectivity story made visible.
- **The context window** (right): exactly what would be handed to the LLM —
  each crossed edge that clears the budget drops its relationship line and (if
  strong enough) its verbatim snippet, in real time.
- **Step** advances one event at a time; the speed slider controls playback.

Try the presets, e.g. *"Tell me about my trip to Rome"* (watch it cross the
bridge), *"Who is my manager?"* (relation-aware seeding — "manager" has no node,
it matches the `HAS_MANAGER` relation), or something off-topic (the graph stays
dark — nothing to recall, which is the point).

## Files

| file | role |
|------|------|
| `demo_data.py` | the hand-authored pre-seeded graph (nodes, edges, snippets, episodes) |
| `demo_trace.py` | runs the **real** retriever and emits an ordered animation trace + the authoritative injected context |
| `server.py` | stdlib HTTP server: `/` (UI), `/graph`, `/trace?q=…` |
| `index.html` | the visualisation (vis-network via CDN) |

The animation is driven by the real pipeline: `spreading_activation` accepts an
opt-in `events` list (behaviour-preserving) that records every crossing, so what
you see is the actual traversal, not a re-creation.
