# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This repo has **two parts that serve one goal**: design and then build an *ontology-based long-term memory system for LLMs*.

| Directory | Role |
|-----------|------|
| `Documentation/` | **The design** — authoritative spec + essay for the ontology memory system. *What* to build. No code. |
| `engine/` | **The implementation** — the standalone Python memory engine (spec v0.2.1), built test-first. *The* memory system. |
| `opencode/` | **The host codebase** — a fork of [`sst/opencode`](https://github.com/sst/opencode), a production AI coding agent (Bun + Effect + SolidJS monorepo). *Where* the engine plugs in. |

The memory engine is **implemented and tested** under `engine/` (Phase 1 complete: all six build-order layers, 200+ tests — `engine/README.md`'s "171 tests" figure is stale, recount with the commands below rather than trusting either number). It is a standalone Python service; `opencode` integrates with it via a thin TS plugin adapter that already exists — the runtime copy lives at `opencode/.opencode/plugins/ontomem-plugin.ts` (see "How the memory system integrates" below for why it's there and not under `engine/`, and this is implemented, not just planned). The engine itself contains no opencode code — read `engine/README.md` for its layout and the design spec before changing it.

---

## Part 0 — The engine (`engine/`)

Standalone Python implementation of the spec, built test-first. Work from `engine/` with `uv`.

```bash
cd engine && uv venv && uv pip install -e ".[dev,llm]"
uv run pytest                 # hermetic suite (no network) — default loop guard
uv run pytest -m integration  # live Gemini tests (needs GEMINI_API_KEY in engine/.env)
GEMINI_API_KEY=... uv run python -m ontomem.service   # run the HTTP service, default http://127.0.0.1:8765

PYTHONPATH=src uv run python demo/server.py           # visual Read-pipeline demo at http://127.0.0.1:8800 (no key needed)
PYTHONPATH=src uv run python scripts/dogfood.py       # multi-turn live-Gemini harness run -> dogfood_report.json
```

- **Test discipline (non-negotiable):** every unit ships with its test; the full hermetic suite stays green at each step; deterministic logic is asserted against hand-checked values; LLM-dependent code is covered by opt-in `-m integration` tests (auto-skip without a key). The default `pytest` run is hermetic (network tests deselected via `addopts`).
- **Module map:** `model` → `store`/`decay`/`journal` → `extractor`(+`extractor_prompt`) → `merge`(2a)/`merge_llm`(2b) → `embeddings`/`retriever` → `consolidate` (Stage 0 + write orchestration, calling `supersede` for the relationship supersede/coexist/contradict judgment when a new edge shares (source, target) with an existing edge under a different relation — spec §9.2c) → `engine` (facade) → `service` (HTTP). `genai_keys` is an operational (non-spec) helper that round-robins multiple `GEMINI_API_KEYS` to work around free-tier per-key rate limits during dogfooding. All v0.2.1 corrections (edge-only strength, Stage 0, deferred reinforcement, JSON-snapshot storage) are implemented.
- **Secrets:** `engine/.env` (git-ignored) holds `GEMINI_API_KEY`; never hardcode it. Model: `gemini-3.1-flash-lite`; embeddings: `gemini-embedding-001`.
- **The `pyright`/editor "import could not be resolved" warnings are a harness venv-config artifact** — the passing pytest run is the ground truth that imports resolve.

## Part 1 — The design (`Documentation/`)

Files (note the version vs. filename mapping — it is counterintuitive):
- `Documentation/ontology_memory_architecture.md` — **the authoritative spec** (v0.2.1, "Full Architecture Specification"). Most complete; includes the full extractor prompt (Section 7), the detailed merge sub-stages, and the long-term roadmap. **Treat this as the source of truth.**
- `Documentation/ontology_memory_design.md` — earlier draft (v0.1). Superseded by v0.2; thinner, missing the merge sub-stages, asymmetric traversal, deep-retrieval tool, and extractor prompt. Keep only for history.
- `Documentation/*.pdf` — original PDF exports of the two `.md` files above (same content; read the `.md` versions).
- `Documentation/Ontology as Long Term memory.ipynb` — the originating essay (markdown only, no code). Explains the *why* and the human-brain analogy. Read for intent; read the v0.2 `.md` for the spec.

When sources disagree, `ontology_memory_architecture.md` (v0.2.1) wins.

### Core idea

Replace the standard "flat text summary appended to the system prompt" memory with a **knowledge graph** (ontology) that mirrors human associative memory: recall is selective, context-triggered, and proportional to connection strength rather than recency of an explicit write. The graph is the *navigation structure*; episodic content (emotion, nuance, exact wording) lives in **edge snippets**, not in the graph topology. Grounded in Complementary Learning Systems theory and Spreading Activation Theory (Collins & Loftus, 1975) — but it is a plain graph data structure, **not** a neural network.

This graph system is **Phase 1**. The long-term roadmap (v0.2 §2.2) is **Phase 2**: replace the external graph store with an encoder-only transformer whose weights *are* the memory. Phase 1 exists partly to produce `(graph, conversation)` pairs as training data for Phase 2 — so its inspectability is itself a deliverable, not just a convenience.

### Architecture at a glance

Three strictly separated processes that never run concurrently and share no mutable state (separation is a *correctness* requirement, not a performance one):

| Process | When | Role |
|---------|------|------|
| **Write** | After a conversation ends | Extract → merge → persist. Analogous to sleep-phase consolidation. |
| **Read** | Before every LLM response | Seed → spread activation → inject context. Analogous to associative recall. |
| **Decay** | Daily background job | Exponentially decays edge strength. Independent of Read/Write. |

#### Data model (v0.2 §3)
- **Nodes** = canonical named concepts, keyed `TYPE::canonical_name` (e.g. `PERSON::sarah`). Store only **time-invariant** facts in `properties` (DOB, nationality, field_of_work). Everything relational/time-variant (even current employment) lives on edges. Kinds: PERSON, ORG, PLACE, EVENT, THING, TOPIC, PREFERENCE, OTHER. **Nodes carry no strength, stability, or ttl** — a node never decays or expires; only edges do. Its persistence is implied by whether any live edge still references it (v0.2.1 correction).
- **Edges** = directional relationships, keyed `source::RELATION::target`. The **primary unit of memory**. `relation` = freeform `UPPER_SNAKE_CASE`, verb-first, ≤4 words (`WORKS_AT`, `COMPLAINED_ABOUT`). Carry `strength` (starts 100.0, decays/reinforces), `confidence`, `stability`, `cardinality` (one_to_one | one_to_many), and a **`snippet`** (2–6 sentence verbatim excerpt — the episodic content). Direction convention is **user-outward** (`User -WORKS_AT-> Org`, not `Org -EMPLOYS-> User`).
- **Episodes** = per-conversation records. Not directly retrievable; used for provenance and snippet ranking. Every node/edge from one conversation shares an episode id (`source_episode_ids`) — this **co-episode binding** is why there is no "conversation"/session/date node. The graph stays purely associative.

#### Write pipeline (Stage 0 context assembly, then 3 sequential stages; 2 can't start before 1, etc.)
0. **Context assembly — "Stage 0"** (v0.2.1 §4.1) — deterministic, no LLM. Assembles the extractor's `{existing_graph_context}` by **reusing the Read seeder over the whole conversation**: noun-extract all turns (both roles) → embed (same model as Read) → ANN → anchor nodes → **structural 2-hop expansion** (nodes + edges + relation labels). Deliberately **no** spreading-activation weighting and **no** node budget — Stage 0 wants breadth, not lean injection. The **2-hop radius is the explicit lever for the "connecting distant nodes" open problem (§9.1)**. Cold start (empty graph) → empty context, which the extractor prompt handles.
1. **Extraction** (v0.2 §4.2, §7) — one extractor LLM reads the full conversation JSONL (both roles) + a 2-hop subgraph context block; outputs entities, relationships, one episode. **Only user turns** produce entities by default; assistant turns are context-only, *except* the "acknowledged agent concepts" rule (explicit affirmation word + no hedge → extract at 0.70 confidence vs. default 0.85). Applies a **durability threshold**: extract only what a thoughtful person would still remember in a month. An empty entities/relationships array + episode-only output is **correct** for an undurable conversation. The 2-hop context drives importance calibration, canonicalisation, and relationship enrichment. **The full extractor prompt is the single most important artifact — see v0.2 §7 (`Documentation/ontology_memory_architecture.md` lines ~508–859) and reproduce it faithfully.**
2. **Merge / entity resolution** (v0.2 §4.3) — *hardest, highest-risk stage.* Two sub-stages:
   - **2a deterministic pre-processing** (no LLM): exact match → alias match → abbreviation dictionary (Sam/Samuel…) → fuzzy match (Levenshtein >80%) → token overlap. No candidates after all five ⇒ definitively new node.
   - **2b LLM disambiguation** (only if 2a produced candidates): receives entity + context + candidate 2-hop neighbourhoods. Confidence thresholds: **>0.90 auto-merge; 0.70–0.90 merge + journal flag; <0.70 new node.** Handles **relational singularity** (one_to_one relations like `HAS_SPOUSE`, `HAS_MOTHER`, `HAS_FATHER`, `BORN_IN`, `HAS_PRIMARY_RESIDENCE`) and network-topology matching.
   - **Merge asymmetry is the key invariant:** a false negative (missed merge) creates a duplicate — recoverable. A false positive (wrong merge) permanently fuses two distinct concepts — *unrecoverable*. **Default conservative: when in doubt, do not merge.** A weekly retroactive scan only *flags* suspected duplicates, never auto-merges.
3. **Write** (v0.2 §4.4) — fully deterministic, no LLM. Create/reinforce nodes and edges, attach snippets + co-episode bindings, write episodes with backlinks, append an audit journal entry. Existing nodes' strength is **not** reset on update — it keeps decaying naturally.

#### Read pipeline (v0.2 §5)
1. **Noun extraction** — tokenise the user message, keep **only nouns** (drop function words) so the search signal isn't diluted.
2. **Seed finding** — embed each noun independently, ANN/cosine search against precomputed node embeddings, return seeds above threshold with activation 1.0. Embeddings precomputed at write time; recomputed when name/aliases change. (Embedding similarity is used *only* for seeding; the graph is primary.)
3. **Multi-source BFS spreading activation** — `activation(target) = Σ activation(source) * (edge.strength/100) * hop_decay^hop` (`hop_decay = 0.6`, max depth 2). Contributions from multiple paths **sum** (multi-source convergence). **Asymmetric traversal: 70% of node budget forward, 30% reverse** — reverse is needed so e.g. seed `Walmart` surfaces `Bill -WORKS_AT-> Walmart`.
4. **Co-episode activation boost** — nodes sharing an episode id with a fired node get boosted, surfacing whole conversations as units.
5. **Threshold + node budget** — primary 0.30 (inject structural descriptor), secondary 0.60 (also inject best edge snippet). Node budget cap = 8 (ranked by activation; lowest silently dropped). The budget is the primary control against context flooding.
6. **Tiered context injection** — descriptors for weakly-fired nodes, full snippets for strongly-fired ones; snippet score = recency × episode importance × co-episode match. Output is a terse `[MEMORY]` (one relationship per line) + `[CONTEXT]` (snippets) block.
7. **Deep retrieval tool** — `retrieve_memory(node_name: str, depth: int = 1) -> dict`, invoked by the model on demand when injected context is insufficient. Keeps the default context lean while preserving full-depth access.

#### Decay (v0.2 §6)
`strength(t) = S0 * e^(-λt)` (t in days). λ by `stability`: immutable 0.0, stable 0.005, mutable 0.020, time_bound 0.050, ephemeral 0.200. Traversal reinforces via `S0_new = min(strength(t_now) + boost, 100.0)` (boost = 15.0) — Hebbian: repeated activation keeps an edge strong. **Reinforcement is deferred**: traversals are logged during Read and the boost is applied at conversation close (part of the Write phase) — Read never mutates strength inline and never deletes edges ("non-destructive" = no deletion + no inline mutation). Daily decay job recomputes all **edges** (nodes never decay). Edges below the dormancy threshold (strength < 2.0) are flagged, **never deleted** (nothing is truly forgotten — only weakened to near-inaccessibility).

### Build order (v0.2 §10 — build/test each layer in isolation before the next)
1. **Extractor** — prompt runner, JSON parsing + partial-JSON salvage, output normalisation. Test on ~10 synthetic conversations with ground-truth entity/relationship sets.
2. **Store** — graph data model, exponential decay, journal, snapshot save/load. Verify decay rates per stability class.
3. **Merge** — all six rules in order (exact, alias, abbreviation, fuzzy, token overlap, LLM disambiguation) incl. relational singularity. **Highest-risk — allocate extra test time.**
4. **Retriever** — noun extraction, per-noun embedding, ANN, asymmetric BFS + co-episode boost, thresholds, tiered formatting. Test against hand-crafted graphs with known correct retrieval.
5. **Harness** — wrap full pipeline with the observability log; run write-then-read on ~10 real conversations and inspect every run manually before any automated eval.
6. **Decay schedule** — wire the background job; verify strength evolution over a simulated 30-day period; tune dormancy threshold.

### Tunable parameters are provisional

`node_budget=8`, `primary_threshold=0.30`, `secondary_threshold=0.60`, `hop_decay=0.6`, `reinforce_boost=15.0`, `max_depth=2`, dormancy `strength<2.0`, fuzzy match >80%, merge confidence bands. These are **initial estimates** (v0.2 §9.3) — treat as provisional until ≥50 real conversations have been processed via the harness. Don't hardcode them as if final; make them configurable.

### Known constraints & open problems (don't "solve" silently — v0.2 §9)

- **Connecting distant nodes during write** (high impact, unsolved): a bridging edge is missed if both entities aren't in the 2-hop context. Deferred pending empirical data; may warrant a larger hop radius.
- **Snippet staleness** (medium): edge snippets capture the moment a relationship was established and don't auto-update when it evolves. Current policy overwrites only on higher-confidence evidence.
- **Context budget tuning** (medium): see provisional parameters above; the harness log is the tuning instrument.
- **Embedding drift** (low prob, high severity): store the embedding model id alongside each embedding and detect mismatch at query time; re-embed on model change. Implement the id check before any production deployment.
- **Silent retrieval failure** is the primary operational risk — the model answers confidently whether or not correct context was injected. The **observability harness** (v0.2 §8) exists specifically to make Read/Write behaviour inspectable; preserve and extend it, don't bypass it.

### Conventions implied by the spec

- Keep key/label formats exact and stable: node `TYPE::canonical_name`, edge `source::RELATION::target`, relations `UPPER_SNAKE_CASE` verb-first ≤4 words, edge direction user-outward.
- Canonicalisation: most specific name available ("my boss Sarah" → `Sarah`, role captured by `HAS_BOSS`); resolve pronouns/possessives before extracting; partially-known names → reduce confidence to ≤0.60.
- Timestamps are ISO 8601 UTC. The extractor prompt must receive current UTC time so TTLs are assigned correctly. `ttl_days` is null unless `time_bound`/`ephemeral`.
- No chunking/windowing of conversations — process the whole conversation in one extraction call.
- Storage (v0.2.1 §3.5, decided): **no graph database (no Neo4j/Cypher).** Runtime = in-memory object graph of Node/Edge **dataclasses with real pointer references** (`node.outgoing`/`incoming` are `Edge` objects; `Edge` points at source/target `Node`s) for pointer-style BFS. Persistence = **human-readable JSON snapshot**, written **atomically** (temp file + `os.replace()`). On-disk form is **flat/key-based** (edges reference endpoints by `source_key`/`target_key`); rehydrate the pointer graph on load — never `json.dump` the live cyclic object graph. **Embeddings live in a sidecar** (e.g. numpy `.npz` keyed by node id, with the embedding-model id), NOT inline in the graph JSON. Keep a rolling last-N snapshots + the append-only journal for recovery. numpy brute-force cosine at this scale; FAISS only if it ever outgrows that.
- Extractor returns strict JSON (no prose, no markdown fences) — implement a salvage path for partial JSON.

> Note: the design spec sketches a Python-flavoured implementation (numpy, FAISS, a `retrieve_memory` tool). The host codebase (`opencode/`) is **TypeScript/Bun**. When implementing inside opencode, translate the spec's intent into the host stack rather than copying its language; the algorithms and invariants above are the contract, not the pseudocode.

---

## Part 2 — The host codebase (`opencode/`)

A fork of `sst/opencode`: an AI coding agent. Bun monorepo using **Effect** (effect-ts), **Hono** (server/API), **Drizzle** (SQLite), and **SolidJS** (web/desktop/TUI). `opencode/` is its own git repo — work inside it with its own conventions, separate from the parent `Memory` repo.

### Authoritative guides inside `opencode/` (read these before editing code there)
- **`opencode/AGENTS.md`** — the binding style guide and workflow rules. Non-negotiable highlights:
  - Default branch is **`dev`** (local `main` may not exist; diff against `dev`/`origin/dev`). Branch names ≤3 hyphen-separated words, **no** `feat/` prefixes.
  - Conventional commits/PR titles: `type(scope): summary` (types: feat, fix, docs, chore, refactor, test).
  - Style: keep logic inline unless genuinely reusable; avoid `try`/`catch`, `any`, `else`, and reassignment (prefer early returns + ternaries + `const`); **no** aliased imports, **no** star imports; use Bun APIs (`Bun.file()`); prefer functional array methods; Drizzle schemas use snake_case field names. Prefer Effect schema helpers (`Schema.UnknownFromJsonString`, `Schema.decodeUnknownOption`) over manual `JSON.parse`.
- **`opencode/CONTEXT.md`** — the **System Context / Session Runtime** domain model and glossary. This is the subsystem most relevant to the memory project: it defines how *Context Sources* are composed into a *System Context*, admitted at *Safe Provider-Turn Boundaries*, and revised via *Mid-Conversation System Messages* across *Context Epochs*. The memory **Read** pipeline would most naturally surface as a new **Context Source**; consolidation (**Write**) would hook a session-end boundary. Read this before designing the integration.
- **`opencode/CONTRIBUTING.md`** — contribution workflow.
- **`opencode/specs/`** — design specs, including `specs/v2/` (session, tools, config, provider) for the V2 session core described in AGENTS.md's "V2 Session Core" section.

### Commands (run inside `opencode/`)
Bun is the package manager and runtime (`bun@1.3.x`). From the **repo root** (`opencode/`):
```bash
bun install                 # install all workspace deps
bun run dev                 # run the CLI agent (packages/opencode)
bun run dev:web             # web app (packages/app)
bun run dev:desktop         # desktop app
bun lint                    # oxlint across the repo
bun typecheck               # turbo-driven typecheck across all packages
```
**Critical gotchas (enforced):**
- **Never run tests from the repo root** — `bun test` at root is rigged to fail (`do-not-run-tests-from-root` guard). Run tests from a package dir, e.g. `cd packages/opencode && bun test`. Single test: `bun test --timeout 30000 path/to/file.test.ts` or `bun test -t "test name"`.
- **Typecheck via the package script, not raw `tsc`** — `bun typecheck` (root) uses turbo; per package it runs `tsgo --noEmit` (TypeScript native preview). Don't call `tsc` directly.
- After changing the API surface, **regenerate the JS SDK**: `./packages/sdk/js/script/build.ts`.

### Package map (`opencode/packages/`)
`core` (domain logic, Drizzle schema, sessions, providers), `opencode` (the CLI agent / main entrypoint, `src/index.ts`), `server` & `http-recorder` (Hono API), `llm` (provider/protocol adapters), `tui` (terminal UI, OpenTUI/SolidJS), `app`/`web`/`desktop`/`ui` (SolidJS frontends), `sdk` (generated JS SDK), `plugin` (plugin API), `console`/`stats` (dashboards), `slack`, `github`, `infra`/`function`/`containers` (SST/AWS deploy). Build/test each package from its own directory.

### How the memory system integrates with opencode (decided — v0.2.1 §11)

The memory **engine** stays a **standalone Python service** (graph store, Read/Write/Decay, extractor — keeps §3.5 intact and reusable). opencode integrates via a **thin TS plugin adapter, NOT an MCP server** (MCP is pull-based and cannot push system-prompt context before a turn). The adapter wires four plugin hooks (`packages/plugin/src/index.ts:222`):

| Memory process | Hook | Status |
|---|---|---|
| Read — capture user msg, run retrieval | `chat.message` (`:234`) | stable |
| Read — inject `[MEMORY]`/`[CONTEXT]` block | `experimental.chat.system.transform` (`:291`), output `system: string[]` | **experimental** |
| `retrieve_memory(node_name, depth)` deep-retrieval tool | `tool` (`:226`) — native plugin tool | stable |
| Write trigger (once per session, at app exit → extract→merge→write) | `dispose`, detached child process | stable |
| Decay | external cron against the engine | n/a |

**Invariant — the boundary is plain data.** The engine exposes `read(conversation_so_far)`, `write(transcript)`, `retrieve_memory(...)`, `decay()` taking/returning only strings/JSON; it never imports opencode types (`Message`/`Part`/`Model`). This is what makes the experimental injection hook a non-issue: if it changes or graduates to a Context Source (opencode's intended long-term home for this — see `opencode/CONTEXT.md`), only the adapter changes; the engine is untouched. Fallback injection seams exist (`experimental.chat.messages.transform`, appending a part via `chat.message`), so injection isn't a single point of failure.

**Write trigger, and why it's `dispose` not `chat.message`/`session.idle`:** the spec's Write process fires once, "after a conversation ends" — not after every turn. An earlier version fired on `session.idle` (which turned out to fire after *every* turn, not just at session end) — that re-extracts the whole growing transcript from scratch each time (wasteful; everything relevant to the *current* conversation is already in its live context window — the graph only pays off in a *later* conversation) and creates one new episode per turn instead of one per conversation, fragmenting co-episode binding. `dispose` is a plugin-level Effect finalizer that opencode *awaits* before actually exiting, so it can't just synchronously call the real write (an LLM extraction call, seconds-long) without visibly hanging the user's `/exit`. The adapter's `dispose` hook instead does the cheap part inline (reading the already-local transcript via the still-live client) and hands the slow extract/merge/persist call to a **detached child process** (`curl`, spawned with `detached: true` + `.unref()` so it outlives opencode's own process) before returning immediately — opencode exits right away, and consolidation genuinely finishes in the background after the terminal is already back at the prompt.

This adapter is **already implemented**. Its runtime location is **`opencode/.opencode/plugins/ontomem-plugin.ts`**, registered in `opencode/.opencode/opencode.jsonc` (`"plugin": ["./plugins/ontomem-plugin.ts"]`) — **not** under `engine/adapters/opencode/` (that directory now holds only a README pointer explaining why). Reason, discovered the hard way: Bun/Node resolves a bare specifier like `@opencode-ai/plugin` by walking up from the *importing file's own directory* looking for `node_modules`; `engine/adapters/opencode/` is a sibling of `opencode/`, outside its workspace, so that resolution always fails there (`Cannot find module '@opencode-ai/plugin'`) — and opencode's plugin loader reports import failures via an in-chat `Session.Event.Error`, not the file log or `bun run dev`'s stdout, so a plugin failing to load this way is silent and looks identical to "loaded fine, nothing to recall." Verify a plugin actually loads with `bun -e "await import('<path>')"` from its real location, not just by checking logs. Related, separately-discovered gotcha: the default export shape for a **server** plugin must be `{ id?, server: Plugin }`, not a bare exported function (which fails `isRecord()` in the loader's validation and is dropped with no error either). This differs from a **TUI** plugin's shape (`{ id?, tui: TuiPlugin }`, registered separately in `opencode/.opencode/tui.json`, e.g. `opencode/.opencode/plugins/memory-viewer.tsx` — the "Open Memory Graph" command palette entry) — the two plugin systems are independent, with different config files, different validation, and files under `.opencode/plugins/*.ts`/`*.js` (not `.tsx`) get auto-discovered as server plugins regardless of `tui.json`, so a TUI-only plugin must use a `.tsx` extension to dodge that glob.

To run it end-to-end: start the engine service (`GEMINI_API_KEY=... ONTOMEM_DIR=~/.ontomem uv run python -m ontomem.service`, listens on `127.0.0.1:8765` by default, override with `ONTOMEM_HOST`/`ONTOMEM_PORT`/`ONTOMEM_URL`) — `scripts/openagent.sh` at the repo root does this automatically. The write hook fetches the full transcript via the opencode SDK (`client.session.messages`), so both roles reach the extractor. `bun typecheck` on this file needs the opencode workspace's `@opencode-ai/plugin` installed — it isn't checked from the Python package (and, per the above, that same resolution requirement is why the file must live inside `opencode/` at runtime, not just at typecheck time). Full detail: `engine/adapters/opencode/README.md`.
