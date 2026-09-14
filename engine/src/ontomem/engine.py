"""Engine facade — the host-agnostic service contract (spec v0.2.1 §11).

Exposes read / write / retrieve_memory / decay taking and returning plain data
(strings, JSON-able dicts) only. Never imports host types. This is the single
seam an adapter (e.g. the opencode plugin) talks to.

Persistence layout under base_dir:
  graph.json       atomic JSON snapshot (the object graph, flat/key-based)
  embeddings.npz   sidecar vectors keyed by node id, tagged with model id
  journal.jsonl    append-only write audit
  reinforce.jsonl  pending Read traversals, drained at write (deferred boost)
  read_trace.jsonl observability harness log (one line per Read)

The two LLM-dependent steps are injectable (extract_fn, disambiguate_fn) so the
whole orchestration can be driven offline in tests.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .consolidate import apply_write, assemble_context, plan_merges, plan_supersessions
from .decay import DORMANCY_THRESHOLD, decayed_strength, is_dormant
from .embeddings import EmbeddingIndex, HashingEmbedder, _node_text, relation_phrase
from .extractor import ExtractionResult, extract as _llm_extract, to_jsonl, build_prompt
from .journal import append_event, read_events
from .merge_llm import disambiguate as _llm_disambiguate
from .model import Edge, Episode, Node, canonicalize, utcnow_iso
from .retriever import RELATION_KEY_PREFIX, RetrievalConfig, read as _retrieve
from .supersede import decide_supersession as _llm_supersede
from .store import Store

# Safety cap on the agentic extraction loop (see Engine._write_agentic): a
# reasoning-capable backend that never calls finish_extraction would otherwise
# spin forever. Chosen generously relative to any real conversation's entity
# + relationship count.
MAX_AGENTIC_TOOL_CALLS = 80

# Self-gating interval for Engine.decay() (see its docstring): matches the
# spec's "daily background job" cadence (v0.2 §6), so a caller that fires
# decay on every process start (no cron) still only does real work ~once/day.
MIN_DECAY_INTERVAL = timedelta(hours=24)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _clamp01(value, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _agentic_null(value):
    """Coerce a nullable agentic tool-call arg to real None. Observed against
    the live vLLM/gemma4 backend: tool_choice="auto" does NOT enforce the tool
    schema's declared types/enums (they're descriptive text to the model, not
    constrained decoding) -- so a field typed ["integer", "null"] sometimes
    comes back as the literal JSON string "null" instead of an actual null,
    which would otherwise slip past a bare `if value:` check and get treated
    as a real value (e.g. "null" used as a candidate_merge_key)."""
    if isinstance(value, str) and value.strip().lower() in ("null", "none", ""):
        return None
    return value


def _agentic_optional_int(value) -> int | None:
    """Same backend quirk as _agentic_null, plus: an int-typed field can come
    back as a numeric string (e.g. ttl_days="30") since nothing enforces the
    schema's declared type under tool_choice="auto"."""
    value = _agentic_null(value)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class Engine:
    def __init__(
        self,
        base_dir,
        embedder=None,
        config: RetrievalConfig | None = None,
        *,
        extract_fn=None,
        disambiguate_fn=None,
        supersede_fn=None,
        generate_fn=None,
        chat_fn=None,
        model: str = "gemini-3.1-flash-lite",
        api_key: str | None = None,
        extraction_prompt_template: str | None = None,
        user_only_extraction: bool = False,
        agentic_extraction: bool = False,
    ) -> None:
        self.dir = Path(base_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or HashingEmbedder()
        self.config = config or RetrievalConfig()
        self.model = model
        self.api_key = api_key
        self.generate_fn = generate_fn
        # Only used by agentic extraction (chat_with_tools-shaped: takes a
        # growing messages list + tools, returns one assistant message dict).
        # Separate from generate_fn because the two have incompatible
        # signatures -- one is single-shot-prompt, the other is multi-turn.
        self.chat_fn = chat_fn
        # None keeps extractor.extract()'s own default (EXTRACTOR_SYSTEM_PROMPT).
        # Set to EXTRACTOR_REASONING_PROMPT for a tool-calling backend.
        self.extraction_prompt_template = extraction_prompt_template
        # Experimental: see write() for what this changes and why.
        self.user_only_extraction = user_only_extraction
        # Experimental: see _write_agentic for what this changes and why.
        self.agentic_extraction = agentic_extraction
        # service.py's ThreadingHTTPServer handles each HTTP request on its own
        # thread with no synchronization otherwise -- a client-side timeout on
        # one /write call does not stop the server from continuing to process
        # it, so a naive retry can end up running concurrently with the
        # original attempt against this same in-memory store. Confirmed live:
        # two overlapping write() calls for the same conversation both reached
        # apply_write()/_persist() around the same time. This lock forces
        # writes to run one at a time; reads are unaffected.
        self._write_lock = threading.Lock()
        self._extract_fn = extract_fn or self._default_extract
        self._disambiguate_fn = disambiguate_fn or self._default_disambiguate
        self._supersede_fn = supersede_fn or self._default_supersede
        self.store = self._load_store()
        self.index = self._load_or_build_index()
        self.last_read: dict | None = None

    # --- paths -----------------------------------------------------------------

    @property
    def _snapshot_path(self):
        return self.dir / "graph.json"

    @property
    def _index_path(self):
        return self.dir / "embeddings.npz"

    @property
    def _journal_path(self):
        return self.dir / "journal.jsonl"

    @property
    def _reinforce_path(self):
        return self.dir / "reinforce.jsonl"

    @property
    def _trace_path(self):
        return self.dir / "read_trace.jsonl"

    @property
    def _decay_state_path(self):
        return self.dir / "decay_state.json"

    # --- persistence -----------------------------------------------------------

    def _load_store(self) -> Store:
        if self._snapshot_path.exists():
            return Store.load(self._snapshot_path)
        return Store()

    def _index_inputs(self) -> dict[str, str]:
        """The combined key->text map embedded in one index: every node (by name
        + aliases) AND every distinct relation label (as a phrase, keyed
        REL::<RELATION>). Sharing one index gives relation-aware seeding a single
        well-estimated mean for anisotropy centring (spec §5.2)."""
        items = {n.key: _node_text(n) for n in self.store.nodes.values()}
        for relation in {e.relation for e in self.store.edges.values()}:
            items[f"{RELATION_KEY_PREFIX}{relation}"] = relation_phrase(relation)
        return items

    def _load_or_build_index(self) -> EmbeddingIndex:
        inputs = self._index_inputs()
        if self._index_path.exists():
            index = EmbeddingIndex.load(self._index_path)
            if index.model_id == self.embedder.model_id and set(index.keys) == set(inputs):
                return index
        return self._rebuild_index()

    def _rebuild_index(self) -> EmbeddingIndex:
        inputs = self._index_inputs()
        index = EmbeddingIndex.build_keyed(list(inputs), list(inputs.values()), self.embedder)
        index.save(self._index_path)
        return index

    def _persist(self) -> None:
        self.store.save(self._snapshot_path)
        self.index.save(self._index_path)

    # --- default LLM steps -----------------------------------------------------

    def _default_extract(self, conversation, existing_graph_context):
        existing_keys: dict[str, str] = {}
        for node in self.store.nodes.values():
            existing_keys[canonicalize(node.name)] = node.key
            for alias in node.aliases:
                existing_keys[canonicalize(alias)] = node.key
        return _llm_extract(
            conversation, existing_graph_context,
            existing_keys=existing_keys, model=self.model, api_key=self.api_key,
            generate_fn=self.generate_fn, prompt_template=self.extraction_prompt_template,
        )

    def _default_disambiguate(self, entity, resolution, store, ctx):
        return _llm_disambiguate(
            entity, resolution, store, ctx, model=self.model, api_key=self.api_key,
            generate_fn=self.generate_fn,
        )

    def _default_supersede(self, old_edge, new_edge, store, ctx):
        src = store.get_node(old_edge.source_key)
        old_tgt = store.get_node(old_edge.target_key)
        new_tgt = store.get_node(new_edge.target_key)
        return _llm_supersede(
            old_edge, new_edge,
            src.name if src else old_edge.source_key,
            old_tgt.name if old_tgt else old_edge.target_key,
            new_tgt.name if new_tgt else new_edge.target_key,
            ctx, model=self.model, api_key=self.api_key, generate_fn=self.generate_fn,
        )

    # --- READ ------------------------------------------------------------------

    def read(self, message: str) -> dict:
        """Run the Read pipeline. Non-destructive: logs traversed edges for
        deferred reinforcement, never mutates the graph."""
        result = _retrieve(message, self.store, self.index, self.embedder, self.config)
        for key in result.traversed_edge_keys:
            append_event(self._reinforce_path, {"edge": key})
        trace = {
            "event": "read",
            "message_preview": message[:200],
            "seeds": result.seeds,
            "selected": result.selected,
            "fired": {k: round(a.score, 4) for k, a in result.activations.items()},
            "context_tokens_estimate": len(result.text.split()),
        }
        append_event(self._trace_path, trace)
        response = {
            "memory_block": result.memory_block,
            "context_block": result.context_block,
            "text": result.text,
            "injections": result.injections,
            "trace": trace,
        }
        # Cached for GET /last_read -- lets an external viewer show what got
        # injected for the actual live conversation, not just a manual query
        # typed into the viewer itself.
        self.last_read = {"message": message, **response}
        return response

    # --- WRITE -----------------------------------------------------------------

    def write(self, conversation: list[dict]) -> dict:
        """Stage 0 -> extract -> merge plan -> deterministic write. Drains the
        pending Read-reinforcement log and rebuilds the embedding index.

        If `user_only_extraction` is set, assistant turns are stripped before
        extraction (Stage 0 context assembly still sees the full conversation,
        since anchor-noun extraction is meant to read both roles) -- an
        experiment to test whether a reasoning-capable backend's runaway
        "thinking" on long conversations is being driven by having to read
        back its own (often much longer) prior replies, which establish
        nothing about the user, rather than by conversation length itself."""
        if self.agentic_extraction:
            return self._write_agentic(conversation)
        with self._write_lock:
            extract_conversation = (
                [t for t in conversation if t.get("role") == "user"]
                if self.user_only_extraction
                else conversation
            )
            context = assemble_context(conversation, self.store, self.index, self.embedder, self.config)
            extraction = self._extract_fn(extract_conversation, context)
            conversation_context = " ".join(t.get("text", "") for t in conversation)[:2000]
            plan = plan_merges(extraction, self.store, self._disambiguate_fn, conversation_context)
            supersessions = plan_supersessions(extraction, plan, self.store, self._supersede_fn, conversation_context)
            reinforcement_keys = [e["edge"] for e in read_events(self._reinforce_path)]

            result = apply_write(
                extraction, plan, self.store,
                reinforcement_edge_keys=reinforcement_keys,
                supersessions=supersessions,
                config=self.config,
                journal_path=self._journal_path,
            )
            self._drain_reinforcement_log()
            # incremental: embed only new nodes / new relation labels
            self.index.sync(self._index_inputs(), self.embedder)
            self.index.save(self._index_path)
            self._persist()
        return {
            "episode_id": result.episode_id,
            "nodes_created": result.nodes_created,
            "nodes_merged": result.nodes_merged,
            "edges_created": result.edges_created,
            "edges_reinforced": result.edges_reinforced,
            "edges_dropped": result.edges_dropped,
            "edges_superseded": result.edges_superseded,
            "reinforced_from_read": result.reinforced_from_read,
            "flagged": result.flagged,
            "warnings": result.warnings,
        }

    def _drain_reinforcement_log(self) -> None:
        self._reinforce_path.unlink(missing_ok=True)

    # --- AGENTIC (per-item) WRITE ------------------------------------------------

    def _write_agentic(self, conversation: list[dict]) -> dict:
        """Agentic (per-item) extraction: the model calls add_entity /
        add_relationship / finish_extraction one at a time, each committed to
        the real store immediately via the same deterministic apply_write()
        the batch path uses, instead of returning one big structured blob
        that only becomes parseable once the whole response finishes. See
        extractor_prompt_agentic.py for the rationale: the batch path's
        internal reasoning WAS sequential (confirmed by watching its raw
        trace), but that order was invisible to the graph, since a single
        tool call's JSON arguments aren't valid/parseable until generation
        completes. This mode gives the model a genuine read-then-append loop
        within one conversation, the same shape the system already runs
        BETWEEN conversations via Stage 0 context."""
        if self.chat_fn is None:
            raise RuntimeError("agentic_extraction=True requires chat_fn (a chat_with_tools-compatible callable)")

        from .extraction_schema import AGENTIC_TOOLS
        from .extractor_prompt_agentic import EXTRACTOR_AGENTIC_PROMPT

        with self._write_lock:
            extract_conversation = (
                [t for t in conversation if t.get("role") == "user"]
                if self.user_only_extraction
                else conversation
            )
            context = assemble_context(conversation, self.store, self.index, self.embedder, self.config)
            convo_jsonl = to_jsonl(extract_conversation)
            prompt = build_prompt(convo_jsonl, context, utcnow_iso(), template=EXTRACTOR_AGENTIC_PROMPT)

            # One shared episode for the whole conversation, persisted now so
            # every incremental apply_write() call below can bind
            # source_episode_ids to it. finish_extraction fills in the real
            # summary/importance/tags by mutating this SAME object (stored by
            # reference in store.episodes, so no re-add/update call needed).
            episode = Episode.create("(pending -- finish_extraction not reached)", 0.3)
            self.store.add_episode(episode)

            messages: list[dict] = [{"role": "user", "content": prompt}]
            totals = dict(
                nodes_created=0, nodes_merged=0, edges_created=0,
                edges_reinforced=0, edges_dropped=0, edges_superseded=0,
                reinforced_from_read=0,
            )
            warnings: list[str] = []
            flagged: list[dict] = []
            tool_call_log: list[dict] = []
            touched_entity_keys: list[str] = []
            reinforcement_keys = [e["edge"] for e in read_events(self._reinforce_path)]
            finished = False

            for _ in range(MAX_AGENTIC_TOOL_CALLS):
                response = self.chat_fn(messages, model=self.model, tools=AGENTIC_TOOLS)
                tool_calls = response.get("tool_calls") or []
                if not tool_calls:
                    warnings.append("agentic extraction ended without finish_extraction (model stopped calling tools)")
                    break

                messages.append({
                    "role": "assistant",
                    "content": response.get("content") or "",
                    "tool_calls": tool_calls,
                })

                for call in tool_calls:
                    name = call.get("function", {}).get("name", "")
                    raw_args = call.get("function", {}).get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except json.JSONDecodeError as exc:
                        result_text = f"error: arguments were not valid JSON ({exc})"
                        tool_call_log.append({"tool": name, "raw_arguments": raw_args, "result": result_text})
                        messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result_text})
                        warnings.append(f"agentic {name} call had unparseable arguments")
                        continue

                    if name == "add_entity":
                        result_text = self._agentic_add_entity(args, episode, totals, warnings, touched_entity_keys)
                    elif name == "add_relationship":
                        result_text = self._agentic_add_relationship(args, episode, totals, warnings, flagged)
                    elif name == "finish_extraction":
                        episode.summary = str(args.get("summary") or "").strip() or episode.summary
                        episode.importance = _clamp01(args.get("importance"), episode.importance)
                        episode.tags = [str(t).strip() for t in (args.get("tags") or []) if str(t).strip()][:5]
                        result_text = "episode finalized; stop calling tools now"
                        finished = True
                    else:
                        result_text = f"error: unknown tool {name!r}"
                        warnings.append(f"agentic loop received unknown tool call {name!r}")

                    tool_call_log.append({"tool": name, "args": args, "result": result_text})
                    messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result_text})

                if finished:
                    break
            else:
                warnings.append(f"agentic extraction hit the {MAX_AGENTIC_TOOL_CALLS}-call safety cap without finish_extraction")

            if episode.summary.startswith("(pending"):
                episode.summary = "(extraction ended without a summary -- see warnings)"

            # Orphan check deferred to here (see apply_write's check_orphans):
            # a node added via add_entity is often connected by a LATER,
            # separate add_relationship call, so checking connectivity after
            # each individual call would flag every entity as orphaned the
            # moment it's created. Now that the whole session's calls have
            # all landed, this is the same "unreachable by traversal" check
            # _flag_orphans does, just run once over everything touched.
            seen_orphan_check: set[str] = set()
            for key in touched_entity_keys:
                if key in seen_orphan_check:
                    continue
                seen_orphan_check.add(key)
                node = self.store.get_node(key)
                if node is not None and not node.outgoing and not node.incoming:
                    warnings.append(f"orphan entity {key}: extracted but participates in no relationship (unreachable)")

            for key in reinforcement_keys:
                if key in self.store.edges:
                    self.store.reinforce_edge(key, boost=self.config.reinforce_boost)
                    totals["reinforced_from_read"] += 1

            self._drain_reinforcement_log()
            self.index.sync(self._index_inputs(), self.embedder)
            self.index.save(self._index_path)
            self._persist()

        return {
            "episode_id": episode.id,
            **totals,
            "flagged": flagged,
            "warnings": warnings,
            "agentic_tool_calls": tool_call_log,
        }

    def _resolve_agentic_entity_key(self, name: str) -> str | None:
        canon = canonicalize(name)
        for node in self.store.nodes.values():
            if canonicalize(node.name) == canon or any(canonicalize(a) == canon for a in node.aliases):
                return node.key
        return None

    def _agentic_synthesize_endpoint(self, name: str, warnings: list[str]) -> str:
        """Mirrors extractor._synthesize_endpoint's anchored-auto-create
        policy: recover the edge (and the concept) with a provisional,
        low-confidence node rather than dropping the relationship outright."""
        node = Node.create(kind="OTHER", name=name, confidence=0.5, properties={"provisional_from_relationship": True})
        self.store.add_node(node)
        warnings.append(
            f"agentic: endpoint {name!r} referenced in add_relationship but never added via "
            f"add_entity; synthesised provisional node {node.key}"
        )
        return node.key

    def _agentic_add_entity(
        self, args: dict, episode: Episode, totals: dict, warnings: list[str], touched_entity_keys: list[str],
    ) -> str:
        name = str(args.get("text", "")).strip()
        kind = str(args.get("type", "")).strip().upper()
        if not name or not kind:
            warnings.append("agentic add_entity call missing text or type; skipped")
            return "error: 'text' and 'type' are required"
        try:
            node = Node.create(
                kind=kind,
                name=name,
                confidence=_clamp01(args.get("confidence"), 0.85),
                properties=args.get("properties") or {},
                aliases=[str(a).strip() for a in (args.get("aliases") or []) if str(a).strip()],
            )
        except ValueError as exc:
            warnings.append(f"agentic add_entity({name!r}) rejected: {exc}")
            return f"error: {exc}"

        extraction = ExtractionResult(episode=episode, nodes=[node], edges=[])
        candidate = _agentic_null(args.get("candidate_merge_key"))
        if candidate:
            extraction.candidate_merge_keys[node.key] = str(candidate)
        plan = plan_merges(extraction, self.store, self._disambiguate_fn, name)
        result = apply_write(
            extraction, plan, self.store,
            config=self.config, journal_path=self._journal_path, add_episode=False,
            check_orphans=False,
        )
        totals["nodes_created"] += result.nodes_created
        totals["nodes_merged"] += result.nodes_merged
        warnings.extend(result.warnings)
        final_key = plan.remap.get(node.key, node.key)
        touched_entity_keys.append(final_key)
        if result.nodes_created:
            return f"created new entity {final_key}"
        return f"merged into existing entity {final_key} (already in the graph)"

    def _agentic_add_relationship(
        self, args: dict, episode: Episode, totals: dict, warnings: list[str], flagged: list[dict],
    ) -> str:
        source_name = str(args.get("source", "")).strip()
        target_name = str(args.get("target", "")).strip()
        relation = str(args.get("relation", "")).strip().upper().replace(" ", "_")
        if not source_name or not target_name or not relation:
            warnings.append("agentic add_relationship call missing source/target/relation; skipped")
            return "error: 'source', 'target', and 'relation' are required"

        src_key = self._resolve_agentic_entity_key(source_name)
        tgt_key = self._resolve_agentic_entity_key(target_name)
        if not src_key and not tgt_key:
            warnings.append(
                f"agentic add_relationship({source_name!r}-{relation}->{target_name!r}) dropped: neither endpoint exists yet"
            )
            return f"error: neither {source_name!r} nor {target_name!r} exist yet -- call add_entity for at least one first"
        if not src_key:
            src_key = self._agentic_synthesize_endpoint(source_name, warnings)
        if not tgt_key:
            tgt_key = self._agentic_synthesize_endpoint(target_name, warnings)

        try:
            edge = Edge.create(
                source_key=src_key, relation=relation, target_key=tgt_key,
                snippet=str(args.get("snippet", "")), evidence=str(args.get("evidence", "")),
                confidence=_clamp01(args.get("confidence"), 0.85),
                stability=str(args.get("stability") or "stable"),
                ttl_days=_agentic_optional_int(args.get("ttl_days")),
                cardinality=str(args.get("cardinality") or "one_to_many"),
                properties=args.get("properties") or {},
            )
        except ValueError as exc:
            warnings.append(f"agentic add_relationship rejected: {exc}")
            return f"error: {exc}"

        extraction = ExtractionResult(episode=episode, nodes=[], edges=[edge])
        ctx = f"{source_name} {relation} {target_name}"
        plan = plan_merges(extraction, self.store, self._disambiguate_fn, ctx)
        supersessions = plan_supersessions(extraction, plan, self.store, self._supersede_fn, ctx)
        result = apply_write(
            extraction, plan, self.store,
            supersessions=supersessions,
            config=self.config, journal_path=self._journal_path, add_episode=False,
        )
        totals["edges_created"] += result.edges_created
        totals["edges_reinforced"] += result.edges_reinforced
        totals["edges_dropped"] += result.edges_dropped
        totals["edges_superseded"] += result.edges_superseded
        flagged.extend(result.flagged)
        warnings.extend(result.warnings)

        if result.edges_dropped:
            return f"dropped: {result.warnings[-1] if result.warnings else 'invalid edge'}"
        if result.edges_superseded:
            return (
                f"added {source_name} -[{relation}]-> {target_name}; this superseded "
                f"{result.edges_superseded} earlier relationship(s) between the same entities"
            )
        if result.edges_reinforced:
            return f"reinforced existing relationship {source_name} -[{relation}]-> {target_name} (already existed)"
        return f"added {source_name} -[{relation}]-> {target_name}"

    # --- RETRIEVE_MEMORY (deep retrieval tool) ---------------------------------

    def retrieve_memory(self, node_name: str, depth: int = 1) -> dict:
        """Deep retrieval (spec §5.7): full neighbourhood + snippets + episodes
        for a named node, on demand."""
        node = self._find_node(node_name)
        if node is None:
            return {"found": False, "node_name": node_name}
        hood = self.store.neighbourhood(node.key, depth)
        relationships = [
            {
                "source": e.source_key, "relation": e.relation, "target": e.target_key,
                "strength": round(e.strength, 2), "snippet": e.snippet,
            }
            for e in hood["edges"]
        ]
        episode_ids = {eid for e in hood["edges"] for eid in e.source_episode_ids}
        episode_ids.update(node.source_episode_ids)
        episodes = [
            {"summary": ep.summary, "importance": ep.importance, "tags": ep.tags}
            for eid in episode_ids
            if (ep := self.store.get_episode(eid)) is not None
        ]
        return {
            "found": True,
            "node": {"key": node.key, "name": node.name, "kind": node.kind,
                     "properties": node.properties, "aliases": node.aliases},
            "relationships": relationships,
            "episodes": episodes,
        }

    def _find_node(self, name: str):
        canon = canonicalize(name)
        for node in self.store.nodes.values():
            if canonicalize(node.name) == canon or canon in {canonicalize(a) for a in node.aliases}:
                return node
        return self.store.nodes.get(name)  # allow passing a raw key too

    # --- DECAY (Layer 6) -------------------------------------------------------

    def decay(self, *, now: datetime | None = None, force: bool = False) -> dict:
        """Daily pass: recompute strength for every EDGE from elapsed time since
        its last strength change. Nodes never decay. Dormant edges (strength <
        2.0) are flagged, never deleted.

        Self-gated: a no-op (no store mutation, no persist) if less than
        MIN_DECAY_INTERVAL has elapsed since the last run, tracked in
        `_decay_state_path` rather than relying on a caller-side schedule.
        This exists so the packaged opencode plugin can fire-and-forget
        POST /decay once per opencode process start with no cron -- most of
        those calls will be well inside 24h of the last one and should cost
        nothing. The gate compares against the `now` parameter (not real
        wall-clock), so simulated multi-day test runs that step `now` forward
        deterministically are unaffected. `force=True` (used by the /decay
        cron path some deployments still run explicitly) bypasses the gate."""
        now = now or _now()
        if not force:
            last = self._last_decay_at()
            if last is not None and (now - last) < MIN_DECAY_INTERVAL:
                return {"edges_decayed": 0, "edges_dormant": 0, "skipped": True}
        decayed = 0
        dormant = 0
        for edge in self.store.edges.values():
            ts = _parse_iso(edge.updated_at)
            if ts is None:
                continue
            elapsed_days = max(0.0, (now - ts).total_seconds() / 86400.0)
            edge.strength = decayed_strength(edge.strength, edge.stability, elapsed_days)
            edge.updated_at = now.isoformat()
            decayed += 1
            if is_dormant(edge.strength):
                dormant += 1
        self._persist()
        self._decay_state_path.write_text(json.dumps({"last_decay_at": now.isoformat()}))
        append_event(self._journal_path, {
            "event": "decay", "edges_decayed": decayed, "edges_dormant": dormant,
            "dormancy_threshold": DORMANCY_THRESHOLD,
        })
        return {"edges_decayed": decayed, "edges_dormant": dormant}

    def _last_decay_at(self) -> datetime | None:
        if not self._decay_state_path.exists():
            return None
        try:
            data = json.loads(self._decay_state_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        return _parse_iso(data.get("last_decay_at", ""))
