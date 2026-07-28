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

from datetime import datetime, timezone
from pathlib import Path

from .consolidate import apply_write, assemble_context, plan_merges, plan_supersessions
from .decay import DORMANCY_THRESHOLD, decayed_strength, is_dormant
from .embeddings import EmbeddingIndex, HashingEmbedder, _node_text, relation_phrase
from .extractor import extract as _llm_extract
from .journal import append_event, read_events
from .merge_llm import disambiguate as _llm_disambiguate
from .model import canonicalize
from .retriever import RELATION_KEY_PREFIX, RetrievalConfig, read as _retrieve
from .supersede import decide_supersession as _llm_supersede
from .store import Store


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
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
        model: str = "gemini-3.1-flash-lite",
        api_key: str | None = None,
    ) -> None:
        self.dir = Path(base_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or HashingEmbedder()
        self.config = config or RetrievalConfig()
        self.model = model
        self.api_key = api_key
        self.generate_fn = generate_fn
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
            generate_fn=self.generate_fn,
        )

    def _default_disambiguate(self, entity, resolution, store, ctx):
        return _llm_disambiguate(
            entity, resolution, store, ctx, model=self.model, api_key=self.api_key,
            generate_fn=self.generate_fn,
        )

    def _default_supersede(self, old_edge, new_edge, store, ctx):
        src = store.get_node(old_edge.source_key)
        tgt = store.get_node(old_edge.target_key)
        return _llm_supersede(
            old_edge, new_edge,
            src.name if src else old_edge.source_key,
            tgt.name if tgt else old_edge.target_key,
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
        pending Read-reinforcement log and rebuilds the embedding index."""
        context = assemble_context(conversation, self.store, self.index, self.embedder, self.config)
        extraction = self._extract_fn(conversation, context)
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

    def decay(self, *, now: datetime | None = None) -> dict:
        """Daily pass: recompute strength for every EDGE from elapsed time since
        its last strength change. Nodes never decay. Dormant edges (strength <
        2.0) are flagged, never deleted."""
        now = now or _now()
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
        append_event(self._journal_path, {
            "event": "decay", "edges_decayed": decayed, "edges_dormant": dormant,
            "dormancy_threshold": DORMANCY_THRESHOLD,
        })
        return {"edges_decayed": decayed, "edges_dormant": dormant}
