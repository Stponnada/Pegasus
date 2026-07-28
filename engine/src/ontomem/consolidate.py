"""Write pipeline orchestration (spec v0.2.1 §4): Stage 0 context assembly,
Stage 2 merge planning, and the deterministic Stage 3 graph write.

The two LLM-dependent steps (extraction, 2b disambiguation) are injected so the
merge-plan application — the part where bugs corrupt the graph — is pure and
hermetically testable. Reinforcement is applied here at conversation close: both
write-time (a re-mentioned relationship, §4.4) and the deferred Read traversals
logged during the session (§6.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .extractor import ExtractionResult
from .journal import append_event
from .merge import AUTO_MERGE, NEEDS_REVIEW, NEW, Resolution, resolve_entity
from .model import Edge, edge_key
from .retriever import RetrievalConfig, extract_nouns, resolve_seed_hits
from .supersede import CONTRADICTS, SUPERSEDES


# --- Stage 0: existing-graph context assembly ----------------------------------


def assemble_context(conversation, store, index, embedder, config: RetrievalConfig | None = None) -> str:
    """Reuse the Read seeder over the WHOLE conversation to find anchor nodes,
    then render their 2-hop neighbourhoods. Breadth, not lean injection: no
    activation weighting, no node budget (spec §4.1)."""
    config = config or RetrievalConfig()
    if len(index) == 0:
        return ""
    if index.model_id != embedder.model_id:
        raise ValueError("embedding model mismatch in Stage 0; re-embed required")
    text = " ".join(t.get("text", "") for t in conversation) if isinstance(conversation, list) else str(conversation)
    nouns = extract_nouns(text)
    if not nouns:
        return ""
    hits: dict[str, None] = {}
    for vec in embedder.embed(nouns):
        for key, _ in index.search(
            vec, top_k=config.seed_top_k, threshold=config.seed_similarity_threshold, center=True
        ):
            hits.setdefault(key, None)
    anchors = resolve_seed_hits(list(hits), store)  # map relation hits -> endpoint nodes
    return store.render_subgraph(anchors, depth=config.max_depth)


# --- Stage 2: merge planning ---------------------------------------------------


@dataclass
class MergePlan:
    remap: dict[str, str] = field(default_factory=dict)  # extracted key -> final canonical key
    flagged: list[dict] = field(default_factory=list)  # 0.70-0.90 band merges needing inspection
    notes: list[dict] = field(default_factory=list)  # per-entity audit trail
    inferred_self_key: str | None = None  # set only when store.self_key was None coming in


def plan_merges(extraction: ExtractionResult, store, disambiguate_fn, conversation_context: str = "") -> MergePlan:
    """Resolve each extracted entity against the graph (2a), escalating
    NEEDS_REVIEW to the injected `disambiguate_fn` (2b). Returns a key remap from
    extracted node keys to final canonical keys (merges only)."""
    plan = MergePlan()
    existing = list(store.nodes.values())
    for node in extraction.nodes:
        res = resolve_entity(node, existing, self_key=store.self_key)
        # incorporate the extractor's soft candidate_merge_key hint
        if res.decision == NEW:
            hint = extraction.candidate_merge_keys.get(node.key)
            if hint and hint in store.nodes and store.nodes[hint].kind == node.kind:
                res = Resolution(node.key, NEEDS_REVIEW, "extractor_hint", candidates=[hint])

        if res.decision == AUTO_MERGE:
            plan.remap[node.key] = res.target_key
            plan.notes.append({"entity": node.key, "action": "auto_merge", "rule": res.rule, "target": res.target_key})
        elif res.decision == NEEDS_REVIEW:
            decision = disambiguate_fn(node, res, store, conversation_context)
            if decision.decision == AUTO_MERGE:
                plan.remap[node.key] = decision.target_key
                plan.notes.append({"entity": node.key, "action": "llm_merge", "rule": res.rule, "target": decision.target_key, "confidence": decision.confidence})
                if decision.flagged:
                    plan.flagged.append({"entity": node.key, "target": decision.target_key, "confidence": decision.confidence, "reason": decision.reason})
            else:
                plan.notes.append({"entity": node.key, "action": "new_after_review", "rule": res.rule})
        else:
            plan.notes.append({"entity": node.key, "action": "new", "rule": res.rule})

    # Designate the self-node the first time a graph ever sees a relationship:
    # per spec §4.2 direction convention, edges are "user-outward," so the
    # source of the first-ever extracted relationship is, in practice, the
    # user themselves. Only inferred once (store.self_key stays sticky after).
    if store.self_key is None and extraction.edges:
        first_source = extraction.edges[0].source_key
        plan.inferred_self_key = plan.remap.get(first_source, first_source)

    return plan


# --- Stage 2.5: relationship supersession planning -----------------------------


@dataclass
class Supersession:
    old_edge_key: str
    new_relation: str
    decision: str  # SUPERSEDES | CONTRADICTS (COEXIST never reaches here)
    reason: str = ""


def plan_supersessions(extraction, plan, store, supersede_fn, conversation_context: str = "") -> list[Supersession]:
    """For each new edge, find existing edges between the SAME (remapped) endpoints
    under a DIFFERENT relation and ask the injected `supersede_fn` how to reconcile
    them. Only SUPERSEDES / CONTRADICTS produce an action; COEXIST is dropped. Pure
    orchestration — the network/LLM lives behind supersede_fn, like 2b's
    disambiguate_fn — so the deterministic write stage stays LLM-free."""
    out: list[Supersession] = []
    checked: set[tuple[str, str]] = set()
    for edge in extraction.edges:
        src = plan.remap.get(edge.source_key, edge.source_key)
        tgt = plan.remap.get(edge.target_key, edge.target_key)
        if src not in store.nodes or tgt not in store.nodes:
            continue
        for existing in list(store.edges.values()):
            if existing.source_key != src or existing.target_key != tgt:
                continue
            if existing.relation == edge.relation:
                continue  # same relation -> a re-mention (reinforce), not supersession
            if existing.properties.get("superseded_by"):
                continue  # already demoted; don't re-evaluate
            pair = (existing.key, edge.relation)
            if pair in checked:
                continue
            checked.add(pair)
            decision = supersede_fn(existing, edge, store, conversation_context)
            if decision.decision in (SUPERSEDES, CONTRADICTS):
                out.append(Supersession(existing.key, edge.relation, decision.decision, decision.reason))
    return out


# --- Stage 3: deterministic graph write ----------------------------------------


@dataclass
class WriteResult:
    episode_id: str
    nodes_created: int = 0
    nodes_merged: int = 0
    edges_created: int = 0
    edges_reinforced: int = 0
    edges_dropped: int = 0
    edges_superseded: int = 0
    reinforced_from_read: int = 0
    flagged: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def apply_write(
    extraction: ExtractionResult,
    plan: MergePlan,
    store,
    *,
    reinforcement_edge_keys=None,
    supersessions=None,
    config: RetrievalConfig | None = None,
    journal_path=None,
) -> WriteResult:
    """Execute the merge plan deterministically. No LLM. The one place graph
    corruption could originate, so it is the most carefully tested."""
    config = config or RetrievalConfig()
    result = WriteResult(episode_id=extraction.episode.id, flagged=list(plan.flagged), warnings=list(extraction.warnings))
    if store.self_key is None and plan.inferred_self_key:
        store.self_key = plan.inferred_self_key
    store.add_episode(extraction.episode)
    _write_nodes(extraction, plan, store, result)
    _write_edges(extraction, plan, store, result, config)
    _apply_supersessions(supersessions or [], store, result, config)
    _flag_orphans(extraction, plan, store, result)
    _apply_read_reinforcement(reinforcement_edge_keys or [], store, result, config)
    _journal(journal_path, extraction, plan, result)
    return result


def _write_nodes(extraction, plan, store, result) -> None:
    ep_id = extraction.episode.id
    for node in extraction.nodes:
        final = plan.remap.get(node.key, node.key)
        existing = store.get_node(final)
        if existing is None:
            node.source_episode_ids = [ep_id]
            store.add_node(node)
            result.nodes_created += 1
            continue
        # merge into the canonical node: keep its surface form as an alias too
        new_aliases = [a for a in (*node.aliases, node.name) if a]
        store.update_node(
            final,
            add_aliases=new_aliases,
            add_properties=node.properties,
            add_episode_ids=[ep_id],
            confidence=max(existing.confidence, node.confidence),
        )
        result.nodes_merged += 1


def _write_edges(extraction, plan, store, result, config) -> None:
    ep_id = extraction.episode.id
    for edge in extraction.edges:
        src = plan.remap.get(edge.source_key, edge.source_key)
        tgt = plan.remap.get(edge.target_key, edge.target_key)
        if src == tgt:
            result.edges_dropped += 1
            result.warnings.append(f"edge {edge.key} dropped: endpoints merged to same node {src}")
            continue
        if src not in store.nodes or tgt not in store.nodes:
            result.edges_dropped += 1
            result.warnings.append(f"edge {edge.key} dropped: missing endpoint after remap")
            continue
        # bind this episode to existing endpoint nodes too (co-episode across
        # conversations) — freshly-created nodes already carry it
        for endpoint in (src, tgt):
            node = store.get_node(endpoint)
            if node is not None and ep_id not in node.source_episode_ids:
                node.source_episode_ids.append(ep_id)
        final_key = edge_key(src, edge.relation, tgt)
        existing = store.get_edge(final_key)
        if existing is None:
            remapped = Edge.create(
                source_key=src, relation=edge.relation, target_key=tgt,
                snippet=edge.snippet, evidence=edge.evidence, confidence=edge.confidence,
                stability=edge.stability, ttl_days=edge.ttl_days, cardinality=edge.cardinality,
                properties=dict(edge.properties), source_episode_ids=[ep_id],
            )
            store.add_edge(remapped)
            result.edges_created += 1
            continue
        # re-mentioned relationship: reinforce, refresh snippet only on higher confidence
        store.reinforce_edge(final_key, boost=config.reinforce_boost)
        store.update_edge(
            final_key,
            add_episode_ids=[ep_id],
            **({"snippet": edge.snippet, "evidence": edge.evidence, "confidence": edge.confidence}
               if edge.confidence > existing.confidence else {}),
        )
        result.edges_reinforced += 1


def _flag_orphans(extraction, plan, store, result) -> None:
    """Observability only: an entity extracted this turn that ends up with no
    incident edge is unreachable by traversal — a silent write failure the
    extractor's NO-ORPHAN rule is meant to prevent but cannot guarantee (model
    non-determinism). Flag it; never drop or auto-link (that is a policy choice,
    spec §9.2c). Mirrors the flag-don't-mutate philosophy of dormancy/duplicate
    scanning."""
    seen: set[str] = set()
    for node in extraction.nodes:
        final = plan.remap.get(node.key, node.key)
        if final in seen:
            continue
        seen.add(final)
        live = store.get_node(final)
        if live is not None and not live.outgoing and not live.incoming:
            result.warnings.append(
                f"orphan entity {final}: extracted but participates in no relationship (unreachable)"
            )


def _apply_supersessions(supersessions, store, result, config) -> None:
    """Demote each superseded edge: strength -> near-dormant and tag it
    `superseded_by`. NON-DESTRUCTIVE — the edge stays in the graph (provenance,
    and a later conversation could revive it), it just stops dominating activation
    and surfacing as current. CONTRADICTS additionally flags for inspection."""
    for s in supersessions:
        edge = store.get_edge(s.old_edge_key)
        if edge is None:
            continue
        store.update_edge(
            s.old_edge_key,
            strength=config.supersede_demotion_strength,
            add_properties={"superseded_by": s.new_relation},
        )
        result.edges_superseded += 1
        result.warnings.append(
            f"edge {s.old_edge_key} superseded by {s.new_relation} (demoted to "
            f"{config.supersede_demotion_strength}, not deleted)"
        )
        if s.decision == CONTRADICTS:
            result.flagged.append({
                "edge": s.old_edge_key, "superseded_by": s.new_relation,
                "kind": "contradiction", "reason": s.reason,
            })


def _apply_read_reinforcement(edge_keys, store, result, config) -> None:
    for key in edge_keys:
        if key in store.edges:
            store.reinforce_edge(key, boost=config.reinforce_boost)
            result.reinforced_from_read += 1


def _journal(journal_path, extraction, plan, result) -> None:
    if journal_path is None:
        return
    append_event(journal_path, {
        "event": "write",
        "episode_id": result.episode_id,
        "nodes_created": result.nodes_created,
        "nodes_merged": result.nodes_merged,
        "edges_created": result.edges_created,
        "edges_reinforced": result.edges_reinforced,
        "edges_dropped": result.edges_dropped,
        "reinforced_from_read": result.reinforced_from_read,
        "merge_notes": plan.notes,
        "flagged": result.flagged,
        "warnings": result.warnings,
    })
