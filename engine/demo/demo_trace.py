"""Produce an animation-ready trace of a real Read.

Runs the actual retriever pipeline (find_seeds -> spreading_activation ->
co-episode boost -> budget -> format_context) and returns an ordered event list
plus the authoritative injected [MEMORY]/[CONTEXT] blocks. Each edge-crossing
event is annotated with whether it ends up injected into context, so the UI can
drop the snippet in exactly when that edge is crossed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ontomem.retriever import (
    RetrievalConfig,
    apply_co_episode_boost,
    extract_nouns,
    find_seeds,
    format_context,
    select_within_budget,
    spreading_activation,
)


def graph_json(store) -> dict:
    """Full graph for the initial render."""
    return {
        "nodes": [
            {"id": n.key, "label": n.name, "kind": n.kind,
             "episodes": list(n.source_episode_ids)}
            for n in store.nodes.values()
        ],
        "edges": [
            {"id": e.key, "from": e.source_key, "to": e.target_key,
             "label": e.relation, "strength": round(e.strength, 1),
             "snippet": e.snippet,
             "superseded": bool(e.properties.get("superseded_by"))}
            for e in store.edges.values()
        ],
    }


def trace_read(message, store, index, embedder, config: RetrievalConfig | None = None) -> dict:
    config = config or RetrievalConfig()
    nouns = extract_nouns(message)
    seeds = find_seeds(message, index, embedder, store, config)

    events: list = []
    spread = spreading_activation(store, seeds, config, events=events)
    activations = dict(spread["activations"])

    # co-episode boost (record which nodes got lifted by sharing a fired episode)
    before = {k: a.score for k, a in activations.items()}
    activations = apply_co_episode_boost(activations, store, config)
    for key, act in activations.items():
        lifted = act.score - before.get(key, 0.0)
        if act.co_episode and lifted > 0:
            events.append({"type": "boost", "node": key, "name": store.nodes[key].name,
                           "amount": round(lifted, 4), "cumulative": round(act.score, 4)})

    selected = select_within_budget(activations, config)
    selected_set = set(selected)
    blocks = format_context(store, selected, activations, config)

    # annotate each crossing: does this edge end up injected as a [MEMORY] line,
    # and does its snippet land in [CONTEXT]? (mirrors format_context's rules)
    context_block = blocks["context_block"]
    for ev in events:
        if ev["type"] != "cross":
            continue
        edge = store.get_edge(ev["edge"])
        both_selected = edge.source_key in selected_set and edge.target_key in selected_set
        ev["injected_memory"] = bool(both_selected and not ev["superseded"])
        ev["injected_snippet"] = bool(ev["injected_memory"] and edge.snippet and edge.snippet in context_block)

    selected_detail = [
        {"key": k, "name": store.nodes[k].name,
         "activation": round(activations[k].score, 4),
         "tier": "secondary" if activations[k].score >= config.secondary_threshold else "primary",
         "direction": activations[k].direction}
        for k in selected
    ]

    return {
        "query": message,
        "nouns": nouns,
        "seeds": [{"key": s, "name": store.nodes[s].name} for s in seeds if s in store.nodes],
        "events": events,
        "selected": selected_detail,
        "thresholds": {"primary": config.primary_threshold, "secondary": config.secondary_threshold,
                       "node_budget": config.node_budget},
        "memory_block": blocks["memory_block"],
        "context_block": blocks["context_block"],
        "final_context": blocks["text"],
    }
