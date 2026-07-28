"""Read pipeline / Retriever (spec v0.2.1 §5). Mostly pure graph mechanics.

Pipeline: noun extraction -> embedding seed finding -> multi-source spreading
activation (asymmetric forward/reverse) -> co-episode boost -> threshold + node
budget -> tiered [MEMORY]/[CONTEXT] injection. Seed finding is the only
embedding-dependent step; everything else is pure and hermetically tested.

All tunables live in RetrievalConfig (provisional per spec §9.3 — configurable,
not hardcoded). Read is non-destructive: it records the edges it traversed (for
deferred reinforcement at conversation close) but never mutates the graph.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

# directions
FORWARD = "forward"
REVERSE = "reverse"
SEED = "seed"

# keys in the embedding index prefixed with this are relation-phrase vectors, not
# nodes. A query noun matching one ('partner' -> REL::HAS_PARTNER) seeds the
# relation's endpoint nodes rather than a node by name (spec §5.2).
RELATION_KEY_PREFIX = "REL::"

_STOPWORDS = frozenset(
    """
    a an the this that these those my your his her its our their me you he she it we they him them us
    i and or but nor so yet for of to in on at by with from into onto over under as is am are was were be
    been being do does did done have has had having will would shall should can could may might must
    not no yes if then else when where why how what which who whom whose all any some each few more most
    other such only own same than too very just about up out off down then once here there
    me about get got like really actually maybe think know want need help thanks please okay ok hey hi
    """.split()
)


@dataclass
class RetrievalConfig:
    hop_decay: float = 0.6
    max_depth: int = 2
    primary_threshold: float = 0.30
    secondary_threshold: float = 0.60
    node_budget: int = 8
    forward_ratio: float = 0.70
    # threshold is over MEAN-CENTRED cosine (anisotropy-corrected, see
    # EmbeddingIndex.search center=True). On raw semantic cosine the baseline is
    # ~0.55 so any absolute cut over-seeds; in centred space unrelated pairs sit
    # near 0 and a true entity match clears ~0.30+. (Found via dogfooding.)
    seed_similarity_threshold: float = 0.30
    # Margin gate: a noun whose best match is moderate but clearly separated from
    # the runner-up should still seed. On real data 'partner' matches HAS_PARTNER
    # at 0.26 (just under the absolute floor) but with a 0.20 gap to the next
    # candidate — an unambiguous winner. Pure-noise abstract nouns ('career',
    # 'decision') top out ~0.10 with ~0.02 gaps and are still rejected. Accept the
    # top hit if score >= seed_margin_floor AND (top - runner_up) >= seed_margin.
    seed_margin: float = 0.15
    seed_margin_floor: float = 0.15
    seed_top_k: int = 3
    co_episode_boost: float = 0.15
    reinforce_boost: float = 15.0
    # Cap on how many of a single hot node's own snippets get injected. Without
    # this, a heavily-connected hub node (the user themselves, almost always
    # at least weakly relevant) crossing the secondary threshold could dump
    # every relationship it has, defeating the terse-context goal (spec §1.1).
    # Provisional (spec §9.3).
    max_snippets_per_node: int = 4
    # strength a superseded edge is demoted to (below dormancy 2.0 -> effectively
    # non-firing but never deleted; spec §9.2c). Provisional.
    supersede_demotion_strength: float = 1.0


@dataclass
class Activation:
    key: str
    score: float
    direction: str  # FORWARD | REVERSE | SEED
    hop: int
    co_episode: bool = False


@dataclass
class RetrievalResult:
    memory_block: str
    context_block: str
    text: str
    seeds: list[str] = field(default_factory=list)
    activations: dict = field(default_factory=dict)  # key -> Activation (post-boost)
    selected: list[str] = field(default_factory=list)
    traversed_edge_keys: list[str] = field(default_factory=list)
    injections: list = field(default_factory=list)  # structured per-relationship view, see format_context


# --- step 1: noun extraction ---------------------------------------------------


def extract_nouns(message: str) -> list[str]:
    """Pragmatic noun/content-word extraction: tokenise, drop function words and
    1-char tokens, dedupe preserving order. (Not a POS tagger — a lexical filter
    to concentrate the seeding signal, per spec §5.1.)"""
    seen: dict[str, None] = {}
    for raw in "".join(c if c.isalnum() else " " for c in message.lower()).split():
        if len(raw) > 1 and raw not in _STOPWORDS:
            seen.setdefault(raw, None)
    return list(seen)


# --- step 2: seed finding (embedding-dependent) --------------------------------


def resolve_seed_hits(hit_keys, store) -> list[str]:
    """Map raw index hits to node keys. A node-key hit resolves to itself; a
    relation-key hit (REL::<RELATION>) resolves to the endpoint nodes of every
    edge carrying that relation. Stale keys not in the graph are dropped. With
    store=None, hits are returned unchanged (node-only index, used in tests)."""
    if store is None:
        return list(dict.fromkeys(hit_keys))
    resolved: dict[str, None] = {}
    for key in hit_keys:
        if key.startswith(RELATION_KEY_PREFIX):
            relation = key[len(RELATION_KEY_PREFIX):]
            for edge in store.edges.values():
                if edge.relation == relation:
                    resolved.setdefault(edge.source_key, None)
                    resolved.setdefault(edge.target_key, None)
        elif key in store.nodes:
            resolved.setdefault(key, None)
    return list(resolved)


def _gate_noun_hits(ranked, config: RetrievalConfig) -> list[str]:
    """Decide which of a noun's ranked (key, score) hits become seeds: everything
    above the absolute threshold, plus the top hit if it is a clearly-separated
    winner (margin gate)."""
    accepted = [key for key, score in ranked if score >= config.seed_similarity_threshold]
    if ranked:
        top_key, top_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        if (
            top_key not in accepted
            and top_score >= config.seed_margin_floor
            and (top_score - runner_up) >= config.seed_margin
        ):
            accepted.append(top_key)
    return accepted


def find_seeds(message, index, embedder, store=None, config: RetrievalConfig | None = None) -> list[str]:
    config = config or RetrievalConfig()
    if index.model_id != embedder.model_id:
        raise ValueError(
            f"embedding model mismatch: index={index.model_id!r} embedder={embedder.model_id!r}; re-embed required"
        )
    nouns = extract_nouns(message)
    if not nouns or len(index) == 0:
        return []
    vectors = embedder.embed(nouns)
    hits: dict[str, None] = {}
    for vec in vectors:
        # fetch ranked candidates (threshold 0) so the margin gate can see the gap
        ranked = index.search(vec, top_k=max(config.seed_top_k, 2), threshold=0.0, center=True)
        for key in _gate_noun_hits(ranked, config):
            hits.setdefault(key, None)
    return resolve_seed_hits(list(hits), store)


# --- step 3: multi-source spreading activation (pure) --------------------------


def _record_crossing(events, store, hop, from_key, edge, reached_key, direction, contribution, cumulative) -> None:
    if events is None:
        return
    events.append({
        "type": "cross", "hop": hop, "edge": edge.key, "direction": direction,
        "from": from_key, "from_name": store.nodes[from_key].name,
        "reached": reached_key, "reached_name": store.nodes[reached_key].name,
        "relation": edge.relation, "strength": round(edge.strength, 1),
        "contribution": round(contribution, 4), "cumulative": round(cumulative, 4),
        "snippet": edge.snippet,
        "superseded": bool(edge.properties.get("superseded_by")),
    })


def spreading_activation(store, seed_keys, config: RetrievalConfig | None = None, *, events: list | None = None) -> dict:
    """Multi-source degree-normalised BFS. If `events` is supplied, ordered
    crossing events are appended to it (seed, then each edge crossed with its
    contribution) for observability/visualisation — behaviour is unchanged."""
    config = config or RetrievalConfig()
    total: dict[str, float] = defaultdict(float)
    forward: dict[str, float] = defaultdict(float)
    reverse: dict[str, float] = defaultdict(float)
    hop_of: dict[str, int] = {}
    traversed: set[str] = set()

    frontier: dict[str, float] = {}
    for key in seed_keys:
        if key in store.nodes:
            total[key] = max(total[key], 1.0)
            hop_of[key] = 0
            frontier[key] = 1.0
            if events is not None:
                events.append({"type": "seed", "node": key, "name": store.nodes[key].name, "activation": 1.0})

    for hop in range(1, config.max_depth + 1):
        nxt: dict[str, float] = defaultdict(float)
        for node_key, act in frontier.items():
            node = store.nodes[node_key]
            # Degree normalisation: a node distributes its activation across its
            # incident edges (weighted by strength) instead of handing full
            # strength to every neighbour. This stops a high-degree hub (e.g. the
            # single user node) from re-radiating above-threshold activation to
            # all its spokes — the §9.2 dense-graph overload problem. A spoke is
            # surfaced only when several paths genuinely converge on it.
            incident = (*node.outgoing, *node.incoming)
            total_strength = sum(e.strength for e in incident) or 1.0
            for edge in node.outgoing:
                contrib = act * (edge.strength / total_strength) * config.hop_decay
                nxt[edge.target_key] += contrib
                forward[edge.target_key] += contrib
                traversed.add(edge.key)
                _record_crossing(events, store, hop, node_key, edge, edge.target_key, FORWARD,
                                  contrib, total.get(edge.target_key, 0.0) + nxt[edge.target_key])
            for edge in node.incoming:
                contrib = act * (edge.strength / total_strength) * config.hop_decay
                nxt[edge.source_key] += contrib
                reverse[edge.source_key] += contrib
                traversed.add(edge.key)
                _record_crossing(events, store, hop, node_key, edge, edge.source_key, REVERSE,
                                  contrib, total.get(edge.source_key, 0.0) + nxt[edge.source_key])
        for key, contrib in nxt.items():
            total[key] += contrib
            hop_of.setdefault(key, hop)
        frontier = nxt

    activations: dict[str, Activation] = {}
    for key, score in total.items():
        if hop_of.get(key) == 0:
            direction = SEED
        elif forward[key] >= reverse[key]:
            direction = FORWARD
        else:
            direction = REVERSE
        activations[key] = Activation(key, score, direction, hop_of.get(key, 0))
    return {"activations": activations, "traversed_edge_keys": sorted(traversed)}


# --- step 4: co-episode boost (pure) -------------------------------------------


def apply_co_episode_boost(activations: dict, store, config: RetrievalConfig | None = None) -> dict:
    config = config or RetrievalConfig()
    fired_keys = [k for k, a in activations.items() if a.score >= config.primary_threshold]
    fired_episodes: set[str] = set()
    for key in fired_keys:
        node = store.nodes.get(key)
        if node:
            fired_episodes.update(node.source_episode_ids)
    if not fired_episodes:
        return activations
    for key, node in store.nodes.items():
        if fired_episodes.intersection(node.source_episode_ids):
            act = activations.get(key)
            if act is None:
                activations[key] = Activation(key, config.co_episode_boost, FORWARD, 1, co_episode=True)
            elif act.direction != SEED:
                act.score += config.co_episode_boost
                act.co_episode = True
    return activations


# --- step 5: threshold + asymmetric node budget (pure) -------------------------


def select_within_budget(activations: dict, config: RetrievalConfig | None = None) -> list[str]:
    config = config or RetrievalConfig()
    fired = sorted(
        (a for a in activations.values() if a.score >= config.primary_threshold),
        key=lambda a: (-a.score, a.key),
    )
    forward_class = [a for a in fired if a.direction in (FORWARD, SEED)]
    reverse_class = [a for a in fired if a.direction == REVERSE]

    fwd_budget = round(config.node_budget * config.forward_ratio)
    rev_budget = config.node_budget - fwd_budget

    chosen = forward_class[:fwd_budget] + reverse_class[:rev_budget]
    # don't waste budget: backfill from whichever class has leftovers
    if len(chosen) < config.node_budget:
        remaining = [a for a in fired if a not in chosen][: config.node_budget - len(chosen)]
        chosen += remaining
    chosen.sort(key=lambda a: (-a.score, a.key))
    return [a.key for a in chosen[: config.node_budget]]


# --- step 6: tiered context injection (pure) -----------------------------------


def _relation_phrase(relation: str) -> str:
    """UPPER_SNAKE_CASE -> lowercase, space-separated, for human-readable
    display (e.g. debug/live-injection views, not the terse LLM-facing block)."""
    return relation.replace("_", " ").lower()


def _node_snippets(node, limit: int):
    """Every qualifying snippet on a node's OWN incident edges (both
    directions), ranked by strength and capped at `limit`.

    Spec design decision (§3.2): "when an edge fires during retrieval, the
    snippet loads the episodic content of that relationship" -- firing is a
    property of the node crossing the secondary threshold, and every edge
    incident to a hot node is relevant context, not just whichever single
    edge happens to score highest on an unrelated strength*importance
    formula. The old single-best-edge selection could surface e.g. a WORKS_AS
    snippet in response to a query about a completely unrelated relationship
    on the same node, simply because that edge scored marginally higher.
    Capped per node (not globally) so one heavily-connected hub can't alone
    dump every relationship it has."""
    edges = [
        e for e in (*node.outgoing, *node.incoming)
        if e.snippet and not e.properties.get("superseded_by")
    ]
    edges.sort(key=lambda e: e.strength, reverse=True)
    return edges[:limit]


def format_context(store, selected, activations, config: RetrievalConfig | None = None) -> dict:
    config = config or RetrievalConfig()
    selected_set = set(selected)
    memory_lines: list[str] = []
    seen_lines: set[str] = set()
    # Structured, per-relationship view for observability/live-injection
    # display (e.g. the memory viewer) -- NOT part of the terse LLM-facing
    # text block, which stays exactly [MEMORY]/[CONTEXT] as before. Keyed by
    # edge key so the two passes below (structural lines, then snippets) can
    # fill in the same entry without fragile name-based matching.
    injections: dict[str, dict] = {}

    def _entry(edge, source_node, target_node) -> dict:
        return injections.setdefault(edge.key, {
            "source": source_node.name, "relation": edge.relation,
            "relation_phrase": _relation_phrase(edge.relation),
            "target": target_node.name, "snippet": None,
        })

    for key in selected:
        node = store.nodes.get(key)
        if not node:
            continue
        for edge in node.outgoing:
            # only render edges WITHIN the activated set — a weakly-reached hub
            # must not dump every spoke that never fired (selectivity, spec §1.1)
            if edge.target_key not in selected_set:
                continue
            if edge.properties.get("superseded_by"):
                continue  # stale edge demoted by a later one (§9.2c) — hide it
            target = store.nodes.get(edge.target_key)
            if target is None:
                continue
            line = f"{node.name} -[{edge.relation}]-> {target.name}"
            if line not in seen_lines:
                seen_lines.add(line)
                memory_lines.append(line)
            _entry(edge, node, target)

    context_snippets: list[str] = []
    seen_snippets: set[str] = set()
    for key in selected:
        act = activations.get(key)
        node = store.nodes.get(key)
        if not node or act is None or act.score < config.secondary_threshold:
            continue
        for edge in _node_snippets(node, config.max_snippets_per_node):
            if edge.snippet not in seen_snippets:
                seen_snippets.add(edge.snippet)
                context_snippets.append(f'"{edge.snippet}"')
            # Attach the snippet to its own edge's entry regardless of whether
            # the structural line above rendered it — this is exactly the
            # case test_format_context_injects_snippet_above_secondary
            # exercises: a node's own relationship snippet is relevant on its
            # own merit even when the other endpoint never independently fired.
            if edge.source is not None and edge.target is not None:
                entry = _entry(edge, edge.source, edge.target)
                entry["snippet"] = edge.snippet

    memory_block = "[MEMORY]\n" + "\n".join(memory_lines) if memory_lines else ""
    context_block = "[CONTEXT]\n" + "\n".join(context_snippets) if context_snippets else ""
    text = "\n\n".join(b for b in (memory_block, context_block) if b)
    return {
        "memory_block": memory_block, "context_block": context_block, "text": text,
        "injections": list(injections.values()),
    }


# --- orchestration -------------------------------------------------------------


def read(message, store, index, embedder, config: RetrievalConfig | None = None) -> RetrievalResult:
    config = config or RetrievalConfig()
    seeds = find_seeds(message, index, embedder, store, config)
    spread = spreading_activation(store, seeds, config)
    activations = apply_co_episode_boost(spread["activations"], store, config)
    selected = select_within_budget(activations, config)
    blocks = format_context(store, selected, activations, config)
    return RetrievalResult(
        memory_block=blocks["memory_block"],
        context_block=blocks["context_block"],
        text=blocks["text"],
        seeds=seeds,
        activations=activations,
        selected=selected,
        traversed_edge_keys=spread["traversed_edge_keys"],
        injections=blocks["injections"],
    )
