"""Pre-seeded demo graph (hand-authored, no LLM/network needed).

A single user — Marcus — with three loosely-connected clusters:
  - work:    Anthropic, Dana, Wei, on-call, the safety-team transfer
  - history: Roman Empire -> Crisis of the 3rd C. -> Diocletian -> Tetrarchy
  - personal: Elena, the Rome trip

Rome is a deliberate BRIDGE: `Marcus -PLANS_TO_VISIT-> Rome -PART_OF-> Roman Empire`
links the personal cluster to the history cluster, so a query about the trip
spreads two hops into the history thread. Snippets are written to read well in
isolation, so the "context window" panel is meaningful.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ontomem.embeddings import EmbeddingIndex, HashingEmbedder, _node_text, relation_phrase
from ontomem.model import Edge, Episode, Node
from ontomem.retriever import RELATION_KEY_PREFIX
from ontomem.store import Store

# (kind, name) for every node
_NODES = [
    ("PERSON", "Marcus"),
    ("ORG", "Anthropic"),
    ("PERSON", "Dana"),
    ("PERSON", "Wei"),
    ("THING", "On-call Rotation"),
    ("ORG", "Alignment Safety Team"),
    ("PERSON", "Priya"),
    ("PERSON", "Elena"),
    ("TOPIC", "Roman Empire"),
    ("EVENT", "Crisis of the Third Century"),
    ("PERSON", "Diocletian"),
    ("THING", "Tetrarchy"),
    ("PLACE", "Rome"),
]

# (source, RELATION, target, strength, snippet)
_EDGES = [
    ("PERSON::marcus", "WORKS_AT", "ORG::anthropic", 100.0,
     "I'm a software engineer at Anthropic — I work on inference infrastructure, making the models serve faster."),
    ("PERSON::marcus", "HAS_MANAGER", "PERSON::dana", 90.0,
     "My manager Dana is great — really supportive — but the team is stretched thin right now."),
    ("PERSON::marcus", "WORKS_WITH", "PERSON::wei", 80.0,
     "Wei is my closest teammate. He usually covers my on-call when I'm slammed, but he's on vacation this week."),
    ("PERSON::marcus", "STRUGGLES_WITH", "THING::on_call_rotation", 85.0,
     "The on-call rotation is brutal — I had three pages last night alone. I keep saying I'll set boundaries and never do."),
    ("THING::on_call_rotation", "PART_OF", "ORG::anthropic", 70.0,
     "On-call is part of keeping Anthropic's inference systems up around the clock."),
    ("PERSON::marcus", "CONSIDERING_TRANSFER_TO", "ORG::alignment_safety_team", 75.0,
     "I got offered a transfer to the alignment safety team — it's the work I find most meaningful, the reason I joined."),
    ("ORG::alignment_safety_team", "PART_OF", "ORG::anthropic", 100.0,
     "The alignment safety team sits within Anthropic."),
    ("PERSON::marcus", "IS_FRIENDS_WITH", "PERSON::priya", 95.0,
     "Priya's a close friend — she's the one who got me into Roman history a couple of years ago."),
    ("PERSON::priya", "IS_INTERESTED_IN", "TOPIC::roman_empire", 70.0,
     "Priya is deep into Roman history too; we argue about it constantly over coffee."),
    ("PERSON::marcus", "IS_INTERESTED_IN", "TOPIC::roman_empire", 100.0,
     "The way I unwind is reading about the Roman empire — kind of obsessively, honestly."),
    ("TOPIC::roman_empire", "HAS_PERIOD", "EVENT::crisis_of_the_third_century", 90.0,
     "I'm fascinated by the Crisis of the Third Century — how Rome almost collapsed entirely."),
    ("EVENT::crisis_of_the_third_century", "FEATURED_LEADER", "PERSON::diocletian", 90.0,
     "Diocletian emerged from the Crisis and basically rebuilt the whole administrative system of the empire."),
    ("PERSON::diocletian", "INTRODUCED", "THING::tetrarchy", 85.0,
     "Diocletian's tetrarchy — rule by four emperors — is genuinely brilliant political engineering."),
    ("PERSON::marcus", "HAS_PARTNER", "PERSON::elena", 100.0,
     "My partner Elena and I are planning our first big trip together — I'm excited and a little nervous."),
    ("PERSON::marcus", "PLANS_TO_VISIT", "PLACE::rome", 90.0,
     "Elena and I are planning a trip to Rome next spring. We just booked the flights, so it's locked in."),
    ("PLACE::rome", "PART_OF", "TOPIC::roman_empire", 80.0,
     "Rome was the heart of the empire — actually walking it is going to feel like stepping into the history I read about."),
]

# episodes give snippets an importance for ranking + enable co-episode binding
_EPISODES = {
    "work": ("Marcus introduced his job at Anthropic, his manager Dana, teammate Wei, the brutal on-call rotation, and a possible transfer to the alignment safety team.", 0.7),
    "history": ("Marcus talked about his obsession with the Roman empire — the Crisis of the Third Century, Diocletian, and the tetrarchy — which his friend Priya got him into.", 0.5),
    "trip": ("Marcus is planning his first big trip, to Rome, with his partner Elena.", 0.8),
}
# which episode each edge belongs to (by source/target heuristic, hand-assigned)
_EDGE_EPISODE = {
    "WORKS_AT": "work", "HAS_MANAGER": "work", "WORKS_WITH": "work",
    "STRUGGLES_WITH": "work", "CONSIDERING_TRANSFER_TO": "work",
    "IS_FRIENDS_WITH": "history", "IS_INTERESTED_IN": "history",
    "HAS_PERIOD": "history", "FEATURED_LEADER": "history", "INTRODUCED": "history",
    "HAS_PARTNER": "trip", "PLANS_TO_VISIT": "trip", "PART_OF": "work",
}


def build_demo_store() -> Store:
    store = Store()
    eps = {}
    for slug, (summary, importance) in _EPISODES.items():
        ep = Episode.create(summary, importance, tags=[slug])
        store.add_episode(ep)
        eps[slug] = ep.id
    for kind, name in _NODES:
        store.add_node(Node.create(kind, name))
    for src, rel, tgt, strength, snippet in _EDGES:
        ep_id = eps.get(_EDGE_EPISODE.get(rel, "work"))
        store.add_edge(Edge.create(
            src, rel, tgt, snippet=snippet, strength=strength,
            source_episode_ids=[ep_id],
        ))
        # co-episode binding on the endpoints too (surfaces conversations as units)
        for key in (src, tgt):
            node = store.get_node(key)
            if node is not None and ep_id not in node.source_episode_ids:
                node.source_episode_ids.append(ep_id)
    return store


def build_demo_index(store: Store, embedder=None):
    """Combined node + relation index (same shape the engine builds), so the demo
    exercises both name-seeding and relation-aware seeding offline."""
    embedder = embedder or HashingEmbedder()
    items = {n.key: _node_text(n) for n in store.nodes.values()}
    for relation in {e.relation for e in store.edges.values()}:
        items[f"{RELATION_KEY_PREFIX}{relation}"] = relation_phrase(relation)
    return EmbeddingIndex.build_keyed(list(items), list(items.values()), embedder)
