"""Unit tests for the retriever. Activation values are hand-derived from the
spec formula: contribution = source_activation * (strength/100) * hop_decay^hop.
"""

import pytest

from ontomem.embeddings import EmbeddingIndex, HashingEmbedder
from ontomem.model import Edge, Episode, Node
from ontomem.embeddings import relation_phrase
from ontomem.retriever import (
    FORWARD,
    RELATION_KEY_PREFIX,
    REVERSE,
    SEED,
    Activation,
    RetrievalConfig,
    _gate_noun_hits,
    apply_co_episode_boost,
    extract_nouns,
    find_seeds,
    format_context,
    read,
    resolve_seed_hits,
    select_within_budget,
    spreading_activation,
)
from ontomem.store import Store


# --- noun extraction -----------------------------------------------------------


def test_extract_nouns_drops_function_words():
    nouns = extract_nouns("Can you help me think through how to talk to my boss about the salary situation")
    assert "boss" in nouns and "salary" in nouns and "situation" in nouns
    for fw in ("you", "me", "to", "my", "the", "about", "how"):
        assert fw not in nouns


def test_extract_nouns_dedupes_preserving_order():
    assert extract_nouns("Walmart walmart WALMART") == ["walmart"]


# --- spreading activation (hand-computed) --------------------------------------


def _chain() -> Store:
    # seed Bill -WORKS_AT-> Walmart -OWNED_BY-> WaltonFamily   (a directed chain)
    store = Store()
    for kind, name in [("PERSON", "Bill"), ("ORG", "Walmart"), ("ORG", "WaltonFamily")]:
        store.add_node(Node.create(kind, name))
    store.add_edge(Edge.create("PERSON::bill", "WORKS_AT", "ORG::walmart"))
    store.add_edge(Edge.create("ORG::walmart", "OWNED_BY", "ORG::waltonfamily"))
    return store


def test_activation_values_match_formula():
    store = _chain()
    cfg = RetrievalConfig()
    acts = spreading_activation(store, ["PERSON::bill"], cfg)["activations"]
    # Degree-normalised: contrib = act * (edge.strength / total_incident) * hop_decay.
    # hop1: Bill has 1 incident edge (S=100) -> Walmart = 1.0*(100/100)*0.6 = 0.6
    assert acts["ORG::walmart"].score == pytest.approx(0.6)
    assert acts["ORG::walmart"].direction == FORWARD
    assert acts["ORG::walmart"].hop == 1
    # hop2: Walmart has 2 incident edges (S=200) -> WaltonFamily = 0.6*(100/200)*0.6 = 0.18
    assert acts["ORG::waltonfamily"].score == pytest.approx(0.18)
    assert acts["ORG::waltonfamily"].hop == 2
    # seed stays at >= 1.0 and is tagged SEED
    assert acts["PERSON::bill"].direction == SEED
    assert acts["PERSON::bill"].score >= 1.0


def test_reverse_traversal_surfaces_incoming_neighbour():
    # seed Walmart should surface Bill via the incoming WORKS_AT edge (reverse)
    store = _chain()
    acts = spreading_activation(store, ["ORG::walmart"], RetrievalConfig())["activations"]
    assert "PERSON::bill" in acts
    assert acts["PERSON::bill"].direction == REVERSE
    # Walmart has 2 incident edges (S=200) -> Bill = 1.0*(100/200)*0.6 = 0.3
    assert acts["PERSON::bill"].score == pytest.approx(0.3)


def test_hub_does_not_radiate_above_threshold_to_all_spokes():
    # A user hub with many spokes, reached via two converging seeds. Degree
    # normalisation must keep unrelated spokes BELOW the primary threshold so the
    # whole life-history doesn't dump into every query (spec §1.1 / §9.2).
    store = Store()
    store.add_node(Node.create("PERSON", "Marcus"))
    spokes = ["Anthropic", "Dana", "Wei", "Priya", "Elena", "Rome", "RomanEmpire"]
    for name in spokes:
        store.add_node(Node.create("THING", name))
        store.add_edge(Edge.create("PERSON::marcus", "REL_TO", f"THING::{name.lower()}"))
    cfg = RetrievalConfig()
    # two seeds (Rome, RomanEmpire) converge on Marcus
    acts = spreading_activation(store, ["THING::rome", "THING::romanempire"], cfg)["activations"]
    fired = {k for k, a in acts.items() if a.score >= cfg.primary_threshold}
    # Marcus and the two seeds fire; the unrelated work spokes must NOT
    assert "PERSON::marcus" in fired
    assert "THING::anthropic" not in fired
    assert "THING::dana" not in fired
    assert "THING::wei" not in fired


def test_multi_source_convergence_sums():
    # two seeds both point at the same target -> contributions add
    store = Store()
    for name in ("A", "B", "C"):
        store.add_node(Node.create("THING", name))
    store.add_edge(Edge.create("THING::a", "LINKS_TO", "THING::c"))
    store.add_edge(Edge.create("THING::b", "LINKS_TO", "THING::c"))
    acts = spreading_activation(store, ["THING::a", "THING::b"], RetrievalConfig())["activations"]
    # C = 0.6 (from A) + 0.6 (from B) = 1.2
    assert acts["THING::c"].score == pytest.approx(1.2)


def test_traversed_edges_recorded():
    store = _chain()
    out = spreading_activation(store, ["PERSON::bill"], RetrievalConfig())
    assert "PERSON::bill::WORKS_AT::ORG::walmart" in out["traversed_edge_keys"]


def test_event_trace_is_ordered_and_behaviour_preserving():
    store = _chain()
    cfg = RetrievalConfig()
    baseline = spreading_activation(store, ["PERSON::bill"], cfg)["activations"]
    events: list = []
    traced = spreading_activation(store, ["PERSON::bill"], cfg, events=events)["activations"]
    # activations identical whether or not we trace
    assert {k: round(a.score, 6) for k, a in traced.items()} == {k: round(a.score, 6) for k, a in baseline.items()}
    # first event is the seed; a crossing to Walmart is recorded with its contribution
    assert events[0] == {"type": "seed", "node": "PERSON::bill", "name": "Bill", "activation": 1.0}
    cross = next(e for e in events if e["type"] == "cross" and e["reached"] == "ORG::walmart")
    assert cross["relation"] == "WORKS_AT"
    assert cross["contribution"] == pytest.approx(0.6)


# --- co-episode boost ----------------------------------------------------------


def test_co_episode_boost_lifts_shared_episode_node():
    store = Store()
    store.add_node(Node.create("PERSON", "Bill", source_episode_ids=["ep1"]))
    store.add_node(Node.create("THING", "Guitar", source_episode_ids=["ep1"]))
    cfg = RetrievalConfig()
    # Bill fired as a seed; Guitar shares ep1 but was not reached by BFS
    acts = {"PERSON::bill": Activation("PERSON::bill", 1.0, SEED, 0)}
    boosted = apply_co_episode_boost(acts, store, cfg)
    assert "THING::guitar" in boosted
    assert boosted["THING::guitar"].co_episode is True
    assert boosted["THING::guitar"].score == pytest.approx(cfg.co_episode_boost)


# --- budget & directionality ---------------------------------------------------


def test_primary_threshold_filters():
    cfg = RetrievalConfig()
    acts = {
        "A": Activation("A", 0.5, FORWARD, 1),
        "B": Activation("B", 0.2, FORWARD, 1),  # below 0.30
    }
    assert select_within_budget(acts, cfg) == ["A"]


def test_budget_caps_and_ranks_by_activation():
    cfg = RetrievalConfig(node_budget=3)
    acts = {k: Activation(k, score, FORWARD, 1) for k, score in
            [("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6)]}
    assert select_within_budget(acts, cfg) == ["A", "B", "C"]


def test_asymmetric_budget_reserves_reverse_slots():
    cfg = RetrievalConfig(node_budget=10, forward_ratio=0.7)  # 7 fwd / 3 rev
    acts = {}
    for i in range(9):
        acts[f"F{i}"] = Activation(f"F{i}", 0.9 - i * 0.01, FORWARD, 1)
    for i in range(3):
        acts[f"R{i}"] = Activation(f"R{i}", 0.4 - i * 0.01, REVERSE, 1)
    chosen = select_within_budget(acts, cfg)
    # despite 9 stronger forward nodes, reverse nodes are not all crowded out
    assert any(k.startswith("R") for k in chosen)
    assert len([k for k in chosen if k.startswith("F")]) <= 7 + 0  # forward capped at its share
    assert len(chosen) == 10


# --- tiered formatting ---------------------------------------------------------


def _bill_graph_with_snippet() -> Store:
    store = Store()
    ep = Episode.create("Bill vented about his boss.", 0.8)
    store.add_episode(ep)
    store.add_node(Node.create("PERSON", "Bill"))
    store.add_node(Node.create("ORG", "Walmart"))
    store.add_edge(
        Edge.create(
            "PERSON::bill", "WORKS_AT", "ORG::walmart",
            snippet="Bill said he has worked at Walmart for three years and finds it draining.",
            source_episode_ids=[ep.id],
        )
    )
    return store


def test_format_memory_renders_edges_between_cofired_nodes():
    store = _bill_graph_with_snippet()
    acts = {
        "PERSON::bill": Activation("PERSON::bill", 0.5, SEED, 0),
        "ORG::walmart": Activation("ORG::walmart", 0.5, FORWARD, 1),
    }
    out = format_context(store, ["PERSON::bill", "ORG::walmart"], acts, RetrievalConfig())
    assert "Bill -[WORKS_AT]-> Walmart" in out["memory_block"]
    assert out["context_block"] == ""  # below secondary threshold -> no snippet


def test_hub_does_not_flood_when_spokes_did_not_fire():
    # a hub with many edges, reached weakly; only the co-fired spoke should show
    store = Store()
    store.add_node(Node.create("PERSON", "Marcus"))
    for org in ("Anthropic", "Walmart", "Costco", "Google"):
        store.add_node(Node.create("ORG", org))
        store.add_edge(Edge.create("PERSON::marcus", "WORKS_AT" if org == "Anthropic" else "KNOWS_OF", f"ORG::{org.lower()}"))
    # only Marcus and Anthropic fired
    acts = {
        "PERSON::marcus": Activation("PERSON::marcus", 0.6, REVERSE, 1),
        "ORG::anthropic": Activation("ORG::anthropic", 1.0, SEED, 0),
    }
    out = format_context(store, ["PERSON::marcus", "ORG::anthropic"], acts, RetrievalConfig())
    assert "Marcus -[WORKS_AT]-> Anthropic" in out["memory_block"]
    # the three non-fired spokes must NOT appear
    assert "Walmart" not in out["memory_block"]
    assert "Costco" not in out["memory_block"]
    assert "Google" not in out["memory_block"]


def test_format_context_hides_superseded_edge():
    # a demoted (superseded) edge must not render even when both endpoints fired
    store = Store()
    store.add_node(Node.create("PERSON", "Marcus"))
    store.add_node(Node.create("ORG", "Team"))
    store.add_edge(Edge.create("PERSON::marcus", "TRANSFERS_TO", "ORG::team"))
    stale = Edge.create("PERSON::marcus", "CONSIDERING_TRANSFER_TO", "ORG::team")
    stale.properties["superseded_by"] = "TRANSFERS_TO"
    store.add_edge(stale)
    acts = {
        "PERSON::marcus": Activation("PERSON::marcus", 0.9, SEED, 0),
        "ORG::team": Activation("ORG::team", 0.9, FORWARD, 1),
    }
    out = format_context(store, ["PERSON::marcus", "ORG::team"], acts, RetrievalConfig())
    assert "TRANSFERS_TO" in out["memory_block"]
    assert "CONSIDERING_TRANSFER_TO" not in out["memory_block"]


def test_format_context_dedupes_shared_snippet():
    # both endpoints of one edge select that edge's snippet -> inject it once only
    store = _bill_graph_with_snippet()
    acts = {
        "PERSON::bill": Activation("PERSON::bill", 0.9, SEED, 0),
        "ORG::walmart": Activation("ORG::walmart", 0.9, FORWARD, 1),
    }
    out = format_context(store, ["PERSON::bill", "ORG::walmart"], acts, RetrievalConfig())
    assert out["context_block"].count("three years") == 1


def test_format_context_injects_snippet_above_secondary():
    store = _bill_graph_with_snippet()
    acts = {"PERSON::bill": Activation("PERSON::bill", 0.9, SEED, 0)}
    out = format_context(store, ["PERSON::bill"], acts, RetrievalConfig())
    assert "three years" in out["context_block"]
    assert out["context_block"].startswith("[CONTEXT]")


# --- edge-centric snippet injection (spec §3.2 design decision: snippets load
# per firing EDGE, not one "best" edge picked for the whole node) --------------


def _multi_relationship_node() -> Store:
    # One hot node (Jordan) with several relationships of differing strength/
    # importance. Reproduces the exact dogfooding bug report: a single
    # "highest-scored" pick could surface an unrelated relationship's snippet
    # (e.g. WORKS_AS) instead of the one the user actually asked about.
    store = Store()
    store.add_node(Node.create("PERSON", "User"))
    store.add_node(Node.create("PERSON", "Jordan"))
    store.add_node(Node.create("OTHER", "Architect"))
    store.add_node(Node.create("PERSON", "Nate"))
    # highest strength/importance of the three -- would have "won" under the
    # old single-best-edge selection despite being irrelevant to a trip query
    ep_work = Episode.create("Jordan's job", 0.9)
    store.add_episode(ep_work)
    store.add_edge(Edge.create(
        "PERSON::jordan", "WORKS_AS", "OTHER::architect",
        snippet="Jordan works as an architect and has been incredibly busy lately.",
        source_episode_ids=[ep_work.id], strength=100.0,
    ))
    ep_trip = Episode.create("planning the trip", 0.5)
    store.add_episode(ep_trip)
    store.add_edge(Edge.create(
        "PERSON::user", "IS_PLANNING_TRIP_WITH", "PERSON::jordan",
        snippet="Jordan and I are finally booking the Japan trip we've talked about for two years.",
        source_episode_ids=[ep_trip.id], strength=90.0,
    ))
    store.add_edge(Edge.create(
        "PERSON::jordan", "IS_FRIENDS_WITH", "PERSON::nate",
        snippet="Jordan's best friend Nate knows about my plan and has been helping keep it a secret.",
        source_episode_ids=[ep_trip.id], strength=80.0,
    ))
    return store


def test_hot_node_injects_all_its_snippets_not_just_the_highest_scored():
    store = _multi_relationship_node()
    acts = {"PERSON::jordan": Activation("PERSON::jordan", 0.9, SEED, 0)}
    out = format_context(store, ["PERSON::jordan"], acts, RetrievalConfig())
    # previously only the WORKS_AS snippet (highest strength*importance) would
    # have been shown; now every one of Jordan's relationships is present
    assert "architect" in out["context_block"]
    assert "Japan trip" in out["context_block"]
    assert "Nate" in out["context_block"]


def test_node_snippets_capped_per_node():
    store = Store()
    store.add_node(Node.create("PERSON", "Hub"))
    for i in range(6):
        store.add_node(Node.create("PERSON", f"Person{i}"))
        store.add_edge(Edge.create(
            "PERSON::hub", "KNOWS", f"PERSON::person{i}",
            snippet=f"Hub talked about person {i} for a while in detail.",
            strength=float(90 - i),
        ))
    acts = {"PERSON::hub": Activation("PERSON::hub", 0.9, SEED, 0)}
    out = format_context(store, ["PERSON::hub"], acts, RetrievalConfig(max_snippets_per_node=4))
    assert out["context_block"].count("talked about person") == 4
    # the highest-strength edges are the ones kept
    assert "person 0" in out["context_block"]
    assert "person 3" in out["context_block"]
    assert "person 5" not in out["context_block"]


def test_format_context_injections_are_structured_per_relationship():
    store = _multi_relationship_node()
    acts = {"PERSON::jordan": Activation("PERSON::jordan", 0.9, SEED, 0)}
    out = format_context(store, ["PERSON::jordan"], acts, RetrievalConfig())
    by_relation = {i["relation"]: i for i in out["injections"]}
    assert by_relation["WORKS_AS"]["relation_phrase"] == "works as"
    assert by_relation["WORKS_AS"]["target"] == "Architect"
    assert "architect" in by_relation["WORKS_AS"]["snippet"]
    assert by_relation["IS_FRIENDS_WITH"]["source"] == "Jordan"
    assert by_relation["IS_FRIENDS_WITH"]["target"] == "Nate"


# --- relation-aware seeding ----------------------------------------------------


def test_resolve_seed_hits_maps_relation_to_endpoints():
    store = Store()
    store.add_node(Node.create("PERSON", "Marcus"))
    store.add_node(Node.create("PERSON", "Elena"))
    store.add_edge(Edge.create("PERSON::marcus", "HAS_PARTNER", "PERSON::elena"))
    out = resolve_seed_hits([f"{RELATION_KEY_PREFIX}HAS_PARTNER"], store)
    assert set(out) == {"PERSON::marcus", "PERSON::elena"}


def test_resolve_seed_hits_passes_nodes_and_drops_stale():
    store = Store()
    store.add_node(Node.create("PERSON", "Marcus"))
    out = resolve_seed_hits(["PERSON::marcus", "PERSON::ghost", f"{RELATION_KEY_PREFIX}NOPE"], store)
    assert out == ["PERSON::marcus"]


def test_margin_gate_accepts_separated_winner_below_absolute():
    cfg = RetrievalConfig()
    # 0.26 < 0.30 absolute, but 0.20 clear of the runner-up -> seed it ('partner')
    assert _gate_noun_hits([("REL::HAS_PARTNER", 0.26), ("X", 0.06), ("Y", 0.02)], cfg) == ["REL::HAS_PARTNER"]


def test_margin_gate_rejects_low_ambiguous_cluster():
    cfg = RetrievalConfig()
    # abstract noise: low and tightly clustered ('career', 'decision') -> nothing
    assert _gate_noun_hits([("A", 0.11), ("B", 0.09), ("C", 0.07)], cfg) == []


def test_absolute_threshold_still_passes_strong_hits():
    cfg = RetrievalConfig()
    assert _gate_noun_hits([("DIOCLETIAN", 1.0), ("X", 0.05)], cfg) == ["DIOCLETIAN"]


def test_margin_gate_below_floor_rejected_despite_gap():
    cfg = RetrievalConfig()
    # big relative gap but top is beneath the floor -> still rejected
    assert _gate_noun_hits([("A", 0.10), ("B", 0.0)], cfg) == []


def _combined_index(store, emb):
    inputs = {n.key: " ".join([n.name, *n.aliases]) for n in store.nodes.values()}
    for rel in {e.relation for e in store.edges.values()}:
        inputs[f"{RELATION_KEY_PREFIX}{rel}"] = relation_phrase(rel)
    return EmbeddingIndex.build_keyed(list(inputs), list(inputs.values()), emb)


def test_relation_seeding_surfaces_endpoint_by_relation_not_name():
    # 'partner' matches no node NAME; it must seed via the HAS_PARTNER relation
    # and surface Elena — the spec §5.2 'boss -> Sarah' capability.
    store = Store()
    store.add_node(Node.create("PERSON", "Marcus"))
    store.add_node(Node.create("PERSON", "Elena"))
    store.add_node(Node.create("PERSON", "Dana"))
    store.add_edge(Edge.create("PERSON::marcus", "HAS_PARTNER", "PERSON::elena",
                               snippet="Marcus said his partner Elena is travelling with him."))
    store.add_edge(Edge.create("PERSON::marcus", "HAS_MANAGER", "PERSON::dana"))
    # realistic hub: extra spokes so degree-normalisation keeps the user node from
    # re-radiating to unrelated neighbours (the real operating condition)
    for name in ("Wei", "Priya", "Anthropic"):
        store.add_node(Node.create("PERSON", name))
        store.add_edge(Edge.create("PERSON::marcus", "KNOWS", f"PERSON::{name.lower()}"))
    emb = HashingEmbedder()
    index = _combined_index(store, emb)
    seeds = find_seeds("what was my partner's name again", index, emb, store, RetrievalConfig())
    assert "PERSON::elena" in seeds
    res = read("what was my partner's name again", store, index, emb, RetrievalConfig())
    assert "Marcus -[HAS_PARTNER]-> Elena" in res.memory_block
    # the unrelated manager relation must NOT surface for a 'partner' query
    assert "Dana" not in res.memory_block


# --- end to end (hashing embedder, lexical seeding) ---------------------------


def test_read_end_to_end_lexical():
    store = _bill_graph_with_snippet()
    emb = HashingEmbedder()
    index = EmbeddingIndex.build(store, emb)
    result = read("Tell me about Walmart", store, index, emb, RetrievalConfig())
    assert "ORG::walmart" in result.seeds
    # Bill should surface via reverse traversal from the Walmart seed
    assert "PERSON::bill" in result.activations
    assert "Bill -[WORKS_AT]-> Walmart" in result.memory_block


def test_read_model_mismatch_raises():
    store = _bill_graph_with_snippet()
    index = EmbeddingIndex.build(store, HashingEmbedder(dim=64))
    other = HashingEmbedder(dim=128)  # different model_id
    with pytest.raises(ValueError):
        read("Walmart", store, index, other)
