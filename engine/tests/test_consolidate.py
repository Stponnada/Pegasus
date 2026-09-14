"""Hermetic tests for the write pipeline: Stage 0 assembly, merge planning with
an injected disambiguator, and the deterministic Stage 3 write."""

from ontomem.consolidate import apply_write, assemble_context, plan_merges, plan_supersessions
from ontomem.embeddings import EmbeddingIndex, HashingEmbedder
from ontomem.extractor import ExtractionResult
from ontomem.merge import AUTO_MERGE, NEW
from ontomem.merge_llm import DisambiguationDecision
from ontomem.model import Edge, Episode, Node
from ontomem.retriever import RetrievalConfig
from ontomem.store import Store
from ontomem.supersede import COEXIST, CONTRADICTS, SUPERSEDES, SupersessionDecision


def _extraction(nodes, edges, *, summary="conv", importance=0.5, merge_keys=None):
    return ExtractionResult(
        episode=Episode.create(summary, importance),
        nodes=nodes,
        edges=edges,
        candidate_merge_keys=merge_keys or {},
    )


# stub 2b: never merge (used when we want to test the NEW path deterministically)
def _never_merge(entity, resolution, store, ctx):
    return DisambiguationDecision(entity.key, NEW, None, 0.0)


# stub 2b: always merge to the first candidate at high confidence
def _always_merge(entity, resolution, store, ctx):
    return DisambiguationDecision(entity.key, AUTO_MERGE, resolution.candidates[0], 0.95)


# --- Stage 0 -------------------------------------------------------------------


def test_assemble_context_empty_graph_is_empty():
    store = Store()
    index = EmbeddingIndex.build(store, HashingEmbedder())
    ctx = assemble_context([{"text": "hi"}], store, index, HashingEmbedder())
    assert ctx == ""


def test_assemble_context_surfaces_anchor_neighbourhood():
    store = Store()
    store.add_node(Node.create("PERSON", "Bill"))
    store.add_node(Node.create("ORG", "Walmart"))
    store.add_edge(Edge.create("PERSON::bill", "WORKS_AT", "ORG::walmart"))
    emb = HashingEmbedder()
    index = EmbeddingIndex.build(store, emb)
    ctx = assemble_context([{"role": "user", "text": "How is Walmart treating you?"}], store, index, emb)
    assert "ORG::walmart" in ctx
    assert "PERSON::bill" in ctx  # surfaced via 2-hop neighbourhood


# --- merge planning ------------------------------------------------------------


def test_plan_auto_merges_exact_existing():
    store = Store()
    store.add_node(Node.create("PERSON", "Bill"))
    extraction = _extraction([Node.create("PERSON", "Bill")], [])
    plan = plan_merges(extraction, store, _never_merge)
    assert plan.remap == {"PERSON::bill": "PERSON::bill"}


def test_plan_new_entity_has_no_remap():
    store = Store()
    extraction = _extraction([Node.create("PERSON", "Zara")], [])
    plan = plan_merges(extraction, store, _never_merge)
    assert "PERSON::zara" not in plan.remap


def test_plan_review_escalates_to_disambiguator():
    store = Store()
    store.add_node(Node.create("PERSON", "Elon Musk"))
    extraction = _extraction([Node.create("PERSON", "Elon")], [])  # token-overlap -> review
    plan = plan_merges(extraction, store, _always_merge)
    assert plan.remap["PERSON::elon"] == "PERSON::elon_musk"


def test_plan_flags_low_band_merge():
    store = Store()
    store.add_node(Node.create("PERSON", "Elon Musk"))

    def flagged_merge(entity, resolution, store, ctx):
        return DisambiguationDecision(entity.key, AUTO_MERGE, resolution.candidates[0], 0.80, flagged=True)

    extraction = _extraction([Node.create("PERSON", "Elon")], [])
    plan = plan_merges(extraction, store, flagged_merge)
    assert len(plan.flagged) == 1
    assert plan.flagged[0]["confidence"] == 0.80


def test_plan_uses_extractor_merge_hint():
    store = Store()
    store.add_node(Node.create("PERSON", "William"))
    # 2a would say NEW (no name relation), but the extractor hinted a match
    extraction = _extraction(
        [Node.create("PERSON", "Bilbo")], [], merge_keys={"PERSON::bilbo": "PERSON::william"}
    )
    plan = plan_merges(extraction, store, _always_merge)
    assert plan.remap["PERSON::bilbo"] == "PERSON::william"


# --- Stage 3 deterministic write ----------------------------------------------


def test_write_creates_new_nodes_and_edges():
    store = Store()
    extraction = _extraction(
        [Node.create("PERSON", "Bill"), Node.create("ORG", "Walmart")],
        [Edge.create("PERSON::bill", "WORKS_AT", "ORG::walmart", snippet="Bill works at Walmart, three years.")],
    )
    plan = plan_merges(extraction, store, _never_merge)
    result = apply_write(extraction, plan, store)
    assert result.nodes_created == 2
    assert result.edges_created == 1
    assert store.get_edge("PERSON::bill::WORKS_AT::ORG::walmart") is not None
    # episode recorded and bound
    assert store.get_node("PERSON::bill").source_episode_ids == [extraction.episode.id]


def test_write_flags_orphan_entity():
    # an entity extracted with one connected node and one unconnected node:
    # only the unconnected one is unreachable -> warn (but never drop)
    store = Store()
    extraction = _extraction(
        [Node.create("PERSON", "Bill"), Node.create("ORG", "Walmart"), Node.create("EVENT", "Crisis")],
        [Edge.create("PERSON::bill", "WORKS_AT", "ORG::walmart")],
    )
    plan = plan_merges(extraction, store, _never_merge)
    result = apply_write(extraction, plan, store)
    assert result.nodes_created == 3
    orphan_warnings = [w for w in result.warnings if "orphan entity" in w]
    assert any("EVENT::crisis" in w for w in orphan_warnings)
    # Bill and Walmart are connected -> not flagged
    assert not any("PERSON::bill" in w or "ORG::walmart" in w for w in orphan_warnings)
    # but the node is NOT dropped
    assert store.get_node("EVENT::crisis") is not None


def test_write_does_not_flag_connected_entities():
    store = Store()
    extraction = _extraction(
        [Node.create("PERSON", "Bill"), Node.create("ORG", "Walmart")],
        [Edge.create("PERSON::bill", "WORKS_AT", "ORG::walmart", snippet="Bill works at Walmart, three years.")],
    )
    plan = plan_merges(extraction, store, _never_merge)
    result = apply_write(extraction, plan, store)
    assert not any("orphan entity" in w for w in result.warnings)


# --- supersession (§9.2c) ------------------------------------------------------


def _store_with_edge(relation):
    store = Store()
    store.add_node(Node.create("PERSON", "Marcus"))
    store.add_node(Node.create("ORG", "SafetyTeam"))
    store.add_edge(Edge.create("PERSON::marcus", relation, "ORG::safetyteam", snippet="was weighing it"))
    return store


def _transfer_extraction():
    return _extraction(
        [Node.create("PERSON", "Marcus"), Node.create("ORG", "SafetyTeam")],
        [Edge.create("PERSON::marcus", "TRANSFERS_TO", "ORG::safetyteam", snippet="going for it")],
    )


def _supersede_stub(decision):
    return lambda old, new, store, ctx: SupersessionDecision(old.key, new.relation, decision, "r")


def test_plan_supersessions_detects_same_endpoint_different_relation():
    store = _store_with_edge("CONSIDERING_TRANSFER_TO")
    ext = _transfer_extraction()
    plan = plan_merges(ext, store, _never_merge)
    sups = plan_supersessions(ext, plan, store, _supersede_stub(SUPERSEDES))
    assert len(sups) == 1
    assert sups[0].old_edge_key == "PERSON::marcus::CONSIDERING_TRANSFER_TO::ORG::safetyteam"


def test_plan_supersessions_coexist_yields_no_action():
    store = _store_with_edge("CONSIDERING_TRANSFER_TO")
    ext = _transfer_extraction()
    plan = plan_merges(ext, store, _never_merge)
    assert plan_supersessions(ext, plan, store, _supersede_stub(COEXIST)) == []


def test_plan_supersessions_ignores_same_relation_rementions():
    store = _store_with_edge("TRANSFERS_TO")  # same relation as the new edge
    ext = _transfer_extraction()
    plan = plan_merges(ext, store, _never_merge)
    # a re-mention is reinforcement, not supersession; stub must not even fire
    def _boom(old, new, store, ctx):
        raise AssertionError("supersede_fn called for a same-relation re-mention")
    assert plan_supersessions(ext, plan, store, _boom) == []


def _managed_by_extraction(cardinality="one_to_one"):
    return _extraction(
        [Node.create("PERSON", "Renee"), Node.create("PERSON", "Desmond")],
        [Edge.create("PERSON::renee", "MANAGED_BY", "PERSON::desmond", snippet="new manager", cardinality=cardinality)],
    )


def _store_with_managed_by(cardinality="one_to_one"):
    store = Store()
    store.add_node(Node.create("PERSON", "Renee"))
    store.add_node(Node.create("PERSON", "Priya"))
    store.add_edge(Edge.create("PERSON::renee", "MANAGED_BY", "PERSON::priya", snippet="old manager", cardinality=cardinality))
    return store


def test_plan_supersessions_detects_same_relation_different_target_when_one_to_one():
    store = _store_with_managed_by("one_to_one")
    ext = _managed_by_extraction("one_to_one")
    plan = plan_merges(ext, store, _never_merge)
    sups = plan_supersessions(ext, plan, store, _supersede_stub(SUPERSEDES))
    assert len(sups) == 1
    assert sups[0].old_edge_key == "PERSON::renee::MANAGED_BY::PERSON::priya"


def test_plan_supersessions_ignores_same_relation_different_target_when_one_to_many():
    # both edges one_to_many (e.g. HAS_FRIEND priya, HAS_FRIEND desmond) -- a second
    # target is a genuinely independent fact, not a replacement; stub must not fire
    store = _store_with_managed_by("one_to_many")
    ext = _managed_by_extraction("one_to_many")
    plan = plan_merges(ext, store, _never_merge)
    def _boom(old, new, store, ctx):
        raise AssertionError("supersede_fn called for a one_to_many relation with a new target")
    assert plan_supersessions(ext, plan, store, _boom) == []


def test_apply_write_demotes_superseded_edge_without_deleting():
    store = _store_with_edge("CONSIDERING_TRANSFER_TO")
    ext = _transfer_extraction()
    plan = plan_merges(ext, store, _never_merge)
    sups = plan_supersessions(ext, plan, store, _supersede_stub(SUPERSEDES))
    result = apply_write(ext, plan, store, supersessions=sups, config=RetrievalConfig())

    old = store.get_edge("PERSON::marcus::CONSIDERING_TRANSFER_TO::ORG::safetyteam")
    assert old is not None  # NOT deleted — non-destructive
    assert old.strength == 1.0  # demoted below dormancy
    assert old.properties.get("superseded_by") == "TRANSFERS_TO"
    assert result.edges_superseded == 1
    new = store.get_edge("PERSON::marcus::TRANSFERS_TO::ORG::safetyteam")
    assert new is not None and new.strength == 100.0  # current edge at full strength


def test_apply_write_contradiction_is_flagged():
    store = _store_with_edge("WORKS_AT")
    ext = _transfer_extraction()
    plan = plan_merges(ext, store, _never_merge)
    sups = plan_supersessions(ext, plan, store, _supersede_stub(CONTRADICTS))
    result = apply_write(ext, plan, store, supersessions=sups, config=RetrievalConfig())
    assert any(f.get("kind") == "contradiction" for f in result.flagged)
    assert store.get_edge("PERSON::marcus::WORKS_AT::ORG::safetyteam").strength == 1.0


def test_write_merges_into_existing_node_and_keeps_alias():
    store = Store()
    store.add_node(Node.create("PERSON", "Samuel"))
    extraction = _extraction([Node.create("PERSON", "Sam")], [])  # abbreviation auto-merge
    plan = plan_merges(extraction, store, _never_merge)
    result = apply_write(extraction, plan, store)
    assert result.nodes_merged == 1
    assert result.nodes_created == 0
    assert store.get_node("PERSON::sam") is None  # not created separately
    assert "Sam" in store.get_node("PERSON::samuel").aliases  # surface form retained


def test_write_reinforces_existing_edge_and_keeps_strength_capped():
    store = Store()
    store.add_node(Node.create("PERSON", "Bill"))
    store.add_node(Node.create("ORG", "Walmart"))
    store.add_edge(Edge.create("PERSON::bill", "WORKS_AT", "ORG::walmart"))
    store.update_edge("PERSON::bill::WORKS_AT::ORG::walmart", strength=50.0, confidence=0.6)

    extraction = _extraction(
        [Node.create("PERSON", "Bill"), Node.create("ORG", "Walmart")],
        [Edge.create("PERSON::bill", "WORKS_AT", "ORG::walmart", confidence=0.9, snippet="updated snippet")],
    )
    plan = plan_merges(extraction, store, _never_merge)
    result = apply_write(extraction, plan, store)
    edge = store.get_edge("PERSON::bill::WORKS_AT::ORG::walmart")
    assert result.edges_reinforced == 1
    assert edge.strength == 65.0  # 50 + 15 boost
    assert edge.snippet == "updated snippet"  # higher confidence -> snippet refreshed


def test_write_drops_edge_when_endpoints_merge_together():
    # Bill and Billy merge to the same node -> a Bill-KNOWS-Billy edge self-collapses
    store = Store()
    store.add_node(Node.create("PERSON", "William"))
    extraction = _extraction(
        [Node.create("PERSON", "Bill"), Node.create("PERSON", "Billy")],
        [Edge.create("PERSON::bill", "KNOWS", "PERSON::billy")],
    )
    # both Bill and Billy abbreviation-map to William (unique) -> auto-merge
    plan = plan_merges(extraction, store, _never_merge)
    result = apply_write(extraction, plan, store)
    assert result.edges_dropped == 1
    assert result.edges_created == 0


def test_write_applies_deferred_read_reinforcement():
    store = Store()
    store.add_node(Node.create("PERSON", "Bill"))
    store.add_node(Node.create("ORG", "Walmart"))
    store.add_edge(Edge.create("PERSON::bill", "WORKS_AT", "ORG::walmart"))
    store.update_edge("PERSON::bill::WORKS_AT::ORG::walmart", strength=40.0)
    extraction = _extraction([], [])  # nothing new extracted this turn
    plan = plan_merges(extraction, store, _never_merge)
    result = apply_write(
        extraction, plan, store,
        reinforcement_edge_keys=["PERSON::bill::WORKS_AT::ORG::walmart"],
    )
    assert result.reinforced_from_read == 1
    assert store.get_edge("PERSON::bill::WORKS_AT::ORG::walmart").strength == 55.0


def test_write_journals_when_path_given(tmp_path):
    from ontomem.journal import read_events

    store = Store()
    extraction = _extraction([Node.create("PERSON", "Bill")], [])
    plan = plan_merges(extraction, store, _never_merge)
    journal = tmp_path / "journal.jsonl"
    apply_write(extraction, plan, store, journal_path=journal)
    events = read_events(journal)
    assert events[0]["event"] == "write"
    assert events[0]["nodes_created"] == 1


def test_write_journal_persists_orphan_warnings(tmp_path):
    # Found via dogfooding: _flag_orphans populates result.warnings correctly
    # (returned live in the HTTP /write response) but the journal write was
    # missing the "warnings" key entirely, silently losing this observability
    # signal for good the moment the response was discarded -- exactly what a
    # fire-and-forget background write does (see ontomem-plugin.ts). The
    # journal is the ONLY durable record of a write; anything not in it is
    # gone.
    from ontomem.journal import read_events

    store = Store()
    extraction = _extraction(
        [Node.create("PERSON", "Bill"), Node.create("EVENT", "Crisis")],
        [],
    )
    journal = tmp_path / "journal.jsonl"
    plan = plan_merges(extraction, store, _never_merge)
    apply_write(extraction, plan, store, journal_path=journal)
    events = read_events(journal)
    assert any("orphan entity" in w and "EVENT::crisis" in w for w in events[0]["warnings"])


# --- self-node designation (end-to-end across two separate writes/episodes,
# reproducing the exact dogfooding failure: a later conversation that doesn't
# restate the user's name must attach to the SAME self node, not a fresh
# "User" placeholder) --------------------------------------------------------


def test_first_write_designates_self_key_from_first_relationship_source():
    store = Store()
    maya = Node.create("PERSON", "Maya")
    derek = Node.create("PERSON", "Derek Osei")
    extraction = _extraction(
        [maya, derek],
        [Edge.create(maya.key, "HAS_MANAGER", derek.key, snippet="My manager Derek keeps changing things.")],
    )
    plan = plan_merges(extraction, store, _never_merge)
    apply_write(extraction, plan, store)
    assert store.self_key == "PERSON::maya"


def test_later_conversation_self_placeholder_attaches_to_designated_self_node():
    store = Store()
    maya = Node.create("PERSON", "Maya")
    derek = Node.create("PERSON", "Derek Osei")
    plan1 = plan_merges(
        _extraction([maya, derek], [Edge.create(maya.key, "HAS_MANAGER", derek.key, snippet="s1 s1 s1 s1")]),
        store, _never_merge,
    )
    apply_write(
        _extraction([maya, derek], [Edge.create(maya.key, "HAS_MANAGER", derek.key, snippet="s1 s1 s1 s1")]),
        plan1, store,
    )

    # Second, unrelated conversation never says "Maya" -- extractor falls back
    # to a generic placeholder, wrongly typed OTHER (exactly what dogfooding saw).
    user_placeholder = Node.create("OTHER", "User")
    liam = Node.create("PERSON", "Liam Fitzgerald")
    extraction2 = _extraction(
        [user_placeholder, liam],
        [Edge.create(user_placeholder.key, "IS_CLIMBING_WITH", liam.key, snippet="Liam and I climb together often.")],
    )
    plan2 = plan_merges(extraction2, store, _never_merge)
    assert plan2.remap[user_placeholder.key] == "PERSON::maya"

    result2 = apply_write(extraction2, plan2, store)
    assert "OTHER::user" not in store.nodes
    assert result2.nodes_merged == 1  # the placeholder merged away, only Liam is new
    edge = store.get_edge("PERSON::maya::IS_CLIMBING_WITH::PERSON::liam_fitzgerald")
    assert edge is not None
    assert edge.source_key == "PERSON::maya"
