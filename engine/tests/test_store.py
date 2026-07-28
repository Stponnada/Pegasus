"""Unit tests for the graph store: mutations, wiring, and atomic persistence."""

import pytest

from ontomem.model import Edge, Episode, Node
from ontomem.store import Store


def _bill_at_walmart() -> Store:
    store = Store()
    store.add_node(Node.create("PERSON", "Bill"))
    store.add_node(Node.create("ORG", "Walmart"))
    store.add_edge(Edge.create("PERSON::bill", "WORKS_AT", "ORG::walmart"))
    return store


# --- node mutations ------------------------------------------------------------


def test_add_and_get_node():
    store = Store()
    store.add_node(Node.create("PERSON", "Bill"))
    assert store.get_node("PERSON::bill").name == "Bill"
    assert store.get_node("PERSON::nobody") is None


def test_add_duplicate_node_raises():
    store = Store()
    store.add_node(Node.create("PERSON", "Bill"))
    with pytest.raises(KeyError):
        store.add_node(Node.create("PERSON", "Bill"))


def test_upsert_node_returns_existing_unchanged():
    store = Store()
    first = store.add_node(Node.create("PERSON", "Bill", confidence=0.85))
    # a second node with the same key but different fields must NOT overwrite
    returned = store.upsert_node(Node.create("PERSON", "Bill", confidence=0.10))
    assert returned is first
    assert returned.confidence == 0.85  # unchanged — upsert never mutates


def test_upsert_node_inserts_when_absent():
    store = Store()
    n = store.upsert_node(Node.create("PERSON", "Bill"))
    assert store.get_node("PERSON::bill") is n


def test_update_node_merges_fields_explicitly():
    store = Store()
    store.add_node(Node.create("PERSON", "Bill", aliases=["Billy"]))
    store.update_node(
        "PERSON::bill",
        confidence=0.95,
        add_aliases=["Billy", "William"],  # "Billy" is a dup, must not duplicate
        add_properties={"nationality": "American"},
        add_episode_ids=["ep1", "ep1"],  # dup must collapse
    )
    node = store.get_node("PERSON::bill")
    assert node.confidence == 0.95
    assert node.aliases == ["Billy", "William"]
    assert node.properties == {"nationality": "American"}
    assert node.source_episode_ids == ["ep1"]


def test_update_unknown_node_raises():
    with pytest.raises(KeyError):
        Store().update_node("PERSON::ghost", confidence=0.5)


def test_update_node_rejects_bad_confidence():
    store = Store()
    store.add_node(Node.create("PERSON", "Bill"))
    with pytest.raises(ValueError):
        store.update_node("PERSON::bill", confidence=2.0)


# --- edge mutations & wiring ---------------------------------------------------


def test_add_edge_wires_pointers_both_directions():
    store = _bill_at_walmart()
    edge = store.get_edge("PERSON::bill::WORKS_AT::ORG::walmart")
    bill = store.get_node("PERSON::bill")
    walmart = store.get_node("ORG::walmart")
    # pointer identity, not copies
    assert edge.source is bill
    assert edge.target is walmart
    assert edge in bill.outgoing
    assert edge in walmart.incoming


def test_add_edge_missing_endpoint_raises():
    store = Store()
    store.add_node(Node.create("PERSON", "Bill"))
    with pytest.raises(KeyError):
        store.add_edge(Edge.create("PERSON::bill", "WORKS_AT", "ORG::walmart"))


def test_add_duplicate_edge_raises():
    store = _bill_at_walmart()
    with pytest.raises(KeyError):
        store.add_edge(Edge.create("PERSON::bill", "WORKS_AT", "ORG::walmart"))


def test_upsert_edge_returns_existing_unchanged():
    store = _bill_at_walmart()
    key = "PERSON::bill::WORKS_AT::ORG::walmart"
    first = store.get_edge(key)
    returned = store.upsert_edge(
        Edge.create("PERSON::bill", "WORKS_AT", "ORG::walmart", snippet="new snippet")
    )
    assert returned is first
    assert returned.snippet == ""  # unchanged


def test_update_edge_overwrites_fields():
    store = _bill_at_walmart()
    key = "PERSON::bill::WORKS_AT::ORG::walmart"
    store.update_edge(key, snippet="Bill mentioned he works at Walmart.", confidence=0.9)
    edge = store.get_edge(key)
    assert edge.snippet == "Bill mentioned he works at Walmart."
    assert edge.confidence == 0.9


def test_reinforce_edge_bumps_and_caps():
    store = _bill_at_walmart()
    key = "PERSON::bill::WORKS_AT::ORG::walmart"
    store.update_edge(key, strength=50.0)
    store.reinforce_edge(key, boost=15.0)
    assert store.get_edge(key).strength == 65.0
    store.reinforce_edge(key, boost=100.0)
    assert store.get_edge(key).strength == 100.0  # capped


# --- episodes ------------------------------------------------------------------


def test_add_episode_and_duplicate_raises():
    store = Store()
    ep = Episode.create("Bill talked about work.", 0.5)
    store.add_episode(ep)
    assert store.get_episode(ep.id) is ep
    with pytest.raises(KeyError):
        store.add_episode(ep)


# --- persistence ---------------------------------------------------------------


def test_snapshot_roundtrip_preserves_graph_and_rewires(tmp_path):
    store = _bill_at_walmart()
    ep = Episode.create("Bill works at Walmart.", 0.6)
    store.add_episode(ep)
    path = tmp_path / "graph.json"
    store.save(path)

    loaded = Store.load(path)
    # data preserved
    assert loaded.get_node("PERSON::bill") == store.get_node("PERSON::bill")
    assert loaded.get_edge("PERSON::bill::WORKS_AT::ORG::walmart") == store.get_edge(
        "PERSON::bill::WORKS_AT::ORG::walmart"
    )
    assert loaded.get_episode(ep.id) == ep
    # pointers rehydrated on load
    edge = loaded.get_edge("PERSON::bill::WORKS_AT::ORG::walmart")
    assert edge.source is loaded.get_node("PERSON::bill")
    assert edge.target is loaded.get_node("ORG::walmart")
    assert edge in loaded.get_node("PERSON::bill").outgoing


def test_load_dangling_edge_raises(tmp_path):
    path = tmp_path / "bad.json"
    bad = {
        "version": 1,
        "nodes": [Node.create("PERSON", "Bill").to_dict()],
        "edges": [Edge.create("PERSON::bill", "WORKS_AT", "ORG::walmart").to_dict()],
        "episodes": [],
    }
    import json

    path.write_text(json.dumps(bad))
    with pytest.raises(KeyError):
        Store.load(path)


def test_save_is_atomic_on_replace_failure(tmp_path, monkeypatch):
    path = tmp_path / "graph.json"
    _bill_at_walmart().save(path)
    original = path.read_text()

    # mutate, then make the critical replace step fail mid-save
    store = Store.load(path)
    store.add_node(Node.create("PERSON", "Fred"))

    import ontomem.store as store_mod

    def boom(*_args, **_kwargs):
        raise OSError("simulated crash during replace")

    monkeypatch.setattr(store_mod.os, "replace", boom)
    with pytest.raises(OSError):
        store.save(path)

    # previous snapshot is byte-for-byte intact
    assert path.read_text() == original
    # no leftover temp files
    assert [p.name for p in tmp_path.iterdir() if "tmp" in p.name] == []
