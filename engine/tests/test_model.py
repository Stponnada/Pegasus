"""Unit tests for the core data model. Hand-checked expected values throughout."""

import pytest

from ontomem.model import (
    INITIAL_STRENGTH,
    Edge,
    Episode,
    Node,
    canonicalize,
    edge_key,
    node_key,
)

# --- canonicalisation & keys ---------------------------------------------------


def test_canonicalize_basic():
    assert canonicalize("Sarah") == "sarah"
    assert canonicalize("Ontology-Based LLM Memory") == "ontology_based_llm_memory"
    assert canonicalize("  New   Jersey ") == "new_jersey"
    assert canonicalize("Walmart") == "walmart"


def test_node_key_format():
    assert node_key("PERSON", "Sarah") == "PERSON::sarah"
    assert node_key("ORG", "Walmart") == "ORG::walmart"


def test_edge_key_format():
    assert (
        edge_key("PERSON::bill", "WORKS_AT", "ORG::walmart")
        == "PERSON::bill::WORKS_AT::ORG::walmart"
    )


# --- Node ----------------------------------------------------------------------


def test_node_create_populates_fields():
    n = Node.create("PERSON", "Sarah")
    assert n.key == "PERSON::sarah"
    assert n.name == "Sarah"
    assert n.kind == "PERSON"
    assert n.id  # non-empty uuid
    assert n.created_at and n.updated_at
    assert n.confidence == 0.85
    # nodes carry no strength/stability/ttl
    assert not hasattr(n, "strength")
    assert not hasattr(n, "stability")
    assert not hasattr(n, "ttl_days")


def test_node_invalid_kind_raises():
    with pytest.raises(ValueError):
        Node.create("ALIEN", "Zorp")


def test_node_confidence_out_of_range_raises():
    with pytest.raises(ValueError):
        Node.create("PERSON", "Sarah", confidence=1.5)


def test_node_roundtrip():
    n = Node.create("ORG", "Walmart", aliases=["Wal-Mart"], properties={"sector": "retail"})
    assert Node.from_dict(n.to_dict()) == n


# --- Edge ----------------------------------------------------------------------


def test_edge_create_defaults():
    e = Edge.create("PERSON::bill", "WORKS_AT", "ORG::walmart")
    assert e.key == "PERSON::bill::WORKS_AT::ORG::walmart"
    assert e.strength == INITIAL_STRENGTH
    assert e.stability == "stable"
    assert e.cardinality == "one_to_many"
    assert e.ttl_days is None


@pytest.mark.parametrize("rel", ["WORKS_AT", "BORN_IN", "COMPLAINED_ABOUT_BOSS", "KNOWS"])
def test_edge_valid_relations(rel):
    e = Edge.create("PERSON::a", rel, "PERSON::b")
    assert e.relation == rel


@pytest.mark.parametrize("rel", ["works_at", "Works_At", "A_B_C_D_E", "HAS-SPOUSE", "", "WORKS AT"])
def test_edge_invalid_relations_raise(rel):
    with pytest.raises(ValueError):
        Edge.create("PERSON::a", rel, "PERSON::b")


def test_edge_self_relationship_raises():
    with pytest.raises(ValueError):
        Edge.create("PERSON::bill", "KNOWS", "PERSON::bill")


def test_edge_invalid_stability_raises():
    with pytest.raises(ValueError):
        Edge.create("PERSON::a", "KNOWS", "PERSON::b", stability="permanent")


def test_edge_invalid_cardinality_raises():
    with pytest.raises(ValueError):
        Edge.create("PERSON::a", "KNOWS", "PERSON::b", cardinality="many_to_many")


def test_edge_ttl_allowed_only_for_timebound_or_ephemeral():
    # allowed
    Edge.create("PERSON::a", "PLANS_TRIP_TO", "PLACE::paris", stability="ephemeral", ttl_days=3)
    Edge.create("PERSON::a", "ATTENDING", "EVENT::wedding", stability="time_bound", ttl_days=30)
    # rejected
    with pytest.raises(ValueError):
        Edge.create("PERSON::a", "KNOWS", "PERSON::b", stability="stable", ttl_days=10)


def test_edge_roundtrip():
    e = Edge.create(
        "PERSON::bill",
        "COMPLAINED_ABOUT",
        "PERSON::sarah",
        snippet="Bill said his boss Sarah keeps moving deadlines.",
        evidence="keeps moving deadlines",
        cardinality="one_to_many",
    )
    assert Edge.from_dict(e.to_dict()) == e


# --- Episode -------------------------------------------------------------------


def test_episode_create_and_roundtrip():
    ep = Episode.create("User designed the read/write pipeline.", 0.7, tags=["system_design"])
    assert ep.importance == 0.7
    assert ep.tags == ["system_design"]
    assert Episode.from_dict(ep.to_dict()) == ep


def test_episode_importance_out_of_range_raises():
    with pytest.raises(ValueError):
        Episode.create("x", 1.2)
