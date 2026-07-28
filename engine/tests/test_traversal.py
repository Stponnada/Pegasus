"""Unit tests for store traversal: neighbourhood BFS and subgraph rendering."""

from ontomem.model import Edge, Node
from ontomem.store import Store


def _chain_store() -> Store:
    # Bill -WORKS_AT-> Walmart ; Bill -IS_FRIENDS_WITH-> Fred -WORKS_AT-> Costco
    store = Store()
    for kind, name in [("PERSON", "Bill"), ("ORG", "Walmart"), ("PERSON", "Fred"), ("ORG", "Costco")]:
        store.add_node(Node.create(kind, name))
    store.add_edge(Edge.create("PERSON::bill", "WORKS_AT", "ORG::walmart"))
    store.add_edge(Edge.create("PERSON::bill", "IS_FRIENDS_WITH", "PERSON::fred"))
    store.add_edge(Edge.create("PERSON::fred", "WORKS_AT", "ORG::costco"))
    return store


def test_neighbourhood_depth_0_is_seed_only():
    store = _chain_store()
    hood = store.neighbourhood("PERSON::bill", depth=0)
    assert {n.key for n in hood["nodes"]} == {"PERSON::bill"}
    assert hood["edges"] == []


def test_neighbourhood_depth_1():
    store = _chain_store()
    hood = store.neighbourhood("PERSON::bill", depth=1)
    assert {n.key for n in hood["nodes"]} == {"PERSON::bill", "ORG::walmart", "PERSON::fred"}
    # Costco is 2 hops away — excluded
    assert "ORG::costco" not in {n.key for n in hood["nodes"]}


def test_neighbourhood_depth_2_reaches_costco():
    store = _chain_store()
    hood = store.neighbourhood("PERSON::bill", depth=2)
    assert {n.key for n in hood["nodes"]} == {
        "PERSON::bill",
        "ORG::walmart",
        "PERSON::fred",
        "ORG::costco",
    }
    # all three edges fall inside the node set
    assert len(hood["edges"]) == 3


def test_neighbourhood_is_direction_agnostic():
    # seeding from Walmart must still surface Bill via the incoming WORKS_AT edge
    store = _chain_store()
    hood = store.neighbourhood("ORG::walmart", depth=1)
    assert "PERSON::bill" in {n.key for n in hood["nodes"]}


def test_neighbourhood_unknown_key_is_empty():
    assert _chain_store().neighbourhood("PERSON::ghost") == {"nodes": [], "edges": []}


def test_render_subgraph_contains_nodes_and_edges():
    store = _chain_store()
    text = store.render_subgraph(["PERSON::bill"], depth=1)
    assert "PERSON::bill" in text
    assert "ORG::walmart" in text
    assert "-[WORKS_AT]->" in text


def test_render_subgraph_empty_for_unknown():
    assert _chain_store().render_subgraph(["PERSON::ghost"]) == ""
