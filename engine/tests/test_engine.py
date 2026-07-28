"""Hermetic tests for the Engine facade: full read/write/retrieve/decay cycles,
driven offline via an injected extractor stub and the HashingEmbedder."""

from datetime import timedelta

from ontomem.embeddings import HashingEmbedder
from ontomem.engine import Engine, _now
from ontomem.extractor import ExtractionResult
from ontomem.merge_llm import DisambiguationDecision
from ontomem.model import Episode, Node, Edge
from ontomem.merge import NEW


def _bill_extraction():
    return ExtractionResult(
        episode=Episode.create("Bill talked about working at Walmart.", 0.6),
        nodes=[Node.create("PERSON", "Bill"), Node.create("ORG", "Walmart")],
        edges=[
            Edge.create("PERSON::bill", "WORKS_AT", "ORG::walmart",
                        snippet="Bill said he has worked at Walmart for three years and finds it draining.")
        ],
    )


def _engine(tmp_path, extraction):
    return Engine(
        tmp_path,
        embedder=HashingEmbedder(),
        extract_fn=lambda convo, ctx: extraction,
        disambiguate_fn=lambda e, r, s, c: DisambiguationDecision(e.key, NEW, None, 0.0),
    )


def test_write_then_read_cycle(tmp_path):
    eng = _engine(tmp_path, _bill_extraction())
    w = eng.write([{"role": "user", "text": "I work at Walmart, three years now."}])
    assert w["nodes_created"] == 2
    assert w["edges_created"] == 1

    # snapshot + index persisted
    assert (tmp_path / "graph.json").exists()
    assert (tmp_path / "embeddings.npz").exists()

    r = eng.read("Tell me about Walmart")
    assert "Bill -[WORKS_AT]-> Walmart" in r["memory_block"]
    assert "three years" in r["context_block"]


def test_read_returns_structured_injections_and_caches_last_read(tmp_path):
    eng = _engine(tmp_path, _bill_extraction())
    assert eng.last_read is None  # nothing read yet
    eng.write([{"role": "user", "text": "I work at Walmart, three years now."}])

    r = eng.read("Tell me about Walmart")
    assert r["injections"] == [
        {
            "source": "Bill", "relation": "WORKS_AT", "relation_phrase": "works at",
            "target": "Walmart", "snippet": "Bill said he has worked at Walmart for three years and finds it draining.",
        }
    ]
    # cached for GET /last_read so a live viewer can show what a real
    # conversation turn injected, not just a manually-typed query
    assert eng.last_read["message"] == "Tell me about Walmart"
    assert eng.last_read["injections"] == r["injections"]


def test_read_logs_reinforcement_and_write_applies_it(tmp_path):
    eng = _engine(tmp_path, _bill_extraction())
    eng.write([{"role": "user", "text": "Walmart job"}])
    edge_key = "PERSON::bill::WORKS_AT::ORG::walmart"
    # decay the edge down so a boost is observable
    eng.store.get_edge(edge_key).strength = 50.0

    eng.read("how's Walmart")  # should traverse and log the edge
    assert (tmp_path / "reinforce.jsonl").exists()

    # an empty subsequent write drains the log and applies the deferred boost
    eng2 = Engine(
        tmp_path, embedder=HashingEmbedder(),
        extract_fn=lambda convo, ctx: ExtractionResult(episode=Episode.create("nothing", 0.1)),
        disambiguate_fn=lambda e, r, s, c: DisambiguationDecision(e.key, NEW, None, 0.0),
    )
    eng2.store.get_edge(edge_key).strength = 50.0
    res = eng2.write([{"role": "user", "text": "bye"}])
    assert res["reinforced_from_read"] >= 1
    assert eng2.store.get_edge(edge_key).strength == 65.0
    assert not (tmp_path / "reinforce.jsonl").exists()  # drained


def test_persistence_survives_reload(tmp_path):
    eng = _engine(tmp_path, _bill_extraction())
    eng.write([{"role": "user", "text": "Walmart"}])

    reloaded = Engine(tmp_path, embedder=HashingEmbedder())
    assert reloaded.store.get_node("PERSON::bill") is not None
    assert reloaded.store.get_edge("PERSON::bill::WORKS_AT::ORG::walmart") is not None
    # pointer graph rehydrated
    edge = reloaded.store.get_edge("PERSON::bill::WORKS_AT::ORG::walmart")
    assert edge.source is reloaded.store.get_node("PERSON::bill")


def test_retrieve_memory_returns_neighbourhood(tmp_path):
    eng = _engine(tmp_path, _bill_extraction())
    eng.write([{"role": "user", "text": "Walmart"}])
    out = eng.retrieve_memory("Bill", depth=1)
    assert out["found"] is True
    assert out["node"]["key"] == "PERSON::bill"
    assert any(r["relation"] == "WORKS_AT" for r in out["relationships"])
    assert any("three years" in r["snippet"] for r in out["relationships"])


def test_retrieve_memory_unknown(tmp_path):
    eng = _engine(tmp_path, _bill_extraction())
    assert eng.retrieve_memory("Nobody")["found"] is False


def test_decay_reduces_edge_strength_by_class(tmp_path):
    eng = _engine(tmp_path, _bill_extraction())
    eng.write([{"role": "user", "text": "Walmart"}])
    edge_key = "PERSON::bill::WORKS_AT::ORG::walmart"
    edge = eng.store.get_edge(edge_key)
    edge.stability = "mutable"  # lambda 0.020
    edge.strength = 100.0
    # backdate the edge clock by 10 days
    edge.updated_at = (_now() - timedelta(days=10)).isoformat()

    stats = eng.decay()
    # 100 * e^(-0.02*10) = 81.87
    assert abs(eng.store.get_edge(edge_key).strength - 81.873) < 0.1
    assert stats["edges_decayed"] == 1


def test_decay_flags_dormant_without_deleting(tmp_path):
    eng = _engine(tmp_path, _bill_extraction())
    eng.write([{"role": "user", "text": "Walmart"}])
    edge_key = "PERSON::bill::WORKS_AT::ORG::walmart"
    edge = eng.store.get_edge(edge_key)
    edge.stability = "ephemeral"
    edge.strength = 100.0
    edge.updated_at = (_now() - timedelta(days=60)).isoformat()  # decays far below 2.0

    stats = eng.decay()
    assert stats["edges_dormant"] == 1
    # never deleted — still retrievable
    assert eng.store.get_edge(edge_key) is not None
