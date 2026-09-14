"""Hermetic tests for the Engine facade: full read/write/retrieve/decay cycles,
driven offline via an injected extractor stub and the HashingEmbedder."""

from datetime import timedelta

from ontomem.embeddings import HashingEmbedder
from ontomem.engine import AGENTIC_SESSION_MAX_AGE, MAX_AGENTIC_TOOL_CALLS, Engine, _now
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


def test_user_only_extraction_strips_assistant_turns(tmp_path):
    captured = {}

    def spy_extract(convo, ctx):
        captured["convo"] = convo
        return _bill_extraction()

    eng = Engine(
        tmp_path,
        embedder=HashingEmbedder(),
        extract_fn=spy_extract,
        disambiguate_fn=lambda e, r, s, c: DisambiguationDecision(e.key, NEW, None, 0.0),
        user_only_extraction=True,
    )
    conversation = [
        {"role": "user", "turn": 1, "text": "I work at Walmart, three years now."},
        {"role": "assistant", "turn": 2, "text": "That's a long tenure, how do you find it?"},
        {"role": "user", "turn": 3, "text": "It's fine, tiring some days."},
    ]

    eng.write(conversation)

    assert [t["role"] for t in captured["convo"]] == ["user", "user"]
    assert all(t["role"] != "assistant" for t in captured["convo"])


def test_user_only_extraction_off_by_default_keeps_both_roles(tmp_path):
    captured = {}

    def spy_extract(convo, ctx):
        captured["convo"] = convo
        return _bill_extraction()

    eng = Engine(
        tmp_path,
        embedder=HashingEmbedder(),
        extract_fn=spy_extract,
        disambiguate_fn=lambda e, r, s, c: DisambiguationDecision(e.key, NEW, None, 0.0),
    )
    conversation = [
        {"role": "user", "turn": 1, "text": "I work at Walmart, three years now."},
        {"role": "assistant", "turn": 2, "text": "That's a long tenure, how do you find it?"},
    ]

    eng.write(conversation)

    assert [t["role"] for t in captured["convo"]] == ["user", "assistant"]


def _tool_call(call_id, name, args):
    return {"id": call_id, "function": {"name": name, "arguments": __import__("json").dumps(args)}}


def test_agentic_write_commits_incrementally_and_finishes(tmp_path):
    """Scripted chat_fn simulating: add Bill, add Walmart, connect them,
    finish. Each step should already be persisted in the store by the time
    the NEXT chat_fn call happens (so a real backend could use that as
    context), and the final result should look like a normal write()."""
    calls = []

    def chat_fn(messages, *, model, tools):
        calls.append([m.get("role") for m in messages])
        step = len(calls)
        if step == 1:
            return {"content": "", "tool_calls": [_tool_call("c1", "add_entity",
                {"text": "Bill", "type": "PERSON", "confidence": 0.9, "properties": {}, "aliases": [], "candidate_merge_key": None})]}
        if step == 2:
            # By now Bill must already be in the store (incremental, not batched).
            assert "PERSON::bill" in eng.store.nodes
            return {"content": "", "tool_calls": [_tool_call("c2", "add_entity",
                {"text": "Walmart", "type": "ORG", "confidence": 0.9, "properties": {}, "aliases": [], "candidate_merge_key": None})]}
        if step == 3:
            return {"content": "", "tool_calls": [_tool_call("c3", "add_relationship", {
                "source": "Bill", "relation": "WORKS_AT", "target": "Walmart",
                "confidence": 0.9, "stability": "mutable", "ttl_days": None,
                "cardinality": "one_to_one", "evidence": "works at Walmart",
                "snippet": "Bill said he works at Walmart.", "properties": {},
            })]}
        return {"content": "", "tool_calls": [_tool_call("c4", "finish_extraction",
            {"summary": "Bill mentioned working at Walmart.", "importance": 0.5, "tags": ["career"]})]}

    eng = Engine(
        tmp_path,
        embedder=HashingEmbedder(),
        disambiguate_fn=lambda e, r, s, c: DisambiguationDecision(e.key, NEW, None, 0.0),
        chat_fn=chat_fn,
        agentic_extraction=True,
    )

    result = eng.write([{"role": "user", "turn": 1, "text": "I work at Walmart."}])

    assert result["nodes_created"] == 2
    assert result["edges_created"] == 1
    assert result["warnings"] == []
    assert len(calls) == 4  # stopped right after finish_extraction, no extra call
    episode = eng.store.get_episode(result["episode_id"])
    assert episode.summary == "Bill mentioned working at Walmart."
    assert episode.tags == ["career"]
    assert eng.store.get_edge("PERSON::bill::WORKS_AT::ORG::walmart") is not None


def test_agentic_synthesizes_missing_endpoint(tmp_path):
    """add_relationship naming a target that was never add_entity'd should
    get a provisional node instead of being silently dropped -- same
    anchored-auto-create policy as the batch path (extractor.py's
    _normalize_relationships): synthesise when exactly ONE endpoint is
    missing. (If BOTH are missing, dropping is correct -- too speculative to
    trust -- which is covered by the neither-exists branch's own warning
    path, not this test.)"""
    def chat_fn(messages, *, model, tools):
        step = sum(1 for m in messages if m.get("role") == "assistant") + 1
        if step == 1:
            return {"content": "", "tool_calls": [_tool_call("c1", "add_entity",
                {"text": "Bill", "type": "PERSON", "confidence": 0.9, "properties": {}, "aliases": [], "candidate_merge_key": None})]}
        if step == 2:
            return {"content": "", "tool_calls": [_tool_call("c2", "add_relationship", {
                "source": "Bill", "relation": "WORKS_AT", "target": "Walmart",
                "confidence": 0.9, "stability": "mutable", "ttl_days": None,
                "cardinality": "one_to_one", "evidence": "", "snippet": "x" * 20, "properties": {},
            })]}
        return {"content": "", "tool_calls": [_tool_call("c3", "finish_extraction",
            {"summary": "s", "importance": 0.3, "tags": []})]}

    eng = Engine(
        tmp_path, embedder=HashingEmbedder(),
        disambiguate_fn=lambda e, r, s, c: DisambiguationDecision(e.key, NEW, None, 0.0),
        chat_fn=chat_fn, agentic_extraction=True,
    )

    result = eng.write([{"role": "user", "turn": 1, "text": "I work at Walmart."}])

    assert result["edges_created"] == 1
    assert any("synthesised provisional node" in w for w in result["warnings"])
    assert "PERSON::bill" in eng.store.nodes and "OTHER::walmart" in eng.store.nodes


def test_agentic_hits_safety_cap_without_finish_extraction(tmp_path):
    def chat_fn(messages, *, model, tools):
        return {"content": "", "tool_calls": [_tool_call("cX", "add_entity", {
            "text": f"Node{len(messages)}", "type": "THING", "confidence": 0.5,
            "properties": {}, "aliases": [], "candidate_merge_key": None,
        })]}

    eng = Engine(
        tmp_path, embedder=HashingEmbedder(),
        disambiguate_fn=lambda e, r, s, c: DisambiguationDecision(e.key, NEW, None, 0.0),
        chat_fn=chat_fn, agentic_extraction=True,
    )

    result = eng.write([{"role": "user", "turn": 1, "text": "..."}])

    assert any("safety cap" in w for w in result["warnings"])


# --- split-request agentic extraction (start_agentic_write / apply_agentic_tool_call) ---
# Same deterministic graph-mutation logic as the chat_fn-driven tests above,
# but driven the way an external caller with its own native tool-calling
# loop would (e.g. the opencode plugin) -- one HTTP-shaped call per tool
# call, no chat_fn/messages list involved at all.


def _agentic_engine(tmp_path):
    return Engine(
        tmp_path, embedder=HashingEmbedder(),
        disambiguate_fn=lambda e, r, s, c: DisambiguationDecision(e.key, NEW, None, 0.0),
    )


def test_split_agentic_write_commits_incrementally_and_finishes(tmp_path):
    eng = _agentic_engine(tmp_path)

    start = eng.start_agentic_write([{"role": "user", "turn": 1, "text": "I work at Walmart."}])
    assert isinstance(start["session_id"], str) and start["session_id"]
    assert "prompt_text" in start

    r1 = eng.apply_agentic_tool_call(start["session_id"], "add_entity", {
        "text": "Bill", "type": "PERSON", "confidence": 0.9, "properties": {}, "aliases": [], "candidate_merge_key": None,
    })
    assert r1["finished"] is False
    assert "PERSON::bill" in eng.store.nodes  # committed immediately, not batched

    r2 = eng.apply_agentic_tool_call(start["session_id"], "add_entity", {
        "text": "Walmart", "type": "ORG", "confidence": 0.9, "properties": {}, "aliases": [], "candidate_merge_key": None,
    })
    assert r2["finished"] is False
    assert "ORG::walmart" in eng.store.nodes

    r3 = eng.apply_agentic_tool_call(start["session_id"], "add_relationship", {
        "source": "Bill", "relation": "WORKS_AT", "target": "Walmart",
        "confidence": 0.9, "stability": "mutable", "ttl_days": None,
        "cardinality": "one_to_one", "evidence": "works at Walmart",
        "snippet": "Bill said he works at Walmart.", "properties": {},
    })
    assert r3["finished"] is False
    assert eng.store.get_edge("PERSON::bill::WORKS_AT::ORG::walmart") is not None

    r4 = eng.apply_agentic_tool_call(start["session_id"], "finish_extraction", {
        "summary": "Bill mentioned working at Walmart.", "importance": 0.5, "tags": ["career"],
    })
    assert r4["finished"] is True
    stats = r4["stats"]
    assert stats["nodes_created"] == 2
    assert stats["edges_created"] == 1
    assert stats["warnings"] == []
    episode = eng.store.get_episode(stats["episode_id"])
    assert episode.summary == "Bill mentioned working at Walmart."
    assert episode.tags == ["career"]

    # session is gone once finished -- a stray extra call must not resurrect it
    r5 = eng.apply_agentic_tool_call(start["session_id"], "add_entity", {
        "text": "Extra", "type": "THING", "confidence": 0.5, "properties": {}, "aliases": [], "candidate_merge_key": None,
    })
    assert r5["finished"] is True
    assert "already-finished" in r5["result_text"] or "unknown" in r5["result_text"]


def test_split_agentic_synthesizes_missing_endpoint(tmp_path):
    eng = _agentic_engine(tmp_path)
    session_id = eng.start_agentic_write([{"role": "user", "turn": 1, "text": "I work at Walmart."}])["session_id"]

    eng.apply_agentic_tool_call(session_id, "add_entity", {
        "text": "Bill", "type": "PERSON", "confidence": 0.9, "properties": {}, "aliases": [], "candidate_merge_key": None,
    })
    r = eng.apply_agentic_tool_call(session_id, "add_relationship", {
        "source": "Bill", "relation": "WORKS_AT", "target": "Walmart",
        "confidence": 0.9, "stability": "mutable", "ttl_days": None,
        "cardinality": "one_to_one", "evidence": "", "snippet": "x" * 20, "properties": {},
    })
    final = eng.apply_agentic_tool_call(session_id, "finish_extraction", {"summary": "s", "importance": 0.3, "tags": []})

    assert final["stats"]["edges_created"] == 1
    assert any("synthesised provisional node" in w for w in final["stats"]["warnings"])
    assert "PERSON::bill" in eng.store.nodes and "OTHER::walmart" in eng.store.nodes


def test_split_agentic_hits_safety_cap_without_finish_extraction(tmp_path):
    eng = _agentic_engine(tmp_path)
    session_id = eng.start_agentic_write([{"role": "user", "turn": 1, "text": "..."}])["session_id"]

    result = None
    for i in range(MAX_AGENTIC_TOOL_CALLS + 1):
        result = eng.apply_agentic_tool_call(session_id, "add_entity", {
            "text": f"Node{i}", "type": "THING", "confidence": 0.5,
            "properties": {}, "aliases": [], "candidate_merge_key": None,
        })
        if result["finished"]:
            break

    assert result["finished"] is True
    assert any("safety cap" in w for w in result["stats"]["warnings"])


def test_split_agentic_unknown_session_returns_finished_error(tmp_path):
    eng = _agentic_engine(tmp_path)
    result = eng.apply_agentic_tool_call("not-a-real-session", "add_entity", {
        "text": "X", "type": "THING", "confidence": 0.5, "properties": {}, "aliases": [], "candidate_merge_key": None,
    })
    assert result["finished"] is True
    assert "unknown or already-finished" in result["result_text"]


def test_split_agentic_sweeps_stale_session(tmp_path):
    eng = _agentic_engine(tmp_path)
    stale_id = eng.start_agentic_write([{"role": "user", "turn": 1, "text": "stale one"}])["session_id"]
    # Commit one entity to the stale session so we can confirm the sweep
    # finalizes with work already committed (nothing lost), not discarded.
    eng.apply_agentic_tool_call(stale_id, "add_entity", {
        "text": "Ghost", "type": "THING", "confidence": 0.5, "properties": {}, "aliases": [], "candidate_merge_key": None,
    })
    eng._agentic_sessions[stale_id].started_at -= AGENTIC_SESSION_MAX_AGE + timedelta(minutes=1)

    fresh_id = eng.start_agentic_write([{"role": "user", "turn": 1, "text": "fresh one"}])["session_id"]

    assert stale_id not in eng._agentic_sessions  # swept as a side effect of the next tool_call/start call
    assert "THING::ghost" in eng.store.nodes  # already-committed work survived the sweep
    assert fresh_id in eng._agentic_sessions


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


def test_decay_second_call_within_24h_is_a_no_op(tmp_path):
    """Self-gating (see Engine.decay's docstring): a caller that fires /decay
    on every opencode process start, with no cron, must not redo real work
    (and reset every edge's updated_at clock) each time."""
    eng = _engine(tmp_path, _bill_extraction())
    eng.write([{"role": "user", "text": "Walmart"}])
    edge_key = "PERSON::bill::WORKS_AT::ORG::walmart"
    edge = eng.store.get_edge(edge_key)
    edge.updated_at = (_now() - timedelta(days=10)).isoformat()

    first = eng.decay()
    strength_after_first = eng.store.get_edge(edge_key).strength
    assert first["edges_decayed"] == 1

    second = eng.decay()  # same process, minutes later in wall-clock terms

    assert second == {"edges_decayed": 0, "edges_dormant": 0, "skipped": True}
    assert eng.store.get_edge(edge_key).strength == strength_after_first


def test_decay_runs_again_after_24h_elapsed(tmp_path):
    eng = _engine(tmp_path, _bill_extraction())
    eng.write([{"role": "user", "text": "Walmart"}])
    edge_key = "PERSON::bill::WORKS_AT::ORG::walmart"
    edge = eng.store.get_edge(edge_key)
    edge.stability = "mutable"
    edge.updated_at = (_now() - timedelta(days=10)).isoformat()

    first_now = _now()
    eng.decay(now=first_now)
    strength_after_first = eng.store.get_edge(edge_key).strength

    stats = eng.decay(now=first_now + timedelta(hours=25))

    assert stats["edges_decayed"] == 1
    assert eng.store.get_edge(edge_key).strength < strength_after_first


def test_decay_force_bypasses_the_gate(tmp_path):
    eng = _engine(tmp_path, _bill_extraction())
    eng.write([{"role": "user", "text": "Walmart"}])
    eng.decay()

    stats = eng.decay(force=True)

    assert "skipped" not in stats
    assert stats["edges_decayed"] == 1
