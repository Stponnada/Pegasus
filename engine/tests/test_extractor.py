"""Hermetic unit tests for the extractor scaffolding (no network).

Covers the deterministic halves: JSON parse/salvage and payload normalisation.
"""

import json

import pytest

from ontomem.extractor import (
    build_prompt,
    extract,
    normalize_payload,
    parse_extractor_json,
    to_jsonl,
)


# --- parse / salvage -----------------------------------------------------------


def test_parse_clean_json():
    assert parse_extractor_json('{"a": 1}') == {"a": 1}


def test_parse_strips_code_fences():
    raw = '```json\n{"a": 1}\n```'
    assert parse_extractor_json(raw) == {"a": 1}


def test_parse_with_leading_and_trailing_prose():
    raw = 'Here is the JSON you asked for:\n{"a": 1}\nHope that helps!'
    assert parse_extractor_json(raw) == {"a": 1}


def test_parse_trailing_commas_salvaged():
    raw = '{"entities": [1, 2,], "episode": {"importance": 0.5,},}'
    assert parse_extractor_json(raw) == {"entities": [1, 2], "episode": {"importance": 0.5}}


def test_parse_unrecoverable_raises():
    with pytest.raises(ValueError):
        parse_extractor_json("this is not json at all")


# --- normalisation: happy path -------------------------------------------------

FULL_PAYLOAD = {
    "entities": [
        {"text": "Bill", "type": "PERSON", "confidence": 0.9, "aliases": ["Billy"]},
        {"text": "Walmart", "type": "ORG", "confidence": 0.85},
    ],
    "relationships": [
        {
            "source": "Bill",
            "relation": "WORKS_AT",
            "target": "Walmart",
            "confidence": 0.9,
            "stability": "mutable",
            "cardinality": "one_to_many",
            "snippet": "Bill said he works at Walmart and it has been stressful lately.",
            "evidence": "works at Walmart",
        }
    ],
    "episode": {"summary": "Bill talked about his job at Walmart.", "importance": 0.6, "tags": ["career"]},
}


def test_normalize_full_payload():
    result = normalize_payload(FULL_PAYLOAD)
    assert {n.key for n in result.nodes} == {"PERSON::bill", "ORG::walmart"}
    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge.key == "PERSON::bill::WORKS_AT::ORG::walmart"
    assert edge.stability == "mutable"
    assert result.episode.summary.startswith("Bill talked")
    assert result.episode.importance == 0.6
    assert result.warnings == []


def test_normalize_episode_only_is_valid():
    result = normalize_payload({"entities": [], "relationships": [], "episode": {"summary": "Small talk.", "importance": 0.1}})
    assert result.nodes == []
    assert result.edges == []
    assert result.episode.summary == "Small talk."


def test_normalize_missing_episode_synthesises_fallback():
    result = normalize_payload({"entities": [], "relationships": []})
    assert result.episode.summary == "(no durable summary)"
    assert result.episode.importance == 0.1
    assert any("fallback" in w for w in result.warnings)


# --- normalisation: defensive behaviour ---------------------------------------


def test_synthesizes_anchored_unknown_endpoint():
    # Bill is real; Walmart was referenced but not extracted -> synthesise it
    # (provisional OTHER node) and KEEP the edge rather than dropping the concept.
    payload = {
        "entities": [{"text": "Bill", "type": "PERSON"}],
        "relationships": [{"source": "Bill", "relation": "WORKS_AT", "target": "Walmart"}],
        "episode": {"summary": "x", "importance": 0.2},
    }
    result = normalize_payload(payload)
    assert {n.key for n in result.nodes} == {"PERSON::bill", "OTHER::walmart"}
    assert len(result.edges) == 1
    assert result.edges[0].target_key == "OTHER::walmart"
    synth = next(n for n in result.nodes if n.key == "OTHER::walmart")
    assert synth.properties.get("provisional_from_relationship") is True
    assert synth.confidence <= 0.5
    assert any("synthesised provisional node" in w for w in result.warnings)


def test_strict_mode_drops_unknown_endpoint():
    # the synthesis safety-net can be turned off (back to conservative drop)
    payload = {
        "entities": [{"text": "Bill", "type": "PERSON"}],
        "relationships": [{"source": "Bill", "relation": "WORKS_AT", "target": "Walmart"}],
        "episode": {"summary": "x", "importance": 0.2},
    }
    result = normalize_payload(payload, synthesize_missing_endpoints=False)
    assert result.edges == []
    assert any("endpoint not in entities" in w for w in result.warnings)


def test_drops_relationship_when_both_endpoints_unknown():
    # neither endpoint resolves -> too speculative to anchor; drop (no synthesis)
    payload = {
        "entities": [{"text": "Bill", "type": "PERSON"}],
        "relationships": [{"source": "Ghost", "relation": "KNOWS", "target": "Phantom"}],
        "episode": {"summary": "x", "importance": 0.2},
    }
    result = normalize_payload(payload)
    assert result.edges == []
    assert {n.key for n in result.nodes} == {"PERSON::bill"}  # nothing synthesised
    assert any("dropped: endpoint not in entities" in w for w in result.warnings)


def test_drops_self_relationship():
    payload = {
        "entities": [{"text": "Bill", "type": "PERSON"}],
        "relationships": [{"source": "Bill", "relation": "KNOWS", "target": "Bill"}],
        "episode": {"summary": "x", "importance": 0.2},
    }
    result = normalize_payload(payload)
    assert result.edges == []
    assert any("self-relationship" in w for w in result.warnings)


def test_coerces_relation_casing_and_spaces():
    payload = {
        "entities": [{"text": "Bill", "type": "person"}, {"text": "Sarah", "type": "PERSON"}],
        "relationships": [{"source": "Bill", "relation": "complained about", "target": "Sarah"}],
        "episode": {"summary": "x", "importance": 0.2},
    }
    result = normalize_payload(payload)
    assert result.edges[0].relation == "COMPLAINED_ABOUT"
    # lowercase entity type was coerced to a valid kind
    assert result.nodes[0].kind == "PERSON"


def test_invalid_entity_type_endpoint_is_recovered_as_other():
    # Zorp had an invalid type (ALIEN) and is dropped as an entity, but it is
    # anchored by Bill in a relationship -> recovered as a provisional OTHER node
    # rather than losing the edge. (The invalid type degrades to the catch-all.)
    payload = {
        "entities": [{"text": "Zorp", "type": "ALIEN"}, {"text": "Bill", "type": "PERSON"}],
        "relationships": [{"source": "Bill", "relation": "KNOWS", "target": "Zorp"}],
        "episode": {"summary": "x", "importance": 0.2},
    }
    result = normalize_payload(payload)
    assert {n.key for n in result.nodes} == {"PERSON::bill", "OTHER::zorp"}
    assert len(result.edges) == 1
    assert any("ALIEN" in w or "Zorp" in w for w in result.warnings)


def test_strict_mode_drops_invalid_type_dependent_relationship():
    payload = {
        "entities": [{"text": "Zorp", "type": "ALIEN"}, {"text": "Bill", "type": "PERSON"}],
        "relationships": [{"source": "Bill", "relation": "KNOWS", "target": "Zorp"}],
        "episode": {"summary": "x", "importance": 0.2},
    }
    result = normalize_payload(payload, synthesize_missing_endpoints=False)
    assert {n.key for n in result.nodes} == {"PERSON::bill"}
    assert result.edges == []


def test_clamps_out_of_range_confidence():
    payload = {
        "entities": [{"text": "Bill", "type": "PERSON", "confidence": 1.5}],
        "relationships": [],
        "episode": {"summary": "x", "importance": 9.0},
    }
    result = normalize_payload(payload)
    assert result.nodes[0].confidence == 1.0
    assert result.episode.importance == 1.0


def test_strips_ttl_when_stability_not_timebound():
    payload = {
        "entities": [{"text": "Bill", "type": "PERSON"}, {"text": "Walmart", "type": "ORG"}],
        "relationships": [
            {"source": "Bill", "relation": "WORKS_AT", "target": "Walmart", "stability": "stable", "ttl_days": 30}
        ],
        "episode": {"summary": "x", "importance": 0.2},
    }
    result = normalize_payload(payload)
    assert result.edges[0].ttl_days is None  # coerced away, edge not dropped


def test_relationship_resolves_endpoint_from_existing_graph():
    # the user 'Marcus' exists in the graph but wasn't re-extracted this turn;
    # a relationship referencing him must connect to the existing node, not drop
    payload = {
        "entities": [{"text": "Priya", "type": "PERSON"}],
        "relationships": [{"source": "Marcus", "relation": "IS_FRIENDS_WITH", "target": "Priya"}],
        "episode": {"summary": "x", "importance": 0.3},
    }
    result = normalize_payload(payload, existing_keys={"marcus": "PERSON::marcus"})
    assert len(result.edges) == 1
    assert result.edges[0].source_key == "PERSON::marcus"
    assert result.edges[0].target_key == "PERSON::priya"
    assert result.warnings == []


def test_synthesizes_endpoint_anchored_by_existing_graph_node():
    # the anchor can come from the existing graph (Marcus), not just this turn's
    # entities: 'On-call Rotation' is referenced but unextracted -> synthesised
    payload = {
        "entities": [{"text": "Priya", "type": "PERSON"}],
        "relationships": [{"source": "Marcus", "relation": "STRUGGLES_WITH", "target": "On-call Rotation"}],
        "episode": {"summary": "x", "importance": 0.3},
    }
    result = normalize_payload(payload, existing_keys={"marcus": "PERSON::marcus"})
    assert len(result.edges) == 1
    assert result.edges[0].source_key == "PERSON::marcus"
    assert result.edges[0].target_key == "OTHER::on_call_rotation"
    assert any(n.key == "OTHER::on_call_rotation" for n in result.nodes)


def test_candidate_merge_key_captured():
    payload = {
        "entities": [{"text": "Bill", "type": "PERSON", "candidate_merge_key": "PERSON::william"}],
        "relationships": [],
        "episode": {"summary": "x", "importance": 0.2},
    }
    result = normalize_payload(payload)
    assert result.candidate_merge_keys == {"PERSON::bill": "PERSON::william"}


# --- prompt building -----------------------------------------------------------


def test_to_jsonl():
    turns = [{"role": "user", "turn": 1, "text": "hi"}, {"role": "assistant", "turn": 2, "text": "hello"}]
    lines = to_jsonl(turns).splitlines()
    assert len(lines) == 2
    assert lines[0] == '{"role": "user", "turn": 1, "text": "hi"}'


def test_build_prompt_fills_all_placeholders():
    prompt = build_prompt("CONVO_HERE", "GRAPH_HERE", "2026-06-19T00:00:00+00:00")
    assert "{conversation_jsonl}" not in prompt
    assert "{existing_graph_context}" not in prompt
    assert "{current_utc_time}" not in prompt
    assert "CONVO_HERE" in prompt and "GRAPH_HERE" in prompt and "2026-06-19" in prompt


def test_build_prompt_empty_graph_placeholder():
    prompt = build_prompt("CONVO", "", "t")
    assert "(empty - no existing graph)" in prompt


def test_build_prompt_accepts_alternate_template():
    prompt = build_prompt("CONVO_HERE", "GRAPH_HERE", "t", template="custom {conversation_jsonl} / {existing_graph_context} / {current_utc_time}")
    assert prompt == "custom CONVO_HERE / GRAPH_HERE / t"


def test_extract_prompt_template_overrides_default():
    captured = {}

    def fake_generate(prompt, *, model, purpose):
        captured["prompt"] = prompt
        return json.dumps({"entities": [], "relationships": [], "episode": {"summary": "s", "importance": 0.1, "tags": []}})

    result = extract(
        [{"role": "user", "turn": 1, "text": "hi"}],
        generate_fn=fake_generate,
        prompt_template="ALTERNATE PROMPT {conversation_jsonl}",
    )

    assert "ALTERNATE PROMPT" in captured["prompt"]
    assert result.episode.summary == "s"


def test_extract_prompt_template_none_uses_default():
    captured = {}

    def fake_generate(prompt, *, model, purpose):
        captured["prompt"] = prompt
        return json.dumps({"entities": [], "relationships": [], "episode": {"summary": "s", "importance": 0.1, "tags": []}})

    extract([{"role": "user", "turn": 1, "text": "hi"}], generate_fn=fake_generate)

    assert "WHAT TO EXTRACT" in captured["prompt"]  # a heading unique to EXTRACTOR_SYSTEM_PROMPT
