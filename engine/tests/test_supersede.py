"""Hermetic tests for relationship supersession (spec §9.2c). The networked
decision (decide_supersession) is covered by the deterministic parse/prompt parts
here; the live call is exercised via the engine integration tests."""

from ontomem.supersede import (
    COEXIST,
    CONTRADICTS,
    SUPERSEDES,
    build_supersession_prompt,
    parse_supersession_json,
)


def test_parse_valid_decisions():
    assert parse_supersession_json('{"decision": "supersedes", "reason": "now committed"}') == {
        "decision": SUPERSEDES, "reason": "now committed"
    }
    assert parse_supersession_json('{"decision": "coexist"}')["decision"] == COEXIST
    assert parse_supersession_json('{"decision": "contradicts"}')["decision"] == CONTRADICTS


def test_parse_defaults_to_coexist_on_unknown_decision():
    # an unexpected label must fall back to the safe option: keep both
    assert parse_supersession_json('{"decision": "delete_it_all"}')["decision"] == COEXIST


def test_parse_defaults_to_coexist_on_garbage():
    assert parse_supersession_json("not json at all")["decision"] == COEXIST


def test_parse_strips_fences():
    raw = '```json\n{"decision": "supersedes", "reason": "x"}\n```'
    assert parse_supersession_json(raw)["decision"] == SUPERSEDES


def test_build_prompt_contains_both_relations():
    prompt = build_supersession_prompt(
        "Marcus", "Safety Team", "CONSIDERING_TRANSFER_TO", "TRANSFERS_TO",
        old_snippet="Marcus was weighing a transfer.", conversation_context="going for it",
    )
    assert "CONSIDERING_TRANSFER_TO" in prompt
    assert "TRANSFERS_TO" in prompt
    assert "Marcus" in prompt and "Safety Team" in prompt
    assert "going for it" in prompt
