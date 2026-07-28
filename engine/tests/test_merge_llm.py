"""Hermetic unit tests for merge 2b: confidence banding, parsing, prompt build."""

import pytest

from ontomem.merge import Resolution
from ontomem.merge_llm import (
    AUTO_MERGE,
    NEW,
    band_decision,
    build_disambiguation_prompt,
    parse_disambiguation_json,
)
from ontomem.model import Node


# --- banding -------------------------------------------------------------------


def test_band_above_090_auto_merges_unflagged():
    d = band_decision("PERSON::sam", [{"key": "PERSON::samuel", "confidence": 0.95}])
    assert d.decision == AUTO_MERGE
    assert d.target_key == "PERSON::samuel"
    assert d.flagged is False


def test_band_between_070_090_merges_flagged():
    d = band_decision("PERSON::sam", [{"key": "PERSON::samuel", "confidence": 0.80}])
    assert d.decision == AUTO_MERGE
    assert d.flagged is True


def test_band_exactly_070_is_flagged_merge():
    d = band_decision("PERSON::sam", [{"key": "PERSON::samuel", "confidence": 0.70}])
    assert d.decision == AUTO_MERGE
    assert d.flagged is True


def test_band_below_070_is_new():
    d = band_decision("PERSON::sam", [{"key": "PERSON::samuel", "confidence": 0.55}])
    assert d.decision == NEW
    assert d.target_key is None


def test_band_picks_highest_qualifying_candidate():
    d = band_decision(
        "PERSON::sam",
        [
            {"key": "PERSON::samuel", "confidence": 0.72},
            {"key": "PERSON::samson", "confidence": 0.93},
        ],
    )
    assert d.target_key == "PERSON::samson"
    assert d.flagged is False


def test_band_no_candidates_is_new():
    d = band_decision("PERSON::sam", [])
    assert d.decision == NEW


# --- parsing -------------------------------------------------------------------


def test_parse_filters_unknown_keys_and_clamps():
    raw = '{"candidates": [{"key": "PERSON::samuel", "confidence": 1.4, "reason": "same job"}, {"key": "PERSON::ghost", "confidence": 0.9}]}'
    out = parse_disambiguation_json(raw, ["PERSON::samuel"])
    assert len(out) == 1
    assert out[0]["key"] == "PERSON::samuel"
    assert out[0]["confidence"] == 1.0  # clamped
    assert out[0]["reason"] == "same job"


def test_parse_salvages_fenced_json():
    raw = '```json\n{"candidates": [{"key": "ORG::walmart", "confidence": 0.8}]}\n```'
    out = parse_disambiguation_json(raw, ["ORG::walmart"])
    assert out[0]["confidence"] == 0.8


# --- prompt build --------------------------------------------------------------


def test_prompt_includes_entity_and_candidate_blocks():
    entity = Node.create("PERSON", "Sam")
    prompt = build_disambiguation_prompt(
        entity, "User mentioned Sam who works at Walmart.", {"PERSON::samuel": "- PERSON::samuel ..."}
    )
    assert "PERSON::sam" in prompt
    assert "PERSON::samuel" in prompt
    assert "works at Walmart" in prompt


# --- guard ---------------------------------------------------------------------


def test_disambiguate_rejects_non_review_resolution():
    from ontomem.merge_llm import disambiguate

    res = Resolution("PERSON::sam", "new", "none")
    with pytest.raises(ValueError):
        disambiguate(Node.create("PERSON", "Sam"), res, store=None)
