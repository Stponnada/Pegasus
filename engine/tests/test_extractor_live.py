"""Live integration tests for the extractor (Gemini). Auto-skip without a key.

LLM output is non-deterministic, so assertions check structural invariants and
spec-mandated behaviour, never exact strings. Kept to a minimum of calls to
respect the free-tier rate limit. Run explicitly with:  pytest -m integration
"""

import pytest

from ontomem.extractor import extract

pytestmark = pytest.mark.integration


def test_extracts_durable_facts_and_canonicalises_role(gemini_key):
    convo = [
        {"role": "user", "turn": 1, "text": "I'm Bill. I work at Walmart and my boss Sarah keeps moving deadlines on me."},
        {"role": "assistant", "turn": 2, "text": "That sounds stressful. How long have you been there?"},
        {"role": "user", "turn": 3, "text": "About three years. I live in New Jersey and commute in."},
    ]
    result = extract(convo, api_key=gemini_key)

    keys = {n.key for n in result.nodes}
    # Bill, Walmart, Sarah, New Jersey should all surface in some canonical form.
    assert any(k.startswith("PERSON::bill") for k in keys)
    assert any("walmart" in k for k in keys)
    # Spec canonicalisation rule: "my boss Sarah" -> name 'Sarah', role on the edge.
    assert any(k == "PERSON::sarah" for k in keys)
    assert not any("boss" in k.lower() for k in keys)

    relations = {e.relation for e in result.edges}
    assert any("WORK" in r for r in relations)  # WORKS_AT or similar

    # Clean structured output: parse + normalise produced no warnings.
    assert result.warnings == []
    # Every edge endpoint resolves to an extracted node (no dangling edges).
    for edge in result.edges:
        assert edge.source_key in keys and edge.target_key in keys
    assert result.episode.summary


def test_undurable_conversation_yields_no_entities(gemini_key):
    convo = [
        {"role": "user", "turn": 1, "text": "what's 17 times 23?"},
        {"role": "assistant", "turn": 2, "text": "391."},
        {"role": "user", "turn": 3, "text": "thanks"},
    ]
    result = extract(convo, api_key=gemini_key)
    # An arithmetic query reveals nothing durable about the user.
    assert result.nodes == []
    assert result.edges == []
    # An episode is still always produced.
    assert result.episode is not None
