"""Full-pipeline live integration test (Gemini extractor + Gemini embeddings).

Writes a real conversation, then reads a related query and asserts the memory
graph surfaces the right context. Auto-skips without GEMINI_API_KEY.
Run with:  pytest -m integration
"""

import pytest

from ontomem.embeddings import GeminiEmbedder
from ontomem.engine import Engine

pytestmark = pytest.mark.integration


def test_full_write_then_read(gemini_key, tmp_path):
    eng = Engine(tmp_path, embedder=GeminiEmbedder(api_key=gemini_key), api_key=gemini_key)

    conversation = [
        {"role": "user", "turn": 1, "text": "I'm Bill. I work at Walmart and my boss Sarah keeps moving my deadlines — it's exhausting."},
        {"role": "assistant", "turn": 2, "text": "That sounds draining. How long have you been there?"},
        {"role": "user", "turn": 3, "text": "Three years. My friend Fred works at Costco and keeps telling me to switch."},
    ]
    w = eng.write(conversation)
    assert w["nodes_created"] >= 3  # Bill, Walmart, Sarah, Fred, Costco-ish
    assert w["edges_created"] >= 2

    # A later, topically-related query should surface the employer context.
    r = eng.read("I'm thinking about my job situation at Walmart again")
    assert "Walmart" in r["text"]
    # semantic seeding + reverse traversal should connect Bill to Walmart
    assert "Bill" in r["memory_block"]


def test_semantic_seeding_via_synonym(gemini_key, tmp_path):
    eng = Engine(tmp_path, embedder=GeminiEmbedder(api_key=gemini_key), api_key=gemini_key)
    eng.write([
        {"role": "user", "turn": 1, "text": "My friend John is a commercial airline pilot and absolutely loves flying."},
    ])
    # 'aviation' / 'planes' should activate the pilot/flying nodes without the exact words.
    r = eng.read("I've been reading about aviation lately")
    assert r["text"] != ""  # something semantically related fired
