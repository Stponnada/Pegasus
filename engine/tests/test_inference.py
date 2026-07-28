"""Hermetic tests for the OpenAI-compatible generation and embedding clients."""

from __future__ import annotations

import json
import urllib.request

import numpy as np

from ontomem.embeddings import OpenAICompatibleEmbedder
from ontomem.inference import OpenAICompatibleGenerator
from ontomem.journal import read_events


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_openai_generator_posts_chat_completion_and_audits(tmp_path, monkeypatch):
    requests = []

    def urlopen(request, timeout):
        requests.append((request, timeout))
        return _Response({"choices": [{"message": {"content": '{"entities":[]}'}}]})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    audit = tmp_path / "llm.jsonl"
    generator = OpenAICompatibleGenerator(
        "http://localhost:18000/v1",
        api_key="secret",
        audit_path=audit,
        audit_content=True,
    )

    result = generator.generate("prompt text", model="ontomem-llm", purpose="extract")

    assert result == '{"entities":[]}'
    request, timeout = requests[0]
    assert request.full_url == "http://localhost:18000/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer secret"
    payload = json.loads(request.data)
    assert payload["model"] == "ontomem-llm"
    assert payload["temperature"] == 0.0
    assert payload["messages"] == [{"role": "user", "content": "prompt text"}]
    assert timeout == 300.0

    event = read_events(audit)[0]
    assert event["purpose"] == "extract"
    assert event["prompt"] == "prompt text"
    assert event["response"] == '{"entities":[]}'
    assert event["latency_seconds"] >= 0


def test_openai_embedder_preserves_index_order_and_normalises(monkeypatch):
    def urlopen(request, timeout):
        payload = json.loads(request.data)
        assert request.full_url == "http://localhost:18001/v1/embeddings"
        assert payload["model"] == "ontomem-embed"
        assert payload["input"] == ["alpha", "beta"]
        assert timeout == 42
        return _Response(
            {
                "data": [
                    {"index": 1, "embedding": [0.0, 3.0]},
                    {"index": 0, "embedding": [4.0, 0.0]},
                ]
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    embedder = OpenAICompatibleEmbedder(
        "http://localhost:18001/v1",
        model="ontomem-embed",
        api_key="secret",
        timeout=42,
    )

    vectors = embedder.embed(["alpha", "beta"])

    assert vectors.dtype == np.float32
    np.testing.assert_allclose(vectors, [[1.0, 0.0], [0.0, 1.0]])


def test_openai_embedder_rejects_missing_vectors(monkeypatch):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: _Response({"data": [{"index": 0, "embedding": [1.0]}]}),
    )
    embedder = OpenAICompatibleEmbedder("http://localhost/v1", model="embed")

    try:
        embedder.embed(["one", "two"])
    except RuntimeError as exc:
        assert "1 vectors for 2 inputs" in str(exc)
    else:
        raise AssertionError("missing embedding vector was accepted")
