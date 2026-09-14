"""Hermetic tests for the OpenAI-compatible generation and embedding clients."""

from __future__ import annotations

import json
import urllib.request

import numpy as np

from ontomem.embeddings import OpenAICompatibleEmbedder
from ontomem.inference import HostCallbackGenerator, OpenAICompatibleGenerator
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


def test_openai_generator_uses_content_path_when_no_tool_schema(monkeypatch):
    """purpose='extract' without a configured tool schema is unchanged behavior."""
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        return _Response({"choices": [{"message": {"content": '{"entities":[]}'}}]})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    generator = OpenAICompatibleGenerator("http://localhost:18000/v1")

    result = generator.generate("prompt text", model="ontomem-llm", purpose="extract")

    assert result == '{"entities":[]}'
    payload = json.loads(requests[0].data)
    assert "tools" not in payload


def test_openai_generator_forces_tool_call_for_extract_purpose(monkeypatch):
    schema = {"type": "function", "function": {"name": "submit_graph_extraction", "parameters": {}}}
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        return _Response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "submit_graph_extraction", "arguments": '{"entities":[]}'}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    generator = OpenAICompatibleGenerator("http://localhost:18000/v1", extraction_tool_schema=schema)

    result = generator.generate("prompt text", model="ontomem-llm", purpose="extract")

    assert result == '{"entities":[]}'
    payload = json.loads(requests[0].data)
    assert payload["tools"] == [schema]
    assert payload["tool_choice"] == "auto"


def test_openai_generator_other_purposes_ignore_tool_schema(monkeypatch):
    """The tool schema is extraction-only -- disambiguate/supersede calls (any
    purpose other than 'extract') must not be forced through it."""
    schema = {"type": "function", "function": {"name": "submit_graph_extraction", "parameters": {}}}
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        return _Response({"choices": [{"message": {"content": "plain text answer"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    generator = OpenAICompatibleGenerator("http://localhost:18000/v1", extraction_tool_schema=schema)

    result = generator.generate("prompt text", model="ontomem-llm", purpose="disambiguate")

    assert result == "plain text answer"
    payload = json.loads(requests[0].data)
    assert "tools" not in payload


def test_openai_generator_raises_on_missing_tool_call(monkeypatch):
    schema = {"type": "function", "function": {"name": "submit_graph_extraction", "parameters": {}}}
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: _Response({"choices": [{"message": {}, "finish_reason": "stop"}]}),
    )
    generator = OpenAICompatibleGenerator("http://localhost:18000/v1", extraction_tool_schema=schema)

    try:
        generator.generate("prompt", model="ontomem-llm", purpose="extract")
    except RuntimeError as exc:
        assert "no tool_calls" in str(exc)
    else:
        raise AssertionError("missing tool_calls was silently accepted")


def test_host_callback_generator_posts_prompt_and_ignores_model(monkeypatch):
    requests = []

    def urlopen(request, timeout):
        requests.append((request, timeout))
        return _Response({"text": "recalled answer"})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    generator = HostCallbackGenerator("http://127.0.0.1:54321")

    result = generator.generate("prompt text", model="whatever-the-engine-thinks", purpose="extract")

    assert result == "recalled answer"
    request, timeout = requests[0]
    assert request.full_url == "http://127.0.0.1:54321/generate"
    assert request.get_header("Authorization") is None
    payload = json.loads(request.data)
    assert payload == {"prompt": "prompt text", "purpose": "extract"}
    assert timeout == 120.0


def test_host_callback_generator_audits_and_raises_on_empty_text(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: _Response({"text": "  "}))
    audit = tmp_path / "llm.jsonl"
    generator = HostCallbackGenerator("http://127.0.0.1:54321", audit_path=audit, audit_content=True)

    try:
        generator.generate("prompt text", model="ignored", purpose="disambiguate")
    except RuntimeError as exc:
        assert "no text" in str(exc)
    else:
        raise AssertionError("empty host callback text was silently accepted")

    event = read_events(audit)[0]
    assert event["purpose"] == "disambiguate"
    assert event["model"] == "host-callback"
    assert event["prompt"] == "prompt text"
    assert event["error"]


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
