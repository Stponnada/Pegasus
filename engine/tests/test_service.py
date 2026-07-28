"""Hermetic tests for the service dispatch layer (no sockets, no network)."""

import pytest

from ontomem.embeddings import HashingEmbedder, OpenAICompatibleEmbedder
from ontomem.engine import Engine
from ontomem.extractor import ExtractionResult
from ontomem.merge import NEW
from ontomem.merge_llm import DisambiguationDecision
from ontomem.model import Edge, Episode, Node
from ontomem.service import ServiceError, dispatch, graph_snapshot, make_engine_from_env


def _engine(tmp_path):
    extraction = ExtractionResult(
        episode=Episode.create("Bill works at Walmart.", 0.6),
        nodes=[Node.create("PERSON", "Bill"), Node.create("ORG", "Walmart")],
        edges=[Edge.create("PERSON::bill", "WORKS_AT", "ORG::walmart", snippet="Bill has worked at Walmart for three years.")],
    )
    return Engine(
        tmp_path, embedder=HashingEmbedder(),
        extract_fn=lambda c, ctx: extraction,
        disambiguate_fn=lambda e, r, s, c: DisambiguationDecision(e.key, NEW, None, 0.0),
    )


def test_health(tmp_path):
    out = dispatch(_engine(tmp_path), "/health", {})
    assert out["ok"] is True


def test_write_then_read_then_retrieve(tmp_path):
    eng = _engine(tmp_path)
    w = dispatch(eng, "/write", {"conversation": [{"role": "user", "text": "Walmart job"}]})
    assert w["nodes_created"] == 2

    r = dispatch(eng, "/read", {"message": "Tell me about Walmart"})
    assert "Walmart" in r["text"]

    d = dispatch(eng, "/retrieve_memory", {"node_name": "Bill", "depth": 1})
    assert d["found"] is True


def test_decay_route(tmp_path):
    out = dispatch(_engine(tmp_path), "/decay", {})
    assert "edges_decayed" in out


def test_bad_input_raises_400(tmp_path):
    eng = _engine(tmp_path)
    with pytest.raises(ServiceError) as exc:
        dispatch(eng, "/read", {})
    assert exc.value.status == 400


def test_unknown_route_raises_404(tmp_path):
    with pytest.raises(ServiceError) as exc:
        dispatch(_engine(tmp_path), "/nope", {})
    assert exc.value.status == 404


def test_graph_snapshot_shape(tmp_path):
    eng = _engine(tmp_path)
    dispatch(eng, "/write", {"conversation": [{"role": "user", "text": "Walmart job"}]})

    snapshot = graph_snapshot(eng)
    assert {n["id"] for n in snapshot["nodes"]} == {"PERSON::bill", "ORG::walmart"}
    assert len(snapshot["edges"]) == 1
    edge = snapshot["edges"][0]
    assert edge["from"] == "PERSON::bill"
    assert edge["to"] == "ORG::walmart"
    assert edge["relation"] == "WORKS_AT"
    assert edge["snippet"]
    assert edge["dormant"] is False


def test_graph_snapshot_decays_strength_to_now(tmp_path):
    eng = _engine(tmp_path)
    dispatch(eng, "/write", {"conversation": [{"role": "user", "text": "Walmart job"}]})

    edge = next(iter(eng.store.edges.values()))
    edge.stability = "ephemeral"  # fast decay so an artificial backdate is visible
    edge.updated_at = "2020-01-01T00:00:00+00:00"

    snapshot = graph_snapshot(eng)
    assert snapshot["edges"][0]["strength"] < edge.strength
    assert snapshot["edges"][0]["dormant"] is True


def test_make_engine_from_env_leaves_api_key_unpinned_when_pool_present(tmp_path, monkeypatch):
    # A pinned api_key disables genai_keys.call_rotating at every Gemini call
    # site (embed/extract/disambiguate/supersede each do `if api_key: <single>
    # else: call_rotating(...)`), so a key pool must NOT be collapsed to one
    # key here -- that was the actual cause of dogfooding runs hitting 429
    # RESOURCE_EXHAUSTED on a single free-tier key under sustained real use.
    monkeypatch.delenv("GEMINI_EMBED_API_KEYS", raising=False)
    monkeypatch.setenv("GEMINI_API_KEYS", "key-a,key-b,key-c")
    monkeypatch.setenv("GEMINI_API_KEY", "key-a")
    engine = make_engine_from_env(str(tmp_path))
    assert engine.api_key is None
    assert engine.embedder._api_key is None


def test_make_engine_from_env_pins_single_key_when_no_pool(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_EMBED_API_KEYS", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "solo-key")
    engine = make_engine_from_env(str(tmp_path))
    assert engine.api_key == "solo-key"
    assert engine.embedder._api_key == "solo-key"


def test_make_engine_from_env_falls_back_offline_with_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_EMBED_API_KEYS", raising=False)
    engine = make_engine_from_env(str(tmp_path))
    assert isinstance(engine.embedder, HashingEmbedder)


def test_make_engine_from_env_uses_separate_embed_pool_when_set(tmp_path, monkeypatch):
    # GEMINI_EMBED_API_KEYS must be its OWN rotation pool, independent of
    # GEMINI_API_KEYS -- Gemini quotas are per-model, so a main-pool key
    # exhausted on the embedding model's daily quota shouldn't dilute (or be
    # diluted by) a dedicated embedding pool, and vice versa for the
    # generative calls the main pool serves.
    monkeypatch.setenv("GEMINI_API_KEYS", "main-a,main-b")
    monkeypatch.setenv("GEMINI_EMBED_API_KEYS", "embed-x,embed-y,embed-z")
    engine = make_engine_from_env(str(tmp_path))
    # main pool stays unpinned for extract/disambiguate/supersede
    assert engine.api_key is None
    # embedder rotates its OWN pool, not the main one
    assert engine.embedder._api_key is None
    assert engine.embedder._keys == ["embed-x", "embed-y", "embed-z"]


def test_make_engine_from_env_embed_pool_works_even_with_single_pinned_main_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "solo-main-key")
    monkeypatch.setenv("GEMINI_EMBED_API_KEYS", "embed-x,embed-y")
    engine = make_engine_from_env(str(tmp_path))
    # main single key still pins the engine (extract/disambiguate/supersede)
    assert engine.api_key == "solo-main-key"
    # but the embedder ignores that pin entirely and rotates its own pool
    assert engine.embedder._api_key is None
    assert engine.embedder._keys == ["embed-x", "embed-y"]


def test_make_engine_from_env_embed_pool_ignored_when_blank(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "solo-key")
    monkeypatch.setenv("GEMINI_EMBED_API_KEYS", "  ")
    engine = make_engine_from_env(str(tmp_path))
    assert engine.embedder._api_key == "solo-key"
    assert engine.embedder._keys is None


def test_make_engine_from_env_uses_openai_compatible_servers(tmp_path, monkeypatch):
    for key in ("GEMINI_API_KEYS", "GEMINI_API_KEY", "GEMINI_EMBED_API_KEYS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:18000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-secret")
    monkeypatch.setenv("OPENAI_MODEL", "ontomem-llm")
    monkeypatch.setenv("OPENAI_EMBED_BASE_URL", "http://localhost:18001/v1")
    monkeypatch.setenv("OPENAI_EMBED_API_KEY", "embed-secret")
    monkeypatch.setenv("OPENAI_EMBED_MODEL", "ontomem-embed")

    engine = make_engine_from_env(str(tmp_path))

    assert isinstance(engine.embedder, OpenAICompatibleEmbedder)
    assert engine.embedder.model_id == "ontomem-embed"
    assert engine.embedder._api_key == "embed-secret"
    assert engine.model == "ontomem-llm"
    assert engine.api_key is None
    assert engine.generate_fn is not None


def test_openai_embedding_defaults_to_generation_base_url(tmp_path, monkeypatch):
    for key in (
        "GEMINI_API_KEYS", "GEMINI_API_KEY", "GEMINI_EMBED_API_KEYS",
        "OPENAI_EMBED_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "shared-secret")
    monkeypatch.setenv("OPENAI_MODEL", "provider-chat-model")
    monkeypatch.setenv("OPENAI_EMBED_MODEL", "provider-embedding-model")

    engine = make_engine_from_env(str(tmp_path))

    assert engine.embedder.base_url == "https://provider.example/v1"
    assert engine.embedder._api_key == "shared-secret"


def test_openai_endpoint_requires_explicit_model(tmp_path, monkeypatch):
    for key in ("OPENAI_MODEL", "ONTOMEM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    with pytest.raises(RuntimeError, match="OPENAI_MODEL"):
        make_engine_from_env(str(tmp_path))
