"""Unit tests for the embedding layer (deterministic HashingEmbedder)."""

import sys
import types

import numpy as np

from ontomem.embeddings import EmbeddingIndex, HashingEmbedder, LocalEmbedder, _batched, _l2_normalise
from ontomem.model import Node
from ontomem.store import Store


def _store():
    store = Store()
    store.add_node(Node.create("ORG", "Walmart"))
    store.add_node(Node.create("PERSON", "Sarah"))
    store.add_node(Node.create("PLACE", "New Jersey"))
    return store


def test_hashing_embedder_is_deterministic_and_normalised():
    emb = HashingEmbedder(dim=64)
    a = emb.embed(["Walmart"])
    b = emb.embed(["Walmart"])
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)


def test_identical_text_cosine_is_one():
    emb = HashingEmbedder()
    v = emb.embed(["Walmart", "Walmart"])
    assert np.isclose(v[0] @ v[1], 1.0)


def test_index_search_finds_exact_match():
    store = _store()
    emb = HashingEmbedder()
    index = EmbeddingIndex.build(store, emb)
    assert len(index) == 3
    query = emb.embed(["Walmart"])[0]
    hits = index.search(query, top_k=3, threshold=0.5)
    assert hits[0][0] == "ORG::walmart"
    assert np.isclose(hits[0][1], 1.0)


def test_index_threshold_filters():
    store = _store()
    emb = HashingEmbedder()
    index = EmbeddingIndex.build(store, emb)
    query = emb.embed(["CompletelyUnrelatedToken"])[0]
    # nothing should clear a high threshold
    assert index.search(query, threshold=0.9) == []


def test_empty_store_index_is_empty():
    index = EmbeddingIndex.build(Store(), HashingEmbedder())
    assert len(index) == 0
    assert index.search(np.zeros(8, dtype=np.float32)) == []


def test_batched_respects_limit():
    # Gemini caps embedding batches at 100; the helper must never exceed it.
    items = list(range(250))
    batches = list(_batched(items, 100))
    assert [len(b) for b in batches] == [100, 100, 50]
    assert [x for b in batches for x in b] == items  # order + completeness preserved


class _CountingEmbedder(HashingEmbedder):
    def __init__(self, dim=64):
        super().__init__(dim)
        self.embed_calls = 0
        self.texts_embedded = 0

    def embed(self, texts):
        self.embed_calls += 1
        self.texts_embedded += len(texts)
        return super().embed(texts)


def test_update_embeds_only_new_nodes():
    store = _store()  # Walmart, Sarah, New Jersey
    emb = _CountingEmbedder()
    index = EmbeddingIndex.build(store, emb)
    assert emb.texts_embedded == 3

    # add one node; update must embed only the new one, not all four
    store.add_node(Node.create("PERSON", "Bill"))
    emb.embed_calls = 0
    emb.texts_embedded = 0
    index.update(store, emb)
    assert emb.texts_embedded == 1  # only Bill re-embedded
    assert set(index.keys) == set(store.nodes)


def test_update_drops_removed_nodes():
    store = _store()
    emb = HashingEmbedder()
    index = EmbeddingIndex.build(store, emb)
    del store.nodes["PERSON::sarah"]
    index.update(store, emb)
    assert "PERSON::sarah" not in index.keys
    assert len(index) == 2


def test_update_rebuilds_on_model_change():
    store = _store()
    index = EmbeddingIndex.build(store, HashingEmbedder(dim=64))
    other = HashingEmbedder(dim=128)  # different model_id
    rebuilt = index.update(store, other)
    assert rebuilt.model_id == other.model_id
    assert len(rebuilt) == 3


def _anisotropic_index() -> EmbeddingIndex:
    # three nodes sharing a large common component -> raw pairwise cosine is high
    # for ALL pairs (simulates gemini-embedding-001 anisotropy, ~0.55 baseline).
    bias = np.array([5, 5, 5, 5], dtype=np.float32)
    raw = _l2_normalise(np.array(
        [bias + [1, 0, 0, 0], bias + [0, 1, 0, 0], bias + [0, 0, 1, 0]], dtype=np.float32
    ))
    index = EmbeddingIndex("synthetic")
    index._keys = ["A", "B", "C"]
    index._matrix = raw
    index._recompute_centered()
    return index


def test_raw_search_floods_under_anisotropy():
    index = _anisotropic_index()
    query = index._matrix[0]  # exactly node A
    hits = index.search(query, top_k=3, threshold=0.5)
    # the shared component makes B and C also clear 0.5 -> flooding
    assert len(hits) == 3


def test_centered_search_suppresses_anisotropic_baseline():
    index = _anisotropic_index()
    query = index._matrix[0]  # exactly node A (raw, pre-centering)
    hits = index.search(query, top_k=3, threshold=0.30, center=True)
    # centring collapses the unrelated baseline: only A clears the threshold
    assert [k for k, _ in hits] == ["A"]
    assert hits[0][1] > 0.30


def test_centered_search_preserves_exact_match_rank():
    index = _anisotropic_index()
    query = index._matrix[1]  # node B
    hits = index.search(query, top_k=3, threshold=0.0, center=True)
    assert hits[0][0] == "B"  # correct node still ranks first


def test_centering_survives_sidecar_roundtrip(tmp_path):
    index = _anisotropic_index()
    path = tmp_path / "emb.npz"
    index.save(path)
    loaded = EmbeddingIndex.load(path)
    hits = loaded.search(index._matrix[2], top_k=3, threshold=0.30, center=True)
    assert [k for k, _ in hits] == ["C"]  # centred view rebuilt on load


def test_build_keyed_holds_mixed_node_and_relation_keys():
    emb = HashingEmbedder()
    index = EmbeddingIndex.build_keyed(
        ["PERSON::alice", "REL::HAS_PARTNER"], ["Alice", "has partner"], emb
    )
    assert set(index.keys) == {"PERSON::alice", "REL::HAS_PARTNER"}


def test_sync_adds_new_drops_removed_embeds_only_new():
    emb = _CountingEmbedder()
    index = EmbeddingIndex.build_keyed(["A", "REL::X"], ["alpha", "rel x"], emb)
    emb.texts_embedded = 0
    # drop REL::X, keep A, add REL::Y -> only REL::Y is embedded
    index.sync({"A": "alpha", "REL::Y": "rel y"}, emb)
    assert set(index.keys) == {"A", "REL::Y"}
    assert emb.texts_embedded == 1


def test_index_sidecar_roundtrip(tmp_path):
    store = _store()
    emb = HashingEmbedder()
    index = EmbeddingIndex.build(store, emb)
    path = tmp_path / "emb.npz"
    index.save(path)
    loaded = EmbeddingIndex.load(path)
    assert loaded.model_id == emb.model_id
    assert set(loaded.keys) == set(index.keys)
    query = emb.embed(["Sarah"])[0]
    assert loaded.search(query, threshold=0.5)[0][0] == "PERSON::sarah"


def test_local_embedder_lazy_loads_fastembed_and_normalises(monkeypatch):
    """fastembed isn't a hard dependency (it's behind the `local` extra), so
    this injects a fake module into sys.modules rather than requiring it
    installed -- same trick test_inference.py uses for urllib.request."""
    calls = []
    vectors_by_text = {"a": [1.0, 0.0], "b": [0.0, 2.0]}

    class FakeTextEmbedding:
        def __init__(self, model_name, cache_dir=None):
            calls.append((model_name, cache_dir))

        def embed(self, texts):
            return (np.array(vectors_by_text[t]) for t in texts)

    fake_module = types.ModuleType("fastembed")
    fake_module.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake_module)

    embedder = LocalEmbedder(cache_dir="/tmp/ontomem-fastembed-cache")
    vectors = embedder.embed(["a", "b"])

    assert calls == [("BAAI/bge-small-en-v1.5", "/tmp/ontomem-fastembed-cache")]
    np.testing.assert_allclose(vectors, [[1.0, 0.0], [0.0, 1.0]])

    embedder.embed(["a"])
    assert len(calls) == 1, "model should be loaded once and reused, not per embed() call"


def test_local_embedder_default_model_id():
    assert LocalEmbedder().model_id == "BAAI/bge-small-en-v1.5"
