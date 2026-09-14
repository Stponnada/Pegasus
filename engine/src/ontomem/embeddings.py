"""Embedding layer for seed finding (spec v0.2.1 §5.2). Brute-force numpy cosine.

The embedder is an injectable interface so the retriever's graph mechanics can be
tested without a network. `HashingEmbedder` is a deterministic, dependency-free
embedder (lexical, not semantic) used for hermetic tests; `GeminiEmbedder` and
`OpenAICompatibleEmbedder` are live, remote ones; `LocalEmbedder` is live but
fully offline (runs on-device, no API key, no data leaves the machine).
Embeddings are stored in a sidecar keyed by node id, alongside the
embedding-model id (open problem 9.4: detect drift, re-embed on model change).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

from .inference import _endpoint, post_json

GEMINI_EMBED_MODEL = "gemini-embedding-001"
EMBED_BATCH = 100  # Gemini BatchEmbedContents hard limit: <=100 texts per request


def _batched(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


class Embedder(Protocol):
    model_id: str

    def embed(self, texts: list[str]) -> np.ndarray:  # shape (n, dim), L2-normalised
        ...


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class HashingEmbedder:
    """Deterministic hashing embedder — lexical, not semantic. Identical strings
    map to identical vectors (cosine 1.0); unrelated strings are near-orthogonal.
    For hermetic tests and as an offline fallback only."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.model_id = f"hashing-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in _tokenise(text):
                h = hashlib.sha1(token.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "big") % self.dim
                sign = 1.0 if h[4] & 1 else -1.0
                vecs[i, idx] += sign
        return _l2_normalise(vecs)


class GeminiEmbedder:
    """Live embedder using the Gemini embedding API.

    `keys`, if given, is a dedicated rotation pool for THIS embedder only —
    separate from the global GEMINI_API_KEYS pool that extraction/merge/
    supersede draw from via genai_keys.load_keys(). Worth keeping separate:
    Gemini's rate limits are per-model, so a key exhausted on the embedding
    model's daily quota may still be perfectly fine for the generative model
    (gemini-3.1-flash-lite) used elsewhere -- mixing pools would dilute a
    fresh embedding-only key pool with keys that don't need the help, and
    vice versa. `api_key` (a single pinned key) takes precedence over `keys`
    if both are given, matching the rest of the codebase's convention."""

    def __init__(
        self,
        *,
        model: str = GEMINI_EMBED_MODEL,
        api_key: str | None = None,
        keys: list[str] | None = None,
    ) -> None:
        self.model_id = model
        self._api_key = api_key
        self._keys = keys

    def embed(self, texts: list[str]) -> np.ndarray:
        from google import genai

        from .genai_keys import call_rotating

        def embed_batch(batch: list[str], key: str | None):
            client = genai.Client(api_key=key)
            return client.models.embed_content(model=self.model_id, contents=batch)

        rows: list = []
        for batch in _batched(list(texts), EMBED_BATCH):
            # rotate keys per batch on rate-limit (free-tier embedding quota)
            result = (
                embed_batch(batch, self._api_key)
                if self._api_key
                else call_rotating(lambda key: embed_batch(batch, key), keys=self._keys)
            )
            rows.extend(e.values for e in (result.embeddings or []))
        vecs = np.array(rows, dtype=np.float32)
        if vecs.size == 0:
            raise RuntimeError("embedding model returned no vectors")
        return _l2_normalise(vecs)


LOCAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


class LocalEmbedder:
    """Fully local, offline semantic embedder using fastembed (ONNX runtime,
    no torch, no network call per embed -- only a one-time model-weight
    download on first use, cached on disk). This is the default embedder for
    the packaged opencode plugin (see service.py's make_engine_from_env):
    unlike GeminiEmbedder/OpenAICompatibleEmbedder it needs no API key and,
    more importantly, node/edge text never leaves the machine to be embedded
    -- there is no third-party embedding endpoint in the loop at all.

    The model is lazy-loaded on first `embed()` call (not in __init__), same
    convention as GeminiEmbedder's lazy `from google import genai`: importing
    this module, or constructing this class, must not require fastembed to
    be installed -- only actually embedding does."""

    def __init__(self, *, model: str = LOCAL_EMBED_MODEL, cache_dir: str | Path | None = None) -> None:
        self.model_id = model
        self._cache_dir = str(cache_dir) if cache_dir else None
        self._model = None

    def _loaded_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_id, cache_dir=self._cache_dir)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._loaded_model()
        vecs = np.array(list(model.embed(list(texts))), dtype=np.float32)
        if vecs.size == 0:
            raise RuntimeError("local embedding model returned no vectors")
        return _l2_normalise(vecs)


class OpenAICompatibleEmbedder:
    """Live embedder using an OpenAI Embeddings-compatible HTTP server."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        batch_size: int = 100,
    ) -> None:
        self.base_url = base_url
        self.model_id = model
        self._api_key = api_key
        self.timeout = timeout
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for batch in _batched(list(texts), self.batch_size):
            payload = post_json(
                _endpoint(self.base_url, "embeddings"),
                {"model": self.model_id, "input": batch, "encoding_format": "float"},
                api_key=self._api_key,
                timeout=self.timeout,
            )
            data = sorted(payload.get("data", []), key=lambda item: int(item.get("index", 0)))
            rows.extend(item["embedding"] for item in data)
        if len(rows) != len(texts):
            raise RuntimeError(
                f"embedding server returned {len(rows)} vectors for {len(texts)} inputs"
            )
        vecs = np.asarray(rows, dtype=np.float32)
        if vecs.ndim != 2 or vecs.size == 0:
            raise RuntimeError("embedding server returned no usable vectors")
        return _l2_normalise(vecs)


def _tokenise(text: str) -> list[str]:
    return [t for t in "".join(c if c.isalnum() else " " for c in text.lower()).split() if t]


class EmbeddingIndex:
    """In-memory map of node key -> unit vector, with brute-force cosine search.

    Persisted as a numpy .npz sidecar (NOT inline in the graph JSON), tagged with
    the embedding-model id so a model change can be detected and re-embedded.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._keys: list[str] = []
        self._matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        # mean-centred view of the matrix, for anisotropy-corrected seeding.
        # Semantic embedding spaces (e.g. gemini-embedding-001) are strongly
        # anisotropic: random pairs sit at cosine ~0.55, so an absolute threshold
        # over raw cosine seeds nearly every node. Subtracting the corpus mean
        # ("all-but-the-top" / common-component removal) collapses the unrelated
        # baseline toward 0 and restores a meaningful threshold. Derived from
        # _matrix (no extra API calls), recomputed whenever the matrix changes.
        self._mu: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._centered: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def _recompute_centered(self) -> None:
        n = self._matrix.shape[0] if self._matrix.ndim == 2 else 0
        if n == 0:
            self._mu = np.zeros((0,), dtype=np.float32)
            self._centered = np.zeros((0, 0), dtype=np.float32)
            return
        if n < 2:
            # a single vector centres to zero — degenerate. With nothing to
            # estimate the common component from, centring is a no-op (mu=0).
            self._mu = np.zeros((self._matrix.shape[1],), dtype=np.float32)
            self._centered = self._matrix.copy()
            return
        self._mu = self._matrix.mean(axis=0)
        self._centered = _l2_normalise(self._matrix - self._mu)

    @property
    def keys(self) -> list[str]:
        return list(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    @classmethod
    def build(cls, store, embedder: Embedder) -> "EmbeddingIndex":
        """Build a node-only index (key = node key, text = name + aliases)."""
        return cls.build_keyed(
            [n.key for n in store.nodes.values()],
            [_node_text(n) for n in store.nodes.values()],
            embedder,
        )

    @classmethod
    def build_keyed(cls, keys: list[str], texts: list[str], embedder: Embedder) -> "EmbeddingIndex":
        """Build an index from an arbitrary key->text mapping. Lets a single
        index hold both node-name vectors AND relation-phrase vectors (keyed
        `REL::<RELATION>`) so relation-aware seeding shares one well-estimated
        mean for centring (spec §5.2)."""
        index = cls(embedder.model_id)
        if not keys:
            return index
        index._keys = list(keys)
        index._matrix = embedder.embed(list(texts))
        index._recompute_centered()
        return index

    def sync(self, key_to_text: dict[str, str], embedder: Embedder) -> "EmbeddingIndex":
        """Incrementally reconcile the index to a key->text mapping: embed ONLY
        keys not already indexed, drop keys no longer present. Avoids re-embedding
        the whole corpus on every write (which would blow embedding quotas as the
        graph grows). If a key's text changes, drop its key before calling sync."""
        if self.model_id != embedder.model_id:
            return EmbeddingIndex.build_keyed(list(key_to_text), list(key_to_text.values()), embedder)
        keep = [(k, row) for k, row in zip(self._keys, self._matrix) if k in key_to_text]
        kept_keys = {k for k, _ in keep}
        new_keys = [k for k in key_to_text if k not in kept_keys]

        rows = [row for _, row in keep]
        keys = [k for k, _ in keep]
        if new_keys:
            new_rows = embedder.embed([key_to_text[k] for k in new_keys])
            rows = [*rows, *new_rows]
            keys = [*keys, *new_keys]
        self._keys = keys
        self._matrix = np.array(rows, dtype=np.float32) if rows else np.zeros((0, 0), dtype=np.float32)
        self._recompute_centered()
        return self

    def update(self, store, embedder: Embedder) -> "EmbeddingIndex":
        """Node-only incremental sync (back-compat). For the combined node+relation
        index the engine calls `sync` with both kinds of key."""
        return self.sync({n.key: _node_text(n) for n in store.nodes.values()}, embedder)

    def search(self, query_vec: np.ndarray, *, top_k: int = 5, threshold: float = 0.0, center: bool = False):
        """Return [(key, cosine_score)] above threshold, highest first.
        Vectors are unit-normalised, so cosine == dot product.

        `center=True` searches in the mean-centred space (anisotropy-corrected):
        the query is centred by the same corpus mean and renormalised. Used for
        seed finding on semantic embeddings, where raw cosine over-seeds."""
        if len(self._keys) == 0:
            return []
        if center:
            q = query_vec - self._mu
            q = q / (float(np.linalg.norm(q)) or 1.0)
            scores = self._centered @ q
        else:
            scores = self._matrix @ query_vec
        order = np.argsort(-scores)[:top_k]
        return [(self._keys[i], float(scores[i])) for i in order if scores[i] >= threshold]

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            model_id=np.array(self.model_id),
            keys=np.array(self._keys, dtype=object),
            matrix=self._matrix,
        )

    @classmethod
    def load(cls, path) -> "EmbeddingIndex":
        data = np.load(path, allow_pickle=True)
        index = cls(str(data["model_id"]))
        index._keys = list(data["keys"])
        index._matrix = data["matrix"]
        index._recompute_centered()
        return index


def _node_text(node) -> str:
    """The text embedded for a node: its name plus any aliases."""
    return " ".join([node.name, *node.aliases])


def relation_phrase(relation: str) -> str:
    """The text embedded for a relation label: the UPPER_SNAKE_CASE relation as a
    natural phrase, so a query noun seeds via the relation (spec §5.2: 'boss'
    surfaces Sarah through HAS_BOSS). `HAS_PARTNER` -> 'has partner'."""
    return relation.lower().replace("_", " ")
