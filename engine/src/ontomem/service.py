"""Local HTTP service exposing the host-agnostic engine contract (spec §11).

Plain JSON in, plain JSON out — the only seam the opencode plugin adapter talks
to. Stdlib http.server only (no web framework) to keep the engine dependency
light. Request routing is factored into a pure `dispatch()` so it is unit-tested
without sockets.

Endpoints (all POST, JSON body):
  /read            {"message": str}                -> {memory_block, context_block, text, trace}
  /write           {"conversation": [turn, ...]}   -> write stats
  /retrieve_memory {"node_name": str, "depth": int}-> neighbourhood
  /decay           {}                              -> {edges_decayed, edges_dormant}
  /health          {}                              -> {ok: true, nodes, edges}

GET endpoints (read-only, for the memory viewer):
  /graph           -> {nodes: [...], edges: [...]}  live snapshot, strength decayed to "now"
  /viewer, /        -> the viewer HTML page (engine/src/ontomem/static/viewer.html)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .decay import DORMANCY_THRESHOLD, decayed_strength
from .embeddings import HashingEmbedder, OpenAICompatibleEmbedder
from .engine import Engine
from .inference import OpenAICompatibleGenerator

_STATIC_DIR = Path(__file__).resolve().parent / "static"


class ServiceError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def dispatch(engine: Engine, path: str, payload: dict) -> dict:
    """Pure routing from (path, payload) to an engine call. Raises ServiceError
    with an HTTP status on bad input or unknown route."""
    if path == "/read":
        message = payload.get("message")
        if not isinstance(message, str):
            raise ServiceError(400, "'message' (str) is required")
        return engine.read(message)
    if path == "/write":
        conversation = payload.get("conversation")
        if not isinstance(conversation, list):
            raise ServiceError(400, "'conversation' (list of turns) is required")
        return engine.write(conversation)
    if path == "/retrieve_memory":
        name = payload.get("node_name")
        if not isinstance(name, str):
            raise ServiceError(400, "'node_name' (str) is required")
        return engine.retrieve_memory(name, int(payload.get("depth", 1)))
    if path == "/decay":
        return engine.decay()
    if path == "/health":
        return {"ok": True, "nodes": len(engine.store.nodes), "edges": len(engine.store.edges)}
    raise ServiceError(404, f"unknown route: {path}")


def _parse_iso(ts: str):
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def graph_snapshot(engine: Engine) -> dict:
    """Live, read-only view of the whole graph for the memory viewer. Edge
    strength is decayed to "now" (not the stale value from last write/decay
    run) so the viewer reflects the real current state (spec §6)."""
    now = datetime.now(timezone.utc)
    nodes = [
        {
            "id": node.key,
            "label": node.name,
            "kind": node.kind,
            "aliases": node.aliases,
            "properties": node.properties,
            "confidence": node.confidence,
        }
        for node in engine.store.nodes.values()
    ]
    edges = []
    for edge in engine.store.edges.values():
        updated = _parse_iso(edge.updated_at)
        elapsed_days = max((now - updated).total_seconds() / 86400, 0.0) if updated else 0.0
        strength = decayed_strength(edge.strength, edge.stability, elapsed_days)
        edges.append(
            {
                "id": edge.key,
                "from": edge.source_key,
                "to": edge.target_key,
                "relation": edge.relation,
                "strength": round(strength, 2),
                "confidence": edge.confidence,
                "stability": edge.stability,
                "cardinality": edge.cardinality,
                "snippet": edge.snippet,
                "dormant": strength < DORMANCY_THRESHOLD,
                "superseded_by": edge.properties.get("superseded_by"),
                "updated_at": edge.updated_at,
            }
        )
    return {"nodes": nodes, "edges": edges}


def _parse_key_pool(env_var: str) -> list[str] | None:
    raw = os.environ.get(env_var, "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return keys or None


def make_engine_from_env(base_dir: str | None = None) -> Engine:
    """Build an Engine using Gemini if a key is present, else the offline
    HashingEmbedder. Keeps the service runnable with or without network.

    When `GEMINI_API_KEYS` (a pool) is set, api_key is deliberately left None
    rather than pinned to one key: every Gemini call site (embed, extract,
    disambiguate, supersede) checks `if api_key: <single key> else:
    genai_keys.call_rotating(...)`, so pinning here silently disables rotation
    across the whole engine. Confirmed via dogfooding: under sustained real
    use a single pinned key hits the free tier's 429 RESOURCE_EXHAUSTED after
    a handful of conversations, and the write fails with no visible signal to
    the user. A lone `GEMINI_API_KEY` with no pool still pins to that one key
    (nothing to rotate across).

    `GEMINI_EMBED_API_KEYS`, if set, is a SEPARATE rotation pool used only for
    embedding calls, independent of GEMINI_API_KEYS (used by extract/
    disambiguate/supersede). Worth keeping separate rather than merging into
    one pool: Gemini's quotas are per-model, so a key can be fully exhausted
    on the embedding model's daily quota while still working fine for the
    generative model, and vice versa -- merging pools would let embedding
    traffic (far higher volume: every /read noun, every write's Stage 0
    context) burn through keys the generative calls also depend on."""
    base_dir = base_dir or os.environ.get("ONTOMEM_DIR", "./ontomem_data")
    single_key = os.environ.get("GEMINI_API_KEY")
    has_pool = bool(os.environ.get("GEMINI_API_KEYS", "").strip())
    embed_keys = _parse_key_pool("GEMINI_EMBED_API_KEYS")
    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ONTOMEM_LLM_API_KEY")
    llm_base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("ONTOMEM_LLM_BASE_URL")
    if openai_key and not llm_base_url:
        llm_base_url = "https://api.openai.com/v1"
    model = os.environ.get("OPENAI_MODEL") or os.environ.get("ONTOMEM_MODEL")
    if llm_base_url and not model:
        raise RuntimeError(
            "OPENAI_MODEL is required with an OpenAI-compatible inference endpoint"
        )
    model = model or "gemini-3.1-flash-lite"

    embed_model = os.environ.get("OPENAI_EMBED_MODEL") or os.environ.get("ONTOMEM_EMBED_MODEL")
    embed_base_url = (
        os.environ.get("OPENAI_EMBED_BASE_URL")
        or os.environ.get("ONTOMEM_EMBED_BASE_URL")
        or (llm_base_url if embed_model else None)
    )
    embed_key = (
        os.environ.get("OPENAI_EMBED_API_KEY")
        or os.environ.get("ONTOMEM_EMBED_API_KEY")
        or openai_key
    )
    generate_fn = None
    if llm_base_url:
        audit_path = os.environ.get("ONTOMEM_LLM_AUDIT_PATH")
        generator = OpenAICompatibleGenerator(
            llm_base_url,
            api_key=openai_key,
            timeout=float(os.environ.get("ONTOMEM_LLM_TIMEOUT", "300")),
            max_tokens=int(os.environ.get("ONTOMEM_LLM_MAX_TOKENS", "8192")),
            audit_path=audit_path,
            audit_content=os.environ.get("ONTOMEM_LLM_AUDIT_CONTENT") == "1",
        )
        generate_fn = generator.generate

    if embed_base_url:
        if not embed_model:
            raise RuntimeError(
                "OPENAI_EMBED_MODEL is required with an OpenAI-compatible embedding endpoint"
            )
        embedder = OpenAICompatibleEmbedder(
            embed_base_url,
            model=embed_model,
            api_key=embed_key,
            timeout=float(os.environ.get("ONTOMEM_EMBED_TIMEOUT", "120")),
        )
    elif has_pool or single_key or embed_keys:
        from .embeddings import GeminiEmbedder

        pinned_key = None if has_pool else single_key
        embedder = GeminiEmbedder(api_key=None if embed_keys else pinned_key, keys=embed_keys)
    else:
        pinned_key = None
        embedder = HashingEmbedder()

    pinned_key = None if has_pool or llm_base_url else single_key
    return Engine(
        base_dir, embedder=embedder, model=model, api_key=pinned_key,
        generate_fn=generate_fn,
    )


def _make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence default stderr logging
            pass

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/graph":
                return self._send(200, graph_snapshot(engine))
            if path == "/last_read":
                return self._send(200, engine.last_read or {})
            if path in ("/", "/viewer"):
                viewer = _STATIC_DIR / "viewer.html"
                if not viewer.exists():
                    return self._send(404, {"error": "viewer.html not found"})
                return self._send(200, viewer.read_bytes(), "text/html; charset=utf-8")
            return self._send(404, {"error": f"unknown route: {path}"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw or b"{}")
                response = dispatch(engine, self.path, payload)
                self._send(200, response)
            except ServiceError as exc:
                self._send(exc.status, {"error": exc.message})
            except json.JSONDecodeError:
                self._send(400, {"error": "invalid JSON body"})
            except Exception as exc:  # surface engine errors as 500 with a message
                self._send(500, {"error": str(exc)})

        def _send(self, status: int, body, content_type: str = "application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def run(host: str = "127.0.0.1", port: int = 8765, base_dir: str | None = None) -> None:
    engine = make_engine_from_env(base_dir)
    server = ThreadingHTTPServer((host, port), _make_handler(engine))
    print(f"ontomem service on http://{host}:{port}  (dir={engine.dir})")
    server.serve_forever()


if __name__ == "__main__":
    run(
        host=os.environ.get("ONTOMEM_HOST", "127.0.0.1"),
        port=int(os.environ.get("ONTOMEM_PORT", "8765")),
    )
