"""Local HTTP service exposing the host-agnostic engine contract (spec §11).

Plain JSON in, plain JSON out — the only seam the opencode plugin adapter talks
to. Stdlib http.server only (no web framework) to keep the engine dependency
light. Request routing is factored into a pure `dispatch()` so it is unit-tested
without sockets.

Endpoints (all POST, JSON body):
  /read            {"message": str}                -> {memory_block, context_block, text, trace}
  /write           {"conversation": [turn, ...]}   -> write stats
  /retrieve_memory {"node_name": str, "depth": int}-> neighbourhood
  /agentic/start    {"conversation": [...]}         -> {session_id, prompt_text}
  /agentic/tool_call {"session_id", "tool_name", "arguments"} -> {result_text, finished, stats?}
    -- split-request agentic extraction (spec's sequential-commit fix, see
    engine.py's Engine.start_agentic_write/apply_agentic_tool_call): the
    caller (the opencode plugin, driving opencode's own native tool-calling
    loop) calls /agentic/start once, then /agentic/tool_call once per
    add_entity/add_relationship/finish_extraction call the model makes.
    Each call commits immediately; `finished` flips true (with `stats`)
    once finish_extraction lands or the safety cap is hit. This is an
    alternative to /write for callers whose backend supports a native
    per-call tool loop; /write's single-shot batch path is unaffected.
  /decay           {}                              -> {edges_decayed, edges_dormant}
  /health          {}                              -> {ok: true, nodes, edges}

GET endpoints (read-only, for the memory viewer):
  /graph           -> {nodes: [...], edges: [...]}  live snapshot, strength decayed to "now"
  /viewer, /        -> the viewer HTML page (engine/src/ontomem/static/viewer.html)

Process lifecycle: this is a plain long-running process, not a system service
-- nothing here daemonises it, restarts it on crash, or starts it at boot.
The packaged opencode plugin spawns it detached (so it survives the spawning
opencode process exiting) and relies on ONTOMEM_IDLE_SHUTDOWN_MINUTES (see
run()) for it to wind itself down after actual inactivity, rather than
running forever on a machine no opencode session is using. A deployment that
wants it always-on regardless of activity (e.g. the cluster dogfooding setup)
sets that to 0.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .decay import DORMANCY_THRESHOLD, decayed_strength
from .embeddings import HashingEmbedder, OpenAICompatibleEmbedder
from .engine import Engine
from .inference import HostCallbackGenerator, OpenAICompatibleGenerator

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
    if path == "/agentic/start":
        conversation = payload.get("conversation")
        if not isinstance(conversation, list):
            raise ServiceError(400, "'conversation' (list of turns) is required")
        return engine.start_agentic_write(conversation)
    if path == "/agentic/tool_call":
        session_id = payload.get("session_id")
        tool_name = payload.get("tool_name")
        arguments = payload.get("arguments")
        if not isinstance(session_id, str):
            raise ServiceError(400, "'session_id' (str) is required")
        if not isinstance(tool_name, str):
            raise ServiceError(400, "'tool_name' (str) is required")
        if not isinstance(arguments, dict):
            raise ServiceError(400, "'arguments' (dict) is required")
        return engine.apply_agentic_tool_call(session_id, tool_name, arguments)
    if path == "/refresh_callback":
        url = payload.get("url")
        if not isinstance(url, str) or not url:
            raise ServiceError(400, "'url' (str) is required")
        engine.refresh_callback_url(url)
        return {"ok": True}
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
                "properties": dict(edge.properties),
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
    chat_fn = None
    extraction_prompt_template = None
    # Opt-in, mutually exclusive with ONTOMEM_EXTRACTION_TOOL_CALLING: the
    # agentic (per-item) extraction mode dispatches entirely inside
    # Engine._write_agentic before generate_fn/extraction_prompt_template are
    # ever consulted, so it needs only chat_fn wired -- see engine.py's
    # write().
    agentic_extraction = os.environ.get("ONTOMEM_AGENTIC_EXTRACTION") == "1"
    # ONTOMEM_HOST_CALLBACK_URL: delegate generation to a local host process
    # (the opencode plugin) instead of any model API the engine itself holds
    # a key for -- the host fulfills each call using whatever provider it's
    # already authenticated for (e.g. the user's own OpenCode Zen/Go plan).
    # Mutually exclusive with the OpenAI-compatible cluster path below: they
    # are two different deployment stories (self-hosted backend vs. riding
    # on the host's own model access), not something to combine.
    host_callback_url = os.environ.get("ONTOMEM_HOST_CALLBACK_URL")
    # Kept as a reference (not just its bound .generate) so it can be handed
    # to Engine for refresh_callback_url() to mutate later -- see that
    # method's docstring for why this engine's callback target needs to be
    # refreshable independent of this process's own startup-time env var.
    host_callback_generator = None
    if host_callback_url:
        if agentic_extraction:
            raise RuntimeError(
                "ONTOMEM_AGENTIC_EXTRACTION=1 is not supported with ONTOMEM_HOST_CALLBACK_URL "
                "-- the agentic tool-calling loop needs chat_fn, which the host callback "
                "contract (single prompt in, text out) does not provide"
            )
        host_callback_generator = HostCallbackGenerator(
            host_callback_url,
            # 120s (the old default) is too short for a reasoning model's
            # single-shot extraction call -- confirmed live, a 3-turn test
            # conversation's plain-mode extraction took ~204s end to end,
            # and agentic mode's first tool call separately measured at
            # ~236s (see plugin/src/index.ts's default-flip comment). 300s
            # covers both with margin; still overridable for longer
            # conversations that need more.
            timeout=float(os.environ.get("ONTOMEM_HOST_CALLBACK_TIMEOUT", "300")),
            audit_path=os.environ.get("ONTOMEM_LLM_AUDIT_PATH"),
            audit_content=os.environ.get("ONTOMEM_LLM_AUDIT_CONTENT") == "1",
        )
        generate_fn = host_callback_generator.generate
    elif llm_base_url:
        audit_path = os.environ.get("ONTOMEM_LLM_AUDIT_PATH")
        # Opt-in: forces extraction through a tool-call JSON Schema instead of
        # freeform JSON, and swaps in the motivation-driven reasoning prompt.
        # Off by default so existing OpenAI-compatible setups (e.g. a backend
        # without reliable tool-call support) are unaffected. Gemini's path
        # never touches this.
        use_tool_calling = os.environ.get("ONTOMEM_EXTRACTION_TOOL_CALLING") == "1"
        extraction_tool_schema = None
        if use_tool_calling:
            from .extraction_schema import EXTRACTION_TOOL_SCHEMA
            from .extractor_prompt_reasoning import EXTRACTOR_REASONING_PROMPT

            extraction_tool_schema = EXTRACTION_TOOL_SCHEMA
            extraction_prompt_template = EXTRACTOR_REASONING_PROMPT
        generator = OpenAICompatibleGenerator(
            llm_base_url,
            api_key=openai_key,
            timeout=float(os.environ.get("ONTOMEM_LLM_TIMEOUT", "300")),
            max_tokens=int(os.environ.get("ONTOMEM_LLM_MAX_TOKENS", "8192")),
            audit_path=audit_path,
            audit_content=os.environ.get("ONTOMEM_LLM_AUDIT_CONTENT") == "1",
            extraction_tool_schema=extraction_tool_schema,
        )
        generate_fn = generator.generate
        if agentic_extraction:
            chat_fn = generator.chat_with_tools
    elif agentic_extraction:
        raise RuntimeError(
            "ONTOMEM_AGENTIC_EXTRACTION=1 requires an OpenAI-compatible inference "
            "endpoint (set OPENAI_BASE_URL or ONTOMEM_LLM_BASE_URL)"
        )

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
        # No explicit embedding backend configured -- this is the packaged
        # plugin's default path (host-callback generation, no embedding API
        # of its own to reuse). Prefer a real local semantic embedder over
        # pure lexical hashing when the `local` extra (fastembed) is
        # installed; fall back to HashingEmbedder only if it truly isn't.
        # The presence check happens here, eagerly, rather than deferring to
        # LocalEmbedder's own lazy import inside embed() -- surfacing a
        # missing dependency at service startup beats an opaque failure deep
        # inside the first retrieval/write call.
        pinned_key = None
        try:
            import fastembed  # noqa: F401

            from .embeddings import LocalEmbedder

            embedder = LocalEmbedder(cache_dir=os.path.join(base_dir, "fastembed_cache"))
        except ImportError:
            embedder = HashingEmbedder()

    pinned_key = None if has_pool or llm_base_url else single_key
    return Engine(
        base_dir, embedder=embedder, model=model, api_key=pinned_key,
        generate_fn=generate_fn, chat_fn=chat_fn, extraction_prompt_template=extraction_prompt_template,
        user_only_extraction=os.environ.get("ONTOMEM_USER_ONLY_EXTRACTION") == "1",
        agentic_extraction=agentic_extraction,
        host_callback_generator=host_callback_generator,
    )


class _ActivityTracker:
    """Tracks time since the last request, across the ThreadingHTTPServer's
    one-thread-per-request model -- see _idle_watchdog. Lock-protected since
    `touch()` (every request thread) and `idle_seconds()` (the single
    watchdog thread) run concurrently."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last = time.monotonic()

    def touch(self) -> None:
        with self._lock:
            self._last = time.monotonic()

    def idle_seconds(self) -> float:
        with self._lock:
            return time.monotonic() - self._last


def _idle_watchdog(activity: _ActivityTracker, *, limit_seconds: float, poll_seconds: float = 60.0, exit_fn=None) -> None:
    """Runs on its own daemon thread (see run()): polls every `poll_seconds`
    and calls `exit_fn` once the server has gone `limit_seconds` without a
    request. Exists so the packaged opencode plugin's detached-spawned
    service (see plugin/src/index.ts's startEngineService) doesn't run
    forever on a machine no opencode session is actually using -- it winds
    itself down, and the plugin's own health-check-and-restart-if-needed
    logic brings it back on the next real use. `exit_fn` is injectable so
    this loop is unit-testable without actually killing the test process;
    production default is os._exit(0) (a plain HTTP-serving daemon thread has
    nothing to flush -- every write already persists synchronously in
    Engine._persist -- so a hard exit is safe and simpler than coordinating a
    graceful ThreadingHTTPServer.shutdown() from a different thread)."""
    exit_fn = exit_fn or (lambda: os._exit(0))
    while True:
        time.sleep(poll_seconds)
        if activity.idle_seconds() >= limit_seconds:
            exit_fn()
            return


def _make_handler(engine: Engine, activity: _ActivityTracker | None = None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence default stderr logging
            pass

        def do_GET(self):
            if activity:
                activity.touch()
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
            if activity:
                activity.touch()
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
    activity = _ActivityTracker()
    server = ThreadingHTTPServer((host, port), _make_handler(engine, activity))

    # ONTOMEM_IDLE_SHUTDOWN_MINUTES: how long with no request before this
    # process exits itself (0 disables it -- always-on, e.g. the cluster
    # dogfooding deployment). Defaults on for the packaged-plugin path, where
    # a detached, spawn-and-forget service would otherwise run forever on a
    # machine no opencode session is using; the plugin's own health-check
    # brings it back on the next real use.
    idle_minutes = float(os.environ.get("ONTOMEM_IDLE_SHUTDOWN_MINUTES", "60"))
    if idle_minutes > 0:
        watchdog = threading.Thread(
            target=_idle_watchdog, args=(activity,), kwargs={"limit_seconds": idle_minutes * 60}, daemon=True,
        )
        watchdog.start()

    print(f"ontomem service on http://{host}:{port}  (dir={engine.dir}, idle_shutdown={idle_minutes}min)")
    server.serve_forever()


if __name__ == "__main__":
    run(
        host=os.environ.get("ONTOMEM_HOST", "127.0.0.1"),
        port=int(os.environ.get("ONTOMEM_PORT", "8765")),
    )
