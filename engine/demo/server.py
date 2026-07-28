"""Tiny stdlib HTTP server for the memory-graph demo. No deps, no API key.

  python demo/server.py            # then open http://127.0.0.1:8800

Routes:
  GET /              -> the visualisation (index.html)
  GET /graph         -> full graph JSON for the initial render
  GET /trace?q=...   -> ordered animation trace of a real Read for query q

By default it uses the offline HashingEmbedder (deterministic, lexical) so the
demo runs with zero setup. Set ONTOMEM_DEMO_GEMINI=1 (and GEMINI_API_KEY) to seed
with real semantic embeddings instead.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_data import build_demo_index, build_demo_store
from demo_trace import graph_json, trace_read

from ontomem.embeddings import HashingEmbedder

_HERE = Path(__file__).resolve().parent


def _make_embedder():
    if os.environ.get("ONTOMEM_DEMO_GEMINI") == "1":
        from ontomem.embeddings import GeminiEmbedder

        # load engine/.env so GEMINI_API_KEY(S) resolve
        env = _HERE.parents[0] / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.strip() and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
        print("demo embedder: gemini-embedding-001 (semantic)")
        return GeminiEmbedder()
    print("demo embedder: hashing (offline, lexical)")
    return HashingEmbedder()


STORE = build_demo_store()
EMBEDDER = _make_embedder()
INDEX = build_demo_index(STORE, EMBEDDER)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802 (stdlib name)
        route = urlparse(self.path)
        if route.path in ("/", "/index.html"):
            return self._send(200, (_HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if route.path == "/graph":
            return self._send(200, json.dumps(graph_json(STORE)))
        if route.path == "/trace":
            q = (parse_qs(route.query).get("q") or [""])[0]
            if not q.strip():
                return self._send(400, json.dumps({"error": "missing query ?q="}))
            return self._send(200, json.dumps(trace_read(q, STORE, INDEX, EMBEDDER)))
        return self._send(404, json.dumps({"error": "not found"}))

    def log_message(self, *args):  # quiet the per-request console spam
        return


PRESETS = [
    "Tell me about my trip to Rome",
    "Who is my manager?",
    "Tell me about Diocletian",
    "What do Priya and I talk about?",
    "How is the on-call situation?",
    "What's my partner's name?",
]


def main():
    host = os.environ.get("ONTOMEM_DEMO_HOST", "127.0.0.1")
    port = int(os.environ.get("ONTOMEM_DEMO_PORT", "8800"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"\n  memory-graph demo: http://{host}:{port}\n  ({len(STORE.nodes)} nodes, {len(STORE.edges)} edges)\n  Ctrl-C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
