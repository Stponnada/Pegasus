"""Drive a diverse dogfood corpus against a LIVE ontomem HTTP service (e.g. the
Qwen-backed engine served via the HPC bridge / `qwen-code`), so you can watch
exactly what goes in and judge output quality yourself instead of having it
run unattended.

Talks only the plain-JSON HTTP contract (spec v0.2.1 SS11) via stdlib
urllib -- no Gemini/google-genai import, no engine-internal import. Point it
at whatever ONTOMEM_URL is currently live.

Usage (from engine/):
    uv run python scripts/run_local_dogfood.py
    uv run python scripts/run_local_dogfood.py --url http://127.0.0.1:8765
    uv run python scripts/run_local_dogfood.py --filter lighthouse --filter hackathon
    uv run python scripts/run_local_dogfood.py --probes-only     # re-run probes on existing graph
    uv run python scripts/run_local_dogfood.py --skip-probes     # writes only

Every conversation is written IN ORDER (the corpus is one persona's graph
growing over time -- several probes assume earlier conversations already
landed). A failure on one conversation is logged and does not abort the rest.

Outputs, under --out-dir (default engine/dogfood_local/):
    graph_<timestamp>.json    -- full /graph snapshot after all writes
    report_<timestamp>.json   -- everything: transcripts, write results, probe results
    report_<timestamp>.md     -- human-readable version of the above, read this first
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dogfood_corpus_diverse import CONVERSATIONS, PROBES  # noqa: E402


def post(url: str, path: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{url}{path}", data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def get(url: str, path: str, timeout: float) -> dict:
    with urllib.request.urlopen(f"{url}{path}", timeout=timeout) as resp:
        return json.loads(resp.read())


def check_health(url: str) -> dict:
    try:
        return post(url, "/health", {}, timeout=10)
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        print(f"Cannot reach ontomem service at {url}: {exc}")
        print("Is it running? On the cluster side: `ontomem status`. On the Mac: `qwen-code` "
              "should already have brought up the local engine and bridge -- check "
              "`ontomem-bridge status all` and that ONTOMEM_URL / the service on 127.0.0.1:8765 "
              "is actually up.")
        sys.exit(1)


def to_turns(turns: list[tuple[str, str]]) -> list[dict]:
    return [{"role": role, "turn": i + 1, "text": text} for i, (role, text) in enumerate(turns)]


def run_writes(url: str, timeout: float, delay: float, name_filter: list[str] | None) -> list[dict]:
    results = []
    selected = CONVERSATIONS
    if name_filter:
        selected = [c for c in CONVERSATIONS if any(f in c[0] for f in name_filter)]
        print(f"Filter matched {len(selected)}/{len(CONVERSATIONS)} conversations: "
              f"{[c[0] for c in selected]}")

    for i, (title, turns) in enumerate(selected, 1):
        conversation = to_turns(turns)
        print(f"\n[{i}/{len(selected)}] writing '{title}' ({len(conversation)} turns)...")
        start = time.time()
        try:
            result = post(url, "/write", {"conversation": conversation}, timeout=timeout)
            elapsed = time.time() - start
            stats = {k: result.get(k) for k in (
                "nodes_created", "nodes_merged", "edges_created", "edges_reinforced",
                "edges_dropped", "edges_superseded", "flagged", "warnings",
            )}
            print(f"  done in {elapsed:.1f}s -- {json.dumps(stats)}")
            if result.get("warnings"):
                print(f"  WARNINGS: {result['warnings']}")
            results.append({"title": title, "turns": turns, "elapsed_s": round(elapsed, 1),
                             "result": result, "error": None})
        except Exception as exc:  # keep going -- one bad conversation shouldn't sink the run
            elapsed = time.time() - start
            print(f"  FAILED after {elapsed:.1f}s: {exc}")
            results.append({"title": title, "turns": turns, "elapsed_s": round(elapsed, 1),
                             "result": None, "error": str(exc)})
        time.sleep(delay)
    return results


def run_probes(url: str, timeout: float, delay: float) -> list[dict]:
    results = []
    print(f"\n=== running {len(PROBES)} retrieval probes ===")
    for i, (query, expect) in enumerate(PROBES, 1):
        print(f"\n[{i}/{len(PROBES)}] Q: {query}")
        print(f"    expect: {expect}")
        start = time.time()
        try:
            r = post(url, "/read", {"message": query}, timeout=timeout)
            elapsed = time.time() - start
            text = r.get("text") or "(nothing fired)"
            print(f"    -> ({elapsed:.1f}s)\n{text}")
            results.append({"query": query, "expect": expect, "elapsed_s": round(elapsed, 1),
                             "memory_block": r.get("memory_block"), "context_block": r.get("context_block"),
                             "text": r.get("text"), "trace": r.get("trace"), "error": None})
        except Exception as exc:
            elapsed = time.time() - start
            print(f"    FAILED after {elapsed:.1f}s: {exc}")
            results.append({"query": query, "expect": expect, "elapsed_s": round(elapsed, 1),
                             "memory_block": None, "context_block": None, "text": None,
                             "trace": None, "error": str(exc)})
        time.sleep(delay)
    return results


def write_markdown_report(path: Path, writes: list[dict], probes: list[dict], graph: dict) -> None:
    lines = ["# Local dogfood report", ""]
    lines.append(f"Graph after run: **{len(graph.get('nodes', []))} nodes, "
                 f"{len(graph.get('edges', []))} edges**")
    lines.append("")
    lines.append("## Writes")
    for w in writes:
        lines.append(f"\n### {w['title']}  ({w['elapsed_s']}s)")
        for role, text in w["turns"]:
            lines.append(f"- **{role}**: {text}")
        if w["error"]:
            lines.append(f"\n**ERROR**: {w['error']}")
        else:
            r = w["result"]
            lines.append(f"\n`{json.dumps({k: r.get(k) for k in ('nodes_created','nodes_merged','edges_created','edges_reinforced','edges_dropped','edges_superseded','flagged','warnings')})}`")
    lines.append("\n## Probes")
    for p in probes:
        lines.append(f"\n### {p['query']}")
        lines.append(f"*expect*: {p['expect']}")
        if p["error"]:
            lines.append(f"\n**ERROR**: {p['error']}")
        else:
            lines.append(f"\n```\n{p['text'] or '(nothing fired)'}\n```")
    lines.append("\n## Final graph")
    lines.append("\n### Nodes")
    for n in graph.get("nodes", []):
        lines.append(f"- `{n['id']}` ({n['kind']}) aliases={n.get('aliases')} props={n.get('properties')}")
    lines.append("\n### Edges")
    for e in graph.get("edges", []):
        dormant = " DORMANT" if e.get("dormant") else ""
        lines.append(f"- `{e['from']}` -[{e['relation']}]-> `{e['to']}` "
                     f"strength={e['strength']} stability={e['stability']}{dormant}")
        if e.get("snippet"):
            lines.append(f"  > {e['snippet']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "dogfood_local"))
    ap.add_argument("--filter", action="append", default=None,
                     help="substring match on conversation title; repeatable")
    ap.add_argument("--probes-only", action="store_true", help="skip writes, only run probes")
    ap.add_argument("--skip-probes", action="store_true", help="only write, skip probes")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--timeout", type=float, default=300.0, help="per-request timeout (s) -- local LLM writes can be slow")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    health = check_health(args.url)
    print(f"connected to {args.url} -- {health.get('nodes')} nodes, {health.get('edges')} edges currently")

    writes: list[dict] = []
    if not args.probes_only:
        writes = run_writes(args.url, args.timeout, args.delay, args.filter)

    probes: list[dict] = []
    if not args.skip_probes:
        probes = run_probes(args.url, args.timeout, args.delay)

    graph = get(args.url, "/graph", timeout=60)
    print(f"\n=== final graph: {len(graph.get('nodes', []))} nodes, {len(graph.get('edges', []))} edges ===")

    report = {"url": args.url, "timestamp": ts, "writes": writes, "probes": probes, "graph": graph}
    report_json = out_dir / f"report_{ts}.json"
    report_md = out_dir / f"report_{ts}.md"
    graph_json = out_dir / f"graph_{ts}.json"
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    graph_json.write_text(json.dumps(graph, indent=2, ensure_ascii=False))
    write_markdown_report(report_md, writes, probes, graph)

    print(f"\nfull report -> {report_json}")
    print(f"readable report -> {report_md}  (read this one first)")
    print(f"graph snapshot -> {graph_json}")

    failures = [w for w in writes if w["error"]] + [p for p in probes if p["error"]]
    if failures:
        print(f"\n{len(failures)} request(s) failed -- see report for details.")


if __name__ == "__main__":
    main()
