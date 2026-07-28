"""Resumable, monitored end-to-end campaign runner for synthetic lives.

The runner exercises the real Engine read/write boundary with whichever Gemini
or OpenAI-compatible provider is configured in the environment. It checkpoints
after every conversation and writes enough evidence to inspect extraction,
graph mutation, retrieval activation, and expectation failures offline.

Run from engine/:
  PYTHONPATH=src .venv/bin/python scripts/e2e_campaign.py \
    --campaign e2e/synthetic_life_v1.json \
    --output e2e_runs/synthetic_life_v1 \
    --audit-content
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo.demo_trace import trace_read
from ontomem.retriever import _gate_noun_hits, extract_nouns, resolve_seed_hits
from ontomem.service import make_engine_from_env


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    os.replace(temp, path)


def _store_state(engine) -> dict:
    return {
        "self_key": engine.store.self_key,
        "nodes": {
            node.key: {
                "name": node.name,
                "kind": node.kind,
                "aliases": list(node.aliases),
                "properties": dict(node.properties),
                "confidence": node.confidence,
                "episodes": list(node.source_episode_ids),
            }
            for node in engine.store.nodes.values()
        },
        "edges": {
            edge.key: {
                "source": edge.source_key,
                "relation": edge.relation,
                "target": edge.target_key,
                "strength": edge.strength,
                "confidence": edge.confidence,
                "stability": edge.stability,
                "ttl_days": edge.ttl_days,
                "properties": dict(edge.properties),
                "snippet": edge.snippet,
                "episodes": list(edge.source_episode_ids),
            }
            for edge in engine.store.edges.values()
        },
        "episodes": len(engine.store.episodes),
    }


def _graph_delta(before: dict, after: dict) -> dict:
    before_nodes = set(before["nodes"])
    after_nodes = set(after["nodes"])
    before_edges = set(before["edges"])
    after_edges = set(after["edges"])
    changed_nodes = [
        key
        for key in sorted(before_nodes & after_nodes)
        if before["nodes"][key] != after["nodes"][key]
    ]
    changed_edges = [
        key
        for key in sorted(before_edges & after_edges)
        if before["edges"][key] != after["edges"][key]
    ]
    return {
        "nodes_created": [after["nodes"][key] | {"key": key} for key in sorted(after_nodes - before_nodes)],
        "nodes_changed": [
            {"key": key, "before": before["nodes"][key], "after": after["nodes"][key]}
            for key in changed_nodes
        ],
        "edges_created": [after["edges"][key] | {"key": key} for key in sorted(after_edges - before_edges)],
        "edges_changed": [
            {"key": key, "before": before["edges"][key], "after": after["edges"][key]}
            for key in changed_edges
        ],
        "nodes_removed": sorted(before_nodes - after_nodes),
        "edges_removed": sorted(before_edges - after_edges),
    }


def _probe_trace(engine, query: str) -> dict:
    """Run the pure demo trace plus per-noun centred-cosine rankings.

    Unlike Engine.read(), this does not append reinforcement events, so synthetic
    evaluation probes cannot change the graph they are measuring.
    """
    trace = trace_read(query, engine.store, engine.index, engine.embedder, engine.config)
    nouns = extract_nouns(query)
    rankings = []
    if nouns and len(engine.index):
        vectors = engine.embedder.embed(nouns)
        for noun, vector in zip(nouns, vectors):
            ranked = engine.index.search(
                vector,
                top_k=max(engine.config.seed_top_k, 5),
                threshold=-1.0,
                center=True,
            )
            accepted = _gate_noun_hits(ranked, engine.config)
            rankings.append(
                {
                    "noun": noun,
                    "ranked": [
                        {
                            "key": key,
                            "score": round(score, 6),
                            "kind": "relation" if key.startswith("REL::") else "node",
                        }
                        for key, score in ranked
                    ],
                    "accepted_index_keys": accepted,
                    "resolved_seed_nodes": resolve_seed_hits(accepted, engine.store),
                }
            )
    trace["seed_rankings"] = rankings
    return trace


def _evaluate_probe(probe: dict, trace: dict) -> list[dict]:
    text = trace["final_context"].lower()
    failures = []

    for expected in probe.get("contains_all", []):
        if expected.lower() not in text:
            failures.append({"check": "contains_all", "expected": expected})

    contains_any = probe.get("contains_any", [])
    if contains_any and not any(value.lower() in text for value in contains_any):
        failures.append({"check": "contains_any", "expected": contains_any})

    for excluded in probe.get("excludes", []):
        if excluded.lower() in text:
            failures.append({"check": "excludes", "unexpected": excluded})

    if probe.get("expect_empty") is True and trace["final_context"]:
        failures.append({"check": "expect_empty", "actual": trace["final_context"]})
    if probe.get("expect_nonempty") is True and not trace["final_context"]:
        failures.append({"check": "expect_nonempty"})
    if len(trace["selected"]) > probe.get("max_selected", 8):
        failures.append(
            {
                "check": "max_selected",
                "expected": probe.get("max_selected", 8),
                "actual": len(trace["selected"]),
            }
        )
    return failures


def _evaluate_graph(expect: dict, write: dict, state: dict) -> list[dict]:
    failures = []
    names = [node["name"].lower() for node in state["nodes"].values()]
    for alternatives in expect.get("has_node_any", []):
        values = alternatives if isinstance(alternatives, list) else [alternatives]
        if not any(any(value.lower() in name for name in names) for value in values):
            failures.append({"check": "has_node_any", "expected": values})
    for term in expect.get("lacks_node_terms", []):
        if any(term.lower() in name for name in names):
            failures.append({"check": "lacks_node_terms", "unexpected": term})
    if write["nodes_created"] > expect.get("max_nodes_created", float("inf")):
        failures.append(
            {
                "check": "max_nodes_created",
                "expected": expect["max_nodes_created"],
                "actual": write["nodes_created"],
            }
        )
    if write["edges_created"] < expect.get("min_edges_created", 0):
        failures.append(
            {
                "check": "min_edges_created",
                "expected": expect["min_edges_created"],
                "actual": write["edges_created"],
            }
        )
    if write["edges_superseded"] < expect.get("min_edges_superseded", 0):
        failures.append(
            {
                "check": "min_edges_superseded",
                "expected": expect["min_edges_superseded"],
                "actual": write["edges_superseded"],
            }
        )
    return failures


def _number_turns(turns: list[dict]) -> list[dict]:
    return [
        {"role": turn["role"], "turn": index, "text": turn["text"]}
        for index, turn in enumerate(turns, start=1)
    ]


def run(campaign_path: Path, output: Path, *, limit: int | None, audit_content: bool) -> int:
    campaign_text = campaign_path.read_text()
    campaign = json.loads(campaign_text)
    output.mkdir(parents=True, exist_ok=True)
    state_dir = output / "state"
    report_path = output / "report.json"
    checkpoint_path = output / "checkpoint.json"

    os.environ.setdefault("ONTOMEM_LLM_AUDIT_PATH", str(output / "llm_calls.jsonl"))
    if audit_content:
        os.environ["ONTOMEM_LLM_AUDIT_CONTENT"] = "1"

    checkpoint = (
        json.loads(checkpoint_path.read_text())
        if checkpoint_path.exists()
        else {"written_sessions": [], "completed_sessions": []}
    )
    checkpoint.setdefault("written_sessions", list(checkpoint["completed_sessions"]))
    report = (
        json.loads(report_path.read_text())
        if report_path.exists()
        else {
            "campaign": campaign["name"],
            "campaign_sha256": hashlib.sha256(campaign_text.encode()).hexdigest(),
            "started_at": _utcnow(),
            "sessions": [],
            "failures": [],
        }
    )
    expected_hash = hashlib.sha256(campaign_text.encode()).hexdigest()
    if report["campaign_sha256"] != expected_hash:
        raise RuntimeError("campaign file changed since this run started; use a new output directory")

    engine = make_engine_from_env(str(state_dir))
    completed = set(checkpoint["completed_sessions"])
    written = set(checkpoint["written_sessions"])
    sessions = campaign["sessions"][:limit] if limit else campaign["sessions"]

    for position, session in enumerate(sessions, start=1):
        if session["id"] in completed:
            print(f"[{position}/{len(sessions)}] {session['id']}: already complete")
            continue
        entry = next((item for item in report["sessions"] if item["id"] == session["id"]), None)
        if session["id"] not in written:
            print(f"[{position}/{len(sessions)}] {session['id']}: reading turns")
            transcript = _number_turns(session["turns"])
            turn_reads = []
            for turn in transcript:
                if turn["role"] != "user":
                    continue
                result = engine.read(turn["text"])
                turn_reads.append(
                    {
                        "turn": turn["turn"],
                        "query": turn["text"],
                        "memory_block": result["memory_block"],
                        "context_block": result["context_block"],
                        "trace": result["trace"],
                    }
                )

            before = _store_state(engine)
            print(f"[{position}/{len(sessions)}] {session['id']}: writing conversation")
            write = engine.write(transcript)
            after = _store_state(engine)
            graph_failures = _evaluate_graph(session.get("expect_graph", {}), write, after)
            entry = {
                "id": session["id"],
                "description": session["description"],
                "transcript": transcript,
                "turn_reads": turn_reads,
                "write": write,
                "graph_delta": _graph_delta(before, after),
                "graph_counts": {
                    "nodes": len(after["nodes"]),
                    "edges": len(after["edges"]),
                    "episodes": after["episodes"],
                },
                "graph_failures": graph_failures,
                "probes": [],
                "written_at": _utcnow(),
            }
            report["sessions"].append(entry)
            written.add(session["id"])
            checkpoint["written_sessions"] = [s["id"] for s in sessions if s["id"] in written]
            _atomic_json(report_path, report)
            _atomic_json(checkpoint_path, checkpoint)
        else:
            if entry is None:
                raise RuntimeError(
                    f"checkpoint says {session['id']} was written but its report entry is missing"
                )
            write = entry["write"]
            graph_failures = entry["graph_failures"]
            print(f"[{position}/{len(sessions)}] {session['id']}: resuming after write")

        probes = []
        for probe in session.get("probes", []):
            print(f"[{position}/{len(sessions)}] {session['id']}: probe {probe['id']}")
            trace = _probe_trace(engine, probe["query"])
            failures = _evaluate_probe(probe, trace)
            probes.append({"id": probe["id"], "query": probe["query"], "trace": trace, "failures": failures})

        session_failures = [
            {"session": session["id"], "scope": "graph", **failure}
            for failure in graph_failures
        ]
        session_failures.extend(
            {"session": session["id"], "scope": probe["id"], **failure}
            for probe in probes
            for failure in probe["failures"]
        )
        report["failures"] = [
            failure for failure in report["failures"]
            if failure.get("session") != session["id"]
        ]
        report["failures"].extend(session_failures)
        entry["probes"] = probes
        entry["completed_at"] = _utcnow()
        completed.add(session["id"])
        checkpoint["completed_sessions"] = [s["id"] for s in sessions if s["id"] in completed]
        _atomic_json(report_path, report)
        _atomic_json(checkpoint_path, checkpoint)
        print(
            f"[{position}/{len(sessions)}] {session['id']}: "
            f"{write['nodes_created']} nodes, {write['edges_created']} edges, "
            f"{len(session_failures)} expectation failures"
        )

    final_state = _store_state(engine)
    report["completed_at"] = _utcnow()
    report["final_graph"] = final_state
    report["summary"] = {
        "sessions_completed": len(report["sessions"]),
        "expectation_failures": len(report["failures"]),
        "nodes": len(final_state["nodes"]),
        "edges": len(final_state["edges"]),
        "episodes": final_state["episodes"],
    }
    _atomic_json(report_path, report)
    print(json.dumps(report["summary"], indent=2))
    return 1 if report["failures"] else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--audit-content", action="store_true")
    args = parser.parse_args()
    return run(args.campaign, args.output, limit=args.limit, audit_content=args.audit_content)


if __name__ == "__main__":
    raise SystemExit(main())
