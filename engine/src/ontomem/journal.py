"""Append-only JSONL audit journal (spec v0.2.1 §3.5, §4.4).

One event per line, never rewritten. Kept in a file SEPARATE from the snapshot
so a snapshot corruption can never take the audit trail down with it.
"""

from __future__ import annotations

import json
from pathlib import Path

from .model import utcnow_iso


def append_event(path, event: dict) -> dict:
    """Append one event as a single JSON line. Stamps 'ts' (UTC) if absent.

    Returns the exact record written. Append-only: existing lines are never touched.
    """
    record = dict(event)
    record.setdefault("ts", utcnow_iso())
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read_events(path) -> list[dict]:
    """Read all events back, in order. Missing file -> empty list."""
    p = Path(path)
    if not p.exists():
        return []
    return [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
