"""Unit tests for the append-only JSONL journal."""

from ontomem.journal import append_event, read_events


def test_append_then_read(tmp_path):
    path = tmp_path / "journal.jsonl"
    rec = append_event(path, {"event": "node_created", "key": "PERSON::bill"})
    assert rec["event"] == "node_created"
    assert "ts" in rec  # auto-stamped
    events = read_events(path)
    assert len(events) == 1
    assert events[0]["key"] == "PERSON::bill"


def test_append_is_one_line_per_event_and_ordered(tmp_path):
    path = tmp_path / "journal.jsonl"
    append_event(path, {"event": "a"})
    append_event(path, {"event": "b"})
    append_event(path, {"event": "c"})
    # exactly three physical lines
    assert path.read_text().count("\n") == 3
    assert [e["event"] for e in read_events(path)] == ["a", "b", "c"]


def test_append_only_does_not_rewrite_existing(tmp_path):
    path = tmp_path / "journal.jsonl"
    append_event(path, {"event": "first"})
    first_line = path.read_text().splitlines()[0]
    append_event(path, {"event": "second"})
    # original first line is byte-for-byte unchanged
    assert path.read_text().splitlines()[0] == first_line


def test_explicit_ts_is_preserved(tmp_path):
    path = tmp_path / "journal.jsonl"
    append_event(path, {"event": "x", "ts": "2020-01-01T00:00:00+00:00"})
    assert read_events(path)[0]["ts"] == "2020-01-01T00:00:00+00:00"


def test_read_missing_file_returns_empty(tmp_path):
    assert read_events(tmp_path / "nope.jsonl") == []
