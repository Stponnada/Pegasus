"""Hermetic tests for Gemini key rotation (no network)."""

import pytest

from ontomem import genai_keys
from ontomem.genai_keys import call_rotating, is_rate_limit, load_keys


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # reset the global cursor so each test starts deterministically
    import itertools

    genai_keys._cursor = itertools.count()


def test_load_keys_pool_over_single(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "a, b ,c")
    monkeypatch.setenv("GEMINI_API_KEY", "z")
    assert load_keys() == ["a", "b", "c"]


def test_load_keys_falls_back_to_single(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "solo")
    assert load_keys() == ["solo"]


def test_load_keys_dedupes_and_drops_blanks(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "a,,a, b ,")
    assert load_keys() == ["a", "b"]


def test_load_keys_raises_when_empty(monkeypatch):
    with pytest.raises(RuntimeError):
        load_keys()


def test_is_rate_limit_classification():
    assert is_rate_limit(Exception("429 RESOURCE_EXHAUSTED"))
    assert is_rate_limit(Exception("quota exceeded for this key"))
    assert not is_rate_limit(Exception("invalid argument: bad model"))


def test_call_rotating_returns_first_success():
    used = []
    out = call_rotating(lambda k: used.append(k) or f"ok-{k}", keys=["a", "b"])
    assert out == "ok-a"
    assert used == ["a"]


def test_call_rotating_rotates_on_rate_limit():
    used = []

    def attempt(key):
        used.append(key)
        if key == "a":
            raise Exception("429 RESOURCE_EXHAUSTED")
        return f"ok-{key}"

    out = call_rotating(attempt, keys=["a", "b"])
    assert out == "ok-b"
    assert used == ["a", "b"]


def test_call_rotating_reraises_non_rate_limit_immediately():
    used = []

    def attempt(key):
        used.append(key)
        raise Exception("invalid argument")

    with pytest.raises(Exception, match="invalid argument"):
        call_rotating(attempt, keys=["a", "b"])
    assert used == ["a"]  # did NOT rotate on a real error


def test_call_rotating_backs_off_then_raises_when_all_exhausted(monkeypatch):
    sleeps = []
    monkeypatch.setattr(genai_keys.time, "sleep", lambda s: sleeps.append(s))

    def always_429(key):
        raise Exception("429 quota")

    with pytest.raises(Exception, match="429"):
        call_rotating(always_429, keys=["a", "b"], base_backoff=1.0)
    # 4 sweeps -> 3 backoffs between them
    assert sleeps == [1.0, 2.0, 4.0]


def test_global_cursor_spreads_start_key():
    first = call_rotating(lambda k: k, keys=["a", "b", "c"])
    second = call_rotating(lambda k: k, keys=["a", "b", "c"])
    assert first == "a" and second == "b"  # cursor advanced between calls
