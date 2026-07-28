"""Gemini API key rotation (operational, not part of the spec).

The free-tier keys have small per-key rate limits, so a single key stalls under
the dogfood load. This pools several keys (env `GEMINI_API_KEYS`, comma-separated,
falling back to the single `GEMINI_API_KEY`) and rotates through them:

  - a global round-robin cursor spreads load across keys between calls;
  - on a rate-limit error the call is retried immediately on the next key;
  - only after the whole pool has been tried once in a row do we back off.

Non-rate-limit errors propagate immediately — rotation only masks quota, never
real failures. `call_rotating` takes a one-argument function that performs a
single attempt with the supplied key.
"""

from __future__ import annotations

import itertools
import os
import time

_RATE_LIMIT_MARKERS = ("429", "resource_exhausted", "rate limit", "quota", "exhausted")


def load_keys() -> list[str]:
    """Pool from `GEMINI_API_KEYS` (comma-separated), else the single
    `GEMINI_API_KEY`. Order preserved, duplicates and blanks removed."""
    pooled = os.environ.get("GEMINI_API_KEYS", "")
    raw = pooled.split(",") if pooled.strip() else [os.environ.get("GEMINI_API_KEY", "")]
    seen: dict[str, None] = {}
    for k in raw:
        key = k.strip()
        if key:
            seen.setdefault(key, None)
    if not seen:
        raise RuntimeError("no Gemini API key: set GEMINI_API_KEYS or GEMINI_API_KEY")
    return list(seen)


# global cursor so consecutive calls (extract, embed, disambiguate) spread load
_cursor = itertools.count()


def is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def call_rotating(make_call, *, keys: list[str] | None = None, base_backoff: float = 4.0):
    """Run `make_call(api_key)`, rotating keys on rate-limit errors.

    Tries each key once per pool-sweep; after a full sweep of rate-limit errors,
    sleeps with exponential backoff before the next sweep. Non-rate-limit errors
    raise immediately."""
    keys = keys or load_keys()
    start = next(_cursor)
    sweeps = 4
    last_exc: Exception | None = None
    for sweep in range(sweeps):
        for i in range(len(keys)):
            key = keys[(start + sweep * len(keys) + i) % len(keys)]
            try:
                return make_call(key)
            except Exception as exc:  # noqa: BLE001 - classify then re-raise
                last_exc = exc
                if not is_rate_limit(exc):
                    raise
        if sweep < sweeps - 1:
            time.sleep(base_backoff * (2**sweep))
    raise last_exc  # type: ignore[misc]
