"""Shared test fixtures and configuration.

Loads engine/.env (if present) so integration tests can find GEMINI_API_KEY,
and registers the `integration` marker for tests that hit the live LLM.
"""

import os
from pathlib import Path

import pytest

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def _load_dotenv() -> None:
    if not _ENV_PATH.exists():
        return
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: hits the live LLM; requires GEMINI_API_KEY"
    )


@pytest.fixture
def gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY not set; skipping live integration test")
    return key
