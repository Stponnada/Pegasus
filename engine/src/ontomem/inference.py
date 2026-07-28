"""Text-generation clients for the non-deterministic pipeline stages.

Gemini remains supported at its existing call sites. This module adds a small
OpenAI-compatible client so extraction, entity disambiguation, and relationship
supersession can share a self-hosted vLLM server without taking a dependency on
the OpenAI SDK.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .journal import append_event


def _endpoint(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def post_json(
    url: str,
    payload: dict,
    *,
    api_key: str | None = None,
    timeout: float = 300.0,
) -> dict:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"inference server returned HTTP {exc.code}: {detail[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach inference server at {url}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"inference server returned invalid JSON at {url}") from exc


class OpenAICompatibleGenerator:
    """Call an OpenAI Chat Completions-compatible text-generation server."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 300.0,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        audit_path: str | Path | None = None,
        audit_content: bool = False,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.audit_path = Path(audit_path) if audit_path else None
        self.audit_content = audit_content

    def generate(self, prompt: str, *, model: str, purpose: str = "generation") -> str:
        started = time.monotonic()
        event = {
            "event": "llm_call",
            "purpose": purpose,
            "model": model,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_chars": len(prompt),
        }
        if self.audit_content:
            event["prompt"] = prompt
        try:
            payload = post_json(
                _endpoint(self.base_url, "chat/completions"),
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                api_key=self.api_key,
                timeout=self.timeout,
            )
            content = payload["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("inference server returned no message content")
            event["response_chars"] = len(content)
            if self.audit_content:
                event["response"] = content
            return content
        except Exception as exc:
            event["error"] = str(exc)
            raise
        finally:
            event["latency_seconds"] = round(time.monotonic() - started, 4)
            if self.audit_path:
                append_event(self.audit_path, event)
