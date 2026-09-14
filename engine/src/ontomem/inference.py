"""Text-generation clients for the non-deterministic pipeline stages.

Gemini remains supported at its existing call sites. This module adds a small
OpenAI-compatible client so extraction, entity disambiguation, and relationship
supersession can share a self-hosted vLLM server without taking a dependency on
the OpenAI SDK. It also adds HostCallbackGenerator, for when the model call
should be delegated entirely to a local host process (e.g. an opencode plugin
proxying through the user's own already-authenticated model access) instead of
the engine holding any API key itself.
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


class HostCallbackGenerator:
    """Call back into a locally running host process (e.g. an opencode plugin)
    instead of any model API directly. The host owns model selection and
    auth entirely -- this client just hands it a prompt and gets text back,
    over a plain localhost HTTP contract, matching the plain-data boundary
    the rest of this engine already keeps with its host (see engine.py's
    module docstring). Used when ONTOMEM_HOST_CALLBACK_URL is set
    (service.py's make_engine_from_env) so extraction/merge/supersede can run
    without the engine ever holding a model API key: the plugin fulfills each
    call using whatever provider opencode itself is already authenticated
    for (e.g. the user's own OpenCode Zen/Go subscription).

    Same `generate(prompt, *, model, purpose)` shape as
    OpenAICompatibleGenerator so it drops into Engine(generate_fn=...)
    unchanged; `model` is accepted but ignored -- the host process decides
    which model to use, not the engine.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 120.0,
        audit_path: str | Path | None = None,
        audit_content: bool = False,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.audit_path = Path(audit_path) if audit_path else None
        self.audit_content = audit_content

    def generate(self, prompt: str, *, model: str, purpose: str = "generation") -> str:
        started = time.monotonic()
        event = {
            "event": "llm_call",
            "purpose": purpose,
            "model": "host-callback",
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_chars": len(prompt),
        }
        if self.audit_content:
            event["prompt"] = prompt
        try:
            payload = post_json(
                _endpoint(self.base_url, "generate"),
                {"prompt": prompt, "purpose": purpose},
                timeout=self.timeout,
            )
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("host callback returned no text")
            event["response_chars"] = len(text)
            if self.audit_content:
                event["response"] = text
            return text
        except Exception as exc:
            event["error"] = str(exc)
            raise
        finally:
            event["latency_seconds"] = round(time.monotonic() - started, 4)
            if self.audit_path:
                append_event(self.audit_path, event)


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
        extraction_tool_schema: dict | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.audit_path = Path(audit_path) if audit_path else None
        self.audit_content = audit_content
        # Only consulted for purpose="extract" calls (see generate() below).
        # Other purposes (disambiguate, supersede) are untouched by this —
        # forcing the model's final answer through a JSON Schema via tool
        # calling is specifically an extraction-reliability mechanism, not a
        # blanket change to how this generator talks to the backend.
        self.extraction_tool_schema = extraction_tool_schema

    def generate(self, prompt: str, *, model: str, purpose: str = "generation") -> str:
        if purpose == "extract" and self.extraction_tool_schema is not None:
            return self._generate_via_tool_call(prompt, model=model, purpose=purpose)
        return self._generate_via_content(prompt, model=model, purpose=purpose)

    def _generate_via_tool_call(self, prompt: str, *, model: str, purpose: str) -> str:
        """Force the final answer through the extraction tool's JSON Schema
        instead of asking the model to freehand JSON. Returns the tool call's
        arguments string (already JSON) — parse_extractor_json accepts it
        unchanged, since valid JSON is trivially its own salvage candidate."""
        schema = self.extraction_tool_schema
        assert schema is not None  # generate() only routes here when it's set
        started = time.monotonic()
        tool_name = schema["function"]["name"]
        event = {
            "event": "llm_call",
            "purpose": purpose,
            "model": model,
            "mode": "tool_call",
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
                    "tools": [schema],
                    # "auto" rather than forcing {"type": "function", ...}: forced
                    # tool_choice on this vLLM build's gemma4 parser hit a real
                    # upstream bug on a large extraction call -- the model's
                    # response was apparently pure tool-call with no leading
                    # text, and vLLM's chat_completion_full_generator asserts
                    # `content is not None` when parsing tool calls out of the
                    # response, which fails in exactly that case (confirmed via
                    # the server's own traceback). Only one tool is offered, so
                    # "auto" still reliably calls it in practice while avoiding
                    # this specific empty-content parse path.
                    "tool_choice": "auto",
                },
                api_key=self.api_key,
                timeout=self.timeout,
            )
            message = payload["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                raise RuntimeError(
                    f"inference server returned no tool_calls for tool={tool_name!r} "
                    f"(finish_reason={payload['choices'][0].get('finish_reason')!r} -- "
                    f"'length' means the reasoning trace exhausted max_tokens before the "
                    f"tool call was emitted; raise max_tokens)"
                )
            arguments = tool_calls[0]["function"]["arguments"]
            if not isinstance(arguments, str) or not arguments.strip():
                raise RuntimeError("inference server returned an empty tool_call arguments string")
            event["response_chars"] = len(arguments)
            if self.audit_content:
                event["response"] = arguments
                if message.get("reasoning_content"):
                    event["reasoning"] = message["reasoning_content"]
            return arguments
        except Exception as exc:
            event["error"] = str(exc)
            raise
        finally:
            event["latency_seconds"] = round(time.monotonic() - started, 4)
            if self.audit_path:
                append_event(self.audit_path, event)

    def chat_with_tools(self, messages: list[dict], *, model: str, tools: list[dict], purpose: str = "extract") -> dict:
        """One turn of a multi-turn tool-calling conversation: send the given
        messages + tools, return the raw assistant message dict (role,
        content, and tool_calls if any). The caller (engine.py's agentic
        extraction loop) owns the conversation state -- appending this
        message plus tool-result messages before calling again -- since this
        client is a dumb transport, not an orchestrator."""
        started = time.monotonic()
        event = {
            "event": "llm_call",
            "purpose": purpose,
            "model": model,
            "mode": "chat_with_tools",
            "message_count": len(messages),
        }
        if self.audit_content:
            event["messages"] = messages
        try:
            payload = post_json(
                _endpoint(self.base_url, "chat/completions"),
                {
                    "model": model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "tools": tools,
                    "tool_choice": "auto",
                },
                api_key=self.api_key,
                timeout=self.timeout,
            )
            message = payload["choices"][0]["message"]
            event["finish_reason"] = payload["choices"][0].get("finish_reason")
            event["tool_call_count"] = len(message.get("tool_calls") or [])
            if self.audit_content:
                event["response_message"] = message
            return message
        except Exception as exc:
            event["error"] = str(exc)
            raise
        finally:
            event["latency_seconds"] = round(time.monotonic() - started, 4)
            if self.audit_path:
                append_event(self.audit_path, event)

    def _generate_via_content(self, prompt: str, *, model: str, purpose: str) -> str:
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
