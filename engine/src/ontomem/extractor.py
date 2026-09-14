"""Extractor (spec v0.2.1 Layer 1): conversation -> structured graph payload.

Two halves, deliberately separated so the hard part stays testable:
  - Deterministic scaffolding (parse + salvage + normalise) — pure, hermetic.
  - The LLM call (call_gemini / extract) — the only networked, non-deterministic part.

Normalisation is defensive: it coerces recoverable LLM mistakes (casing, stray
ttl, unknown stability) and DROPS unrecoverable items (self-edges, dangling
endpoints, malformed relations) with a warning, rather than failing the whole
extraction. An empty entities/relationships payload is valid output.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .extractor_prompt import EXTRACTOR_SYSTEM_PROMPT
from .model import (
    CARDINALITIES,
    STABILITIES,
    Edge,
    Episode,
    Node,
    canonicalize,
    utcnow_iso,
)

DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_CONFIDENCE = 0.85
_TTL_STABILITIES = ("time_bound", "ephemeral")

_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_REL_CLEAN_RE = re.compile(r"[^A-Z0-9]+")


@dataclass
class ExtractionResult:
    episode: Episode
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    candidate_merge_keys: dict[str, str] = field(default_factory=dict)  # node.key -> existing key
    warnings: list[str] = field(default_factory=list)


# --- parsing / salvage ---------------------------------------------------------


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _salvage_candidates(text: str):
    yield text
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        sliced = text[start : end + 1]
        yield sliced
        yield _TRAILING_COMMA_RE.sub(r"\1", sliced)


def parse_extractor_json(raw: str) -> dict:
    """Parse the extractor output into a dict, salvaging common malformations:
    code fences, leading/trailing prose, trailing commas."""
    text = _strip_fences(raw)
    for candidate in _salvage_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("could not parse extractor JSON output")


# --- normalisation -------------------------------------------------------------


def _clamp01(value, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, v))


def _normalize_relation(raw) -> str:
    return _REL_CLEAN_RE.sub("_", str(raw).strip().upper()).strip("_")


def normalize_payload(
    payload: dict,
    *,
    source_prompt_text: str = "",
    default_confidence: float = DEFAULT_CONFIDENCE,
    existing_keys: dict | None = None,
    synthesize_missing_endpoints: bool = True,
) -> ExtractionResult:
    warnings: list[str] = []
    episode = _normalize_episode(payload.get("episode"), source_prompt_text, warnings)
    nodes, text_to_key, candidate_merge_keys = _normalize_entities(
        payload.get("entities", []), default_confidence, warnings
    )
    edges = _normalize_relationships(
        payload.get("relationships", []), nodes, text_to_key, default_confidence, warnings,
        existing_keys or {}, synthesize_missing_endpoints,
    )
    return ExtractionResult(
        episode=episode,
        nodes=nodes,
        edges=edges,
        candidate_merge_keys=candidate_merge_keys,
        warnings=warnings,
    )


def _normalize_episode(ep_raw, source_prompt_text: str, warnings: list[str]) -> Episode:
    ep_raw = ep_raw or {}
    summary = str(ep_raw.get("summary", "")).strip()
    if not summary:
        warnings.append("episode missing or empty; synthesised fallback")
        return Episode.create("(no durable summary)", 0.1, source_prompt_text=source_prompt_text)
    tags = [str(t).strip() for t in ep_raw.get("tags", []) if str(t).strip()][:5]
    return Episode.create(
        summary=summary,
        importance=_clamp01(ep_raw.get("importance", 0.3), 0.3),
        tags=tags,
        source_prompt_text=source_prompt_text,
    )


def _normalize_entities(entities, default_confidence, warnings):
    nodes: list[Node] = []
    text_to_key: dict[str, str] = {}
    candidate_merge_keys: dict[str, str] = {}
    for ent in entities:
        name = str(ent.get("text", "")).strip()
        if not name:
            warnings.append("entity with empty text dropped")
            continue
        kind = str(ent.get("type", "")).strip().upper()
        try:
            node = Node.create(
                kind=kind,
                name=name,
                confidence=_clamp01(ent.get("confidence", default_confidence), default_confidence),
                properties=ent.get("properties") or {},
                aliases=[str(a).strip() for a in ent.get("aliases", []) if str(a).strip()],
            )
        except ValueError as exc:
            warnings.append(f"entity {name!r} dropped: {exc}")
            continue
        nodes.append(node)
        text_to_key[name] = node.key
        text_to_key[canonicalize(name)] = node.key
        cmk = ent.get("candidate_merge_key")
        if cmk:
            candidate_merge_keys[node.key] = str(cmk)
    return nodes, text_to_key, candidate_merge_keys


def _resolve_key(name: str, text_to_key: dict[str, str], existing_keys: dict[str, str]) -> str | None:
    # current extraction first, then fall back to the existing graph (so a
    # relationship to a known node from an earlier conversation is not dropped)
    canon = canonicalize(name)
    return text_to_key.get(name) or text_to_key.get(canon) or existing_keys.get(canon)


def _synthesize_endpoint(name, nodes, text_to_key, default_confidence, warnings) -> str | None:
    """Create a provisional OTHER node for a relationship endpoint the extractor
    referenced but did not emit as an entity (the inverse of the NO-ORPHAN rule).
    Recovers the edge — and the concept — instead of dropping it (spec §9.5: a
    stray low-confidence node is recoverable via decay/merge/flagging; a dropped
    edge is lost). Marked provisional and low-confidence so a later proper
    extraction is preferred. Returns the new node key, or None if unusable."""
    try:
        node = Node.create(
            kind="OTHER", name=name,
            confidence=min(default_confidence, 0.5),
            properties={"provisional_from_relationship": True},
        )
    except ValueError:
        return None
    nodes.append(node)
    text_to_key[name] = node.key
    text_to_key[canonicalize(name)] = node.key
    warnings.append(
        f"endpoint {name!r} referenced in a relationship but not extracted; "
        f"synthesised provisional node {node.key}"
    )
    return node.key


def _normalize_relationships(
    relationships, nodes, text_to_key, default_confidence, warnings, existing_keys,
    synthesize_missing_endpoints=True,
):
    edges: list[Edge] = []
    for rel in relationships:
        src_name = str(rel.get("source", "")).strip()
        tgt_name = str(rel.get("target", "")).strip()
        relation = _normalize_relation(rel.get("relation", ""))
        src_key = _resolve_key(src_name, text_to_key, existing_keys)
        tgt_key = _resolve_key(tgt_name, text_to_key, existing_keys)
        label = f"{src_name!r}-{relation}->{tgt_name!r}"

        # Anchored auto-create: if exactly one endpoint is unknown, synthesise it
        # (the relationship anchors it to something real). If BOTH are unknown the
        # relationship is too speculative to trust — drop it.
        if synthesize_missing_endpoints and bool(src_key) != bool(tgt_key):
            if not src_key:
                src_key = _synthesize_endpoint(src_name, nodes, text_to_key, default_confidence, warnings)
            else:
                tgt_key = _synthesize_endpoint(tgt_name, nodes, text_to_key, default_confidence, warnings)

        if not src_key or not tgt_key:
            warnings.append(f"relationship {label} dropped: endpoint not in entities or graph")
            continue

        stability = str(rel.get("stability") or "stable")
        if stability not in STABILITIES:
            warnings.append(f"relationship {label}: unknown stability {stability!r} -> 'stable'")
            stability = "stable"
        cardinality = str(rel.get("cardinality") or "one_to_many")
        if cardinality not in CARDINALITIES:
            warnings.append(f"relationship {label}: unknown cardinality {cardinality!r} -> 'one_to_many'")
            cardinality = "one_to_many"
        ttl_days = rel.get("ttl_days") if stability in _TTL_STABILITIES else None

        try:
            edge = Edge.create(
                source_key=src_key,
                relation=relation,
                target_key=tgt_key,
                snippet=str(rel.get("snippet", "")),
                evidence=str(rel.get("evidence", "")),
                confidence=_clamp01(rel.get("confidence", default_confidence), default_confidence),
                stability=stability,
                ttl_days=ttl_days,
                cardinality=cardinality,
                properties=rel.get("properties") or {},
            )
        except ValueError as exc:
            warnings.append(f"relationship {label} dropped: {exc}")
            continue
        edges.append(edge)
    return edges


# --- prompt building & the LLM call -------------------------------------------


def to_jsonl(turns: list[dict]) -> str:
    """Serialise a list of turn dicts to one JSON object per line."""
    return "\n".join(json.dumps(t, ensure_ascii=False) for t in turns)


def build_prompt(
    conversation_jsonl: str,
    existing_graph_context: str,
    current_utc_time: str,
    *,
    template: str = EXTRACTOR_SYSTEM_PROMPT,
) -> str:
    return (
        template.replace("{current_utc_time}", current_utc_time)
        .replace("{existing_graph_context}", existing_graph_context or "(empty - no existing graph)")
        .replace("{conversation_jsonl}", conversation_jsonl)
    )


def call_gemini(prompt: str, *, model: str = DEFAULT_MODEL, api_key: str | None = None) -> str:
    """The one networked, non-deterministic call. Lazily imports the SDK.

    With no explicit `api_key`, rotates across the pooled keys (free-tier rate
    limits) via `genai_keys.call_rotating`."""
    from google import genai

    from .genai_keys import call_rotating

    def attempt(key: str) -> str:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(model=model, contents=prompt)
        if response.text is None:
            raise RuntimeError("model returned no text (possibly blocked or empty response)")
        return response.text

    return attempt(api_key) if api_key else call_rotating(attempt)


def extract(
    conversation,
    existing_graph_context: str = "",
    *,
    existing_keys: dict | None = None,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    now_iso: str | None = None,
    generate_fn=None,
    prompt_template: str | None = None,
) -> ExtractionResult:
    """End-to-end: build prompt -> call LLM -> parse -> normalise.

    `existing_keys` (canonical name/alias -> node key) lets relationships
    reference nodes already in the graph that weren't re-extracted this turn.

    `prompt_template` swaps in an alternate system prompt (e.g. the
    motivation-driven EXTRACTOR_REASONING_PROMPT for a tool-calling backend)
    without touching the default Gemini path. None keeps EXTRACTOR_SYSTEM_PROMPT."""
    convo_jsonl = to_jsonl(conversation) if isinstance(conversation, list) else str(conversation)
    prompt = build_prompt(
        convo_jsonl, existing_graph_context, now_iso or utcnow_iso(),
        template=prompt_template or EXTRACTOR_SYSTEM_PROMPT,
    )
    raw = (
        generate_fn(prompt, model=model, purpose="extract")
        if generate_fn
        else call_gemini(prompt, model=model, api_key=api_key)
    )
    return normalize_payload(parse_extractor_json(raw), source_prompt_text=convo_jsonl, existing_keys=existing_keys)
