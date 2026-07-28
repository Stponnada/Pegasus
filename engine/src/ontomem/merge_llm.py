"""Merge Sub-stage 2b — LLM disambiguation (spec v0.2.1 §4.3). Runs ONLY on the
NEEDS_REVIEW resolutions produced by 2a.

The LLM is asked to compare one extracted entity against the candidate existing
nodes (each shown with its 2-hop neighbourhood) and return, per candidate, a
match confidence. The confidence is then banded deterministically:

  > 0.90        -> merge (auto)
  0.70 .. 0.90  -> merge, but flag in the journal for inspection
  < 0.70        -> do not merge

If several candidates clear the bar, the highest-confidence one wins. If none
do, the entity is new. The banding is pure, deterministic, and hermetically
tested; only the call itself touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass

from .merge import AUTO_MERGE, NEEDS_REVIEW, NEW, Resolution
from .model import Node

DEFAULT_MODEL = "gemini-3.1-flash-lite"

AUTO_MERGE_BAND = 0.90  # strictly above -> auto merge
FLAG_BAND = 0.70  # [FLAG_BAND, AUTO_MERGE_BAND] -> merge + journal flag


@dataclass
class DisambiguationDecision:
    entity_key: str
    decision: str  # AUTO_MERGE | NEW
    target_key: str | None = None
    confidence: float = 0.0
    flagged: bool = False  # merged within the 0.70-0.90 band -> needs inspection
    reason: str = ""


# --- prompt building -----------------------------------------------------------

_DISAMBIGUATION_SYSTEM = """You are an entity-resolution component of a personal memory system.

You are given ONE newly extracted entity, the conversation it came from, and a
list of CANDIDATE existing entities that might be the same real-world thing.
Each candidate is shown with its 2-hop neighbourhood from the memory graph.

Decide, for EACH candidate, the probability that it refers to the SAME
real-world entity as the extracted entity. Consider names, aliases, entity type,
and especially the surrounding relationships (network topology): if the extracted
entity and a candidate share connections (same employer, same spouse, same
location), that strongly raises confidence. For one-to-one relations (HAS_SPOUSE,
HAS_MOTHER, HAS_FATHER, BORN_IN, HAS_PRIMARY_RESIDENCE) a shared such relation is
very strong evidence of identity.

Be conservative. Merging two DISTINCT entities is unrecoverable; a missed merge
only creates a duplicate. When genuinely unsure, assign low confidence.

Return ONLY valid JSON, no prose, no markdown fences:
{
  "candidates": [
    {"key": "EXISTING_NODE_KEY", "confidence": 0.0, "reason": "short phrase"}
  ]
}
Confidence is in [0.0, 1.0]. Include every candidate key you were given.
"""


def build_disambiguation_prompt(
    entity: Node, conversation_context: str, candidate_blocks: dict[str, str]
) -> str:
    parts = [
        _DISAMBIGUATION_SYSTEM,
        "\n=== EXTRACTED ENTITY ===",
        f"key (proposed): {entity.key}",
        f"name: {entity.name}",
        f"type: {entity.kind}",
        f"aliases: {entity.aliases}",
        f"properties: {entity.properties}",
        "\n=== CONVERSATION CONTEXT ===",
        conversation_context or "(none)",
        "\n=== CANDIDATES (each with its 2-hop neighbourhood) ===",
    ]
    for key, block in candidate_blocks.items():
        parts.append(f"\n--- candidate: {key} ---")
        parts.append(block or "(no neighbourhood)")
    return "\n".join(parts)


# --- banding (pure) ------------------------------------------------------------


def band_decision(entity_key: str, scored_candidates: list[dict]) -> DisambiguationDecision:
    """Apply the confidence bands to the LLM's per-candidate scores.

    `scored_candidates`: [{"key": str, "confidence": float, "reason": str}, ...]
    Highest qualifying confidence wins; ties broken by key for determinism.
    """
    ranked = sorted(
        scored_candidates,
        key=lambda c: (float(c.get("confidence", 0.0)), str(c.get("key", ""))),
        reverse=True,
    )
    for cand in ranked:
        conf = float(cand.get("confidence", 0.0))
        if conf > AUTO_MERGE_BAND:
            return DisambiguationDecision(
                entity_key, AUTO_MERGE, cand["key"], conf, flagged=False,
                reason=str(cand.get("reason", "")),
            )
        if conf >= FLAG_BAND:
            return DisambiguationDecision(
                entity_key, AUTO_MERGE, cand["key"], conf, flagged=True,
                reason=str(cand.get("reason", "")),
            )
        break  # ranked desc: if the top candidate is < FLAG_BAND, none qualify
    return DisambiguationDecision(entity_key, NEW, None, ranked[0]["confidence"] if ranked else 0.0)


# --- parsing -------------------------------------------------------------------


def parse_disambiguation_json(raw: str, allowed_keys) -> list[dict]:
    """Parse the LLM output, keeping only candidates whose key was offered.
    Reuses the extractor's tolerant JSON salvage."""
    from .extractor import parse_extractor_json

    data = parse_extractor_json(raw)
    allowed = set(allowed_keys)
    out = []
    for cand in data.get("candidates", []):
        key = str(cand.get("key", ""))
        if key in allowed:
            out.append(
                {
                    "key": key,
                    "confidence": max(0.0, min(1.0, float(cand.get("confidence", 0.0)))),
                    "reason": str(cand.get("reason", "")),
                }
            )
    return out


# --- the networked call --------------------------------------------------------


def call_gemini(prompt: str, *, model: str = DEFAULT_MODEL, api_key: str | None = None) -> str:
    from google import genai

    from .genai_keys import call_rotating

    def attempt(key: str) -> str:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(model=model, contents=prompt)
        if response.text is None:
            raise RuntimeError("model returned no text")
        return response.text

    return attempt(api_key) if api_key else call_rotating(attempt)


def disambiguate(
    entity: Node,
    resolution: Resolution,
    store,
    conversation_context: str = "",
    *,
    depth: int = 2,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    generate_fn=None,
) -> DisambiguationDecision:
    """Run 2b for a single NEEDS_REVIEW resolution. Non-review resolutions are a
    programming error here — 2b must only be invoked on candidates."""
    if resolution.decision != NEEDS_REVIEW:
        raise ValueError("disambiguate() expects a NEEDS_REVIEW resolution")
    candidate_blocks = {
        key: store.render_subgraph([key], depth=depth) for key in resolution.candidates
    }
    prompt = build_disambiguation_prompt(entity, conversation_context, candidate_blocks)
    raw = (
        generate_fn(prompt, model=model, purpose="disambiguate")
        if generate_fn
        else call_gemini(prompt, model=model, api_key=api_key)
    )
    scored = parse_disambiguation_json(raw, resolution.candidates)
    return band_decision(entity.key, scored)
