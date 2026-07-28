"""Write-stage relationship supersession (spec v0.2.1 §9.2c).

When a newly extracted edge shares its (source, target) with an existing edge
under a DIFFERENT relation, the new one may *supersede* the old — a user who was
`CONSIDERING_TRANSFER_TO` a team later `TRANSFERS_TO` it — or the two may
legitimately `COEXIST` (`HAS_PARTNER` and `TRAVELS_WITH`), or they may
`CONTRADICT`. The judgement is not deterministic (it needs to know that
'transfers' deprecates 'considering transfer' but 'travels with' does not
deprecate 'has partner'), so it is delegated to an injected decision function —
an LLM by default, stubbed in tests.

The decision is pure data. The deterministic write stage (consolidate.apply_write)
acts on it by DEMOTING the superseded edge — reducing its strength to near-dormant
and tagging it `superseded_by` — and NEVER deleting it (non-destructive, §6/§9.2c).
The safe default when the model is unsure is COEXIST: keep both, weaken nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MODEL = "gemini-3.1-flash-lite"

SUPERSEDES = "supersedes"
COEXIST = "coexist"
CONTRADICTS = "contradicts"
_VALID = (SUPERSEDES, COEXIST, CONTRADICTS)


@dataclass
class SupersessionDecision:
    old_edge_key: str
    new_relation: str
    decision: str  # SUPERSEDES | COEXIST | CONTRADICTS
    reason: str = ""


_SUPERSESSION_SYSTEM = """You are a component of a personal memory system that keeps a knowledge graph current.

A new relationship has just been observed between two entities that ALREADY have
a different relationship recorded between them, in the SAME direction. Decide how
the new relationship relates to the existing one:

- "supersedes": the new relationship is a later state of the SAME underlying fact
  and makes the old one stale. Example: old CONSIDERING_TRANSFER_TO, new
  TRANSFERS_TO (the person was considering it, now they have done it). Old
  HAS_CRUSH_ON, new HAS_PARTNER. The old should be demoted.
- "coexist": the two relationships are both independently true and neither
  invalidates the other. Example: HAS_PARTNER and TRAVELS_WITH; WORKS_AT and
  FOUNDED. Keep both.
- "contradicts": the new relationship directly conflicts with the old in a way
  that is not a natural progression. Example: old WORKS_AT, new QUIT followed by
  a competing WORKS_AT to a different org would contradict. Flag it.

Be conservative: only answer "supersedes" or "contradicts" when you are clearly
confident the old relationship is no longer current. When in doubt, answer
"coexist" — keeping a slightly stale edge is recoverable; wrongly demoting a true
one loses information.

Return ONLY valid JSON, no prose, no markdown fences:
{"decision": "supersedes|coexist|contradicts", "reason": "short phrase"}
"""


def build_supersession_prompt(
    source_name: str, target_name: str, old_relation: str, new_relation: str,
    old_snippet: str = "", conversation_context: str = "",
) -> str:
    return "\n".join([
        _SUPERSESSION_SYSTEM,
        "\n=== ENTITIES ===",
        f"source: {source_name}",
        f"target: {target_name}",
        "\n=== EXISTING RELATIONSHIP (may be stale) ===",
        f"{source_name} -[{old_relation}]-> {target_name}",
        f"recorded context: {old_snippet or '(none)'}",
        "\n=== NEW RELATIONSHIP (just observed) ===",
        f"{source_name} -[{new_relation}]-> {target_name}",
        "\n=== CONVERSATION CONTEXT ===",
        conversation_context or "(none)",
    ])


def parse_supersession_json(raw: str) -> dict:
    """Parse the decision, defaulting to the safe COEXIST on anything unexpected.
    Reuses the extractor's tolerant JSON salvage."""
    from .extractor import parse_extractor_json

    try:
        data = parse_extractor_json(raw)
    except ValueError:
        return {"decision": COEXIST, "reason": "unparseable decision; kept both"}
    decision = str(data.get("decision", "")).strip().lower()
    if decision not in _VALID:
        decision = COEXIST
    return {"decision": decision, "reason": str(data.get("reason", ""))}


def decide_supersession(
    old_edge, new_edge, source_name: str, target_name: str, conversation_context: str = "",
    *, model: str = DEFAULT_MODEL, api_key: str | None = None, generate_fn=None,
) -> SupersessionDecision:
    """The networked decision. Compares an existing edge with a newly extracted
    one between the same endpoints."""
    from .extractor import call_gemini

    prompt = build_supersession_prompt(
        source_name, target_name, old_edge.relation, new_edge.relation,
        old_edge.snippet, conversation_context,
    )
    raw = (
        generate_fn(prompt, model=model, purpose="supersede")
        if generate_fn
        else call_gemini(prompt, model=model, api_key=api_key)
    )
    parsed = parse_supersession_json(raw)
    return SupersessionDecision(old_edge.key, new_edge.relation, parsed["decision"], parsed["reason"])
