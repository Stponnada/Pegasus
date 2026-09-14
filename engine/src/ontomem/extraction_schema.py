"""JSON Schema for the extraction tool-call contract, used when the
generation backend supports OpenAI-compatible function calling with
guided/constrained decoding (e.g. vLLM's tool-call parsers).

This exists to split two previously-conflated jobs the extractor prompt used
to do at once: dictating the exact JSON shape (a structural, non-judgment
concern) and teaching the model what's worth extracting (a judgment concern).
Forcing the model's final answer through this schema means the DECODER
enforces field names, types, and enums -- the prompt (extractor_prompt_reasoning.py)
no longer has to, and can focus entirely on judgment. See inference.py's
OpenAICompatibleGenerator for where this is actually sent as a tool
definition.

Field names and shapes here are the single source of truth for the wire
format; keep in sync with model.py's Node/Edge/Episode and
extractor.normalize_payload, which consumes the resulting arguments dict
exactly the same way it consumes parsed freeform JSON.
"""

from __future__ import annotations

from .model import CARDINALITIES, NODE_KINDS, STABILITIES

EXTRACTION_TOOL_NAME = "submit_graph_extraction"

# Shared field definitions -- used by both the single-shot batch tool below and
# the granular per-item agentic tools in this module, so the wire format for
# "one entity" / "one relationship" never drifts between the two modes.
_ENTITY_PROPERTIES: dict = {
    "text": {
        "type": "string",
        "description": "Canonical entity name, as specific as possible.",
    },
    "type": {"type": "string", "enum": sorted(NODE_KINDS)},
    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "properties": {
        "type": "object",
        "description": "Time-invariant facts only. Empty {} for most entities.",
    },
    "aliases": {"type": "array", "items": {"type": "string"}},
    "candidate_merge_key": {
        "type": ["string", "null"],
        "description": "Null unless you identified a likely existing node match.",
    },
}
_ENTITY_REQUIRED = ["text", "type", "confidence", "properties", "aliases", "candidate_merge_key"]

_RELATIONSHIP_PROPERTIES: dict = {
    "source": {"type": "string", "description": "Canonical entity name, must already exist (added earlier or in the graph)."},
    "relation": {
        "type": "string",
        "description": "UPPER_SNAKE_CASE, verb-first, max 4 words.",
    },
    "target": {"type": "string", "description": "Canonical entity name, must already exist (added earlier or in the graph)."},
    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "stability": {"type": "string", "enum": sorted(STABILITIES)},
    "ttl_days": {"type": ["integer", "null"]},
    "cardinality": {
        "type": "string",
        "enum": sorted(CARDINALITIES),
        "description": (
            "one_to_one: at any given moment, only ONE target can be true for this relation from this "
            "source, even if that target can change over time (current manager, current employer, current "
            "primary residence, current partner). Marking this correctly is what lets the system replace "
            "the old value non-destructively when a new one is stated later, instead of leaving both as "
            "if the person had two managers at once. one_to_many: genuinely multiple simultaneous targets "
            "are normal and expected (friends, hobbies, places visited)."
        ),
    },
    "evidence": {"type": "string", "description": "Short verbatim phrase from the text."},
    "snippet": {
        "type": "string",
        "description": "2-6 sentence verbatim excerpt, meaningful read in isolation.",
    },
    "properties": {
        "type": "object",
        "description": (
            "Quantifiers/qualifiers about this relationship that don't belong "
            "in the relation label — duration, frequency, degree."
        ),
    },
}
_RELATIONSHIP_REQUIRED = [
    "source", "relation", "target", "confidence", "stability", "ttl_days",
    "cardinality", "evidence", "snippet", "properties",
]

EXTRACTION_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": EXTRACTION_TOOL_NAME,
        "description": (
            "Submit the structured knowledge graph extracted from this conversation: "
            "every durable entity, every stated relationship between them, and one "
            "episode summarising the conversation. Call this exactly once, after you "
            "have finished reasoning about what belongs in the graph."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {"type": "object", "properties": _ENTITY_PROPERTIES, "required": _ENTITY_REQUIRED},
                },
                "relationships": {
                    "type": "array",
                    "items": {"type": "object", "properties": _RELATIONSHIP_PROPERTIES, "required": _RELATIONSHIP_REQUIRED},
                },
                "episode": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["summary", "importance", "tags"],
                },
            },
            "required": ["entities", "relationships", "episode"],
        },
    },
}

# --- Agentic (per-item) tools --------------------------------------------------
# Used by engine.py's agentic extraction loop (ONTOMEM_AGENTIC_EXTRACTION=1):
# instead of one call returning the whole graph, the model calls one of these
# per entity/relationship as it finds them, sees the result, and continues --
# see extractor_prompt_agentic.py for the accompanying prompt.

ADD_ENTITY_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "add_entity",
        "description": (
            "Add ONE durable entity to the graph as soon as you identify it. Call this once per entity, "
            "not in a batch. 'type' MUST be exactly one of: " + ", ".join(sorted(NODE_KINDS)) + " -- "
            "an organization/company is ORG, not ORGANIZATION or COMPANY; anything else durable that "
            "isn't a person/org/place/event/topic/preference is OTHER, not a made-up category."
        ),
        "parameters": {"type": "object", "properties": _ENTITY_PROPERTIES, "required": _ENTITY_REQUIRED},
    },
}

ADD_RELATIONSHIP_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "add_relationship",
        "description": (
            "Add ONE relationship to the graph as soon as you identify it. Both source and target must "
            "already exist -- either added earlier in this same session via add_entity, or already present "
            "in the existing graph shown to you. Call this once per relationship, not in a batch."
        ),
        "parameters": {"type": "object", "properties": _RELATIONSHIP_PROPERTIES, "required": _RELATIONSHIP_REQUIRED},
    },
}

FINISH_EXTRACTION_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "finish_extraction",
        "description": (
            "Call this exactly once, after every durable entity and relationship in the conversation has "
            "already been added via add_entity/add_relationship, to close out the episode."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "1-2 sentence specific summary of the conversation."},
                "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "2-5 snake_case domain labels."},
            },
            "required": ["summary", "importance", "tags"],
        },
    },
}

AGENTIC_TOOLS: list[dict] = [ADD_ENTITY_TOOL, ADD_RELATIONSHIP_TOOL, FINISH_EXTRACTION_TOOL]
