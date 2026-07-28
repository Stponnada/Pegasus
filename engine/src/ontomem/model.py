"""Core graph data model: Node, Edge, Episode.

Pure, serialisable dataclasses. No graph wiring (in-memory pointers) and no I/O
live here — those belong to the store layer. See architecture spec v0.2.1 §3.

Post-v0.2.1 corrections enforced here:
  - Strength and decay are EDGE-only. Nodes carry no strength/stability/ttl.
  - Self-relationships (source == target) are rejected.
  - ttl_days is permitted only for time_bound / ephemeral edges.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

# --- controlled vocabularies (plain strings for clean JSON; validated against these) ---

NODE_KINDS = frozenset(
    {"PERSON", "ORG", "PLACE", "EVENT", "THING", "TOPIC", "PREFERENCE", "OTHER"}
)
STABILITIES = frozenset({"immutable", "stable", "mutable", "time_bound", "ephemeral"})
CARDINALITIES = frozenset({"one_to_one", "one_to_many"})

# stabilities for which a hard ttl is meaningful (everything else never expires)
_TTL_STABILITIES = frozenset({"time_bound", "ephemeral"})

INITIAL_STRENGTH = 100.0
MAX_STRENGTH = 100.0  # reinforcement cap — strength never exceeds its initial value

# UPPER_SNAKE_CASE, verb-first, 1-4 words (spec §3.2). "Verb-first" can't be
# checked mechanically; shape and word-count can.
_RELATION_RE = re.compile(r"^[A-Z]+(?:_[A-Z]+){0,3}$")
_CANON_STRIP_RE = re.compile(r"[^a-z0-9]+")


def utcnow_iso() -> str:
    """Current time as an ISO 8601 UTC string."""
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


def canonicalize(name: str) -> str:
    """Lowercase, collapse non-alphanumeric runs to single '_', strip edges.

    'Ontology-Based LLM Memory' -> 'ontology_based_llm_memory'
    """
    return _CANON_STRIP_RE.sub("_", name.strip().lower()).strip("_")


def node_key(kind: str, name: str) -> str:
    return f"{kind}::{canonicalize(name)}"


def edge_key(source_key: str, relation: str, target_key: str) -> str:
    return f"{source_key}::{relation}::{target_key}"


def _require_range(value: float, lo: float, hi: float, label: str) -> None:
    if not (lo <= value <= hi):
        raise ValueError(f"{label} must be in [{lo}, {hi}], got {value!r}")


@dataclass
class Node:
    """A canonical named concept. Time-invariant facts only; no strength/decay."""

    id: str
    key: str
    name: str
    kind: str
    properties: dict = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    confidence: float = 0.85
    source_episode_ids: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    # runtime-only graph adjacency (wired by the store; never serialised)
    outgoing: list = field(default_factory=list, compare=False, repr=False)
    incoming: list = field(default_factory=list, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.kind not in NODE_KINDS:
            raise ValueError(f"invalid node kind: {self.kind!r}")
        _require_range(self.confidence, 0.0, 1.0, "confidence")

    @classmethod
    def create(
        cls,
        kind: str,
        name: str,
        *,
        confidence: float = 0.85,
        properties: dict | None = None,
        aliases: list[str] | None = None,
        source_episode_ids: list[str] | None = None,
    ) -> "Node":
        ts = utcnow_iso()
        return cls(
            id=new_id(),
            key=node_key(kind, name),
            name=name,
            kind=kind,
            properties=dict(properties or {}),
            aliases=list(aliases or []),
            confidence=confidence,
            source_episode_ids=list(source_episode_ids or []),
            created_at=ts,
            updated_at=ts,
        )

    def to_dict(self) -> dict:
        # Persistent fields only; runtime pointers (outgoing/incoming) are excluded.
        return {
            "id": self.id,
            "key": self.key,
            "name": self.name,
            "kind": self.kind,
            "properties": dict(self.properties),
            "aliases": list(self.aliases),
            "confidence": self.confidence,
            "source_episode_ids": list(self.source_episode_ids),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(**d)


@dataclass
class Edge:
    """A directional relationship. The primary unit of memory; carries strength,
    decay (via stability), and the episodic snippet."""

    id: str
    key: str
    source_key: str
    relation: str
    target_key: str
    strength: float = INITIAL_STRENGTH
    confidence: float = 0.85
    stability: str = "stable"
    ttl_days: int | None = None
    cardinality: str = "one_to_many"
    snippet: str = ""
    evidence: str = ""
    properties: dict = field(default_factory=dict)
    source_episode_ids: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    # runtime-only graph adjacency (wired by the store; never serialised)
    source: Node | None = field(default=None, compare=False, repr=False)
    target: Node | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not _RELATION_RE.match(self.relation):
            raise ValueError(
                f"relation must be UPPER_SNAKE_CASE, 1-4 words: {self.relation!r}"
            )
        if self.stability not in STABILITIES:
            raise ValueError(f"invalid stability: {self.stability!r}")
        if self.cardinality not in CARDINALITIES:
            raise ValueError(f"invalid cardinality: {self.cardinality!r}")
        if self.source_key == self.target_key:
            raise ValueError("self-relationships are not allowed (source == target)")
        if self.ttl_days is not None and self.stability not in _TTL_STABILITIES:
            raise ValueError(
                f"ttl_days is only allowed for {sorted(_TTL_STABILITIES)}, "
                f"not stability={self.stability!r}"
            )
        _require_range(self.confidence, 0.0, 1.0, "confidence")

    @classmethod
    def create(
        cls,
        source_key: str,
        relation: str,
        target_key: str,
        *,
        snippet: str = "",
        evidence: str = "",
        confidence: float = 0.85,
        stability: str = "stable",
        ttl_days: int | None = None,
        cardinality: str = "one_to_many",
        strength: float = INITIAL_STRENGTH,
        properties: dict | None = None,
        source_episode_ids: list[str] | None = None,
    ) -> "Edge":
        ts = utcnow_iso()
        return cls(
            id=new_id(),
            key=edge_key(source_key, relation, target_key),
            source_key=source_key,
            relation=relation,
            target_key=target_key,
            strength=strength,
            confidence=confidence,
            stability=stability,
            ttl_days=ttl_days,
            cardinality=cardinality,
            snippet=snippet,
            evidence=evidence,
            properties=dict(properties or {}),
            source_episode_ids=list(source_episode_ids or []),
            created_at=ts,
            updated_at=ts,
        )

    def to_dict(self) -> dict:
        # Persistent fields only; runtime pointers (source/target) are excluded.
        return {
            "id": self.id,
            "key": self.key,
            "source_key": self.source_key,
            "relation": self.relation,
            "target_key": self.target_key,
            "strength": self.strength,
            "confidence": self.confidence,
            "stability": self.stability,
            "ttl_days": self.ttl_days,
            "cardinality": self.cardinality,
            "snippet": self.snippet,
            "evidence": self.evidence,
            "properties": dict(self.properties),
            "source_episode_ids": list(self.source_episode_ids),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        return cls(**d)


@dataclass
class Episode:
    """A per-conversation provenance record. Not directly retrievable."""

    id: str
    summary: str
    importance: float
    tags: list[str] = field(default_factory=list)
    source_prompt_text: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        _require_range(self.importance, 0.0, 1.0, "importance")

    @classmethod
    def create(
        cls,
        summary: str,
        importance: float,
        *,
        tags: list[str] | None = None,
        source_prompt_text: str = "",
    ) -> "Episode":
        return cls(
            id=new_id(),
            summary=summary,
            importance=importance,
            tags=list(tags or []),
            source_prompt_text=source_prompt_text,
            created_at=utcnow_iso(),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "summary": self.summary,
            "importance": self.importance,
            "tags": list(self.tags),
            "source_prompt_text": self.source_prompt_text,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        return cls(**d)
