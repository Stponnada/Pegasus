"""ontomem — ontology-based long-term memory engine (Phase 1).

Host-agnostic. Plain-data in, plain-data out. See spec v0.2.1.
"""

from .decay import (
    DORMANCY_THRESHOLD,
    LAMBDAS,
    REINFORCE_BOOST,
    decayed_strength,
    is_dormant,
    reinforce,
)
from .journal import append_event, read_events
from .merge import (
    AUTO_MERGE,
    NEEDS_REVIEW,
    NEW,
    Resolution,
    resolve_entities,
    resolve_entity,
)
from .model import (
    CARDINALITIES,
    INITIAL_STRENGTH,
    MAX_STRENGTH,
    NODE_KINDS,
    STABILITIES,
    Edge,
    Episode,
    Node,
    canonicalize,
    edge_key,
    node_key,
)
from .extractor import (
    ExtractionResult,
    build_prompt,
    extract,
    normalize_payload,
    parse_extractor_json,
    to_jsonl,
)
from .engine import Engine
from .retriever import RetrievalConfig
from .store import Store

__all__ = [
    # model
    "Node",
    "Edge",
    "Episode",
    "canonicalize",
    "node_key",
    "edge_key",
    "NODE_KINDS",
    "STABILITIES",
    "CARDINALITIES",
    "INITIAL_STRENGTH",
    "MAX_STRENGTH",
    # store
    "Store",
    # engine facade
    "Engine",
    "RetrievalConfig",
    # extractor
    "extract",
    "ExtractionResult",
    "normalize_payload",
    "parse_extractor_json",
    "build_prompt",
    "to_jsonl",
    # decay
    "decayed_strength",
    "reinforce",
    "is_dormant",
    "LAMBDAS",
    "REINFORCE_BOOST",
    "DORMANCY_THRESHOLD",
    # journal
    "append_event",
    "read_events",
    # merge (2a)
    "resolve_entity",
    "resolve_entities",
    "Resolution",
    "AUTO_MERGE",
    "NEEDS_REVIEW",
    "NEW",
]
