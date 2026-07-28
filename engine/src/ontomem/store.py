"""In-memory graph store + atomic JSON snapshot persistence (spec v0.2.1 §3.5).

Runtime representation is an object graph of Node/Edge with real pointer
references (node.outgoing/incoming, edge.source/target) for pointer-style BFS.
The on-disk form is flat and key-based; the pointer graph is rehydrated on load
via a strict two-pass (all nodes first, then wire edges).

Mutation methods are single-responsibility:
  - add_*      : insert a brand-new entity; raise if the key already exists.
  - upsert_*   : ensure the entity exists; if present, return it UNCHANGED.
  - update_*   : explicitly merge/overwrite fields of an existing entity.
  - reinforce_edge : the one Hebbian strength bump.
upsert never silently mutates — all field changes are explicit via update_*.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from .model import MAX_STRENGTH, Edge, Episode, Node, utcnow_iso

SNAPSHOT_VERSION = 1


def _require_confidence(value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"confidence must be in [0.0, 1.0], got {value!r}")


class Store:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self.episodes: dict[str, Episode] = {}
        # The node key representing the graph's own user (there is exactly one
        # per personal graph, structurally -- unlike any other entity, so
        # merging into it is never a false-positive risk the way merging two
        # arbitrary same-named people would be). Set on first write; used to
        # force-resolve generic self-placeholders ("User", "the user") that
        # the extractor emits when it doesn't yet know the person's real name
        # (dogfooding found these fragmenting into separate nodes per
        # conversation, e.g. OTHER::user / PERSON::user / OTHER::other_user).
        self.self_key: str | None = None

    # --- reads -----------------------------------------------------------------

    def get_node(self, key: str) -> Node | None:
        return self.nodes.get(key)

    def get_edge(self, key: str) -> Edge | None:
        return self.edges.get(key)

    def get_episode(self, episode_id: str) -> Episode | None:
        return self.episodes.get(episode_id)

    # --- node mutations --------------------------------------------------------

    def add_node(self, node: Node) -> Node:
        if node.key in self.nodes:
            raise KeyError(f"node already exists: {node.key}")
        self.nodes[node.key] = node
        return node

    def upsert_node(self, node: Node) -> Node:
        """Ensure a node with this key exists. Insert if absent; if present,
        return the EXISTING node unchanged. Field merges belong to update_node."""
        existing = self.nodes.get(node.key)
        if existing is not None:
            return existing
        self.nodes[node.key] = node
        return node

    def update_node(
        self,
        key: str,
        *,
        confidence: float | None = None,
        add_aliases: list[str] | None = None,
        add_properties: dict | None = None,
        add_episode_ids: list[str] | None = None,
    ) -> Node:
        node = self.nodes.get(key)
        if node is None:
            raise KeyError(f"unknown node: {key}")
        if confidence is not None:
            _require_confidence(confidence)
            node.confidence = confidence
        if add_aliases:
            for alias in add_aliases:
                if alias not in node.aliases:
                    node.aliases.append(alias)
        if add_properties:
            node.properties.update(add_properties)
        if add_episode_ids:
            for eid in add_episode_ids:
                if eid not in node.source_episode_ids:
                    node.source_episode_ids.append(eid)
        node.updated_at = utcnow_iso()
        return node

    # --- edge mutations --------------------------------------------------------

    def add_edge(self, edge: Edge) -> Edge:
        if edge.key in self.edges:
            raise KeyError(f"edge already exists: {edge.key}")
        self._require_endpoints(edge)
        self.edges[edge.key] = edge
        self._wire_edge(edge)
        return edge

    def upsert_edge(self, edge: Edge) -> Edge:
        """Ensure an edge with this key exists. Insert (and wire) if absent; if
        present, return the EXISTING edge unchanged. Merges belong to update_edge."""
        existing = self.edges.get(edge.key)
        if existing is not None:
            return existing
        return self.add_edge(edge)

    def update_edge(
        self,
        key: str,
        *,
        snippet: str | None = None,
        evidence: str | None = None,
        confidence: float | None = None,
        strength: float | None = None,
        add_episode_ids: list[str] | None = None,
        add_properties: dict | None = None,
    ) -> Edge:
        edge = self.edges.get(key)
        if edge is None:
            raise KeyError(f"unknown edge: {key}")
        if snippet is not None:
            edge.snippet = snippet
        if evidence is not None:
            edge.evidence = evidence
        if confidence is not None:
            _require_confidence(confidence)
            edge.confidence = confidence
        if strength is not None:
            edge.strength = strength
        if add_properties:
            edge.properties.update(add_properties)
        if add_episode_ids:
            for eid in add_episode_ids:
                if eid not in edge.source_episode_ids:
                    edge.source_episode_ids.append(eid)
        edge.updated_at = utcnow_iso()
        return edge

    def reinforce_edge(self, key: str, *, boost: float) -> Edge:
        """The one Hebbian bump: strength = min(strength + boost, MAX_STRENGTH)."""
        edge = self.edges.get(key)
        if edge is None:
            raise KeyError(f"unknown edge: {key}")
        edge.strength = min(edge.strength + boost, MAX_STRENGTH)
        edge.updated_at = utcnow_iso()
        return edge

    # --- episodes --------------------------------------------------------------

    def add_episode(self, episode: Episode) -> Episode:
        if episode.id in self.episodes:
            raise KeyError(f"episode already exists: {episode.id}")
        self.episodes[episode.id] = episode
        return episode

    # --- adjacency wiring ------------------------------------------------------

    def _require_endpoints(self, edge: Edge) -> None:
        for endpoint in (edge.source_key, edge.target_key):
            if endpoint not in self.nodes:
                raise KeyError(f"edge {edge.key} references missing node: {endpoint}")

    def _wire_edge(self, edge: Edge) -> None:
        source = self.nodes[edge.source_key]
        target = self.nodes[edge.target_key]
        edge.source = source
        edge.target = target
        source.outgoing.append(edge)
        target.incoming.append(edge)

    # --- traversal -------------------------------------------------------------

    def neighbourhood(self, key: str, depth: int = 2) -> dict:
        """BFS out to `depth` hops over the undirected adjacency from `key`.

        Returns {"nodes": [Node], "edges": [Edge]} — nodes reachable within
        `depth` hops (including the seed) and every edge whose endpoints are both
        in that node set. Direction-agnostic: follows both outgoing and incoming.
        Used by merge 2b (candidate neighbourhoods), Stage 0, and rendering.
        """
        start = self.nodes.get(key)
        if start is None:
            return {"nodes": [], "edges": []}
        seen = {start.key}
        frontier = [start]
        for _ in range(max(0, depth)):
            nxt = []
            for n in frontier:
                for edge in (*n.outgoing, *n.incoming):
                    other = edge.target if edge.source is n else edge.source
                    if other is not None and other.key not in seen:
                        seen.add(other.key)
                        nxt.append(other)
            if not nxt:
                break
            frontier = nxt
        nodes = [self.nodes[k] for k in seen]
        edges = [
            e
            for e in self.edges.values()
            if e.source_key in seen and e.target_key in seen
        ]
        return {"nodes": nodes, "edges": edges}

    def render_subgraph(self, keys, depth: int = 2) -> str:
        """Render the union of the 2-hop neighbourhoods of `keys` as a compact,
        human-readable text block (for the extractor/merge LLM context)."""
        collected_nodes: dict[str, Node] = {}
        collected_edges: dict[str, Edge] = {}
        for key in keys:
            hood = self.neighbourhood(key, depth)
            for n in hood["nodes"]:
                collected_nodes[n.key] = n
            for e in hood["edges"]:
                collected_edges[e.key] = e
        if not collected_nodes:
            return ""
        lines = []
        for n in sorted(collected_nodes.values(), key=lambda x: x.key):
            descriptor = f"- {n.key} (name={n.name!r}"
            if n.aliases:
                descriptor += f", aliases={n.aliases}"
            if n.properties:
                descriptor += f", properties={n.properties}"
            descriptor += ")"
            lines.append(descriptor)
        for e in sorted(collected_edges.values(), key=lambda x: x.key):
            lines.append(f"  {e.source_key} -[{e.relation}]-> {e.target_key}")
        return "\n".join(lines)

    # --- serialisation ---------------------------------------------------------

    def to_snapshot(self) -> dict:
        return {
            "version": SNAPSHOT_VERSION,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "episodes": [ep.to_dict() for ep in self.episodes.values()],
            "self_key": self.self_key,
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "Store":
        store = cls()
        # Pass 1: build all nodes, keyed by node.key.
        for node_dict in data.get("nodes", []):
            node = Node.from_dict(node_dict)
            store.nodes[node.key] = node
        # Pass 2: build edges and wire pointers (endpoints now guaranteed present).
        for edge_dict in data.get("edges", []):
            edge = Edge.from_dict(edge_dict)
            store._require_endpoints(edge)
            store.edges[edge.key] = edge
            store._wire_edge(edge)
        for episode_dict in data.get("episodes", []):
            episode = Episode.from_dict(episode_dict)
            store.episodes[episode.id] = episode
        store.self_key = data.get("self_key")
        return store

    def save(self, path) -> None:
        """Atomic snapshot write: temp file in the SAME directory, then os.replace.

        A crash mid-write never corrupts the live snapshot — a reader sees either
        the whole old file or the whole new one.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
        try:
            tmp.write_text(json.dumps(self.to_snapshot(), indent=2), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

    @classmethod
    def load(cls, path) -> "Store":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_snapshot(data)
