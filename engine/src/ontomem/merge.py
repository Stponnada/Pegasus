"""Merge / entity resolution — Sub-stage 2a (spec v0.2.1 §4.3). Deterministic, no LLM.

Generates candidate matches for each freshly extracted entity against the
existing graph. Rule 0 (self-reference) then five more are applied in order:

  0. self-reference (generic placeholder,   -> AUTO_MERGE (unconditional)
                     e.g. "User", "me")
  1. exact         (name + type)            -> AUTO_MERGE
  2. alias         (extracted name is an    -> AUTO_MERGE (unique) / REVIEW (ambiguous)
                    existing node's alias)
  3. abbreviation  (shared name-abbrev group) -> AUTO_MERGE (unique) / REVIEW (ambiguous)
  4. fuzzy         (Levenshtein sim > 0.80)  -> REVIEW
  5. token overlap (subset / Jaccard >= 0.5) -> REVIEW

The merge-asymmetry invariant governs the control flow: a wrong merge is
*unrecoverable*, a missed merge only makes a duplicate. So only the
high-certainty rules (1-3) auto-merge, and they do so ONLY when the match is
unambiguous; rules 4-5 and every ambiguous case defer to LLM review (Sub-stage
2b). No candidates after all five rules => the entity is definitively new.

Candidate generation is restricted to same-kind nodes — a PERSON named "Sam"
must never become a candidate for an ORG.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import Node, canonicalize

# decisions
AUTO_MERGE = "auto_merge"
NEEDS_REVIEW = "needs_review"
NEW = "new"

# which rule fired
RULE_SELF = "self_reference"
RULE_EXACT = "exact"
RULE_ALIAS = "alias"
RULE_ABBREVIATION = "abbreviation"
RULE_FUZZY = "fuzzy"
RULE_TOKEN = "token_overlap"
RULE_NONE = "none"

# Generic self-references the extractor falls back to when it doesn't yet know
# the user's real name (canonicalize()'d: lowercased, non-alphanumeric -> "_").
# Unlike every other merge rule, this one is safe to force unconditionally:
# a false-positive merge between two DIFFERENT real people is unrecoverable
# (the spec's core asymmetry concern), but there is structurally only ever one
# "the user" per personal graph, so there is no false-positive to protect
# against. Found via dogfooding: without this, each conversation that doesn't
# restate the user's name spawns its own placeholder (OTHER::user,
# PERSON::user, OTHER::other_user, ...), fragmenting the single most important
# node in the graph.
SELF_PLACEHOLDER_NAMES = frozenset({"user", "the_user", "me", "myself"})

FUZZY_THRESHOLD = 0.80  # strictly greater than, per spec
JACCARD_THRESHOLD = 0.5

# Static name-abbreviation groups (canonicalised). A name may appear in more than
# one group (e.g. "ted" -> edward & theodore); such ambiguity is resolved to
# REVIEW, never auto-merged.
_ABBREVIATION_GROUPS: list[set[str]] = [
    {"william", "bill", "will", "billy"},
    {"samuel", "sam", "sammy"},
    {"elizabeth", "liz", "beth", "eliza", "lizzie", "betty"},
    {"robert", "rob", "bob", "bobby"},
    {"richard", "rick", "rich", "richie"},
    {"michael", "mike", "mikey", "mick"},
    {"james", "jim", "jimmy", "jamie"},
    {"john", "johnny", "jack"},
    {"margaret", "maggie", "meg", "peggy"},
    {"katherine", "kate", "katie", "kathy", "kat", "catherine"},
    {"thomas", "tom", "tommy"},
    {"charles", "charlie", "chuck"},
    {"joseph", "joe", "joey"},
    {"daniel", "dan", "danny"},
    {"matthew", "matt", "matty"},
    {"anthony", "tony"},
    {"christopher", "chris"},
    {"nicholas", "nick", "nicky"},
    {"alexander", "alex", "sasha"},
    {"benjamin", "ben", "benny"},
    {"edward", "ed", "eddie", "ted", "ned"},
    {"andrew", "andy", "drew"},
    {"frederick", "fred", "freddy"},
    {"theodore", "theo", "ted", "teddy"},
    {"jonathan", "jon", "jonny"},
    {"stephen", "steve", "stevie", "steven"},
    {"timothy", "tim", "timmy"},
]


@dataclass
class Resolution:
    """The 2a verdict for one extracted entity."""

    entity_key: str
    decision: str  # AUTO_MERGE | NEEDS_REVIEW | NEW
    rule: str  # which rule fired
    target_key: str | None = None  # set iff AUTO_MERGE
    candidates: list[str] = field(default_factory=list)  # existing keys, for REVIEW


# --- string similarity primitives ---------------------------------------------


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def name_similarity(a: str, b: str) -> float:
    """1 - normalised Levenshtein distance over canonicalised names. [0, 1]."""
    ca, cb = canonicalize(a), canonicalize(b)
    if not ca and not cb:
        return 1.0
    longest = max(len(ca), len(cb))
    if longest == 0:
        return 1.0
    return 1.0 - levenshtein(ca, cb) / longest


def _tokens(name: str) -> set[str]:
    return {t for t in canonicalize(name).split("_") if t}


def token_overlap(a: str, b: str) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    shared = ta & tb
    if not any(len(t) >= 2 for t in shared):
        return False
    if ta <= tb or tb <= ta:
        return True
    return len(shared) / len(ta | tb) >= JACCARD_THRESHOLD


def _shares_abbreviation(a: str, b: str) -> bool:
    ca, cb = canonicalize(a), canonicalize(b)
    return any(ca in g and cb in g for g in _ABBREVIATION_GROUPS)


# --- resolution ----------------------------------------------------------------


def _decide_from_matches(entity_key, rule, matches: list[str]) -> Resolution | None:
    """Auto-merge rules (alias, abbreviation): unique match -> AUTO_MERGE,
    multiple matches -> NEEDS_REVIEW (ambiguity is never auto-merged)."""
    if not matches:
        return None
    if len(matches) == 1:
        return Resolution(entity_key, AUTO_MERGE, rule, target_key=matches[0])
    return Resolution(entity_key, NEEDS_REVIEW, rule, candidates=sorted(matches))


def resolve_entity(entity: Node, existing_nodes, self_key: str | None = None) -> Resolution:
    """Run the deterministic rules in order against existing nodes.

    `existing_nodes` is any iterable of Node. The entity itself, if already
    present by key, counts as an exact match (idempotent re-extraction).
    `self_key`, if the graph has one designated, forces any generic
    self-placeholder name straight to it, bypassing kind-matching entirely
    (rule 0 — see SELF_PLACEHOLDER_NAMES)."""
    entity_canon = canonicalize(entity.name)

    # 0. self-reference — force-merge regardless of kind (see module docstring
    # on SELF_PLACEHOLDER_NAMES for why this is the one rule allowed to be
    # unconditional rather than confidence-banded).
    if self_key is not None and entity_canon in SELF_PLACEHOLDER_NAMES:
        return Resolution(entity.key, AUTO_MERGE, RULE_SELF, target_key=self_key)

    same_kind = [n for n in existing_nodes if n.kind == entity.kind]
    by_key = {n.key: n for n in same_kind}

    # 1. exact (name + type, encoded in the key)
    if entity.key in by_key:
        return Resolution(entity.key, AUTO_MERGE, RULE_EXACT, target_key=entity.key)

    # 2. alias — existing node lists the extracted name as an alias
    alias_matches = [
        n.key for n in same_kind if entity_canon in {canonicalize(a) for a in n.aliases}
    ]
    decided = _decide_from_matches(entity.key, RULE_ALIAS, alias_matches)
    if decided:
        return decided

    # 3. abbreviation — shared name-abbreviation group
    abbrev_matches = [n.key for n in same_kind if _shares_abbreviation(entity.name, n.name)]
    decided = _decide_from_matches(entity.key, RULE_ABBREVIATION, abbrev_matches)
    if decided:
        return decided

    # 4. fuzzy — Levenshtein similarity strictly above threshold (review only)
    fuzzy_matches = sorted(
        n.key for n in same_kind if name_similarity(entity.name, n.name) > FUZZY_THRESHOLD
    )
    if fuzzy_matches:
        return Resolution(entity.key, NEEDS_REVIEW, RULE_FUZZY, candidates=fuzzy_matches)

    # 5. token overlap (review only)
    token_matches = sorted(
        n.key for n in same_kind if token_overlap(entity.name, n.name)
    )
    if token_matches:
        return Resolution(entity.key, NEEDS_REVIEW, RULE_TOKEN, candidates=token_matches)

    # no candidates after all five rules — definitively new
    return Resolution(entity.key, NEW, RULE_NONE)


def resolve_entities(entities, existing_nodes, self_key: str | None = None) -> list[Resolution]:
    existing = list(existing_nodes)
    return [resolve_entity(e, existing, self_key=self_key) for e in entities]
