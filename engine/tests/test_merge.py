"""Unit tests for merge Sub-stage 2a (deterministic candidate generation).

This is the highest-risk layer; coverage is deliberately heavy. The governing
invariant under test: rules 4-5 and every ambiguous match NEVER auto-merge.
"""

import pytest

from ontomem.merge import (
    AUTO_MERGE,
    NEEDS_REVIEW,
    NEW,
    RULE_ABBREVIATION,
    RULE_ALIAS,
    RULE_EXACT,
    RULE_FUZZY,
    RULE_NONE,
    RULE_SELF,
    RULE_TOKEN,
    levenshtein,
    name_similarity,
    resolve_entity,
    token_overlap,
)
from ontomem.model import Node


def node(kind, name, **kw):
    return Node.create(kind, name, **kw)


# --- string primitives ---------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,dist",
    [("", "", 0), ("abc", "abc", 0), ("abc", "abd", 1), ("kitten", "sitting", 3), ("", "abc", 3)],
)
def test_levenshtein(a, b, dist):
    assert levenshtein(a, b) == dist


def test_name_similarity_identical_and_close():
    assert name_similarity("Sarah", "Sarah") == 1.0
    assert name_similarity("Jonathan", "Jonathon") == pytest.approx(1 - 1 / 8)  # 0.875


def test_name_similarity_canonicalises():
    assert name_similarity("New Jersey", "new_jersey") == 1.0


def test_token_overlap_subset_and_jaccard():
    assert token_overlap("Elon", "Elon Musk") is True  # subset
    assert token_overlap("Elon Musk", "Elon") is True  # superset
    assert token_overlap("New York", "New Jersey") is False  # only "new" shared, jaccard 1/3
    assert token_overlap("Sarah", "Bill") is False


# --- rule 1: exact -------------------------------------------------------------


def test_exact_match_auto_merges():
    existing = [node("PERSON", "Sarah")]
    r = resolve_entity(node("PERSON", "Sarah"), existing)
    assert r.decision == AUTO_MERGE
    assert r.rule == RULE_EXACT
    assert r.target_key == "PERSON::sarah"


def test_same_name_different_type_is_not_exact():
    # PERSON::bill vs ORG::bill must never merge (different kind)
    existing = [node("ORG", "Bill")]
    r = resolve_entity(node("PERSON", "Bill"), existing)
    assert r.decision == NEW
    assert r.rule == RULE_NONE


# --- rule 2: alias -------------------------------------------------------------


def test_alias_match_auto_merges():
    existing = [node("ORG", "Walmart", aliases=["Wal-Mart", "WMT"])]
    r = resolve_entity(node("ORG", "Wal-Mart"), existing)
    assert r.decision == AUTO_MERGE
    assert r.rule == RULE_ALIAS
    assert r.target_key == "ORG::walmart"


def test_ambiguous_alias_downgrades_to_review():
    # two existing nodes both claim the alias -> never auto-merge
    existing = [
        node("PERSON", "Robert", aliases=["Bob"]),
        node("PERSON", "Roberto", aliases=["Bob"]),
    ]
    r = resolve_entity(node("PERSON", "Bob"), existing)
    assert r.decision == NEEDS_REVIEW
    assert r.rule == RULE_ALIAS
    assert set(r.candidates) == {"PERSON::robert", "PERSON::roberto"}


# --- rule 3: abbreviation ------------------------------------------------------


def test_abbreviation_auto_merges_unique():
    existing = [node("PERSON", "Samuel")]
    r = resolve_entity(node("PERSON", "Sam"), existing)
    assert r.decision == AUTO_MERGE
    assert r.rule == RULE_ABBREVIATION
    assert r.target_key == "PERSON::samuel"


def test_abbreviation_reversed_direction():
    existing = [node("PERSON", "Bill")]
    r = resolve_entity(node("PERSON", "William"), existing)
    assert r.decision == AUTO_MERGE
    assert r.rule == RULE_ABBREVIATION
    assert r.target_key == "PERSON::bill"


def test_ambiguous_abbreviation_downgrades_to_review():
    # "Ted" belongs to both edward and theodore groups
    existing = [node("PERSON", "Edward"), node("PERSON", "Theodore")]
    r = resolve_entity(node("PERSON", "Ted"), existing)
    assert r.decision == NEEDS_REVIEW
    assert r.rule == RULE_ABBREVIATION
    assert set(r.candidates) == {"PERSON::edward", "PERSON::theodore"}


def test_abbreviation_respects_kind():
    existing = [node("ORG", "Samuel")]  # an org, not a person
    r = resolve_entity(node("PERSON", "Sam"), existing)
    assert r.decision == NEW


# --- rule 4: fuzzy (review only) ----------------------------------------------


def test_fuzzy_match_needs_review_never_auto():
    existing = [node("PERSON", "Jonathan")]
    r = resolve_entity(node("PERSON", "Jonathon"), existing)
    assert r.decision == NEEDS_REVIEW
    assert r.rule == RULE_FUZZY
    assert r.candidates == ["PERSON::jonathan"]
    assert r.target_key is None


def test_fuzzy_below_threshold_is_new():
    existing = [node("PERSON", "Sarah")]
    r = resolve_entity(node("PERSON", "Steven"), existing)
    assert r.decision == NEW


# --- rule 5: token overlap (review only) --------------------------------------


def test_token_overlap_needs_review():
    existing = [node("PERSON", "Elon Musk")]
    r = resolve_entity(node("PERSON", "Elon"), existing)
    assert r.decision == NEEDS_REVIEW
    assert r.rule == RULE_TOKEN
    assert r.candidates == ["PERSON::elon_musk"]


def test_distinct_places_sharing_one_token_are_new():
    existing = [node("PLACE", "New York")]
    r = resolve_entity(node("PLACE", "New Jersey"), existing)
    assert r.decision == NEW


# --- rule ordering & no-match --------------------------------------------------


def test_exact_takes_precedence_over_alias():
    existing = [
        node("PERSON", "Sam"),  # exact
        node("PERSON", "Samuel", aliases=["Sam"]),  # alias + abbreviation
    ]
    r = resolve_entity(node("PERSON", "Sam"), existing)
    assert r.rule == RULE_EXACT
    assert r.target_key == "PERSON::sam"


def test_no_candidates_is_new():
    existing = [node("PERSON", "Sarah"), node("ORG", "Walmart")]
    r = resolve_entity(node("PERSON", "Zorp"), existing)
    assert r.decision == NEW
    assert r.rule == RULE_NONE
    assert r.candidates == []
    assert r.target_key is None


def test_empty_graph_is_new():
    r = resolve_entity(node("PERSON", "Bill"), [])
    assert r.decision == NEW


# --- rule 0: self-reference (dogfooding found real graphs fragmenting the
# user's own node across conversations that don't restate their name) --------


def test_self_placeholder_force_merges_to_self_key_ignoring_kind():
    # OTHER::user must resolve to the designated self_key even though the
    # existing self node is PERSON-kind -- this is the whole bug: node keys
    # embed kind, so exact/alias matching alone can never bridge a kind
    # mismatch, but self-reference is exempt from kind-matching entirely.
    existing = [node("PERSON", "Maya")]
    r = resolve_entity(node("OTHER", "User"), existing, self_key="PERSON::maya")
    assert r.decision == AUTO_MERGE
    assert r.rule == RULE_SELF
    assert r.target_key == "PERSON::maya"


@pytest.mark.parametrize("placeholder", ["User", "the user", "Me", "myself", "USER"])
def test_self_placeholder_variants_all_merge(placeholder):
    r = resolve_entity(node("PERSON", placeholder), [], self_key="PERSON::maya")
    assert r.decision == AUTO_MERGE
    assert r.rule == RULE_SELF
    assert r.target_key == "PERSON::maya"


def test_self_rule_inert_without_a_designated_self_key():
    # No self_key yet (e.g. the very first extraction in an empty graph) ->
    # falls through to the normal rules, so "User" becomes a plain new node
    # rather than merging into nothing.
    r = resolve_entity(node("PERSON", "User"), [], self_key=None)
    assert r.decision == NEW


def test_self_rule_does_not_catch_an_unrelated_real_name():
    # "Us" or "Userman" etc. must not accidentally match -- only the exact
    # fixed placeholder set does.
    r = resolve_entity(node("PERSON", "Userman"), [], self_key="PERSON::maya")
    assert r.decision != AUTO_MERGE or r.rule != RULE_SELF
