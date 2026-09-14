"""Hermetic checks on the extraction tool-call JSON Schema (no network)."""

from ontomem.extraction_schema import EXTRACTION_TOOL_NAME, EXTRACTION_TOOL_SCHEMA
from ontomem.model import CARDINALITIES, NODE_KINDS, STABILITIES


def test_schema_top_level_shape():
    assert EXTRACTION_TOOL_SCHEMA["type"] == "function"
    fn = EXTRACTION_TOOL_SCHEMA["function"]
    assert fn["name"] == EXTRACTION_TOOL_NAME
    params = fn["parameters"]
    assert params["required"] == ["entities", "relationships", "episode"]


def test_entity_enum_matches_node_kinds():
    entity_schema = EXTRACTION_TOOL_SCHEMA["function"]["parameters"]["properties"]["entities"]["items"]
    assert set(entity_schema["properties"]["type"]["enum"]) == set(NODE_KINDS)


def test_relationship_enums_match_model_vocab():
    rel_schema = EXTRACTION_TOOL_SCHEMA["function"]["parameters"]["properties"]["relationships"]["items"]
    props = rel_schema["properties"]
    assert set(props["stability"]["enum"]) == set(STABILITIES)
    assert set(props["cardinality"]["enum"]) == set(CARDINALITIES)


def test_relationship_required_fields_cover_edge_create_args():
    rel_schema = EXTRACTION_TOOL_SCHEMA["function"]["parameters"]["properties"]["relationships"]["items"]
    required = set(rel_schema["required"])
    for field in ("source", "relation", "target", "stability", "cardinality", "properties"):
        assert field in required


def test_episode_required_fields():
    episode_schema = EXTRACTION_TOOL_SCHEMA["function"]["parameters"]["properties"]["episode"]
    assert set(episode_schema["required"]) == {"summary", "importance", "tags"}
