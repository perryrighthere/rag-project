import pytest

from rag_project.knowledge_base import MetadataSchema, MetadataValidationError
from rag_project.retrieval import MilvusFilterBuilder


def build_schema() -> MetadataSchema:
    return MetadataSchema(
        fields=[
            {"name": "doc_type", "type": "string", "required": True, "filterable": True},
            {"name": "department", "type": "string", "required": False, "filterable": True},
            {"name": "year", "type": "int", "required": False, "filterable": True},
            {"name": "tags", "type": "string_array", "required": False, "filterable": True},
            {"name": "internal_note", "type": "string", "required": False, "filterable": False},
        ]
    )


def test_document_metadata_validation_accepts_declared_fields() -> None:
    schema = build_schema()

    metadata = schema.validate_document_metadata(
        {"doc_type": "policy", "department": "finance", "year": 2025, "tags": ["travel"]}
    )

    assert metadata["doc_type"] == "policy"


def test_document_metadata_validation_rejects_missing_unknown_and_wrong_type() -> None:
    schema = build_schema()

    with pytest.raises(MetadataValidationError, match="missing required"):
        schema.validate_document_metadata({"department": "finance"})
    with pytest.raises(MetadataValidationError, match="unknown metadata fields"):
        schema.validate_document_metadata({"doc_type": "policy", "unknown": "x"})
    with pytest.raises(MetadataValidationError, match="year must be int"):
        schema.validate_document_metadata({"doc_type": "policy", "year": "2025"})


def test_milvus_filter_builder_validates_and_escapes_expression() -> None:
    expr = MilvusFilterBuilder().build(
        kb_id='kb_"policy',
        metadata_schema=build_schema(),
        filters={
            "doc_type": {"$eq": 'po"licy'},
            "year": {"$gte": 2024},
            "department": {"$in": ["finance", "hr"]},
            "tags": {"$contains": "travel"},
        },
    )

    assert expr.startswith('kb_id == "kb_\\"policy" and')
    assert 'doc_type == "po\\"licy"' in expr
    assert "year >= 2024" in expr
    assert 'department in ["finance", "hr"]' in expr
    assert 'array_contains(tags, "travel")' in expr


def test_milvus_filter_builder_rejects_unfilterable_and_bad_operator() -> None:
    builder = MilvusFilterBuilder()
    schema = build_schema()

    with pytest.raises(MetadataValidationError, match="not filterable"):
        builder.build(kb_id="kb", metadata_schema=schema, filters={"internal_note": {"$eq": "x"}})
    with pytest.raises(MetadataValidationError, match="not supported"):
        builder.build(kb_id="kb", metadata_schema=schema, filters={"department": {"$gt": "finance"}})
