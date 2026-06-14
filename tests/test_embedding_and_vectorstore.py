import pytest

from rag_project.chunking import ChunkRecord
from rag_project.embeddings import EmbeddingConfig, EmbeddingDimensionError, OpenAICompatibleEmbeddingClient
from rag_project.knowledge_base import MetadataSchema
from rag_project.vectorstores import MilvusVectorStoreAdapter, VectorStoreConfig


def test_embedding_dimension_validation() -> None:
    client = OpenAICompatibleEmbeddingClient.__new__(OpenAICompatibleEmbeddingClient)
    client.config = EmbeddingConfig(base_url=None, api_key="EMPTY", model="embed", dim=3)

    client.validate_vectors([[1.0, 2.0, 3.0]])
    with pytest.raises(EmbeddingDimensionError):
        client.validate_vectors([[1.0, 2.0]])


def test_milvus_chunk_row_contains_required_payload_fields() -> None:
    adapter = MilvusVectorStoreAdapter.__new__(MilvusVectorStoreAdapter)
    adapter.config = VectorStoreConfig(uri="unused")
    schema = MetadataSchema(
        fields=[
            {"name": "doc_type", "type": "string", "filterable": True},
            {"name": "year", "type": "int", "filterable": True},
            {"name": "tags", "type": "string_array", "filterable": True},
        ]
    )
    chunk = ChunkRecord(
        kb_id="kb",
        document_id="doc",
        chunk_index=2,
        text="正文",
        heading_path="制度 > 报销",
        source_uri="minio://rag/parsed/kb/doc/markdown/demo.md",
        metadata={"doc_type": "policy", "year": 2025, "tags": ["travel"]},
    )

    row = adapter._chunk_row(chunk, [0.1, 0.2, 0.3], schema)  # noqa: SLF001

    assert row["id"] == chunk.chunk_id
    assert row["kb_id"] == "kb"
    assert row["document_id"] == "doc"
    assert row["chunk_index"] == 2
    assert row["metadata_json"]["doc_type"] == "policy"
    assert row["doc_type"] == "policy"
    assert row["year"] == 2025
    assert row["tags"] == ["travel"]
    assert row["embedding"] == [0.1, 0.2, 0.3]
