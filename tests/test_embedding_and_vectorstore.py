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
            {"name": "internal_note", "type": "string", "filterable": False},
        ]
    )
    chunk = ChunkRecord(
        kb_id="kb",
        document_id="doc",
        chunk_index=2,
        text="正文",
        heading_path="制度 > 报销",
        source_uri="minio://rag/parsed/kb/doc/markdown/demo.md",
        metadata={"doc_type": "policy", "year": 2025, "tags": ["travel"], "internal_note": "draft"},
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
    assert row["internal_note"] == "draft"
    assert row["embedding"] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_milvus_metadata_update_preserves_existing_vector() -> None:
    class FakeMilvusClient:
        def __init__(self) -> None:
            self.rows = {"chunk_1": {"id": "chunk_1", "embedding": [0.1, 0.2, 0.3]}}
            self.upserted = []

        def has_collection(self, collection_name):
            return True

        def get(self, *, collection_name, ids, output_fields):
            return [self.rows[chunk_id] for chunk_id in ids if chunk_id in self.rows]

        def upsert(self, *, collection_name, data):
            self.upserted.extend(data)

    client = FakeMilvusClient()
    adapter = MilvusVectorStoreAdapter(VectorStoreConfig(uri="unused"), client=client)
    schema = MetadataSchema(
        fields=[
            {"name": "doc_type", "type": "string", "filterable": True},
            {"name": "internal_note", "type": "string", "filterable": False},
        ]
    )
    chunk = ChunkRecord(
        chunk_id="chunk_1",
        kb_id="kb",
        document_id="doc",
        chunk_index=0,
        text="正文",
        metadata={"doc_type": "guide", "internal_note": "published"},
    )

    await adapter.update_chunk_metadata([chunk], metadata_schema=schema)

    assert len(client.upserted) == 1
    assert client.upserted[0]["embedding"] == [0.1, 0.2, 0.3]
    assert client.upserted[0]["metadata_json"] == {"doc_type": "guide", "internal_note": "published"}
    assert client.upserted[0]["doc_type"] == "guide"
    assert client.upserted[0]["internal_note"] == "published"
