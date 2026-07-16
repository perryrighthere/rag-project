import asyncio

from fastapi.testclient import TestClient

from rag_project.api import routes
from rag_project.chunking import ChunkRecord
from rag_project.db import get_store
from rag_project.main import create_app


def test_health_endpoint() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_kb_and_upload_document() -> None:
    client = TestClient(create_app())
    kb_response = client.post(
        "/knowledge-bases",
        json={
            "name": "policy",
            "metadata_schema": {
                "fields": [
                    {"name": "doc_type", "type": "string", "required": True, "filterable": True}
                ]
            },
        },
    )
    assert kb_response.status_code == 201
    kb_id = kb_response.json()["kb_id"]

    doc_response = client.post(
        f"/knowledge-bases/{kb_id}/documents",
        files={"file": ("demo.pdf", b"%PDF", "application/pdf")},
        data={"metadata": '{"doc_type": "policy"}'},
    )

    assert doc_response.status_code == 201
    payload = doc_response.json()
    assert payload["kb_id"] == kb_id
    assert payload["filename"] == "demo.pdf"
    assert payload["metadata"] == {"doc_type": "policy"}
    assert "file_content" not in payload

    metadata_response = client.patch(
        f"/documents/{payload['document_id']}/metadata",
        json={"metadata": {"doc_type": "guide"}},
    )
    assert metadata_response.status_code == 200
    assert metadata_response.json()["metadata"] == {"doc_type": "guide"}


def test_update_indexed_document_metadata_synchronizes_chunks_and_milvus(monkeypatch) -> None:
    class FakeVectorStore:
        def __init__(self) -> None:
            self.chunks = []

        async def update_chunk_metadata(self, chunks, *, metadata_schema):
            self.chunks = chunks

    client = TestClient(create_app())
    kb_response = client.post(
        "/knowledge-bases",
        json={
            "name": "indexed policy",
            "metadata_schema": {
                "fields": [
                    {"name": "doc_type", "type": "string", "required": True, "filterable": True},
                    {"name": "note", "type": "string", "filterable": False},
                ]
            },
        },
    )
    kb_id = kb_response.json()["kb_id"]
    doc_response = client.post(
        f"/knowledge-bases/{kb_id}/documents",
        files={"file": ("indexed.pdf", b"%PDF", "application/pdf")},
        data={"metadata": '{"doc_type": "policy", "note": "draft"}'},
    )
    document_id = doc_response.json()["document_id"]
    chunk = ChunkRecord(
        kb_id=kb_id,
        document_id=document_id,
        chunk_index=0,
        text="正文",
        metadata={"doc_type": "policy", "note": "draft", "chunk_type": "text"},
    )
    asyncio.run(get_store().replace_document_chunks(document_id, [chunk]))
    vector_store = FakeVectorStore()
    monkeypatch.setattr(routes, "get_vector_store", lambda: vector_store)

    response = client.patch(
        f"/documents/{document_id}/metadata",
        json={"metadata": {"doc_type": "guide"}},
    )

    assert response.status_code == 200
    assert response.json()["metadata"] == {"doc_type": "guide"}
    assert vector_store.chunks[0].metadata == {"doc_type": "guide", "chunk_type": "text"}
    assert get_store().list_document_chunks(document_id)[0].metadata == {
        "doc_type": "guide",
        "chunk_type": "text",
    }


def test_upload_document_rejects_metadata_outside_schema() -> None:
    client = TestClient(create_app())
    kb_response = client.post(
        "/knowledge-bases",
        json={
            "name": "strict policy",
            "metadata_schema": {
                "fields": [
                    {"name": "doc_type", "type": "string", "required": True, "filterable": True}
                ]
            },
        },
    )
    kb_id = kb_response.json()["kb_id"]

    response = client.post(
        f"/knowledge-bases/{kb_id}/documents",
        files={"file": ("demo.pdf", b"%PDF", "application/pdf")},
        data={"metadata": '{"doc_type": "policy", "department": "finance"}'},
    )

    assert response.status_code == 422
    assert "unknown metadata fields" in response.json()["detail"]
