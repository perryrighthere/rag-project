from fastapi.testclient import TestClient

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
