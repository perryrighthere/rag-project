from io import BytesIO
from zipfile import ZipFile

import pytest

from rag_project.parsers import ParseOptions
from rag_project.parsers.base import UploadedFile
from rag_project.parsers.mineru import MinerUApiParser


class FakeStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put_bytes(self, object_key: str, payload: bytes, content_type: str | None = None):
        self.objects[object_key] = payload
        return None

    def object_url(self, object_key: str) -> str:
        return f"http://minio/rag/{object_key}"


def build_zip() -> bytes:
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("demo/demo.md", "正文\n![](images/page 1.png)\n")
        archive.writestr("demo/images/page 1.png", b"png-bytes")
        archive.writestr("demo/demo_middle.json", '{"image": "images/page 1.png"}')
        archive.writestr("demo/demo_content_list.json", "[]")
    return stream.getvalue()


@pytest.mark.asyncio
async def test_persist_zip_artifacts_rewrites_and_uploads() -> None:
    storage = FakeStorage()
    parser = MinerUApiParser(base_url="http://mineru", storage=storage)

    artifacts = await parser._persist_zip_artifacts(  # noqa: SLF001 - targeted adapter boundary test
        build_zip(),
        "demo.pdf",
        ParseOptions(kb_id="kb", document_id="doc"),
    )

    assert artifacts["markdown_object_key"] == "parsed/kb/doc/markdown/demo.md"
    assert artifacts["middle_json_object_key"] == "parsed/kb/doc/json/demo_middle.json"
    assert artifacts["content_list_object_key"] == "parsed/kb/doc/json/demo_content_list.json"
    assert artifacts["image_object_keys"] == ["parsed/kb/doc/images/page_1.png"]
    assert "http://minio/rag/parsed/kb/doc/images/page_1.png" in artifacts["markdown_text"]
    assert storage.objects["parsed/kb/doc/images/page_1.png"] == b"png-bytes"


def test_mineru_task_form_matches_fastapi_contract() -> None:
    form = MinerUApiParser._to_form_data(ParseOptions(kb_id="kb", document_id="doc"))  # noqa: SLF001

    assert form["lang_list"] == ["ch"]
    assert form["return_md"] == "true"
    assert form["return_middle_json"] == "true"
    assert form["return_content_list"] == "true"
    assert form["return_images"] == "true"
    assert form["response_format_zip"] == "true"


@pytest.mark.asyncio
async def test_submit_task_uses_mineru_files_field() -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"task_id": "mineru_task_1"}

    class FakeClient:
        def __init__(self) -> None:
            self.calls = []

        async def post(self, path, *, data, files):
            self.calls.append({"path": path, "data": data, "files": files})
            return FakeResponse()

    client = FakeClient()
    parser = MinerUApiParser(base_url="http://mineru", storage=FakeStorage())

    task_id = await parser._submit_task(  # noqa: SLF001 - verifies MinerU wire contract
        client,
        UploadedFile(filename="demo.pdf", content=b"%PDF", content_type="application/pdf"),
        ParseOptions(kb_id="kb", document_id="doc"),
    )

    assert task_id == "mineru_task_1"
    assert client.calls[0]["path"] == "/tasks"
    assert client.calls[0]["files"][0][0] == "files"
    assert client.calls[0]["files"][0][1] == ("demo.pdf", b"%PDF", "application/pdf")
