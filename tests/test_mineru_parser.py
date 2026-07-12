from io import BytesIO
from zipfile import ZipFile

import pytest

from rag_project.parsers import ParseOptions
from rag_project.parsers.base import UploadedFile
from rag_project.parsers.image_explanations import ImageExplanationResult
from rag_project.parsers.mineru import MinerUApiParser


class FakeStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put_bytes(self, object_key: str, payload: bytes, content_type: str | None = None):
        self.objects[object_key] = payload
        return None

    def object_url(self, object_key: str) -> str:
        return f"http://minio/rag/{object_key}"


class FakeImageExplainer:
    async def enrich_markdown(self, markdown_text, *, kb_id, document_id, image_assets, language):
        return ImageExplanationResult(markdown_text=f"{markdown_text}> 图片解释：测试说明\n", chunks=[])


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


@pytest.mark.asyncio
async def test_persist_zip_artifacts_can_write_image_explanations_back_to_markdown() -> None:
    storage = FakeStorage()
    parser = MinerUApiParser(base_url="http://mineru", storage=storage, image_explainer=FakeImageExplainer())

    artifacts = await parser._persist_zip_artifacts(  # noqa: SLF001 - parser artifact boundary test
        build_zip(),
        "demo.pdf",
        ParseOptions(kb_id="kb", document_id="doc"),
    )

    assert "> 图片解释：测试说明" in artifacts["markdown_text"]
    stored_markdown = storage.objects["parsed/kb/doc/markdown/demo.md"].decode("utf-8")
    assert "> 图片解释：测试说明" in stored_markdown


def test_mineru_task_form_matches_fastapi_contract() -> None:
    form = MinerUApiParser._to_form_data(ParseOptions(kb_id="kb", document_id="doc"))  # noqa: SLF001

    assert form["lang_list"] == ["ch"]
    assert form["backend"] == "pipeline"
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


@pytest.mark.asyncio
async def test_parse_reports_progress_after_mineru_submit(monkeypatch) -> None:
    class FakeResponse:
        def __init__(self, payload=None, content=b""):
            self._payload = payload or {}
            self.content = content

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, path):
            if path == "/health":
                return FakeResponse({"status": "healthy"})
            if path == "/tasks/mineru_task_1":
                return FakeResponse({"status": "completed"})
            if path == "/tasks/mineru_task_1/result":
                return FakeResponse(content=build_zip())
            raise AssertionError(f"unexpected GET {path}")

        async def post(self, path, *, data, files):
            assert path == "/tasks"
            return FakeResponse({"task_id": "mineru_task_1"})

    events = []
    storage = FakeStorage()
    parser = MinerUApiParser(base_url="http://mineru", storage=storage)
    monkeypatch.setattr("rag_project.parsers.mineru.httpx.AsyncClient", FakeClient)

    async def record_progress(stage, details):
        events.append((stage, details))

    parsed = await parser.parse(
        UploadedFile(filename="demo.pdf", content=b"%PDF", content_type="application/pdf"),
        ParseOptions(kb_id="kb", document_id="doc"),
        progress_callback=record_progress,
    )

    assert parsed.parser_task_id == "mineru_task_1"
    assert [stage for stage, _ in events] == [
        "raw_saved",
        "mineru_submitted",
        "mineru_finished",
        "result_downloaded",
        "artifacts_persisted",
    ]
    assert events[1][1]["parser_task_id"] == "mineru_task_1"
