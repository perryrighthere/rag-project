import asyncio
import base64
import json
import mimetypes
import zipfile
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any

import httpx

from rag_project.parsers.base import DocumentParser, ParsedDocument, ParseOptions, UploadedFile
from rag_project.parsers.image_explanations import ImageAsset, ImageExplanationGenerator
from rag_project.storage import (
    MinioStorage,
    build_parsed_image_key,
    build_parsed_json_key,
    build_parsed_markdown_key,
    build_raw_object_key,
    rewrite_relative_image_paths,
    sanitize_object_part,
)


class MinerUApiError(RuntimeError):
    pass


ParseProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


class MinerUApiParser(DocumentParser):
    """MinerU HTTP API adapter with MinIO artifact persistence."""

    success_statuses = {"success", "succeeded", "finished", "completed", "done"}
    failure_statuses = {"failed", "failure", "error", "cancelled", "canceled"}

    def __init__(
        self,
        *,
        base_url: str,
        storage: MinioStorage,
        request_timeout: float = 60.0,
        poll_interval_seconds: float = 2.0,
        max_wait_seconds: float = 600.0,
        image_explainer: ImageExplanationGenerator | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.storage = storage
        self.request_timeout = request_timeout
        self.poll_interval_seconds = poll_interval_seconds
        self.max_wait_seconds = max_wait_seconds
        self.image_explainer = image_explainer

    async def parse(
        self,
        file: UploadedFile,
        options: ParseOptions,
        progress_callback: ParseProgressCallback | None = None,
    ) -> ParsedDocument:
        filename = sanitize_object_part(file.filename)
        raw_key = build_raw_object_key(options.kb_id, options.document_id, filename)
        await self.storage.put_bytes(raw_key, file.content, file.content_type)
        if progress_callback is not None:
            await progress_callback("raw_saved", {"raw_object_key": raw_key})

        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.request_timeout) as client:
            await self._check_health(client)
            parser_task_id = await self._submit_task(client, file, options)
            if progress_callback is not None:
                await progress_callback(
                    "mineru_submitted",
                    {"parser_task_id": parser_task_id, "raw_object_key": raw_key},
                )
            await self._wait_for_task(client, parser_task_id)
            if progress_callback is not None:
                await progress_callback("mineru_finished", {"parser_task_id": parser_task_id})
            zip_payload = await self._download_result_zip(client, parser_task_id)
            if progress_callback is not None:
                await progress_callback("result_downloaded", {"parser_task_id": parser_task_id})

        if progress_callback is not None:
            await progress_callback("persisting_artifacts", {"parser_task_id": parser_task_id})
        artifacts = await self._persist_zip_artifacts(zip_payload, filename, options)
        if progress_callback is not None:
            await progress_callback(
                "artifacts_persisted",
                {
                    "parser_task_id": parser_task_id,
                    "markdown_object_key": artifacts["markdown_object_key"],
                },
            )
        return ParsedDocument(
            document_id=options.document_id,
            parser="mineru",
            parser_task_id=parser_task_id,
            markdown_text=artifacts["markdown_text"],
            markdown_object_key=artifacts["markdown_object_key"],
            content_list_object_key=artifacts.get("content_list_object_key"),
            middle_json_object_key=artifacts.get("middle_json_object_key"),
            image_object_keys=artifacts["image_object_keys"],
            image_explanation_chunks=artifacts["image_explanation_chunks"],
            raw_object_key=raw_key,
            parse_options=options.model_dump(mode="json"),
            created_at=datetime.now(timezone.utc),
        )

    async def _check_health(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/health")
        response.raise_for_status()

    async def _submit_task(self, client: httpx.AsyncClient, file: UploadedFile, options: ParseOptions) -> str:
        response = await client.post(
            "/tasks",
            data=self._to_form_data(options),
            files=[
                (
                    "files",
                    (
                        file.filename,
                        file.content,
                        file.content_type or "application/octet-stream",
                    ),
                )
            ],
        )
        response.raise_for_status()
        payload = response.json()
        task_id = self._first(payload, "task_id", "id", "data.task_id", "data.id")
        if not task_id:
            raise MinerUApiError(f"MinerU task response did not include a task id: {payload}")
        return str(task_id)

    async def _wait_for_task(self, client: httpx.AsyncClient, task_id: str) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + self.max_wait_seconds
        last_payload: dict[str, Any] = {}

        while True:
            response = await client.get(f"/tasks/{task_id}")
            response.raise_for_status()
            last_payload = response.json()
            status = str(self._first(last_payload, "status", "state", "data.status", "data.state") or "").lower()

            if status in self.success_statuses:
                return last_payload
            if status in self.failure_statuses:
                message = self._first(last_payload, "error", "message", "data.error", "data.message") or last_payload
                raise MinerUApiError(f"MinerU task {task_id} failed: {message}")
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"MinerU task {task_id} did not finish within {self.max_wait_seconds} seconds")

            await asyncio.sleep(self.poll_interval_seconds)

    async def _download_result_zip(self, client: httpx.AsyncClient, task_id: str) -> bytes:
        response = await client.get(f"/tasks/{task_id}/result")
        response.raise_for_status()
        if self._looks_like_zip(response.content):
            return response.content

        payload = response.json()
        encoded = self._first(payload, "zip_base64", "data.zip_base64", "result_zip_base64")
        if encoded:
            return base64.b64decode(encoded)

        download_url = self._first(payload, "download_url", "result_url", "zip_url", "data.download_url", "data.result_url")
        if download_url:
            followup = await client.get(str(download_url))
            followup.raise_for_status()
            if self._looks_like_zip(followup.content):
                return followup.content

        raise MinerUApiError(f"MinerU result for task {task_id} was not a zip payload")

    async def _persist_zip_artifacts(self, zip_payload: bytes, filename: str, options: ParseOptions) -> dict[str, Any]:
        doc_stem = PurePosixPath(filename).stem or "document"
        markdown_text = ""
        markdown_object_key = ""
        content_list_object_key: str | None = None
        middle_json_object_key: str | None = None
        image_object_keys: list[str] = []
        image_assets: list[ImageAsset] = []
        image_explanation_chunks = []

        files = await asyncio.to_thread(self._extract_zip_files, zip_payload)
        image_members = {name: data for name, data in files if self._is_image_member(name)}

        def sync_image_url_for_name(image_name: str) -> str:
            object_key = build_parsed_image_key(options.kb_id, options.document_id, image_name)
            return self.storage.object_url(object_key)

        async def persist_image(member_name: str, payload: bytes) -> tuple[str, ImageAsset]:
            image_name = self._image_name(member_name)
            object_key = build_parsed_image_key(options.kb_id, options.document_id, image_name)
            content_type = mimetypes.guess_type(image_name)[0] or "application/octet-stream"
            await self.storage.put_bytes(object_key, payload, content_type)
            return object_key, ImageAsset(
                image_url=self.storage.object_url(object_key),
                object_key=object_key,
                content=payload,
            )

        persisted_images = await asyncio.gather(
            *(persist_image(member_name, payload) for member_name, payload in sorted(image_members.items()))
        )
        for object_key, image_asset in persisted_images:
            image_object_keys.append(object_key)
            image_assets.append(image_asset)

        for member_name, payload in sorted(files):
            if self._is_ignored_member(member_name) or self._is_image_member(member_name):
                continue

            suffix = PurePosixPath(member_name).suffix.lower()
            if suffix == ".md":
                text = payload.decode("utf-8")
                text = rewrite_relative_image_paths(text, sync_image_url_for_name)
                if self.image_explainer is not None:
                    result = await self.image_explainer.enrich_markdown(
                        text,
                        kb_id=options.kb_id,
                        document_id=options.document_id,
                        image_assets=image_assets,
                        language=options.lang_list[0] if options.lang_list else "ch",
                    )
                    text = result.markdown_text
                    image_explanation_chunks.extend(result.chunks)
                object_key = build_parsed_markdown_key(options.kb_id, options.document_id, filename)
                await self.storage.put_bytes(object_key, text.encode("utf-8"), "text/markdown; charset=utf-8")
                markdown_text = text
                markdown_object_key = object_key
            elif suffix == ".json":
                kind = self._json_kind(member_name)
                text = payload.decode("utf-8")
                text = rewrite_relative_image_paths(text, sync_image_url_for_name)
                object_key = build_parsed_json_key(options.kb_id, options.document_id, doc_stem, kind)
                await self.storage.put_bytes(object_key, text.encode("utf-8"), "application/json")
                if kind == "content_list":
                    content_list_object_key = object_key
                elif kind == "middle":
                    middle_json_object_key = object_key

        if not markdown_object_key:
            raise FileNotFoundError("MinerU result zip did not contain a Markdown file")

        return {
            "markdown_text": markdown_text,
            "markdown_object_key": markdown_object_key,
            "content_list_object_key": content_list_object_key,
            "middle_json_object_key": middle_json_object_key,
            "image_object_keys": image_object_keys,
            "image_explanation_chunks": image_explanation_chunks,
        }

    @staticmethod
    def _extract_zip_files(zip_payload: bytes) -> list[tuple[str, bytes]]:
        with zipfile.ZipFile(BytesIO(zip_payload)) as archive:
            return [(info.filename, archive.read(info)) for info in archive.infolist() if not info.is_dir()]

    @staticmethod
    def _to_form_data(options: ParseOptions) -> dict[str, str | list[str]]:
        payload = options.model_dump(exclude={"kb_id", "document_id"})
        form: dict[str, str | list[str]] = {}
        for key, value in payload.items():
            if isinstance(value, bool):
                form[key] = str(value).lower()
            elif isinstance(value, list):
                form[key] = [str(item) for item in value]
            elif isinstance(value, dict):
                form[key] = json.dumps(value, ensure_ascii=False)
            elif value is not None:
                form[key] = str(value)
        return form

    @staticmethod
    def _first(payload: dict[str, Any], *paths: str) -> Any:
        for path in paths:
            current: Any = payload
            for part in path.split("."):
                if not isinstance(current, dict) or part not in current:
                    current = None
                    break
                current = current[part]
            if current is not None:
                return current
        return None

    @staticmethod
    def _looks_like_zip(payload: bytes) -> bool:
        return payload.startswith(b"PK\x03\x04") or payload.startswith(b"PK\x05\x06") or payload.startswith(b"PK\x07\x08")

    @staticmethod
    def _is_ignored_member(name: str) -> bool:
        normalized = PurePosixPath(name)
        return any(part.startswith("__MACOSX") or part.startswith(".") for part in normalized.parts)

    @classmethod
    def _is_image_member(cls, name: str) -> bool:
        if cls._is_ignored_member(name):
            return False
        return PurePosixPath(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

    @staticmethod
    def _image_name(member_name: str) -> str:
        path = PurePosixPath(member_name)
        parts = list(path.parts)
        if "images" in parts:
            return "/".join(parts[parts.index("images") + 1 :])
        return path.name

    @staticmethod
    def _json_kind(member_name: str) -> str:
        lowered = PurePosixPath(member_name).name.lower()
        if "content_list" in lowered or "content-list" in lowered:
            return "content_list"
        if "middle" in lowered:
            return "middle"
        return PurePosixPath(member_name).stem
