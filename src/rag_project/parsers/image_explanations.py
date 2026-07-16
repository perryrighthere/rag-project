import asyncio
import base64
import mimetypes
import re
import uuid
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from rag_project.parsers.base import ImageExplanationChunk

try:
    from openai import AsyncOpenAI
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    AsyncOpenAI = None


DEFAULT_IMAGE_EXPLANATION_PROMPT = (
    "你是文档图片理解助手。请结合图片内容和给出的文档上下文，"
    "用1到3句中文解释这张图片在文档中的关键信息。"
    "不要虚构，不要输出项目符号，不要使用 Markdown 代码块，只输出解释正文。"
)
IMAGE_EXPLANATION_PREFIX = "图片解释："
IMAGE_EXPLANATION_PREFIX_EN = "Image explanation:"
MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*]\((?P<url>[^)\s]+)(?:\s+\"[^\"]*\")?\)")


class ImageExplanationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    base_url: str | None = None
    api_key: str = "EMPTY"
    model: str | None = None
    prompt: str = DEFAULT_IMAGE_EXPLANATION_PROMPT
    timeout: float = 120.0
    max_tokens: int = 300
    concurrency: int = Field(default=4, ge=1)

    @property
    def is_configured(self) -> bool:
        return bool(self.enabled and self.base_url and self.model)


@dataclass(frozen=True)
class ImageAsset:
    image_url: str
    object_key: str
    content: bytes


class ImageExplanationResult(BaseModel):
    markdown_text: str
    chunks: list[ImageExplanationChunk] = Field(default_factory=list)


class ImageExplanationGenerator:
    def __init__(self, config: ImageExplanationConfig):
        self.config = config

    async def enrich_markdown(
        self,
        markdown_text: str,
        *,
        kb_id: str,
        document_id: str,
        image_assets: list[ImageAsset],
        language: str,
    ) -> ImageExplanationResult:
        if not self.config.enabled:
            return ImageExplanationResult(markdown_text=markdown_text)
        if not self.config.is_configured:
            raise ValueError("VLM image explanation is enabled but base_url or model is missing")
        if AsyncOpenAI is None:
            raise ModuleNotFoundError("The optional dependency 'openai' is required for VLM image explanations")

        image_by_url = {asset.image_url: asset for asset in image_assets}
        if not image_by_url or not MARKDOWN_IMAGE_PATTERN.search(markdown_text):
            return ImageExplanationResult(markdown_text=markdown_text)

        client = self._create_client()
        lines = markdown_text.splitlines()
        requests_by_url: dict[str, tuple[ImageAsset, str]] = {}
        for line_index, line in enumerate(lines):
            image_urls = self._line_image_urls(line)
            if not image_urls:
                continue
            next_line = lines[line_index + 1].strip() if line_index + 1 < len(lines) else ""
            if self._has_existing_explanation(next_line):
                continue
            prompt = self._build_vlm_prompt(self.config.prompt, self._build_markdown_context(lines, line_index))
            for image_url in image_urls:
                asset = image_by_url.get(image_url)
                if asset is not None and image_url not in requests_by_url:
                    requests_by_url[image_url] = (asset, prompt)

        semaphore = asyncio.Semaphore(self.config.concurrency)

        async def explain(asset: ImageAsset, prompt: str) -> str:
            async with semaphore:
                return await self._request_image_explanation(client, prompt=prompt, asset=asset)

        urls = list(requests_by_url)
        explanations = await asyncio.gather(
            *(explain(*requests_by_url[image_url]) for image_url in urls)
        )
        explanations_by_url = dict(zip(urls, explanations, strict=True))

        enriched_lines: list[str] = []
        chunks: list[ImageExplanationChunk] = []
        for line_index, line in enumerate(lines):
            enriched_lines.append(line)
            image_urls = self._line_image_urls(line)
            if not image_urls:
                continue
            next_line = lines[line_index + 1].strip() if line_index + 1 < len(lines) else ""
            if self._has_existing_explanation(next_line):
                continue
            for image_url in image_urls:
                asset = image_by_url.get(image_url)
                explanation = explanations_by_url.get(image_url)
                if asset is None or not explanation:
                    continue
                enriched_lines.append(self._format_image_explanation(explanation, language))
                chunks.append(
                    self._build_chunk(
                        kb_id=kb_id,
                        document_id=document_id,
                        chunk_index=len(chunks),
                        asset=asset,
                        explanation=explanation,
                    )
                )

        enriched_text = "\n".join(enriched_lines)
        if markdown_text.endswith("\n"):
            enriched_text += "\n"
        return ImageExplanationResult(markdown_text=enriched_text, chunks=chunks)

    async def _request_image_explanation(self, client, *, prompt: str, asset: ImageAsset) -> str:
        completion = await client.chat.completions.create(
            model=self.config.model,
            temperature=0.2,
            max_tokens=self.config.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._build_image_data_url(asset.content, asset.object_key)},
                        },
                    ],
                }
            ],
        )
        if not completion.choices:
            return ""
        return self._normalize_vlm_response(completion.choices[0].message.content)

    def _create_client(self):
        return AsyncOpenAI(api_key=self.config.api_key, base_url=self.config.base_url, timeout=self.config.timeout)

    @staticmethod
    def _line_image_urls(line: str) -> list[str]:
        urls: list[str] = []
        for match in MARKDOWN_IMAGE_PATTERN.finditer(line):
            url = match.group("url")
            if url not in urls:
                urls.append(url)
        return urls

    @staticmethod
    def _has_existing_explanation(line: str) -> bool:
        return line.startswith(f"> {IMAGE_EXPLANATION_PREFIX}") or line.startswith(f"> {IMAGE_EXPLANATION_PREFIX_EN}")

    @staticmethod
    def _build_markdown_context(lines: list[str], line_index: int, radius: int = 3) -> str:
        context_lines: list[str] = []
        for offset in range(-radius, radius + 1):
            if offset == 0:
                continue
            candidate_index = line_index + offset
            if candidate_index < 0 or candidate_index >= len(lines):
                continue
            candidate = lines[candidate_index].strip()
            if not candidate or MARKDOWN_IMAGE_PATTERN.search(candidate):
                continue
            context_lines.append(candidate)
        return "\n".join(context_lines[:4])

    @staticmethod
    def _build_vlm_prompt(base_prompt: str, context_text: str) -> str:
        if not context_text:
            return f"{base_prompt}\n\n请只输出最终解释。"
        return (
            f"{base_prompt}\n\n"
            f"文档上下文如下，请将其作为辅助信息而不是必须复述的内容：\n{context_text}\n\n"
            "请只输出最终解释。"
        )

    @staticmethod
    def _build_image_data_url(image_bytes: bytes, object_key: str) -> str:
        mime_type = mimetypes.guess_type(object_key)[0] or "application/octet-stream"
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _normalize_vlm_response(content) -> str:
        if isinstance(content, str):
            normalized = content
        elif isinstance(content, list):
            normalized = "".join(
                item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
                for item in content
            )
        else:
            normalized = str(content or "")

        normalized = normalized.strip()
        if normalized.startswith("```"):
            normalized = re.sub(r"^```[^\n]*\n?", "", normalized)
            normalized = re.sub(r"\n?```$", "", normalized).strip()
        return normalized

    @staticmethod
    def _format_image_explanation(explanation: str, language: str) -> str:
        label = IMAGE_EXPLANATION_PREFIX if language.lower().startswith("ch") else IMAGE_EXPLANATION_PREFIX_EN
        compact = " ".join(explanation.split())
        return f"> {label}{compact}"

    @staticmethod
    def _build_chunk(
        *,
        kb_id: str,
        document_id: str,
        chunk_index: int,
        asset: ImageAsset,
        explanation: str,
    ) -> ImageExplanationChunk:
        chunk_id = f"img_chunk_{uuid.uuid5(uuid.NAMESPACE_URL, f'{document_id}:{asset.object_key}:{chunk_index}').hex}"
        page_content = f"图片说明：{explanation}\n\n图片地址：{asset.image_url}"
        return ImageExplanationChunk(
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            text=explanation,
            page_content=page_content,
            image_url=asset.image_url,
            image_object_key=asset.object_key,
            metadata={
                "kb_id": kb_id,
                "document_id": document_id,
                "chunk_type": "image_explanation",
                "image_object_key": asset.object_key,
                "image_url": asset.image_url,
            },
        )
