from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class UploadedFile:
    filename: str
    content: bytes
    content_type: str | None = None


class ParseOptions(BaseModel):
    model_config = ConfigDict(extra="allow")

    kb_id: str
    document_id: str
    backend: str = "hybrid-auto-engine"
    parse_method: str = "auto"
    lang_list: list[str] = Field(default_factory=lambda: ["ch"])
    formula_enable: bool = True
    table_enable: bool = True
    return_md: bool = True
    return_middle_json: bool = True
    return_model_output: bool = False
    return_content_list: bool = True
    return_images: bool = True
    response_format_zip: bool = True
    return_original_file: bool = False


class ImageExplanationChunk(BaseModel):
    chunk_id: str
    chunk_index: int
    text: str
    page_content: str
    image_url: str
    image_object_key: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    document_id: str
    parser: str
    parser_task_id: str
    markdown_text: str
    markdown_object_key: str
    content_list_object_key: str | None = None
    middle_json_object_key: str | None = None
    image_object_keys: list[str] = Field(default_factory=list)
    image_explanation_chunks: list[ImageExplanationChunk] = Field(default_factory=list)
    raw_object_key: str
    parse_options: dict[str, Any]
    created_at: datetime


class DocumentParser(Protocol):
    async def parse(self, file: UploadedFile, options: ParseOptions) -> ParsedDocument:
        ...
