from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def new_chunk_id() -> str:
    return f"chunk_{uuid4().hex}"


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(default=800, ge=100, le=8000)
    chunk_overlap: int = Field(default=120, ge=0, le=2000)
    separators: list[str] = Field(
        default_factory=lambda: ["\n## ", "\n### ", "\n\n", "\n", "。", "，", " "]
    )

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if not self.separators:
            raise ValueError("separators cannot be empty")
        return self


class ChunkRecord(BaseModel):
    chunk_id: str = Field(default_factory=new_chunk_id)
    kb_id: str
    document_id: str
    chunk_index: int
    text: str
    heading_path: str = ""
    page_start: int | None = None
    page_end: int | None = None
    source_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str | None = None
    embedding_dim: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
