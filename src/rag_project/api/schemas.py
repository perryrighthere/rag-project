from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from rag_project.chunking import ChunkRecord, ChunkingConfig
from rag_project.knowledge_base import MetadataSchema
from rag_project.parsers import ParsedDocument


DocumentStatus = Literal["uploaded", "parsing", "parsed", "chunking", "embedding", "indexed", "failed", "deleted"]
TaskStatus = Literal["pending", "running", "succeeded", "failed"]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class KnowledgeBaseCreate(BaseModel):
    name: str
    description: str | None = None
    metadata_schema: MetadataSchema = Field(default_factory=MetadataSchema)
    chunking_config: ChunkingConfig = Field(default_factory=ChunkingConfig)


class KnowledgeBaseUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    chunking_config: ChunkingConfig | None = None


class KnowledgeBaseRecord(BaseModel):
    kb_id: str = Field(default_factory=lambda: new_id("kb"))
    name: str
    description: str | None = None
    metadata_schema: MetadataSchema = Field(default_factory=MetadataSchema)
    chunking_config: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding_model: str | None = None
    embedding_dim: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentRecord(BaseModel):
    document_id: str = Field(default_factory=lambda: new_id("doc"))
    kb_id: str
    filename: str
    content_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: DocumentStatus = "uploaded"
    raw_object_key: str | None = None
    parsed_document: ParsedDocument | None = None
    chunk_count: int = 0
    embedding_model: str | None = None
    embedding_dim: int | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    file_content: bytes | None = Field(default=None, exclude=True)


class TaskRecord(BaseModel):
    task_id: str = Field(default_factory=lambda: new_id("task"))
    task_type: str
    status: TaskStatus = "pending"
    document_id: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RetrievalSearchRequest(BaseModel):
    kb_id: str
    query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=10, ge=1, le=100)
    top_n: int | None = Field(default=None, ge=1, le=100)


class RetrievalMatch(BaseModel):
    chunk_id: str
    document_id: str
    score: float | None = None
    rerank_score: float | None = None
    text: str
    source_uri: str | None = None
    heading_path: str = ""
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalSearchResponse(BaseModel):
    query: str
    filter_expr: str
    matches: list[RetrievalMatch] = Field(default_factory=list)
    rerank_error: str | None = None


class DocumentChunksResponse(BaseModel):
    document_id: str
    chunks: list[ChunkRecord]


class ChatRequest(BaseModel):
    kb_id: str
    query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=10, ge=1, le=100)
    top_n: int | None = Field(default=None, ge=1, le=100)


class ChatCitation(BaseModel):
    chunk_id: str | None = None
    document_id: str | None = None
    source_uri: str | None = None
    heading_path: str = ""
    page_start: int | None = None
    page_end: int | None = None
    score: float | None = None
    rerank_score: float | None = None


class ChatResponse(BaseModel):
    query: str
    answer: str
    filter_expr: str
    citations: list[ChatCitation] = Field(default_factory=list)
    matches: list[RetrievalMatch] = Field(default_factory=list)
    rerank_error: str | None = None
