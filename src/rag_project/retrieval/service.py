from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.documents import Document

from rag_project.embeddings import OpenAICompatibleEmbeddingClient
from rag_project.knowledge_base import MetadataSchema
from rag_project.rerankers import Reranker
from rag_project.retrieval.filters import MilvusFilterBuilder
from rag_project.services.memory_store import InMemoryStore
from rag_project.vectorstores import MilvusSearchMatch, MilvusVectorStoreAdapter


EmbeddingClientFactory = Callable[[], OpenAICompatibleEmbeddingClient]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    score: float | None
    rerank_score: float | None
    text: str
    source_uri: str | None
    heading_path: str
    page_start: int | None
    page_end: int | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    filter_expr: str
    matches: list[RetrievedChunk]
    rerank_error: str | None = None


class KnowledgeBaseRetriever:
    """Coordinates schema-safe filter building, vector search, and rerank."""

    def __init__(
        self,
        *,
        store: InMemoryStore,
        embedding_client_factory: EmbeddingClientFactory,
        vector_store: MilvusVectorStoreAdapter,
        reranker: Reranker,
        filter_builder: MilvusFilterBuilder | None = None,
    ):
        self.store = store
        self.embedding_client_factory = embedding_client_factory
        self.vector_store = vector_store
        self.reranker = reranker
        self.filter_builder = filter_builder or MilvusFilterBuilder()

    def metadata_schema_for(self, kb_id: str) -> MetadataSchema:
        knowledge_base = self.store.knowledge_bases.get(kb_id)
        if knowledge_base is None:
            raise KeyError(f"Knowledge base not found: {kb_id}")
        return knowledge_base.metadata_schema

    def build_filter_expr(self, *, kb_id: str, filters: dict[str, Any]) -> str:
        return self.filter_builder.build(
            kb_id=kb_id,
            metadata_schema=self.metadata_schema_for(kb_id),
            filters=filters,
        )

    async def retrieve_candidates(
        self,
        *,
        query: str,
        filter_expr: str,
        top_k: int,
    ) -> list[Document]:
        query_vector = await self.embedding_client_factory().embed_query(query)
        matches = await self.vector_store.search(
            query_vector=query_vector,
            filter_expr=filter_expr,
            top_k=top_k,
        )
        return [_search_match_to_document(match) for match in matches]

    async def rerank_documents(
        self,
        *,
        query: str,
        documents: list[Document],
        top_n: int,
    ) -> tuple[list[Document], str | None]:
        try:
            return await self.reranker.rerank(query, documents, top_n), None
        except Exception as exc:
            return documents[:top_n], str(exc)

    async def search(
        self,
        *,
        kb_id: str,
        query: str,
        filters: dict[str, Any],
        top_k: int,
        top_n: int | None = None,
    ) -> RetrievalResult:
        limit = min(top_n or top_k, top_k)
        filter_expr = self.build_filter_expr(kb_id=kb_id, filters=filters)
        candidates = await self.retrieve_candidates(query=query, filter_expr=filter_expr, top_k=top_k)
        reranked, rerank_error = await self.rerank_documents(query=query, documents=candidates, top_n=limit)
        return RetrievalResult(
            query=query,
            filter_expr=filter_expr,
            matches=[_document_to_retrieved_chunk(document) for document in reranked],
            rerank_error=rerank_error,
        )


def _search_match_to_document(match: MilvusSearchMatch) -> Document:
    source_metadata = dict(match.metadata)
    metadata = {
        **source_metadata,
        "_source_metadata": source_metadata,
        "chunk_id": match.chunk_id,
        "document_id": match.document_id,
        "score": match.score,
        "source_uri": match.source_uri,
        "heading_path": match.heading_path,
        "page_start": match.page_start,
        "page_end": match.page_end,
    }
    return Document(page_content=match.text, metadata=metadata)


def _document_to_retrieved_chunk(document: Document) -> RetrievedChunk:
    metadata = document.metadata
    return RetrievedChunk(
        chunk_id=str(metadata.get("chunk_id") or ""),
        document_id=str(metadata.get("document_id") or ""),
        score=_optional_float(metadata.get("score")),
        rerank_score=_optional_float(metadata.get("rerank_score")),
        text=document.page_content,
        source_uri=metadata.get("source_uri"),
        heading_path=str(metadata.get("heading_path") or ""),
        page_start=metadata.get("page_start"),
        page_end=metadata.get("page_end"),
        metadata=dict(metadata.get("_source_metadata") or {}),
    )


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
