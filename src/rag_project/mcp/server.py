from __future__ import annotations

from typing import Any, Callable

from langchain_core.documents import Document
from mcp.server.fastmcp import FastMCP

from rag_project.graphs import QAGraph
from rag_project.retrieval import KnowledgeBaseRetriever
from rag_project.services.store import Store

RetrieverFactory = Callable[[], KnowledgeBaseRetriever]
QAGraphFactory = Callable[[], QAGraph]
StoreFactory = Callable[[], Store]


def build_mcp_server(
    *,
    retriever_factory: RetrieverFactory,
    qa_graph_factory: QAGraphFactory,
    store_factory: StoreFactory,
    name: str = "rag-project",
) -> FastMCP:
    """Build the RAG-project MCP server exposing retrieval and QA tools."""
    mcp = FastMCP(
        name,
        instructions=(
            "RAG 知识库检索与问答服务。可通过 retrieval_search 进行元数据过滤的"
            "向量检索，或通过 qa_chat 让知识库驱动模型回答带引用的中文问题。"
            "调用前请先用 list_knowledge_bases 确认可用的知识库 ID。"
        ),
        streamable_http_path="/",
    )

    @mcp.tool()
    async def list_knowledge_bases() -> list[dict[str, Any]]:
        """List all knowledge bases with their ids and names."""
        return [
            {
                "kb_id": kb.kb_id,
                "name": kb.name,
                "description": kb.description,
                "embedding_model": kb.embedding_model,
            }
            for kb in store_factory().list_knowledge_bases()
        ]

    @mcp.tool()
    async def retrieval_search(
        kb_id: str,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        top_n: int | None = None,
    ) -> dict[str, Any]:
        """Search chunks in a knowledge base using metadata-filtered vector retrieval.

        Args:
            kb_id: Knowledge base id.
            query: Natural language search query.
            filters: Optional structured metadata filters validated against the kb schema.
            top_k: Number of candidate chunks to retrieve.
            top_n: Number of chunks to return after reranking (defaults to top_k).
        """
        result = await retriever_factory().search(
            kb_id=kb_id,
            query=query,
            filters=filters or {},
            top_k=top_k,
            top_n=top_n,
        )
        return {
            "query": result.query,
            "filter_expr": result.filter_expr,
            "rerank_error": result.rerank_error,
            "matches": [
                {
                    "chunk_id": match.chunk_id,
                    "document_id": match.document_id,
                    "score": match.score,
                    "rerank_score": match.rerank_score,
                    "text": match.text,
                    "source_uri": match.source_uri,
                    "heading_path": match.heading_path,
                    "page_start": match.page_start,
                    "page_end": match.page_end,
                    "metadata": match.metadata,
                }
                for match in result.matches
            ],
        }

    @mcp.tool()
    async def qa_chat(
        kb_id: str,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        top_n: int | None = None,
        orchestrator: str | None = None,
    ) -> dict[str, Any]:
        """Answer a question against a knowledge base with cited sources.

        Args:
            kb_id: Knowledge base id.
            query: Natural language question.
            filters: Optional structured metadata filters validated against the kb schema.
            top_k: Number of candidate chunks to retrieve.
            top_n: Number of chunks to rerank and pass as context.
            orchestrator: Optional QA orchestrator (single, langgraph_multi, crewai, autogen).
        """
        graph = qa_graph_factory()
        result = await graph.run(
            kb_id=kb_id,
            query=query,
            filters=filters or {},
            top_k=top_k,
            top_n=top_n,
            orchestrator=_parse_orchestrator(orchestrator),
        )
        return {
            "query": result.query,
            "answer": result.answer,
            "filter_expr": result.filter_expr,
            "orchestrator": result.orchestrator,
            "rerank_error": result.rerank_error,
            "review_notes": result.review_notes,
            "citations": [
                {
                    "chunk_id": citation.get("chunk_id"),
                    "document_id": citation.get("document_id"),
                    "source_uri": citation.get("source_uri"),
                    "heading_path": citation.get("heading_path") or "",
                }
                for citation in result.citations
            ],
            "matches": [_document_to_match(document) for document in result.reranked_documents],
        }

    @mcp.tool()
    async def get_task_status(task_id: str) -> dict[str, Any]:
        """Get the status of a parse, index, or ingest task."""
        task = store_factory().get_task(task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        return task.model_dump(mode="json", exclude_none=False)

    return mcp


def _parse_orchestrator(value: str | None) -> Any:
    if value is None:
        return None
    return value


def _document_to_match(document: Document) -> dict[str, Any]:
    metadata = document.metadata
    return {
        "chunk_id": str(metadata.get("chunk_id") or ""),
        "document_id": str(metadata.get("document_id") or ""),
        "score": metadata.get("score"),
        "rerank_score": metadata.get("rerank_score"),
        "text": document.page_content,
        "source_uri": metadata.get("source_uri"),
        "heading_path": str(metadata.get("heading_path") or ""),
        "page_start": metadata.get("page_start"),
        "page_end": metadata.get("page_end"),
        "metadata": dict(metadata.get("_source_metadata") or {}),
    }
