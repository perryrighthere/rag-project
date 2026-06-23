from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from rag_project.chat import OpenAICompatibleChatClient
from rag_project.retrieval import KnowledgeBaseRetriever


class QAState(TypedDict, total=False):
    query: str
    kb_id: str
    filters: dict[str, Any]
    top_k: int
    top_n: int
    filter_expr: str
    candidates: list[Document]
    reranked: list[Document]
    context: str
    answer: str
    citations: list[dict[str, Any]]
    rerank_error: str | None
    error: str | None


@dataclass(frozen=True)
class QAResult:
    query: str
    answer: str
    filter_expr: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    reranked_documents: list[Document] = field(default_factory=list)
    rerank_error: str | None = None


class QAGraph:
    """Minimal LangGraph QA workflow for metadata-filtered retrieval and answer generation."""

    def __init__(self, *, retriever: KnowledgeBaseRetriever, chat_client: OpenAICompatibleChatClient):
        self.retriever = retriever
        self.chat_client = chat_client
        self._app = self._build_graph().compile()

    async def run(
        self,
        *,
        kb_id: str,
        query: str,
        filters: dict[str, Any],
        top_k: int,
        top_n: int | None = None,
    ) -> QAResult:
        limit = min(top_n or top_k, top_k)
        state = await self._app.ainvoke(
            {
                "kb_id": kb_id,
                "query": query,
                "filters": filters,
                "top_k": top_k,
                "top_n": limit,
            }
        )
        return QAResult(
            query=query,
            answer=state.get("answer") or "",
            filter_expr=state.get("filter_expr") or "",
            citations=list(state.get("citations") or []),
            reranked_documents=list(state.get("reranked") or []),
            rerank_error=state.get("rerank_error"),
        )

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(QAState)
        graph.add_node("receive_query", self._receive_query)
        graph.add_node("build_metadata_filter", self._build_metadata_filter)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("rerank", self._rerank)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("return_answer", self._return_answer)

        graph.set_entry_point("receive_query")
        graph.add_edge("receive_query", "build_metadata_filter")
        graph.add_edge("build_metadata_filter", "retrieve")
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "generate_answer")
        graph.add_edge("generate_answer", "return_answer")
        graph.add_edge("return_answer", END)
        return graph

    async def _receive_query(self, state: QAState) -> dict[str, Any]:
        return {
            "query": state["query"].strip(),
            "filters": state.get("filters") or {},
        }

    async def _build_metadata_filter(self, state: QAState) -> dict[str, Any]:
        return {
            "filter_expr": self.retriever.build_filter_expr(
                kb_id=state["kb_id"],
                filters=state.get("filters") or {},
            )
        }

    async def _retrieve(self, state: QAState) -> dict[str, Any]:
        candidates = await self.retriever.retrieve_candidates(
            query=state["query"],
            filter_expr=state["filter_expr"],
            top_k=int(state["top_k"]),
        )
        return {"candidates": candidates}

    async def _rerank(self, state: QAState) -> dict[str, Any]:
        reranked, rerank_error = await self.retriever.rerank_documents(
            query=state["query"],
            documents=list(state.get("candidates") or []),
            top_n=int(state["top_n"]),
        )
        return {
            "reranked": reranked,
            "rerank_error": rerank_error,
            "citations": [_citation(document) for document in reranked],
        }

    async def _generate_answer(self, state: QAState) -> dict[str, Any]:
        answer = await self.chat_client.generate_answer(
            query=state["query"],
            documents=list(state.get("reranked") or []),
        )
        return {"answer": answer}

    async def _return_answer(self, state: QAState) -> dict[str, Any]:
        return state


def _citation(document: Document) -> dict[str, Any]:
    metadata = document.metadata
    return {
        "chunk_id": metadata.get("chunk_id"),
        "document_id": metadata.get("document_id"),
        "source_uri": metadata.get("source_uri"),
        "heading_path": metadata.get("heading_path") or "",
        "page_start": metadata.get("page_start"),
        "page_end": metadata.get("page_end"),
        "score": metadata.get("score"),
        "rerank_score": metadata.get("rerank_score"),
    }
