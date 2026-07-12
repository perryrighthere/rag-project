from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from rag_project.chat import OpenAICompatibleChatClient
from rag_project.qa import QAOrchestratorName, create_qa_orchestrator
from rag_project.retrieval import KnowledgeBaseRetriever


class QAState(TypedDict, total=False):
    query: str
    kb_id: str
    filters: dict[str, Any]
    top_k: int
    top_n: int
    orchestrator: QAOrchestratorName
    include_agent_trace: bool
    filter_expr: str
    candidates: list[Document]
    reranked: list[Document]
    context: str
    answer: str
    citations: list[dict[str, Any]]
    agent_trace: list[dict[str, Any]]
    review_notes: str | None
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
    orchestrator: QAOrchestratorName = "single"
    agent_trace: list[dict[str, Any]] = field(default_factory=list)
    review_notes: str | None = None


class QAGraph:
    """Minimal LangGraph QA workflow for metadata-filtered retrieval and answer generation."""

    def __init__(
        self,
        *,
        retriever: KnowledgeBaseRetriever,
        chat_client: OpenAICompatibleChatClient,
        default_orchestrator: QAOrchestratorName = "single",
        agent_max_rounds: int = 3,
    ):
        self.retriever = retriever
        self.chat_client = chat_client
        self.default_orchestrator = default_orchestrator
        self.agent_max_rounds = max(1, agent_max_rounds)
        self._app = self._build_graph().compile()

    async def run(
        self,
        *,
        kb_id: str,
        query: str,
        filters: dict[str, Any],
        top_k: int,
        top_n: int | None = None,
        orchestrator: QAOrchestratorName | None = None,
        include_agent_trace: bool = False,
    ) -> QAResult:
        limit = min(top_n or top_k, top_k)
        selected_orchestrator = orchestrator or self.default_orchestrator
        state = await self._app.ainvoke(
            {
                "kb_id": kb_id,
                "query": query,
                "filters": filters,
                "top_k": top_k,
                "top_n": limit,
                "orchestrator": selected_orchestrator,
                "include_agent_trace": include_agent_trace,
            }
        )
        return QAResult(
            query=query,
            answer=state.get("answer") or "",
            filter_expr=state.get("filter_expr") or "",
            citations=list(state.get("citations") or []),
            reranked_documents=list(state.get("reranked") or []),
            rerank_error=state.get("rerank_error"),
            orchestrator=selected_orchestrator,
            agent_trace=list(state.get("agent_trace") or []),
            review_notes=state.get("review_notes"),
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
        orchestrator = create_qa_orchestrator(
            name=state.get("orchestrator") or self.default_orchestrator,
            chat_client=self.chat_client,
            max_rounds=self.agent_max_rounds,
        )
        result = await orchestrator.answer(
            query=state["query"],
            documents=list(state.get("reranked") or []),
            citations=list(state.get("citations") or []),
        )
        return {
            "answer": result.answer,
            "review_notes": result.review_notes,
            "agent_trace": [step.to_dict() for step in result.agent_trace] if state.get("include_agent_trace") else [],
        }

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
