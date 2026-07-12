from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from rag_project.chat import OpenAICompatibleChatClient
from rag_project.qa.base import (
    AgentStep,
    AgenticQAResult,
    QAOrchestratorExecutionError,
    QAOrchestratorName,
    format_citations,
    format_documents,
)


class LangGraphMultiState(TypedDict, total=False):
    query: str
    documents: list[Document]
    citations: list[dict[str, Any]]
    context: str
    citation_context: str
    query_analysis: str
    evidence_map: str
    draft_answer: str
    review_notes: str
    answer: str
    agent_trace: list[AgentStep]


class LangGraphMultiQAOrchestrator:
    name: QAOrchestratorName = "langgraph_multi"

    def __init__(self, *, chat_client: OpenAICompatibleChatClient, max_rounds: int = 3):
        self.chat_client = chat_client
        self.max_rounds = max(1, max_rounds)
        self._app = self._build_graph().compile()

    async def answer(
        self,
        *,
        query: str,
        documents: list[Document],
        citations: list[dict[str, Any]],
    ) -> AgenticQAResult:
        if not documents:
            return AgenticQAResult(
                answer="当前知识库无法确认。",
                orchestrator=self.name,
                review_notes="未检索到可用上下文，无法生成有依据的答案。",
                agent_trace=[
                    AgentStep(role="Citation Reviewer", content="未检索到可用上下文，答案必须说明无法确认。")
                ],
            )
        try:
            state = await self._app.ainvoke(
                {
                    "query": query,
                    "documents": documents,
                    "citations": citations,
                    "context": format_documents(documents),
                    "citation_context": format_citations(citations),
                    "agent_trace": [],
                }
            )
        except Exception as exc:
            raise QAOrchestratorExecutionError(f"langgraph_multi orchestration failed: {exc}") from exc
        return AgenticQAResult(
            answer=str(state.get("answer") or "").strip(),
            orchestrator=self.name,
            review_notes=str(state.get("review_notes") or "").strip() or None,
            agent_trace=list(state.get("agent_trace") or []),
        )

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(LangGraphMultiState)
        graph.add_node("query_analyst", self._query_analyst)
        graph.add_node("evidence_mapper", self._evidence_mapper)
        graph.add_node("answer_writer", self._answer_writer)
        graph.add_node("citation_reviewer", self._citation_reviewer)

        graph.set_entry_point("query_analyst")
        graph.add_edge("query_analyst", "evidence_mapper")
        graph.add_edge("evidence_mapper", "answer_writer")
        graph.add_edge("answer_writer", "citation_reviewer")
        graph.add_edge("citation_reviewer", END)
        return graph

    async def _query_analyst(self, state: LangGraphMultiState) -> dict[str, Any]:
        content = await self.chat_client.generate_text(
            "你是 Query Analyst。请只分析用户问题需要从上下文中确认什么，不要回答问题。\n\n"
            f"问题：{state['query']}\n\n"
            "请用中文输出 1-3 条检索意图。"
        )
        return {"query_analysis": content, "agent_trace": _append_step(state, "Query Analyst", content)}

    async def _evidence_mapper(self, state: LangGraphMultiState) -> dict[str, Any]:
        content = await self.chat_client.generate_text(
            "你是 Evidence Mapper。请只根据给定上下文整理可支持回答的证据。"
            "每条证据必须包含已有 chunk_id；不要引入外部知识。\n\n"
            f"问题：{state['query']}\n\n"
            f"问题分析：{state.get('query_analysis', '')}\n\n"
            f"上下文：\n{state['context']}\n\n"
            f"可用引用：\n{state['citation_context']}\n\n"
            "请用中文输出证据清单。"
        )
        return {"evidence_map": content, "agent_trace": _append_step(state, "Evidence Mapper", content)}

    async def _answer_writer(self, state: LangGraphMultiState) -> dict[str, Any]:
        content = await self.chat_client.generate_text(
            "你是 Answer Writer。请只根据 Evidence Mapper 给出的证据回答。"
            "答案必须是中文，引用依据时使用已有 chunk_id。"
            "如果证据不足，必须明确说“当前知识库无法确认”。\n\n"
            f"问题：{state['query']}\n\n"
            f"证据：\n{state.get('evidence_map', '')}\n\n"
            "请输出简洁答案。"
        )
        return {"draft_answer": content, "agent_trace": _append_step(state, "Answer Writer", content)}

    async def _citation_reviewer(self, state: LangGraphMultiState) -> dict[str, Any]:
        content = await self.chat_client.generate_text(
            "你是 Citation Reviewer。请只基于给定上下文和可用引用审查草稿答案是否越界。"
            "不能引入新事实；如果答案缺少依据，请指出并给出修正后的最终答案。"
            "输出必须包含两段：FINAL_ANSWER: 和 REVIEW_NOTES:。\n\n"
            f"问题：{state['query']}\n\n"
            f"上下文：\n{state['context']}\n\n"
            f"可用引用：\n{state['citation_context']}\n\n"
            f"草稿答案：\n{state.get('draft_answer', '')}\n"
        )
        answer, review_notes = _split_review(content, fallback_answer=str(state.get("draft_answer") or ""))
        return {
            "answer": answer,
            "review_notes": review_notes,
            "agent_trace": _append_step(state, "Citation Reviewer", content),
        }


def _append_step(state: LangGraphMultiState, role: str, content: str) -> list[AgentStep]:
    return [*list(state.get("agent_trace") or []), AgentStep(role=role, content=content.strip())]


def _split_review(content: str, *, fallback_answer: str) -> tuple[str, str]:
    text = content.strip()
    if "FINAL_ANSWER:" not in text:
        return fallback_answer.strip(), text
    _, tail = text.split("FINAL_ANSWER:", 1)
    if "REVIEW_NOTES:" not in tail:
        return tail.strip(), ""
    answer, review_notes = tail.split("REVIEW_NOTES:", 1)
    return answer.strip(), review_notes.strip()
