from __future__ import annotations

import inspect
from typing import Any

from langchain_core.documents import Document

from rag_project.chat import ChatConfig
from rag_project.qa.base import (
    AgentStep,
    AgenticQAResult,
    QAOrchestratorDependencyError,
    QAOrchestratorExecutionError,
    QAOrchestratorName,
    format_citations,
    format_documents,
    split_answer_and_review,
)


class AutoGenQAOrchestrator:
    name: QAOrchestratorName = "autogen"

    def __init__(self, *, chat_config: ChatConfig, max_rounds: int = 3):
        self.chat_config = chat_config
        self.max_rounds = max(1, max_rounds)

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
            )
        try:
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
            from autogen_agentchat.teams import RoundRobinGroupChat
            from autogen_ext.models.openai import OpenAIChatCompletionClient
        except ImportError as exc:
            raise QAOrchestratorDependencyError(
                "AutoGen is not installed. Install optional dependencies with requirements-agentic.txt."
            ) from exc

        model_client = OpenAIChatCompletionClient(
            model=self.chat_config.model,
            api_key=self.chat_config.api_key,
            base_url=self.chat_config.base_url,
        )
        try:
            agents = [
                AssistantAgent(
                    "query_analyst",
                    model_client=model_client,
                    system_message="你是 Query Analyst。只分析问题需要从上下文确认什么，不回答问题。",
                ),
                AssistantAgent(
                    "evidence_mapper",
                    model_client=model_client,
                    system_message="你是 Evidence Mapper。只使用传入上下文整理带 chunk_id 的证据。",
                ),
                AssistantAgent(
                    "answer_writer",
                    model_client=model_client,
                    system_message=(
                        "你是 Answer Writer。只基于证据生成中文答案，引用已有 chunk_id；"
                        "证据不足时必须说“当前知识库无法确认”。"
                    ),
                ),
                AssistantAgent(
                    "citation_reviewer",
                    model_client=model_client,
                    system_message=(
                        "你是 Citation Reviewer。只基于传入上下文审查答案。"
                        "最终输出必须包含 FINAL_ANSWER: 和 REVIEW_NOTES:。"
                    ),
                ),
            ]
            termination = TextMentionTermination("REVIEW_NOTES:") | MaxMessageTermination(
                max_messages=max(4, self.max_rounds * len(agents))
            )
            team = RoundRobinGroupChat(agents, termination_condition=termination)
            result = await team.run(
                task=(
                    "请完成基于 RAG 上下文的多 Agent 问答。不能访问外部知识，不能重新检索。\n\n"
                    f"问题：{query}\n\n"
                    f"上下文：\n{format_documents(documents)}\n\n"
                    f"可用引用：\n{format_citations(citations)}"
                )
            )
        except Exception as exc:
            raise QAOrchestratorExecutionError(f"AutoGen orchestration failed: {exc}") from exc
        finally:
            close = getattr(model_client, "close", None)
            if close is not None:
                close_result = close()
                if inspect.isawaitable(close_result):
                    await close_result

        messages = list(getattr(result, "messages", []) or [])
        trace = [
            AgentStep(
                role=str(getattr(message, "source", getattr(message, "role", "AutoGen"))),
                content=str(getattr(message, "content", message)).strip(),
            )
            for message in messages
        ]
        final_content = trace[-1].content if trace else str(result)
        answer, review_notes = split_answer_and_review(final_content)
        return AgenticQAResult(
            answer=answer,
            orchestrator=self.name,
            review_notes=review_notes,
            agent_trace=trace or [AgentStep(role="AutoGen", content=str(result).strip())],
        )
