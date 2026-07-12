from __future__ import annotations

import asyncio
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


class CrewAIQAOrchestrator:
    name: QAOrchestratorName = "crewai"

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
            from crewai import Agent, Crew, LLM, Process, Task
        except ImportError as exc:
            raise QAOrchestratorDependencyError(
                "CrewAI is not installed. Install optional dependencies with requirements-agentic.txt."
            ) from exc

        try:
            llm = LLM(
                model=self.chat_config.model,
                api_key=self.chat_config.api_key,
                base_url=self.chat_config.base_url,
                temperature=self.chat_config.temperature,
                max_tokens=self.chat_config.max_tokens,
            )
            analyst = Agent(
                role="Query Analyst",
                goal="分析用户问题需要从上下文确认的要点。",
                backstory="你只做问题分析，不回答问题，也不引入外部知识。",
                llm=llm,
                allow_delegation=False,
                verbose=False,
            )
            mapper = Agent(
                role="Evidence Mapper",
                goal="从传入上下文中整理带 chunk_id 的证据。",
                backstory="你只能使用传入的上下文和引用。",
                llm=llm,
                allow_delegation=False,
                verbose=False,
            )
            writer = Agent(
                role="Answer Writer",
                goal="基于证据生成简洁中文答案。",
                backstory="你必须使用已有 chunk_id 引用依据，证据不足时说明当前知识库无法确认。",
                llm=llm,
                allow_delegation=False,
                verbose=False,
            )
            reviewer = Agent(
                role="Citation Reviewer",
                goal="审查答案是否只基于传入证据，并输出最终答案和审查说明。",
                backstory="你不能引入新事实，只能基于 reranked documents 检查答案。",
                llm=llm,
                allow_delegation=False,
                verbose=False,
            )
            context = format_documents(documents)
            citation_context = format_citations(citations)
            tasks = [
                Task(
                    description=f"分析问题需要确认什么，不要回答。\n问题：{query}",
                    expected_output="1-3 条中文问题分析。",
                    agent=analyst,
                ),
                Task(
                    description=(
                        "整理可支持回答的证据，每条必须包含已有 chunk_id，不得引入外部知识。\n\n"
                        f"问题：{query}\n\n上下文：\n{context}\n\n可用引用：\n{citation_context}"
                    ),
                    expected_output="中文证据清单。",
                    agent=mapper,
                ),
                Task(
                    description=(
                        "基于上一步证据回答问题。答案必须是中文，引用依据时使用已有 chunk_id。"
                        "如果证据不足，必须明确说“当前知识库无法确认”。\n\n"
                        f"问题：{query}"
                    ),
                    expected_output="简洁中文答案。",
                    agent=writer,
                ),
                Task(
                    description=(
                        "审查答案是否只基于上下文和可用引用。输出必须包含 FINAL_ANSWER: 和 REVIEW_NOTES:。\n\n"
                        f"上下文：\n{context}\n\n可用引用：\n{citation_context}"
                    ),
                    expected_output="FINAL_ANSWER: <最终答案>\nREVIEW_NOTES: <审查说明>",
                    agent=reviewer,
                ),
            ]
            crew = Crew(agents=[analyst, mapper, writer, reviewer], tasks=tasks, process=Process.sequential, verbose=False)
            output = await asyncio.to_thread(crew.kickoff)
        except QAOrchestratorDependencyError:
            raise
        except Exception as exc:
            raise QAOrchestratorExecutionError(f"CrewAI orchestration failed: {exc}") from exc

        answer, review_notes = split_answer_and_review(str(output))
        return AgenticQAResult(
            answer=answer,
            orchestrator=self.name,
            review_notes=review_notes,
            agent_trace=[AgentStep(role="CrewAI", content=str(output).strip())],
        )
