from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from langchain_core.documents import Document

from rag_project.chat import ChatConfig, OpenAICompatibleChatClient


QAOrchestratorName = Literal["single", "langgraph_multi", "crewai", "autogen"]


@dataclass(frozen=True)
class AgentStep:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class AgenticQAResult:
    answer: str
    orchestrator: QAOrchestratorName
    review_notes: str | None = None
    agent_trace: list[AgentStep] = field(default_factory=list)


class QAOrchestrator(Protocol):
    name: QAOrchestratorName

    async def answer(
        self,
        *,
        query: str,
        documents: list[Document],
        citations: list[dict[str, Any]],
    ) -> AgenticQAResult:
        ...


class QAOrchestratorError(RuntimeError):
    """Base error for agentic QA orchestration failures."""


class QAOrchestratorDependencyError(QAOrchestratorError):
    """Raised when an optional multi-agent framework is not installed."""


class QAOrchestratorExecutionError(QAOrchestratorError):
    """Raised when an orchestrator fails during execution."""


def format_documents(documents: list[Document]) -> str:
    if not documents:
        return "无可用上下文"
    return "\n\n".join(_format_document(index, document) for index, document in enumerate(documents, start=1))


def format_citations(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return "无可用引用"
    lines = []
    for index, citation in enumerate(citations, start=1):
        lines.append(
            " ".join(
                [
                    f"[{index}]",
                    f"chunk_id={citation.get('chunk_id')}",
                    f"document_id={citation.get('document_id')}",
                    f"source_uri={citation.get('source_uri')}",
                    f"heading_path={citation.get('heading_path') or ''}",
                ]
            )
        )
    return "\n".join(lines)


def split_answer_and_review(content: str) -> tuple[str, str | None]:
    text = content.strip()
    if "REVIEW_NOTES:" not in text:
        return _strip_answer_marker(text), None
    answer_part, review_part = text.split("REVIEW_NOTES:", 1)
    return _strip_answer_marker(answer_part), review_part.strip() or None


def _strip_answer_marker(text: str) -> str:
    return text.replace("FINAL_ANSWER:", "", 1).strip()


def _format_document(index: int, document: Document) -> str:
    metadata = document.metadata
    return (
        f"[{index}] chunk_id={metadata.get('chunk_id', '')} "
        f"source_uri={metadata.get('source_uri', '')} "
        f"heading_path={metadata.get('heading_path', '')}\n"
        f"{document.page_content}"
    )


def build_chat_config(chat_client: OpenAICompatibleChatClient) -> ChatConfig:
    return chat_client.config
