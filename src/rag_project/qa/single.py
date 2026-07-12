from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from rag_project.chat import OpenAICompatibleChatClient
from rag_project.qa.base import AgenticQAResult, QAOrchestratorName


class SingleQAOrchestrator:
    name: QAOrchestratorName = "single"

    def __init__(self, chat_client: OpenAICompatibleChatClient):
        self.chat_client = chat_client

    async def answer(
        self,
        *,
        query: str,
        documents: list[Document],
        citations: list[dict[str, Any]],
    ) -> AgenticQAResult:
        answer = await self.chat_client.generate_answer(query=query, documents=documents)
        return AgenticQAResult(answer=answer, orchestrator=self.name)
