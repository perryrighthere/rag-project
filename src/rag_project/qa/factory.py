from __future__ import annotations

from rag_project.chat import OpenAICompatibleChatClient
from rag_project.qa.autogen import AutoGenQAOrchestrator
from rag_project.qa.base import QAOrchestrator, QAOrchestratorName, build_chat_config
from rag_project.qa.crewai import CrewAIQAOrchestrator
from rag_project.qa.langgraph_multi import LangGraphMultiQAOrchestrator
from rag_project.qa.single import SingleQAOrchestrator


def create_qa_orchestrator(
    *,
    name: QAOrchestratorName,
    chat_client: OpenAICompatibleChatClient,
    max_rounds: int,
) -> QAOrchestrator:
    if name == "single":
        return SingleQAOrchestrator(chat_client)
    if name == "langgraph_multi":
        return LangGraphMultiQAOrchestrator(chat_client=chat_client, max_rounds=max_rounds)
    if name == "crewai":
        return CrewAIQAOrchestrator(chat_config=build_chat_config(chat_client), max_rounds=max_rounds)
    if name == "autogen":
        return AutoGenQAOrchestrator(chat_config=build_chat_config(chat_client), max_rounds=max_rounds)
    raise ValueError(f"Unsupported QA orchestrator: {name}")
