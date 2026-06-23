from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class ChatConfig:
    base_url: str | None
    api_key: str
    model: str
    timeout: float = 60.0
    temperature: float = 0.2
    max_tokens: int = 1200


class OpenAICompatibleChatClient:
    """OpenAI-compatible chat facade used by the QA graph."""

    def __init__(self, config: ChatConfig):
        if not config.model:
            raise ValueError("chat model is required")
        self.config = config
        self._chat = ChatOpenAI(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    async def generate_answer(self, *, query: str, documents: list[Document]) -> str:
        prompt = _build_prompt(query, documents)
        response = await self._chat.ainvoke([HumanMessage(content=prompt)])
        return _normalize_content(response.content)


def _build_prompt(query: str, documents: list[Document]) -> str:
    context = "\n\n".join(_format_context_item(index, document) for index, document in enumerate(documents, start=1))
    return (
        "你是一个严谨的 RAG 问答助手。请只根据给定上下文回答问题。"
        "如果上下文不足，请明确说明无法从当前知识库确认。"
        "回答中引用依据时使用 chunk_id。\n\n"
        f"问题：{query}\n\n"
        f"上下文：\n{context or '无可用上下文'}\n\n"
        "请给出简洁中文答案。"
    )


def _format_context_item(index: int, document: Document) -> str:
    metadata = document.metadata
    chunk_id = metadata.get("chunk_id", "")
    source_uri = metadata.get("source_uri", "")
    heading_path = metadata.get("heading_path", "")
    return (
        f"[{index}] chunk_id={chunk_id} source_uri={source_uri} heading_path={heading_path}\n"
        f"{document.page_content}"
    )


def _normalize_content(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
            for item in content
        ).strip()
    return str(content or "").strip()
