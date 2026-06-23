from typing import Protocol

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda


class Reranker(Protocol):
    async def rerank(self, query: str, documents: list[Document], top_n: int) -> list[Document]:
        ...


class NoopReranker:
    """Fallback reranker used when no provider is configured."""

    async def rerank(self, query: str, documents: list[Document], top_n: int) -> list[Document]:
        return documents[:top_n]

    def as_runnable(self) -> Runnable:
        async def run(payload: dict) -> list[Document]:
            documents = list(payload.get("documents") or [])
            top_n = int(payload.get("top_n") or len(documents))
            return await self.rerank(str(payload.get("query") or ""), documents, top_n)

        return RunnableLambda(run)
