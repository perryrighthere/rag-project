from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda


@dataclass(frozen=True)
class RerankConfig:
    base_url: str
    api_key: str
    model: str
    timeout: float = 60.0


class OpenAICompatibleReranker:
    """Small adapter for common OpenAI-compatible rerank endpoints."""

    def __init__(self, config: RerankConfig):
        if not config.base_url:
            raise ValueError("rerank base_url is required")
        if not config.model:
            raise ValueError("rerank model is required")
        self.config = config

    async def rerank(self, query: str, documents: list[Document], top_n: int) -> list[Document]:
        if not documents:
            return []

        payload = {
            "model": self.config.model,
            "query": query,
            "documents": [document.page_content for document in documents],
            "top_n": min(top_n, len(documents)),
            "return_documents": False,
        }
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(self._endpoint(), json=payload, headers=headers)
            response.raise_for_status()
            ranked = self._parse_results(response.json(), documents)

        return ranked[:top_n]

    def as_runnable(self) -> Runnable:
        async def run(payload: dict) -> list[Document]:
            documents = list(payload.get("documents") or [])
            top_n = int(payload.get("top_n") or len(documents))
            return await self.rerank(str(payload.get("query") or ""), documents, top_n)

        return RunnableLambda(run)

    def _endpoint(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/rerank"

    @staticmethod
    def _parse_results(payload: dict[str, Any], documents: list[Document]) -> list[Document]:
        raw_results = payload.get("results") or payload.get("data") or []
        ranked: list[Document] = []
        for item in raw_results:
            index = _coerce_index(item)
            if index is None or index < 0 or index >= len(documents):
                continue
            score = _coerce_score(item)
            metadata = dict(documents[index].metadata)
            if score is not None:
                metadata["rerank_score"] = score
            ranked.append(Document(page_content=documents[index].page_content, metadata=metadata))
        if not ranked:
            raise ValueError(f"rerank response did not contain usable results: {payload}")
        return ranked


def _coerce_index(item: dict[str, Any]) -> int | None:
    for key in ("index", "document_index"):
        value = item.get(key)
        if isinstance(value, int):
            return value
    document = item.get("document")
    if isinstance(document, dict) and isinstance(document.get("index"), int):
        return document["index"]
    return None


def _coerce_score(item: dict[str, Any]) -> float | None:
    for key in ("relevance_score", "score"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None
