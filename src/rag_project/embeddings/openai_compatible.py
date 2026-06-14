import asyncio
from dataclasses import dataclass

from langchain_openai import OpenAIEmbeddings


class EmbeddingDimensionError(ValueError):
    pass


@dataclass(frozen=True)
class EmbeddingConfig:
    base_url: str | None
    api_key: str
    model: str
    dim: int
    batch_size: int = 32
    timeout: float = 60.0


class OpenAICompatibleEmbeddingClient:
    """OpenAI-compatible embedding facade with explicit dimension validation."""

    def __init__(self, config: EmbeddingConfig):
        if not config.model:
            raise ValueError("embedding model is required")
        if config.dim <= 0:
            raise ValueError("embedding dim must be positive")
        self.config = config
        self._embeddings = OpenAIEmbeddings(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            batch_vectors = await self._embed_batch(batch)
            self.validate_vectors(batch_vectors)
            vectors.extend(batch_vectors)
        return vectors

    async def embed_query(self, query: str) -> list[float]:
        vector = await self._embed_query(query)
        self.validate_vectors([vector])
        return vector

    def validate_vectors(self, vectors: list[list[float]]) -> None:
        for index, vector in enumerate(vectors):
            if len(vector) != self.config.dim:
                raise EmbeddingDimensionError(
                    f"embedding vector {index} has dim {len(vector)}, expected {self.config.dim}"
                )

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        if hasattr(self._embeddings, "aembed_documents"):
            return await self._embeddings.aembed_documents(batch)
        return await asyncio.to_thread(self._embeddings.embed_documents, batch)

    async def _embed_query(self, query: str) -> list[float]:
        if hasattr(self._embeddings, "aembed_query"):
            return await self._embeddings.aembed_query(query)
        return await asyncio.to_thread(self._embeddings.embed_query, query)
