from functools import lru_cache

from rag_project.core.config import Settings, get_settings
from rag_project.chat import ChatConfig, OpenAICompatibleChatClient
from rag_project.db import get_store
from rag_project.embeddings import EmbeddingConfig, OpenAICompatibleEmbeddingClient
from rag_project.graphs import IngestionGraph, QAGraph
from rag_project.parsers import ImageExplanationConfig, ImageExplanationGenerator, MinerUApiParser
from rag_project.rerankers import NoopReranker, OpenAICompatibleReranker, RerankConfig, Reranker
from rag_project.retrieval import KnowledgeBaseRetriever
from rag_project.storage import MinioStorage, MinioStorageConfig
from rag_project.vectorstores import MilvusVectorStoreAdapter, VectorStoreConfig


@lru_cache
def get_storage() -> MinioStorage:
    settings = get_settings()
    return MinioStorage(
        MinioStorageConfig(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
            secure=settings.minio_secure,
            public_endpoint=settings.minio_public_endpoint,
        )
    )


def get_parser() -> MinerUApiParser:
    settings: Settings = get_settings()
    image_explainer = None
    if settings.vlm_image_explanations_enabled:
        config_kwargs = {
            "enabled": True,
            "base_url": settings.vlm_base_url,
            "api_key": settings.vlm_api_key,
            "model": settings.vlm_model,
            "timeout": settings.vlm_timeout,
            "max_tokens": settings.vlm_max_tokens,
        }
        if settings.vlm_prompt:
            config_kwargs["prompt"] = settings.vlm_prompt
        image_explainer = ImageExplanationGenerator(ImageExplanationConfig(**config_kwargs))

    return MinerUApiParser(
        base_url=str(settings.mineru_base_url),
        storage=get_storage(),
        request_timeout=settings.mineru_request_timeout,
        poll_interval_seconds=settings.mineru_poll_interval_seconds,
        max_wait_seconds=settings.mineru_max_wait_seconds,
        image_explainer=image_explainer,
    )


def get_embedding_client() -> OpenAICompatibleEmbeddingClient:
    settings = get_settings()
    if not settings.embedding_model:
        raise ValueError("EMBEDDING_MODEL must be configured before indexing or retrieval")
    return OpenAICompatibleEmbeddingClient(
        EmbeddingConfig(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            batch_size=settings.embedding_batch_size,
            timeout=settings.embedding_timeout,
        )
    )


@lru_cache
def get_vector_store() -> MilvusVectorStoreAdapter:
    settings = get_settings()
    return MilvusVectorStoreAdapter(
        VectorStoreConfig(
            uri=settings.milvus_uri,
            collection=settings.milvus_collection,
        )
    )


@lru_cache
def get_reranker() -> Reranker:
    settings = get_settings()
    if settings.rerank_base_url and settings.rerank_model:
        return OpenAICompatibleReranker(
            RerankConfig(
                base_url=settings.rerank_base_url,
                api_key=settings.rerank_api_key,
                model=settings.rerank_model,
                timeout=settings.rerank_timeout,
            )
        )
    return NoopReranker()


def get_retriever() -> KnowledgeBaseRetriever:
    return KnowledgeBaseRetriever(
        store=get_store(),
        embedding_client_factory=get_embedding_client,
        vector_store=get_vector_store(),
        reranker=get_reranker(),
    )


def get_chat_client() -> OpenAICompatibleChatClient:
    settings = get_settings()
    if not settings.chat_model:
        raise ValueError("CHAT_MODEL must be configured before using /chat")
    return OpenAICompatibleChatClient(
        ChatConfig(
            base_url=settings.chat_base_url,
            api_key=settings.chat_api_key,
            model=settings.chat_model,
            timeout=settings.chat_timeout,
            temperature=settings.chat_temperature,
            max_tokens=settings.chat_max_tokens,
        )
    )


def get_qa_graph() -> QAGraph:
    settings = get_settings()
    return QAGraph(
        retriever=get_retriever(),
        chat_client=get_chat_client(),
        default_orchestrator=settings.qa_orchestrator,
        agent_max_rounds=settings.qa_agent_max_rounds,
    )


def get_ingestion_graph() -> IngestionGraph:
    return IngestionGraph(
        store=get_store(),
        parser=get_parser(),
        embedding_client_factory=get_embedding_client,
        vector_store=get_vector_store(),
    )
