from functools import lru_cache

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "rag-project"
    app_env: str = "local"
    api_prefix: str = ""

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "rag-project"
    minio_secure: bool = False
    minio_public_endpoint: str | None = None

    mineru_base_url: HttpUrl = Field(default="http://localhost:8001")
    mineru_request_timeout: float = 60.0
    mineru_poll_interval_seconds: float = 2.0
    mineru_max_wait_seconds: float = 600.0

    vlm_image_explanations_enabled: bool = False
    vlm_base_url: str | None = None
    vlm_api_key: str = "EMPTY"
    vlm_model: str | None = None
    vlm_prompt: str | None = None
    vlm_timeout: float = 120.0
    vlm_max_tokens: int = 300

    embedding_base_url: str | None = None
    embedding_api_key: str = "EMPTY"
    embedding_model: str | None = None
    embedding_dim: int = 1024
    embedding_batch_size: int = 32
    embedding_timeout: float = 60.0

    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection: str = "rag_chunks"


@lru_cache
def get_settings() -> Settings:
    return Settings()
