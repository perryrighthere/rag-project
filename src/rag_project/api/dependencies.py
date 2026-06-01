from functools import lru_cache

from rag_project.core.config import Settings, get_settings
from rag_project.parsers import MinerUApiParser
from rag_project.storage import MinioStorage, MinioStorageConfig


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
    return MinerUApiParser(
        base_url=str(settings.mineru_base_url),
        storage=get_storage(),
        request_timeout=settings.mineru_request_timeout,
        poll_interval_seconds=settings.mineru_poll_interval_seconds,
        max_wait_seconds=settings.mineru_max_wait_seconds,
    )
