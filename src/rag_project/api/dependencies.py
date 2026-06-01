from functools import lru_cache

from rag_project.core.config import Settings, get_settings
from rag_project.parsers import ImageExplanationConfig, ImageExplanationGenerator, MinerUApiParser
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
