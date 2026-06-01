import asyncio
from dataclasses import dataclass
from io import BytesIO
from urllib.parse import urlparse

from minio import Minio

from rag_project.storage.object_keys import build_http_object_url


@dataclass(frozen=True)
class MinioStorageConfig:
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    secure: bool = False
    public_endpoint: str | None = None


@dataclass(frozen=True)
class StoredObject:
    bucket: str
    object_key: str
    content_type: str | None = None
    size: int | None = None
    url: str | None = None

    @property
    def uri(self) -> str:
        return f"minio://{self.bucket}/{self.object_key}"


class MinioStorage:
    """Small async facade over the MinIO Python client."""

    def __init__(self, config: MinioStorageConfig):
        self.config = config
        endpoint, secure = self._client_endpoint(config.endpoint, config.secure)
        self._client = Minio(
            endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key,
            secure=secure,
        )

    @staticmethod
    def _client_endpoint(endpoint: str, secure: bool) -> tuple[str, bool]:
        if endpoint.startswith(("http://", "https://")):
            parsed = urlparse(endpoint)
            return parsed.netloc, parsed.scheme == "https"
        return endpoint, secure

    @property
    def public_endpoint(self) -> str:
        return self.config.public_endpoint or self.config.endpoint

    async def ensure_bucket(self) -> None:
        def ensure() -> None:
            if not self._client.bucket_exists(self.config.bucket):
                self._client.make_bucket(self.config.bucket)

        await asyncio.to_thread(ensure)

    async def put_bytes(self, object_key: str, payload: bytes, content_type: str | None = None) -> StoredObject:
        await self.ensure_bucket()

        def put() -> None:
            self._client.put_object(
                self.config.bucket,
                object_key,
                BytesIO(payload),
                length=len(payload),
                content_type=content_type or "application/octet-stream",
            )

        await asyncio.to_thread(put)
        return StoredObject(
            bucket=self.config.bucket,
            object_key=object_key,
            content_type=content_type,
            size=len(payload),
            url=self.object_url(object_key),
        )

    async def get_bytes(self, object_key: str) -> bytes:
        def get() -> bytes:
            response = self._client.get_object(self.config.bucket, object_key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()

        return await asyncio.to_thread(get)

    async def delete_object(self, object_key: str) -> None:
        await asyncio.to_thread(self._client.remove_object, self.config.bucket, object_key)

    def object_url(self, object_key: str) -> str:
        return build_http_object_url(
            self.public_endpoint,
            self.config.bucket,
            object_key,
            secure=self.config.secure,
        )

