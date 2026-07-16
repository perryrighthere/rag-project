import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pymilvus import DataType, MilvusClient

from rag_project.chunking import ChunkRecord
from rag_project.knowledge_base import MetadataSchema


@dataclass(frozen=True)
class VectorStoreConfig:
    uri: str
    collection: str = "rag_chunks"
    vector_field: str = "embedding"
    metric_type: str = "COSINE"


@dataclass(frozen=True)
class MilvusSearchMatch:
    chunk_id: str
    document_id: str
    score: float | None
    text: str
    source_uri: str | None
    heading_path: str
    page_start: int | None
    page_end: int | None
    metadata: dict[str, Any]


class MilvusVectorStoreAdapter:
    """Project-owned Milvus boundary for chunk upsert and filtered search."""

    def __init__(self, config: VectorStoreConfig, *, client: MilvusClient | None = None):
        self.config = config
        self.client = client or MilvusClient(uri=config.uri)

    async def ensure_collection(self, *, embedding_dim: int, metadata_schema: MetadataSchema) -> None:
        await asyncio.to_thread(self._ensure_collection, embedding_dim, metadata_schema)

    async def upsert_chunks(
        self,
        chunks: list[ChunkRecord],
        vectors: list[list[float]],
        *,
        metadata_schema: MetadataSchema,
        embedding_dim: int,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        await self.ensure_collection(embedding_dim=embedding_dim, metadata_schema=metadata_schema)
        rows = [self._chunk_row(chunk, vector, metadata_schema) for chunk, vector in zip(chunks, vectors)]
        if rows:
            await asyncio.to_thread(self.client.upsert, collection_name=self.config.collection, data=rows)

    async def delete_document_chunks(self, *, kb_id: str, document_id: str) -> None:
        exists = await asyncio.to_thread(self.client.has_collection, self.config.collection)
        if not exists:
            return
        expr = f'kb_id == "{_escape_string(kb_id)}" and document_id == "{_escape_string(document_id)}"'
        await asyncio.to_thread(self.client.delete, collection_name=self.config.collection, filter=expr)

    async def update_chunk_metadata(
        self,
        chunks: list[ChunkRecord],
        *,
        metadata_schema: MetadataSchema,
    ) -> None:
        """Update Milvus metadata while preserving the existing chunk vectors."""
        if not chunks:
            return
        exists = await asyncio.to_thread(self.client.has_collection, self.config.collection)
        if not exists:
            raise ValueError(f"Milvus collection does not exist: {self.config.collection}")
        await asyncio.to_thread(self._update_chunk_metadata, chunks, metadata_schema)

    async def search(
        self,
        *,
        query_vector: list[float],
        filter_expr: str,
        top_k: int,
    ) -> list[MilvusSearchMatch]:
        results = await asyncio.to_thread(
            self.client.search,
            collection_name=self.config.collection,
            data=[query_vector],
            filter=filter_expr,
            limit=top_k,
            output_fields=[
                "kb_id",
                "document_id",
                "chunk_id",
                "chunk_index",
                "text",
                "source_uri",
                "heading_path",
                "page_start",
                "page_end",
                "metadata_json",
            ],
        )
        return [self._search_hit_to_match(hit) for hit in (results[0] if results else [])]

    def _ensure_collection(self, embedding_dim: int, metadata_schema: MetadataSchema) -> None:
        if self.client.has_collection(self.config.collection):
            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("kb_id", DataType.VARCHAR, max_length=128)
        schema.add_field("document_id", DataType.VARCHAR, max_length=128)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=128)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("source_uri", DataType.VARCHAR, max_length=2048)
        schema.add_field("heading_path", DataType.VARCHAR, max_length=2048)
        schema.add_field("page_start", DataType.INT64)
        schema.add_field("page_end", DataType.INT64)
        schema.add_field("created_at", DataType.INT64)
        schema.add_field("metadata_json", DataType.JSON)
        for field in metadata_schema.fields:
            if field.filterable:
                _add_metadata_field(schema, field.name, field.type)
        schema.add_field(self.config.vector_field, DataType.FLOAT_VECTOR, dim=embedding_dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=self.config.vector_field,
            index_type="HNSW",
            metric_type=self.config.metric_type,
            params={"M": 16, "efConstruction": 200},
        )
        self.client.create_collection(
            collection_name=self.config.collection,
            schema=schema,
            index_params=index_params,
        )

    def _chunk_row(
        self,
        chunk: ChunkRecord,
        vector: list[float],
        metadata_schema: MetadataSchema,
    ) -> dict[str, Any]:
        row = {
            "id": chunk.chunk_id,
            "kb_id": chunk.kb_id,
            "document_id": chunk.document_id,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "source_uri": chunk.source_uri or "",
            "heading_path": chunk.heading_path,
            "page_start": chunk.page_start if chunk.page_start is not None else -1,
            "page_end": chunk.page_end if chunk.page_end is not None else -1,
            "created_at": int(chunk.created_at.timestamp()),
            "metadata_json": chunk.metadata,
            self.config.vector_field: vector,
        }
        # Every schema-declared metadata value is copied to a Milvus field. Fields
        # not declared in the collection schema are stored as dynamic fields, so a
        # later filterability change does not require re-embedding the chunk.
        for field in metadata_schema.fields:
            if field.name in chunk.metadata:
                row[field.name] = chunk.metadata[field.name]
        return row

    def _update_chunk_metadata(self, chunks: list[ChunkRecord], metadata_schema: MetadataSchema) -> None:
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        updated_ids: set[str] = set()
        for chunk_batch in _batched(chunks, 1000):
            rows = self.client.get(
                collection_name=self.config.collection,
                ids=[chunk.chunk_id for chunk in chunk_batch],
                output_fields=["id", self.config.vector_field],
            )
            upsert_rows = []
            for row in rows:
                chunk_id = str(row.get("id") or "")
                chunk = chunks_by_id.get(chunk_id)
                vector = row.get(self.config.vector_field)
                if chunk is None or vector is None:
                    continue
                upsert_rows.append(self._chunk_row(chunk, vector, metadata_schema))
                updated_ids.add(chunk_id)
            if upsert_rows:
                self.client.upsert(collection_name=self.config.collection, data=upsert_rows)

        missing_ids = sorted(set(chunks_by_id) - updated_ids)
        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            raise ValueError(f"Milvus chunks not found while updating metadata: {preview}")

    @staticmethod
    def _search_hit_to_match(hit: dict[str, Any]) -> MilvusSearchMatch:
        entity = hit.get("entity", hit)
        metadata = entity.get("metadata_json") or {}
        page_start = entity.get("page_start")
        page_end = entity.get("page_end")
        score = hit.get("distance")
        if score is None:
            score = hit.get("score")
        return MilvusSearchMatch(
            chunk_id=str(entity["chunk_id"]),
            document_id=str(entity["document_id"]),
            score=score,
            text=str(entity.get("text") or ""),
            source_uri=entity.get("source_uri") or None,
            heading_path=str(entity.get("heading_path") or ""),
            page_start=page_start if page_start != -1 else None,
            page_end=page_end if page_end != -1 else None,
            metadata=metadata,
        )


def _add_metadata_field(schema, name: str, type_name: str) -> None:
    data_type = _milvus_scalar_type(type_name)
    if data_type == DataType.VARCHAR:
        schema.add_field(name, data_type, max_length=512, nullable=True)
    elif data_type == DataType.ARRAY:
        schema.add_field(
            name,
            data_type,
            element_type=DataType.VARCHAR,
            max_capacity=128,
            max_length=512,
            nullable=True,
        )
    else:
        schema.add_field(name, data_type, nullable=True)


def _batched(items: list[ChunkRecord], batch_size: int) -> list[list[ChunkRecord]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _milvus_scalar_type(type_name: str) -> DataType:
    if type_name in {"int", "date", "datetime"}:
        return DataType.INT64 if type_name == "int" else DataType.VARCHAR
    if type_name == "float":
        return DataType.DOUBLE
    if type_name == "bool":
        return DataType.BOOL
    if type_name == "string_array":
        return DataType.ARRAY
    return DataType.VARCHAR


def _escape_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
