from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from rag_project.api.schemas import DocumentRecord, KnowledgeBaseRecord, TaskRecord
from rag_project.chunking import ChunkRecord, ChunkingConfig
from rag_project.db.models import ChunkModel, DocumentModel, IngestionTaskModel, KnowledgeBaseModel
from rag_project.knowledge_base import MetadataSchema
from rag_project.parsers import ParsedDocument


SessionFactory = Callable[[], Session]


class SQLAlchemyStore:
    """Small repository facade over SQLAlchemy ORM models.

    The public methods intentionally mirror `InMemoryStore` so API routes,
    retrieval, and graph code can depend on one narrow persistence contract.
    """

    def __init__(self, session_factory: SessionFactory):
        self._session_factory = session_factory

    @property
    def knowledge_bases(self) -> dict[str, KnowledgeBaseRecord]:
        with self._session_factory() as session:
            records = session.scalars(select(KnowledgeBaseModel)).all()
            return {record.id: _kb_to_record(record) for record in records}

    @property
    def documents(self) -> dict[str, DocumentRecord]:
        with self._session_factory() as session:
            records = session.scalars(select(DocumentModel)).all()
            return {record.id: _document_to_record(record) for record in records}

    @property
    def tasks(self) -> dict[str, TaskRecord]:
        with self._session_factory() as session:
            records = session.scalars(select(IngestionTaskModel)).all()
            return {record.id: _task_to_record(record) for record in records}

    def list_knowledge_bases(self) -> list[KnowledgeBaseRecord]:
        with self._session_factory() as session:
            records = session.scalars(select(KnowledgeBaseModel).order_by(KnowledgeBaseModel.created_at)).all()
            return [_kb_to_record(record) for record in records]

    def get_knowledge_base(self, kb_id: str) -> KnowledgeBaseRecord | None:
        with self._session_factory() as session:
            model = session.get(KnowledgeBaseModel, kb_id)
            return _kb_to_record(model) if model else None

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._session_factory() as session:
            model = session.get(DocumentModel, document_id)
            return _document_to_record(model) if model else None

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._session_factory() as session:
            model = session.get(IngestionTaskModel, task_id)
            return _task_to_record(model) if model else None

    async def add_knowledge_base(self, record: KnowledgeBaseRecord) -> KnowledgeBaseRecord:
        with self._session_factory() as session:
            model = KnowledgeBaseModel(
                id=record.kb_id,
                name=record.name,
                description=record.description,
                metadata_schema_json=record.metadata_schema.model_dump(mode="json"),
                chunking_config_json=record.chunking_config.model_dump(mode="json"),
                embedding_model=record.embedding_model,
                embedding_dim=record.embedding_dim,
                created_at=record.created_at,
                updated_at=record.updated_at,
            )
            session.add(model)
            session.commit()
            session.refresh(model)
            return _kb_to_record(model)

    async def update_knowledge_base(self, kb_id: str, **changes: Any) -> KnowledgeBaseRecord | None:
        with self._session_factory() as session:
            model = session.get(KnowledgeBaseModel, kb_id)
            if model is None:
                return None
            for key, value in changes.items():
                if key == "metadata_schema":
                    model.metadata_schema_json = value.model_dump(mode="json")
                elif key == "chunking_config":
                    model.chunking_config_json = value.model_dump(mode="json")
                elif hasattr(model, key):
                    setattr(model, key, value)
            model.updated_at = _now()
            session.commit()
            session.refresh(model)
            return _kb_to_record(model)

    async def add_document(self, record: DocumentRecord) -> DocumentRecord:
        with self._session_factory() as session:
            model = DocumentModel(
                id=record.document_id,
                kb_id=record.kb_id,
                filename=record.filename,
                content_type=record.content_type,
                file_size=len(record.file_content) if record.file_content is not None else None,
                status=record.status,
                raw_object_key=record.raw_object_key,
                metadata_json=record.metadata,
                parsed_document_json=_parsed_document_json(record.parsed_document),
                parser=record.parsed_document.parser if record.parsed_document else None,
                parser_task_id=record.parsed_document.parser_task_id if record.parsed_document else None,
                markdown_object_key=record.parsed_document.markdown_object_key if record.parsed_document else None,
                content_list_object_key=record.parsed_document.content_list_object_key if record.parsed_document else None,
                middle_json_object_key=record.parsed_document.middle_json_object_key if record.parsed_document else None,
                chunk_count=record.chunk_count,
                embedding_model=record.embedding_model,
                embedding_dim=record.embedding_dim,
                error_message=record.error,
                file_content=record.file_content,
                created_at=record.created_at,
                updated_at=record.updated_at,
            )
            session.add(model)
            session.commit()
            session.refresh(model)
            return _document_to_record(model)

    async def update_document(self, document_id: str, **changes: Any) -> DocumentRecord | None:
        with self._session_factory() as session:
            model = session.get(DocumentModel, document_id)
            if model is None:
                return None
            for key, value in changes.items():
                if key == "metadata":
                    model.metadata_json = value
                elif key == "parsed_document":
                    model.parsed_document_json = _parsed_document_json(value)
                    model.parser = value.parser if value else None
                    model.parser_task_id = value.parser_task_id if value else None
                    model.markdown_object_key = value.markdown_object_key if value else None
                    model.content_list_object_key = value.content_list_object_key if value else None
                    model.middle_json_object_key = value.middle_json_object_key if value else None
                elif key == "error":
                    model.error_message = value
                elif hasattr(model, key):
                    setattr(model, key, value)
            model.updated_at = _now()
            session.commit()
            session.refresh(model)
            return _document_to_record(model)

    async def replace_document_chunks(self, document_id: str, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        with self._session_factory() as session:
            session.query(ChunkModel).filter(ChunkModel.document_id == document_id).delete()
            for index, chunk in enumerate(chunks):
                indexed = chunk.model_copy(update={"chunk_index": index})
                session.add(_chunk_to_model(indexed))
            session.commit()
        return self.list_document_chunks(document_id)

    def list_document_chunks(self, document_id: str) -> list[ChunkRecord]:
        with self._session_factory() as session:
            records = session.scalars(
                select(ChunkModel).where(ChunkModel.document_id == document_id).order_by(ChunkModel.chunk_index)
            ).all()
            return [_chunk_to_record(record) for record in records]

    async def add_task(self, record: TaskRecord) -> TaskRecord:
        with self._session_factory() as session:
            model = IngestionTaskModel(
                id=record.task_id,
                task_type=record.task_type,
                status=record.status,
                document_id=record.document_id,
                result_json=record.result,
                error_message=record.error,
                created_at=record.created_at,
                updated_at=record.updated_at,
            )
            session.add(model)
            session.commit()
            session.refresh(model)
            return _task_to_record(model)

    async def update_task(self, task_id: str, **changes: Any) -> TaskRecord | None:
        with self._session_factory() as session:
            model = session.get(IngestionTaskModel, task_id)
            if model is None:
                return None
            for key, value in changes.items():
                if key == "result":
                    model.result_json = value
                elif key == "error":
                    model.error_message = value
                elif hasattr(model, key):
                    setattr(model, key, value)
            model.updated_at = _now()
            session.commit()
            session.refresh(model)
            return _task_to_record(model)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _kb_to_record(model: KnowledgeBaseModel) -> KnowledgeBaseRecord:
    return KnowledgeBaseRecord(
        kb_id=model.id,
        name=model.name,
        description=model.description,
        metadata_schema=MetadataSchema.model_validate(model.metadata_schema_json or {}),
        chunking_config=ChunkingConfig.model_validate(model.chunking_config_json or {}),
        embedding_model=model.embedding_model,
        embedding_dim=model.embedding_dim,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def _document_to_record(model: DocumentModel) -> DocumentRecord:
    parsed_document = None
    if model.parsed_document_json:
        parsed_document = ParsedDocument.model_validate(model.parsed_document_json)
    return DocumentRecord(
        document_id=model.id,
        kb_id=model.kb_id,
        filename=model.filename,
        content_type=model.content_type,
        metadata=dict(model.metadata_json or {}),
        status=model.status,
        raw_object_key=model.raw_object_key,
        parsed_document=parsed_document,
        chunk_count=model.chunk_count,
        embedding_model=model.embedding_model,
        embedding_dim=model.embedding_dim,
        error=model.error_message,
        created_at=model.created_at,
        updated_at=model.updated_at,
        file_content=model.file_content,
    )


def _chunk_to_model(record: ChunkRecord) -> ChunkModel:
    return ChunkModel(
        id=record.chunk_id,
        kb_id=record.kb_id,
        document_id=record.document_id,
        chunk_index=record.chunk_index,
        text=record.text,
        heading_path=record.heading_path,
        page_start=record.page_start,
        page_end=record.page_end,
        source_uri=record.source_uri,
        metadata_json=record.metadata,
        embedding_model=record.embedding_model,
        embedding_dim=record.embedding_dim,
        created_at=record.created_at,
        updated_at=record.created_at,
    )


def _chunk_to_record(model: ChunkModel) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=model.id,
        kb_id=model.kb_id,
        document_id=model.document_id,
        chunk_index=model.chunk_index,
        text=model.text,
        heading_path=model.heading_path,
        page_start=model.page_start,
        page_end=model.page_end,
        source_uri=model.source_uri,
        metadata=dict(model.metadata_json or {}),
        embedding_model=model.embedding_model,
        embedding_dim=model.embedding_dim,
        created_at=model.created_at,
    )


def _task_to_record(model: IngestionTaskModel) -> TaskRecord:
    return TaskRecord(
        task_id=model.id,
        task_type=model.task_type,
        status=model.status,
        document_id=model.document_id,
        result=model.result_json,
        error=model.error_message,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def _parsed_document_json(parsed_document: ParsedDocument | None) -> dict[str, Any] | None:
    if parsed_document is None:
        return None
    return parsed_document.model_dump(mode="json")
