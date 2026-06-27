import asyncio
from datetime import datetime, timezone

from rag_project.api.schemas import DocumentRecord, KnowledgeBaseRecord, TaskRecord
from rag_project.chunking import ChunkRecord


class InMemoryStore:
    """Development store until the SQLAlchemy repositories from 4.4 are added."""

    def __init__(self) -> None:
        self.knowledge_bases: dict[str, KnowledgeBaseRecord] = {}
        self.documents: dict[str, DocumentRecord] = {}
        self.chunks: dict[str, ChunkRecord] = {}
        self.tasks: dict[str, TaskRecord] = {}
        self._lock = asyncio.Lock()

    async def add_knowledge_base(self, record: KnowledgeBaseRecord) -> KnowledgeBaseRecord:
        async with self._lock:
            self.knowledge_bases[record.kb_id] = record
            return record

    def list_knowledge_bases(self) -> list[KnowledgeBaseRecord]:
        return list(self.knowledge_bases.values())

    def get_knowledge_base(self, kb_id: str) -> KnowledgeBaseRecord | None:
        return self.knowledge_bases.get(kb_id)

    def get_document(self, document_id: str) -> DocumentRecord | None:
        return self.documents.get(document_id)

    def get_task(self, task_id: str) -> TaskRecord | None:
        return self.tasks.get(task_id)

    async def update_knowledge_base(self, kb_id: str, **changes) -> KnowledgeBaseRecord | None:
        async with self._lock:
            record = self.knowledge_bases.get(kb_id)
            if record is None:
                return None
            updated = record.model_copy(update={**changes, "updated_at": datetime.now(timezone.utc)})
            self.knowledge_bases[kb_id] = updated
            return updated

    async def add_document(self, record: DocumentRecord) -> DocumentRecord:
        async with self._lock:
            self.documents[record.document_id] = record
            return record

    async def update_document(self, document_id: str, **changes) -> DocumentRecord | None:
        async with self._lock:
            record = self.documents.get(document_id)
            if record is None:
                return None
            updated = record.model_copy(update={**changes, "updated_at": datetime.now(timezone.utc)})
            self.documents[document_id] = updated
            return updated

    async def replace_document_chunks(self, document_id: str, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        async with self._lock:
            for chunk_id, chunk in list(self.chunks.items()):
                if chunk.document_id == document_id:
                    del self.chunks[chunk_id]
            for index, chunk in enumerate(chunks):
                indexed = chunk.model_copy(update={"chunk_index": index})
                self.chunks[indexed.chunk_id] = indexed
            return self.list_document_chunks(document_id)

    def list_document_chunks(self, document_id: str) -> list[ChunkRecord]:
        return sorted(
            [chunk for chunk in self.chunks.values() if chunk.document_id == document_id],
            key=lambda chunk: chunk.chunk_index,
        )

    async def add_task(self, record: TaskRecord) -> TaskRecord:
        async with self._lock:
            self.tasks[record.task_id] = record
            return record

    async def update_task(self, task_id: str, **changes) -> TaskRecord | None:
        async with self._lock:
            record = self.tasks.get(task_id)
            if record is None:
                return None
            updated = record.model_copy(update={**changes, "updated_at": datetime.now(timezone.utc)})
            self.tasks[task_id] = updated
            return updated


store = InMemoryStore()
