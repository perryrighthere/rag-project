import asyncio
from datetime import datetime, timezone

from rag_project.api.schemas import DocumentRecord, KnowledgeBaseRecord, TaskRecord


class InMemoryStore:
    """Development store until the SQLAlchemy repositories from 4.4 are added."""

    def __init__(self) -> None:
        self.knowledge_bases: dict[str, KnowledgeBaseRecord] = {}
        self.documents: dict[str, DocumentRecord] = {}
        self.tasks: dict[str, TaskRecord] = {}
        self._lock = asyncio.Lock()

    async def add_knowledge_base(self, record: KnowledgeBaseRecord) -> KnowledgeBaseRecord:
        async with self._lock:
            self.knowledge_bases[record.kb_id] = record
            return record

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

