import asyncio
from datetime import datetime, timezone

from rag_project.api.schemas import DocumentRecord, KnowledgeBaseRecord, TaskRecord
from rag_project.chunking import ChunkRecord


class InMemoryStore:
    """Test fake implementing the production store contract."""

    def __init__(self) -> None:
        self.knowledge_bases: dict[str, KnowledgeBaseRecord] = {}
        self.documents: dict[str, DocumentRecord] = {}
        self.chunks: dict[str, ChunkRecord] = {}
        self.tasks: dict[str, TaskRecord] = {}
        self._lock = asyncio.Lock()

    def list_documents(self, kb_id: str) -> list[DocumentRecord]:
        return [record for record in self.documents.values() if record.kb_id == kb_id]

    async def delete_knowledge_base(self, kb_id: str) -> bool:
        async with self._lock:
            if kb_id not in self.knowledge_bases:
                return False
            ids = {doc.document_id for doc in self.list_documents(kb_id)}
            if any(task.document_id in ids and task.status in {"pending", "running"} for task in self.tasks.values()):
                raise ValueError("知识库中有正在执行的任务，请等待任务结束后再删除。")
            self.tasks = {key: task for key, task in self.tasks.items() if task.document_id not in ids}
            self.chunks = {key: chunk for key, chunk in self.chunks.items() if chunk.kb_id != kb_id}
            self.documents = {key: doc for key, doc in self.documents.items() if doc.kb_id != kb_id}
            del self.knowledge_bases[kb_id]
            return True

    async def bind_embedding(self, kb_id: str, model: str, dim: int) -> None:
        async with self._lock:
            record = self.knowledge_bases[kb_id]
            if (record.embedding_model, record.embedding_dim) not in {(None, None), (model, dim)}:
                raise ValueError("Embedding model/dimension mismatch. Use the original model or a new knowledge base and collection.")
            self.knowledge_bases[kb_id] = record.model_copy(update={"embedding_model": model, "embedding_dim": dim})

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
            if record.kb_id not in self.knowledge_bases:
                raise ValueError("Knowledge base not found")
            self.documents[record.document_id] = record
            return record

    async def update_document(self, document_id: str, **changes) -> DocumentRecord | None:
        async with self._lock:
            record = self.documents.get(document_id)
            if record is None:
                return None
            if record.status == "deleted":
                return record
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
            if record.document_id and record.document_id not in self.documents:
                raise ValueError("Knowledge base or document not found")
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
