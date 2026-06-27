from typing import Protocol

from rag_project.api.schemas import DocumentRecord, KnowledgeBaseRecord, TaskRecord
from rag_project.chunking import ChunkRecord


class Store(Protocol):
    @property
    def knowledge_bases(self) -> dict[str, KnowledgeBaseRecord]:
        ...

    @property
    def documents(self) -> dict[str, DocumentRecord]:
        ...

    @property
    def tasks(self) -> dict[str, TaskRecord]:
        ...

    def list_knowledge_bases(self) -> list[KnowledgeBaseRecord]:
        ...

    def get_knowledge_base(self, kb_id: str) -> KnowledgeBaseRecord | None:
        ...

    def get_document(self, document_id: str) -> DocumentRecord | None:
        ...

    def get_task(self, task_id: str) -> TaskRecord | None:
        ...

    async def add_knowledge_base(self, record: KnowledgeBaseRecord) -> KnowledgeBaseRecord:
        ...

    async def update_knowledge_base(self, kb_id: str, **changes) -> KnowledgeBaseRecord | None:
        ...

    async def add_document(self, record: DocumentRecord) -> DocumentRecord:
        ...

    async def update_document(self, document_id: str, **changes) -> DocumentRecord | None:
        ...

    async def replace_document_chunks(self, document_id: str, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        ...

    def list_document_chunks(self, document_id: str) -> list[ChunkRecord]:
        ...

    async def add_task(self, record: TaskRecord) -> TaskRecord:
        ...

    async def update_task(self, task_id: str, **changes) -> TaskRecord | None:
        ...
