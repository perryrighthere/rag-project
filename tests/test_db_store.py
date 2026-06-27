from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from rag_project.api.schemas import DocumentRecord, KnowledgeBaseRecord, TaskRecord
from rag_project.chunking import ChunkRecord, ChunkingConfig
from rag_project.db.models import Base
from rag_project.db.store import SQLAlchemyStore
from rag_project.knowledge_base import MetadataSchema
from rag_project.parsers import ParsedDocument


@pytest.fixture
def db_store() -> SQLAlchemyStore:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return SQLAlchemyStore(sessionmaker(bind=engine, autoflush=False, expire_on_commit=False))


@pytest.mark.asyncio
async def test_sqlalchemy_store_persists_knowledge_base_document_chunks_and_task(db_store: SQLAlchemyStore) -> None:
    schema = MetadataSchema(
        fields=[{"name": "doc_type", "type": "string", "required": True, "filterable": True}]
    )
    chunking_config = ChunkingConfig(chunk_size=400, chunk_overlap=40)
    kb = await db_store.add_knowledge_base(
        KnowledgeBaseRecord(
            kb_id="kb",
            name="policy",
            metadata_schema=schema,
            chunking_config=chunking_config,
        )
    )

    assert db_store.knowledge_bases["kb"].metadata_schema == schema
    assert kb.chunking_config.chunk_size == 400

    parsed = ParsedDocument(
        document_id="doc",
        parser="mineru",
        parser_task_id="mineru_task",
        markdown_text="# Title\n\nBody",
        markdown_object_key="parsed/kb/doc/markdown/demo.md",
        content_list_object_key="parsed/kb/doc/json/demo_content_list.json",
        middle_json_object_key="parsed/kb/doc/json/demo_middle.json",
        raw_object_key="raw/kb/doc/demo.pdf",
        parse_options={"backend": "pipeline"},
        created_at=datetime.now(timezone.utc),
    )
    await db_store.add_document(
        DocumentRecord(
            document_id="doc",
            kb_id="kb",
            filename="demo.pdf",
            metadata={"doc_type": "policy"},
            parsed_document=parsed,
            file_content=b"%PDF",
        )
    )

    stored_document = db_store.documents["doc"]
    assert stored_document.parsed_document == parsed
    assert stored_document.file_content == b"%PDF"

    chunks = await db_store.replace_document_chunks(
        "doc",
        [
            ChunkRecord(
                chunk_id="chunk_2",
                kb_id="kb",
                document_id="doc",
                chunk_index=99,
                text="second",
                metadata={"doc_type": "policy"},
            ),
            ChunkRecord(
                chunk_id="chunk_1",
                kb_id="kb",
                document_id="doc",
                chunk_index=99,
                text="first",
                metadata={"doc_type": "policy"},
            ),
        ],
    )

    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert [chunk.chunk_id for chunk in db_store.list_document_chunks("doc")] == ["chunk_2", "chunk_1"]

    task = await db_store.add_task(TaskRecord(task_id="task", task_type="index", document_id="doc"))
    assert task.status == "pending"
    updated_task = await db_store.update_task("task", status="succeeded", result={"chunk_count": 2})

    assert updated_task is not None
    assert db_store.tasks["task"].result == {"chunk_count": 2}
