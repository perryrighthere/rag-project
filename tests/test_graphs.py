from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from rag_project.api.schemas import DocumentRecord, KnowledgeBaseRecord, TaskRecord
from rag_project.graphs import IngestionGraph, QAGraph
from rag_project.parsers import ParsedDocument
from rag_project.services.memory_store import InMemoryStore


@pytest.mark.asyncio
async def test_qa_graph_runs_minimal_flow() -> None:
    class FakeRetriever:
        def build_filter_expr(self, *, kb_id, filters):
            return f'kb_id == "{kb_id}"'

        async def retrieve_candidates(self, *, query, filter_expr, top_k):
            return [
                Document(
                    page_content="报销需要审批。",
                    metadata={"chunk_id": "chunk_1", "document_id": "doc", "score": 0.8},
                )
            ]

        async def rerank_documents(self, *, query, documents, top_n):
            return [
                Document(
                    page_content=documents[0].page_content,
                    metadata={**documents[0].metadata, "rerank_score": 0.9},
                )
            ], None

    class FakeChatClient:
        async def generate_answer(self, *, query, documents):
            return "报销需要审批。"

    result = await QAGraph(retriever=FakeRetriever(), chat_client=FakeChatClient()).run(
        kb_id="kb",
        query="报销政策是什么？",
        filters={},
        top_k=3,
        top_n=1,
    )

    assert result.answer == "报销需要审批。"
    assert result.filter_expr == 'kb_id == "kb"'
    assert result.citations[0]["chunk_id"] == "chunk_1"
    assert result.citations[0]["rerank_score"] == 0.9


@pytest.mark.asyncio
async def test_ingestion_graph_indexes_document_with_in_memory_store() -> None:
    class FakeParser:
        async def parse(self, file, options):
            return ParsedDocument(
                document_id=options.document_id,
                parser="mineru",
                parser_task_id="mineru_task",
                markdown_text="# 标题\n\n正文",
                markdown_object_key="parsed/kb/doc/markdown/demo.md",
                raw_object_key="raw/kb/doc/demo.pdf",
                parse_options=options.model_dump(mode="json"),
                created_at=datetime.now(timezone.utc),
            )

    class FakeEmbeddingClient:
        config = SimpleNamespace(model="embed-model", dim=2)

        async def embed_documents(self, texts):
            return [[0.1, 0.2] for _ in texts]

    class FakeVectorStore:
        def __init__(self) -> None:
            self.upsert_count = 0

        async def delete_document_chunks(self, *, kb_id, document_id):
            return None

        async def upsert_chunks(self, chunks, vectors, *, metadata_schema, embedding_dim):
            self.upsert_count = len(chunks)

    store = InMemoryStore()
    await store.add_knowledge_base(KnowledgeBaseRecord(kb_id="kb", name="policy"))
    await store.add_document(
        DocumentRecord(
            document_id="doc",
            kb_id="kb",
            filename="demo.pdf",
            content_type="application/pdf",
            file_content=b"%PDF",
        )
    )
    await store.add_task(TaskRecord(task_id="task", task_type="ingest", document_id="doc"))
    vector_store = FakeVectorStore()

    state = await IngestionGraph(
        store=store,
        parser=FakeParser(),
        embedding_client_factory=FakeEmbeddingClient,
        vector_store=vector_store,
    ).run(task_id="task", document_id="doc")

    assert state.get("error") is None
    assert store.documents["doc"].status == "indexed"
    assert store.tasks["task"].status == "succeeded"
    assert vector_store.upsert_count == store.documents["doc"].chunk_count
