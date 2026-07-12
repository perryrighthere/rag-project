from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from rag_project.api.schemas import ChatRequest, DocumentRecord, KnowledgeBaseRecord, TaskRecord
from rag_project.chat import ChatConfig
from rag_project.graphs import IngestionGraph, QAGraph
from rag_project.parsers import ParsedDocument
from rag_project.qa import QAOrchestratorDependencyError
from rag_project.qa.autogen import AutoGenQAOrchestrator
from rag_project.qa.crewai import CrewAIQAOrchestrator
from rag_project.services.memory_store import InMemoryStore
from rag_project.services.task_recovery import INTERRUPTED_TASK_ERROR, mark_interrupted_tasks_failed


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
async def test_qa_graph_runs_langgraph_multi_with_trace() -> None:
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
            return documents[:top_n], None

    class FakeChatClient:
        async def generate_answer(self, *, query, documents):
            return "single answer"

        async def generate_text(self, prompt):
            if "Citation Reviewer" in prompt:
                return "FINAL_ANSWER: 报销需要审批。chunk_1\nREVIEW_NOTES: 引用 chunk_1 有依据。"
            if "Evidence Mapper" in prompt:
                return "chunk_1 支持报销需要审批。"
            if "Answer Writer" in prompt:
                return "报销需要审批。chunk_1"
            return "需要确认报销审批要求。"

    result = await QAGraph(
        retriever=FakeRetriever(),
        chat_client=FakeChatClient(),
        default_orchestrator="langgraph_multi",
    ).run(
        kb_id="kb",
        query="报销政策是什么？",
        filters={},
        top_k=3,
        top_n=1,
        include_agent_trace=True,
    )

    assert result.orchestrator == "langgraph_multi"
    assert result.answer == "报销需要审批。chunk_1"
    assert result.review_notes == "引用 chunk_1 有依据。"
    assert [step["role"] for step in result.agent_trace] == [
        "Query Analyst",
        "Evidence Mapper",
        "Answer Writer",
        "Citation Reviewer",
    ]


@pytest.mark.asyncio
async def test_qa_graph_omits_agent_trace_by_default_for_langgraph_multi() -> None:
    class FakeRetriever:
        def build_filter_expr(self, *, kb_id, filters):
            return f'kb_id == "{kb_id}"'

        async def retrieve_candidates(self, *, query, filter_expr, top_k):
            return [Document(page_content="上下文", metadata={"chunk_id": "chunk_1"})]

        async def rerank_documents(self, *, query, documents, top_n):
            return documents, None

    class FakeChatClient:
        async def generate_answer(self, *, query, documents):
            return "single answer"

        async def generate_text(self, prompt):
            if "Citation Reviewer" in prompt:
                return "FINAL_ANSWER: 当前知识库无法确认。\nREVIEW_NOTES: 证据不足。"
            return "中间步骤"

    result = await QAGraph(
        retriever=FakeRetriever(),
        chat_client=FakeChatClient(),
        default_orchestrator="langgraph_multi",
    ).run(kb_id="kb", query="问题", filters={}, top_k=1)

    assert result.answer == "当前知识库无法确认。"
    assert result.agent_trace == []
    assert result.review_notes == "证据不足。"


@pytest.mark.asyncio
async def test_crewai_orchestrator_reports_missing_optional_dependency(monkeypatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "crewai", None)
    orchestrator = CrewAIQAOrchestrator(
        chat_config=ChatConfig(base_url=None, api_key="EMPTY", model="test-model"),
    )

    with pytest.raises(QAOrchestratorDependencyError, match="requirements-agentic.txt"):
        await orchestrator.answer(
            query="问题",
            documents=[Document(page_content="上下文", metadata={"chunk_id": "chunk_1"})],
            citations=[{"chunk_id": "chunk_1"}],
        )


@pytest.mark.asyncio
async def test_autogen_orchestrator_reports_missing_optional_dependency(monkeypatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "autogen_agentchat", None)
    orchestrator = AutoGenQAOrchestrator(
        chat_config=ChatConfig(base_url=None, api_key="EMPTY", model="test-model"),
    )

    with pytest.raises(QAOrchestratorDependencyError, match="requirements-agentic.txt"):
        await orchestrator.answer(
            query="问题",
            documents=[Document(page_content="上下文", metadata={"chunk_id": "chunk_1"})],
            citations=[{"chunk_id": "chunk_1"}],
        )


def test_chat_request_rejects_invalid_orchestrator() -> None:
    with pytest.raises(ValueError):
        ChatRequest.model_validate(
            {
                "kb_id": "kb",
                "query": "问题",
                "orchestrator": "invalid",
            }
        )


@pytest.mark.asyncio
async def test_mark_interrupted_tasks_failed_marks_stale_background_tasks() -> None:
    store = InMemoryStore()
    await store.add_knowledge_base(KnowledgeBaseRecord(kb_id="kb", name="policy"))
    await store.add_document(
        DocumentRecord(
            document_id="doc",
            kb_id="kb",
            filename="demo.pdf",
            status="parsing",
            file_content=b"%PDF",
        )
    )
    await store.add_task(TaskRecord(task_id="task", task_type="parse", document_id="doc", status="running"))

    count = await mark_interrupted_tasks_failed(store)

    assert count == 1
    assert store.tasks["task"].status == "failed"
    assert store.tasks["task"].error == INTERRUPTED_TASK_ERROR
    assert store.documents["doc"].status == "failed"
    assert store.documents["doc"].error == INTERRUPTED_TASK_ERROR


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
