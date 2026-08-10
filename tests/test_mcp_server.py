from datetime import datetime, timezone

import pytest
from langchain_core.documents import Document

from rag_project.api.schemas import KnowledgeBaseRecord, TaskRecord
from rag_project.mcp import build_mcp_server
from rag_project.qa import AgentStep, AgenticQAResult


class FakeStore:
    def __init__(self) -> None:
        self.knowledge_bases = [
            KnowledgeBaseRecord(kb_id="kb_1", name="policy", description="报销政策"),
        ]
        self.tasks = {
            "task_1": TaskRecord(
                task_id="task_1",
                task_type="ingest",
                document_id="doc_1",
                status="succeeded",
                result={"chunk_count": 3},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        }

    def list_knowledge_bases(self):
        return self.knowledge_bases

    def get_task(self, task_id):
        return self.tasks.get(task_id)


class FakeRetriever:
    def __init__(self, error=None) -> None:
        self.error = error
        self.last_filters = None

    async def search(self, *, kb_id, query, filters, top_k, top_n=None):
        self.last_filters = filters
        if self.error:
            raise self.error
        return type(
            "Result",
            (),
            {
                "query": query,
                "filter_expr": 'kb_id == "kb_1"',
                "rerank_error": None,
                "matches": [
                    type(
                        "Match",
                        (),
                        {
                            "chunk_id": "chunk_1",
                            "document_id": "doc_1",
                            "score": 0.9,
                            "rerank_score": 0.95,
                            "text": "报销需要审批。",
                            "source_uri": "minio://bucket/raw/kb_1/doc_1/demo.pdf",
                            "heading_path": "报销政策",
                            "page_start": 1,
                            "page_end": 1,
                            "metadata": {"doc_type": "policy"},
                        },
                    )
                ],
            },
        )()


class FakeQAGraph:
    def __init__(self) -> None:
        self.last_orchestrator = None

    async def run(self, *, kb_id, query, filters, top_k, top_n=None, orchestrator=None, include_agent_trace=False):
        self.last_orchestrator = orchestrator
        return type(
            "Result",
            (),
            {
                "query": query,
                "answer": "报销需要审批。",
                "filter_expr": 'kb_id == "kb_1"',
                "orchestrator": orchestrator or "single",
                "rerank_error": None,
                "review_notes": None,
                "citations": [
                    {
                        "chunk_id": "chunk_1",
                        "document_id": "doc_1",
                        "source_uri": "minio://bucket/raw/kb_1/doc_1/demo.pdf",
                        "heading_path": "报销政策",
                    }
                ],
                "reranked_documents": [
                    Document(
                        page_content="报销需要审批。",
                        metadata={
                            "chunk_id": "chunk_1",
                            "document_id": "doc_1",
                            "score": 0.9,
                            "rerank_score": 0.95,
                            "source_uri": "minio://bucket/raw/kb_1/doc_1/demo.pdf",
                            "heading_path": "报销政策",
                            "page_start": 1,
                            "page_end": 1,
                            "_source_metadata": {"doc_type": "policy"},
                        },
                    )
                ],
            },
        )()


def build_server():
    return build_mcp_server(
        retriever_factory=lambda: FakeRetriever(),
        qa_graph_factory=lambda: FakeQAGraph(),
        store_factory=lambda: FakeStore(),
    )


def test_mcp_server_exposes_expected_tools() -> None:
    import asyncio

    server = build_server()
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == {"list_knowledge_bases", "retrieval_search", "qa_chat", "get_task_status"}


def test_list_knowledge_bases_tool_returns_records() -> None:
    import asyncio

    server = build_server()
    result = asyncio.run(server.call_tool("list_knowledge_bases", {}))
    content = result[0][0].text
    assert '"kb_id": "kb_1"' in content
    assert '"name": "policy"' in content


def test_retrieval_search_tool_returns_matches() -> None:
    import asyncio

    server = build_server()
    result = asyncio.run(
        server.call_tool(
            "retrieval_search",
            {"kb_id": "kb_1", "query": "报销政策是什么？", "top_k": 10, "top_n": 5},
        )
    )
    content = result[0][0].text
    assert '"chunk_id": "chunk_1"' in content
    assert '"text": "报销需要审批。"' in content
    assert '"rerank_score": 0.95' in content


def test_qa_chat_tool_returns_answer_with_citations() -> None:
    import asyncio

    server = build_server()
    result = asyncio.run(
        server.call_tool(
            "qa_chat",
            {"kb_id": "kb_1", "query": "报销需要审批吗？", "top_k": 10, "orchestrator": "single"},
        )
    )
    content = result[0][0].text
    assert '"answer": "报销需要审批。"' in content
    assert '"chunk_id": "chunk_1"' in content
    assert '"orchestrator": "single"' in content


def test_get_task_status_tool_returns_record() -> None:
    import asyncio

    server = build_server()
    result = asyncio.run(server.call_tool("get_task_status", {"task_id": "task_1"}))
    content = result[0][0].text
    assert '"status": "succeeded"' in content
    assert '"chunk_count": 3' in content


def test_get_task_status_tool_returns_not_found() -> None:
    import asyncio

    server = build_server()
    result = asyncio.run(server.call_tool("get_task_status", {"task_id": "missing"}))
    content = result[0][0].text
    assert '"status": "not_found"' in content


def test_retrieval_search_tool_propagates_missing_kb_error() -> None:
    import asyncio

    from mcp.server.fastmcp.exceptions import ToolError

    class MissingKBRetriever:
        async def search(self, *, kb_id, query, filters, top_k, top_n=None):
            raise KeyError(f"Knowledge base not found: {kb_id}")

    server = build_mcp_server(
        retriever_factory=MissingKBRetriever,
        qa_graph_factory=FakeQAGraph,
        store_factory=FakeStore,
    )
    with pytest.raises(ToolError, match="Knowledge base not found"):
        asyncio.run(server.call_tool("retrieval_search", {"kb_id": "missing", "query": "问题"}))


def test_mcp_server_is_importable_from_package() -> None:
    import rag_project.mcp as mcp_pkg

    assert mcp_pkg.build_mcp_server is build_mcp_server
