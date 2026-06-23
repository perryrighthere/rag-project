import pytest
from langchain_core.documents import Document

from rag_project.api.schemas import KnowledgeBaseRecord
from rag_project.knowledge_base import MetadataSchema
from rag_project.rerankers import OpenAICompatibleReranker
from rag_project.retrieval import KnowledgeBaseRetriever
from rag_project.services.memory_store import InMemoryStore
from rag_project.vectorstores import MilvusSearchMatch


def test_openai_compatible_reranker_parses_ranked_scores() -> None:
    documents = [
        Document(page_content="first", metadata={"chunk_id": "c1"}),
        Document(page_content="second", metadata={"chunk_id": "c2"}),
    ]

    ranked = OpenAICompatibleReranker._parse_results(  # noqa: SLF001 - adapter response parser test
        {"results": [{"index": 1, "relevance_score": 0.91}, {"index": 0, "relevance_score": 0.42}]},
        documents,
    )

    assert [document.metadata["chunk_id"] for document in ranked] == ["c2", "c1"]
    assert ranked[0].metadata["rerank_score"] == 0.91


@pytest.mark.asyncio
async def test_retriever_degrades_to_vector_order_when_rerank_fails() -> None:
    class FailingReranker:
        async def rerank(self, query, documents, top_n):
            raise RuntimeError("rerank offline")

    retriever = KnowledgeBaseRetriever(
        store=InMemoryStore(),
        embedding_client_factory=lambda: None,
        vector_store=None,
        reranker=FailingReranker(),
    )
    documents = [
        Document(page_content="a", metadata={"chunk_id": "c1"}),
        Document(page_content="b", metadata={"chunk_id": "c2"}),
    ]

    reranked, error = await retriever.rerank_documents(query="q", documents=documents, top_n=1)

    assert [document.metadata["chunk_id"] for document in reranked] == ["c1"]
    assert "rerank offline" in error


@pytest.mark.asyncio
async def test_retrieval_service_builds_filter_searches_and_reranks() -> None:
    class FakeEmbeddingClient:
        async def embed_query(self, query):
            return [0.1, 0.2]

    class FakeVectorStore:
        def __init__(self) -> None:
            self.filter_expr = None

        async def search(self, *, query_vector, filter_expr, top_k):
            self.filter_expr = filter_expr
            return [
                MilvusSearchMatch(
                    chunk_id="chunk_1",
                    document_id="doc",
                    score=0.7,
                    text="first",
                    source_uri="minio://bucket/one.md",
                    heading_path="A",
                    page_start=1,
                    page_end=1,
                    metadata={"doc_type": "policy"},
                ),
                MilvusSearchMatch(
                    chunk_id="chunk_2",
                    document_id="doc",
                    score=0.8,
                    text="second",
                    source_uri="minio://bucket/two.md",
                    heading_path="B",
                    page_start=2,
                    page_end=2,
                    metadata={"doc_type": "policy"},
                ),
            ]

    class FakeReranker:
        async def rerank(self, query, documents, top_n):
            ranked = []
            for score, document in zip([0.95, 0.5], reversed(documents)):
                ranked.append(
                    Document(
                        page_content=document.page_content,
                        metadata={**document.metadata, "rerank_score": score},
                    )
                )
            return ranked[:top_n]

    store = InMemoryStore()
    schema = MetadataSchema(fields=[{"name": "doc_type", "type": "string", "filterable": True}])
    await store.add_knowledge_base(KnowledgeBaseRecord(kb_id="kb", name="policy", metadata_schema=schema))
    vector_store = FakeVectorStore()
    retriever = KnowledgeBaseRetriever(
        store=store,
        embedding_client_factory=FakeEmbeddingClient,
        vector_store=vector_store,
        reranker=FakeReranker(),
    )

    result = await retriever.search(
        kb_id="kb",
        query="报销",
        filters={"doc_type": {"$eq": "policy"}},
        top_k=2,
        top_n=1,
    )

    assert vector_store.filter_expr == 'kb_id == "kb" and ((doc_type == "policy"))'
    assert [match.chunk_id for match in result.matches] == ["chunk_2"]
    assert result.matches[0].rerank_score == 0.95
    assert result.matches[0].metadata == {"doc_type": "policy"}
