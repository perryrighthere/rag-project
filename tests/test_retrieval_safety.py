import json
from types import SimpleNamespace

import pytest

from rag_project.api.schemas import DocumentRecord, KnowledgeBaseRecord
from rag_project.knowledge_base import MetadataSchema
from rag_project.rerankers import NoopReranker
from rag_project.retrieval import KnowledgeBaseRetriever
from rag_project.services.memory_store import InMemoryStore
from rag_project.vectorstores import MilvusSearchMatch, MilvusVectorStoreAdapter, VectorStoreConfig


class Embedding:
    config = SimpleNamespace(model="model-a", dim=2)

    async def embed_query(self, query):
        return [0.1, 0.2]


def match(document_id):
    return MilvusSearchMatch(chunk_id=document_id + "_chunk", document_id=document_id,
        score=0.9, text="内容", source_uri=None, heading_path="", page_start=None, page_end=None, metadata={})


@pytest.fixture
async def records():
    store = InMemoryStore()
    await store.add_knowledge_base(KnowledgeBaseRecord(kb_id="other", name="other"))
    await store.add_knowledge_base(KnowledgeBaseRecord(kb_id="kb", name="test", embedding_model="model-a", embedding_dim=2))
    for document_id, kb_id, status in [("live", "kb", "indexed"), ("deleted", "kb", "deleted"),
                                       ("failed", "kb", "failed"), ("foreign", "other", "indexed")]:
        await store.add_document(DocumentRecord(document_id=document_id, kb_id=kb_id, filename="test.md", status=status))
    return store


@pytest.mark.asyncio
async def test_deleted_failed_and_foreign_documents_are_excluded_before_and_after_search(records):
    class Vector:
        async def validate_embedding(self, **kwargs):
            pass

        async def search(self, **kwargs):
            assert 'document_id in ["live"]' in kwargs["filter_expr"]
            assert 'kb_id == "kb"' in kwargs["filter_expr"]
            # Deliberately return stale results to exercise the DB recheck.
            return [match(doc_id) for doc_id in ("live", "deleted", "failed", "foreign")]

    retriever = KnowledgeBaseRetriever(store=records, embedding_client_factory=Embedding,
        vector_store=Vector(), reranker=NoopReranker())
    result = await retriever.search(kb_id="kb", query="q", filters={}, top_k=10)
    assert [item.document_id for item in result.matches] == ["live"]


@pytest.mark.asyncio
async def test_deletion_during_vector_search_is_also_excluded(records):
    class Vector:
        async def validate_embedding(self, **kwargs):
            pass

        async def search(self, **kwargs):
            await records.update_document("live", status="deleted")
            return [match("live")]

    retriever = KnowledgeBaseRetriever(store=records, embedding_client_factory=Embedding,
        vector_store=Vector(), reranker=NoopReranker())
    assert not (await retriever.search(kb_id="kb", query="q", filters={}, top_k=10)).matches


@pytest.mark.asyncio
async def test_same_dimension_different_model_is_rejected_before_search(records):
    await records.update_knowledge_base("kb", embedding_model="different-model")
    retriever = KnowledgeBaseRetriever(store=records, embedding_client_factory=Embedding,
        vector_store=None, reranker=NoopReranker())
    with pytest.raises(ValueError, match="model/dimension mismatch"):
        await retriever.search(kb_id="kb", query="q", filters={}, top_k=10)


@pytest.mark.asyncio
async def test_empty_or_deleted_knowledge_base_does_not_call_models(records):
    await records.update_document("live", status="deleted")
    retriever = KnowledgeBaseRetriever(store=records, embedding_client_factory=None,
        vector_store=None, reranker=NoopReranker())
    result = await retriever.search(kb_id="kb", query="q", filters={}, top_k=10)
    assert result.matches == []


@pytest.mark.parametrize("description,dim,valid", [
    ({"embedding_model":"model-a", "embedding_dim":2}, 2, True),
    ({"embedding_model":"model-b", "embedding_dim":2}, 2, False),
    ({"embedding_model":"model-a", "embedding_dim":2}, 3, False),
    ({}, 2, False),
])
@pytest.mark.asyncio
async def test_collection_checks_model_signature_and_actual_dimension(description, dim, valid):
    class Client:
        def describe_collection(self, **kwargs):
            return {"description":json.dumps(description), "fields":[{"name":"embedding", "params":{"dim":dim}}]}

    vector = MilvusVectorStoreAdapter(VectorStoreConfig(uri="unused"), client=Client())
    if valid:
        await vector.validate_embedding(model="model-a", dim=2)
    else:
        with pytest.raises(ValueError):
            await vector.validate_embedding(model="model-a", dim=2)


@pytest.mark.parametrize("name", ["kb_id", "text", "embedding", "metadata_json", "_source_metadata"])
def test_metadata_cannot_shadow_system_fields(name):
    with pytest.raises(ValueError, match="reserved"):
        MetadataSchema(fields=[{"name":name,"type":"string"}])
