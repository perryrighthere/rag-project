from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from rag_project.api import routes as api_routes
from rag_project.chunking import ChunkRecord
from rag_project.core.config import Settings
from rag_project.db.models import Base
from rag_project.db.store import SQLAlchemyStore
from rag_project.main import create_app
from rag_project.teaching import routes
from rag_project.teaching.dataset import load_dataset
from rag_project.teaching.reports import ReportStore


@pytest.fixture
def classroom(monkeypatch):
    engine = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    store = SQLAlchemyStore(factory)
    monkeypatch.setattr(routes, "get_store", lambda: store)
    monkeypatch.setattr(api_routes, "get_store", lambda: store)
    monkeypatch.setattr(routes, "report_store", lambda: ReportStore(factory))
    monkeypatch.setattr(routes, "get_settings", lambda: Settings(_env_file=None,
        embedding_model="classroom-embed", embedding_dim=2, chat_model="classroom-chat",
        rerank_model="classroom-rerank", rerank_base_url="http://unused"))
    yield TestClient(create_app()), store, factory
    engine.dispose()


def test_fixed_dataset_preview_and_independent_imports(classroom):
    client, store, _ = classroom
    data = client.get("/teaching/dataset").json()
    assert len(data["documents"]) == 5 and len(data["questions"]) == 7
    assert data["sha256"] == load_dataset()["sha256"]
    preview = client.post("/teaching/preview", json={"document_id": "travel-2025",
        "chunking_config": {"chunk_size": 100, "chunk_overlap": 10}})
    assert preview.status_code == 200
    assert len(preview.json()["chunks"]) > 3
    assert client.post("/teaching/preview", json={"document_id":"travel-2025",
        "chunking_config":{"chunk_size":100,"chunk_overlap":100}}).status_code == 422
    first = client.post("/teaching/dataset/import", json={}).json()
    second = client.post("/teaching/dataset/import", json={}).json()
    assert first["knowledge_base"]["kb_id"] != second["knowledge_base"]["kb_id"]
    for doc in first["documents"]:
        assert doc["status"] == "parsed"
        assert doc["parsed_document"]["source_uri"].startswith("teaching://")
        assert "file_content" not in doc
        assert "reference_answer" not in doc["parsed_document"]["markdown_text"]
    kb_id = first["knowledge_base"]["kb_id"]
    assert len(client.get(f"/knowledge-bases/{kb_id}/documents").json()) == 5
    assert len(store.list_documents(kb_id)) == 5


@pytest.mark.asyncio
async def test_experiment_records_actual_stages_and_survives_store_recreation(classroom, monkeypatch):
    client, store, factory = classroom
    seed = client.post("/teaching/dataset/import", json={}).json()
    kb_id = seed["knowledge_base"]["kb_id"]
    doc = seed["documents"][1]
    await store.update_document(doc["document_id"], status="indexed")
    await store.replace_document_chunks(doc["document_id"], [ChunkRecord(
        chunk_id="c1", kb_id=kb_id, document_id=doc["document_id"], chunk_index=0, text="450元")])

    class Retriever:
        def build_filter_expr(self, **kwargs):
            return f'kb_id == "{kb_id}"'

        async def retrieve_candidates(self, **kwargs):
            return [Document(page_content="旧排序", metadata={"chunk_id":"c2", "document_id":doc["document_id"], "score":0.9}),
                    Document(page_content="住宿上限450元", metadata={"chunk_id":"c1", "document_id":doc["document_id"], "score":0.8})]

        async def rerank_documents(self, *, query, documents, top_n):
            return [Document(page_content=documents[1].page_content,
                metadata={**documents[1].metadata, "rerank_score":0.95})][:top_n], None

    class Chat:
        config = SimpleNamespace(model="classroom-chat")

        async def generate_text(self, prompt):
            assert "住宿上限450元" in prompt
            assert "learning_goal" not in prompt
            return "每晚450元 [c1]"

    monkeypatch.setattr(routes, "get_retriever", Retriever)
    monkeypatch.setattr(routes, "get_chat_client", Chat)
    request = {"kb_id":kb_id, "query":load_dataset()["questions"][0]["query"], "mode":"chat", "top_n":1}
    response = client.post("/teaching/experiments", json=request)
    assert response.status_code == 201
    report = response.json()
    assert report["status"] == "succeeded"
    assert report["candidates"][0]["chunk_id"] == "c2"
    assert report["matches"][0]["chunk_id"] == "c1"
    assert report["matches"][0]["rerank_score"] == 0.95
    assert report["answer"] == "每晚450元 [c1]"
    assert report["prompts"] == [report["prompt"]]
    assert report["rerank_status"] == "applied"
    assert set(report["timings_ms"]) == {"retrieval", "rerank", "generation"}
    assert ReportStore(factory).get(report["id"]) == report
    # Reports are snapshots: deleting/changing live documents cannot rewrite the result.
    await store.update_document(doc["document_id"], status="deleted")
    saved = client.get(f'/teaching/experiments/{report["id"]}').json()
    assert saved["answer"] == report["answer"]
    assert saved["documents"][1]["status"] == "indexed"
    markdown = client.get(f'/teaching/experiments/{report["id"]}/markdown')
    assert "attachment" in markdown.headers["content-disposition"]
    assert "450元" in markdown.text and "classroom-embed" in markdown.text
    assert "api_key" not in markdown.text
    assert any(row["id"] == report["id"] for row in client.get("/teaching/experiments").json())


def test_failed_experiments_are_saved_without_external_services(classroom):
    client, _, _ = classroom
    seed = client.post("/teaching/dataset/import", json={}).json()
    response = client.post("/teaching/experiments", json={"kb_id":seed["knowledge_base"]["kb_id"], "query":"住宿？"})
    report = response.json()
    assert report["status"] == "failed" and "索引" in report["error"]
    assert client.get(f'/teaching/experiments/{report["id"]}').json() == report
    assert client.post("/teaching/experiments", json={"kb_id":"kb", "query":" ", "top_k":0}).status_code == 422


def test_workbench_serves_local_assets(classroom):
    client, _, _ = classroom
    assert "RAG 课堂实验室" in client.get("/workbench/").text
    assert client.get("/workbench/app.js").status_code == 200
    assert client.get("/workbench/style.css").status_code == 200


@pytest.mark.asyncio
async def test_index_model_mismatch_fails_before_embedding_or_deleting_old_vectors(classroom):
    client, store, _ = classroom
    seed = client.post("/teaching/dataset/import", json={}).json()
    kb_id = seed["knowledge_base"]["kb_id"]
    document_id = seed["documents"][0]["document_id"]
    await store.bind_embedding(kb_id, "original-model", 2)
    from rag_project.api.schemas import TaskRecord
    task = await store.add_task(TaskRecord(task_type="index", document_id=document_id))
    embedding = SimpleNamespace(config=SimpleNamespace(model="new-model", dim=2))
    # Neither fake has external methods: a preflight failure must not call them.
    await api_routes._run_index_task(task.task_id, document_id, lambda: embedding, object())
    assert store.get_task(task.task_id).status == "failed"
    assert "mismatch" in store.get_task(task.task_id).error
    assert store.get_knowledge_base(kb_id).embedding_model == "original-model"


@pytest.mark.asyncio
async def test_active_tasks_block_duplicate_index_and_delete(classroom):
    client, store, _ = classroom
    seed = client.post("/teaching/dataset/import", json={}).json()
    document_id = seed["documents"][0]["document_id"]
    from rag_project.api.schemas import TaskRecord
    await store.add_task(TaskRecord(task_type="index", document_id=document_id))
    assert client.delete(f"/documents/{document_id}").status_code == 409
    assert client.post(f"/documents/{document_id}/index").status_code == 409
    assert store.get_document(document_id).status == "parsed"


def test_rest_search_preserves_rerank_score(classroom, monkeypatch):
    client, _, _ = classroom
    from rag_project.retrieval.service import RetrievalResult, RetrievedChunk

    class Retriever:
        async def search(self, **kwargs):
            return RetrievalResult(query="q", filter_expr='kb_id == "kb"', matches=[
                RetrievedChunk(chunk_id="c", document_id="doc", score=0.8, rerank_score=0.95,
                    text="正文", source_uri=None, heading_path="", page_start=None, page_end=None, metadata={})])

    monkeypatch.setattr(api_routes, "get_retriever", Retriever)
    result = client.post("/retrieval/search", json={"kb_id":"kb", "query":"q"})
    assert result.json()["matches"][0]["rerank_score"] == 0.95


def test_import_index_experiment_delete_and_search_flow(classroom, monkeypatch):
    client, store, _ = classroom
    from rag_project.rerankers import NoopReranker
    from rag_project.retrieval import KnowledgeBaseRetriever
    from rag_project.vectorstores import MilvusSearchMatch

    class Embedding:
        config = SimpleNamespace(model="classroom-embed", dim=2)

        async def embed_documents(self, texts):
            return [[0.1, 0.2] for _ in texts]

        async def embed_query(self, query):
            return [0.1, 0.2]

    class Vector:
        chunks = []

        async def ensure_collection(self, **kwargs):
            assert kwargs["embedding_model"] == "classroom-embed"

        async def validate_embedding(self, **kwargs):
            assert kwargs == {"model":"classroom-embed", "dim":2}

        async def delete_document_chunks(self, **kwargs):
            self.chunks = []

        async def upsert_chunks(self, chunks, vectors, **kwargs):
            self.chunks = chunks

        async def search(self, **kwargs):
            return [MilvusSearchMatch(chunk_id=c.chunk_id, document_id=c.document_id, score=0.8,
                text=c.text, source_uri=c.source_uri, heading_path=c.heading_path,
                page_start=c.page_start, page_end=c.page_end, metadata=c.metadata) for c in self.chunks]

    vector = Vector()
    retriever = lambda: KnowledgeBaseRetriever(store=store, embedding_client_factory=Embedding,
        vector_store=vector, reranker=NoopReranker())
    monkeypatch.setattr(api_routes, "get_vector_store", lambda: vector)
    monkeypatch.setattr(api_routes, "get_embedding_client", Embedding)
    monkeypatch.setattr(api_routes, "get_retriever", retriever)
    monkeypatch.setattr(routes, "get_retriever", retriever)
    seed = client.post("/teaching/dataset/import", json={}).json()
    document_id = seed["documents"][1]["document_id"]
    kb_id = seed["knowledge_base"]["kb_id"]
    task_response = client.post(f"/documents/{document_id}/index")
    assert task_response.status_code == 202
    task = client.get(f'/tasks/{task_response.json()["task_id"]}').json()
    assert task["status"] == "succeeded"
    chunks = client.get(f"/documents/{document_id}/chunks").json()["chunks"]
    assert chunks and chunks[0]["source_uri"].startswith("teaching://")
    report = client.post("/teaching/experiments", json={"kb_id":kb_id, "query":"住宿上限", "rerank":False}).json()
    assert report["status"] == "succeeded" and report["matches"]
    assert report["rerank_status"] == "disabled"
    assert client.delete(f"/documents/{document_id}").status_code == 200
    assert vector.chunks  # Logical deletion: stale vectors intentionally remain in this fake.
    search = client.post("/retrieval/search", json={"kb_id":kb_id,"query":"住宿上限"})
    assert search.status_code == 200 and search.json()["matches"] == []
    assert client.get(f'/teaching/experiments/{report["id"]}').json()["matches"] == report["matches"]


@pytest.mark.parametrize("operation", ["parse", "ingest"])
def test_uploaded_file_processing_and_task_progress(classroom, monkeypatch, operation):
    from datetime import datetime, timezone
    from rag_project.graphs import IngestionGraph
    from rag_project.parsers import ParsedDocument

    client, store, _ = classroom
    kb = client.post("/knowledge-bases", json={"name":"上传实验", "metadata_schema":{"fields":[
        {"name":"year", "type":"int", "required":True, "filterable":True}]}}).json()
    upload = client.post(f'/knowledge-bases/{kb["kb_id"]}/documents',
        files={"file":("课程.pdf", b"%PDF classroom fixture", "application/pdf")}, data={"metadata":'{"year":2026}'})
    assert upload.status_code == 201
    doc_id = upload.json()["document_id"]
    progress = []

    class Parser:
        async def parse(self, file, options, progress_callback=None):
            assert file.content == b"%PDF classroom fixture"
            await progress_callback("raw_saved", {"raw_object_key":"raw/course.pdf"})
            await progress_callback("mineru_submitted", {"parser_task_id":"mineru-upload-test"})
            task = next(task for task in store.tasks.values() if task.document_id == doc_id)
            progress.append(task.result)
            assert store.get_document(doc_id).raw_object_key == "raw/course.pdf"
            return ParsedDocument(document_id=options.document_id, parser="mineru", parser_task_id="mineru-upload-test",
                markdown_text="# 课程资料\n\n提交作业需附实验报告。", markdown_object_key="parsed/course.md",
                raw_object_key="raw/course.pdf", parse_options=options.model_dump(mode="json"), created_at=datetime.now(timezone.utc))

    class Embedding:
        config = SimpleNamespace(model="classroom-embed", dim=2)

        async def embed_documents(self, texts):
            return [[0.1,0.2] for _ in texts]

    class Vector:
        async def ensure_collection(self, **kwargs):
            pass

        async def delete_document_chunks(self, **kwargs):
            pass

        async def upsert_chunks(self, chunks, vectors, **kwargs):
            assert chunks[0].metadata["year"] == 2026

    monkeypatch.setattr(api_routes, "get_parser", Parser)
    monkeypatch.setattr(api_routes, "get_ingestion_graph", lambda: IngestionGraph(store=store, parser=Parser(),
        embedding_client_factory=Embedding, vector_store=Vector()))
    response = client.post(f"/documents/{doc_id}/{operation}")
    assert response.status_code == 202
    task = client.get(f'/tasks/{response.json()["task_id"]}').json()
    assert task["status"] == "succeeded"
    assert progress[0]["stage"] == "mineru_submitted"
    assert progress[0]["raw_object_key"] == "raw/course.pdf"
    assert task["result"]["parser_task_id"] == "mineru-upload-test"
    record = client.get(f"/documents/{doc_id}").json()
    assert record["status"] == ("parsed" if operation == "parse" else "indexed")
    assert record["parsed_document"]["markdown_text"].startswith("# 课程资料")
    assert client.get(f'/knowledge-bases/{kb["kb_id"]}/tasks').json()[0]["task_id"] == task["task_id"]
    other = client.post("/knowledge-bases", json={"name":"other"}).json()
    assert client.get(f'/knowledge-bases/{other["kb_id"]}/tasks').json() == []
    assert client.get("/knowledge-bases/missing/tasks").status_code == 404


def test_failed_ingestion_can_retry_without_reuploading(classroom, monkeypatch):
    from rag_project.graphs import IngestionGraph
    client, store, _ = classroom
    kb = client.post("/knowledge-bases", json={"name":"失败重试"}).json()
    document = client.post(f'/knowledge-bases/{kb["kb_id"]}/documents',
        files={"file":("failure.pdf", b"%PDF", "application/pdf")}).json()

    class Parser:
        async def parse(self, file, options, progress_callback=None):
            await progress_callback("mineru_submitted", {"parser_task_id":"failed-parser"})
            raise RuntimeError("MinerU unavailable")

    monkeypatch.setattr(api_routes, "get_ingestion_graph", lambda: IngestionGraph(store=store, parser=Parser(),
        embedding_client_factory=None, vector_store=None))
    first = client.post(f'/documents/{document["document_id"]}/ingest').json()
    first_status = client.get(f'/tasks/{first["task_id"]}').json()
    assert first_status["status"] == "failed"
    assert first_status["result"]["failed_node"] == "parse_with_mineru"
    assert first_status["result"]["parser_task_id"] == "failed-parser"
    second = client.post(f'/documents/{document["document_id"]}/ingest')
    assert second.status_code == 202
    assert second.json()["task_id"] != first["task_id"]
    assert len(store.list_documents(kb["kb_id"])) == 1
    assert len(client.get(f'/knowledge-bases/{kb["kb_id"]}/tasks').json()) == 2


@pytest.mark.parametrize("operation,factory", [("parse","get_parser"), ("ingest","get_ingestion_graph")])
def test_configuration_failure_does_not_leave_pending_tasks(classroom, monkeypatch, operation, factory):
    client, store, _ = classroom
    kb = client.post("/knowledge-bases", json={"name":"配置失败"}).json()
    document = client.post(f'/knowledge-bases/{kb["kb_id"]}/documents', files={"file":("a.pdf",b"%PDF")}).json()
    def broken():
        raise ValueError("missing service configuration")
    monkeypatch.setattr(api_routes, factory, broken)
    assert client.post(f'/documents/{document["document_id"]}/{operation}').status_code == 503
    assert store.get_document(document["document_id"]).status == "uploaded"
    assert store.tasks == {}


@pytest.mark.asyncio
async def test_delete_knowledge_base_cascades_and_preserves_reports(classroom):
    from rag_project.api.schemas import TaskRecord
    from rag_project.retrieval.service import KnowledgeBaseRetriever
    client, store, factory = classroom
    seed = client.post('/teaching/dataset/import', json={}).json()
    other = client.post('/teaching/dataset/import', json={}).json()
    kb_id = seed['knowledge_base']['kb_id']
    doc_id = seed['documents'][0]['document_id']
    await store.replace_document_chunks(doc_id, [ChunkRecord(chunk_id='delete-chunk', kb_id=kb_id,
        document_id=doc_id, chunk_index=0, text='test')])
    await store.add_task(TaskRecord(task_id='delete-task', task_type='index', document_id=doc_id, status='succeeded'))
    reports = ReportStore(factory)
    reports.save({'id':'preserved-report', 'knowledge_base':seed['knowledge_base'], 'documents':seed['documents']})
    assert client.patch(f'/knowledge-bases/{kb_id}', json={'name':'改名', 'description':'课堂资料'}).json()['name'] == '改名'
    assert client.delete(f'/knowledge-bases/{kb_id}').status_code == 200
    assert client.get(f'/knowledge-bases/{kb_id}').status_code == 404
    assert client.get(f'/knowledge-bases/{kb_id}/documents').status_code == 404
    assert client.get(f'/documents/{doc_id}').status_code == 404
    assert store.list_document_chunks(doc_id) == []
    assert store.get_task('delete-task') is None
    assert reports.get('preserved-report')['documents'] == seed['documents']
    assert len(store.list_documents(other['knowledge_base']['kb_id'])) == 5
    assert client.delete(f'/knowledge-bases/{kb_id}').status_code == 404
    retriever = KnowledgeBaseRetriever(store=store, embedding_client_factory=lambda: None, vector_store=None, reranker=None)
    with pytest.raises(KeyError):
        retriever.build_filter_expr(kb_id=kb_id, filters={})
    with pytest.raises(ValueError):
        await store.add_task(TaskRecord(task_type='parse', document_id=doc_id))


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['pending', 'running'])
async def test_delete_knowledge_base_rejects_active_tasks_atomically(classroom, status):
    from rag_project.api.schemas import TaskRecord
    client, store, _ = classroom
    seed = client.post('/teaching/dataset/import', json={}).json()
    kb_id = seed['knowledge_base']['kb_id']
    await store.add_task(TaskRecord(task_id='active', task_type='parse', status=status,
                                   document_id=seed['documents'][0]['document_id']))
    assert client.delete(f'/knowledge-bases/{kb_id}').status_code == 409
    assert len(store.list_documents(kb_id)) == 5
    assert store.get_knowledge_base(kb_id) is not None
    await store.update_task('active', status='failed')
    assert client.delete(f'/knowledge-bases/{kb_id}').status_code == 200


@pytest.mark.asyncio
async def test_auto_intent_experiment_uses_rewrite_but_answers_original(classroom, monkeypatch):
    import json
    client, store, factory = classroom
    seed = client.post('/teaching/dataset/import',json={}).json()
    kb_id = seed['knowledge_base']['kb_id']
    await store.update_document(seed['documents'][0]['document_id'],status='indexed')
    original = '请问2025年财务处住宿能报多少？'
    rewritten = '财务处2025年差旅住宿报销上限'
    class Chat:
        config = SimpleNamespace(model='test-intent',base_url=None,api_key='EMPTY')
        async def generate_text(self,prompt):
            if 'RAG Intent Agent' in prompt:
                return json.dumps({'kb_id':kb_id,'retrieval_query':rewritten,
                    'filters':{'year':2025,'department':'财务处'},'rationale':'指定部门和年份',
                    'needs_clarification':False,'clarification_question':None})
            assert original in prompt
            return '依据片段回答'
    class Retriever:
        def build_filter_expr(self,**kwargs):
            assert kwargs == {'kb_id':kb_id,'filters':{'year':2025,'department':'财务处'}}
            return 'kb_id == "test"'
        async def retrieve_candidates(self,**kwargs):
            assert kwargs['query'] == rewritten
            return [Document(page_content='住宿标准',metadata={'chunk_id':'c1'})]
        async def rerank_documents(self,**kwargs):
            assert kwargs['query'] == original
            return kwargs['documents'],None
    monkeypatch.setattr(routes,'get_chat_client',Chat)
    monkeypatch.setattr(routes,'get_retriever',Retriever)
    response = client.post('/teaching/experiments',json={'auto_intent':True,'query':original,'mode':'chat'})
    report = response.json()
    assert report['status'] == 'succeeded', report.get('error')
    assert report['request']['query'] == original
    assert report['effective_request']['retrieval_query'] == rewritten
    assert report['intent']['kb_id'] == kb_id
    assert report['models']['intent'] == 'test-intent'
    assert 'intent' in report['timings_ms']
    assert ReportStore(factory).get(report['id']) == report


def test_intent_clarification_and_failures_are_saved_without_retrieval(classroom,monkeypatch):
    import json
    client, store, _ = classroom
    client.post('/teaching/dataset/import',json={})
    class Chat:
        config = SimpleNamespace(model='test-intent')
        async def generate_text(self,prompt):
            return json.dumps({'kb_id':None,'retrieval_query':'申请流程','filters':{},'rationale':'范围不明确',
                'needs_clarification':True,'clarification_question':'请说明需要申请什么？'})
    monkeypatch.setattr(routes,'get_chat_client',Chat)
    monkeypatch.setattr(routes,'get_retriever',lambda:pytest.fail('must not retrieve'))
    report = client.post('/teaching/experiments',json={'auto_intent':True,'query':'怎么申请？'}).json()
    assert report['status'] == 'failed' and report['failed_stage'] == 'intent'
    assert report['intent']['needs_clarification']
    assert '申请什么' in report['error']
    conflict = client.post('/teaching/experiments',json={'auto_intent':True,'query':'报销','filters':{'year':2025}}).json()
    assert conflict['status'] == 'failed' and conflict['failed_stage'] == 'intent'


@pytest.mark.asyncio
async def test_intent_api_search_and_chat_use_validated_routing(classroom,monkeypatch):
    import json
    from rag_project.retrieval.intent import IntentAgent
    from rag_project.graphs.qa import QAGraph
    client,store,_ = classroom
    seed = client.post('/teaching/dataset/import',json={}).json()
    kb_id = seed['knowledge_base']['kb_id']
    original, rewritten = '帮我找2025年的住宿标准', '2025年差旅住宿报销上限'
    class Chat:
        config = SimpleNamespace(model='test-intent')
        async def generate_text(self,prompt):
            return json.dumps({'kb_id':kb_id,'retrieval_query':rewritten,'filters':{'year':2025},
                'rationale':'指定年份','needs_clarification':False,'clarification_question':None})
        async def generate_answer(self,*,query,documents):
            assert query == original
            return 'answer'
    class Retriever:
        async def search(self,**kwargs):
            assert kwargs['query'] == original and kwargs['retrieval_query'] == rewritten
            assert kwargs['kb_id'] == kb_id and kwargs['filters'] == {'year':2025}
            return SimpleNamespace(query=original,filter_expr='safe',matches=[],rerank_error=None)
        def build_filter_expr(self,**kwargs):
            assert kwargs == {'kb_id':kb_id,'filters':{'year':2025}}
            return 'safe'
        async def retrieve_candidates(self,**kwargs):
            assert kwargs['query'] == rewritten
            return []
        async def rerank_documents(self,**kwargs):
            assert kwargs['query'] == original
            return [],None
    monkeypatch.setattr(api_routes,'get_intent_agent',lambda:IntentAgent(store=store,chat_factory=Chat))
    monkeypatch.setattr(api_routes,'get_retriever',Retriever)
    monkeypatch.setattr(api_routes,'get_qa_graph',lambda:QAGraph(retriever=Retriever(),chat_client=Chat()))
    plan = client.post('/intent/plan',json={'query':original})
    assert plan.status_code == 200 and plan.json()['kb_id'] == kb_id
    for path in ('/retrieval/search','/chat'):
        response = client.post(path,json={'auto_intent':True,'query':original})
        assert response.status_code == 200, response.text
        assert response.json()['query'] == original
        assert response.json()['intent']['retrieval_query'] == rewritten
        assert client.post(path,json={'query':original}).status_code == 422
