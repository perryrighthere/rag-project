import json
from types import SimpleNamespace

import pytest

from rag_project.api.schemas import DocumentRecord, KnowledgeBaseRecord
from rag_project.knowledge_base import MetadataSchema, MetadataValidationError
from rag_project.retrieval.filters import MilvusFilterBuilder
from rag_project.retrieval.intent import IntentAgent, IntentError, build_catalog
from rag_project.services.memory_store import InMemoryStore


@pytest.fixture
async def store():
    store = InMemoryStore()
    schema = MetadataSchema(fields=[
        {'name':'department','type':'string','filterable':True},
        {'name':'year','type':'int','filterable':True},
        {'name':'tags','type':'string_array','filterable':True},
        {'name':'private_note','type':'string','filterable':False}])
    await store.add_knowledge_base(KnowledgeBaseRecord(kb_id='finance', name='财务制度', metadata_schema=schema))
    await store.add_knowledge_base(KnowledgeBaseRecord(kb_id='library', name='图书馆规定'))
    await store.add_document(DocumentRecord(kb_id='finance', document_id='d1', filename='travel.pdf',
        status='indexed', metadata={'department':'财务处','year':2025,'tags':['教学'],'private_note':'not-in-prompt'}))
    await store.add_document(DocumentRecord(kb_id='finance', filename='deleted.pdf', status='deleted',
        metadata={'department':'deleted-category'}))
    return store


def decision(**changes):
    return {'kb_id':'finance','retrieval_query':'财务处2025年差旅住宿报销标准',
            'filters':{'department':'财务处','year':2025},'rationale':'问题明确指定财务处和2025年',
            'needs_clarification':False,'clarification_question':None, **changes}


class Chat:
    config = SimpleNamespace(model='test-intent')

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []

    async def generate_text(self, prompt):
        self.prompts.append(prompt)
        return next(self.outputs)


@pytest.mark.asyncio
async def test_intent_reads_all_catalogs_and_validates_plan(store):
    chat = Chat([json.dumps(decision(),ensure_ascii=False)])
    plan = await IntentAgent(store=store,chat_factory=lambda:chat).plan('请问财务处2025年出差住宿能报多少？')
    assert plan.kb_id == 'finance' and plan.filters['year'] == 2025
    assert plan.original_query != plan.retrieval_query
    assert plan.model == 'test-intent'
    assert {kb['kb_id'] for kb in plan.catalog} == {'finance','library'}
    assert '图书馆规定' in chat.prompts[0] and '财务处' in chat.prompts[0]
    assert 'not-in-prompt' not in chat.prompts[0] and 'deleted-category' not in chat.prompts[0]
    assert plan.require_ready() is plan


@pytest.mark.asyncio
@pytest.mark.parametrize('bad', [
    'not-json', json.dumps(decision(kb_id='invented')),
    json.dumps(decision(filters={'year':'2025'})),
    json.dumps(decision(filters={'kb_id':'library'})),
    json.dumps(decision(filters={'private_note':'secret'})),
    json.dumps(decision(filters={'$or':[{}]})),
    json.dumps(decision(filters={'year':None})),
    json.dumps(decision(filters={'tags':['教学']})),
    json.dumps(decision(filters={'year':{'$raw':'1 == 1'}})),
    json.dumps({**decision(),'expr':'kb_id != "finance"'}),
])
async def test_invalid_model_output_never_becomes_a_filter(store,bad):
    chat = Chat([bad,bad])
    with pytest.raises(IntentError,match='连续返回无效'):
        await IntentAgent(store=store,chat_factory=lambda:chat).plan('报销')
    assert len(chat.prompts) == 2


@pytest.mark.asyncio
async def test_intent_repairs_once_and_handles_clarification(store):
    chat = Chat(['bad', '```json\n'+json.dumps(decision())+'\n```'])
    plan = await IntentAgent(store=store,chat_factory=lambda:chat).plan('财务处2025年报销')
    assert plan.attempts == 2
    chat = Chat([json.dumps(decision(kb_id=None,filters={},needs_clarification=True,clarification_question='您指哪个部门？'))])
    plan = await IntentAgent(store=store,chat_factory=lambda:chat).plan('怎么申请？')
    with pytest.raises(IntentError,match='哪个部门'):
        plan.require_ready()


@pytest.mark.asyncio
async def test_missing_catalog_and_service_fail_closed(store):
    with pytest.raises(IntentError,match='没有知识库'):
        await IntentAgent(store=InMemoryStore(),chat_factory=lambda:None).plan('报销')
    with pytest.raises(IntentError,match='调用失败'):
        await IntentAgent(store=store,chat_factory=lambda:None).plan('报销')


@pytest.mark.parametrize('filters', [{'year':None},{'$or':[{}]},{'tags':{'$eq':['教学']}},{'year':float('inf')}])
@pytest.mark.asyncio
async def test_filter_boundary_rejects_ambiguous_values(store,filters):
    with pytest.raises(MetadataValidationError):
        MilvusFilterBuilder().build(kb_id='finance',metadata_schema=store.get_knowledge_base('finance').metadata_schema,filters=filters)
