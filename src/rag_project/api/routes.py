import json

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status

from rag_project.api.dependencies import get_embedding_client, get_parser, get_vector_store
from rag_project.api.schemas import (
    ChatRequest,
    DocumentChunksResponse,
    DocumentRecord,
    KnowledgeBaseCreate,
    KnowledgeBaseRecord,
    KnowledgeBaseUpdate,
    MetadataSchema,
    RetrievalMatch,
    RetrievalSearchRequest,
    RetrievalSearchResponse,
    TaskRecord,
)
from rag_project.chunking import MarkdownChunker
from rag_project.core.config import get_settings
from rag_project.embeddings import OpenAICompatibleEmbeddingClient
from rag_project.knowledge_base import MetadataValidationError
from rag_project.parsers import MinerUApiParser, ParseOptions, UploadedFile as ParserUploadedFile
from rag_project.retrieval import MilvusFilterBuilder
from rag_project.services.memory_store import store
from rag_project.vectorstores import MilvusVectorStoreAdapter


router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/knowledge-bases", response_model=KnowledgeBaseRecord, status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(payload: KnowledgeBaseCreate) -> KnowledgeBaseRecord:
    return await store.add_knowledge_base(KnowledgeBaseRecord(**payload.model_dump()))


@router.get("/knowledge-bases", response_model=list[KnowledgeBaseRecord])
async def list_knowledge_bases() -> list[KnowledgeBaseRecord]:
    return list(store.knowledge_bases.values())


@router.get("/knowledge-bases/{kb_id}", response_model=KnowledgeBaseRecord)
async def get_knowledge_base(kb_id: str) -> KnowledgeBaseRecord:
    record = store.knowledge_bases.get(kb_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found")
    return record


@router.patch("/knowledge-bases/{kb_id}", response_model=KnowledgeBaseRecord)
async def update_knowledge_base(kb_id: str, payload: KnowledgeBaseUpdate) -> KnowledgeBaseRecord:
    changes = payload.model_dump(exclude_unset=True)
    record = await store.update_knowledge_base(kb_id, **changes)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found")
    return record


@router.patch("/knowledge-bases/{kb_id}/metadata-schema", response_model=KnowledgeBaseRecord)
async def update_metadata_schema(kb_id: str, payload: MetadataSchema) -> KnowledgeBaseRecord:
    record = await store.update_knowledge_base(kb_id, metadata_schema=payload)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found")
    return record


@router.post("/knowledge-bases/{kb_id}/documents", response_model=DocumentRecord, status_code=status.HTTP_201_CREATED)
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    metadata: str = Form(default="{}"),
) -> DocumentRecord:
    knowledge_base = store.knowledge_bases.get(kb_id)
    if knowledge_base is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found")
    try:
        parsed_metadata = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="metadata must be a JSON object") from exc
    if not isinstance(parsed_metadata, dict):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="metadata must be a JSON object")
    try:
        knowledge_base.metadata_schema.validate_document_metadata(parsed_metadata)
    except MetadataValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    content = await file.read()
    record = DocumentRecord(
        kb_id=kb_id,
        filename=file.filename or "document",
        content_type=file.content_type,
        metadata=parsed_metadata,
        file_content=content,
    )
    return await store.add_document(record)


@router.get("/documents/{document_id}", response_model=DocumentRecord)
async def get_document(document_id: str) -> DocumentRecord:
    record = store.documents.get(document_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return record


@router.get("/documents/{document_id}/chunks", response_model=DocumentChunksResponse)
async def list_document_chunks(document_id: str) -> DocumentChunksResponse:
    if document_id not in store.documents:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return DocumentChunksResponse(document_id=document_id, chunks=store.list_document_chunks(document_id))


@router.delete("/documents/{document_id}", response_model=DocumentRecord)
async def delete_document(document_id: str) -> DocumentRecord:
    record = await store.update_document(document_id, status="deleted")
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return record


@router.post("/documents/{document_id}/parse", response_model=TaskRecord, status_code=status.HTTP_202_ACCEPTED)
async def parse_document(document_id: str, background_tasks: BackgroundTasks) -> TaskRecord:
    document = store.documents.get(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if document.status == "deleted":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Deleted documents cannot be parsed")
    if document.file_content is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Document file content is unavailable")

    task = await store.add_task(TaskRecord(task_type="parse", document_id=document_id))
    await store.update_document(document_id, status="parsing", error=None)
    background_tasks.add_task(_run_parse_task, task.task_id, document_id, get_parser())
    return task


@router.post("/documents/{document_id}/index", response_model=TaskRecord, status_code=status.HTTP_202_ACCEPTED)
async def index_document(document_id: str, background_tasks: BackgroundTasks) -> TaskRecord:
    document = store.documents.get(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if document.status == "deleted":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Deleted documents cannot be indexed")
    if document.parsed_document is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Document must be parsed before indexing")

    task = await store.add_task(TaskRecord(task_type="index", document_id=document_id))
    await store.update_document(document_id, status="chunking", error=None)
    background_tasks.add_task(
        _run_index_task,
        task.task_id,
        document_id,
        get_embedding_client,
        get_vector_store(),
    )
    return task


@router.post("/documents/{document_id}/reindex", response_model=TaskRecord, status_code=status.HTTP_202_ACCEPTED)
async def reindex_document(document_id: str, background_tasks: BackgroundTasks) -> TaskRecord:
    document = store.documents.get(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if document.parsed_document is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Document must be parsed before reindexing")
    return await index_document(document_id, background_tasks)


@router.get("/tasks/{task_id}", response_model=TaskRecord)
async def get_task(task_id: str) -> TaskRecord:
    record = store.tasks.get(task_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    return record


@router.post("/retrieval/search", response_model=RetrievalSearchResponse)
async def search(payload: RetrievalSearchRequest) -> RetrievalSearchResponse:
    knowledge_base = store.knowledge_bases.get(payload.kb_id)
    if knowledge_base is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found")
    try:
        filter_expr = MilvusFilterBuilder().build(
            kb_id=payload.kb_id,
            metadata_schema=knowledge_base.metadata_schema,
            filters=payload.filters,
        )
        query_vector = await get_embedding_client().embed_query(payload.query)
        matches = await get_vector_store().search(
            query_vector=query_vector,
            filter_expr=filter_expr,
            top_k=payload.top_k,
        )
    except MetadataValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return RetrievalSearchResponse(
        query=payload.query,
        filter_expr=filter_expr,
        matches=[
            RetrievalMatch(
                chunk_id=match.chunk_id,
                document_id=match.document_id,
                score=match.score,
                text=match.text,
                source_uri=match.source_uri,
                heading_path=match.heading_path,
                page_start=match.page_start,
                page_end=match.page_end,
                metadata=match.metadata,
            )
            for match in matches
        ],
    )


@router.post("/chat")
async def chat(_: ChatRequest) -> None:
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="QA graph belongs to architecture section 4.11")


async def _run_parse_task(task_id: str, document_id: str, parser: MinerUApiParser) -> None:
    document = store.documents[document_id]
    assert document.file_content is not None
    await store.update_task(task_id, status="running")
    try:
        parsed = await parser.parse(
            ParserUploadedFile(
                filename=document.filename,
                content=document.file_content,
                content_type=document.content_type,
            ),
            ParseOptions(kb_id=document.kb_id, document_id=document.document_id),
        )
        await store.update_document(
            document_id,
            status="parsed",
            raw_object_key=parsed.raw_object_key,
            parsed_document=parsed,
            error=None,
        )
        await store.update_task(task_id, status="succeeded", result=parsed.model_dump(mode="json"))
    except Exception as exc:
        await store.update_document(document_id, status="failed", error=str(exc))
        await store.update_task(task_id, status="failed", error=str(exc))


async def _run_index_task(
    task_id: str,
    document_id: str,
    embedding_client_factory,
    vector_store: MilvusVectorStoreAdapter,
) -> None:
    document = store.documents[document_id]
    knowledge_base = store.knowledge_bases[document.kb_id]
    assert document.parsed_document is not None

    await store.update_task(task_id, status="running")
    try:
        source_uri = _source_uri(document.parsed_document.markdown_object_key)
        chunker = MarkdownChunker(knowledge_base.chunking_config)
        chunks = chunker.chunk_parsed_document(
            document.parsed_document,
            kb_id=document.kb_id,
            document_id=document.document_id,
            document_metadata=document.metadata,
            source_uri=source_uri,
        )
        await store.update_document(document_id, status="embedding")
        embedding_client: OpenAICompatibleEmbeddingClient = embedding_client_factory()
        vectors = await embedding_client.embed_documents([chunk.text for chunk in chunks])
        chunks = [
            chunk.model_copy(
                update={
                    "embedding_model": embedding_client.config.model,
                    "embedding_dim": embedding_client.config.dim,
                }
            )
            for chunk in chunks
        ]

        await store.update_document(document_id, status="embedding")
        await vector_store.delete_document_chunks(kb_id=document.kb_id, document_id=document.document_id)
        await vector_store.upsert_chunks(
            chunks,
            vectors,
            metadata_schema=knowledge_base.metadata_schema,
            embedding_dim=embedding_client.config.dim,
        )
        chunks = await store.replace_document_chunks(document_id, chunks)
        await store.update_knowledge_base(
            document.kb_id,
            embedding_model=embedding_client.config.model,
            embedding_dim=embedding_client.config.dim,
        )
        await store.update_document(
            document_id,
            status="indexed",
            chunk_count=len(chunks),
            embedding_model=embedding_client.config.model,
            embedding_dim=embedding_client.config.dim,
            error=None,
        )
        await store.update_task(
            task_id,
            status="succeeded",
            result={"chunk_count": len(chunks), "embedding_model": embedding_client.config.model},
        )
    except Exception as exc:
        await store.update_document(document_id, status="failed", error=str(exc))
        await store.update_task(task_id, status="failed", error=str(exc))


def _source_uri(object_key: str) -> str:
    settings = get_settings()
    return f"minio://{settings.minio_bucket}/{object_key}"
