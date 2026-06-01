import json

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status

from rag_project.api.dependencies import get_parser
from rag_project.api.schemas import (
    ChatRequest,
    DocumentRecord,
    KnowledgeBaseCreate,
    KnowledgeBaseRecord,
    KnowledgeBaseUpdate,
    MetadataSchema,
    RetrievalSearchRequest,
    TaskRecord,
)
from rag_project.parsers import MinerUApiParser, ParseOptions, UploadedFile as ParserUploadedFile
from rag_project.services.memory_store import store


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
    if kb_id not in store.knowledge_bases:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found")
    try:
        parsed_metadata = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="metadata must be a JSON object") from exc
    if not isinstance(parsed_metadata, dict):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="metadata must be a JSON object")

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


@router.post("/documents/{document_id}/index")
async def index_document(document_id: str) -> None:
    if document_id not in store.documents:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Indexing belongs to architecture section 4.5-4.7")


@router.post("/documents/{document_id}/reindex")
async def reindex_document(document_id: str) -> None:
    if document_id not in store.documents:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Reindexing belongs to architecture section 4.5-4.7")


@router.get("/tasks/{task_id}", response_model=TaskRecord)
async def get_task(task_id: str) -> TaskRecord:
    record = store.tasks.get(task_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    return record


@router.post("/retrieval/search")
async def search(_: RetrievalSearchRequest) -> None:
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Retrieval belongs to architecture section 4.8-4.10")


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

