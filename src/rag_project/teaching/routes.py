from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from rag_project.api.dependencies import get_chat_client, get_retriever
from rag_project.chunking import ChunkingConfig, MarkdownChunker
from rag_project.core.config import get_settings
from rag_project.db import get_store
from rag_project.db.session import get_session_factory
from rag_project.teaching.dataset import import_dataset, load_dataset
from rag_project.teaching.experiments import ExperimentRequest, run_experiment
from rag_project.teaching.reports import ReportStore, render_markdown

router = APIRouter(prefix="/teaching", tags=["teaching"])


class DatasetImportRequest(BaseModel):
    chunking_config: ChunkingConfig = Field(default_factory=ChunkingConfig)


class PreviewRequest(DatasetImportRequest):
    document_id: str


def report_store():
    return ReportStore(get_session_factory())


@router.get("/dataset")
async def dataset():
    return load_dataset()


@router.post("/dataset/import", status_code=201)
async def seed(payload: DatasetImportRequest):
    return await import_dataset(get_store(), payload.chunking_config)


@router.post("/preview")
async def preview(payload: PreviewRequest):
    data = load_dataset()
    document = next((item for item in data["documents"] if item["id"] == payload.document_id), None)
    if document is None:
        raise HTTPException(404, "教学文档不存在")
    chunks = MarkdownChunker(payload.chunking_config).chunk_markdown(document["markdown"], kb_id="preview",
        document_id=document["id"], document_metadata=document["metadata"],
        source_uri=f"teaching://{data['id']}/{data['version']}/{document['id']}")
    return {"chunks": chunks, "chunking_config": payload.chunking_config}


@router.post("/experiments", status_code=201)
async def experiment(payload: ExperimentRequest):
    return await run_experiment(payload, store=get_store(), reports=report_store(),
        retriever_factory=get_retriever, chat_factory=get_chat_client, settings=get_settings())


@router.get("/experiments")
async def reports():
    return report_store().list()


@router.get("/experiments/{report_id}")
async def report(report_id: str):
    record = report_store().get(report_id)
    if record is None:
        raise HTTPException(404, "实验报告不存在")
    return record


@router.get("/experiments/{report_id}/markdown", response_class=PlainTextResponse)
async def download_report(report_id: str):
    record = await report(report_id)
    return PlainTextResponse(render_markdown(record), media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{record["id"]}.md"'})
