from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from rag_project.api.dependencies import get_qa_graph, get_retriever
from rag_project.api.routes import router
from rag_project.core.config import get_settings
from rag_project.db import create_db_and_tables, get_store
from rag_project.mcp import build_mcp_server
from rag_project.services.task_recovery import mark_interrupted_tasks_failed
from rag_project.teaching.routes import router as teaching_router


def create_app() -> FastAPI:
    settings = get_settings()
    create_db_and_tables()

    mcp_server = build_mcp_server(
        retriever_factory=get_retriever,
        qa_graph_factory=get_qa_graph,
        store_factory=get_store,
    )
    mcp_app = mcp_server.streamable_http_app()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with mcp_server.session_manager.run():
            await mark_interrupted_tasks_failed(get_store())
            yield

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.include_router(router, prefix=settings.api_prefix)
    app.include_router(teaching_router, prefix=settings.api_prefix)
    app.mount(f"{settings.api_prefix}/workbench", StaticFiles(
        directory=Path(__file__).parent / "teaching" / "static", html=True), name="workbench")
    app.mount("/mcp", mcp_app)
    return app


app = create_app()
