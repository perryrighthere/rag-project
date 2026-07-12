from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag_project.api.routes import router
from rag_project.core.config import get_settings
from rag_project.db import create_db_and_tables, get_store
from rag_project.services.task_recovery import mark_interrupted_tasks_failed


def create_app() -> FastAPI:
    settings = get_settings()
    create_db_and_tables()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await mark_interrupted_tasks_failed(get_store())
        yield

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.include_router(router, prefix=settings.api_prefix)
    return app


app = create_app()
