from fastapi import FastAPI

from rag_project.api.routes import router
from rag_project.core.config import get_settings
from rag_project.db import create_db_and_tables


def create_app() -> FastAPI:
    settings = get_settings()
    create_db_and_tables()
    app = FastAPI(title=settings.app_name)
    app.include_router(router, prefix=settings.api_prefix)
    return app


app = create_app()
