from fastapi import FastAPI

from rag_project.api.routes import router
from rag_project.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.include_router(router, prefix=settings.api_prefix)
    return app


app = create_app()

