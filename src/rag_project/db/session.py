from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from rag_project.core.config import get_settings
from rag_project.db.models import Base
from rag_project.db.store import SQLAlchemyStore


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    kwargs = {"pool_pre_ping": True}
    if settings.database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if settings.database_url.endswith(":memory:"):
            kwargs["poolclass"] = StaticPool
    return create_engine(settings.database_url, **kwargs)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


def create_db_and_tables() -> None:
    Base.metadata.create_all(bind=get_engine())


@lru_cache
def get_store() -> SQLAlchemyStore:
    create_db_and_tables()
    return SQLAlchemyStore(get_session_factory())
