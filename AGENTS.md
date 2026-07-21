# Repository Guide

## Commands

- Use Python 3.12. Set up with `python -m venv .venv && . .venv/bin/activate && python -m pip install -r requirements.txt`.
- Run the API from the repository root with `PYTHONPATH=src uvicorn rag_project.main:app --reload`.
- Start the app plus PostgreSQL, MinIO, Milvus, etcd, and Attu with `docker compose -f docker-compose.yml -f docker-compose.local.yml up --build`.
- Run the isolated unit suite with `DATABASE_URL='sqlite+pysqlite:///:memory:' python -m pytest`. The override matters: API tests use the cached SQLAlchemy store and otherwise load the developer's `.env` database.
- Run one test with the same override, for example `DATABASE_URL='sqlite+pysqlite:///:memory:' python -m pytest tests/test_graphs.py::test_ingestion_graph_indexes_document_with_in_memory_store`.
- Install `requirements-agentic.txt` only for the optional `crewai` or `autogen` QA orchestrators; `single` and `langgraph_multi` do not need it.
- There is no configured lint, formatter, typecheck, codegen, pre-commit, or CI command. Do not invent one as required verification.

## Runtime Shape

- `src/rag_project/main.py` is the runtime entrypoint. App creation immediately runs SQLAlchemy `create_all`; there is no Alembic configuration or migration tree despite Alembic being a dependency and appearing in architecture prose.
- `api/routes.py` owns HTTP orchestration and the separate parse/index background-task paths. `graphs/ingestion.py` powers only the combined `/documents/{document_id}/ingest` flow; `graphs/qa.py` powers `/chat`.
- Production persistence is `db/store.py` through `services/store.py`'s contract. `services/memory_store.py` is a test fake; its docstring describing it as the development store is stale.
- External-service construction is centralized in `api/dependencies.py`. Settings and several clients are `lru_cache`d, so tests that change environment variables after import must clear the relevant caches or inject fakes before constructing the app.
- Unit tests fake MinerU, MinIO, model APIs, and Milvus and require no running services. The local Compose stack still does not provide MinerU or OpenAI-compatible embedding/chat/rerank endpoints.

## Behavioral Constraints

- Parse, index, and ingest jobs use FastAPI in-process `BackgroundTasks`. Startup deliberately marks leftover `pending`/`running` tasks and their documents failed; do not imply resumability without changing this design.
- Metadata filters are a security boundary: callers provide structured filters, `retrieval/filters.py` validates them against the knowledge-base schema, and generated Milvus expressions must always include `kb_id` isolation. Never accept raw Milvus expressions.
- Reranking is optional and failure intentionally falls back to vector-search order. Embedding model and dimension are recorded on knowledge bases, documents, and chunks and must stay consistent with Milvus writes.
- Document deletion is currently logical only; it does not remove MinIO objects or Milvus chunks.
- Prefer the two Compose files for local Milvus. `standalone_embed.sh` uses `sudo`, a different Milvus version, and its `delete` action removes local data.

## Project Guidance

- `docs/technical_architecture.md` is a design blueprint and contains stale planned names/defaults; current code, tests, `.env.example`, and Compose files win when they disagree.
- For domain changes, consult the matching repo guide: `skills/rag-ingestion-builder/SKILL.md`, `skills/rag-retrieval-builder/SKILL.md`, or `skills/rag-project-architect/SKILL.md`. Preserve their verified module boundaries and retrieval safety rules, but check planned details against current code.
