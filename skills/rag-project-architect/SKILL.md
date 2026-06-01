---
name: rag-project-architect
description: Guide architecture, repository structure, implementation sequencing, and technical consistency for this RAG project. Use when planning, reviewing, scaffolding, or refactoring the FastAPI, LangChain, LangGraph, MinerU, MinIO, Milvus, metadata, ingestion, retrieval, and QA modules in /Users/perryhe/Projects/rag-project.
---

# RAG Project Architect

## Core Rule

Keep the project architecture aligned with `docs/technical_architecture.md`. Use LangChain for model, document, splitter, retriever, prompt, and Runnable abstractions. Use LangGraph for stateful ingestion and QA workflows. Keep MinerU parsing, metadata schema, Milvus filter building, document state, and storage layout under project-owned business code.

## Before Implementing

1. Read `docs/technical_architecture.md` when the task touches architecture, module boundaries, dependencies, or implementation order.
2. For concise architecture constraints and checklists, read `references/architecture-boundaries.md`.
3. Check existing files before adding new abstractions. Follow the established layout once code exists.
4. Keep edits scoped to the requested milestone or module.

## Module Boundaries

- Put API routes under `src/rag_project/api/`.
- Put settings, logging, and shared errors under `src/rag_project/core/`.
- Put SQLAlchemy models, sessions, and repositories under `src/rag_project/db/`.
- Put MinIO object handling under `src/rag_project/storage/`.
- Put MinerU integration under `src/rag_project/parsers/`.
- Put knowledge base, metadata schema, and document state under `src/rag_project/knowledge_base/`.
- Put chunking under `src/rag_project/chunking/`.
- Put embedding clients under `src/rag_project/embeddings/`.
- Put Milvus schema and adapters under `src/rag_project/vectorstores/`.
- Put filter building, retrieval, context formatting, and rerank orchestration under `src/rag_project/retrieval/` and `src/rag_project/rerankers/`.
- Put LangGraph graphs and states under `src/rag_project/graphs/`.

## Implementation Order

Prefer this order unless the user asks otherwise:

1. Project skeleton, settings, logging, health check.
2. Database models and migrations for knowledge bases, documents, chunks, and ingestion tasks.
3. Metadata schema validation.
4. MinIO storage client and object key conventions.
5. MinerU parser adapter.
6. Chunking and LangChain `Document` conversion.
7. Embedding and Milvus indexing.
8. Metadata filter builder.
9. Retrieval and rerank.
10. LangGraph ingestion graph.
11. LangGraph QA graph.
12. Observability, tests, and evaluation fixtures.

## Quality Bar

- Keep business contracts explicit with typed models.
- Do not let clients pass raw Milvus expressions.
- Record model name, embedding dimension, parser options, and document status.
- Prefer async I/O for MinerU, MinIO, model APIs, and graph execution where practical.
- Add focused tests for schema validation, filter building, chunking, and graph routing.
- Update `docs/technical_architecture.md` when architecture decisions change.
