---
name: rag-retrieval-builder
description: Build and maintain metadata-filtered retrieval, rerank, and QA workflows for this RAG project. Use when implementing knowledge-base search, Milvus filter expression generation, LangChain retrievers, OpenAI-compatible rerank adapters, context assembly, citation handling, or the LangGraph QA graph.
---

# RAG Retrieval Builder

## Core Rule

Treat metadata filtering as a business-critical security boundary. Validate user filters against the knowledge base metadata schema, generate Milvus expressions internally, retrieve with `kb_id` isolation, rerank candidates, and return cited answers with source metadata.

## Before Implementing

1. Read `docs/technical_architecture.md` sections 4.4, 4.7, 4.8, 4.9, 4.10, and 4.11.
2. Read `references/retrieval-workflow.md` for filter, retriever, rerank, and QA graph contracts.
3. Use `rag-project-architect` guidance for repo layout and cross-module boundaries.

## Retrieval Flow

Implement search in this order:

1. Receive `query`, `kb_id`, and optional structured `filters`.
2. Load the knowledge base metadata schema.
3. Validate filter fields, operators, and value types.
4. Build a Milvus expression that always includes `kb_id`.
5. Embed the query with the configured embedding model.
6. Retrieve top K candidates from Milvus.
7. Rerank candidates with the configured reranker.
8. Return top N chunks with scores, rerank scores, source fields, and metadata.

## Filter Builder Rules

- Never accept raw Milvus expressions from callers.
- Only allow fields declared in the metadata schema.
- Only allow filtering on fields marked `filterable=true`.
- Enforce value types for `string`, `int`, `float`, `bool`, `date`, `datetime`, and `string_array`.
- Support only the project-approved operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, and `$contains`.
- Escape strings and limit expression depth and condition count.

## Rerank Rules

- Keep rerank behind a project-owned adapter.
- Expose rerank as a LangChain Runnable-compatible component.
- Write rerank scores to `Document.metadata`.
- Support top N selection.
- If rerank fails, degrade to vector-search order and log the failure.

## QA Graph

Start with the minimal graph:

```text
receive_query -> build_metadata_filter -> retrieve -> rerank -> generate_answer -> return_answer
```

Add optional nodes only after the base flow is stable:

- `classify_intent`
- `rewrite_query`
- `compress_context`
- `verify_citations`
- `repair_answer`

## Tests

Add focused tests for:

- Filter schema validation.
- Operator type checking.
- Milvus expression generation.
- Query isolation by `kb_id`.
- Rerank fallback.
- Citation payload shape.
- QA graph branch behavior.
