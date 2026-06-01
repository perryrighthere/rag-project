# RAG Project Architecture Boundaries

## Purpose

Use this reference when planning or reviewing project-wide changes.

## Non-Negotiable Boundaries

- LangChain is a component layer for `Document`, splitters, embeddings, chat models, retrievers, prompts, and Runnables.
- LangGraph is a workflow layer for stateful ingestion and QA graphs.
- MinerU integration stays behind `DocumentParser` and `MinerUApiParser`.
- MinIO stores raw and parsed artifacts; the database stores business state.
- Milvus stores vector-search fields and selected scalar metadata.
- Metadata schema validation and Milvus filter expression generation are project-owned code.
- User input must never become raw Milvus expressions.

## Recommended Package Layout

```text
src/rag_project/
  api/
  core/
  db/
  storage/
  parsers/
  knowledge_base/
  chunking/
  embeddings/
  rerankers/
  vectorstores/
  retrieval/
  graphs/
```

## Implementation Milestones

1. Skeleton: FastAPI, settings, logging, health.
2. Persistence: SQLAlchemy models and migrations.
3. Knowledge base: metadata schema and document status.
4. Storage: MinIO client and object keys.
5. Parser: MinerU API adapter.
6. Chunking: Markdown normalization and LangChain `Document`.
7. Indexing: embeddings and Milvus.
8. Filtering: metadata filter builder.
9. Retrieval: vector search and rerank.
10. Graphs: ingestion and QA.
11. Quality: observability and tests.

## Review Checklist

- Does the change preserve module ownership?
- Does it avoid leaking provider-specific model code into business services?
- Does it keep metadata validation separate from vector search execution?
- Does it record parser options, embedding model, embedding dimension, and document status?
- Does it include tests for boundary-sensitive logic?
