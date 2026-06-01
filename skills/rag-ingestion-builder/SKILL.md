---
name: rag-ingestion-builder
description: Build and maintain the document ingestion pipeline for this RAG project. Use when implementing MinerU API parsing, MinIO artifact storage, Markdown normalization, configurable chunking, LangChain Document conversion, OpenAI-compatible embeddings, Milvus indexing, document status transitions, or the LangGraph ingestion graph.
---

# RAG Ingestion Builder

## Core Rule

Build ingestion as a resumable stateful pipeline: validate upload, store raw file, parse with MinerU, persist artifacts in MinIO, normalize Markdown, merge metadata, chunk into LangChain `Document` objects, embed, upsert to Milvus, verify index, and update document state.

## Before Implementing

1. Read `docs/technical_architecture.md` sections 4.2, 4.3, 4.5, 4.6, 4.7, and 4.12.
2. Read `references/ingestion-workflow.md` for node contracts and acceptance checks.
3. Use `rag-project-architect` guidance for repo layout and cross-module boundaries.

## Required Pipeline

Implement the ingestion graph in this order:

1. `validate_upload`
2. `save_raw_file`
3. `parse_with_mineru`
4. `normalize_markdown`
5. `merge_metadata`
6. `chunk_document`
7. `embed_chunks`
8. `upsert_milvus`
9. `verify_index`
10. `mark_indexed` or `mark_failed`

## MinerU Integration

- Call MinerU API rather than importing MinerU internals in the RAG service.
- Support `/health`, `/tasks`, `/tasks/{task_id}`, and `/tasks/{task_id}/result`.
- Default to `backend=hybrid-auto-engine`, `parse_method=auto`, `lang_list=["ch"]`, `return_md=true`, `return_images=true`, `return_middle_json=true`, and `return_content_list=true`.
- Save raw file and parsed outputs to MinIO. Store business state in the database.
- Rewrite Markdown image references from relative `images/...` paths to MinIO-accessible URLs.

## Chunking Rules

- Preserve Markdown heading hierarchy.
- Protect tables, code blocks, formulas, and image explanation blocks where possible.
- Use LangChain splitters for standard splitting, but keep project-owned normalization and chunk metadata.
- Apply user-provided `chunk_size`, `chunk_overlap`, and `separators`.
- Include heading text in `page_content` so embedding captures title semantics.

## Indexing Rules

- Store full chunk records in the database.
- Store vector-search fields in Milvus.
- Always include `kb_id`, `document_id`, `chunk_id`, `chunk_index`, `source_uri`, `heading_path`, and metadata scalar fields.
- Validate embedding dimension before writing to Milvus.
- Make document reindex delete or supersede previous chunks deterministically.

## Tests

Add focused tests for:

- MinerU API response handling and failure states.
- MinIO object key generation.
- Markdown image URL rewriting.
- Chunk parameter behavior.
- LangChain `Document` metadata shape.
- Milvus upsert payload shape and embedding dimension checks.
