# RAG Ingestion Workflow Reference

## Graph State

```python
class IngestionState(TypedDict):
    task_id: str
    kb_id: str
    document_id: str
    raw_file_uri: str | None
    user_metadata: dict
    parse_options: dict
    parsed_document: dict | None
    chunks: list[dict]
    documents: list[Document]
    error: str | None
    failed_node: str | None
```

## MinerU API Contract

Submit parse jobs to MinerU with multipart files and form data:

```json
{
  "backend": "hybrid-auto-engine",
  "parse_method": "auto",
  "lang_list": ["ch"],
  "formula_enable": true,
  "table_enable": true,
  "return_md": true,
  "return_middle_json": true,
  "return_content_list": true,
  "return_images": true,
  "response_format_zip": true
}
```

Expected endpoints:

```text
GET  /health
POST /tasks
GET  /tasks/{task_id}
GET  /tasks/{task_id}/result
```

## MinIO Object Keys

```text
raw/{kb_id}/{document_id}/{filename}
parsed/{kb_id}/{document_id}/markdown/{filename}.md
parsed/{kb_id}/{document_id}/images/{image_name}
parsed/{kb_id}/{document_id}/json/{filename}_middle.json
parsed/{kb_id}/{document_id}/json/{filename}_content_list.json
```

## Chunk Metadata

Each chunk must include:

```text
kb_id
document_id
chunk_id
chunk_index
heading_path
page_start
page_end
source_uri
metadata_json
embedding_model
embedding_dim
```

Common scalar metadata fields should also be copied to Milvus columns:

```text
doc_type
department
year
author
```

## Acceptance Checks

- Raw files and parsed artifacts are persisted in MinIO.
- Document state reaches `indexed` only after Milvus write verification.
- Failed nodes set both `error` and `failed_node`.
- Markdown image paths are MinIO-accessible after normalization.
- Chunk settings are honored for size, overlap, and separators.
- Reindexing does not leave duplicate active chunks.
