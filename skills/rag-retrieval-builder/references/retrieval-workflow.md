# RAG Retrieval and QA Workflow Reference

## Filter Request Shape

```json
{
  "kb_id": "kb_policy",
  "query": "报销政策是什么？",
  "filters": {
    "doc_type": {"$eq": "policy"},
    "year": {"$gte": 2024},
    "department": {"$in": ["finance", "hr"]}
  }
}
```

## Filter Operators

```text
$eq
$ne
$gt
$gte
$lt
$lte
$in
$nin
$contains
```

## Filter Validation

Validate in this order:

1. The field exists in the knowledge base metadata schema.
2. The field is marked `filterable=true`.
3. The operator is allowed for the field type.
4. The value matches the field type.
5. The condition count and nesting depth are within project limits.

Always add `kb_id == "<kb_id>"` internally.

## Retrieval Response Shape

```json
{
  "query": "报销政策是什么？",
  "matches": [
    {
      "chunk_id": "chunk_x",
      "document_id": "doc_x",
      "score": 0.82,
      "rerank_score": 0.91,
      "text": "...",
      "source_uri": "minio://...",
      "heading_path": "制度 > 报销",
      "page_start": 3,
      "page_end": 4,
      "metadata": {
        "doc_type": "policy",
        "department": "finance",
        "year": 2025
      }
    }
  ]
}
```

## QA State

```python
class QAState(TypedDict):
    query: str
    kb_id: str
    filters: dict
    filter_expr: str | None
    candidates: list[Document]
    reranked: list[Document]
    context: str
    answer: str | None
    citations: list[dict]
    error: str | None
```

## Minimal QA Graph

```text
receive_query -> build_metadata_filter -> retrieve -> rerank -> generate_answer -> return_answer
```

## Acceptance Checks

- Queries are isolated by `kb_id`.
- Invalid filters fail before hitting Milvus.
- Rerank score is preserved in returned metadata.
- Returned answers include citation payloads.
- Rerank failure degrades gracefully to vector order.
