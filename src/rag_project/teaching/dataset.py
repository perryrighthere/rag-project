import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from rag_project.api.schemas import DocumentRecord, KnowledgeBaseRecord
from rag_project.chunking import ChunkingConfig
from rag_project.parsers import ParsedDocument
from rag_project.services.store import Store


def load_dataset() -> dict:
    raw = Path(__file__).with_name("dataset.json").read_bytes()
    return {**json.loads(raw), "sha256": hashlib.sha256(raw).hexdigest()}


async def import_dataset(store: Store, config: ChunkingConfig) -> dict:
    dataset = load_dataset()
    kb = await store.add_knowledge_base(KnowledgeBaseRecord(
        name=f"{dataset['title']} · {dataset['version']}",
        description=f"教学实验副本；dataset={dataset['id']}@{dataset['version']}",
        metadata_schema=dataset["metadata_schema"], chunking_config=config,
    ))
    records = []
    for item in dataset["documents"]:
        document = DocumentRecord(kb_id=kb.kb_id, filename=item["filename"],
            content_type="text/markdown", metadata=item["metadata"], status="parsed",
            file_content=item["markdown"].encode("utf-8"))
        document.parsed_document = ParsedDocument(
            document_id=document.document_id, parser="teaching_fixture", parser_task_id="fixture",
            markdown_text=item["markdown"], markdown_object_key="", raw_object_key="",
            source_uri=f"teaching://{dataset['id']}/{dataset['version']}/{item['id']}",
            parse_options={"dataset_id": dataset["id"], "dataset_version": dataset["version"],
                           "dataset_sha256": dataset["sha256"], "fixture_id": item["id"]},
            created_at=datetime.now(timezone.utc),
        )
        records.append(await store.add_document(document))
    return {"knowledge_base": kb, "documents": records, "dataset_version": dataset["version"]}
