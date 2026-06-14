from datetime import datetime, timezone

from rag_project.chunking import ChunkingConfig, MarkdownChunker
from rag_project.parsers import ParsedDocument


def test_markdown_chunker_preserves_heading_metadata_and_document_shape() -> None:
    chunker = MarkdownChunker(ChunkingConfig(chunk_size=120, chunk_overlap=20, separators=["\n\n", "\n", "。", " "]))
    parsed = ParsedDocument(
        document_id="doc",
        parser="mineru",
        parser_task_id="task",
        markdown_text="# 制度\n\n## 报销\n\n报销正文第一页 page: 3。\n\n第二段继续说明。",
        markdown_object_key="parsed/kb/doc/markdown/demo.md",
        raw_object_key="raw/kb/doc/demo.pdf",
        parse_options={},
        created_at=datetime.now(timezone.utc),
    )

    chunks = chunker.chunk_parsed_document(
        parsed,
        kb_id="kb",
        document_id="doc",
        document_metadata={"doc_type": "policy"},
        source_uri="minio://rag/parsed/kb/doc/markdown/demo.md",
    )
    documents = chunker.to_langchain_documents(chunks)

    assert chunks
    assert chunks[0].heading_path == "制度 > 报销"
    assert chunks[0].metadata["doc_type"] == "policy"
    assert chunks[0].page_start == 3
    assert documents[0].page_content.startswith("# 制度\n## 报销")
    assert documents[0].metadata["chunk_id"] == chunks[0].chunk_id
