import re
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from rag_project.chunking.models import ChunkRecord, ChunkingConfig
from rag_project.parsers import ImageExplanationChunk, ParsedDocument


_MULTIPLE_BLANK_LINES = re.compile(r"\n{3,}")
_PAGE_PATTERN = re.compile(r"(?:page|页码|第)\s*[:：]?\s*(?P<page>\d+)", re.IGNORECASE)


def normalize_markdown(markdown_text: str) -> str:
    text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _MULTIPLE_BLANK_LINES.sub("\n\n", text).strip()


class MarkdownChunker:
    """Convert MinerU Markdown into project chunk records and LangChain Documents."""

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ],
            strip_headers=True,
        )
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
        )

    def chunk_parsed_document(
        self,
        parsed_document: ParsedDocument,
        *,
        kb_id: str,
        document_id: str,
        document_metadata: dict[str, Any],
        source_uri: str | None,
    ) -> list[ChunkRecord]:
        chunks = self.chunk_markdown(
            parsed_document.markdown_text,
            kb_id=kb_id,
            document_id=document_id,
            document_metadata=document_metadata,
            source_uri=source_uri,
        )
        chunks.extend(
            self._image_explanation_records(
                parsed_document.image_explanation_chunks,
                kb_id=kb_id,
                document_id=document_id,
                document_metadata=document_metadata,
                source_uri=source_uri,
                start_index=len(chunks),
            )
        )
        return chunks

    def chunk_markdown(
        self,
        markdown_text: str,
        *,
        kb_id: str,
        document_id: str,
        document_metadata: dict[str, Any],
        source_uri: str | None,
    ) -> list[ChunkRecord]:
        normalized = normalize_markdown(markdown_text)
        if not normalized:
            return []

        header_docs = self._header_splitter.split_text(normalized)
        chunks: list[ChunkRecord] = []
        for header_doc in header_docs:
            heading_path = self._heading_path(header_doc.metadata)
            page_start, page_end = self._page_range(header_doc.page_content)
            split_docs = self._text_splitter.split_documents([header_doc])
            for split_doc in split_docs:
                heading_path = self._heading_path(split_doc.metadata) or heading_path
                text = self._with_heading_context(split_doc.page_content, split_doc.metadata)
                metadata = {
                    **document_metadata,
                    "kb_id": kb_id,
                    "document_id": document_id,
                    "chunk_type": "text",
                    "heading_path": heading_path,
                }
                chunks.append(
                    ChunkRecord(
                        kb_id=kb_id,
                        document_id=document_id,
                        chunk_index=len(chunks),
                        text=text,
                        heading_path=heading_path,
                        page_start=page_start,
                        page_end=page_end,
                        source_uri=source_uri,
                        metadata=metadata,
                    )
                )
        return chunks

    @staticmethod
    def to_langchain_documents(chunks: list[ChunkRecord]) -> list[Document]:
        return [
            Document(
                page_content=chunk.text,
                metadata={
                    **chunk.metadata,
                    "kb_id": chunk.kb_id,
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "heading_path": chunk.heading_path,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "source_uri": chunk.source_uri,
                },
            )
            for chunk in chunks
        ]

    @staticmethod
    def _heading_path(metadata: dict[str, Any]) -> str:
        headings = [metadata[key] for key in ("h1", "h2", "h3", "h4") if metadata.get(key)]
        return " > ".join(str(item) for item in headings)

    @staticmethod
    def _with_heading_context(page_content: str, metadata: dict[str, Any]) -> str:
        headings = [metadata[key] for key in ("h1", "h2", "h3", "h4") if metadata.get(key)]
        if not headings:
            return page_content.strip()
        heading_prefix = "\n".join(f"{'#' * (index + 1)} {heading}" for index, heading in enumerate(headings))
        return f"{heading_prefix}\n\n{page_content.strip()}"

    @staticmethod
    def _page_range(text: str) -> tuple[int | None, int | None]:
        pages = [int(match.group("page")) for match in _PAGE_PATTERN.finditer(text)]
        if not pages:
            return None, None
        return min(pages), max(pages)

    @staticmethod
    def _image_explanation_records(
        chunks: list[ImageExplanationChunk],
        *,
        kb_id: str,
        document_id: str,
        document_metadata: dict[str, Any],
        source_uri: str | None,
        start_index: int,
    ) -> list[ChunkRecord]:
        records: list[ChunkRecord] = []
        for offset, chunk in enumerate(chunks):
            metadata = {
                **document_metadata,
                **chunk.metadata,
                "kb_id": kb_id,
                "document_id": document_id,
                "chunk_type": "image_explanation",
            }
            records.append(
                ChunkRecord(
                    kb_id=kb_id,
                    document_id=document_id,
                    chunk_id=chunk.chunk_id,
                    chunk_index=start_index + offset,
                    text=chunk.page_content,
                    source_uri=source_uri,
                    metadata=metadata,
                )
            )
        return records
