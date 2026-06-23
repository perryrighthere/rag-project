from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph

from rag_project.chunking import ChunkRecord, MarkdownChunker, normalize_markdown
from rag_project.core.config import get_settings
from rag_project.embeddings import OpenAICompatibleEmbeddingClient
from rag_project.parsers import MinerUApiParser, ParseOptions, ParsedDocument, UploadedFile
from rag_project.services.memory_store import InMemoryStore
from rag_project.vectorstores import MilvusVectorStoreAdapter


EmbeddingClientFactory = Callable[[], OpenAICompatibleEmbeddingClient]


class IngestionState(TypedDict, total=False):
    task_id: str
    kb_id: str
    document_id: str
    raw_file_uri: str
    user_metadata: dict[str, Any]
    parse_options: dict[str, Any]
    parsed_document: ParsedDocument | None
    chunks: list[ChunkRecord]
    vectors: list[list[float]]
    error: str | None
    failed_node: str | None


class IngestionGraph:
    """LangGraph document ingestion workflow for the current in-memory service layer."""

    def __init__(
        self,
        *,
        store: InMemoryStore,
        parser: MinerUApiParser,
        embedding_client_factory: EmbeddingClientFactory,
        vector_store: MilvusVectorStoreAdapter,
    ):
        self.store = store
        self.parser = parser
        self.embedding_client_factory = embedding_client_factory
        self.vector_store = vector_store
        self._app = self._build_graph().compile()

    async def run(self, *, task_id: str, document_id: str) -> IngestionState:
        await self.store.update_task(task_id, status="running")
        return await self._app.ainvoke({"task_id": task_id, "document_id": document_id})

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(IngestionState)
        nodes = [
            ("validate_upload", self._validate_upload),
            ("save_raw_file", self._save_raw_file),
            ("parse_with_mineru", self._parse_with_mineru),
            ("normalize_markdown", self._normalize_markdown),
            ("merge_metadata", self._merge_metadata),
            ("chunk_document", self._chunk_document),
            ("embed_chunks", self._embed_chunks),
            ("upsert_milvus", self._upsert_milvus),
            ("verify_index", self._verify_index),
            ("mark_indexed", self._mark_indexed),
        ]
        for name, handler in nodes:
            graph.add_node(name, self._guard(name, handler))
        graph.add_node("mark_failed", self._mark_failed)

        graph.set_entry_point("validate_upload")
        for index, (name, _) in enumerate(nodes):
            next_name = nodes[index + 1][0] if index + 1 < len(nodes) else END
            graph.add_conditional_edges(
                name,
                self._route_after_node,
                {"continue": next_name, "failed": "mark_failed"},
            )
        graph.add_edge("mark_failed", END)
        return graph

    async def _validate_upload(self, state: IngestionState) -> dict[str, Any]:
        document = self.store.documents.get(state["document_id"])
        if document is None:
            raise KeyError(f"Document not found: {state['document_id']}")
        if document.status == "deleted":
            raise ValueError("Deleted documents cannot be ingested")
        if document.file_content is None:
            raise ValueError("Document file content is unavailable")
        knowledge_base = self.store.knowledge_bases.get(document.kb_id)
        if knowledge_base is None:
            raise KeyError(f"Knowledge base not found: {document.kb_id}")
        knowledge_base.metadata_schema.validate_document_metadata(document.metadata)
        return {
            "kb_id": document.kb_id,
            "user_metadata": document.metadata,
        }

    async def _save_raw_file(self, state: IngestionState) -> dict[str, Any]:
        await self.store.update_document(state["document_id"], status="parsing", error=None)
        return {}

    async def _parse_with_mineru(self, state: IngestionState) -> dict[str, Any]:
        document = self.store.documents[state["document_id"]]
        settings = get_settings()
        assert document.file_content is not None
        options = ParseOptions(
            kb_id=document.kb_id,
            document_id=document.document_id,
            backend=settings.mineru_backend,
            parse_method=settings.mineru_parse_method,
            lang_list=settings.mineru_lang_list,
        )
        parsed = await self.parser.parse(
            UploadedFile(
                filename=document.filename,
                content=document.file_content,
                content_type=document.content_type,
            ),
            options,
        )
        await self.store.update_document(
            document.document_id,
            raw_object_key=parsed.raw_object_key,
            parsed_document=parsed,
            error=None,
        )
        return {
            "raw_file_uri": _source_uri(parsed.raw_object_key),
            "parse_options": options.model_dump(mode="json"),
            "parsed_document": parsed,
        }

    async def _normalize_markdown(self, state: IngestionState) -> dict[str, Any]:
        parsed = state["parsed_document"]
        if parsed is None:
            raise ValueError("parsed_document is missing")
        return {"parsed_document": parsed.model_copy(update={"markdown_text": normalize_markdown(parsed.markdown_text)})}

    async def _merge_metadata(self, state: IngestionState) -> dict[str, Any]:
        parsed = state["parsed_document"]
        if parsed is None:
            raise ValueError("parsed_document is missing")
        metadata = {
            **(state.get("user_metadata") or {}),
            "kb_id": state["kb_id"],
            "document_id": state["document_id"],
            "parser": parsed.parser,
            "parser_task_id": parsed.parser_task_id,
        }
        return {"user_metadata": metadata}

    async def _chunk_document(self, state: IngestionState) -> dict[str, Any]:
        document = self.store.documents[state["document_id"]]
        knowledge_base = self.store.knowledge_bases[state["kb_id"]]
        parsed = state["parsed_document"]
        if parsed is None:
            raise ValueError("parsed_document is missing")
        chunks = MarkdownChunker(knowledge_base.chunking_config).chunk_parsed_document(
            parsed,
            kb_id=document.kb_id,
            document_id=document.document_id,
            document_metadata=state.get("user_metadata") or {},
            source_uri=_source_uri(parsed.markdown_object_key),
        )
        return {"chunks": chunks}

    async def _embed_chunks(self, state: IngestionState) -> dict[str, Any]:
        await self.store.update_document(state["document_id"], status="embedding")
        embedding_client = self.embedding_client_factory()
        chunks = [
            chunk.model_copy(
                update={
                    "embedding_model": embedding_client.config.model,
                    "embedding_dim": embedding_client.config.dim,
                }
            )
            for chunk in state.get("chunks", [])
        ]
        vectors = await embedding_client.embed_documents([chunk.text for chunk in chunks])
        return {"chunks": chunks, "vectors": vectors}

    async def _upsert_milvus(self, state: IngestionState) -> dict[str, Any]:
        document = self.store.documents[state["document_id"]]
        knowledge_base = self.store.knowledge_bases[state["kb_id"]]
        embedding_client = self.embedding_client_factory()
        chunks = state.get("chunks", [])
        vectors = state.get("vectors", [])
        await self.vector_store.delete_document_chunks(kb_id=document.kb_id, document_id=document.document_id)
        await self.vector_store.upsert_chunks(
            chunks,
            vectors,
            metadata_schema=knowledge_base.metadata_schema,
            embedding_dim=embedding_client.config.dim,
        )
        await self.store.replace_document_chunks(document.document_id, chunks)
        await self.store.update_knowledge_base(
            document.kb_id,
            embedding_model=embedding_client.config.model,
            embedding_dim=embedding_client.config.dim,
        )
        return {}

    async def _verify_index(self, state: IngestionState) -> dict[str, Any]:
        expected = len(state.get("chunks", []))
        actual = len(self.store.list_document_chunks(state["document_id"]))
        if actual != expected:
            raise ValueError(f"indexed chunk count mismatch: expected {expected}, got {actual}")
        return {}

    async def _mark_indexed(self, state: IngestionState) -> dict[str, Any]:
        document = self.store.documents[state["document_id"]]
        embedding_client = self.embedding_client_factory()
        chunks = self.store.list_document_chunks(document.document_id)
        await self.store.update_document(
            document.document_id,
            status="indexed",
            chunk_count=len(chunks),
            embedding_model=embedding_client.config.model,
            embedding_dim=embedding_client.config.dim,
            error=None,
        )
        await self.store.update_task(
            state["task_id"],
            status="succeeded",
            result={"chunk_count": len(chunks), "embedding_model": embedding_client.config.model},
        )
        return {}

    async def _mark_failed(self, state: IngestionState) -> dict[str, Any]:
        error = state.get("error") or "ingestion failed"
        await self.store.update_document(state["document_id"], status="failed", error=error)
        await self.store.update_task(state["task_id"], status="failed", error=error)
        return {}

    def _guard(self, name: str, handler):
        async def guarded(state: IngestionState) -> dict[str, Any]:
            if state.get("error"):
                return {}
            try:
                return await handler(state)
            except Exception as exc:
                return {"error": str(exc), "failed_node": name}

        return guarded

    @staticmethod
    def _route_after_node(state: IngestionState) -> str:
        return "failed" if state.get("error") else "continue"


def _source_uri(object_key: str) -> str:
    settings = get_settings()
    return f"minio://{settings.minio_bucket}/{object_key}"
