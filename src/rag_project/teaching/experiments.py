import asyncio
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from rag_project.chat.openai_compatible import build_answer_prompt
from rag_project.qa import create_qa_orchestrator
from rag_project.qa.base import format_documents
from rag_project.teaching.dataset import load_dataset


class ExperimentRequest(BaseModel):
    kb_id: str
    query: str = Field(min_length=1, max_length=4000)
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=8, ge=1, le=100)
    top_n: int = Field(default=4, ge=1, le=100)
    rerank: bool = True
    mode: Literal["search", "chat"] = "search"
    orchestrator: Literal["single", "langgraph_multi"] = "single"

    @field_validator("query")
    @classmethod
    def strip_query(cls, value):
        if not value.strip():
            raise ValueError("问题不能为空")
        return value.strip()


def serialize_document(document) -> dict:
    return {"text": document.page_content, **document.metadata}


class RecordingChatClient:
    """Capture the actual generation prompts without logging credentials."""

    def __init__(self, client, report):
        self.client = client
        self.report = report
        self.config = client.config

    async def generate_text(self, prompt):
        self.report.setdefault("prompts", []).append(prompt)
        return await self.client.generate_text(prompt)

    async def generate_answer(self, *, query, documents):
        return await self.generate_text(build_answer_prompt(query, documents))


async def run_experiment(payload, *, store, reports, retriever_factory, chat_factory, settings) -> dict:
    start = perf_counter()
    report = {"id": f"exp_{uuid4().hex}", "created_at": datetime.now(timezone.utc).isoformat(),
              "status": "running", "request": payload.model_dump(mode="json"),
              "timings_ms": {}, "candidates": [], "matches": [], "answer": None,
              "models": {"embedding": settings.embedding_model, "embedding_dim": settings.embedding_dim,
                         "chat": settings.chat_model if payload.mode == "chat" else None,
                         "chat_temperature": settings.chat_temperature,
                         "chat_max_tokens": settings.chat_max_tokens,
                         "rerank": settings.rerank_model if payload.rerank and settings.rerank_base_url else None}}
    reports.save(report)
    phase = "snapshot"
    try:
        kb = store.get_knowledge_base(payload.kb_id)
        if kb is None:
            raise ValueError("知识库不存在")
        report["knowledge_base"] = kb.model_dump(mode="json")
        docs = store.list_documents(payload.kb_id)
        report["documents"] = [{"document_id": doc.document_id, "filename": doc.filename,
            "status": doc.status, "metadata": doc.metadata,
            "parse_options": doc.parsed_document.parse_options if doc.parsed_document else {},
            "chunks": [chunk.model_dump(mode="json") for chunk in store.list_document_chunks(doc.document_id)]}
            for doc in docs]
        if not any(doc.status == "indexed" for doc in docs):
            raise ValueError("请先完成文档索引，再运行检索实验。")
        dataset = load_dataset()
        fixture_ids = {doc.parsed_document.parse_options.get("dataset_id") for doc in docs if doc.parsed_document}
        if dataset["id"] in fixture_ids:
            report["dataset"] = [doc.parsed_document.parse_options for doc in docs
                                 if doc.parsed_document and doc.parsed_document.parser == "teaching_fixture"]
            if all(item.get("dataset_sha256") == dataset["sha256"] for item in report["dataset"]):
                report["reference_questions"] = [q for q in dataset["questions"] if q["query"] == payload.query]
        phase = "retrieval"
        retriever = retriever_factory()
        report["filter_expr"] = retriever.build_filter_expr(kb_id=payload.kb_id, filters=payload.filters)
        stage = perf_counter()
        candidates = await retriever.retrieve_candidates(kb_id=payload.kb_id, query=payload.query,
            filter_expr=report["filter_expr"], top_k=payload.top_k)
        report["timings_ms"]["retrieval"] = round((perf_counter() - stage) * 1000, 2)
        report["candidates"] = [serialize_document(doc) for doc in candidates]
        limit = min(payload.top_k, payload.top_n)
        stage = perf_counter()
        phase = "rerank"
        if payload.rerank:
            matches, report["rerank_error"] = await retriever.rerank_documents(query=payload.query,
                documents=candidates, top_n=limit)
            report["rerank_status"] = "fallback" if report["rerank_error"] else (
                "applied" if report["models"]["rerank"] else "not_configured")
        else:
            matches = candidates[:limit]
            report["rerank_status"] = "disabled"
        report["timings_ms"]["rerank"] = round((perf_counter() - stage) * 1000, 2)
        report["matches"] = [serialize_document(doc) for doc in matches]
        report["context"] = format_documents(matches)
        report["citations"] = [{key: doc.metadata.get(key) for key in
            ("chunk_id", "document_id", "source_uri", "heading_path", "page_start", "page_end")}
            for doc in matches]
        reports.save(report)
        if payload.mode == "chat":
            phase = "generation"
            stage = perf_counter()
            if payload.orchestrator == "single":
                report["prompt"] = build_answer_prompt(payload.query, matches)
            orchestrator = create_qa_orchestrator(name=payload.orchestrator,
                chat_client=RecordingChatClient(chat_factory(), report), max_rounds=settings.qa_agent_max_rounds)
            result = await orchestrator.answer(query=payload.query, documents=matches, citations=report["citations"])
            report.update(answer=result.answer, review_notes=result.review_notes,
                          agent_trace=[step.to_dict() for step in result.agent_trace])
            report["timings_ms"]["generation"] = round((perf_counter() - stage) * 1000, 2)
        report["status"] = "succeeded"
    except asyncio.CancelledError:
        report.update(status="failed", failed_stage=phase, error="实验执行被中断，请重新运行。")
        raise
    except Exception as exc:
        error = str(exc)
        for secret in (settings.embedding_api_key, settings.chat_api_key, settings.rerank_api_key):
            if secret and secret != "EMPTY":
                error = error.replace(secret, "[redacted]")
        report.update(status="failed", failed_stage=phase, error=error)
    finally:
        report["elapsed_ms"] = round((perf_counter() - start) * 1000, 2)
        reports.save(report)
    return report
