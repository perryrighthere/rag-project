import json
import re
from sqlalchemy import select

from rag_project.db.models import ExperimentReportModel


class ReportStore:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def save(self, report: dict) -> None:
        with self.session_factory() as session:
            row = session.get(ExperimentReportModel, report["id"])
            if row is None:
                row = ExperimentReportModel(id=report["id"], payload=report)
                session.add(row)
            else:
                row.payload = report
            session.commit()

    def get(self, report_id: str) -> dict | None:
        with self.session_factory() as session:
            row = session.get(ExperimentReportModel, report_id)
            return dict(row.payload) if row else None

    def list(self) -> list[dict]:
        with self.session_factory() as session:
            rows = session.scalars(select(ExperimentReportModel)
                .order_by(ExperimentReportModel.created_at.desc()).limit(100)).all()
            return [{key: row.payload.get(key) for key in ("id", "created_at", "status", "request", "elapsed_ms")}
                    for row in rows]


def render_markdown(report: dict) -> str:
    # Use a fence longer than any embedded backtick sequence, including model output.
    raw = json.dumps(report, ensure_ascii=False, indent=2)
    fence = "`" * max(3, max((len(part) for part in re.findall(r"`+", raw)), default=0) + 1)
    lines = ["# RAG 课堂实验报告", "", f"- 实验编号：{report['id']}",
             f"- 时间：{report['created_at']}", f"- 状态：{report['status']}",
             f"- 总耗时：{report.get('elapsed_ms', 0)} ms", "",
             "本报告保存实际运行的参数、数据快照、召回、重排、模型上下文及答案。向量分数与重排分数不可直接比较。",
             "参考答案仅用于人工核验，不会发送给模型。报告不自动判定答案正确。", "",
             "## 问题", "", fence + "text", report["request"]["query"], fence, "",
             "## 答案 / 失败原因", "", fence + "text",
             report.get("error") or report.get("answer") or "仅检索实验，未生成答案。", fence, "",
             "## 最终证据", ""]
    for index, match in enumerate(report.get("matches", []), 1):
        lines.extend([f"### 片段 {index}", "", fence + "text",
            f"chunk_id: {match.get('chunk_id')}", f"source: {match.get('source_uri')}",
            f"vector_score: {match.get('score')} / rerank_score: {match.get('rerank_score')}",
            match["text"], fence, ""])
    lines.extend([
             "## 实验观察", "", "1. 过滤条件排除了哪些文档？",
             "2. 重排是否改善了相关材料的位置？", "3. 答案中的每项结论是否有片段支持？",
             "4. 更改一个参数再运行，结果与耗时如何变化？", "", "## 完整运行记录", "",
             fence + "json", raw, fence, ""])
    return "\n".join(lines)
