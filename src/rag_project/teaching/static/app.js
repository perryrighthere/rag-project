"use strict";
const $ = id => document.getElementById(id);
const base = location.pathname.slice(0, location.pathname.lastIndexOf("/workbench"));
let dataset, documents = [], knowledgeBases = [], latestTasks = {}, busy = false;
const selectedKBKey = `rag-classroom:${base}:kb`;
const el = (tag, text, className) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (className) node.className = className; return node; };
function notice(message, error = false) { $("notice").textContent = message; $("notice").className = error ? "error" : ""; }
async function api(path, body, method) {
  const multipart = body instanceof FormData;
  const response = await fetch(base + path, { method: method || (body === undefined ? "GET" : "POST"), headers: multipart ? {} : {"Content-Type":"application/json"}, body: body === undefined ? undefined : multipart ? body : JSON.stringify(body) });
  const text = await response.text(); let result; try { result = JSON.parse(text); } catch { result = text; }
  if (!response.ok) throw new Error(typeof result?.detail === "string" ? result.detail : JSON.stringify(result?.detail || result));
  return result;
}
function tab(name) { document.querySelectorAll("[data-tab]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.tab === name))); ["corpus", "manage", "uploads", "experiment", "reports"].forEach(id => $(id).hidden = id !== name); }
async function action(fn) {
  if (busy) return; busy = true;
  document.querySelectorAll("button,input,textarea,select").forEach(node => node.disabled = true);
  try { await fn(); } catch (error) { notice(error.message, true); }
  finally { busy = false; document.querySelectorAll("button,input,textarea,select").forEach(node => node.disabled = false); }
}
function option(select, value, label) { const node = el("option", label); node.value = value; select.append(node); }
function config() { return {chunk_size: Number($("chunk-size").value), chunk_overlap: Number($("overlap").value)}; }
async function preview() {
  const item = dataset.documents.find(doc => doc.id === $("fixture").value);
  $("source").textContent = item.markdown; $("doc-meta").textContent = `${item.metadata.year} · ${item.metadata.department}`;
  const result = await api("/teaching/preview", {document_id:item.id, chunking_config:config()});
  $("chunks").replaceChildren(); $("chunk-count").textContent = `${result.chunks.length} 个片段`;
  result.chunks.forEach((chunk, index) => { const node = el("div", undefined, "chunk"); node.append(el("strong", `Chunk ${index + 1}`), el("p", `${chunk.text.length} 字符 · ${chunk.heading_path}`, "meta"), el("pre", chunk.text)); $("chunks").append(node); });
  notice("切分预览已更新。调整长度或重叠后，可观察片段边界变化。");
}
async function refreshKB(selected) {
  const oldKB = $("kb").value;
  knowledgeBases = await api("/knowledge-bases"); let previous = selected || $("kb").value;
  if (!previous) { try { previous = localStorage.getItem(selectedKBKey); } catch {} }
  for (const id of ["kb", "upload-kb", "manage-kb"]) {
    $(id).replaceChildren(); option($(id), "", "请选择知识库"); knowledgeBases.forEach(kb => option($(id), kb.kb_id, `${kb.name} · ${kb.kb_id.slice(-6)}`));
    if (knowledgeBases.some(kb => kb.kb_id === previous)) $(id).value = previous;
  }
  renderMetadata(); rememberKB();
  if (oldKB !== $("kb").value) resetUploadedView();
  await refreshDocuments();
}
async function refreshDocuments() {
  const kb = encodeURIComponent($("kb").value);
  const [records, tasks] = kb ? await Promise.all([api(`/knowledge-bases/${kb}/documents`), api(`/knowledge-bases/${kb}/tasks`)]) : [[], []];
  documents = records; latestTasks = {};
  tasks.forEach(task => { if (!latestTasks[task.document_id]) latestTasks[task.document_id] = task; });
  renderDocuments(); renderKBDetails();
}

function rememberKB() { try { localStorage.setItem(selectedKBKey, $("kb").value); } catch {} }
function resetUploadedView() {
  $("uploaded-title").textContent = "解析结果与索引片段";
  $("uploaded-markdown").textContent = "尚未选择文档。"; $("uploaded-chunks").replaceChildren();
  $("upload-progress").textContent = "文档与任务已刷新。";
}
function renderMetadata() {
  $("upload-metadata").replaceChildren();
  if (!$("kb").value) { $("upload-metadata").append(el("p", "请先选择目标知识库，或创建空白知识库。", "hint")); return; }
  const fields = knowledgeBases.find(kb => kb.kb_id === $("kb").value)?.metadata_schema.fields || [];
  if (!fields.length) $("upload-metadata").append(el("p", "此知识库未声明 metadata 字段，上传时使用空对象。", "hint"));
  fields.forEach(field => {
    const label = el("label", `${field.name} · ${field.type} · ${field.required ? "必填" : "选填"}`);
    label.htmlFor = `meta-${field.name}`;
    const input = el(field.type === "bool" ? "select" : "input"); input.id = label.htmlFor; input.dataset.metadataField = field.name;
    if (field.type === "bool") { option(input, "", "未填写"); option(input, "true", "true"); option(input, "false", "false"); }
    else if (["int", "float"].includes(field.type)) { input.type = "number"; input.step = field.type === "int" ? "1" : "any"; }
    else if (field.type === "date") input.type = "date";
    else if (field.type === "datetime") input.placeholder = "2026-09-29T09:00:00+08:00";
    else if (field.type === "string_array") input.placeholder = '["教学", "示例"]';
    input.required = field.required; $("upload-metadata").append(label, input);
  });
}
function readMetadata() {
  const metadata = {}, fields = knowledgeBases.find(kb => kb.kb_id === $("kb").value)?.metadata_schema.fields || [];
  fields.forEach(field => {
    const raw = $(`meta-${field.name}`).value.trim();
    if (!raw) { if (field.required) throw new Error(`请填写必填字段：${field.name}`); return; }
    let value = raw;
    if (["int", "float"].includes(field.type)) {
      value = Number(raw); if (!Number.isFinite(value) || (field.type === "int" && !Number.isSafeInteger(value))) throw new Error(`${field.name} 必须是有效的${field.type === "int" ? "整数" : "数值"}。`);
    } else if (field.type === "bool") value = raw === "true";
    else if (field.type === "string_array") {
      try { value = JSON.parse(raw); } catch { throw new Error(`${field.name} 请填写 JSON 字符串数组。`); }
      if (!Array.isArray(value) || !value.every(item => typeof item === "string")) throw new Error(`${field.name} 必须是字符串数组。`);
    }
    metadata[field.name] = value;
  });
  return metadata;
}
const stageLabels = {validate_upload:"校验文档",save_raw_file:"准备解析",parse_with_mineru:"调用解析服务",raw_saved:"原文件已保存",mineru_submitted:"等待 MinerU 解析",mineru_finished:"解析服务已完成",result_downloaded:"解析结果已下载",persisting_artifacts:"保存解析产物",artifacts_persisted:"解析产物已保存",normalize_markdown:"清洗 Markdown",merge_metadata:"合并 metadata",chunk_document:"切分文档",embed_chunks:"生成向量",upsert_milvus:"写入索引",verify_index:"核验片段记录",mark_indexed:"完成入库",indexed:"完成入库",failed:"失败"};
function taskFailure(task) {
  const stage = task.result?.failed_stage || task.result?.failed_node || task.result?.stage;
  const afterParse = task.task_type === "index" || task.result?.parse_succeeded ||
    ["normalize_markdown", "merge_metadata", "chunk_document", "embed_chunks", "upsert_milvus", "verify_index", "mark_indexed"].includes(stage);
  const reason = task.error || "请查看任务详情";
  const hint = reason.includes("MILVUS_COLLECTION") ? "\n索引集合的模型或维度不匹配，或旧集合缺少模型标记。请为当前模型配置新的 MILVUS_COLLECTION，并用同一模型重建索引。" : "";
  return `${afterParse ? "解析已成功，后续入库失败；解析结果已保留，可查看内容。修复原因后点击“重试索引”，无需重新解析。" : "处理失败。"}${stage && stage !== "failed" ? "\n失败阶段：" + (stageLabels[stage] || stage) : ""}\n${reason}${hint}`;
}
function documentStatus(doc, task) {
  if (doc.status === "failed" && doc.parsed_document && (task?.task_type === "index" || task?.result?.parse_succeeded || ["normalize_markdown", "merge_metadata", "chunk_document", "embed_chunks", "upsert_milvus", "verify_index", "mark_indexed"].includes(task?.result?.failed_node))) return "解析成功 · 入库失败";
  return {uploaded:"已上传",parsing:"解析中",parsed:"已解析，待索引",chunking:"切分中",embedding:"生成向量中",indexed:"已入库",failed:"处理失败",deleted:"已删除"}[doc.status] || doc.status;
}
function taskText(task) {
  const state = {pending:"等待执行",running:"执行中",succeeded:"成功",failed:"失败"}[task.status] || task.status;
  const stage = task.result?.stage; const details = task.result?.failed_stage || task.result?.failed_node || stage;
  return `${task.task_type} · ${state}${details ? " · " + (stageLabels[details] || details) : ""}\n${task.task_id}${task.result?.parser_task_id ? "\nMinerU：" + task.result.parser_task_id : ""}${task.status === "failed" ? "\n" + taskFailure(task) : ""}`;
}
function renderDocuments() {
  for (const target of ["documents", "upload-documents", "manage-documents"]) {
    $(target).replaceChildren(); if (!documents.length) $(target).append(el("p", "暂无文档。", "hint"));
    const visible = target === "manage-documents" ? documents.filter(doc => ($("show-deleted").checked || doc.status !== "deleted") && doc.filename.toLowerCase().includes($("document-search").value.trim().toLowerCase())) : documents;
    if (documents.length && !visible.length) $(target).append(el("p", "没有符合条件的文档。", "hint"));
    visible.forEach(doc => {
      const row = el("div", undefined, "doc-row"), task = latestTasks[doc.document_id];
      row.append(el("strong", doc.filename), el("div", `状态：${documentStatus(doc, task)} · ${doc.chunk_count} 个片段`));
      if (task) row.append(el("pre", taskText(task), "task-info")); else if (doc.error) row.append(el("p", doc.error));
      const buttons = el("div", undefined, "document-actions");
      const button = (label, fn) => { const node = el("button", label, "secondary"); node.onclick = () => action(fn); buttons.append(node); };
      if (task && ["pending", "running"].includes(task.status)) button("继续查看进度", async () => { tab("uploads"); try { await waitTask(task, doc.filename); } finally { await refreshDocuments(); } });
      else if (doc.status !== "deleted") {
        if (target === "upload-documents") {
          if (doc.parsed_document?.parser !== "teaching_fixture") {
            button(doc.parsed_document ? "重新解析" : "仅解析", () => processDocument(doc, "parse"));
            button("解析并入库", () => processDocument(doc, "ingest"));
          }
          if (doc.parsed_document) button(doc.status === "failed" ? "重试索引（无需重新解析）" : "索引 / 重建", () => processDocument(doc, "index"));
        }
        button("删除（停止检索）", async () => { await api(`/documents/${doc.document_id}`, undefined, "DELETE"); await refreshDocuments(); notice("文档已删除，后续检索将排除它。历史报告保留原快照。"); });
      }
      if (["upload-documents", "manage-documents"].includes(target) && doc.parsed_document) button("查看内容", async () => { await inspectDocument(doc.document_id); tab("uploads"); });
      row.append(buttons); $(target).append(row);
    });
  }
}
async function inspectDocument(documentId) {
  const [doc, result] = await Promise.all([api(`/documents/${documentId}`), api(`/documents/${documentId}/chunks`)]);
  $("uploaded-title").textContent = doc.filename; $("uploaded-markdown").textContent = doc.parsed_document?.markdown_text || "尚无解析结果。";
  $("uploaded-chunks").replaceChildren();
  result.chunks.forEach((chunk, index) => { const node = el("div", undefined, "chunk"); node.append(el("strong", `Chunk ${index + 1}`), el("p", chunk.chunk_id, "meta"), el("pre", chunk.text)); $("uploaded-chunks").append(node); });
  if (!result.chunks.length) $("uploaded-chunks").append(el("p", "尚未索引。"));
}
async function waitTask(task, filename) {
  const deadline = Date.now() + 30 * 60 * 1000;
  while (true) {
    $("upload-progress").textContent = `${filename}\n${taskText(task)}`;
    latestTasks[task.document_id] = task; renderDocuments();
    if (task.status === "succeeded") return task;
    if (task.status === "failed") throw new Error(`${filename}：${taskFailure(task)}`);
    if (Date.now() > deadline) throw new Error(`等待超时，任务 ${task.task_id} 可能仍在运行。可刷新后继续查看进度，请勿重复上传。`);
    await new Promise(resolve => setTimeout(resolve, 1500)); task = await api(`/tasks/${task.task_id}`);
  }
}
async function processDocument(doc, operation) {
  notice(`正在处理 ${doc.filename}…`);
  try { const task = await api(`/documents/${doc.document_id}/${operation}`, {}, "POST"); await waitTask(task, doc.filename); notice(`${doc.filename} 处理成功。`); }
  finally { await refreshDocuments(); }
}
function selectQuestion() { const q = dataset.questions.find(item => item.id === $("question").value); if (!q) return; $("query").value = q.query; $("filters").value = JSON.stringify(q.filters, null, 2); $("learning").textContent = q.learning_goal; $("reference").textContent = q.reference_answer; }
function showMatches(target, matches, snapshot) {
  $(target).replaceChildren(); if (!matches.length) { $(target).append(el("p", "没有召回片段。")); return; }
  matches.forEach((match, index) => { const node = el("details", undefined, "chunk"); node.open = index === 0; const doc = (snapshot || []).find(item => item.document_id === match.document_id);
    node.append(el("summary", `${index + 1}. ${doc?.filename || match.heading_path || match.chunk_id}`), el("p", `向量：${match.score ?? "—"} ｜ 重排：${match.rerank_score ?? "—"}`, "meta"), el("p", `${match.chunk_id} · 页码 ${match.page_start ?? "未知"}–${match.page_end ?? "未知"}`, "meta"), el("pre", match.text), el("p", match.source_uri || "", "meta")); $(target).append(node); });
}
function showReport(report) {
  tab("experiment"); $("result-status").textContent = `${report.status} · ${report.elapsed_ms ?? 0} ms`;
  $("filter-expr").textContent = report.filter_expr || "尚未构造过滤表达式";
  $("answer").textContent = report.error ? `实验失败：${report.error}` : report.answer ?? "本次为仅检索实验，未调用 Chat 模型。";
  if (report.rerank_error) $("answer").textContent += `\n\n重排序已降级：${report.rerank_error}`;
  $("context").textContent = report.context || "无上下文";
  $("trace").textContent = JSON.stringify({prompts:report.prompts || [], agent_trace:report.agent_trace || []}, null, 2);
  $("parameters").textContent = JSON.stringify({request:report.request, models:report.models, rerank_status:report.rerank_status, timings_ms:report.timings_ms, knowledge_base:report.knowledge_base, dataset:report.dataset}, null, 2);
  const ref = report.reference_questions?.[0]; $("learning").textContent = ref?.learning_goal || `问题：${report.request.query}`; $("reference").textContent = ref?.reference_answer || "自定义问题暂无参考答案。";
  showMatches("candidates", report.candidates || [], report.documents); showMatches("matches", report.matches || [], report.documents);
  $("report-links").replaceChildren(); [["下载 Markdown 报告", "markdown"], ["查看完整 JSON", ""]].forEach(([label,suffix]) => { const link = el("a", label); link.href = `${base}/teaching/experiments/${report.id}${suffix ? "/" + suffix : ""}`; if (!suffix) { link.target="_blank"; link.rel="noopener"; } $("report-links").append(link); });
}
async function history() {
  const records = await api("/teaching/experiments"); $("history").replaceChildren();
  if (!records.length) $("history").append(el("p", "还没有实验记录。运行检索或问答后会自动保存。", "muted"));
  records.forEach(record => { const row = el("div", undefined, "history-row"), info = el("div"); info.append(el("p", record.request.query), el("small", `${new Date(record.created_at).toLocaleString()} · ${record.status} · ${record.elapsed_ms ?? "—"} ms · ${record.id}`)); const button = el("button", "查看报告", "secondary"); button.onclick = () => action(async () => { showReport(await api(`/teaching/experiments/${record.id}`)); notice("正在查看已保存的运行快照，左侧参数可用于新的实验。"); }); row.append(info, button); $("history").append(row); });
}
document.querySelectorAll("[data-tab]").forEach(button => button.onclick = () => action(async () => { tab(button.dataset.tab); if (button.dataset.tab === "reports") await history(); if (button.dataset.tab === "uploads") await refreshDocuments(); if (button.dataset.tab === "manage") await refreshKB(); }));
$("fixture").onchange = () => action(preview); $("preview").onclick = () => action(preview);
$("import").onclick = () => action(async () => { notice("正在创建独立教学副本…"); const result = await api("/teaching/dataset/import", {chunking_config:config()}); await refreshKB(result.knowledge_base.kb_id); tab("experiment"); notice("教学副本已创建。点击索引按钮，生成真实向量后即可运行实验。"); });
async function chooseKB(value) { $("kb").value = value; $("upload-kb").value = value; $("manage-kb").value = value; rememberKB(); renderMetadata(); resetUploadedView(); await refreshDocuments(); }
function renderKBDetails() {
  const kb = knowledgeBases.find(item => item.kb_id === $("kb").value);
  $("kb-count").textContent = `共 ${knowledgeBases.length} 个知识库`;
  $("kb-editor").hidden = !kb;
  if (!kb) { $("kb-summary").textContent = "请选择知识库，或新建一个知识库。"; return; }
  $("kb-name").value = kb.name; $("kb-description").value = kb.description || "";
  $("kb-details").textContent = `${kb.kb_id}\n模型：${kb.embedding_model || "尚未绑定"}\n维度：${kb.embedding_dim ?? "—"}\n创建：${new Date(kb.created_at).toLocaleString()}`;
  $("kb-config").textContent = JSON.stringify({metadata_schema:kb.metadata_schema, chunking_config:kb.chunking_config}, null, 2);
  const active = documents.filter(doc => doc.status !== "deleted");
  $("kb-summary").textContent = `${kb.name} · ${active.length} 份有效文档 · ${active.filter(doc => doc.status === "indexed").length} 份已入库 · ${documents.length - active.length} 份已删除`;
}
$("manage-kb").onchange = () => action(() => chooseKB($("manage-kb").value));
$("manage-refresh").onclick = () => action(() => refreshKB());
$("manage-create").onclick = () => { tab("uploads"); $("new-kb-name").closest("details").open = true; $("new-kb-name").focus(); };
$("manage-upload").onclick = () => tab("uploads");
$("document-search").oninput = renderDocuments;
$("show-deleted").onchange = renderDocuments;
$("kb-save").onclick = () => action(async () => {
  const id = $("kb").value, name = $("kb-name").value.trim();
  if (!id || !name) throw new Error("请选择知识库并填写名称。");
  await api(`/knowledge-bases/${encodeURIComponent(id)}`, {name, description:$("kb-description").value.trim()}, "PATCH");
  await refreshKB(id); notice("知识库信息已保存。");
});
$("kb-delete").onclick = () => action(async () => {
  const kb = knowledgeBases.find(item => item.kb_id === $("kb").value); if (!kb) return;
  await refreshDocuments();
  if (Object.values(latestTasks).some(task => ["pending", "running"].includes(task.status))) throw new Error("知识库中有正在执行的任务，请等待任务结束后再删除。");
  if (!window.confirm(`确认删除知识库“${kb.name}”？\n将删除 ${documents.length} 份文档及其片段、任务记录，此操作不可撤销。历史实验报告保留。`)) return;
  await api(`/knowledge-bases/${encodeURIComponent(kb.kb_id)}`, undefined, "DELETE");
  await refreshKB(); notice(`知识库“${kb.name}”已删除，库内文档不再参与检索。历史报告保留。`);
});
$("kb").onchange = () => action(() => chooseKB($("kb").value));
$("upload-kb").onchange = () => action(() => chooseKB($("upload-kb").value));
$("refresh").onclick = () => action(async () => { await refreshKB(); notice("文档状态已刷新。"); });
$("upload-refresh").onclick = () => action(async () => { await refreshDocuments(); notice("文档与任务已刷新。活动任务可点击“继续查看进度”。"); });
$("create-kb").onclick = () => action(async () => {
  const name = $("new-kb-name").value.trim(); if (!name) throw new Error("请填写知识库名称。");
  let schema; try { schema = JSON.parse($("new-kb-schema").value); } catch { throw new Error("Metadata schema 不是有效 JSON。"); }
  const kb = await api("/knowledge-bases", {name, metadata_schema:schema});
  await refreshKB(kb.kb_id); $("new-kb-name").value = ""; $("question").value = ""; $("query").value = ""; $("filters").value = "{}";
  notice("空白知识库已创建，可选择文件上传。");
});
$("upload-submit").onclick = () => action(async () => {
  const kbId = $("upload-kb").value, files = Array.from($("upload-files").files), mode = $("upload-mode").value;
  if (!kbId) throw new Error("请选择目标知识库，或先创建空白知识库。");
  if (!files.length) throw new Error("请选择至少一个文件。");
  if (files.some(file => !file.size)) throw new Error("不能上传空文件，请重新选择。");
  const metadata = readMetadata(), uploaded = [];
  try {
    for (const [index, file] of files.entries()) {
      notice(`正在上传 ${index + 1}/${files.length}：${file.name}…`);
      const form = new FormData(); form.append("file", file); form.append("metadata", JSON.stringify(metadata));
      uploaded.push(await api(`/knowledge-bases/${encodeURIComponent(kbId)}/documents`, form));
    }
  } catch (error) {
    const remaining = files.slice(uploaded.length).map(file => file.name).join("、");
    throw new Error(`${error.message}。已上传 ${uploaded.length} 个文件，可在文档列表处理。尚未上传：${remaining}`);
  } finally {
    if (uploaded.length) $("upload-files").value = "";
    await refreshDocuments();
  }
  $("upload-progress").textContent = `已上传 ${uploaded.length} 个文件。`;
  if (mode === "ingest") {
    try { for (const doc of uploaded) await processDocument(doc, "ingest"); }
    catch (error) { throw new Error(`文件已保存，无需重复上传。${error.message}。可在列表中重试或处理其余文件。`); }
    notice(`${uploaded.length} 个文件已完成入库，可以前往检索与问答。`);
  } else notice(`${uploaded.length} 个文件已上传。可在列表中选择解析或完整入库。`);
});
$("go-experiment").onclick = () => action(async () => { tab("experiment"); $("question").value = ""; $("filters").value = "{}"; $("query").value = ""; $("reference").textContent = "自定义资料请自行核验答案依据。"; $("learning").textContent = "输入针对已上传资料的问题。"; $("query").focus(); });
$("question").onchange = selectQuestion;
$("index").onclick = () => action(async () => {
  await refreshDocuments(); const active = documents.filter(doc => doc.status !== "deleted" && doc.parsed_document); if (!active.length) throw new Error("没有已解析的文档。请先在“上传与入库”页面完成解析。");
  for (const doc of active) await processDocument(doc, "index");
  await refreshDocuments(); notice("索引完成。可以运行实验，或修改过滤条件进行对照。");
});
$("run").onclick = () => action(async () => {
  if (!$("kb").value) throw new Error("请选择知识库。");
  let filters; try { filters = JSON.parse($("filters").value); } catch { throw new Error("过滤条件不是有效 JSON。"); }
  if (!filters || Array.isArray(filters) || typeof filters !== "object") throw new Error("过滤条件必须是 JSON 对象。");
  notice("实验运行中，完成后报告将自动保存在数据库…");
  const report = await api("/teaching/experiments", {kb_id:$("kb").value, query:$("query").value, filters, top_k:Number($("top-k").value), top_n:Number($("top-n").value), rerank:$("rerank").checked, mode:$("mode").value, orchestrator:$("orchestrator").value});
  showReport(report); notice(report.status === "failed" ? `实验失败，失败报告已保存：${report.error}` : `实验完成，报告已自动保存：${report.id}`, report.status === "failed");
});
$("reload-reports").onclick = () => action(history);
action(async () => { dataset = await api("/teaching/dataset"); dataset.documents.forEach(doc => option($("fixture"), doc.id, doc.filename)); option($("question"), "", "自定义问题"); dataset.questions.forEach(q => option($("question"), q.id, `${q.id.toUpperCase()} · ${q.query}`)); $("question").value = dataset.questions[0].id; selectQuestion(); await refreshKB(); await preview(); });
