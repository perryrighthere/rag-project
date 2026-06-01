# RAG Project

面向复杂文档的 RAG 服务骨架，当前完成技术规划中 4.1-4.3：

- FastAPI API 服务与知识库、文档、任务路由。
- MinerU HTTP API parser adapter。
- MinIO 原文与解析产物存储封装。

## 目录

```text
src/rag_project/
  api/        FastAPI 路由、请求/响应模型、依赖
  core/       配置
  parsers/    DocumentParser 抽象与 MinerUApiParser
  storage/    MinIO 客户端、对象路径约定、Markdown 图片路径重写
  services/   4.4 数据库落地前的开发期内存状态
```

## 配置

配置可通过环境变量或 `.env` 提供：

```bash
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=rag-project
MINIO_SECURE=false
MINIO_PUBLIC_ENDPOINT=http://localhost:9000

MINERU_BASE_URL=http://localhost:8001
MINERU_REQUEST_TIMEOUT=60
MINERU_POLL_INTERVAL_SECONDS=2
MINERU_MAX_WAIT_SECONDS=600

VLM_IMAGE_EXPLANATIONS_ENABLED=false
VLM_BASE_URL=
VLM_API_KEY=EMPTY
VLM_MODEL=
VLM_TIMEOUT=120
VLM_MAX_TOKENS=300
```

如果本地 MinerU FastAPI 启动在 `18000` 端口，例如日志显示 `Start MinerU FastAPI Service: http://0.0.0.0:18000`，请改成：

```bash
MINERU_BASE_URL=http://127.0.0.1:18000
```

## 运行

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src uvicorn rag_project.main:app --reload
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

启动后也可以打开交互式 API 文档：

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

## 如何使用服务

当前服务完成的是“上传文档 -> 调用 MinerU 解析 -> 保存解析产物到 MinIO -> 查询任务和文档状态”的最小链路。知识库、文档和任务状态暂存在内存里，服务重启后会清空；持久化数据库属于后续 4.4 模块。

使用前需要先确保两个外部服务可访问：

- MinIO：用于保存原始文件、Markdown、图片和 JSON。
- MinerU API：需要提供 `GET /health`、`POST /tasks`、`GET /tasks/{task_id}`、`GET /tasks/{task_id}/result`。

### 1. 创建知识库

```bash
curl -s -X POST http://127.0.0.1:8000/knowledge-bases \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "政策知识库",
    "description": "用于测试文档解析",
    "metadata_schema": {
      "fields": [
        {"name": "doc_type", "type": "string", "required": false, "filterable": true},
        {"name": "department", "type": "string", "required": false, "filterable": true}
      ]
    }
  }'
```

响应中会返回 `kb_id`，后续上传文档时使用它。

### 2. 上传文档

```bash
KB_ID="kb_xxx"

curl -s -X POST "http://127.0.0.1:8000/knowledge-bases/${KB_ID}/documents" \
  -F "file=@/absolute/path/to/document.pdf;type=application/pdf" \
  -F 'metadata={"doc_type":"policy","department":"finance"}'
```

响应中会返回 `document_id`。当前实现会把上传文件内容暂存在内存里，真正写入 MinIO 发生在解析任务启动后。

### 3. 启动 MinerU 解析

```bash
DOCUMENT_ID="doc_xxx"

curl -s -X POST "http://127.0.0.1:8000/documents/${DOCUMENT_ID}/parse"
```

响应中会返回 `task_id`。服务会在后台执行：

1. 将原始文件保存到 MinIO：`raw/{kb_id}/{document_id}/{filename}`。
2. 调用 MinerU `/tasks` 提交解析任务。
3. 轮询 MinerU 任务状态。
4. 下载 MinerU 结果 zip。
5. 解压并上传 Markdown、图片、middle json、content list 到 MinIO。
6. 将 Markdown 和 JSON 里的 `images/...`、`./images/...` 重写为 MinIO HTTP URL。

### 4. 查询任务状态

```bash
TASK_ID="task_xxx"

curl -s "http://127.0.0.1:8000/tasks/${TASK_ID}"
```

任务状态：

- `pending`：任务已创建。
- `running`：正在调用 MinerU 或保存产物。
- `succeeded`：解析成功，`result` 中包含 `ParsedDocument`。
- `failed`：解析失败，`error` 中包含失败原因。

### 5. 查询文档和解析结果

```bash
curl -s "http://127.0.0.1:8000/documents/${DOCUMENT_ID}"
```

解析成功后，文档状态会变为 `parsed`，并包含：

- `raw_object_key`
- `parsed_document.markdown_text`
- `parsed_document.markdown_object_key`
- `parsed_document.middle_json_object_key`
- `parsed_document.content_list_object_key`
- `parsed_document.image_object_keys`

这些 object key 对应 MinIO 中的对象；Markdown 内的图片引用会指向 `MINIO_PUBLIC_ENDPOINT` 或 `MINIO_ENDPOINT` 生成的 HTTP URL。

如果启用了 VLM 图片解释，`parsed_document.markdown_text` 会在图片行后追加引用块形式的图片说明，`parsed_document.image_explanation_chunks` 会返回图片说明的独立 chunk，供后续 4.5 chunking 和索引流程接入。

### 6. 删除文档

```bash
curl -s -X DELETE "http://127.0.0.1:8000/documents/${DOCUMENT_ID}"
```

当前删除只会把内存中的文档状态标记为 `deleted`，不会删除 MinIO 对象。对象生命周期管理会在后续存储治理模块中完善。

## API 状态

已实现：

- `POST /knowledge-bases`
- `GET /knowledge-bases`
- `GET /knowledge-bases/{kb_id}`
- `PATCH /knowledge-bases/{kb_id}`
- `PATCH /knowledge-bases/{kb_id}/metadata-schema`
- `POST /knowledge-bases/{kb_id}/documents`
- `GET /documents/{document_id}`
- `DELETE /documents/{document_id}`
- `POST /documents/{document_id}/parse`
- `GET /tasks/{task_id}`

已预留但返回 `501`：

- `POST /documents/{document_id}/index`
- `POST /documents/{document_id}/reindex`
- `POST /retrieval/search`
- `POST /chat`

这些属于后续 chunking、embedding、Milvus、retrieval 和 QA graph 模块。

## MinerU 解析流程

`MinerUApiParser` 按规划调用：

- `GET /health`
- `POST /tasks`
- `GET /tasks/{task_id}`
- `GET /tasks/{task_id}/result`

默认参数与技术规划保持一致：`backend=hybrid-auto-engine`、`parse_method=auto`、`lang_list=["ch"]`、返回 Markdown、middle json、content list、图片和 zip。

`POST /tasks` 使用 MinerU FastAPI 的实际 multipart 契约：

- 文件字段名必须是 `files`，类型是文件列表；即使只上传一个文件，也按 `files=[...]` 发送。
- `lang_list` 按表单列表发送，例如 `["ch"]`，不是 JSON 字符串。
- 布尔参数按小写字符串发送，例如 `return_md=true`、`response_format_zip=true`。

等价的 MinerU 直连调试请求：

```bash
curl -X POST "${MINERU_BASE_URL}/tasks" \
  -F "files=@/absolute/path/to/document.pdf;type=application/pdf" \
  -F "lang_list=ch" \
  -F "backend=hybrid-auto-engine" \
  -F "parse_method=auto" \
  -F "formula_enable=true" \
  -F "table_enable=true" \
  -F "return_md=true" \
  -F "return_middle_json=true" \
  -F "return_content_list=true" \
  -F "return_images=true" \
  -F "response_format_zip=true"
```

如果 MinerU 日志出现 `POST /tasks HTTP/1.1" 422 Unprocessable Entity`，优先检查字段名是否误写成了单数 `file`，以及 `lang_list` 是否被作为 JSON 字符串发送。

解析结果会写入 MinIO：

```text
raw/{kb_id}/{document_id}/{filename}
parsed/{kb_id}/{document_id}/markdown/{filename}.md
parsed/{kb_id}/{document_id}/images/{image_name}
parsed/{kb_id}/{document_id}/json/{filename}_middle.json
parsed/{kb_id}/{document_id}/json/{filename}_content_list.json
```

Markdown 和 JSON 中的 `images/...`、`./images/...` 会被重写为 MinIO HTTP URL。

## VLM 图片解释增强

4.2 的可选增强可以通过环境变量启用：

```bash
VLM_IMAGE_EXPLANATIONS_ENABLED=true
VLM_BASE_URL=https://your-openai-compatible-vlm.example.com/v1
VLM_API_KEY=your-api-key
VLM_MODEL=your-vlm-model
VLM_TIMEOUT=120
VLM_MAX_TOKENS=300
```

启用后，解析流程会在 MinerU 产物保存阶段额外执行：

1. 使用已上传到 MinIO 的图片内容调用 OpenAI-compatible VLM。
2. 将图片说明写回 Markdown，格式为 `> 图片解释：...`。
3. 为每条图片说明生成独立的 `image_explanation_chunks`。

`image_explanation_chunks` 示例：

```json
{
  "chunk_id": "img_chunk_xxx",
  "chunk_index": 0,
  "text": "图片展示了审批流程的关键节点。",
  "page_content": "图片说明：图片展示了审批流程的关键节点。\n\n图片地址：http://...",
  "image_url": "http://...",
  "image_object_key": "parsed/kb/doc/images/figure.png",
  "metadata": {
    "kb_id": "kb_xxx",
    "document_id": "doc_xxx",
    "chunk_type": "image_explanation",
    "image_object_key": "parsed/kb/doc/images/figure.png",
    "image_url": "http://..."
  }
}
```

默认不启用该能力。启用时必须同时配置 `VLM_BASE_URL` 和 `VLM_MODEL`；否则解析任务会失败并在任务 `error` 中返回配置错误。

## 测试

```bash
PYTHONPATH=src pytest
```
