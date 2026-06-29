# RAG Project

面向复杂文档的 RAG 服务骨架，当前完成技术规划中 4.1-4.12 的最小闭环：

- FastAPI API 服务与知识库、文档、任务路由。
- SQLAlchemy 知识库持久化层，默认使用本地 SQLite，生产可切换 PostgreSQL。
- MinerU HTTP API parser adapter。
- MinIO 原文与解析产物存储封装。
- 知识库 metadata schema、文档 metadata 校验与文档状态管理。
- Markdown chunking 与 LangChain `Document` 转换。
- OpenAI-compatible Embedding 封装与维度校验。
- Milvus chunk upsert/search 适配器。
- 安全的 metadata filter builder，用户不能直接传 Milvus expr。
- OpenAI-compatible rerank adapter，未配置时按向量召回顺序降级。
- LangGraph QA graph 与 ingestion graph。

## 目录

```text
src/rag_project/
  api/        FastAPI 路由、请求/响应模型、依赖
  chat/       OpenAI-compatible ChatModel facade
  chunking/   Markdown 标准化、可配置切分、LangChain Document 转换
  core/       配置
  embeddings/ OpenAI-compatible Embedding 客户端
  graphs/     LangGraph QA 与文档入库工作流
  knowledge_base/ metadata schema 与校验
  parsers/    DocumentParser 抽象与 MinerUApiParser
  rerankers/  OpenAI-compatible rerank adapter 与降级实现
  retrieval/  Milvus filter builder、检索与 rerank 编排
  storage/    MinIO 客户端、对象路径约定、Markdown 图片路径重写
  vectorstores/ Milvus collection/upsert/search 封装
  db/         SQLAlchemy models、session、store repository
  services/   store 契约与测试用内存实现
```

## 配置

配置可通过环境变量或 `.env` 提供：

```bash
cp .env.example .env
```

默认 Docker Compose 本地依赖栈会由 `docker-compose.local.yml` 注入 PostgreSQL、Milvus 和 MinIO 的容器内地址。`.env` 主要用于模型 API、端口和外部服务覆盖。

```bash
APP_PORT=8000

# Docker Compose local dependencies leave these unset.
# App-only mode can override them to external services:
# DATABASE_URL=postgresql+psycopg://rag:rag@host.docker.internal:5432/rag
# MILVUS_URI=http://host.docker.internal:19530
# MINIO_ENDPOINT=host.docker.internal:9000

MINERU_BASE_URL=http://host.docker.internal:18000
MINERU_BACKEND=pipeline
MINERU_PARSE_METHOD=auto
MINERU_LANG_LIST='["ch"]'
MINERU_REQUEST_TIMEOUT=60
MINERU_POLL_INTERVAL_SECONDS=2
MINERU_MAX_WAIT_SECONDS=600

VLM_IMAGE_EXPLANATIONS_ENABLED=false
VLM_BASE_URL=
VLM_API_KEY=EMPTY
VLM_MODEL=
VLM_TIMEOUT=120
VLM_MAX_TOKENS=300

EMBEDDING_BASE_URL=https://your-openai-compatible-provider.example.com/v1
EMBEDDING_API_KEY=your-api-key
EMBEDDING_MODEL=text-embedding-model
EMBEDDING_DIM=1024
EMBEDDING_BATCH_SIZE=32
EMBEDDING_TIMEOUT=60

MILVUS_COLLECTION=rag_chunks

RERANK_BASE_URL=https://your-openai-compatible-provider.example.com/v1
RERANK_API_KEY=your-api-key
RERANK_MODEL=rerank-model
RERANK_TIMEOUT=60

CHAT_BASE_URL=https://your-openai-compatible-provider.example.com/v1
CHAT_API_KEY=your-api-key
CHAT_MODEL=chat-model
CHAT_TIMEOUT=60
CHAT_TEMPERATURE=0.2
CHAT_MAX_TOKENS=1200
```

如果不在 Docker 容器中运行应用，而是在宿主机直接执行 `uvicorn`，本地服务地址可以使用 `localhost` 或 `127.0.0.1`。如果应用在容器里、MinerU 在宿主机上，通常应使用 `host.docker.internal`：

```bash
MINERU_BASE_URL=http://host.docker.internal:18000
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

## Docker Compose

默认本地运行方式会启动应用、PostgreSQL、MinIO、etcd 和 Milvus standalone：

```bash
docker compose -f docker-compose.yml -f docker-compose.local.yml up --build
```

常用访问地址：

- API: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- MinIO API: `http://127.0.0.1:9000`
- MinIO Console: `http://127.0.0.1:9001`
- PostgreSQL: `127.0.0.1:5432`
- Milvus: `127.0.0.1:19530`

只运行应用并连接外部 PostgreSQL、Milvus 和 MinIO 时，只使用基础 compose 文件，并在 `.env` 中覆盖外部地址：

```bash
docker compose up --build app
```

```bash
DATABASE_URL=postgresql+psycopg://rag:rag@host.docker.internal:5432/rag
MILVUS_URI=http://host.docker.internal:19530
MINIO_ENDPOINT=host.docker.internal:9000
MINIO_PUBLIC_ENDPOINT=http://localhost:9000
MINERU_BASE_URL=http://host.docker.internal:18000
```

停止并移除本地依赖栈：

```bash
docker compose -f docker-compose.yml -f docker-compose.local.yml down
```

同时删除本地数据卷：

```bash
docker compose -f docker-compose.yml -f docker-compose.local.yml down -v
```

## 如何使用服务

当前服务完成的是“上传文档 -> 调用 MinerU 解析 -> 保存解析产物到 MinIO -> chunking -> embedding -> 写入 Milvus -> metadata 过滤检索 -> rerank -> QA graph”的最小链路。Docker Compose 本地依赖栈默认使用 PostgreSQL 保存知识库、文档、chunk 和任务状态，使用 Milvus 保存向量索引，使用 MinIO 保存原始文件和解析产物。生产环境建议把 PostgreSQL、MinIO 和 Milvus 单独部署或使用托管服务，再通过环境变量让 app 连接外部服务。

应用启动时会通过 SQLAlchemy `create_all` 创建缺失表，便于本地开发和最小部署。后续进入生产迁移治理时，应把同一组 ORM model 接入 Alembic revision。正在运行的 FastAPI `BackgroundTasks` 仍绑定当前服务进程；服务停止、`uvicorn --reload` 触发重启或多进程 worker 切换后，任务记录会保留在数据库中，但后台执行本身不会自动恢复。真正的可恢复任务队列可在后续复用现有 task 表和 LangGraph state 契约继续接入。

使用前需要先确保两个外部服务可访问：

- MinIO：用于保存原始文件、Markdown、图片和 JSON。
- MinerU API：需要提供 `GET /health`、`POST /tasks`、`GET /tasks/{task_id}`、`GET /tasks/{task_id}/result`。
- Embedding API：OpenAI-compatible `/embeddings`。
- Milvus：用于保存 chunk 向量和可过滤 scalar 字段。
- Rerank API：可选，OpenAI-compatible `/rerank`；未配置时自动降级为向量召回顺序。
- Chat API：使用 `/chat` 时必需，OpenAI-compatible chat completions。

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
        {"name": "department", "type": "string", "required": false, "filterable": true},
        {"name": "year", "type": "int", "required": false, "filterable": true},
        {"name": "tags", "type": "string_array", "required": false, "filterable": true}
      ]
    },
    "chunking_config": {
      "chunk_size": 800,
      "chunk_overlap": 120,
      "separators": ["\n## ", "\n### ", "\n\n", "\n", "。", "，", " "]
    }
  }'
```

响应中会返回 `kb_id`，后续上传文档时使用它。上传文档 metadata 必须符合该 schema；未声明字段、缺少 required 字段或类型错误都会返回 `422`。

### 2. 上传文档

```bash
KB_ID="kb_xxx"

curl -s -X POST "http://127.0.0.1:8000/knowledge-bases/${KB_ID}/documents" \
  -F "file=@/absolute/path/to/document.pdf;type=application/pdf" \
  -F 'metadata={"doc_type":"policy","department":"finance","year":2025,"tags":["travel"]}'
```

响应中会返回 `document_id`。当前实现会把上传文件内容随文档记录保存到数据库，以便后续后台解析任务读取；真正写入 MinIO 发生在解析任务启动后。大文件生产部署建议进一步调整为“上传即写 MinIO，数据库只保存 raw object key”。

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

`POST /documents/{document_id}/parse` 返回 `202` 时，响应里的任务状态可能仍是 `pending`，这是任务刚创建时的快照。后台任务会在响应返回后开始执行，并把任务状态更新为 `running`。如果服务进程停止、`uvicorn --reload` 触发重启或多进程 worker 切换，任务状态仍保存在数据库中，但当前进程内的后台执行不会自动恢复，已提交到 MinerU 的外部任务也不会被本服务继续追踪。

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

### 6. 启动索引

```bash
curl -s -X POST "http://127.0.0.1:8000/documents/${DOCUMENT_ID}/index"
```

响应中会返回 `task_id`。服务会在后台执行：

1. 标准化 MinerU Markdown。
2. 按知识库 `chunking_config` 使用 Markdown header splitter 和 recursive splitter 生成 chunks。
3. 将标题路径拼回 chunk 正文，使标题语义参与 embedding。
4. 合并系统 metadata、用户 metadata、chunk metadata。
5. 调用 OpenAI-compatible Embedding API，并校验向量维度等于 `EMBEDDING_DIM`。
6. 删除该文档旧 chunks，再 upsert 新 chunks 到 Milvus collection。
7. 将文档状态更新为 `indexed`，并记录 `chunk_count`、`embedding_model`、`embedding_dim`。

重新索引使用同一个流程：

```bash
curl -s -X POST "http://127.0.0.1:8000/documents/${DOCUMENT_ID}/reindex"
```

查询当前数据库中的 chunks：

```bash
curl -s "http://127.0.0.1:8000/documents/${DOCUMENT_ID}/chunks"
```

### 7. 使用 LangGraph 完整入库

如果希望一次执行“解析 -> chunk -> embedding -> Milvus upsert -> 标记 indexed”，可以使用 ingestion graph：

```bash
curl -s -X POST "http://127.0.0.1:8000/documents/${DOCUMENT_ID}/ingest"
```

该接口创建 `ingest` 任务，后台按 LangGraph 节点执行：

```text
validate_upload -> save_raw_file -> parse_with_mineru -> normalize_markdown -> merge_metadata -> chunk_document -> embed_chunks -> upsert_milvus -> verify_index -> mark_indexed
```

任一节点失败时会进入 `mark_failed`，任务和文档状态都会记录错误。

### 8. Metadata 过滤检索 + Rerank

```bash
curl -s -X POST http://127.0.0.1:8000/retrieval/search \
  -H 'Content-Type: application/json' \
  -d '{
    "kb_id": "'"${KB_ID}"'",
    "query": "报销政策是什么？",
    "top_k": 5,
    "top_n": 3,
    "filters": {
      "doc_type": {"$eq": "policy"},
      "year": {"$gte": 2024},
      "department": {"$in": ["finance", "hr"]},
      "tags": {"$contains": "travel"}
    }
  }'
```

服务会先用知识库 metadata schema 校验字段、操作符和值类型，再生成 Milvus filter expr。调用方不能传原始 Milvus 表达式。向量检索取 `top_k`，rerank 后返回 `top_n`；如果没有配置 `RERANK_BASE_URL`/`RERANK_MODEL`，服务会按向量检索原始顺序返回；如果已配置但调用失败，服务同样降级，并在响应中带上 `rerank_error`。

支持的操作符：

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

响应包含 `filter_expr`、`matches` 和可选 `rerank_error`。`filter_expr` 会自动附加 `kb_id == "<kb_id>"`，确保检索隔离到当前知识库。每个 match 会返回 `score`、可选 `rerank_score`、文本、来源和 metadata。

### 9. QA Graph 问答

```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "kb_id": "'"${KB_ID}"'",
    "query": "报销政策是什么？",
    "top_k": 8,
    "top_n": 4,
    "filters": {
      "doc_type": {"$eq": "policy"}
    }
  }'
```

`/chat` 使用 LangGraph 最小 QA flow：

```text
receive_query -> build_metadata_filter -> retrieve -> rerank -> generate_answer -> return_answer
```

响应包含 `answer`、`filter_expr`、`citations`、`matches` 和可选 `rerank_error`。使用前必须配置 `CHAT_MODEL`，否则接口返回 `503`。

### 10. 删除文档

```bash
curl -s -X DELETE "http://127.0.0.1:8000/documents/${DOCUMENT_ID}"
```

当前删除只会把数据库中的文档状态标记为 `deleted`，不会删除 MinIO 对象或 Milvus chunk。对象生命周期管理和向量删除治理会在后续存储治理模块中完善。

## API 状态

已实现：

- `POST /knowledge-bases`
- `GET /knowledge-bases`
- `GET /knowledge-bases/{kb_id}`
- `PATCH /knowledge-bases/{kb_id}`
- `PATCH /knowledge-bases/{kb_id}/metadata-schema`
- `POST /knowledge-bases/{kb_id}/documents`
- `GET /documents/{document_id}`
- `GET /documents/{document_id}/chunks`
- `DELETE /documents/{document_id}`
- `POST /documents/{document_id}/parse`
- `POST /documents/{document_id}/index`
- `POST /documents/{document_id}/reindex`
- `POST /documents/{document_id}/ingest`
- `GET /tasks/{task_id}`
- `POST /retrieval/search`
- `POST /chat`

## Metadata Schema 与 Filter 安全边界

知识库 schema 支持字段类型：

```text
string
int
float
bool
date
datetime
string_array
```

上传文档时会校验：

- 字段必须在 schema 中声明。
- `required=true` 字段必须提供。
- 值类型必须与字段类型匹配。

检索时会校验：

- filter 字段必须在 schema 中声明。
- filter 字段必须设置 `filterable=true`。
- 操作符必须受支持且适用于字段类型。
- 字符串会由项目代码转义后再拼入 Milvus expr。
- filter 深度和条件数有限制，避免复杂表达式直接穿透到 Milvus。

## MinerU 解析流程

`MinerUApiParser` 按规划调用：

- `GET /health`
- `POST /tasks`
- `GET /tasks/{task_id}`
- `GET /tasks/{task_id}/result`

默认参数：`backend=pipeline`、`parse_method=auto`、`lang_list=["ch"]`、返回 Markdown、middle json、content list、图片和 zip。可通过 `MINERU_BACKEND` 切换为 MinerU 支持的其他后端，例如 `hybrid-auto-engine`、`vlm-auto-engine`、`vlm-http-client`。

`POST /tasks` 使用 MinerU FastAPI 的实际 multipart 契约：

- 文件字段名必须是 `files`，类型是文件列表；即使只上传一个文件，也按 `files=[...]` 发送。
- `lang_list` 按表单列表发送，例如 `["ch"]`，不是 JSON 字符串。
- 布尔参数按小写字符串发送，例如 `return_md=true`、`response_format_zip=true`。

等价的 MinerU 直连调试请求：

```bash
curl -X POST "${MINERU_BASE_URL}/tasks" \
  -F "files=@/absolute/path/to/document.pdf;type=application/pdf" \
  -F "lang_list=ch" \
  -F "backend=pipeline" \
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
