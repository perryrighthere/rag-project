# RAG 项目技术架构文档

## 1. 背景与目标

本项目是一个面向复杂文档的 RAG 系统。核心能力包括：

- 接入 MinerU 服务，对 PDF、图片、Office 等复杂文档进行解析。
- 使用 MinIO 保存原始文件、Markdown、图片、JSON 等解析产物。
- 使用 Milvus 作为向量数据库。
- Embedding、Rerank、Chat 模型接入外部 OpenAI-compatible API 供应商。
- 支持用户自定义 Chunk 长度、分段符和重叠长度。
- 知识库支持元数据 schema 管理，并在检索时支持安全的元数据过滤。
- 引入 LangChain 和 LangGraph，但只放在适合的抽象层，避免框架侵入核心业务逻辑。

当前项目目录 `/Users/perryhe/Projects/rag-project` 基本为空，后续实施应按新项目初始化方式推进。

## 2. 总体架构

```mermaid
flowchart LR
  User["用户 / 前端"] --> API["FastAPI 服务"]

  API --> KB["Knowledge Base Service<br/>自研业务层"]
  API --> IngestGraph["LangGraph<br/>Ingestion Graph"]
  API --> QAGraph["LangGraph<br/>QA Graph"]

  IngestGraph --> Parser["MinerU Parser Adapter<br/>自研"]
  Parser --> MinerU["MinerU API 服务"]
  Parser --> MinIO["MinIO<br/>原文 / Markdown / 图片 / JSON"]

  IngestGraph --> Splitter["LangChain Splitters<br/>+ 自定义 Markdown Splitter"]
  IngestGraph --> Embedding["LangChain Embeddings<br/>OpenAI-compatible"]
  IngestGraph --> VectorStore["LangChain Milvus VectorStore<br/>+ 自研 wrapper"]
  VectorStore --> Milvus["Milvus"]

  QAGraph --> Retriever["KnowledgeBaseRetriever<br/>自研 filter + LangChain retriever"]
  Retriever --> VectorStore
  QAGraph --> Rerank["Rerank Runnable<br/>OpenAI-compatible adapter"]
  QAGraph --> Chat["LangChain ChatModel<br/>OpenAI-compatible"]

  KB --> DB["Postgres / SQLite<br/>知识库 / 文档 / 元数据 / 任务"]
```

## 3. 框架使用边界

### 3.1 LangChain 适合负责的模块

LangChain 用于统一模型调用、文档对象、切分、检索和 Runnable 编排。

适合使用 LangChain 的模块：

- `Document` 标准对象。
- `MarkdownHeaderTextSplitter`、`RecursiveCharacterTextSplitter` 等文本切分器。
- `OpenAIEmbeddings` 或兼容 OpenAI API 的 Embedding 封装。
- `ChatOpenAI` 或兼容 OpenAI API 的 ChatModel 封装。
- Milvus VectorStore 基础操作。
- Retriever 基础接口。
- PromptTemplate、Runnable、OutputParser。
- Rerank adapter 的 Runnable 化封装。

### 3.2 LangGraph 适合负责的模块

LangGraph 用于长流程、有状态、可恢复、有分支的业务编排。

适合使用 LangGraph 的流程：

- 文档入库流水线：上传、解析、清洗、切分、Embedding、写入 Milvus、状态更新。
- 问答流水线：query 分析、metadata filter 构建、检索、低召回改写、rerank、答案生成、引用校验。
- 后续人工审核、人机协同、失败重试、节点级 trace。

### 3.3 必须自研的模块

以下模块属于业务核心，不建议交给 LangChain 或 LangGraph 直接处理：

- MinerU Parser Adapter。
- MinIO 文件组织与解析产物管理。
- 知识库元数据 schema 管理。
- 用户 metadata 校验。
- Milvus filter builder。
- 文档、任务、知识库状态机。
- 文档删除、重建索引、版本管理。
- API 权限与租户隔离。

核心原则：LangChain 和 LangGraph 是执行框架，不是业务模型。业务边界、数据契约和安全校验应由项目自身控制。

## 4. 功能模块设计

## 4.1 API 服务

技术选型：

- FastAPI
- Pydantic / Pydantic Settings
- SQLAlchemy
- Alembic

职责：

- 对外暴露知识库、文档、任务、检索和问答接口。
- 校验请求参数。
- 创建异步入库任务。
- 查询任务状态。
- 调用 LangGraph 工作流。

建议 API：

```text
POST   /knowledge-bases
GET    /knowledge-bases
GET    /knowledge-bases/{kb_id}
PATCH  /knowledge-bases/{kb_id}
PATCH  /knowledge-bases/{kb_id}/metadata-schema

POST   /knowledge-bases/{kb_id}/documents
GET    /documents/{document_id}
DELETE /documents/{document_id}

POST   /documents/{document_id}/parse
POST   /documents/{document_id}/index
POST   /documents/{document_id}/reindex

GET    /tasks/{task_id}

POST   /retrieval/search
POST   /chat
```

## 4.2 MinerU 文档解析模块

模块名称建议：

```text
src/rag_project/parsers/
```

接口设计：

```python
class DocumentParser:
    async def parse(self, file: UploadedFile, options: ParseOptions) -> ParsedDocument:
        ...
```

第一版实现：

```text
MinerUApiParser
```

调用 MinerU API：

- `GET /health`
- `POST /tasks`
- `GET /tasks/{task_id}`
- `GET /tasks/{task_id}/result`

默认 MinerU 参数：

```json
{
  "backend": "hybrid-auto-engine",
  "parse_method": "auto",
  "lang_list": ["ch"],
  "formula_enable": true,
  "table_enable": true,
  "return_md": true,
  "return_middle_json": true,
  "return_model_output": false,
  "return_content_list": true,
  "return_images": true,
  "response_format_zip": true,
  "return_original_file": false
}
```

解析流程：

1. 接收上传文件。
2. 保存原始文件到 MinIO。
3. 调用 MinerU `/tasks` 提交解析任务。
4. 轮询 MinerU 任务状态。
5. 下载结果 zip。
6. 解压 Markdown、图片、middle json、content list。
7. 上传解析产物到 MinIO。
8. 重写 Markdown 中的图片路径，将 `images/...` 替换为 MinIO HTTP URL。
9. 返回 `ParsedDocument`。

`ParsedDocument` 建议结构：

```json
{
  "document_id": "doc_x",
  "parser": "mineru",
  "parser_task_id": "mineru_task_x",
  "markdown_text": "...",
  "markdown_object_key": "parsed/kb_x/doc_x/main.md",
  "content_list_object_key": "parsed/kb_x/doc_x/content_list.json",
  "middle_json_object_key": "parsed/kb_x/doc_x/middle.json",
  "image_object_keys": ["parsed/kb_x/doc_x/images/xxx.png"],
  "parse_options": {},
  "created_at": "..."
}
```

可选增强：

- 复用 `minio_markdown_demo.py` 中的 VLM 图片解释能力。
- 将图片解释写回 Markdown。
- 为图片说明单独生成 chunk。

该能力不建议放在最小闭环第一阶段。

## 4.3 MinIO 存储模块

职责：

- 保存原始文件。
- 保存 MinerU 解析产物。
- 提供对象 URL。
- 提供对象生命周期管理。

建议对象路径：

```text
raw/{kb_id}/{document_id}/{filename}
parsed/{kb_id}/{document_id}/markdown/{filename}.md
parsed/{kb_id}/{document_id}/images/{image_name}
parsed/{kb_id}/{document_id}/json/{filename}_middle.json
parsed/{kb_id}/{document_id}/json/{filename}_content_list.json
```

MinIO 中只存大文件和解析产物，不负责业务状态。业务状态保存在数据库中。

## 4.4 知识库与元数据管理模块

模块名称建议：

```text
src/rag_project/knowledge_base/
```

核心实体：

```text
KnowledgeBase
Document
DocumentVersion
Chunk
MetadataSchema
IngestionTask
```

知识库 metadata schema 示例：

```json
{
  "kb_id": "kb_policy",
  "fields": [
    {
      "name": "doc_type",
      "type": "string",
      "required": true,
      "filterable": true
    },
    {
      "name": "department",
      "type": "string",
      "required": false,
      "filterable": true
    },
    {
      "name": "year",
      "type": "int",
      "required": false,
      "filterable": true
    },
    {
      "name": "tags",
      "type": "string_array",
      "required": false,
      "filterable": true
    }
  ]
}
```

字段类型建议：

```text
string
int
float
bool
date
datetime
string_array
```

职责：

- 管理知识库基础信息。
- 管理 metadata schema。
- 上传文档时校验 metadata。
- 检索时校验 filter 字段是否合法。
- 管理文档状态。

文档状态建议：

```text
uploaded
parsing
parsed
chunking
embedding
indexed
failed
deleted
```

## 4.5 Chunking 模块

模块名称建议：

```text
src/rag_project/chunking/
```

用户可配置：

```json
{
  "chunk_size": 800,
  "chunk_overlap": 120,
  "separators": ["\n## ", "\n### ", "\n\n", "\n", "。", "，", " "]
}
```

推荐策略：

1. MinerU Markdown 标准化。
2. Markdown 标题结构切分。
3. 表格、图片说明、公式块、代码块保护。
4. 超长段落用 LangChain `RecursiveCharacterTextSplitter` 二次切分。
5. 将标题路径拼回 chunk 正文，保证标题语义参与 embedding。

LangChain 使用点：

```text
MarkdownHeaderTextSplitter
RecursiveCharacterTextSplitter
Document
```

Chunk 转换为 LangChain Document：

```python
Document(
    page_content="# 一级标题\n## 二级标题\n\n正文内容...",
    metadata={
        "kb_id": "kb_x",
        "document_id": "doc_x",
        "chunk_id": "chunk_x",
        "chunk_index": 12,
        "heading_path": "一级标题 > 二级标题",
        "page_start": 3,
        "page_end": 4,
        "source_uri": "minio://...",
        "doc_type": "policy",
        "department": "finance",
        "year": 2025
    }
)
```

Chunk 数据应同时写入：

- 数据库：完整 chunk 记录、状态、版本、审计信息。
- Milvus：向量检索所需字段。

## 4.6 Embedding 模块

模块名称建议：

```text
src/rag_project/embeddings/
```

职责：

- 封装 OpenAI-compatible Embedding API。
- 支持 batch embedding。
- 控制重试、超时、限流。
- 校验 embedding 维度。

LangChain 使用点：

```text
langchain-openai
OpenAIEmbeddings
```

配置项：

```text
EMBEDDING_BASE_URL
EMBEDDING_API_KEY
EMBEDDING_MODEL
EMBEDDING_DIM
EMBEDDING_BATCH_SIZE
EMBEDDING_TIMEOUT
```

注意事项：

- Milvus collection 的 vector dim 必须与 embedding dim 一致。
- 切换 embedding 模型后，通常需要新建 collection 或执行全量 reindex。
- 文档应记录使用的 embedding model 和 embedding dim。

## 4.7 Milvus 向量库模块

模块名称建议：

```text
src/rag_project/vectorstores/
```

推荐 collection 策略：

- 一个环境一个主 collection。
- 通过 `kb_id` 区分知识库。
- 不建议每个知识库一个 collection，除非有强隔离或差异化 embedding 维度要求。

字段建议：

```text
id              VarChar primary key
kb_id           VarChar
document_id     VarChar
chunk_id        VarChar
chunk_index     Int64
text            VarChar
source_uri      VarChar
heading_path    VarChar
page_start      Int64
page_end        Int64
created_at      Int64

doc_type        VarChar
department      VarChar
year            Int64
author          VarChar

metadata_json   JSON
embedding       FloatVector(dim)
```

高频过滤字段应提升为 Milvus scalar 字段，不应只放在 `metadata_json` 中。

LangChain 使用点：

```text
langchain-milvus
Milvus VectorStore
as_retriever()
```

建议封装：

```text
MilvusVectorStoreAdapter
KnowledgeBaseRetriever
MilvusFilterBuilder
```

LangChain 可以负责基础 vector search，但 metadata filter 的构造必须由自研 builder 完成。

## 4.8 Metadata Filter Builder

模块名称建议：

```text
src/rag_project/retrieval/filters.py
```

用户请求示例：

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

内部转换：

```text
kb_id == "kb_policy" and doc_type == "policy" and year >= 2024 and department in ["finance", "hr"]
```

支持操作符：

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

安全要求：

- 用户不能直接传 Milvus expr。
- 字段必须存在于 metadata schema。
- 字段必须 `filterable=true`。
- 值类型必须与 schema 匹配。
- 字符串必须做转义。
- 复杂表达式需要限制最大深度和最大条件数。

## 4.9 Rerank 模块

模块名称建议：

```text
src/rag_project/rerankers/
```

Rerank API 在不同供应商之间差异较大，不建议直接绑定某个 LangChain 内置实现。

推荐抽象：

```python
class Reranker:
    async def rerank(self, query: str, documents: list[Document], top_n: int) -> list[Document]:
        ...
```

同时封装成 LangChain Runnable：

```text
query + documents -> reranked documents
```

职责：

- 调用 OpenAI-compatible 或供应商自定义 rerank API。
- 将 rerank score 写回 Document metadata。
- 支持 top_n。
- 支持失败降级：rerank 不可用时返回 vector search 原始排序。

## 4.10 Retrieval 模块

模块名称建议：

```text
src/rag_project/retrieval/
```

检索流程：

1. 接收 query、kb_id、filters。
2. 校验 filters。
3. 构造 Milvus expr。
4. 使用 LangChain retriever 或 Milvus vector search 取 top K。
5. 调用 rerank。
6. 返回 top N chunks。

检索响应示例：

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

## 4.11 QA Graph

LangGraph 用于组织问答流程。

```mermaid
flowchart TD
  A["receive_query"] --> B["classify_intent"]
  B --> C["build_metadata_filter"]
  C --> D["retrieve"]
  D --> E{"enough_context?"}
  E -- "no" --> F["rewrite_query"]
  F --> D
  E -- "yes" --> G["rerank"]
  G --> H["compress_context"]
  H --> I["generate_answer"]
  I --> J["verify_citations"]
  J --> K{"citation_ok?"}
  K -- "no" --> L["repair_answer"]
  K -- "yes" --> M["return_answer"]
```

节点说明：

```text
receive_query          接收问题、知识库、过滤条件、会话信息
classify_intent        判断是否需要检索、是否需要结构化过滤
build_metadata_filter  构造并校验 metadata filter
retrieve               初始向量检索
rewrite_query          低召回时改写查询
rerank                 对候选 chunk 重排
compress_context       控制上下文长度
generate_answer        调用 ChatModel 生成答案
verify_citations       检查答案是否有可靠引用
repair_answer          引用不足或格式错误时修复
return_answer          返回答案、引用和检索轨迹
```

第一版可以简化为：

```text
receive_query -> build_metadata_filter -> retrieve -> rerank -> generate_answer -> return_answer
```

后续再加入 query rewrite、citation verification 和 answer repair。

## 4.12 Ingestion Graph

LangGraph 用于组织文档入库流程。

```mermaid
flowchart TD
  A["validate_upload"] --> B["save_raw_file"]
  B --> C["parse_with_mineru"]
  C --> D{"parse_ok?"}
  D -- "no" --> E["mark_failed"]
  D -- "yes" --> F["normalize_markdown"]
  F --> G["merge_metadata"]
  G --> H["chunk_document"]
  H --> I["embed_chunks"]
  I --> J["upsert_milvus"]
  J --> K["verify_index"]
  K --> L["mark_indexed"]
```

节点说明：

```text
validate_upload      校验文件类型、知识库、metadata schema
save_raw_file        原文件写入 MinIO
parse_with_mineru    调用 MinerU API 并获取解析产物
normalize_markdown   图片路径重写、Markdown 清洗
merge_metadata       合并用户 metadata、系统 metadata、解析 metadata
chunk_document       结构化切分并生成 LangChain Document
embed_chunks         调用 Embedding 模型
upsert_milvus        写入 Milvus
verify_index         校验写入数量、向量维度、索引状态
mark_indexed         更新文档状态为 indexed
mark_failed          保存错误信息和失败节点
```

Graph state 建议：

```python
class IngestionState(TypedDict):
    task_id: str
    kb_id: str
    document_id: str
    raw_file_uri: str
    user_metadata: dict
    parse_options: dict
    parsed_document: dict | None
    chunks: list[dict]
    documents: list[Document]
    error: str | None
    failed_node: str | None
```

Checkpoint：

- 开发期可用 SQLite。
- 生产建议使用 Postgres checkpoint。
- 任务失败后可从失败节点重试，或从指定节点重跑。

## 5. 数据模型规划

## 5.1 数据库表

### knowledge_bases

```text
id
name
description
metadata_schema_json
chunking_config_json
embedding_model
embedding_dim
created_at
updated_at
```

### documents

```text
id
kb_id
filename
content_type
file_size
status
raw_object_key
markdown_object_key
content_list_object_key
middle_json_object_key
metadata_json
parser
parser_task_id
error_message
created_at
updated_at
```

### chunks

```text
id
kb_id
document_id
chunk_index
text
heading_path
page_start
page_end
token_count
source_uri
metadata_json
embedding_model
embedding_dim
created_at
updated_at
```

### ingestion_tasks

```text
id
kb_id
document_id
status
current_node
error_message
graph_run_id
created_at
started_at
completed_at
updated_at
```

## 5.2 Milvus Collection

Collection 名称：

```text
rag_chunks
```

字段：

```text
id              VarChar primary key
kb_id           VarChar
document_id     VarChar
chunk_id        VarChar
chunk_index     Int64
text            VarChar
source_uri      VarChar
heading_path    VarChar
page_start      Int64
page_end        Int64
created_at      Int64

doc_type        VarChar
department      VarChar
year            Int64
author          VarChar

metadata_json   JSON
embedding       FloatVector(dim)
```

索引建议：

```text
embedding: HNSW 或 IVF_FLAT，按数据规模选择
kb_id: scalar index
document_id: scalar index
doc_type: scalar index
department: scalar index
year: scalar index
```

## 6. 配置规划

`.env` 示例：

```text
APP_ENV=dev
LOG_LEVEL=INFO

DATABASE_URL=postgresql+psycopg://rag:rag@localhost:5432/rag

MINERU_API_BASE_URL=http://127.0.0.1:8000
MINERU_TASK_POLL_INTERVAL_SECONDS=1
MINERU_TASK_TIMEOUT_SECONDS=3600

MINIO_ENDPOINT=http://127.0.0.1:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=rag-bucket
MINIO_PUBLIC_BASE_URL=http://127.0.0.1:9000

MILVUS_URI=http://127.0.0.1:19530
MILVUS_COLLECTION=rag_chunks

EMBEDDING_BASE_URL=https://example.com/v1
EMBEDDING_API_KEY=xxx
EMBEDDING_MODEL=text-embedding-model
EMBEDDING_DIM=1024
EMBEDDING_BATCH_SIZE=32

RERANK_BASE_URL=https://example.com/v1
RERANK_API_KEY=xxx
RERANK_MODEL=rerank-model

CHAT_BASE_URL=https://example.com/v1
CHAT_API_KEY=xxx
CHAT_MODEL=chat-model

LANGGRAPH_CHECKPOINT_DATABASE_URL=postgresql://rag:rag@localhost:5432/rag
```

## 7. 推荐依赖

```text
fastapi
uvicorn
pydantic
pydantic-settings
sqlalchemy
alembic
psycopg
httpx
python-multipart

minio
boto3
pymilvus

langchain
langchain-core
langchain-openai
langchain-milvus
langgraph
langgraph-checkpoint-postgres

tiktoken
python-dotenv
structlog
pytest
pytest-asyncio
```

说明：

- `pymilvus` 作为底层能力保留。
- `langchain-milvus` 用于快速接入 LangChain retriever。
- 如果 LangChain Milvus wrapper 无法满足 schema 或 filter 需求，应以自研 `pymilvus` adapter 为准。

## 8. 实施顺序

## 8.1 第一阶段：项目骨架与基础设施

目标：建立可运行的服务骨架。

任务：

1. 初始化 `pyproject.toml`。
2. 建立 `src/rag_project` 目录结构。
3. 接入 FastAPI。
4. 接入 Pydantic Settings。
5. 配置日志。
6. 添加 Docker Compose：
   - Postgres
   - Milvus
   - MinIO
   - Redis，可选
7. 添加基础健康检查接口。

验收：

- 服务可启动。
- `/health` 返回正常。
- 能连接数据库、MinIO、Milvus。

## 8.2 第二阶段：知识库与元数据 schema

目标：建立业务数据模型。

任务：

1. 创建数据库 migration。
2. 实现知识库 CRUD。
3. 实现 metadata schema 定义与校验。
4. 实现文档记录创建。
5. 实现文档 metadata 校验。

验收：

- 可以创建知识库。
- 可以设置 metadata schema。
- 上传文档 metadata 时能正确校验类型、必填字段、非法字段。

## 8.3 第三阶段：MinerU 解析接入

目标：完成文档到 Markdown 的解析闭环。

任务：

1. 实现 `MinerUApiParser`。
2. 调用 MinerU `/health`。
3. 调用 MinerU `/tasks`。
4. 轮询任务状态。
5. 下载并解压结果 zip。
6. 上传 Markdown、图片、JSON 到 MinIO。
7. 重写 Markdown 图片 URL。
8. 更新文档状态。

验收：

- 上传 PDF 后能得到 Markdown。
- 图片路径可访问。
- 解析失败能记录错误。

## 8.4 第四阶段：Chunking

目标：根据用户配置生成高质量 chunks。

任务：

1. 实现 Markdown 标准化。
2. 接入 LangChain Markdown splitter。
3. 接入 `RecursiveCharacterTextSplitter`。
4. 支持用户自定义 `chunk_size`、`chunk_overlap`、`separators`。
5. 生成 LangChain `Document`。
6. 保存 chunks 到数据库。

验收：

- chunk 数量合理。
- 标题路径保留。
- 用户配置生效。
- 表格、图片说明不被粗暴切坏。

## 8.5 第五阶段：Embedding 与 Milvus 写入

目标：完成文档索引。

任务：

1. 接入 LangChain `OpenAIEmbeddings`。
2. 支持 OpenAI-compatible base URL。
3. 实现 embedding batch。
4. 创建 Milvus collection。
5. 写入 chunks。
6. 校验向量维度。
7. 实现文档删除和重建索引。

验收：

- chunks 能写入 Milvus。
- 可以按 `kb_id` 查询。
- 删除文档时能删除对应向量。

## 8.6 第六阶段：Metadata Filter Builder

目标：实现安全可控的元数据过滤。

任务：

1. 定义 filter JSON 语法。
2. 实现 schema 校验。
3. 实现类型校验。
4. 实现 Milvus expr 生成。
5. 添加注入防护。
6. 添加单元测试。

验收：

- 合法 filter 能生成正确 Milvus expr。
- 非法字段被拒绝。
- 非 filterable 字段被拒绝。
- 类型错误被拒绝。

## 8.7 第七阶段：Retrieval 与 Rerank

目标：完成检索闭环。

任务：

1. 实现 `KnowledgeBaseRetriever`。
2. 集成 LangChain retriever。
3. 集成 metadata filter。
4. 实现 OpenAI-compatible rerank adapter。
5. 封装 rerank Runnable。
6. 返回 chunks、score、rerank_score、引用信息。

验收：

- 能按 query 检索。
- 能按 metadata 过滤。
- rerank 后排序生效。
- rerank 失败时可降级。

## 8.8 第八阶段：LangGraph Ingestion Graph

目标：用 LangGraph 管理入库长流程。

任务：

1. 定义 `IngestionState`。
2. 实现 graph 节点。
3. 接入 checkpoint。
4. 接入任务状态更新。
5. 实现失败节点记录。
6. 支持重试。

验收：

- 文档入库由 graph 驱动。
- 服务重启后任务状态可恢复或可追踪。
- 失败时能看到失败节点和错误原因。

## 8.9 第九阶段：LangGraph QA Graph

目标：用 LangGraph 管理问答流程。

任务：

1. 定义 `QAState`。
2. 实现基础 graph：
   - receive_query
   - build_metadata_filter
   - retrieve
   - rerank
   - generate_answer
   - return_answer
3. 加入上下文压缩。
4. 加入引用返回。
5. 后续扩展 query rewrite 和 citation verification。

验收：

- `/chat` 能返回答案。
- 答案包含引用。
- filters 在问答流程中生效。
- 可以观察每个 graph 节点耗时。

## 8.10 第十阶段：可观测性与质量保障

目标：保证系统可调试、可评估。

任务：

1. 添加结构化日志。
2. 添加请求 trace id。
3. 记录每次检索候选、rerank 结果和最终引用。
4. 记录模型调用耗时和 token 使用量。
5. 添加 RAG evaluation 数据集。
6. 添加端到端测试。

验收：

- 出错时能定位到节点。
- 检索效果可以回放。
- 修改 chunk 参数、embedding 模型后能对比效果。

## 9. 推荐目录结构

```text
rag-project/
  docs/
    technical_architecture.md
  src/
    rag_project/
      __init__.py
      main.py
      api/
        routes_health.py
        routes_knowledge_bases.py
        routes_documents.py
        routes_retrieval.py
        routes_chat.py
      core/
        config.py
        logging.py
        errors.py
      db/
        session.py
        models.py
        repositories.py
      storage/
        minio_client.py
        object_keys.py
      parsers/
        base.py
        mineru_api.py
        markdown_assets.py
      knowledge_base/
        service.py
        metadata_schema.py
        document_state.py
      chunking/
        config.py
        markdown_normalizer.py
        splitter.py
      embeddings/
        client.py
      rerankers/
        base.py
        openai_compatible.py
      vectorstores/
        milvus_schema.py
        milvus_store.py
      retrieval/
        filters.py
        retriever.py
        context.py
      graphs/
        ingestion.py
        qa.py
        state.py
  tests/
    unit/
    integration/
  alembic/
  pyproject.toml
  docker-compose.yml
  .env.example
```

## 10. 关键风险与应对

### 10.1 MinerU 解析耗时长

风险：

- 大文件解析时间长。
- MinerU 服务可能排队。
- 任务可能失败。

应对：

- 所有解析走异步任务。
- 使用 LangGraph checkpoint。
- 保存 MinerU task id。
- 支持失败重试和状态查询。

### 10.2 Metadata filter 失控

风险：

- 用户直接传 Milvus expr 可能造成注入或查询错误。
- JSON metadata 难以高性能过滤。

应对：

- 自研 filter builder。
- 高频字段提升为 scalar 字段。
- 严格 schema 校验。
- 限制 filter 深度和条件数量。

### 10.3 LangChain 抽象不满足 Milvus 需求

风险：

- LangChain Milvus wrapper 对复杂 schema、index、filter 支持不足。

应对：

- 保留 `pymilvus` adapter。
- LangChain 只作为 retriever/vectorstore 接口层。
- 关键写入、删除、filter 查询可直接使用 `pymilvus`。

### 10.4 Embedding 模型切换

风险：

- 不同 embedding 模型维度不同。
- 已有 collection 不兼容新维度。

应对：

- 知识库记录 embedding model 和 dim。
- Milvus collection 维度固定。
- 切换模型时创建新 collection 或执行全量 reindex。

### 10.5 Rerank API 差异

风险：

- Rerank 供应商接口不完全兼容 OpenAI。

应对：

- Rerank 单独 adapter。
- 封装为 LangChain Runnable。
- 支持失败降级。

## 11. 最小可用版本范围

MVP 应包含：

- FastAPI 服务。
- 知识库创建。
- metadata schema 管理。
- 文档上传。
- MinerU API 解析。
- Markdown + 图片入 MinIO。
- 用户自定义 chunk 参数。
- Embedding 写入 Milvus。
- metadata filter 检索。
- rerank。
- 基础 `/chat`。

MVP 暂不强制包含：

- 图片 VLM 解释。
- 多轮复杂 agent。
- 自动 query rewrite。
- citation repair。
- human-in-the-loop。
- 多租户权限体系。

这些能力可以在基础链路稳定后逐步加入。

## 12. 结论

本项目应采用“自研业务核心 + LangChain 组件化接入 + LangGraph 流程编排”的架构。

LangChain 负责模型、文档、切分、检索和 Runnable 组合；LangGraph 负责文档入库和问答这类有状态、多步骤、可恢复的流程；MinerU 解析、元数据 schema、Milvus filter builder、任务状态和文件资产管理必须由项目自身控制。

这样既能利用 LangChain/LangGraph 生态提升开发效率，又能保证复杂文档解析、元数据过滤和知识库管理这些核心能力可控、可测试、可扩展。
