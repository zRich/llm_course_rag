# RAG系统实验4 - 完整答案解析（Lesson 15–18）

## 实验概述
本答案解析面向“工程化数据处理与一致性实验”，覆盖课时 15–18 的四个核心能力：
- 课程15：断点续传与幂等批量管道（Checkpoint + Idempotency + Sink）
- 课程16：三类数据源连接器（CSV / SQL / API）统一接入
- 课程17：结构化规范化与文本清洗流水线
- 课程18：增量更新与失效重建（内容哈希驱动）

文档将解释每个任务的目标、实现方案、关键技术点、测试验证方法与常见问题的解决策略，帮助你端到端完成实验并自证正确性。

---

## 实验任务总览

| 任务编号 | 任务名称 | 核心技术 | 难度等级 |
|---------|---------|---------|---------|
| 任务1 | 断点续传与幂等管道 | Checkpoint、幂等键、批量管道 | ⭐⭐⭐ |
| 任务2 | 连接器统一接入 | CSV/SQL/API 读取与字段映射 | ⭐⭐⭐ |
| 任务3 | 结构化与清洗 | 规范化与文本去噪 | ⭐⭐ |
| 任务4 | 增量更新与重建 | 内容哈希、失效重建 | ⭐⭐⭐⭐ |

---

## 任务1：断点续传与幂等管道（Lesson 15）

### 目标
- 提供可重入、可断点续传的批量加载能力
- 使用幂等键防止重复写入，确保一致性

### 核心实现

文件与模块：
- `src/services/ingestion.py`
  - `CheckpointManager`：使用缓存保存与恢复批次偏移与状态
  - `idempotency_key`：按稳定字段计算 SHA256 幂等键
  - `DocumentSink`：将结构化文档落地到 `Document/Chunk` 表
  - `BatchResumableLoader`：分批处理，具备断点续传与幂等判断
- `src/api/routes/ingestion.py`
  - `/ingestion/run` 通用批量运行
  - `/ingestion/checkpoint/{pipeline_id}` 检查点查询

关键逻辑片段（思路）：
- 幂等键生成：
  ```python
  def idempotency_key(doc, stable_fields):
      payload = "|".join(str(doc.get(f, "")) for f in stable_fields)
      return hashlib.sha256(payload.encode("utf-8")).hexdigest()
  ```
- 存在判断（避免重复）：
  ```python
  async def exists(self, key: str) -> bool:
      q = self.db.query(Document).filter(
          (Document.content_hash == key) |
          (Document.metadata_.ilike(f'%"idempotency_key": "{key}"%'))
      )
      return self.db.query(q.exists()).scalar() or False
  ```
- 批量运行（断点续传）：
  ```python
  async def run(self, pipeline_id, docs, chunk_size=1000):
      start_offset = (self.ckpt.load(pipeline_id) or Checkpoint(pipeline_id, 0, "pending", time.time())).offset
      n = len(docs)
      for i in range(start_offset, n, chunk_size):
          batch = docs[i : min(i + chunk_size, n)]
          self.ckpt.mark(pipeline_id, "processing", i)
          for d in batch:
              key = idempotency_key(d, self.stable_fields)
              if await self.sink.exists(key):
                  continue
              await self.sink.write(key, d)
          self.ckpt.mark(pipeline_id, "completed", i + len(batch))
  ```

### 测试与验证
- 准备一组结构化 `docs`，包含稳定字段（如 `id/title/content`）
- 通过 `/ingestion/run` 执行两次相同批次，第二次应不新增文档（幂等）
- 中途中断后重试，检查点偏移应从中断位置继续，最终完成

示例请求：
```json
POST /ingestion/run
{
  "pipeline_id": "lesson15-demo",
  "stable_fields": ["id", "title", "content"],
  "docs": [
    {"id": "a1", "title": "T1", "content": "hello", "tags": ["lab"], "source": "demo"},
    {"id": "a2", "title": "T2", "content": "world", "tags": [], "source": "demo"}
  ]
}
```

预期结果：
- 第一次：返回 count=2，Document/Chunk 新增记录
- 第二次：返回 count=2，但未新增记录（`exists` 命中，跳过）
- `/ingestion/checkpoint/lesson15-demo` 能看到 `status=completed` 与最终 `offset`

---

## 任务2：连接器统一接入（Lesson 16）

### 目标
- 提供 CSV、SQL、API 三类数据源读取能力
- 将原始数据映射为统一结构 `{id,title,content,tags,source}`

### 核心实现

文件与模块：
- `src/connectors/csv_connector.py` → `load_csv(file_path, field_mapping)`
  - 使用 `csv.DictReader`，按列名映射，`tags` 支持分号分隔
- `src/connectors/sql_connector.py` → `fetch_sql(db_uri, query, field_mapping)`
  - 使用 `sqlalchemy.create_engine` 与 `text(query)`，迭代 `rs.mappings()`
- `src/connectors/api_connector.py` → `fetch_api(url, headers, params, path, field_mapping)`
  - 使用 `requests.get`，解析 `json`，支持嵌套路径 `path`（如 `data.items`）

字段映射示例：
```python
field_mapping = {"id": "doc_id", "title": "doc_title", "content": "body", "tags": "labels"}
```

### 测试与验证
- CSV：准备 `data.csv` 并调用 `/ingestion/csv`，验证返回计数与落库结果
- SQL：使用容器或本地 SQLite，执行 `/ingestion/sql` 验证
- API：调用公共 JSON API 或本地 Mock，使用 `/ingestion/api` 验证

示例请求（CSV）：
```json
POST /ingestion/csv
{
  "file_path": "./fixtures/data.csv",
  "pipeline_id": "csv-lesson16",
  "stable_fields": ["id", "title", "content"],
  "field_mapping": {"id": "id", "title": "title", "content": "content", "tags": "tags"}
}
```

预期结果：
- 接口返回 `success=true, count=<N>`
- Document/Chunk 按统一结构写入；重复运行不重复写入（依赖任务1）

---

## 任务3：结构化规范化与清洗（Lesson 17）

### 目标
- 将不同来源的原始数据规范化为统一结构
- 对文本执行清洗与去噪提升分块与向量化质量

### 核心实现

文件与模块：
- `src/services/ingestion.py` → `make_structured_docs(raws)`
  - 容错提取 `id/title/content/tags/source`，去除空白、类型稳定
- `src/services/cleaning.py` → `CleaningService.clean_text(text)`
  - 统一换行与空格；去除行首尾空白；删除空行

清洗思路：
```python
out = text.replace("\r\n", "\n").replace("\r", "\n")
out = re.sub(r"[\t\u00A0]+", " ", out)
out = re.sub(r"[ ]{2,}", " ", out)
out = "\n".join(line.strip() for line in out.splitlines())
out = "\n".join(line for line in out.splitlines() if line)
```

### 测试与验证
- 使用 `/ingestion/run` 提供含冗余空白与换行的文本，验证清洗后的分块更稳定
- 验证 `make_structured_docs` 能兼容字段名差异（如 `id/doc_id/uuid`）

---

## 任务4：增量更新与失效重建（Lesson 18）

### 目标
- 基于内容哈希检测文档变化
- 有序删除旧分块并重建，保证数据一致性

### 核心实现

文件与模块：
- `src/services/incremental.py`
  - `upsert_document(doc_id, title, content, metadata)`：清洗新内容，计算哈希；未变化则跳过；变化则更新并触发重建
  - `rebuild_document(doc_id)`：删除旧分块，按策略重新分块写入，更新统计与状态

逻辑片段（思路）：
```python
new_content = cleaner.clean_text(content)
new_hash = sha256(new_content)
if d.content_hash and d.content_hash == new_hash:
    return d
# 更新并重建
if title: d.title = title
d.content = new_content
d.content_hash = new_hash
if metadata: d.metadata_ = metadata
return rebuild_document(doc_id)
```

### 测试与验证
- 第一次 `upsert`：文档写入，`status=processing → completed`，分块计数与 token 统计更新
- 第二次 `upsert`：内容相同应跳过，无数据库写操作
- 修改内容再次 `upsert`：触发 `rebuild`，分块数变化且哈希更新

示例请求：
```json
POST /ingestion/incremental/upsert
{ "doc_id": 1, "title": "更新标题", "content": "新的内容", "metadata": "{}" }
```

---

## 关键技术点解析
- 幂等键的选择：优先可稳定复现的字段组合；避免使用易变字段（如时间、随机值）
- 检查点 TTL：与向量缓存一致性对齐，防止长批次过程中过期
- 结构化规范化：以最小字段集统一表示文档，降低跨源差异带来的复杂度
- 清洗策略：以通用规则为主，确保断词与分块稳定；可在特定文本类型上做专门优化
- 增量重建：变更检测以内容哈希为主；必要时辅以元数据变更触发策略

---

## 测试用例与脚本建议
- 集成测试：
  1. 连接器读取 → 规范化 → 管道写入 → 检查点查询
  2. 幂等重复运行 → 验证不重复写入
  3. 增量更新 → 验证跳过与重建两种路径
- 脚本建议：
  - 使用 `requests`/`httpx` 编写三类数据源的调用示例与断言
  - 为清洗与结构化函数编写最小单元测试（空文本、冗余空白、嵌套路径）

---

## 常见问题与解决方案
- 连接器异常：
  - CSV 列名不匹配 → 使用 `field_mapping` 指定正确列名
  - SQL 连接失败 → 检查 `db_uri` 格式与网络连通性；使用 SQLite 进行本地验证
  - API 结构不符 → 指定 `path` 为正确的嵌套路径，如 `data.items`
- 幂等未命中：
  - 检查 `stable_fields` 是否包含足够稳定的字段；避免过少或使用可变字段
- 检查点未续传：
  - 排查缓存服务是否正常；检查 `pipeline_id` 是否一致；确认 TTL 未过期
- 重建后分块异常：
  - 调整分块策略或清洗规则；确认长文本未被截断或误清洗

---

## 运行与验证指南
- 启动：
  - `uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload`
- 主要路由：
  - `/ingestion/csv|sql|api|run|checkpoint|incremental/*`
- 验收：
  - 幂等、续传、连接器统一结构、清洗效果、增量重建均可通过接口验证

---

## 实验总结
本实验完成数据接入到数据一致性的工程化闭环。通过断点续传与幂等保障、连接器统一结构、清洗与规范化、增量更新与失效重建，系统形成“可重入、可观察、可维护”的数据处理链路。建议在生产化时：
- 补充指标与监控（批次处理耗时、跳过率、失败率）
- 对不同类型文本引入差异化清洗策略
- 将重建与向量化联动为事务性流程，进一步提升一致性