# 实验4：工程化数据处理与一致性实验（Lesson 15-18）

## 实验概述
本实验是 RAG 实战课程的第四个综合实验，聚焦于数据处理工程化与一致性保障，确保在大规模数据更新与接入场景下系统具备断点续传、增量更新、结构化接入与文本清洗能力。实验严格对齐 Lesson 15-18 的教学目标，刻意避免与其他实验（如容器化、API 网关、CI/CD、监控与故障注入）重复的实现细节。

## 实验目标
- 构建可断点续传的批量处理管道（幂等保障）
- 实现稳定的增量更新与失效重建策略（一致性保障）
- 完成多源结构化数据的标准化接入（统一元数据）
- 建立文本清洗与去噪流水线（可配置、可评估）

## 涉及课程
- Lesson 15：批量断点续传
- Lesson 16：增量更新与一致性
- Lesson 17：结构化数据接入
- Lesson 18：文本清洗与去噪

## 前置条件
- 熟悉 Python（3.10+）与异步编程（`asyncio`）
- 了解向量数据库（Qdrant、Milvus 或 Weaviate）与常见嵌入模型
- 掌握基础数据工程（ETL）与常见数据源（CSV/SQL/REST API）
- 准备开发环境与必要依赖（见课程仓库 README）

## 实验步骤

### 第一阶段：批量断点续传与幂等键（Lesson 15）

1. 断点续传与检查点
```python
from dataclasses import dataclass
from typing import Optional, Dict, List
import hashlib, time

@dataclass
class Checkpoint:
    batch_id: str
    offset: int
    status: str  # pending|processing|completed|failed
    updated_at: float

class CheckpointManager:
    def __init__(self, store):
        self.store = store  # 可用 Redis/DB
    
    def load(self, pipeline_id: str) -> Optional[Checkpoint]:
        return self.store.get(f"ckpt:{pipeline_id}")
    
    def save(self, pipeline_id: str, ckpt: Checkpoint):
        self.store.set(f"ckpt:{pipeline_id}", ckpt)
    
    def mark(self, pipeline_id: str, status: str, offset: int):
        ckpt = self.load(pipeline_id) or Checkpoint(batch_id=pipeline_id, offset=0, status="pending", updated_at=time.time())
        ckpt.status, ckpt.offset, ckpt.updated_at = status, offset, time.time()
        self.save(pipeline_id, ckpt)

def idempotency_key(doc: dict, stable_fields: List[str]) -> str:
    payload = "|".join(str(doc.get(f, "")) for f in stable_fields)
    return hashlib.sha256(payload.encode()).hexdigest()

class BatchResumableLoader:
    def __init__(self, ckpt: CheckpointManager, sink, stable_fields):
        self.ckpt, self.sink, self.stable_fields = ckpt, sink, stable_fields
    
    async def run(self, pipeline_id: str, docs: List[dict], chunk_size: int = 1000):
        start_offset = (self.ckpt.load(pipeline_id) or Checkpoint(pipeline_id, 0, "pending", time.time())).offset
        for i in range(start_offset, len(docs), chunk_size):
            chunk = docs[i:i+chunk_size]
            self.ckpt.mark(pipeline_id, "processing", i)
            for d in chunk:
                key = idempotency_key(d, self.stable_fields)
                if await self.sink.exists(key):
                    continue  # 幂等：跳过已处理
                await self.sink.write(key, d)
            self.ckpt.mark(pipeline_id, "completed", i+len(chunk))
```

2. 幂等键设计建议
- 优先选择稳定字段组合（如 `source_id + version + content_hash`）
- 当 `content_hash` 引入（见 Lesson 18）后，优先以标题+内容哈希作为去重键

3. 批量执行建议
- 设计 `chunk_size` 与 `concurrency` 的上限，避免超时与内存峰值
- 以 `checkpoint` 粒度进行可重入执行，失败从最近完成的偏移恢复
- 统一幂等字段来源于 Lesson 17 的标准化元数据

4. 质量基线
- 重试成功率 ≥ 98%，重复写入（幂等冲突） ≤ 0.5%
- 批量失败可恢复（恢复至最近检查点）且不产生脏数据

---

### 第二阶段：增量更新与失效重建（Lesson 16）

1. 增量计算与索引更新
```python
class DeltaIndexUpdater:
    def __init__(self, store, embedder, vector_db):
        self.store, self.embedder, self.vector_db = store, embedder, vector_db
    
    async def compute_delta(self, since_ts: float):
        # 读取自 since_ts 之后变更的记录（新增/更新/删除）
        return await self.store.list_changes(since_ts)
    
    async def apply_upserts(self, changes):
        for c in changes:
            if c.type in ("insert", "update"):
                emb = await self.embedder.encode(c.document)
                await self.vector_db.upsert(id=c.id, vector=emb, metadata=c.document)
            elif c.type == "delete":
                await self.vector_db.delete(id=c.id)
    
    async def run_once(self, since_ts: float):
        changes = await self.compute_delta(since_ts)
        await self.apply_upserts(changes)
```

2. 失效重建与一致性保障
```python
class Invalidator:
    def __init__(self, vector_db):
        self.vector_db = vector_db
    
    async def invalidate_by_tag(self, tag: str):
        ids = await self.vector_db.list_ids_by_tag(tag)
        for id_ in ids:
            await self.vector_db.delete(id=id_)
    
    async def rebuild_by_source(self, source_id: str, docs: list):
        await self.vector_db.delete_by_source(source_id)
        for d in docs:
            emb = await embed(d)
            await self.vector_db.upsert(id=d["id"], vector=emb, metadata=d)
```

3. 调度与压缩
- 增量任务以固定频率运行（如每 5 分钟），记录 `last_success_ts`
- 周期性触发向量库压缩/重排，保持检索质量稳定

4. 一致性指标
- 重建后“陈旧命中率” ≤ 1%，增量延迟 ≤ 5 分钟
- 删除操作在索引生效（TTL ≤ 1 分钟），避免幽灵数据

---

### 第三阶段：结构化数据接入（Lesson 17）

1. 统一元数据与接入配置
```json
{
  "sources": [
    {"type": "csv", "path": "data/docs.csv", "id_field": "doc_id"},
    {"type": "sql", "dsn": "postgresql://user:pwd@host:5432/db", "table": "articles", "id_field": "id"},
    {"type": "api", "base_url": "https://api.example.com", "id_field": "uuid"}
  ],
  "normalize": {
    "title": "str",
    "content": "str",
    "tags": "list[str]",
    "source": "str"
  }
}
```

2. 连接器接口与标准化
```python
from typing import Iterable, Dict

class Connector:
    async def read(self) -> Iterable[Dict]:
        ...

class Normalizer:
    def apply(self, raw: Dict) -> Dict:
        return {
            "id": raw.get("id") or raw.get("doc_id") or raw.get("uuid"),
            "title": (raw.get("title") or "").strip(),
            "content": (raw.get("content") or "").strip(),
            "tags": raw.get("tags") or [],
            "source": raw.get("source") or "unknown"
        }
```

3. 接入覆盖与校验
- 至少覆盖三类结构化源（CSV/SQL/API）
- 正确生成稳定 `id`，供 Lesson 15/16 使用

---

### 第四阶段：文本清洗与去噪（Lesson 18）

1. 清洗配置示例
```json
{
  "rules": {
    "strip_html": true,
    "normalize_whitespace": true,
    "remove_boilerplate": ["版权声明", "联系地址"],
    "min_tokens": 20
  },
  "dedup": {"by": ["title", "content_hash"]},
  "quality": {"max_noise_ratio": 0.2}
}
```

2. 清洗流水线
```python
import hashlib

def clean_text(doc):
    txt = strip_html(doc["content"]).strip()
    txt = normalize_spaces(txt)
    txt = remove_boilerplate(txt)
    return txt

def content_hash(txt: str) -> str:
    return hashlib.sha256(txt.encode()).hexdigest()

def run_cleaning(docs):
    out = []
    seen = set()
    for d in docs:
        txt = clean_text(d)
        if len(tokenize(txt)) < 20:
            continue
        h = content_hash(txt)
        key = f"{d.get('title','')}|{h}"
        if key in seen:
            continue
        seen.add(key)
        out.append({**d, "content": txt, "content_hash": h})
    return out
```

3. （省略与部署相关策略，避免与其他实验重复）

## 实验任务（对齐 Lesson 15-18）

### 任务1：批量断点续传与幂等
**目标**：实现可断点续传的批量处理，具备幂等保障。

**具体要求**：
1. 实现检查点持久化（偏移与状态）
2. 设计稳定幂等键并避免重复写入
3. 支持失败重试与可重入执行
4. 记录批量执行日志以便审计

**评估指标**：
- 恢复成功率、重复写入率、批量完成率、平均恢复时间

### 任务2：增量更新与一致性
**目标**：稳定完成增量更新与失效重建，保证索引一致性。

**具体要求**：
1. 计算自时间点后的新增/更新/删除集
2. 设计失效重建（按源/标签）与压缩策略
3. 建立增量任务调度与运行日志
4. 确保删除在索引中及时生效

**评估指标**：
- 增量延迟、陈旧命中率、删除生效时延、重建成功率

### 任务3：结构化数据接入
**目标**：完成多源结构化数据的标准化接入与统一元数据映射。

**具体要求**：
1. 至少覆盖 CSV、SQL、API 三类源
2. 标准化字段（id、title、content、tags、source）
3. 与幂等/增量模块对接（稳定 id 与内容哈希）
4. 输出清洗前统一原始集

**评估指标**：
- 接入覆盖度、字段映射正确率、异常记录率、采集完整率

### 任务4：文本清洗与去噪
**目标**：实现可配置的文本清洗与去噪流水线，保证质量基线。

**具体要求**：
1. 规则化清洗（HTML去除、空白归一、模板噪声移除）
2. 去重策略（标题+内容哈希）
3. 质量阈值（最小 tokens、最大噪声比）
4. 输出清洗后标准集供索引

**评估指标**：
- 噪声比、重复率、可用样本率、清洗耗时

## 数据处理流程示意图
```
┌──────────────────────────────────────────────────────────────┐
│                     Batch & Delta Scheduler                 │
├──────────────────────────────────────────────────────────────┤
│                   Structured Connectors                     │
│                  (CSV/SQL/API)                              │
├──────────────────────────────────────────────────────────────┤
│               Processing Pipeline                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Checkpoint  │  │ Idempotency │  │ Cleaning    │         │
│  │   Manager   │  │   Writer    │  │   Stage     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Metadata DB │  │   Cache     │  │ Vector DB   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└──────────────────────────────────────────────────────────────┘
```

## 监控与度量（范围限定）
建议采集与本实验直接相关的处理指标与日志字段：

metrics/schema.yml（建议）
```yaml
jobs:
  - job_name: 'batch-delta'
    metrics: ['batch_completed', 'retry_success', 'delta_latency_seconds']
  - job_name: 'ingestion-store'
    metrics: ['stale_hit_ratio', 'delete_apply_latency']
  - job_name: 'cleaning'
    metrics: ['noise_ratio', 'dedup_ratio']
```

日志建议：保留关键流水线字段（pipeline_id、checkpoint_offset、idempotency_key、delta_window、quality_scores）。

## 评估标准（对齐 Lesson 15-18）

### 数据处理与一致性（40分）
- [ ] 断点续传与幂等（15分）
- [ ] 增量更新与失效重建（15分）
- [ ] 压缩与延迟控制（10分）

### 接入与清洗质量（30分）
- [ ] 结构化接入覆盖度（10分）
- [ ] 字段映射与稳定ID（10分）
- [ ] 清洗质量与重复率（10分）

### 工程文档与可操作性（30分）
- [ ] 指标采集与日志字段（10分）
- [ ] 故障恢复与可重入（10分）
- [ ] 文档与审计记录（10分）

## 对齐与范围问题（FAQ）
**Q: 是否需要容器化与 CI/CD？**
A: 不需要。本实验范围仅涵盖 Lesson 15-18 的数据处理与一致性内容，容器化与 CI/CD 在其他实验/课程中覆盖。

**Q: 是否要求 API 网关与服务治理？**
A: 不要求。该主题不在本实验范围内，避免与其他实验重复。

**Q: 如何避免与实验 3/5 重复？**
A: 不实现容器化、网关、监控告警与故障注入；仅保留与 Lesson 15-18 直接相关的处理与一致性。

## 参考资源与课程链接
- Lesson 15：`courses/11_rag/lesson15_批量断点续传/`
- Lesson 16：`courses/11_rag/lesson16_增量更新与一致性/`
- Lesson 17：`courses/11_rag/lesson17_结构化数据接入/`
- Lesson 18：`courses/11_rag/lesson18_文本清洗与去噪/`

## 实验时间安排（建议）
- 理论学习：2-3 小时（断点续传、增量一致性、接入与清洗）
- 批量与幂等实现：6-8 小时
- 增量与失效重建：6-8 小时
- 结构化接入：6-8 小时
- 文本清洗：6-8 小时
- 测试与评估：4-6 小时
- 文档与提交：2-3 小时

总计：26-35 小时

## 提交要求（对齐 Lesson 15-18）
1. 批量与增量实现：代码与配置（检查点、幂等、调度）
2. 结构化接入：连接器与字段映射说明
3. 清洗配置与结果：规则、阈值与质量报告
4. 评估与日志：指标采集与关键流水线日志片段
5. 提交说明：复现实验步骤与注意事项

## 后续实验提示
与系统可靠性相关的主题（监控、告警、故障注入、恢复演练）在实验 5 中统一实现，本实验不覆盖，以避免重复。