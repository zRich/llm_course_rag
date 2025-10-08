# 实验 2 答案解析（Hybrid 检索与重排）

本解析针对实验 2 的四个核心任务（Lesson 7–10），给出目标说明、推荐实现步骤、伪代码、接口调用示例、调参建议与故障排查。按照本文完成后，系统应具备：关键词检索、融合策略（RRF/线性）、过滤 DSL（eq/in/range/exists）、CrossEncoder 重排，以及端到端的问答能力。

## 总览与成果
- 课程映射：
  - Lesson 7：关键词检索（TF‑IDF + 字符 n‑gram）
  - Lesson 8：检索融合（RRF 与线性加权）
  - Lesson 9：过滤 DSL（eq/in/range/exists）
  - Lesson 10：CrossEncoder 语义重排（Top‑M）
- 完成后，你可以：
  - 对上传并向量化的文档进行混合检索
  - 使用过滤 DSL 精准筛选候选
  - 开启/关闭重排并比较排序差异
  - 通过问答接口返回答案与引用来源

## 环境准备
- 启动应用：`uv run uvicorn src.main:app --host 0.0.0.0 --port 8000`
- 数据服务：Postgres/Redis/Qdrant 已运行（参考项目 docker-compose）
- 基础数据：上传至少一个文档并完成向量化（见《lab02_操作手册.md》中的 I/O 示例）

## 数据流与模块关系
- 路由：`POST /api/v1/retrieval/search`
  - 关键词候选：`KeywordSearchService.search`
  - 向量候选：`VectorService.search_similar_chunks`
  - 融合：`fusion_service.rrf_fuse | linear_fuse`
  - 过滤：`filter_dsl.apply_filters`
  - 重排：`RerankService.rerank`
- 问答：`POST /api/v1/qa/ask`，内部会复用检索结果聚合上下文，并生成答案与来源。

> 学生版代码中，以上 4 个关键函数带有 `TODO(lab02-Lx)` 占位；路由会在未实现时返回 501，保证“最小可运行”。

## Lesson 7：关键词检索（TF‑IDF + 字符 n‑gram）

目标
- 在 `src/services/keyword_service.py` 的 `search(...)` 中完成 TF‑IDF + 字符 n‑gram 检索逻辑，输出字段与向量检索保持一致。

推荐步骤
- 初始化索引（若未加载）：
  - 从 `Chunk` 表取全部分块文本（`content`）与对应 `chunk_id/document_id`；
  - 使用 `TfidfVectorizer(analyzer="char", ngram_range=(2,4))` 构建索引矩阵；
  - 保存 `chunk_ids/document_ids/chunk_texts/tfidf_matrix`。
- 检索：
  - 将 `query` 转换为 TF‑IDF 向量；
  - 计算与矩阵的 `cosine_similarity`；
  - 可选按 `document_ids` 过滤候选；
  - 选 Top‑K，并补齐 `Chunk/Document` 的额外字段（文件名、标题、metadata）。

伪代码
```
ensure_index()
query_vec = vectorizer.transform([query])
scores = cosine_similarity(query_vec, tfidf_matrix)[0]
mask = build_mask_by_document_ids(document_ids)
top_indices = argsort_desc(scores where mask)[:limit]
results = enrich_with_chunk_and_document(top_indices)
return results
```

调用与验收
- 端到端：
```
POST /api/v1/retrieval/search
{
  "query": "RAG系统",
  "top_k": 6,
  "fusion": {"strategy": "linear", "w_keyword": 0.5, "w_vector": 0.5}
}
```
- 期望：返回 `total_found > 0`，结果包含 `chunk_id/document_id/document_filename/content/score/...`。

常见问题
- 索引为空：未上传或未分块；需先完成向量化与分块。
- 中文召回差：提升 n‑gram 范围（如 2..5）或增加最小 DF；结合向量检索做融合。

## Lesson 8：融合策略（RRF 与线性加权）

目标
- 在 `src/services/fusion_service.py` 中实现 `rrf_fuse(...)` 与 `linear_fuse(...)`，对关键词与向量候选做融合、去重与排序。

RRF 原理
- 输入已按分数降序排列；为每个 `chunk_id` 生成名次 `rank`：从 1 开始；
- 融合分数：`rrf_score = sum(1 / (k + rank))`，默认 `k = 60`；
- 将两路结果（关键词/向量）字段合并去重，回填 `score` 与 `fusion_strategy="rrf"`。

线性加权
- 对两路分数先做归一化到 `[0,1]`；
- 融合分数：`score = w_keyword * norm(keyword) + w_vector * norm(vector)`；
- 字段合并去重，回填 `fusion_strategy="linear"`。

伪代码
```
# RRF
kw_ranks = rank_map(keyword_results)
vec_ranks = rank_map(vector_results)
merged = dedup_merge(keyword_results, vector_results)
for cid in merged:
  s = (cid in kw_ranks ? 1/(k+kw_ranks[cid]) : 0) +
      (cid in vec_ranks ? 1/(k+vec_ranks[cid]) : 0)
  out[cid].score = s
sort_desc(out by score)

# 线性
kw_norm = normalize([r.score for r in keyword_results])
vec_norm = normalize([r.score for r in vector_results])
kw_map/vec_map = map chunk_id -> norm score
merged = dedup_merge(...)
for cid in merged:
  s = w_k * kw_map.get(cid,0) + w_v * vec_map.get(cid,0)
  out[cid].score = s
sort_desc(out by score)
```

调用与验收
```
POST /api/v1/retrieval/search
{
  "query": "RAG系统",
  "top_k": 6,
  "fusion": {"strategy": "rrf", "k": 60}
}
```
- 期望：返回非空结果，`fusion_strategy` 字段正确，排序稳定。

## Lesson 9：过滤 DSL（eq/in/range/exists）

目标
- 在 `src/services/filter_dsl.py` 的 `apply_filters(...)` 中实现四类操作并支持深层字段寻址（如 `metadata.author`）。

字段寻址
- 将顶层字段与 `metadata` 合并为一个字典；
- 解析 `field`，基于 `.` 分段逐层取值；
- 不存在的路径返回 `None`。

操作语义
- `eq`：严格等值；
- `in`：列表包含（`value` 为数组）；
- `range`：`min/max` 为数值；左闭右闭；
- `exists`：值非空且非 `""`。

伪代码
```
for r in results:
  doc = merge_top_and_metadata(r)
  passed = True
  for f in filters:
    v = get_field(doc, f.field)
    if op==eq and not (v == f.value): passed=False; break
    if op==in and not (v in f.value): passed=False; break
    if op==range and not (min<=v<=max): passed=False; break
    if op==exists and not exists(v): passed=False; break
    else if unknown op: passed=False; break
  if passed: out.append(r)
return out
```

调用与验收
```
POST /api/v1/retrieval/search
{
  "query": "RAG系统",
  "top_k": 6,
  "fusion": {"strategy": "rrf", "k": 60},
  "filters": [
    {"op":"eq","field":"document_filename","value":"test_document.txt"},
    {"op":"range","field":"score","min":0.01}
  ]
}
```
- 期望：筛选后仍返回候选，字段匹配规则生效。

## Lesson 10：CrossEncoder 重排（Top‑M）

目标
- 在 `src/services/rerank_service.py` 的 `rerank(...)` 中完成模型加载与 Top‑M 重排，写回 `rerank_score` 并排序；保留未参与重排的尾部。

推荐实现
- 模型惰性加载：首次调用时加载一次 `CrossEncoder(model_name)`，失败时记录日志并回退为“不重排”；
- Top‑M 处理：仅对前 `top_m` 候选计算分数；
- 写回与排序：将分数写入 `rerank_score`，按该分数降序；尾部拼接保持稳定性。

伪代码
```
ensure_model()
pre = candidates[:max(1, top_m)]
pairs = [(query, r.content) for r in pre]
scores = model.predict(pairs)
for r,s in zip(pre, scores): r.rerank_score = s
sort_desc(pre by rerank_score)
return pre + candidates[len(pre):]
```

对比测试
```
# 无重排
POST /api/v1/retrieval/search { "query":"RAG系统", "rerank_top_m": 0, ... }

# 开启重排
POST /api/v1/retrieval/search { "query":"RAG系统", "rerank_top_m": 6, ... }
```
- 期望：两次调用的结果顺序存在差异，均成功返回。

## 端到端问答验证
```
POST /api/v1/qa/ask
{
  "question": "什么是RAG系统？",
  "top_k": 5,
  "score_threshold": 0.2,
  "context_size": 2
}
```
- 期望：`success=true`，`answer` 为合理定义，`sources` 包含引用详情（`document_id/document_filename/chunk_id/score`）。

## 参数调优建议
- `ngram_range`：中文推荐 2..4 或 2..5；过大影响索引规模与速度。
- `score_threshold`：向量检索的分数阈值；过高可能导致候选为空。
- `rrf.k`：一般 60；增大可削弱尾部名次的贡献；与候选规模相关。
- 线性权重：`w_keyword` 与 `w_vector` 按语料与查询类型调配（结构化术语 vs 自然语义）。
- `rerank_top_m`：建议 6–10；过大影响延迟；可按接口超时与吞吐调节。

## 故障排查与常见坑
- 返回 501：表示某 Lesson 的 TODO 未实现；根据错误信息完成对应函数。
- 关键词检索空：文档尚未分块/向量化；或 TF‑IDF 索引未初始化。
- RRF 排序异常：输入列表未按分数降序；确保融合前排序。
- 过滤无效：字段寻址未合并顶层与 `metadata`；或路径不存在返回 `None`。
- 重排失败：CrossEncoder 模型未加载或依赖缺失；建议惰性加载并捕获异常。

## 验收清单（Checklist）
- Lesson 7：`keyword_service.search` 返回非空候选，字段完整。
- Lesson 8：`fusion_service` 两个策略都能返回合理排序，`fusion_strategy` 字段正确。
- Lesson 9：`apply_filters` 能筛选命中与分数范围，深层字段可用。
- Lesson 10：`rerank_top_m=0` 与 `>0` 排序对比测试通过。
- 端到端：问答接口返回答案与引用来源，检索端点处理成功。

## 附录：响应字段说明（检索结果）
- `chunk_id`：分块唯一标识（字符串）
- `document_id`：所属文档 ID（字符串）
- `document_filename`：文档文件名
- `document_title`：文档标题（可空）
- `chunk_index`：分块序号
- `content`：分块文本
- `score`：检索或融合分数（重排后可同时存在 `rerank_score`）
- `start_position/end_position`：在原文档中的位置（如字符偏移）
- `metadata`：分块元数据（JSON 对象）

—— 完 ——