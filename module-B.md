好的，我们按照模块A的详细风格，将 **模块B（检索优化，Lesson 07–14）** 展开成教师讲义 + 学生实验指导 + 思考题版本。

---

# 模块B：检索强化与混合搜索（Lesson 07–14）

---

## Lesson 07 – 关键词检索

### 🎓 教师讲义要点

* **课程目标**：实现基于关键词的全文检索，掌握文本索引和查询优化技术。
* **技术重点**：

  * PostgreSQL 17 全文检索功能（tsvector/tsquery）
  * 中文分词（jieba 或 THULAC）
  * 检索结果排序与评分
  * 查询性能优化（索引、查询计划）

### 🧪 学生实验指导

```python
# 安装中文分词
uv pip install jieba

# 建立全文索引
from sqlalchemy import text
from app.core.db import engine

with engine.connect() as conn:
    conn.execute(text("""
        CREATE INDEX idx_document_content
        ON document
        USING gin(to_tsvector('chinese', content));
    """))

# 查询示例
from sqlalchemy import text
query = "人工智能"
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT * FROM document
        WHERE to_tsvector('chinese', content) @@ plainto_tsquery('chinese', :query)
        ORDER BY ts_rank(to_tsvector('chinese', content), plainto_tsquery('chinese', :query)) DESC
        LIMIT 5;
    """), {"query": query})
    for row in result:
        print(row)
```

### 🤔 思考题

1. 中文文本检索与英文文本检索有什么区别？
2. 如何提高长文档的检索精度和效率？

---

## Lesson 08 – 混合检索融合策略

### 🎓 教师讲义要点

* **课程目标**：融合向量检索与关键词检索，构建混合检索系统。
* **技术重点**：

  * 多路检索结果融合（向量 + 关键词）
  * 权重调节算法
  * 去重和排序策略
  * 简单A/B测试框架

### 🧪 学生实验指导

```python
# 假设已经有向量检索结果 vector_hits 和关键词检索结果 keyword_hits
# 简单融合示例：加权得分
alpha = 0.6  # 向量权重
beta = 0.4   # 关键词权重

combined_results = {}
for hit in vector_hits:
    combined_results[hit["id"]] = combined_results.get(hit["id"], 0) + alpha * hit["score"]
for hit in keyword_hits:
    combined_results[hit["id"]] = combined_results.get(hit["id"], 0) + beta * hit["score"]

# 排序输出
final_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
print(final_results[:5])
```

### 🤔 思考题

1. 如何确定融合权重 alpha/beta？
2. 当两个检索结果冲突时，如何处理优先级？

---

## Lesson 09 – 元数据过滤

### 🎓 教师讲义要点

* **课程目标**：基于文档元数据进行检索过滤，提高精度。
* **技术重点**：

  * 支持动态复合查询条件
  * 标签、作者、日期等元数据索引
  * 索引优化策略
  * 查询性能监控

### 🧪 学生实验指导

```python
# 假设文档有字段：category, author, created_at
query = "人工智能"
filters = {"category": "AI", "author": "张三"}

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT * FROM document
        WHERE to_tsvector('chinese', content) @@ plainto_tsquery('chinese', :query)
          AND category = :category
          AND author = :author
        ORDER BY ts_rank(to_tsvector('chinese', content), plainto_tsquery('chinese', :query)) DESC
        LIMIT 5;
    """), {**filters, "query": query})
    for row in result:
        print(row)
```

### 🤔 思考题

1. 元数据过滤如何影响检索召回率？
2. 如何在大规模数据中保持过滤查询性能？

---

## Lesson 10 – 重排序(Rerank)接入

### 🎓 教师讲义要点

* **课程目标**：集成重排序模型提升检索精度。
* **技术重点**：

  * bge-reranker-v2-m3 模型调用
  * 前向/批量重排序
  * 缓存策略与性能平衡
  * 结果可解释性

### 🧪 学生实验指导

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("bge-reranker-v2-m3")

# 对 top-10 检索结果进行重排序
pairs = [(query, doc["content"]) for doc in final_results[:10]]
scores = reranker.predict(pairs)

reranked = sorted(zip(final_results[:10], scores), key=lambda x: x[1], reverse=True)
for doc, score in reranked:
    print(doc["id"], score)
```

### 🤔 思考题

1. 为什么 Rerank 可以显著提升结果精度？
2. 如何处理大规模检索结果的 rerank 性能问题？

---

## Lesson 11 – Chunk尺寸与重叠实验

### 🎓 教师讲义要点

* **课程目标**：优化文本分块策略，提高检索与生成效果。
* **技术重点**：

  * 动态分块算法
  * 重叠策略优化
  * 分块质量评估
  * 参数调优实验

### 🧪 学生实验指导

```python
chunk_sizes = [200, 400, 600]
overlaps = [50, 100, 150]

for size in chunk_sizes:
    for overlap in overlaps:
        temp_chunks = []
        for page in doc:
            text = page.get_text()
            for i in range(0, len(text), size - overlap):
                temp_chunks.append(text[i:i+size])
        print(f"Chunk size {size}, overlap {overlap}, total chunks: {len(temp_chunks)}")
```

### 🤔 思考题

1. Chunk大小与重叠比例对检索精度和召回率的影响？
2. 如何自动选择最优分块策略？

---

## Lesson 12 – 多文档源处理

### 🎓 教师讲义要点

* **课程目标**：支持多文档格式和来源，统一处理接口。
* **技术重点**：

  * PDF、Word、TXT、HTML 文档解析
  * 批量处理优化
  * 错误处理机制与日志
  * 接口统一化

### 🧪 学生实验指导

```python
from pathlib import Path
import fitz  # PDF
import docx  # Word

for file in Path("docs/").glob("*.*"):
    if file.suffix == ".pdf":
        doc = fitz.open(file)
        text = "".join([page.get_text() for page in doc])
    elif file.suffix == ".docx":
        doc = docx.Document(file)
        text = "\n".join([p.text for p in doc.paragraphs])
    else:
        text = file.read_text()
    # 文本分块入库
```

### 🤔 思考题

1. 如何保证多种文档格式处理一致性？
2. 批量处理大规模文件时的性能优化方法？

---

## Lesson 13 – 引用与可溯源输出

### 🎓 教师讲义要点

* **课程目标**：在回答中提供引用和来源，增强可信度。
* **技术重点**：

  * 引用链追踪
  * 置信度计算
  * 可溯源信息展示
  * 与生成结果结合

### 🧪 学生实验指导

```python
# 假设每个 chunk 有 metadata: source_file, page_no
answer_chunks = reranked[:3]
for idx, (doc, score) in enumerate(answer_chunks):
    print(f"Answer {idx+1}:")
    print(doc["payload"]["text"])
    print(f"Source: {doc['payload']['source_file']}, Page: {doc['payload']['page_no']}, Score: {score}")
```

### 🤔 思考题

1. 如何设计可溯源结构以便于生成端使用？
2. 多来源文档中，如何合并冲突信息？

---

## Lesson 14 – 缓存策略

### 🎓 教师讲义要点

* **课程目标**：通过多层缓存提升系统性能。
* **技术重点**：

  * Redis 多级缓存
  * 缓存失效与更新策略
  * 缓存命中率优化
  * 分布式缓存设计

### 🧪 学生实验指导

```python
import redis

cache = redis.Redis(host='localhost', port=6379, db=0)

# 检查缓存
key = f"query:{query}"
if cache.exists(key):
    result = cache.get(key)
else:
    result = compute_results(query)
    cache.set(key, str(result), ex=3600)  # 1小时过期
```

### 🤔 思考题

1. 缓存策略如何平衡实时性与性能？
2. 分布式环境下，如何保证缓存一致性？

---

我已经完成 **模块B（Lesson 07–14）详细讲义 + 实验指导**。

如果你确认格式和深度，可以让我继续整理 **模块C（Lesson 15–20）**，然后逐模块完成直到 Lesson 40。

你希望我继续模块C吗？
