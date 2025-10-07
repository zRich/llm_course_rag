好的，我们继续，下面是 **模块C（Ingestion工程化，Lesson 15–20）** 的详细讲义 + 学生实验指导 + 思考题版本。

---

# 模块C：Ingestion工程化（Lesson 15–20）

---

## Lesson 15 – 批量/断点续传

### 🎓 教师讲义要点

* **课程目标**：实现大规模文档的批量处理与断点续传，提高系统稳定性。
* **技术重点**：

  * 异步批量处理（asyncio + FastAPI Background Tasks）
  * 断点续传机制设计
  * 处理进度跟踪与日志记录
  * 错误恢复策略

### 🧪 学生实验指导

```python
from fastapi import BackgroundTasks
import asyncio

async def process_document(doc_id):
    # 模拟处理
    await asyncio.sleep(0.1)
    print(f"Processed {doc_id}")

async def batch_process(doc_ids):
    tasks = [process_document(d) for d in doc_ids]
    for i, task in enumerate(asyncio.as_completed(tasks), 1):
        await task
        print(f"Progress: {i}/{len(doc_ids)}")

# FastAPI background task
def start_batch(background_tasks: BackgroundTasks, doc_ids):
    background_tasks.add_task(batch_process, doc_ids)
```

### 🤔 思考题

1. 批量处理大规模文档时，如何保证系统不会因内存过载崩溃？
2. 断点续传的关键点是什么？

---

## Lesson 16 – 增量更新

### 🎓 教师讲义要点

* **课程目标**：实现文档增量更新，提高索引与检索效率。
* **技术重点**：

  * 变更检测算法
  * 增量索引更新（向量库/全文索引）
  * 文档版本控制
  * 冲突解决策略

### 🧪 学生实验指导

```python
# 假设每个文档有 version
def update_index(new_doc):
    existing = get_doc_by_id(new_doc.id)
    if not existing or existing.version < new_doc.version:
        # 更新索引
        index_vector(new_doc)
```

### 🤔 思考题

1. 如何设计文档的版本号与更新策略？
2. 增量更新如何减少系统负载？

---

## Lesson 17 – 结构化数据接入

### 🎓 教师讲义要点

* **课程目标**：支持结构化数据源接入，如数据库或 API。
* **技术重点**：

  * 数据库连接器与查询
  * REST / GraphQL API 数据源集成
  * 数据格式转换与清洗
  * 实时数据同步

### 🧪 学生实验指导

```python
import requests
import pandas as pd

# 从 API 获取数据
response = requests.get("https://api.example.com/data")
data = pd.DataFrame(response.json())

# 转换为 RAG 系统可用格式
for _, row in data.iterrows():
    doc = {"content": row["text"], "metadata": {"source": "api"}}
    index_vector(doc)
```

### 🤔 思考题

1. 如何保证实时数据同步的稳定性和一致性？
2. 结构化数据接入和非结构化文档有何差异？

---

## Lesson 18 – 文本清洗与去噪

### 🎓 教师讲义要点

* **课程目标**：提升文档质量，减少噪声，提高检索和生成效果。
* **技术重点**：

  * 文本预处理算法（去空格、去特殊字符）
  * 噪声检测与清理
  * 文本质量评估指标
  * 自动化清洗流程

### 🧪 学生实验指导

```python
import re

def clean_text(text):
    # 去掉多余空格
    text = re.sub(r'\s+', ' ', text)
    # 去掉非文字字符
    text = re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9.,;!?]', '', text)
    return text

cleaned = [clean_text(doc["content"]) for doc in documents]
```

### 🤔 思考题

1. 清洗过度会不会影响信息完整性？
2. 如何自动评估文本清洗效果？

---

## Lesson 19 – 切分策略插件化

### 🎓 教师讲义要点

* **课程目标**：构建可扩展的文本切分策略系统。
* **技术重点**：

  * 插件化架构设计
  * 策略注册机制
  * 动态策略选择
  * 性能基准测试

### 🧪 学生实验指导

```python
# 定义接口
class ChunkStrategy:
    def split(self, text):
        raise NotImplementedError

# 插件注册
strategies = {}
def register_strategy(name, cls):
    strategies[name] = cls()

# 使用策略
register_strategy("fixed_200", lambda: ChunkStrategy())
chunks = strategies["fixed_200"].split(text)
```

### 🤔 思考题

1. 为什么插件化策略比硬编码策略更灵活？
2. 如何选择最优切分策略？

---

## Lesson 20 – 故障注入与恢复演练

### 🎓 教师讲义要点

* **课程目标**：提升系统可靠性和容错能力。
* **技术重点**：

  * 故障注入框架
  * 自动恢复机制
  * 监控与告警
  * 灾难恢复演练

### 🧪 学生实验指导

```python
import random

def inject_failure():
    if random.random() < 0.1:  # 10% 概率模拟失败
        raise Exception("Simulated failure")

# 使用恢复机制
try:
    inject_failure()
    process_document(doc)
except Exception as e:
    print("Error occurred, retrying...")
    retry_process(doc)
```

### 🤔 思考题

1. 故障注入为什么对系统稳定性测试重要？
2. 如何设计自动恢复策略，保证数据不丢失？

---

我已完成 **模块C（Lesson 15–20）详细讲义 + 实验指导**。

是否现在继续 **模块D（生成控制与防幻觉，Lesson 21–26）**？
