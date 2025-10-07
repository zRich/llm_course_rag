# 模块A：环境与最小可用RAG（Lesson 01–06）

---

## Lesson 01 – 课程导入与环境准备

### 🎓 教师讲义要点

* **课程目标**：让学生掌握企业级 RAG 系统设计与实现，从零到一构建完整架构。
* **技术栈介绍**：Python、FastAPI、Qdrant、PostgreSQL、Redis、MinIO、Docker、Prometheus/Grafana。
* **环境准备**：

  * Python >= 3.10
  * VSCode + Git
  * uv（轻量依赖管理）
  * Docker Desktop 或 Podman
* **最佳实践**：使用 `.env` 管理密钥和环境变量。

### 🧪 学生实验指导

```bash
# 安装 uv
pip install uv

# 新建项目目录并初始化虚拟环境
mkdir rag-course
cd rag-course
uv venv
source .venv/bin/activate

# 安装 FastAPI + Uvicorn
uv pip install fastapi uvicorn
```

创建最小 FastAPI demo：

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def hello():
    return {"msg": "Hello RAG!"}
```

运行：

```bash
uv run uvicorn main:app --reload
```

访问浏览器：[http://127.0.0.1:8000](http://127.0.0.1:8000)

### 🤔 思考题

1. 为什么企业环境中更推荐 uv 或 poetry 来管理依赖，而不是直接 pip install？
2. 本地开发环境与云端部署环境可能会有什么差异？

---

## Lesson 02 – 容器化与依赖服务启动

### 🎓 教师讲义要点

* **课程目标**：构建完整的容器化开发环境，确保服务可复现。
* **依赖服务**：

  * PostgreSQL 17
  * Redis 7
  * Qdrant 1.7
  * MinIO 对象存储
  * InfluxDB 2.7（监控用）
* **核心概念**：Docker Compose、服务依赖管理、端口映射、环境变量注入。

### 🧪 学生实验指导

```yaml
# docker-compose.yml 示例
version: "3.9"
services:
  postgres:
    image: postgres:17
    environment:
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: ragpass
    ports:
      - "5432:5432"
  redis:
    image: redis:7
    ports:
      - "6379:6379"
  qdrant:
    image: qdrant/qdrant:v1.7.0
    ports:
      - "6333:6333"
  minio:
    image: minio/minio
    command: server /data
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: minio123
    ports:
      - "9000:9000"
```

启动服务：

```bash
docker compose up -d
```

### 🤔 思考题

1. 为什么容器化可以提升开发效率和环境一致性？
2. 对于微服务架构，端口管理和网络配置有哪些注意事项？

---

## Lesson 03 – 数据模型与迁移

### 🎓 教师讲义要点

* **课程目标**：设计 RAG 系统核心数据模型，并能完成数据库迁移。
* **技术重点**：

  * SQLModel 0.0.14+
  * PostgreSQL 数据库设计
  * 数据迁移脚本管理
  * 连接池配置优化

### 🧪 学生实验指导

```python
# app/data/models.py
from sqlmodel import SQLModel, Field
from datetime import datetime

class Document(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    title: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

迁移与初始化：

```bash
# 安装依赖
uv pip install sqlmodel psycopg2-binary

# 初始化数据库
python
>>> from app.data.models import SQLModel, engine
>>> SQLModel.metadata.create_all(engine)
```

### 🤔 思考题

1. 为什么要使用 ORM 而不是直接写 SQL？
2. 数据库迁移和版本控制有什么最佳实践？

---

## Lesson 04 – PDF解析与Chunk拆分

### 🎓 教师讲义要点

* **课程目标**：实现文档解析和文本分块处理，为向量化做准备。
* **技术重点**：

  * PyMuPDF 1.23+ 文档解析
  * 文档结构识别（标题、段落、表格）
  * 元数据提取
  * Chunk大小与重叠策略

### 🧪 学生实验指导

```python
import fitz  # PyMuPDF

doc = fitz.open("sample.pdf")
chunks = []

for page in doc:
    text = page.get_text()
    # 简单分块：每500字符为一块
    for i in range(0, len(text), 500):
        chunks.append(text[i:i+500])
```

### 🤔 思考题

1. Chunk大小和重叠比例如何影响检索精度？
2. 文档结构信息如何在向量化中利用？

---

## Lesson 05 – Embedding与向量入库

### 🎓 教师讲义要点

* **课程目标**：实现文本向量化并存入向量数据库。
* **技术重点**：

  * sentence-transformers 2.2+
  * bge-m3 模型调用
  * Qdrant 向量存储
  * 批量向量化处理

### 🧪 学生实验指导

```python
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

model = SentenceTransformer("bge-m3")
qdrant = QdrantClient(host="localhost", port=6333)

vectors = [model.encode(chunk) for chunk in chunks]

qdrant.upsert(
    collection_name="documents",
    points=[
        {"id": i, "vector": vec.tolist(), "payload": {"text": chunks[i]}}
        for i, vec in enumerate(vectors)
    ]
)
```

### 🤔 思考题

1. 为什么向量化是RAG检索的核心？
2. 如何衡量向量化质量？

---

## Lesson 06 – 最小检索与生成（MVP RAG）

### 🎓 教师讲义要点

* **课程目标**：实现基础RAG系统，完成向量检索与LLM生成结合。
* **技术重点**：

  * 向量相似度检索（余弦/点积）
  * 基础LLM调用与Prompt设计
  * 结果格式化输出
  * 简单接口测试

### 🧪 学生实验指导

```python
# 检索
query_vec = model.encode("查询文本")
hits = qdrant.search(collection_name="documents", query_vector=query_vec, limit=5)
for hit in hits:
    print(hit.payload["text"])

# 调用LLM生成
from openai import OpenAI
client = OpenAI(api_key="YOUR_KEY")
prompt = "根据以下文档回答问题:\n" + "\n".join([hit.payload["text"] for hit in hits])
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role":"user","content":prompt}]
)
print(response.choices[0].message.content)
```

### 🤔 思考题

1. 如何设计Prompt以提高生成答案质量？
2. 向量检索结果和生成答案如何结合衡量准确性？
