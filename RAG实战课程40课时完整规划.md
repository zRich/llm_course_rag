# RAG实战课程40课时完整规划

> **课程目标**：从零开始构建生产级 RAG 系统，掌握工程化实践的核心流程
> 
> **适用对象**：大四学生，具备基础 Python 编程能力
> 
> **技术栈**：FastAPI + PostgreSQL + Qdrant + Docker + LLM API
> 
> **最终交付**：完整的 RAG 系统，包含文档处理、向量检索、混合搜索、工程化部署等核心功能
> 
> **课时安排**：
> - 全部20课（Lesson 1-20）：每课2课时，深入学习核心概念和基础技术

---

# RAG实战课程项目指南

## 项目简介

本项目是RAG实战课程的配套实践项目，采用现代化的技术栈和架构设计，旨在帮助学生从零开始构建生产级RAG系统，掌握工程化实践的核心流程。

## 技术架构

### 核心技术栈

- **Python 3.12.x**: 稳定的Python版本，提供良好的性能和类型支持
- **FastAPI**: 高性能异步Web框架
- **PostgreSQL 17.5**: 主数据库，支持向量扩展
- **Qdrant**: 专业向量数据库
- **Redis**: 缓存和会话管理
- **Docker**: 容器化部署
- **uv**: 现代Python包管理器

### 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   用户接口层     │    │    业务逻辑层    │    │    数据存储层    │
│                │    │                │    │                │
│ • REST API     │────│ • 文档处理      │────│ • PostgreSQL   │
│ • API文档      │    │ • 向量检索      │    │ • Qdrant       │
│ • 健康检查     │    │ • 混合搜索      │    │ • Redis        │
│                │    │ • 生成回答      │    │                │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 项目结构

```
rag-system/
├── src/                    # 源代码目录
│   ├── api/                # API接口
│   │   └── health.py       # 健康检查接口
│   ├── core/               # 核心业务逻辑
│   ├── models/             # 数据模型
│   ├── services/           # 业务服务
│   └── utils/              # 工具函数
├── tests/                  # 测试代码
├── scripts/                # 脚本文件
│   └── verify_environment.py  # 环境验证脚本
├── data/                   # 数据目录
│   ├── raw/               # 原始数据
│   ├── processed/         # 处理后数据
│   └── embeddings/        # 向量数据
├── logs/                   # 日志文件
├── pyproject.toml         # 项目配置
├── .env.example           # 环境变量模板
├── .gitignore             # Git忽略文件
├── docker-compose.yml     # Docker编排文件
└── README.md              # 项目说明
```

## 学习环境搭建

### 1. 环境准备

确保系统已安装以下软件：
 - Python 3.12.x
- Docker & Docker Compose
- Git
- uv (Python包管理器)

### 2. 环境验证

运行环境验证脚本：

```bash
python scripts/verify_environment.py
```

### 3. 项目初始化

```bash
# 创建项目目录
mkdir rag-system
cd rag-system

# 初始化项目
uv init

# 复制环境变量文件
cp .env.example .env

# 安装依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows
```

### 4. 启动开发服务

#### 启动API服务

```bash
# 启动FastAPI开发服务器
uvicorn src.api.health:app --reload --host 0.0.0.0 --port 8000
```

#### 启动依赖服务

```bash
# 使用Docker Compose启动数据库等服务
docker-compose up -d postgres qdrant redis
```

### 5. 验证环境

访问以下端点验证服务状态：

- API文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health
- 根路径: http://localhost:8000/

## 开发指南

### 代码规范

项目使用以下工具确保代码质量：

```bash
# 代码格式化
black src/ tests/

# 导入排序
isort src/ tests/

# 类型检查
mypy src/

# 运行测试
pytest tests/
```

### 环境变量配置

复制 `.env.example` 为 `.env` 并根据学习环境修改配置：

```bash
# 数据库配置
DATABASE_URL=postgresql://raguser:ragpass@localhost:5432/ragdb

# 向量数据库配置
QDRANT_URL=http://localhost:6333

# 缓存配置
REDIS_URL=redis://localhost:6379

# AI模型配置（可选）
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## 课程实践指导

### 模块A：环境与最小可用RAG

1. **项目初始化**：使用uv创建项目结构
2. **FastAPI入门**：实现基础API和健康检查
3. **Docker基础**：启动PostgreSQL和Qdrant服务
4. **数据模型**：使用SQLModel定义Document和Chunk模型
5. **PDF解析**：实现文档解析和分块功能
6. **向量检索**：实现embedding和基础检索功能

### 模块B：检索强化与混合搜索

7. **关键词检索**：实现全文检索功能
8. **向量优化**：集成Qdrant向量数据库
9. **混合检索**：实现稠密和稀疏检索融合
10. **重排序**：理解和实现rerank机制
11. **分块优化**：实验不同的分块策略
12. **多源检索**：实现并行多数据源检索
13. **引用溯源**：实现可溯源的答案生成
14. **缓存策略**：实现查询缓存机制

### 模块C：Ingestion工程化

15. **批量处理**：实现异步批量文档处理
16. **断点续传**：实现幂等性和错误恢复
17. **结构化数据**：支持CSV和SQL数据接入
18. **文本清洗**：实现自动化文本预处理
19. **增量更新**：实现文档版本管理和增量索引
20. **监控运维**：实现系统监控和故障恢复

## 学习资源

### 推荐阅读

- FastAPI官方文档
- SQLModel使用指南
- Qdrant向量数据库文档
- Docker容器化最佳实践

### 实践建议

1. **循序渐进**：按照课程模块顺序逐步实现功能
2. **动手实践**：每个概念都要通过代码验证
3. **记录笔记**：记录遇到的问题和解决方案
4. **测试驱动**：为每个功能编写测试用例
5. **代码审查**：定期检查和优化代码质量

## 常见问题

### 环境配置问题

**Q: Python版本不匹配怎么办？**
 A: 使用pyenv或conda管理Python版本，确保使用3.12.x版本。

**Q: Docker服务启动失败？**
A: 检查端口占用情况，确保5432、6333、6379端口可用。

**Q: uv安装依赖失败？**
A: 尝试使用pip作为备选方案，或检查网络连接。

### 开发问题

**Q: API无法访问？**
A: 检查防火墙设置和端口配置，确保8000端口开放。

**Q: 数据库连接失败？**
A: 验证数据库服务状态和连接字符串配置。

**Q: 向量检索结果不准确？**
A: 检查embedding模型和向量维度配置。

## 项目交付标准

### 检查点1（第6课）
- 完整的项目结构
- 可运行的FastAPI服务
- 基础的PDF解析和向量检索功能
- 简单的问答接口

### 检查点2（第14课）
- 混合检索系统
- 重排序功能
- 可溯源的答案生成
- 性能优化和缓存

### 检查点3（第20课）
- 完整的工程化RAG系统
- 批量处理和增量更新
- 监控和故障恢复
- 完整的文档和测试

## 总结

通过本课程的学习和实践，学生将掌握：

1. **技术栈应用**：熟练使用现代Python技术栈
2. **系统设计**：理解RAG系统的架构和组件
3. **工程实践**：掌握软件工程的最佳实践
4. **问题解决**：具备独立解决技术问题的能力
5. **项目管理**：了解项目开发的完整流程

这些技能将为学生在AI应用开发领域的职业发展奠定坚实基础。

## 课程总览

| 模块 | 课程范围 | 课时数 | 核心目标 |
|------|----------|--------|----------|
| A | Lesson 1-6 | 12 课时 | 环境与最小可用 RAG |
| B | Lesson 7-14 | 16 课时 | 检索强化与混合搜索 |
| C | Lesson 15-20 | 12 课时 | Ingestion 工程化 |
| **总计** | **20 课** | **40 课时** | **完整 RAG 系统** |

下面是每课内容（目标 → 课堂活动 → 课后作业/交付物 → 难度 | 典型坑）。

# 模块 A：环境与最小可用 RAG（12 课时，Lesson 01–06）

> 目标：把学生带到"能跑起来的最小闭环"——FastAPI 服务 + 本地向量检索 + 简单 LLM（echo）。
> **每课2课时**，充分时间进行理论讲解、实践操作和问题解决。

1. **Lesson 01 — 项目与工具链入门（初始化）**

   * 目标：建立统一项目模板、介绍 uv、pyproject、基本目录结构。
   * 课堂活动：
     - 理论讲解：企业级RAG系统架构概述、技术栈介绍
     - 实践操作：`uv init`，建立 repo、README、.env.example、基础依赖
     - 问题解答：环境配置问题排查
   * 作业：把本地环境与 repo 同步（提交 scaffold）。
   * 难度：低。坑：uv 使用细节、Python 版本不对。

2. **Lesson 02 — FastAPI 快速上手（API 骨架）**

   * 目标：写第一个 API：`GET /health`、`/api/docs`。
   * 课堂活动：
     - 理论讲解：FastAPI框架特性、RESTful API设计原则
     - 实践操作：建立 `app.main`、路由结构、启动 uvicorn、测试API
     - 问题解答：端口配置、调试技巧
   * 作业：补完 README 的启动步骤。交付：可访问 `/docs`。
   * 难度：低。坑：端口被占用、uvicorn 启动参数。

3. **Lesson 03 — 本地依赖服务入门（Docker 基础）**

   * 目标：掌握 Docker 与 docker-compose 的基本命令（镜像、容器、卷）。
   * 课堂活动：
     - 理论讲解：容器化概念、Docker架构、服务编排原理
     - 实践操作：讲解 `docker-compose.yml` 结构，运行 `docker compose up`，服务健康检查
     - 问题解答：资源配置、网络问题排查
   * 作业：启动 Postgres + Qdrant（或仅 Postgres+Qdrant）。交付：能访问 Postgres、Qdrant 的端口。
   * 难度：低→中。坑：资源限制、端口映射、权限。

4. **Lesson 04 — 最小数据模型 & DB 连接**

   * 目标：用 SQLModel 定义 Document/Chunk/EvalCase，能建表并做简单 CRUD。
   * 课堂活动：
     - 理论讲解：ORM概念、SQLModel特性、数据库设计原则
     - 实践操作：写 model、`init_db`、`get_session`、CRUD操作演示
     - 问题解答：连接池配置、SQL调试技巧
   * 作业：写一个小脚本插入 sample document；交付：数据库可查询到记录。
   * 难度：中。坑：SQLAlchemy URL 配置、迁移可以先用简单脚本（不强制 alembic）。

5. **Lesson 05 — PDF 解析与 chunk 基础**

   * 目标：使用 PyMuPDF 提取文本并实现 `paragraph/title/window` 三种切分（课堂提供工具函数）。
   * 课堂活动：
     - 理论讲解：文档解析原理、分块策略对比、元数据提取
     - 实践操作：演示解析一页 PDF 并返回 chunks；在 API 中实现 `POST /ingest/pdf`（返回 chunk 数）
     - 问题解答：中文处理、噪声过滤技巧
   * 作业：对一份样例 PDF 测试三种策略，提交 chunk 数对比截图。
   * 难度：低→中。坑：中文换行/页眉页脚噪声、编码问题。

6. **Lesson 06 — Embedding + 本地最小索引（MVP）**

   * 目标：引入 sentence-transformers 做 embedding；把向量写入（本节可先用内存列表或 SQLite 代替 Qdrant，降低部署门槛）。
   * 课堂活动：
     - 理论讲解：向量化原理、embedding模型选择、相似度计算
     - 实践操作：实现 `Embedder.encode()`、`POST /ingest/index`（将未索引 chunks 向量化并写入本地 store）
     - 问题解答：模型优化、向量存储方案
   * 作业：用小数据集跑通 Query→embedding→nearest。交付：`/rag/query` 返回 echo 模式答案。
   * 难度：中。坑：模型下载慢（提醒离线/镜像方案）、向量维度约定。

**检查点 1（第 6 课）**：学生能完成 **上传 PDF → 切分 → 向量化 → 最小检索 → echo 返回** 的闭环。

---

# 模块 B：检索强化与混合搜索（16 课时，Lesson 07–14）

> 目标：在最小闭环上加入关键词检索、混合策略与重排序概念（先做轻量实现，再扩展到 Qdrant+FTS）。
> **每课2课时**，深入理解检索优化原理和实践应用。

7. **Lesson 07 — 关键词检索（FTS 概念与实现）**

   * 目标：理解全文检索原理。课堂用 SQLite FTS5 或 Postgres tsvector（视环境选择 SQLite 以降低难度）。
   * 课堂活动：
     - 理论讲解：全文检索原理、倒排索引、TF-IDF算法
     - 实践操作：创建 `rag_index` FTS 表，演示 MATCH 查询，写 `/search/keyword`
     - 问题解答：中文分词处理、FTS配置优化
   * 作业：给 10 条 chunk 写关键词检索示例，提交查询效果对比。
   * 难度：中。坑：中文分词（可先用简单空格分词 + 后续补中文分词库），FTS 配置差异。

8. **Lesson 08 — 向量检索回头优化（Qdrant 简介 + 配置）**

   * 目标：把内存/SQLite 向量替换成 Qdrant（或继续用内存作为轻量可选项）。
   * 课堂活动：
     - 理论讲解：向量数据库架构、HNSW索引原理、性能对比
     - 实践操作：Qdrant client upsert/search，collection schema，dim 对齐
     - 问题解答：性能调优、监控配置
   * 作业：将 ingest 的 chunks 写入 Qdrant，验证搜索结果。
   * 难度：中。坑：Qdrant 启动、网络/端口、维度不对。

9. **Lesson 09 — 混合检索理论（score 融合）**

   * 目标：介绍混合检索融合方法与权重 α 的含义。
   * 课堂活动：
     - 理论讲解：稠密检索vs稀疏检索、融合算法、权重优化理论
     - 实践操作：实现简单融合函数 `score = α*dense + (1-α)*sparse`，演示调参影响
     - 问题解答：归一化策略、参数调优技巧
   * 作业：在 5 个查询上尝试 α=0.3/0.5/0.7 并记录差异。
   * 难度：中。坑：归一化步骤（记得对 dense/sparse 做尺度归一化）。

10. **Lesson 10 — 基本 Rerank 思路（跨编码器概念）**

    * 目标：理解 reranker 为何提高质量与其计算成本（先理论+demo）。
    * 课堂活动：
      - 理论讲解：重排序原理、Cross-encoder架构、计算成本分析
      - 实践操作：演示 cross-encoder 的输入输出（不强制在学生机器上跑大模型，课堂提供小样例或 echo stub）
      - 问题解答：模型选择、性能权衡
    * 作业：写伪代码/流程图描述 rerank 在 pipeline 的位置。
    * 难度：低。坑：不做成"模型必须跑通"的硬要求，只理解流程。

11. **Lesson 11 — Chunk 大小与重叠实验（小规模）**

    * 目标：量化 chunk 尺寸/overlap 的影响（用小数据集跑实验）。
    * 课堂活动：
      - 理论讲解：分块策略理论、重叠机制、实验设计方法
      - 实践操作：学生分组对比 3 个配置（短/中/长），记录召回/延迟
      - 问题解答：实验结果分析、参数选择策略
    * 作业：提交实验结果表格与结论。
    * 难度：中。坑：统计样本太少会导致结论不稳定。

12. **Lesson 12 — 多数据源检索（路由与去重）**

    * 目标：实现多源并行检索（例如 product DB + PDF + FAQ）与去重逻辑。
    * 课堂活动：
      - 理论讲解：多源检索架构、并行处理原理、去重算法
      - 实践操作：演示并行请求（async）实现，做简单去重（文本哈希）
      - 问题解答：异常处理、性能优化
    * 作业：实现一条多源查询（提交日志与时间对比）。
    * 难度：中→高（并行 IO）。坑：并行异常处理、超时。

13. **Lesson 13 — 引用与可溯源（证据返回）**

    * 目标：让模型返回"引用信息"（来源、页码、score）。
    * 课堂活动：
      - 理论讲解：可溯源性原理、引用格式设计、信任度评估
      - 实践操作：设计返回 schema（answer + citations[]），并在 `/rag/query` 中实现
      - 问题解答：引用准确性验证、格式标准化
    * 作业：为 3 个 query 输出引用链并手工核验。
    * 难度：中。坑：payload 过大需要截断。

14. **Lesson 14 — 缓存策略（Redis 简化教学）**

    * 目标：实现热门查询缓存（可用内存 dict 作替代）并讨论 Redis 的优势。
    * 课堂活动：
      - 理论讲解：缓存架构设计、Redis特性、缓存策略对比
      - 实践操作：实现简单 LRU 或 TTL 缓存中间件，记录命中率
      - 问题解答：缓存优化、监控配置
    * 作业：配置 TTL 并证明命中率提高。
    * 难度：中。坑：缓存与权限/多租户冲突（课堂提点）。

**检查点 2（第 14 课）**：学生应能完成 **混合检索 + 基本 rerank 思路 + 可溯源输出**，有具体验证结果（实验报告）。

---

# 模块 C：Ingestion 工程化（12 课时，Lesson 15–20）

> 目标：把单次 ingest 扩展为可靠、可重试的工程化流程（支持批量、增量与插件化策略）。
> **每课2课时**，深入学习工程化数据处理流程。

15. **Lesson 15 — 批量处理与异步任务入门**

    * 目标：讲解异步/后台任务（celery / RQ / 简化的 asyncio worker）。课堂采用简化的 asyncio queue 实现。
    * 课堂活动：
      - 理论讲解：异步编程原理、任务队列架构、并发控制
      - 实践操作：实现 `POST /ingest/batch`，把大文件拆成小任务推入队列
      - 问题解答：内存管理、错误处理
    * 作业：在本地触发 500 条批量任务，观察队列处理。
    * 难度：中。坑：并发控制与内存泄露。

16. **Lesson 16 — 断点续传与幂等性设计**

    * 目标：保证中断后可继续（用对象的 checksum 或数据库 flag 做幂等）。
    * 课堂活动：
      - 理论讲解：幂等性原理、断点续传机制、事务处理
      - 实践操作：实现幂等判定（file_id + chunk_index）与进度表
      - 问题解答：数据一致性、恢复策略
    * 作业：模拟中断与恢复流程并提交日志。
    * 难度：中。坑：事务与重复写入处理。

17. **Lesson 17 — 结构化数据接入（CSV / SQL）**

    * 目标：设计从 CSV/SQL 到 chunk 的转化流程与字段映射。
    * 课堂活动：
      - 理论讲解：结构化数据处理、字段映射策略、数据验证
      - 实践操作：实现 `POST /ingest/csv`（把行转摘要 + chunk + metadata）
      - 问题解答：数据清洗、格式标准化
    * 作业：接入一个示例 CSV（产品表），构建索引并查询。
    * 难度：中。坑：字段标准化（单位/日期）。

18. **Lesson 18 — 文本清洗与去噪（自动化）**

    * 目标：实现基本规则的去噪（页眉/页脚/重复行/非文字噪声）。
    * 课堂活动：
      - 理论讲解：文本预处理原理、噪声识别、清洗策略
      - 实践操作：正则/规则链实现、可视化对比（原文 vs 清洗后）
      - 问题解答：规则优化、效果评估
    * 作业：提交 3 个清洗规则并评估效果。
    * 难度：低→中。坑：过度清洗导致信息丢失。

19. **Lesson 19 — 增量更新与版本管理**

    * 目标：实现文档版本控制与增量索引更新（避免全量重建）。
    * 课堂活动：
      - 理论讲解：版本控制原理、增量更新策略、冲突处理
      - 实践操作：实现文档版本表、增量检测与索引更新
      - 问题解答：数据一致性、性能优化
    * 作业：模拟文档更新场景并验证增量索引。
    * 难度：中→高。坑：并发更新冲突、索引一致性。

20. **Lesson 20 — 监控与故障恢复**

    * 目标：实现基本的系统监控、日志记录与故障恢复机制。
    * 课堂活动：
      - 理论讲解：系统监控原理、故障检测、恢复策略
      - 实践操作：实现健康检查、错误日志、自动重试机制
      - 问题解答：监控指标、告警配置
    * 作业：配置完整的监控系统并模拟故障恢复。
    * 难度：中。坑：监控开销、误报处理。

**检查点 3（第 20 课）**：学生应能完成 **完整的工程化 RAG 系统**，包含批量处理、增量更新、监控告警等生产级功能。

---

## 课程总结

通过这 20 课的学习，学生将掌握：

1. **基础架构**：FastAPI + PostgreSQL + Qdrant 的完整技术栈
2. **核心功能**：文档解析、向量化、混合检索、重排序等 RAG 核心技术
3. **工程化实践**：批量处理、增量更新、监控告警等生产级功能
4. **实战经验**：通过 3 个检查点的验收，确保每个阶段都有可运行的系统

最终交付的 RAG 系统具备完整的生产级功能，为学生进入企业级 AI 应用开发打下坚实基础。

---

# RAG实战课程统一技术栈规范

## 1. 概述

本文档定义了RAG实战课程的技术栈标准，确保所有课程模块使用一致的技术版本和配置，提高教学质量和学生学习体验。

### 1.1 设计原则

- **一致性**: 所有课程使用相同的技术栈版本
- **稳定性**: 选择经过验证的稳定版本
- **实用性**: 贴近企业实际使用的技术组合
- **可维护性**: 便于课程维护和更新
- **学习友好**: 降低学生环境配置难度

### 1.2 适用范围

- RAG实战课程所有模块（A、B、C）
- 课程示例代码和项目模板
- 学生作业和项目实践
- 课程配套工具和脚本

## 2. Python运行时环境

### 2.1 Python版本

```yaml
# 标准配置
 Python: 3.12.x (推荐 3.12.x)
 最低要求: 3.12.0
 支持: 3.12.x (主要依赖均已兼容)
```

**选择理由**:
- Python 3.11 性能提升显著（比3.10快10-60%）
- 生态系统成熟，主要依赖都已支持
- 企业广泛采用的稳定版本
- 避免3.12的兼容性问题

### 2.2 包管理器

```yaml
# 主要工具
uv: latest (推荐，极速包管理)
pip: 23.3+ (备选)

# 虚拟环境
venv: Python内置
conda: 可选（适合Windows用户）
```

**uv优势**:
- 安装速度比pip快10-100倍
- 更好的依赖解析
- 内置虚拟环境管理
- Rust编写，稳定可靠

## 3. Web框架与API

### 3.1 核心框架

```toml
[dependencies]
# Web框架
fastapi = "0.104.1"
uvicorn = {extras = ["standard"], version = "0.24.0"}

# ASGI服务器
gunicorn = "21.2.0"  # 生产环境
```

### 3.2 中间件与扩展

```toml
# HTTP客户端
httpx = "0.25.2"
aiohttp = "3.9.1"

# 跨域支持
fastapi-cors = "0.0.6"

# 请求验证
slowapi = "0.1.9"  # 限流
```

**配置示例**:
```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="RAG API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 4. 数据验证与序列化

### 4.1 数据模型

```toml
[dependencies]
# 数据验证（V2版本）
pydantic = "2.5.0"
pydantic-settings = "2.1.0"

# 类型提示增强
typing-extensions = "4.8.0"
```

### 4.2 配置管理

```python
# app/core/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # 数据库配置
    database_url: str = "postgresql://user:pass@localhost/ragdb"
    
    # Redis配置
    redis_url: str = "redis://localhost:6379/0"
    
    # Qdrant配置
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    
    # AI模型配置
    openai_api_key: Optional[str] = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

## 5. ORM与数据库

### 5.1 ORM框架

```toml
[dependencies]
# ORM（基于SQLAlchemy 2.0）
sqlmodel = "0.0.14"
sqlalchemy = "2.0.23"

# 数据库驱动
psycopg2-binary = "2.9.9"  # PostgreSQL
aiosqlite = "0.19.0"       # SQLite（开发/测试）

# 迁移工具
alembic = "1.13.1"
```

### 5.2 数据库模型示例

```python
# app/models/document.py
from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from datetime import datetime
from uuid import UUID, uuid4

class DocumentBase(SQLModel):
    title: str = Field(max_length=500)
    content: str
    source_type: str = Field(max_length=50)  # pdf, txt, url, etc.
    source_path: str = Field(max_length=1000)
    metadata_: Optional[dict] = Field(default=None, sa_column_kwargs={"name": "metadata"})
    
class Document(DocumentBase, table=True):
    __tablename__ = "documents"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    # 关联关系
    chunks: List["Chunk"] = Relationship(back_populates="document")

class DocumentCreate(DocumentBase):
    pass

class DocumentRead(DocumentBase):
    id: UUID
    created_at: datetime
    updated_at: Optional[datetime]
```

## 6. 关系型数据库

### 6.1 PostgreSQL配置

```yaml
# 推荐版本
PostgreSQL: 15.x (推荐 15.5+)
最低要求: 14.0

# 必需扩展
- pg_vector: 0.5.1+ (向量支持)
- pg_trgm: 内置 (模糊匹配)
- btree_gin: 内置 (复合索引)
```

**Docker配置**:
```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: ragdb
      POSTGRES_USER: raguser
      POSTGRES_PASSWORD: ragpass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    command: >
      postgres
      -c shared_preload_libraries=vector
      -c max_connections=200
      -c shared_buffers=256MB
      -c effective_cache_size=1GB

volumes:
  postgres_data:
```

### 6.2 数据库初始化

```sql
-- init.sql
-- 创建扩展
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- 创建用户和权限
CREATE USER raguser WITH PASSWORD 'ragpass';
GRANT ALL PRIVILEGES ON DATABASE ragdb TO raguser;
GRANT ALL ON SCHEMA public TO raguser;
```

## 7. 缓存系统

### 7.1 Redis配置

```toml
[dependencies]
# Redis客户端
redis = {extras = ["hiredis"], version = "5.0.1"}
aioredis = "2.0.1"

# 缓存框架
fastapi-cache2 = {extras = ["redis"], version = "0.2.1"}
```

**Docker配置**:
```yaml
# docker-compose.yml (Redis部分)
redis:
  image: redis:7.2-alpine
  ports:
    - "6379:6379"
  volumes:
    - redis_data:/data
    - ./redis.conf:/usr/local/etc/redis/redis.conf
  command: redis-server /usr/local/etc/redis/redis.conf
```

### 7.2 缓存配置

```python
# app/core/cache.py
import aioredis
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

async def init_cache():
    redis = aioredis.from_url(
        "redis://localhost:6379",
        encoding="utf8",
        decode_responses=True,
        max_connections=20,
        retry_on_timeout=True
    )
    FastAPICache.init(RedisBackend(redis), prefix="rag-cache")
```

## 8. 向量数据库

### 8.1 Qdrant配置

```toml
[dependencies]
# Qdrant客户端
qdrant-client = "1.7.0"
```

**Docker配置**:
```yaml
# docker-compose.yml (Qdrant部分)
qdrant:
  image: qdrant/qdrant:v1.7.0
  ports:
    - "6333:6333"
    - "6334:6334"
  volumes:
    - qdrant_data:/qdrant/storage
  environment:
    QDRANT__SERVICE__HTTP_PORT: 6333
    QDRANT__SERVICE__GRPC_PORT: 6334
```

### 8.2 向量存储配置

```python
# app/core/vector_store.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List
import uuid

class VectorStore:
    def __init__(self, url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=url)
        self.collection_name = "rag_vectors"
        
    async def init_collection(self, vector_size: int = 384):
        """初始化向量集合"""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
        except Exception as e:
            print(f"Collection already exists or error: {e}")
    
    async def upsert_vectors(self, vectors: List[List[float]], 
                           payloads: List[dict]) -> bool:
        """插入或更新向量"""
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )
            for vector, payload in zip(vectors, payloads)
        ]
        
        result = self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return result.status == "ok"
```

## 9. 对象存储

### 9.1 MinIO配置

```toml
[dependencies]
# S3兼容存储
minio = "7.2.0"
boto3 = "1.34.0"  # AWS SDK
```

**Docker配置**:
```yaml
# docker-compose.yml (MinIO部分)
minio:
  image: minio/minio:RELEASE.2023-12-07T04-16-00Z
  ports:
    - "9000:9000"
    - "9001:9001"
  environment:
    MINIO_ROOT_USER: minioadmin
    MINIO_ROOT_PASSWORD: minioadmin123
  volumes:
    - minio_data:/data
  command: server /data --console-address ":9001"
```

### 9.2 文件存储服务

```python
# app/core/storage.py
from minio import Minio
from minio.error import S3Error
from typing import Optional
import io

class FileStorage:
    def __init__(self, endpoint: str, access_key: str, secret_key: str):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False  # 开发环境
        )
        self.bucket_name = "rag-documents"
        
    async def init_bucket(self):
        """初始化存储桶"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except S3Error as e:
            print(f"Error creating bucket: {e}")
    
    async def upload_file(self, file_path: str, file_data: bytes) -> bool:
        """上传文件"""
        try:
            self.client.put_object(
                self.bucket_name,
                file_path,
                io.BytesIO(file_data),
                len(file_data)
            )
            return True
        except S3Error as e:
            print(f"Error uploading file: {e}")
            return False
```

## 10. 时序数据库（可选）

### 10.1 InfluxDB配置

```toml
[dependencies]
# 时序数据库（用于监控指标）
influxdb-client = {extras = ["async"], version = "1.38.0"}
```

**使用场景**:
- API响应时间监控
- 向量检索性能指标
- 系统资源使用情况
- 用户查询统计

## 11. AI/ML技术栈

### 11.1 核心ML库

```toml
[dependencies]
# 深度学习框架
torch = "2.1.1"
torchvision = "0.16.1"
torchaudio = "2.1.1"

# Transformers生态
transformers = "4.36.0"
sentence-transformers = "2.2.2"
tokenizers = "0.15.0"

# 科学计算
numpy = "1.24.4"
scipy = "1.11.4"
scikit-learn = "1.3.2"
```

### 11.2 文档处理

```toml
[dependencies]
# PDF处理
PyMuPDF = "1.23.8"  # fitz
pdfplumber = "0.10.3"

# 文档解析
python-docx = "1.1.0"
openpyxl = "3.1.2"
python-pptx = "0.6.23"

# 文本处理
nltk = "3.8.1"
spacy = "3.7.2"
jieba = "0.42.1"  # 中文分词
```

### 11.3 Embedding模型配置

```python
# app/core/embeddings.py
from sentence_transformers import SentenceTransformer
from typing import List
import torch

class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # 检查GPU可用性
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder="./models"  # 本地缓存
        )
        
        # 模型配置
        self.model.max_seq_length = 512
        
    def encode(self, texts: List[str]) -> List[List[float]]:
        """文本向量化"""
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    @property
    def dimension(self) -> int:
        """向量维度"""
        return self.model.get_sentence_embedding_dimension()
```

## 12. 前端技术栈

### 12.1 基础框架

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "typescript": "^5.2.2",
    "vite": "^5.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.2.0"
  }
}
```

### 12.2 UI组件库

```json
{
  "dependencies": {
    "@radix-ui/react-dialog": "^1.0.5",
    "@radix-ui/react-dropdown-menu": "^2.0.6",
    "lucide-react": "^0.294.0",
    "tailwindcss": "^3.3.6",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.0.0",
    "tailwind-merge": "^2.0.0"
  }
}
```

## 13. 开发工具链

### 13.1 代码质量

```toml
[tool.ruff]
# Python代码检查和格式化
 target-version = "py312"
line-length = 88
select = ["E", "W", "F", "I", "B", "C4", "ARG", "SIM"]
ignore = ["E501", "W503", "E203"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.mypy]
# 类型检查
 python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### 13.2 测试框架

```toml
[dependencies]
# 测试工具
pytest = "7.4.3"
pytest-asyncio = "0.21.1"
pytest-cov = "4.1.0"
httpx = "0.25.2"  # 测试HTTP客户端
factory-boy = "3.3.0"  # 测试数据工厂
```

### 13.3 项目配置模板

```toml
# pyproject.toml 完整模板
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "rag-system"
version = "1.0.0"
description = "RAG实战课程项目"
authors = [{name = "Student", email = "student@example.com"}]
readme = "README.md"
 requires-python = ">=3.12"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Education",
    "License :: OSI Approved :: MIT License",
 "Programming Language :: Python :: 3.12",
]

dependencies = [
    # Web框架
    "fastapi==0.104.1",
    "uvicorn[standard]==0.24.0",
    
    # 数据库
    "sqlmodel==0.0.14",
    "psycopg2-binary==2.9.9",
    "alembic==1.13.1",
    
    # 缓存
    "redis[hiredis]==5.0.1",
    
    # 向量数据库
    "qdrant-client==1.7.0",
    
    # AI/ML
    "torch==2.1.1",
    "transformers==4.36.0",
    "sentence-transformers==2.2.2",
    
    # 文档处理
    "PyMuPDF==1.23.8",
    "python-docx==1.1.0",
    
    # 工具库
    "pydantic==2.5.0",
    "pydantic-settings==2.1.0",
    "httpx==0.25.2",
    "python-multipart==0.0.6",
]

[project.optional-dependencies]
dev = [
    "pytest==7.4.3",
    "pytest-asyncio==0.21.1",
    "pytest-cov==4.1.0",
    "ruff==0.1.6",
    "mypy==1.7.1",
    "pre-commit==3.6.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "-v --tb=short --cov=app --cov-report=term-missing"
asyncio_mode = "auto"

[tool.coverage.run]
source = ["app"]
omit = [
    "*/tests/*",
    "*/migrations/*",
    "*/__pycache__/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
]
```

## 14. 监控与运维

### 14.1 日志配置

```python
# app/core/logging.py
import logging
import sys
from pathlib import Path

def setup_logging():
    """配置日志系统"""
    
    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = logging.FileHandler(
        log_dir / "rag_system.log",
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # 根日志器配置
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # 第三方库日志级别
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
```

### 14.2 健康检查

```python
# app/api/health.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_session
from app.core.redis import get_redis
from app.core.vector_store import get_vector_store
import asyncio

router = APIRouter()

@router.get("/health")
async def health_check():
    """基础健康检查"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@router.get("/health/detailed")
async def detailed_health_check(
    db: AsyncSession = Depends(get_session),
    redis = Depends(get_redis),
    vector_store = Depends(get_vector_store)
):
    """详细健康检查"""
    checks = {}
    
    # 数据库检查
    try:
        await db.execute("SELECT 1")
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {str(e)}"
    
    # Redis检查
    try:
        await redis.ping()
        checks["redis"] = "healthy"
    except Exception as e:
        checks["redis"] = f"unhealthy: {str(e)}"
    
    # 向量数据库检查
    try:
        vector_store.client.get_collections()
        checks["vector_store"] = "healthy"
    except Exception as e:
        checks["vector_store"] = f"unhealthy: {str(e)}"
    
    # 整体状态
    overall_status = "healthy" if all(
        status == "healthy" for status in checks.values()
    ) else "unhealthy"
    
    return {
        "status": overall_status,
        "checks": checks,
        "timestamp": datetime.utcnow()
    }
```

## 15. 课程设计理念

### 15.1 渐进式学习

1. **模块A（基础）**: 最小可用系统
   - 单一技术栈，避免复杂性
   - 重点理解核心概念
   - 快速获得成就感

2. **模块B（进阶）**: 性能优化
   - 引入更多技术组件
   - 理解系统瓶颈和优化方法
   - 培养工程思维

3. **模块C（高级）**: 生产级系统
   - 完整技术栈整合
   - 关注可靠性和可维护性
   - 企业级开发实践

### 15.2 技术选型原则

1. **教学友好性**
   - 安装配置简单
   - 文档完善，社区活跃
   - 错误信息清晰

2. **企业相关性**
   - 主流企业广泛使用
   - 有良好的职业发展前景
   - 技能可迁移性强

3. **技术先进性**
   - 代表技术发展方向
   - 性能和功能优势明显
   - 持续更新维护

## 16. 标准配置模板

### 16.1 Docker Compose完整配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  # PostgreSQL数据库
  postgres:
    image: pgvector/pgvector:pg15
    container_name: rag_postgres
    environment:
      POSTGRES_DB: ragdb
      POSTGRES_USER: raguser
      POSTGRES_PASSWORD: ragpass
      POSTGRES_INITDB_ARGS: "--encoding=UTF-8 --lc-collate=C --lc-ctype=C"
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    command: >
      postgres
      -c shared_preload_libraries=vector
      -c max_connections=200
      -c shared_buffers=256MB
      -c effective_cache_size=1GB
      -c work_mem=4MB
      -c maintenance_work_mem=64MB
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U raguser -d ragdb"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis缓存
  redis:
    image: redis:7.2-alpine
    container_name: rag_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
      - ./redis.conf:/usr/local/etc/redis/redis.conf
    command: redis-server /usr/local/etc/redis/redis.conf
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  # Qdrant向量数据库
  qdrant:
    image: qdrant/qdrant:v1.7.0
    container_name: rag_qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT__SERVICE__HTTP_PORT: 6333
      QDRANT__SERVICE__GRPC_PORT: 6334
      QDRANT__LOG_LEVEL: INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # MinIO对象存储
  minio:
    image: minio/minio:RELEASE.2023-12-07T04-16-00Z
    container_name: rag_minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin123
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  # RAG应用服务
  rag-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rag_api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://raguser:ragpass@postgres:5432/ragdb
      - REDIS_URL=redis://redis:6379/0
      - QDRANT_URL=http://qdrant:6333
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=minioadmin
      - MINIO_SECRET_KEY=minioadmin123
    volumes:
      - ./app:/app/app
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      qdrant:
        condition: service_healthy
      minio:
        condition: service_healthy
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
  minio_data:

networks:
  default:
    name: rag_network
```

### 16.2 环境变量模板

```bash
# .env.template
# 复制为 .env 并填入实际值

# =============================================================================
# 数据库配置
# =============================================================================
DATABASE_URL=postgresql://raguser:ragpass@localhost:5432/ragdb
DATABASE_ECHO=false  # 生产环境设为false

# =============================================================================
# Redis配置
# =============================================================================
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=20

# =============================================================================
# 向量数据库配置
# =============================================================================
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # 生产环境必须设置
QDRANT_COLLECTION_NAME=rag_vectors

# =============================================================================
# 对象存储配置
# =============================================================================
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_BUCKET_NAME=rag-documents
MINIO_SECURE=false  # 生产环境设为true

# =============================================================================
# AI模型配置
# =============================================================================
# OpenAI API（可选）
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1

# Embedding模型
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu  # 或 cuda
EMBEDDING_BATCH_SIZE=32

# 模型缓存目录
MODEL_CACHE_DIR=./models

# =============================================================================
# 应用配置
# =============================================================================
APP_NAME=RAG System
APP_VERSION=1.0.0
APP_DEBUG=true  # 生产环境设为false

# API配置
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1  # 生产环境根据CPU核心数调整

# CORS配置
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
ALLOWED_METHODS=GET,POST,PUT,DELETE,OPTIONS
ALLOWED_HEADERS=*

# =============================================================================
# 安全配置
# =============================================================================
SECRET_KEY=your-secret-key-here-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# =============================================================================
# 日志配置
# =============================================================================
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE=logs/rag_system.log
LOG_MAX_SIZE=10MB
LOG_BACKUP_COUNT=5

# =============================================================================
# 性能配置
# =============================================================================
# 数据库连接池
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600

# 缓存配置
CACHE_TTL=3600  # 1小时
CACHE_MAX_SIZE=1000

# 限流配置
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_BURST=10

# =============================================================================
# 监控配置
# =============================================================================
# 健康检查
HEALTH_CHECK_INTERVAL=30
HEALTH_CHECK_TIMEOUT=10

# 指标收集
METRICS_ENABLED=true
METRICS_PORT=9090
```

## 17. 版本兼容性矩阵

### 17.1 Python版本兼容性

| 组件 | Python 3.11 | Python 3.12 | 说明 |
|------|-------------|-------------|------|
| FastAPI | ✅ | ✅ | 完全支持 |
| SQLModel | ✅ | ⚠️ | 部分依赖可能有问题 |
| PyTorch | ✅ | ✅ | 官方支持 |
| Transformers | ✅ | ✅ | 官方支持 |
| Qdrant Client | ✅ | ✅ | 官方支持 |
| Redis | ✅ | ✅ | 官方支持 |

 **推荐**: 使用Python 3.12.x以获得最佳兼容性

### 17.2 数据库版本兼容性

| 数据库 | 推荐版本 | 最低版本 | 说明 |
|--------|----------|----------|------|
| PostgreSQL | 15.5+ | 14.0 | 需要pg_vector扩展 |
| Redis | 7.2+ | 6.2 | 建议使用最新稳定版 |
| Qdrant | 1.7.0+ | 1.5.0 | API兼容性 |

### 17.3 依赖版本锁定

```toml
# 关键依赖版本锁定
[project]
dependencies = [
    # 核心框架 - 严格版本控制
    "fastapi==0.104.1",
    "sqlmodel==0.0.14",
    "pydantic==2.5.0",
    
    # AI/ML - 兼容性版本范围
    "torch>=2.1.0,<2.2.0",
    "transformers>=4.35.0,<4.37.0",
    "sentence-transformers>=2.2.0,<2.3.0",
    
    # 数据库驱动 - 稳定版本
    "psycopg2-binary==2.9.9",
    "redis[hiredis]==5.0.1",
    "qdrant-client==1.7.0",
    
    # 工具库 - 兼容性范围
    "httpx>=0.25.0,<0.26.0",
    "uvicorn[standard]>=0.24.0,<0.25.0",
]
```

## 18. 迁移指南

### 18.1 从旧版本迁移

如果你的项目使用了不同的技术栈，按以下步骤迁移：

#### 步骤1：备份现有项目
```bash
# 备份整个项目
cp -r your-project your-project-backup

# 备份数据库
pg_dump your_db > backup.sql

# 备份requirements.txt
cp requirements.txt requirements.txt.backup

# 2. 创建pyproject.toml
# 使用上面的模板，根据requirements.txt内容调整dependencies

# 3. 安装uv（如果还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 4. 使用uv安装依赖
uv pip install -e .

# 5. 验证安装
uv pip list

# 6. 删除旧文件
rm requirements.txt
```

### 11.2 版本升级检查清单

- [ ] 备份现有项目
- [ ] 更新pyproject.toml中的版本号
- [ ] 运行`uv pip install -e .`安装新版本
- [ ] 运行测试套件确保兼容性
- [ ] 检查API文档生成是否正常
- [ ] 验证Docker构建是否成功
- [ ] 更新CI/CD配置（如需要）

## 12. 质量保证

### 12.1 自动化检查

```bash
# 代码质量检查脚本
#!/bin/bash
set -e

echo "🔍 运行代码质量检查..."

# 代码风格检查
echo "📝 检查代码风格..."
ruff check .
ruff format --check .

# 类型检查
echo "🔍 运行类型检查..."
mypy app/

# 运行测试
echo "🧪 运行测试套件..."
pytest --cov=app --cov-report=term-missing

# 安全检查
echo "🔒 运行安全检查..."
bandit -r app/

echo "✅ 所有检查通过！"
```

### 12.2 性能基准

```python
# 性能基准测试
import time
import asyncio
from fastapi.testclient import TestClient
from app.main import app

def test_api_performance():
    """API性能基准测试"""
    client = TestClient(app)
    
    # 健康检查响应时间
    start = time.time()
    response = client.get("/health")
    duration = time.time() - start
    
    assert response.status_code == 200
    assert duration < 0.1  # 100ms内响应
    
    print(f"健康检查响应时间: {duration:.3f}s")
```

## 13. 常见问题解决

### 13.1 依赖冲突

**问题**: 依赖版本冲突
```bash
ERROR: Cannot install package due to conflicting dependencies
```

**解决方案**:
```bash
# 1. 清理环境
uv pip uninstall -r pyproject.toml

# 2. 重新安装
uv pip install -e . --force-reinstall

# 3. 检查冲突
uv pip check
```

### 13.2 Docker构建失败

**问题**: Docker镜像构建缓慢或失败

**解决方案**:
```dockerfile
# 使用多阶段构建优化
FROM python:3.12-slim as builder

# 安装构建依赖
RUN pip install uv

# 复制依赖文件
COPY pyproject.toml ./

# 安装依赖到虚拟环境
RUN uv venv /opt/venv
RUN uv pip install --system -e .

# 生产镜像
FROM python:3.11-slim

# 复制虚拟环境
COPY --from=builder /opt/venv /opt/venv

# 设置PATH
ENV PATH="/opt/venv/bin:$PATH"

# 复制应用代码
COPY app/ /app/app/

WORKDIR /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 13.3 性能优化

**数据库连接池优化**:
```python
# app/core/database.py
from sqlmodel import create_engine

engine = create_engine(
    DATABASE_URL,
    pool_size=20,          # 连接池大小
    max_overflow=30,       # 最大溢出连接
    pool_pre_ping=True,    # 连接预检查
    pool_recycle=3600,     # 连接回收时间
)
```

**Redis连接优化**:
```python
# app/core/redis.py
import redis.asyncio as redis

redis_pool = redis.ConnectionPool(
    host="localhost",
    port=6379,
    db=0,
    max_connections=20,    # 最大连接数
    retry_on_timeout=True, # 超时重试
)

redis_client = redis.Redis(connection_pool=redis_pool)
```

## 14. 总结

### 14.1 核心原则

1. **一致性优先**: 所有课程使用相同的技术栈版本
2. **稳定性保证**: 选择经过验证的稳定版本
3. **性能优化**: 配置优化确保良好性能
4. **可维护性**: 标准化配置便于维护
5. **扩展性**: 模块化设计支持功能扩展

### 14.2 实施建议

1. **立即行动**: 优先统一已有课程的技术栈
2. **分步实施**: 按模块逐步升级，降低风险
3. **充分测试**: 每次变更后进行完整测试
4. **文档更新**: 及时更新相关文档
5. **持续监控**: 建立监控机制确保系统稳定

### 14.3 预期效果

通过统一技术栈，我们将实现：

- ✅ **开发效率提升30%**: 标准化减少重复工作
- ✅ **维护成本降低50%**: 统一配置简化维护
- ✅ **学习体验改善**: 一致的开发环境
- ✅ **代码质量提升**: 自动化工具确保质量
- ✅ **部署成功率99%+**: 标准化配置减少部署问题

**建议立即开始技术栈统一工作，为后续课程开发奠定坚实基础！**
