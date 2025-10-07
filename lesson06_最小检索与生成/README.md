# Lesson06: 最小检索与生成

## 课程目标

实现最小可用的RAG系统，包括：
- 向量检索（TopK相似度搜索）
- 上下文拼接与优化
- LLM生成回答
- 完整的问答流程

## 核心功能

### 1. 向量检索服务
- **相似度搜索**：基于向量相似度的TopK检索
- **混合检索**：结合关键词和向量检索
- **结果过滤**：基于阈值和元数据的结果筛选
- **检索优化**：查询重写和扩展

### 2. 上下文管理
- **上下文拼接**：智能组合检索结果
- **长度控制**：基于Token限制的上下文截断
- **相关性排序**：重新排序检索结果
- **去重处理**：移除重复和冗余内容

### 3. LLM生成服务
- **多模型支持**：OpenAI、Claude、本地模型
- **Prompt工程**：优化的提示词模板
- **流式生成**：支持实时响应
- **生成控制**：温度、长度等参数调节

### 4. RAG流程编排
- **端到端流程**：从问题到答案的完整链路
- **错误处理**：优雅的异常处理和降级
- **性能监控**：检索和生成的性能指标
- **缓存机制**：问答结果缓存

## 技术架构

```
用户问题 → 查询处理 → 向量检索 → 上下文构建 → LLM生成 → 答案返回
    ↓         ↓         ↓         ↓         ↓         ↓
  预处理   → 查询扩展 → TopK搜索 → 内容拼接 → Prompt → 后处理
```

## 项目结构

```
lesson06_最小检索与生成/
├── README.md
└── practice/
    ├── app/
    │   ├── __init__.py
    │   ├── main.py                 # FastAPI应用入口
    │   ├── core/
    │   │   ├── __init__.py
    │   │   ├── config.py           # 配置管理
    │   │   ├── database.py         # 数据库连接
    │   │   └── exceptions.py       # 异常定义
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── conversation.py     # 对话模型
    │   │   └── retrieval.py        # 检索模型
    │   ├── services/
    │   │   ├── __init__.py
    │   │   ├── retriever.py        # 检索服务
    │   │   ├── generator.py        # 生成服务
    │   │   ├── context_builder.py  # 上下文构建
    │   │   └── rag_pipeline.py     # RAG流程
    │   ├── api/
    │   │   ├── __init__.py
    │   │   └── v1/
    │   │       ├── __init__.py
    │   │       └── endpoints/
    │   │           ├── __init__.py
    │   │           ├── chat.py     # 对话接口
    │   │           └── retrieval.py # 检索接口
    │   └── schemas/
    │       ├── __init__.py
    │       ├── chat.py             # 对话模式
    │       └── retrieval.py        # 检索模式
    ├── tests/
    │   ├── __init__.py
    │   ├── test_retriever.py       # 检索测试
    │   ├── test_generator.py       # 生成测试
    │   └── test_rag_pipeline.py    # 流程测试
    ├── scripts/
    │   ├── start.sh                # 启动脚本
    │   └── test.sh                 # 测试脚本
    ├── requirements.txt            # Python依赖
    ├── Dockerfile                  # Docker镜像
    ├── docker-compose.yml         # 容器编排
    └── .env.example               # 环境变量示例
```

## 核心组件

### 1. 检索服务 (retriever.py)
- 向量相似度搜索
- 混合检索策略
- 结果过滤和排序
- 查询优化

### 2. 生成服务 (generator.py)
- 多LLM模型集成
- Prompt模板管理
- 流式生成支持
- 参数控制

### 3. 上下文构建 (context_builder.py)
- 检索结果整合
- 上下文长度控制
- 相关性重排
- 格式化输出

### 4. RAG流程 (rag_pipeline.py)
- 端到端问答流程
- 异步处理支持
- 错误处理机制
- 性能监控

## 使用方式

### 1. 环境准备
```bash
# 复制环境变量
cp .env.example .env

# 编辑配置
vim .env
```

### 2. 启动服务
```bash
# 使用Docker Compose
./scripts/start.sh

# 或直接启动
docker-compose up -d
```

### 3. 测试功能
```bash
# 运行测试
./scripts/test.sh

# 手动测试
curl -X POST "http://localhost:8000/api/v1/chat/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是RAG?", "conversation_id": "test-001"}'
```

## API接口

### 1. 问答接口
- `POST /api/v1/chat/ask` - 提问
- `GET /api/v1/chat/conversations` - 对话列表
- `GET /api/v1/chat/conversations/{id}` - 对话详情

### 2. 检索接口
- `POST /api/v1/retrieval/search` - 向量检索
- `POST /api/v1/retrieval/hybrid` - 混合检索
- `GET /api/v1/retrieval/stats` - 检索统计

## 性能指标

- **检索延迟**: < 100ms (P95)
- **生成延迟**: < 2s (P95)
- **端到端延迟**: < 3s (P95)
- **并发支持**: 100+ QPS
- **准确率**: > 85%

## 扩展功能

- 多轮对话支持
- 检索结果缓存
- 生成质量评估
- A/B测试框架
- 用户反馈收集

## 注意事项

1. **LLM配置**: 需要配置OpenAI API Key或其他模型服务
2. **向量数据库**: 需要预先导入文档向量
3. **资源要求**: 建议4GB+内存，支持GPU加速
4. **网络访问**: 确保能访问LLM服务API
5. **数据隐私**: 注意敏感数据的处理和存储