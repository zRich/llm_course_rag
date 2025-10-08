# Lesson 06：最小检索与生成（MVP RAG）- 教师讲稿

## 课程时间轴（90分钟）

### 开场导入（10分钟）
**教师开场白（可直接照读）：**

"同学们好！今天我们进入第6课：最小检索与生成，也就是MVP RAG系统的实现。

**今天的核心目标是什么？**
- 我们要把前面5课学到的所有技术串联起来
- 实现一个完整的、可以工作的RAG问答系统
- 能够接收用户问题，检索相关文档，生成准确答案

**今天的核心产出是什么？**
- 一个完整的RAG系统类 `RAGSystem`
- 支持同步和异步查询的API接口
- 包含错误处理和会话管理的完整实现

让我们先回顾一下前面几课的成果：
- Lesson 04：我们学会了PDF解析和智能分块
- Lesson 05：我们掌握了文本向量化和向量数据库操作
- 今天，我们要把这些技术整合成一个完整的RAG系统"

### 核心概念讲解（25分钟）

#### 1. RAG系统架构（8分钟）
**讲解要点：**

"RAG系统的核心是什么？Retrieval-Augmented Generation，检索增强生成。

**系统架构包含四个核心组件：**
1. **检索器（Retriever）**：从向量数据库中找到相关文档
2. **上下文管理器（Context Manager）**：整理和格式化检索到的内容
3. **LLM生成器（Generator）**：基于上下文生成答案
4. **流程编排器（Orchestrator）**：协调整个查询流程

**数据流向是这样的：**
```
用户问题 → 向量化 → 向量检索 → 上下文拼接 → LLM生成 → 格式化输出
```

这就是我们今天要实现的完整流程。"

#### 2. 向量检索原理（8分钟）
**讲解要点：**

"向量检索的核心是相似度计算。

**工作原理：**
1. 用户问题转换为向量：`question_vector = embedder.embed(question)`
2. 在向量数据库中搜索：`similar_docs = vector_store.search(question_vector, top_k=5)`
3. 计算相似度分数：余弦相似度、欧几里得距离等

**关键参数：**
- `top_k`：返回最相似的K个文档，通常设置为3-10
- `score_threshold`：相似度阈值，过滤不相关文档
- `rerank`：是否对结果重新排序

**常见问题：**
- 检索结果不相关：调整embedding模型或分块策略
- 检索速度慢：优化向量索引或减少向量维度"

#### 3. Prompt工程与上下文管理（9分钟）
**讲解要点：**

"Prompt设计直接影响生成质量。

**标准RAG Prompt结构：**
```
系统角色定义 + 上下文文档 + 用户问题 + 回答要求
```

**上下文管理策略：**
1. **文档排序**：按相似度分数排序，最相关的放在前面
2. **长度控制**：确保总长度不超过模型上下文限制
3. **格式统一**：统一文档格式，便于LLM理解

**示例Prompt：**
```
你是一个专业的问答助手。基于以下文档内容回答用户问题。

相关文档：
1. [文档1内容]
2. [文档2内容]

用户问题：{question}

请基于上述文档内容回答，如果文档中没有相关信息，请明确说明。
```"

### 实践演示（35分钟）

#### 演示1：RAG系统初始化（10分钟）
**操作步骤：**

"现在我来演示如何创建RAG系统。

**第一步：导入依赖**
```python
from rag_system import RAGSystem, RAGConfig
from llm_client import LLMClient
from text_embedder import TextEmbedder
from qdrant_vector_store import QdrantVectorStore
```

**第二步：配置参数**
```python
config = RAGConfig(
    top_k=5,
    score_threshold=0.7,
    max_context_length=4000,
    temperature=0.1
)
```

**第三步：初始化组件**
```python
rag_system = RAGSystem(
    llm_client=LLMClient(),
    embedder=TextEmbedder(),
    vector_store=QdrantVectorStore(),
    config=config
)
```

注意这里的依赖注入模式，每个组件都是独立的，便于测试和替换。"

#### 演示2：同步查询流程（12分钟）
**操作步骤：**

"现在演示完整的查询流程。

**输入问题：**
```python
question = "什么是向量数据库？"
```

**执行查询：**
```python
response = rag_system.query(question)
print(f"答案：{response.answer}")
print(f"相关文档数：{len(response.retrieved_docs)}")
print(f"置信度：{response.confidence}")
```

**预期输出：**
```
答案：向量数据库是一种专门存储和检索高维向量数据的数据库系统...
相关文档数：3
置信度：0.85
```

**内部流程解析：**
1. 问题向量化：`question_vector = self.embedder.embed(question)`
2. 向量检索：`docs = self.vector_store.search(question_vector, self.config.top_k)`
3. 上下文构建：`context = self._build_context(docs)`
4. LLM生成：`answer = self.llm_client.generate(context + question)`
5. 结果封装：`return RAGResponse(...)`"

#### 演示3：异步查询与流式输出（13分钟）
**操作步骤：**

"对于实时应用，我们需要异步和流式处理。

**异步查询：**
```python
import asyncio

async def async_query_demo():
    response = await rag_system.aquery("解释一下RAG的工作原理")
    return response

# 运行异步查询
response = asyncio.run(async_query_demo())
```

**流式输出：**
```python
async def stream_query_demo():
    async for chunk in rag_system.stream_query("什么是embedding？"):
        print(chunk.content, end='', flush=True)
    print()  # 换行

# 运行流式查询
asyncio.run(stream_query_demo())
```

**预期效果：**
- 异步查询：不阻塞主线程，适合Web应用
- 流式输出：实时显示生成过程，提升用户体验

**性能对比：**
- 同步查询：简单直接，适合批处理
- 异步查询：高并发，适合Web服务
- 流式查询：实时反馈，适合聊天应用"

### 错误处理与调试（10分钟）

**常见错误类型：**

"在RAG系统中，我们会遇到这些错误：

**1. 检索错误（RetrievalError）：**
```python
try:
    response = rag_system.query(question)
except RetrievalError as e:
    print(f"检索失败：{e.message}")
    # 降级策略：使用缓存或默认回答
```

**2. 生成错误（GenerationError）：**
```python
try:
    response = rag_system.query(question)
except GenerationError as e:
    print(f"生成失败：{e.message}")
    # 降级策略：返回检索到的原始文档
```

**3. 上下文过长错误（ContextTooLongError）：**
```python
try:
    response = rag_system.query(question)
except ContextTooLongError as e:
    print(f"上下文过长：{e.message}")
    # 自动截断或重新检索更少文档
```

**调试技巧：**
- 启用详细日志：`rag_system.enable_debug_logging()`
- 检查中间结果：`response.debug_info`
- 性能监控：`response.timing_info`"

### 课堂练习与互动（8分钟）

**练习1：参数调优（3分钟）**
"请大家尝试调整这些参数，观察结果变化：
- `top_k`：从3调整到10
- `score_threshold`：从0.7调整到0.5
- `temperature`：从0.1调整到0.7

哪个参数对结果影响最大？"

**练习2：错误模拟（3分钟）**
"请故意输入一个空问题或超长问题，观察系统如何处理错误。"

**练习3：性能测试（2分钟）**
"测试同一个问题的查询时间，记录：
- 首次查询时间（冷启动）
- 重复查询时间（缓存命中）"

### 总结与预告（2分钟）

**今天的收获：**
"我们今天完成了：
1. ✅ 理解了RAG系统的完整架构
2. ✅ 实现了同步、异步、流式三种查询模式
3. ✅ 掌握了错误处理和调试技巧
4. ✅ 学会了参数调优方法

**下一课预告：**
- Lesson 07：我们将学习RAG系统的评估与优化
- 包括答案质量评估、检索效果评估、系统性能优化等

**课后任务：**
- 完成RAG系统的完整实现
- 测试不同参数组合的效果
- 准备自己的测试问题集"

## 课堂提问参考

### 概念理解类问题
1. **Q**: "RAG系统中，为什么要先检索再生成，而不是直接让LLM回答？"
   **A**: "因为LLM的知识是训练时固定的，无法获取最新信息。通过检索，我们可以为LLM提供实时、准确的上下文信息。"

2. **Q**: "向量检索的相似度阈值应该如何设置？"
   **A**: "通常从0.7开始，根据实际效果调整。阈值太高可能检索不到文档，太低可能引入噪音。"

### 实践操作类问题
3. **Q**: "如果检索到的文档都不相关怎么办？"
   **A**: "可以降低阈值、增加top_k、或者返回'抱歉，我没有找到相关信息'的回答。"

4. **Q**: "异步查询和同步查询在什么场景下使用？"
   **A**: "Web应用用异步，批处理用同步，实时聊天用流式。"

## 常见误区与应对话术

### 误区1：认为RAG就是简单的文档搜索
**学生可能说**："这不就是搜索引擎吗？"
**应对话术**："搜索引擎返回文档链接，RAG返回基于文档内容生成的答案。RAG理解问题语义，生成连贯回答。"

### 误区2：忽视上下文长度限制
**学生可能说**："为什么不把所有相关文档都给LLM？"
**应对话术**："LLM有上下文长度限制，通常2K-8K tokens。太长会截断，太多会影响理解质量。需要精选最相关的内容。"

### 误区3：过度依赖参数调优
**学生可能说**："是不是调好参数就能解决所有问题？"
**应对话术**："参数调优很重要，但数据质量更关键。好的文档分块、准确的向量化、合适的Prompt设计比参数调优更重要。"

## 术语解释

- **RAG**: Retrieval-Augmented Generation，检索增强生成
- **top_k**: 检索返回的最相似文档数量
- **score_threshold**: 相似度分数阈值
- **context window**: LLM的上下文窗口大小
- **temperature**: 生成随机性控制参数
- **rerank**: 对检索结果重新排序
- **streaming**: 流式输出，逐步返回生成内容