# Lesson 06：最小检索与生成 - 术语表

## 核心术语

### RAG (Retrieval-Augmented Generation)
**中文**: 检索增强生成  
**定义**: 一种结合信息检索和文本生成的AI技术，通过先检索相关文档，再基于检索结果生成答案的方法。  
**英文释义**: A technique that combines information retrieval with text generation, first retrieving relevant documents and then generating answers based on the retrieved results.  
**使用场景**: 问答系统、知识库查询、智能客服等需要基于特定知识库回答问题的场景。

### Vector Retrieval
**中文**: 向量检索  
**定义**: 通过计算查询向量与文档向量之间的相似度来检索相关文档的技术。  
**英文释义**: A technique for retrieving relevant documents by calculating similarity between query vectors and document vectors.  
**关键特点**: 
- 基于语义相似度而非关键词匹配
- 支持模糊匹配和语义理解
- 可以处理同义词和相关概念

### Semantic Search
**中文**: 语义搜索  
**定义**: 基于内容语义含义进行搜索，而不仅仅是关键词匹配的搜索方式。  
**英文释义**: Search based on semantic meaning of content rather than just keyword matching.  
**优势**: 
- 理解用户真实意图
- 处理自然语言查询
- 发现隐含的相关内容

### Context Window
**中文**: 上下文窗口  
**定义**: LLM能够处理的最大输入文本长度，通常以token数量计算。  
**英文释义**: The maximum length of input text that an LLM can process, usually measured in tokens.  
**重要性**: 
- 决定了能够输入多少检索文档
- 影响生成质量和相关性
- 需要合理管理以优化性能

### Prompt Engineering
**中文**: 提示工程  
**定义**: 设计和优化输入提示以获得更好LLM输出结果的技术和方法。  
**英文释义**: The technique and methodology of designing and optimizing input prompts to achieve better LLM output results.  
**核心要素**: 
- 清晰的指令描述
- 合适的上下文信息
- 期望输出格式说明

## 技术术语

### Embedding Vector
**中文**: 嵌入向量  
**定义**: 将文本转换为高维数值向量的表示形式，用于计算语义相似度。  
**英文释义**: A high-dimensional numerical vector representation of text used for calculating semantic similarity.  
**特征**: 
- 维度通常为512-1536
- 相似内容的向量距离较近
- 可以进行数学运算

### Similarity Score
**中文**: 相似度分数  
**定义**: 衡量两个向量之间相似程度的数值，通常在0-1之间。  
**英文释义**: A numerical value measuring the degree of similarity between two vectors, typically between 0-1.  
**计算方法**: 
- 余弦相似度 (Cosine Similarity)
- 欧几里得距离 (Euclidean Distance)
- 点积 (Dot Product)

### Score Threshold
**中文**: 分数阈值  
**定义**: 用于过滤检索结果的最小相似度分数，低于此值的文档将被排除。  
**英文释义**: The minimum similarity score used to filter retrieval results; documents below this value will be excluded.  
**设置原则**: 
- 过高：可能检索不到任何文档
- 过低：可能引入不相关的噪音
- 需要根据实际效果调整

### Top-K Retrieval
**中文**: Top-K检索  
**定义**: 返回相似度分数最高的K个文档的检索策略。  
**英文释义**: A retrieval strategy that returns the K documents with the highest similarity scores.  
**参数选择**: 
- K值过小：可能遗漏相关信息
- K值过大：可能引入噪音，增加处理成本
- 通常设置为3-10

### Dense Retrieval
**中文**: 密集检索  
**定义**: 使用密集向量表示进行检索的方法，与稀疏检索（如TF-IDF）相对。  
**英文释义**: A retrieval method using dense vector representations, as opposed to sparse retrieval methods like TF-IDF.  
**优势**: 
- 更好的语义理解能力
- 处理同义词和相关概念
- 支持跨语言检索

## 系统架构术语

### Retrieval Service
**中文**: 检索服务  
**定义**: 负责从向量数据库中检索相关文档的服务组件。  
**英文释义**: A service component responsible for retrieving relevant documents from vector databases.  
**主要功能**: 
- 查询向量化
- 相似度计算
- 结果排序和过滤

### Generation Service
**中文**: 生成服务  
**定义**: 负责基于检索结果生成最终答案的服务组件。  
**英文释义**: A service component responsible for generating final answers based on retrieval results.  
**核心职责**: 
- 上下文构建
- LLM调用
- 答案后处理

### Context Manager
**中文**: 上下文管理器  
**定义**: 负责管理和组织检索文档，构建LLM输入上下文的组件。  
**英文释义**: A component responsible for managing and organizing retrieved documents to construct LLM input context.  
**功能**: 
- 文档排序和选择
- 上下文长度控制
- 格式化和模板应用

### Response Aggregator
**中文**: 响应聚合器  
**定义**: 将检索结果和生成结果整合为最终响应的组件。  
**英文释义**: A component that integrates retrieval results and generation results into final responses.  
**处理内容**: 
- 答案文本
- 来源文档信息
- 置信度分数
- 元数据信息

## 性能术语

### Latency
**中文**: 延迟  
**定义**: 从接收查询到返回结果的时间间隔。  
**英文释义**: The time interval from receiving a query to returning results.  
**组成部分**: 
- 向量化时间
- 检索时间
- 生成时间
- 网络传输时间

### Throughput
**中文**: 吞吐量  
**定义**: 系统在单位时间内能够处理的查询数量。  
**英文释义**: The number of queries a system can process per unit time.  
**影响因素**: 
- 硬件性能
- 并发处理能力
- 缓存策略
- 负载均衡

### Concurrency
**中文**: 并发性  
**定义**: 系统同时处理多个请求的能力。  
**英文释义**: The system's ability to handle multiple requests simultaneously.  
**实现方式**: 
- 异步处理
- 多线程/多进程
- 连接池管理
- 队列机制

### Caching
**中文**: 缓存  
**定义**: 将频繁访问的数据存储在快速存储介质中以提高访问速度的技术。  
**英文释义**: A technique for storing frequently accessed data in fast storage media to improve access speed.  
**缓存类型**: 
- 查询结果缓存
- 向量缓存
- 模型缓存
- 会话缓存

## 质量评估术语

### Relevance
**中文**: 相关性  
**定义**: 检索到的文档或生成的答案与用户查询的相关程度。  
**英文释义**: The degree to which retrieved documents or generated answers are related to the user query.  
**评估方法**: 
- 人工评分
- 自动化指标
- 用户反馈
- A/B测试

### Precision
**中文**: 精确率  
**定义**: 检索结果中相关文档的比例。  
**英文释义**: The proportion of relevant documents in the retrieval results.  
**计算公式**: Precision = 相关文档数 / 检索文档总数  
**意义**: 衡量检索结果的准确性

### Recall
**中文**: 召回率  
**定义**: 所有相关文档中被成功检索到的比例。  
**英文释义**: The proportion of all relevant documents that are successfully retrieved.  
**计算公式**: Recall = 检索到的相关文档数 / 所有相关文档数  
**意义**: 衡量检索的完整性

### F1-Score
**中文**: F1分数  
**定义**: 精确率和召回率的调和平均数，综合评估检索性能。  
**英文释义**: The harmonic mean of precision and recall, providing a comprehensive evaluation of retrieval performance.  
**计算公式**: F1 = 2 × (Precision × Recall) / (Precision + Recall)  
**优势**: 平衡精确率和召回率

### Confidence Score
**中文**: 置信度分数  
**定义**: 系统对生成答案准确性的信心程度，通常在0-1之间。  
**英文释义**: The system's confidence level in the accuracy of generated answers, typically between 0-1.  
**影响因素**: 
- 检索文档相关性
- 生成模型确定性
- 答案一致性
- 历史准确率

## 错误处理术语

### Fallback Strategy
**中文**: 降级策略  
**定义**: 当主要功能失败时采用的备用处理方案。  
**英文释义**: Alternative processing solutions adopted when primary functions fail.  
**常见策略**: 
- 返回缓存结果
- 使用默认回答
- 降低质量要求
- 人工介入

### Circuit Breaker
**中文**: 熔断器  
**定义**: 防止系统过载的保护机制，在检测到故障时暂时停止服务调用。  
**英文释义**: A protection mechanism to prevent system overload by temporarily stopping service calls when failures are detected.  
**状态**: 
- 关闭状态：正常工作
- 开启状态：停止调用
- 半开状态：尝试恢复

### Retry Mechanism
**中文**: 重试机制  
**定义**: 在操作失败时自动重新尝试的机制。  
**英文释义**: A mechanism that automatically retries operations when they fail.  
**策略**: 
- 固定间隔重试
- 指数退避重试
- 随机延迟重试
- 最大重试次数限制

### Graceful Degradation
**中文**: 优雅降级  
**定义**: 在部分功能不可用时，系统仍能提供基本服务的能力。  
**英文释义**: The system's ability to continue providing basic services when some functions are unavailable.  
**实现方式**: 
- 功能分级
- 服务降级
- 资源限制
- 用户提示

## 异步处理术语

### Asynchronous Processing
**中文**: 异步处理  
**定义**: 不阻塞主线程，允许其他操作并行执行的处理方式。  
**英文释义**: A processing method that doesn't block the main thread and allows other operations to execute in parallel.  
**优势**: 
- 提高系统响应性
- 更好的资源利用
- 支持高并发
- 改善用户体验

### Streaming Response
**中文**: 流式响应  
**定义**: 逐步返回生成结果，而不是等待完整结果后一次性返回的响应方式。  
**英文释义**: A response method that gradually returns generated results instead of waiting for complete results before returning all at once.  
**适用场景**: 
- 长文本生成
- 实时聊天
- 大数据处理
- 用户体验优化

### Event Loop
**中文**: 事件循环  
**定义**: 异步编程中用于处理和分发事件的核心机制。  
**英文释义**: The core mechanism for handling and dispatching events in asynchronous programming.  
**工作原理**: 
- 监听事件队列
- 执行回调函数
- 管理异步任务
- 协调资源分配

### Coroutine
**中文**: 协程  
**定义**: 可以暂停和恢复执行的函数，用于实现异步编程。  
**英文释义**: Functions that can pause and resume execution, used for implementing asynchronous programming.  
**特点**: 
- 轻量级并发
- 协作式多任务
- 内存效率高
- 易于调试

## 配置管理术语

### Configuration Management
**中文**: 配置管理  
**定义**: 管理系统配置参数和设置的过程和方法。  
**英文释义**: The process and methods for managing system configuration parameters and settings.  
**内容**: 
- 模型参数
- 服务端点
- 性能阈值
- 安全设置

### Environment Variables
**中文**: 环境变量  
**定义**: 操作系统级别的配置参数，用于存储系统和应用程序设置。  
**英文释义**: Operating system-level configuration parameters used to store system and application settings.  
**用途**: 
- API密钥管理
- 服务地址配置
- 运行环境区分
- 安全信息存储

### Hot Reload
**中文**: 热重载  
**定义**: 在不重启系统的情况下重新加载配置或代码的能力。  
**英文释义**: The ability to reload configuration or code without restarting the system.  
**优势**: 
- 零停机更新
- 快速配置调整
- 开发效率提升
- 服务连续性保证

## 监控术语

### Metrics
**中文**: 指标  
**定义**: 用于衡量系统性能和健康状况的量化数据。  
**英文释义**: Quantitative data used to measure system performance and health.  
**类型**: 
- 性能指标（响应时间、吞吐量）
- 质量指标（准确率、满意度）
- 资源指标（CPU、内存使用率）
- 业务指标（查询量、成功率）

### Logging
**中文**: 日志记录  
**定义**: 记录系统运行过程中的事件和状态信息的过程。  
**英文释义**: The process of recording events and status information during system operation.  
**级别**: 
- DEBUG：调试信息
- INFO：一般信息
- WARNING：警告信息
- ERROR：错误信息
- CRITICAL：严重错误

### Tracing
**中文**: 链路追踪  
**定义**: 跟踪请求在分布式系统中的完整执行路径的技术。  
**英文释义**: Technology for tracking the complete execution path of requests in distributed systems.  
**用途**: 
- 性能分析
- 错误定位
- 依赖关系分析
- 系统优化

## 术语索引

### 按字母排序
- Asynchronous Processing (异步处理)
- Caching (缓存)
- Circuit Breaker (熔断器)
- Concurrency (并发性)
- Confidence Score (置信度分数)
- Configuration Management (配置管理)
- Context Manager (上下文管理器)
- Context Window (上下文窗口)
- Coroutine (协程)
- Dense Retrieval (密集检索)
- Embedding Vector (嵌入向量)
- Environment Variables (环境变量)
- Event Loop (事件循环)
- F1-Score (F1分数)
- Fallback Strategy (降级策略)
- Generation Service (生成服务)
- Graceful Degradation (优雅降级)
- Hot Reload (热重载)
- Latency (延迟)
- Logging (日志记录)
- Metrics (指标)
- Precision (精确率)
- Prompt Engineering (提示工程)
- RAG (检索增强生成)
- Recall (召回率)
- Relevance (相关性)
- Response Aggregator (响应聚合器)
- Retrieval Service (检索服务)
- Retry Mechanism (重试机制)
- Score Threshold (分数阈值)
- Semantic Search (语义搜索)
- Similarity Score (相似度分数)
- Streaming Response (流式响应)
- Throughput (吞吐量)
- Top-K Retrieval (Top-K检索)
- Tracing (链路追踪)
- Vector Retrieval (向量检索)

### 按主题分类
**核心概念**: RAG, Vector Retrieval, Semantic Search, Context Window, Prompt Engineering  
**技术实现**: Embedding Vector, Similarity Score, Score Threshold, Top-K Retrieval, Dense Retrieval  
**系统架构**: Retrieval Service, Generation Service, Context Manager, Response Aggregator  
**性能优化**: Latency, Throughput, Concurrency, Caching  
**质量评估**: Relevance, Precision, Recall, F1-Score, Confidence Score  
**错误处理**: Fallback Strategy, Circuit Breaker, Retry Mechanism, Graceful Degradation  
**异步处理**: Asynchronous Processing, Streaming Response, Event Loop, Coroutine  
**配置管理**: Configuration Management, Environment Variables, Hot Reload  
**监控运维**: Metrics, Logging, Tracing