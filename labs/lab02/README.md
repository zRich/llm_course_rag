# 实验2：检索优化与JWT认证实验（Lesson 7-14）

## 实验概述

本实验是RAG实战课程的第二个综合实验，基于实验1的基础RAG系统，重点优化检索功能并引入JWT身份认证。学生将学习如何实现混合检索、重排序等高级检索技术，同时构建完整的用户认证和权限控制系统。

## 实验目标

- 实现关键词检索功能
- 构建混合检索系统（向量检索 + 关键词检索）
- **引入JWT身份认证系统**
- **实现用户权限控制和数据隔离**
- 集成元数据过滤功能
- 实现Rerank重排序机制
- 构建用户会话管理和缓存系统
- 评估和对比不同检索策略的效果

## 涉及课程

- **Lesson 7**：关键词检索优化
- **Lesson 8**：向量检索优化
- **Lesson 9**：混合检索融合策略
- **Lesson 10**：重排序Rerank接入
- **Lesson 11**：JWT认证系统设计与实现 ⭐
- **Lesson 12**：权限控制与文档归属 ⭐
- **Lesson 13**：引用与可溯源（带权限）
- **Lesson 14**：缓存策略与用户会话管理

## 技术栈

### 新增技术组件
- **JWT认证**：用户身份验证和授权 ⭐
- **密码哈希**：bcrypt或argon2安全密码存储 ⭐
- **权限控制**：基于用户的资源访问控制 ⭐
- **Elasticsearch**：全文检索引擎（可选）
- **PostgreSQL全文检索**：内置全文检索功能
- **Rerank模型**：BGE-reranker或Cohere Rerank API
- **Jieba分词**：中文文本分词

### 继承技术栈
- Python 3.12.x、FastAPI、PostgreSQL、Qdrant、Redis、Docker

## 前置条件

- 完成实验1：RAG系统基础搭建实验
- 具备基础的信息检索理论知识
- 了解向量相似度计算原理
- **了解JWT认证原理和HTTP安全基础** ⭐

## 实验步骤

### 第一阶段：关键词检索实现（Lesson 7）

1. **全文检索配置**
   ```sql
   -- PostgreSQL全文检索配置
   ALTER TABLE chunks ADD COLUMN search_vector tsvector;
   CREATE INDEX idx_chunks_search ON chunks USING gin(search_vector);
   ```

2. **中文分词集成**
   ```python
   import jieba
   
   def tokenize_chinese(text: str) -> List[str]:
       return list(jieba.cut(text))
   ```

3. **关键词检索API**
   - 实现BM25算法
   - 支持中英文混合检索
   - 配置检索权重

### 第二阶段：向量检索优化（Lesson 8）

1. **Qdrant集成优化**
   - 向量索引配置
   - 批量向量操作
   - 性能监控

2. **向量检索参数调优**
   - 相似度阈值
   - 检索数量
   - 向量维度优化

### 第三阶段：混合检索融合（Lesson 9）

1. **检索策略设计**
   - 向量检索：语义相似度
   - 关键词检索：词汇匹配
   - 融合算法：RRF（Reciprocal Rank Fusion）

2. **融合算法实现**
   ```python
   def reciprocal_rank_fusion(
       vector_results: List[SearchResult],
       keyword_results: List[SearchResult],
       k: int = 60
   ) -> List[SearchResult]:
       # RRF融合算法实现
       pass
   ```

### 第四阶段：重排序集成（Lesson 10）

1. **Rerank模型选择**
   - BGE-reranker-large
   - Cohere Rerank API
   - 自定义重排序模型

2. **重排序流程**
   ```python
   def rerank_results(
       query: str,
       candidates: List[SearchResult],
       top_k: int = 10
   ) -> List[SearchResult]:
       # 重排序实现
       pass
   ```

### 第五阶段：JWT认证系统实现（Lesson 11） ⭐

1. **用户模型设计**
   ```python
   class User(SQLModel, table=True):
       id: Optional[int] = Field(default=None, primary_key=True)
       username: str = Field(unique=True, index=True)
       email: str = Field(unique=True, index=True)
       hashed_password: str
       is_active: bool = Field(default=True)
       created_at: datetime = Field(default_factory=datetime.utcnow)
       last_login_at: Optional[datetime] = None
   ```

2. **JWT认证实现**
   ```python
   from jose import JWTError, jwt
   from passlib.context import CryptContext
   
   pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
   
   def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
       to_encode = data.copy()
       if expires_delta:
           expire = datetime.utcnow() + expires_delta
       else:
           expire = datetime.utcnow() + timedelta(minutes=15)
       to_encode.update({"exp": expire})
       encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
       return encoded_jwt
   ```

3. **认证API端点**
   - `POST /auth/register` - 用户注册
   - `POST /auth/login` - 用户登录
   - `POST /auth/refresh` - 令牌刷新
   - `POST /auth/logout` - 用户登出

### 第六阶段：权限控制与文档归属（Lesson 12） ⭐

1. **文档归属设计**
   ```python
   class Document(SQLModel, table=True):
       id: Optional[int] = Field(default=None, primary_key=True)
       title: str
       content: str
       owner_id: int = Field(foreign_key="user.id")  # 新增
       is_public: bool = Field(default=False)  # 新增
       created_at: datetime = Field(default_factory=datetime.utcnow)
   ```

2. **权限装饰器**
   ```python
   from functools import wraps
   
   def require_auth(f):
       @wraps(f)
       async def decorated_function(*args, **kwargs):
           # JWT验证逻辑
           return await f(*args, **kwargs)
       return decorated_function
   ```

3. **数据隔离实现**
   - 用户只能访问自己的文档
   - 支持文档共享机制
   - 管理员权限设计

### 第七阶段：引用与可溯源（带权限）（Lesson 13）

1. **权限过滤的引用系统**
   ```python
   def get_citations_with_permission(
       user_id: int,
       search_results: List[SearchResult]
   ) -> List[Citation]:
       # 过滤用户有权限访问的引用
       pass
   ```

### 第八阶段：缓存策略与用户会话管理（Lesson 14）

1. **用户级缓存实现**
   ```python
   def get_user_cache_key(user_id: int, query: str) -> str:
       return f"user:{user_id}:query:{hash(query)}"
   ```

2. **会话管理**
   - 用户会话状态
   - 令牌黑名单
   - 会话超时处理

## 实验任务

### 任务1：关键词检索系统

**要求**：
- 实现基于PostgreSQL的全文检索
- 支持中文分词和英文检索
- 提供检索结果评分机制

**交付物**：
- 关键词检索API接口
- 检索性能测试报告
- 中英文检索效果对比

### 任务2：混合检索融合

**要求**：
- 实现向量检索和关键词检索的融合
- 对比不同融合算法的效果
- 提供参数调优建议

**交付物**：
- 混合检索API接口
- 融合算法对比报告
- 参数调优实验结果

### 任务3：重排序优化

**要求**：
- 集成至少一种重排序模型
- 评估重排序效果提升
- 优化重排序性能

**交付物**：
- 重排序API接口
- 重排序效果评估报告
- 性能优化方案

### 任务4：JWT认证系统 ⭐

**要求**：
- 实现完整的用户注册、登录、登出功能
- 使用JWT进行身份认证和授权
- 实现密码哈希和安全存储
- 提供令牌刷新机制

**交付物**：
- 用户认证API接口（注册、登录、登出、刷新）
- JWT认证中间件
- 用户管理界面
- 安全性测试报告

### 任务5：权限控制与数据隔离 ⭐

**要求**：
- 实现文档归属和权限控制
- 用户只能访问自己的文档
- 支持文档共享机制
- 实现基于用户的数据隔离

**交付物**：
- 权限控制API接口
- 文档归属管理系统
- 数据隔离测试用例
- 权限验证中间件

### 任务6：带权限的检索系统 ⭐

**要求**：
- 检索结果基于用户权限过滤
- 实现用户级缓存机制
- 提供会话管理功能
- 确保引用和溯源的权限安全

**交付物**：
- 带权限的检索API
- 用户级缓存实现
- 会话管理系统
- 权限安全测试报告

## 评估指标

### 检索质量指标

1. **准确率（Precision）**
   ```
   Precision = 相关结果数 / 检索结果总数
   ```

2. **召回率（Recall）**
   ```
   Recall = 相关结果数 / 相关文档总数
   ```

3. **F1分数**
   ```
   F1 = 2 * (Precision * Recall) / (Precision + Recall)
   ```

4. **NDCG@K**
   - 归一化折损累积增益
   - 考虑结果排序质量

### 性能指标

1. **检索延迟**：平均响应时间
2. **吞吐量**：每秒处理查询数
3. **资源使用**：CPU和内存占用

### JWT认证安全指标 ⭐

1. **认证成功率**：有效令牌验证成功率
2. **令牌安全性**：令牌泄露和伪造防护
3. **会话管理**：会话超时和刷新机制
4. **权限控制**：数据隔离和访问控制准确性

## 实验数据集

### 测试文档集合
- **技术文档**：API文档、技术手册
- **学术论文**：AI/ML相关论文
- **新闻文章**：科技新闻报道
- **产品说明**：产品介绍和使用指南

### 查询集合
- **事实性查询**：具体信息查找
- **概念性查询**：概念解释和定义
- **比较性查询**：技术对比分析
- **操作性查询**：操作步骤和方法

### 用户测试数据 ⭐
- **测试用户**：不同权限级别的用户账户
- **文档归属**：不同用户拥有的文档集合
- **权限场景**：公开文档、私有文档、共享文档

## 评估标准

### 功能实现（60分）

- [ ] 关键词检索功能（10分）
- [ ] 混合检索融合（10分）
- [ ] 重排序功能（10分）
- [ ] JWT认证系统（10分） ⭐
- [ ] 权限控制与数据隔离（10分） ⭐
- [ ] 带权限的检索系统（10分） ⭐

### 性能优化（25分）

- [ ] 检索速度优化（8分）
- [ ] 准确率提升（8分）
- [ ] 认证性能优化（4分） ⭐
- [ ] 资源使用优化（5分）

### 实验报告（15分）

- [ ] 实验设计合理（5分）
- [ ] 数据分析深入（5分）
- [ ] 结论总结清晰（5分）

## 常见问题

### 技术实现问题

**Q: 中文分词效果不好？**
A: 尝试不同的分词工具（jieba、pkuseg、thulac），或使用自定义词典。

**Q: 混合检索融合效果不佳？**
A: 调整向量检索和关键词检索的权重比例，尝试不同的融合算法。

**Q: 重排序模型加载慢？**
A: 使用模型缓存，考虑使用更轻量级的重排序模型。

### JWT认证问题 ⭐

**Q: JWT令牌过期处理？**
A: 实现令牌刷新机制，提供自动续期功能，合理设置令牌过期时间。

**Q: 密码安全存储？**
A: 使用bcrypt进行密码哈希，设置合适的盐值轮数，永不存储明文密码。

**Q: 权限验证失败？**
A: 检查JWT签名验证，确认用户权限设置，验证数据库权限配置。

### 性能优化问题

**Q: 检索速度太慢？**
A: 优化数据库索引，使用缓存机制，考虑异步处理。

**Q: 认证验证延迟高？**
A: 使用JWT本地验证，避免每次数据库查询，实现令牌缓存机制。

**Q: 内存占用过高？**
A: 优化向量存储，使用批量处理，清理无用缓存。

## 参考资源

- [Elasticsearch官方文档](https://www.elastic.co/guide/)
- [PostgreSQL全文检索](https://www.postgresql.org/docs/current/textsearch.html)
- [BGE模型使用指南](https://github.com/FlagOpen/FlagEmbedding)
- [信息检索评估指标](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))

## 实验时间安排

- **理论学习**：2-3小时（检索理论和算法）
- **功能开发**：10-12小时（四个阶段实现）
- **性能调优**：3-4小时（参数调优和优化）
- **实验评估**：2-3小时（效果评估和分析）
- **报告撰写**：2-3小时

**总计**：19-25小时

## 提交要求

1. **完整代码**：包含所有检索优化功能
2. **实验数据**：测试数据集和查询集合
3. **评估报告**：详细的性能评估和对比分析
4. **演示视频**：展示检索优化效果

## 后续实验预告

完成本实验后，学生将进入实验3：系统性能优化实验，学习如何从系统架构层面优化RAG系统的整体性能。