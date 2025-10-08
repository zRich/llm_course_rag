# Lesson 07 对齐与连续性说明

## 与模块目标对齐

### Module B：检索技术优化
**模块目标**：构建多样化的检索技术栈，提升RAG系统的检索能力和适应性

**本课贡献**：
- 实现关键词检索技术，补充向量检索的不足
- 建立基于PostgreSQL的全文检索基础设施
- 掌握中文文本处理和分词技术
- 为混合检索策略提供技术组件

**核心契约对齐**：
- **检索接口统一**：`search(query, limit) -> results`
- **结果格式标准**：`{"results": [...], "total": int, "score": float}`
- **错误处理规范**：`{"error": str, "results": [], "total": 0}`
- **性能指标要求**：响应时间 ≤ 200ms，准确率 ≥ 85%

## 与课程主线对齐

### RAG系统演进主线
```
Lesson 01-05: 基础RAG系统
    ↓
Lesson 06: 向量检索实现 ← 语义检索能力
    ↓
Lesson 07: 关键词检索 ← 精确匹配能力 (当前课)
    ↓
Lesson 08: 混合检索策略 ← 综合检索能力
    ↓
Lesson 09-12: 高级优化技术
```

**主线契约**：
- **数据库schema**：documents表结构保持一致
- **API接口**：检索函数签名统一
- **配置管理**：环境变量和配置文件规范
- **错误码体系**：HTTP状态码 + 业务错误码

### 核心API一致性
```python
# 统一的检索接口规范
def search_interface(query: str, method: str = "vector", limit: int = 10) -> Dict[str, Any]:
    """
    统一检索接口
    
    Args:
        query: 查询字符串
        method: 检索方法 ("vector", "keyword", "hybrid")
        limit: 结果数量限制
    
    Returns:
        标准化结果格式
    """
```

## 承接关系

### 从Lesson 06承接的内容

#### 1. 技术基础
- **PostgreSQL数据库**：已建立的documents表和向量字段
- **Python环境**：已配置的依赖库和开发环境
- **RAG系统架构**：已实现的基础框架和接口

#### 2. 数据资源
- **文档数据集**：已导入的中文文档数据
- **向量数据**：已生成的文档embedding向量
- **测试用例**：已建立的查询测试集

#### 3. 系统组件
- **数据库连接**：已配置的连接池和参数
- **配置管理**：已建立的环境变量体系
- **日志系统**：已实现的日志记录机制

### 具体承接点

#### 数据库Schema扩展
```sql
-- Lesson 06已有结构
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(1536),  -- 向量检索字段
    created_at TIMESTAMP DEFAULT NOW()
);

-- Lesson 07新增字段
ALTER TABLE documents ADD COLUMN content_tsvector tsvector;  -- 关键词检索字段
```

#### 检索接口扩展
```python
# Lesson 06实现
def vector_search(query: str, limit: int = 10) -> Dict[str, Any]:
    """向量检索实现"""
    pass

# Lesson 07新增
def keyword_search(query: str, limit: int = 10) -> Dict[str, Any]:
    """关键词检索实现"""
    pass
```

## 引出关系

### 为Lesson 08准备的内容

#### 1. 技术组件
- **关键词检索函数**：`keyword_search()` - 混合检索的核心组件
- **查询预处理**：`preprocess_query()` - 查询优化的基础
- **分词技术**：jieba分词 - 中文处理的标准工具

#### 2. 性能基准
- **检索速度基准**：关键词检索的响应时间数据
- **准确率基准**：不同查询类型的效果评估
- **资源消耗基准**：内存和CPU使用情况

#### 3. 对比数据
- **检索方式对比**：关键词 vs 向量检索的效果差异
- **适用场景分析**：不同查询类型的最佳检索方法
- **用户体验数据**：检索结果的用户满意度

### 具体引出点

#### 混合检索接口设计
```python
# Lesson 08将实现
def hybrid_search(query: str, 
                 keyword_weight: float = 0.5,
                 vector_weight: float = 0.5,
                 limit: int = 10) -> Dict[str, Any]:
    """
    混合检索实现
    将使用Lesson 07的keyword_search()和Lesson 06的vector_search()
    """
    keyword_results = keyword_search(query, limit)  # 使用本课成果
    vector_results = vector_search(query, limit)    # 使用前课成果
    # 结果融合逻辑...
```

#### 查询分析器扩展
```python
# Lesson 08将基于本课的预处理功能
def analyze_query(query: str) -> Dict[str, Any]:
    """
    查询分析器，决定最佳检索策略
    基于Lesson 07的preprocess_query()功能
    """
    keywords = preprocess_query(query)  # 复用本课功能
    # 分析查询特征，选择检索策略...
```

## 仓库结构对齐

### 目录结构规范
```
/Users/richzhao/dev/llm_courses/courses/11_rag/
├── lesson07_关键词检索优化/           # 教学文档目录
│   ├── README.md                    # 课时总览
│   ├── teacher-script.md            # 教师讲稿
│   ├── blackboard-steps.md          # 演示步骤
│   ├── checklist.md                 # 检查清单
│   ├── questions.md                 # 课堂提问
│   ├── terminology.md               # 术语定义
│   ├── acceptance.md                # 验收标准
│   ├── alignment.md                 # 本文档
│   ├── boundaries.md                # 边界声明
│   ├── SETUP.md                     # 环境准备
│   ├── lab.md                       # 实验指导
│   ├── examples/                    # 演示示例
│   │   ├── exercise/               # 课堂练习
│   │   ├── queries.txt             # 测试查询
│   │   └── expected_results.json   # 预期结果
│   └── templates/                   # 提交模板
│       └── lab07_submission.py     # 作业模板
├── labs/student/lab07/              # 学生实验目录
│   ├── keyword_search.py           # 学生实现
│   ├── test_keyword_search.py      # 测试文件
│   └── README.md                   # 实验说明
└── labs/full/lab07/                # 完整参考实现
    ├── keyword_search.py           # 参考实现
    ├── performance_test.py         # 性能测试
    └── integration_test.py         # 集成测试
```

### 路径命名一致性
- **配置文件路径**：`config/database.py`, `config/search.py`
- **数据文件路径**：`data/documents/`, `data/test_queries/`
- **日志文件路径**：`logs/search.log`, `logs/performance.log`
- **缓存目录路径**：`cache/jieba/`, `cache/queries/`

## 关键字段命名规范

### 数据库字段
```sql
-- 统一命名规范
documents.id              -- 文档ID
documents.content         -- 文档内容
documents.embedding       -- 向量字段 (Lesson 06)
documents.content_tsvector -- 全文检索字段 (Lesson 07)
documents.created_at      -- 创建时间
documents.updated_at      -- 更新时间
```

### API字段
```python
# 检索结果统一格式
{
    "results": [
        {
            "id": str,           # 文档ID
            "content": str,      # 文档内容
            "score": float,      # 相关性得分
            "method": str,       # 检索方法标识
            "metadata": dict     # 额外元数据
        }
    ],
    "total": int,            # 结果总数
    "query_time": str,       # 查询耗时
    "method": str,           # 使用的检索方法
    "parameters": dict       # 查询参数
}
```

### 错误响应格式
```python
# 统一错误格式
{
    "error": {
        "code": str,         # 错误代码 (如: "QUERY_INVALID")
        "message": str,      # 错误描述
        "details": dict,     # 详细信息
        "hints": list        # 解决建议
    },
    "results": [],
    "total": 0
}
```

## 性能指标对齐

### 响应时间标准
- **单次查询**：≤ 200ms (与Lesson 06保持一致)
- **批量查询**：≤ 50ms/query (平均)
- **并发查询**：支持10个并发请求

### 准确率标准
- **关键词匹配准确率**：≥ 85%
- **分词准确率**：≥ 90%
- **用户满意度**：≥ 80% (基于测试反馈)

### 资源使用标准
- **内存使用**：≤ 512MB (单进程)
- **CPU使用**：≤ 50% (单核)
- **数据库连接**：≤ 5个并发连接

## 版本兼容性

### 依赖库版本
```txt
# 与前序课程保持兼容
psycopg2-binary>=2.9.0    # 数据库连接 (Lesson 06使用)
numpy>=1.21.0             # 数值计算 (Lesson 06使用)
jieba>=0.42.1             # 中文分词 (Lesson 07新增)
pytest>=7.0.0             # 测试框架 (全课程使用)
```

### Python版本
- **最低要求**：Python 3.8+
- **推荐版本**：Python 3.9+
- **测试版本**：Python 3.8, 3.9, 3.10

### 数据库版本
- **PostgreSQL**：≥ 12.0 (支持中文全文检索)
- **扩展要求**：zhparser (中文分词扩展，可选)

## 质量保证对齐

### 代码质量标准
- **pylint评分**：≥ 8.0 (与课程标准一致)
- **测试覆盖率**：≥ 80% (与课程标准一致)
- **文档覆盖率**：≥ 90% (函数docstring)

### 安全标准
- **SQL注入防护**：使用参数化查询
- **输入验证**：查询长度和格式检查
- **错误信息安全**：不泄露敏感信息

### 可维护性标准
- **函数复杂度**：圈复杂度 ≤ 10
- **代码重复率**：≤ 5%
- **命名规范**：遵循PEP 8标准

---

## 对齐验证清单

### 技术对齐检查
- [ ] 数据库schema与前序课程兼容
- [ ] API接口格式与课程规范一致
- [ ] 错误处理与全课程标准统一
- [ ] 性能指标符合模块要求

### 教学对齐检查
- [ ] 知识点与模块目标匹配
- [ ] 难度递进符合学习曲线
- [ ] 实践项目与课程主线契合
- [ ] 评估标准与课程体系一致

### 文档对齐检查
- [ ] 文档结构符合最佳实践规范
- [ ] 术语使用与课程词汇表一致
- [ ] 示例代码与课程风格统一
- [ ] 参考资料与课程资源库对齐