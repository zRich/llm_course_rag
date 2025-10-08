# Lesson 07 提交与验收标准

## 验收标准概述

本文档定义了第七课"关键词检索优化"的详细验收标准，确保学生提交的作业符合课程要求和质量标准。

## 功能验收标准

### 1. 核心功能要求

#### 1.1 预处理功能 (25分)
**验收项目**：`preprocess_query()` 函数实现

**必须满足**：
- [ ] 正确使用jieba进行中文分词
- [ ] 过滤长度小于2的无效词汇
- [ ] 移除标点符号和特殊字符
- [ ] 返回清理后的关键词列表
- [ ] 处理空输入和异常情况

**测试用例**：
```python
# 测试用例1：正常中文查询
assert preprocess_query("人工智能在医疗领域的应用") == ["人工智能", "医疗", "领域", "应用"]

# 测试用例2：包含英文和数字
assert preprocess_query("COVID-19疫苗接种") == ["COVID-19", "疫苗", "接种"]

# 测试用例3：空查询
assert preprocess_query("") == []

# 测试用例4：只有标点符号
assert preprocess_query("！@#￥%") == []
```

**评分标准**：
- 基本功能实现：15分
- 边界情况处理：5分
- 代码质量：5分

#### 1.2 关键词检索功能 (35分)
**验收项目**：`keyword_search()` 函数实现

**必须满足**：
- [ ] 调用预处理函数获取关键词
- [ ] 构建正确的PostgreSQL全文检索查询
- [ ] 使用ts_rank计算相关性得分
- [ ] 返回规范格式的结果
- [ ] 支持结果数量限制参数

**返回格式规范**：
```python
{
    "results": [
        {
            "id": "文档ID",
            "content": "文档内容（截断到200字符）",
            "score": 0.8542,  # 保留4位小数
            "matched_keywords": ["关键词1", "关键词2"]
        }
    ],
    "total": 15,
    "keywords": ["提取的关键词列表"],
    "query_time": "< 0.1s"
}
```

**SQL查询要求**：
```sql
-- 必须使用的查询结构
SELECT id, content, 
       ts_rank(content_tsvector, to_tsquery('chinese', %s)) as score
FROM documents 
WHERE content_tsvector @@ to_tsquery('chinese', %s)
ORDER BY score DESC
LIMIT %s
```

**评分标准**：
- SQL查询正确性：15分
- 结果格式规范：10分
- 性能优化：5分
- 错误处理：5分

#### 1.3 数据库配置 (20分)
**验收项目**：PostgreSQL全文检索配置

**必须满足**：
- [ ] documents表包含content_tsvector字段
- [ ] 创建GIN索引：`idx_documents_content_tsvector`
- [ ] 现有数据的tsvector字段已更新
- [ ] 触发器自动更新新插入数据的tsvector

**验证SQL**：
```sql
-- 检查字段存在
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'documents' AND column_name = 'content_tsvector';

-- 检查索引存在
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE tablename = 'documents' AND indexname = 'idx_documents_content_tsvector';

-- 检查触发器存在
SELECT trigger_name, event_manipulation, action_statement
FROM information_schema.triggers 
WHERE event_object_table = 'documents';
```

**评分标准**：
- 字段和索引创建：10分
- 数据更新完整性：5分
- 触发器配置：5分

## 性能验收标准

### 2. 性能指标要求

#### 2.1 响应时间 (10分)
**标准**：单次查询响应时间 ≤ 200ms

**测试方法**：
```python
import time

def test_response_time():
    queries = [
        "人工智能医疗应用",
        "机器学习算法优化",
        "深度学习神经网络",
        "自然语言处理技术"
    ]
    
    total_time = 0
    for query in queries:
        start_time = time.time()
        result = keyword_search(query)
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000  # 转换为毫秒
        total_time += response_time
        
        assert response_time <= 200, f"查询'{query}'响应时间{response_time}ms超过200ms"
    
    avg_time = total_time / len(queries)
    print(f"平均响应时间: {avg_time:.2f}ms")
```

**评分标准**：
- 平均响应时间 ≤ 100ms：10分
- 平均响应时间 ≤ 150ms：8分
- 平均响应时间 ≤ 200ms：6分
- 超过200ms：0分

#### 2.2 检索准确率 (10分)
**标准**：关键词匹配准确率 ≥ 85%

**测试数据集**：
```python
test_cases = [
    {
        "query": "人工智能医疗诊断",
        "expected_keywords": ["人工智能", "医疗", "诊断"],
        "must_contain": ["人工智能", "医疗"]  # 结果必须包含的关键词
    },
    {
        "query": "机器学习算法优化方法",
        "expected_keywords": ["机器学习", "算法", "优化", "方法"],
        "must_contain": ["机器学习", "算法"]
    }
    # ... 更多测试用例
]
```

**评估方法**：
```python
def evaluate_accuracy():
    correct_extractions = 0
    total_tests = len(test_cases)
    
    for case in test_cases:
        result = keyword_search(case["query"])
        extracted_keywords = result["keywords"]
        
        # 检查关键词提取准确性
        if all(kw in extracted_keywords for kw in case["must_contain"]):
            correct_extractions += 1
    
    accuracy = correct_extractions / total_tests
    return accuracy
```

**评分标准**：
- 准确率 ≥ 90%：10分
- 准确率 ≥ 85%：8分
- 准确率 ≥ 80%：6分
- 准确率 < 80%：0分

## 代码质量验收标准

### 3. 代码规范要求

#### 3.1 代码结构 (5分)
**要求**：
- [ ] 函数职责单一，逻辑清晰
- [ ] 适当的代码注释（覆盖率 ≥ 60%）
- [ ] 符合PEP 8编码规范
- [ ] 合理的变量和函数命名

**检查工具**：
```bash
# 代码风格检查
flake8 keyword_search.py

# 代码质量评分
pylint keyword_search.py
```

**评分标准**：
- pylint评分 ≥ 8.0：5分
- pylint评分 ≥ 7.0：3分
- pylint评分 < 7.0：0分

#### 3.2 类型提示 (3分)
**要求**：
- [ ] 函数参数类型提示完整
- [ ] 返回值类型提示准确
- [ ] 复杂数据结构使用适当的类型定义

**示例**：
```python
from typing import List, Dict, Any, Optional

def preprocess_query(query: str) -> List[str]:
    """预处理查询文本"""
    pass

def keyword_search(query: str, limit: int = 10) -> Dict[str, Any]:
    """执行关键词检索"""
    pass
```

#### 3.3 错误处理 (7分)
**要求**：
- [ ] 数据库连接异常处理
- [ ] SQL查询错误处理
- [ ] 输入参数验证
- [ ] 友好的错误信息

**错误处理示例**：
```python
try:
    with psycopg2.connect(DATABASE_URL) as conn:
        # 数据库操作
        pass
except psycopg2.OperationalError as e:
    return {"error": f"数据库连接失败: {str(e)}", "results": [], "total": 0}
except psycopg2.ProgrammingError as e:
    return {"error": f"SQL查询错误: {str(e)}", "results": [], "total": 0}
except Exception as e:
    return {"error": f"系统错误: {str(e)}", "results": [], "total": 0}
```

## 测试验收标准

### 4. 测试覆盖要求

#### 4.1 单元测试 (10分)
**要求**：
- [ ] 测试覆盖率 ≥ 80%
- [ ] 包含正常情况和边界情况测试
- [ ] 使用pytest框架编写测试

**测试文件结构**：
```python
# test_keyword_search.py
import pytest
from keyword_search import preprocess_query, keyword_search

class TestPreprocessQuery:
    def test_normal_chinese_query(self):
        """测试正常中文查询"""
        pass
    
    def test_empty_query(self):
        """测试空查询"""
        pass
    
    def test_mixed_language_query(self):
        """测试中英文混合查询"""
        pass

class TestKeywordSearch:
    def test_successful_search(self):
        """测试成功检索"""
        pass
    
    def test_no_results(self):
        """测试无结果情况"""
        pass
    
    def test_database_error(self):
        """测试数据库错误"""
        pass
```

**运行测试**：
```bash
# 运行测试并生成覆盖率报告
pytest test_keyword_search.py --cov=keyword_search --cov-report=html
```

#### 4.2 集成测试 (5分)
**要求**：
- [ ] 端到端功能测试
- [ ] 数据库集成测试
- [ ] 性能基准测试

## 文档验收标准

### 5. 文档要求

#### 5.1 README文档 (5分)
**必须包含**：
- [ ] 功能说明和特性介绍
- [ ] 安装和配置步骤
- [ ] 使用示例和API说明
- [ ] 性能指标和限制说明

#### 5.2 代码注释 (3分)
**要求**：
- [ ] 函数docstring完整
- [ ] 关键逻辑有行内注释
- [ ] 复杂算法有详细说明

**docstring示例**：
```python
def keyword_search(query: str, limit: int = 10) -> Dict[str, Any]:
    """
    执行关键词检索
    
    Args:
        query: 用户查询字符串
        limit: 返回结果数量限制，默认10
    
    Returns:
        包含检索结果的字典，格式如下：
        {
            "results": [...],  # 检索结果列表
            "total": int,      # 结果总数
            "keywords": [...], # 提取的关键词
            "query_time": str  # 查询耗时
        }
    
    Raises:
        ValueError: 当查询参数无效时
        DatabaseError: 当数据库操作失败时
    """
```

## 提交格式验收标准

### 6. 提交物要求

#### 6.1 文件结构 (5分)
**标准结构**：
```
lab07_keyword_search/
├── keyword_search.py          # 主实现文件
├── test_keyword_search.py     # 测试文件
├── requirements.txt           # 依赖列表
├── README.md                  # 说明文档
├── config.py                  # 配置文件
├── setup.sql                  # 数据库初始化脚本
└── examples/
    ├── test_queries.txt       # 测试查询
    └── performance_report.md  # 性能测试报告
```

#### 6.2 配置文件 (3分)
**requirements.txt**：
```
jieba>=0.42.1
psycopg2-binary>=2.9.0
pytest>=7.0.0
pytest-cov>=4.0.0
```

**config.py**：
```python
import os

# 数据库配置
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost/rag_system')

# 检索配置
DEFAULT_LIMIT = 10
MAX_LIMIT = 100
QUERY_TIMEOUT = 5  # 秒

# 分词配置
MIN_KEYWORD_LENGTH = 2
MAX_QUERY_LENGTH = 1000
```

## 评分权重分布

| 评估维度 | 权重 | 满分 |
|---------|------|------|
| 核心功能实现 | 40% | 80分 |
| 性能指标 | 20% | 20分 |
| 代码质量 | 15% | 15分 |
| 测试覆盖 | 15% | 15分 |
| 文档完整性 | 10% | 8分 |
| **总计** | **100%** | **138分** |

**等级划分**：
- 优秀（A）：≥ 120分（87%）
- 良好（B）：≥ 100分（72%）
- 及格（C）：≥ 80分（58%）
- 不及格（F）：< 80分

## 验收流程

### 7. 验收步骤

1. **自动化检查**（30分钟）
   - 代码风格检查
   - 单元测试运行
   - 性能基准测试

2. **功能验证**（45分钟）
   - 核心功能测试
   - 边界情况验证
   - 错误处理测试

3. **人工评审**（30分钟）
   - 代码质量评估
   - 文档完整性检查
   - 创新性和优化评价

4. **反馈生成**（15分钟）
   - 生成详细评分报告
   - 提供改进建议
   - 记录优秀实践

### 8. 不可虚构声明

**严格禁止**：
- 编造未实现的功能或性能指标
- 伪造测试结果或性能数据
- 抄袭他人代码或文档
- 提交无法运行的代码

**验证方法**：
- 所有功能必须可演示
- 性能数据必须可重现
- 代码必须可独立运行
- 使用代码相似性检测工具

**违规后果**：
- 发现虚构内容直接判定为不及格
- 严重违规报告学术诚信委员会
- 要求重新提交并延期评分