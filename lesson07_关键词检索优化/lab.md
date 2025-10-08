# Lesson 07 课后实验任务说明

## 实验概述

### 实验目标
通过本实验，学生将：
1. 掌握PostgreSQL全文检索的配置和使用
2. 学会使用jieba进行中文分词处理
3. 实现完整的关键词检索功能
4. 理解关键词检索与向量检索的区别和适用场景
5. 建立混合检索系统的技术基础

### 实验时长
- **预计完成时间**：3-4小时
- **建议分配**：
  - 环境配置：30分钟
  - 功能实现：2-2.5小时
  - 测试验证：30-45分钟
  - 文档整理：15-30分钟

### 实验难度
- **技术难度**：⭐⭐⭐☆☆ (中等)
- **工程难度**：⭐⭐☆☆☆ (简单)
- **调试难度**：⭐⭐⭐☆☆ (中等)

## 实验任务

### 任务1：环境配置与验证 (30分钟)

#### 1.1 安装jieba分词库
```bash
# 安装jieba
pip install jieba>=0.42.1

# 验证安装
python -c "import jieba; print('jieba version:', jieba.__version__)"
```

#### 1.2 配置PostgreSQL全文检索
```sql
-- 连接到数据库
psql -U your_username -d your_database

-- 设置默认全文检索配置
ALTER DATABASE your_database SET default_text_search_config = 'simple';

-- 为documents表添加tsvector字段
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_tsvector tsvector;

-- 创建GIN索引
CREATE INDEX IF NOT EXISTS idx_documents_content_tsvector 
ON documents USING GIN(content_tsvector);

-- 更新现有数据
UPDATE documents 
SET content_tsvector = to_tsvector('simple', content) 
WHERE content_tsvector IS NULL;
```

#### 1.3 环境验证
运行提供的环境测试脚本，确保所有组件正常工作：
```bash
python test_environment.py
```

**验收标准**：
- [ ] jieba安装成功，版本 >= 0.42.1
- [ ] PostgreSQL全文检索配置完成
- [ ] GIN索引创建成功
- [ ] 环境测试脚本全部通过

### 任务2：查询预处理功能实现 (45分钟)

#### 2.1 实现preprocess_query函数
在 `keyword_search.py` 文件中实现查询预处理功能：

```python
import jieba
import re
from typing import List, Set

def preprocess_query(query: str, 
                    min_word_length: int = 2,
                    max_word_length: int = 20,
                    stop_words: Set[str] = None) -> List[str]:
    """
    查询预处理函数
    
    Args:
        query: 原始查询字符串
        min_word_length: 最小词长度
        max_word_length: 最大词长度
        stop_words: 停用词集合
    
    Returns:
        处理后的关键词列表
    
    功能要求：
    1. 使用jieba进行中文分词
    2. 过滤停用词
    3. 过滤长度不符合要求的词
    4. 去除特殊字符和数字
    5. 去重并保持顺序
    """
    # TODO: 实现查询预处理逻辑
    pass
```

#### 2.2 实现停用词管理
```python
def load_stop_words(file_path: str = "data/stop_words.txt") -> Set[str]:
    """
    加载停用词列表
    
    Args:
        file_path: 停用词文件路径
    
    Returns:
        停用词集合
    """
    # TODO: 实现停用词加载逻辑
    pass

def get_default_stop_words() -> Set[str]:
    """
    获取默认停用词列表
    
    Returns:
        默认停用词集合
    """
    # TODO: 返回默认停用词集合
    pass
```

**验收标准**：
- [ ] 正确使用jieba进行分词
- [ ] 有效过滤停用词和无效词汇
- [ ] 处理边界情况（空查询、特殊字符等）
- [ ] 返回格式正确的关键词列表

### 任务3：关键词检索功能实现 (60分钟)

#### 3.1 实现keyword_search函数
```python
import psycopg2
from typing import Dict, Any, List
import time
import logging

def keyword_search(query: str, 
                  limit: int = 10,
                  min_score: float = 0.1) -> Dict[str, Any]:
    """
    关键词检索函数
    
    Args:
        query: 查询字符串
        limit: 返回结果数量限制
        min_score: 最小相关性得分
    
    Returns:
        检索结果字典，包含results、total、query_time等字段
    
    功能要求：
    1. 调用preprocess_query处理查询
    2. 构建PostgreSQL全文检索查询
    3. 执行查询并获取结果
    4. 计算相关性得分
    5. 格式化返回结果
    6. 处理异常情况
    """
    # TODO: 实现关键词检索逻辑
    pass
```

#### 3.2 实现数据库连接管理
```python
def get_database_connection():
    """
    获取数据库连接
    
    Returns:
        数据库连接对象
    """
    # TODO: 实现数据库连接逻辑
    pass

def execute_search_query(keywords: List[str], 
                        limit: int = 10,
                        min_score: float = 0.1) -> List[tuple]:
    """
    执行搜索查询
    
    Args:
        keywords: 关键词列表
        limit: 结果数量限制
        min_score: 最小得分
    
    Returns:
        查询结果列表
    """
    # TODO: 实现SQL查询执行逻辑
    pass
```

#### 3.3 实现结果格式化
```python
def format_search_results(raw_results: List[tuple], 
                         query_time: float,
                         original_query: str,
                         processed_keywords: List[str]) -> Dict[str, Any]:
    """
    格式化搜索结果
    
    Args:
        raw_results: 原始查询结果
        query_time: 查询耗时
        original_query: 原始查询
        processed_keywords: 处理后的关键词
    
    Returns:
        格式化的结果字典
    """
    # TODO: 实现结果格式化逻辑
    pass
```

**验收标准**：
- [ ] 正确构建PostgreSQL全文检索查询
- [ ] 有效处理数据库连接和异常
- [ ] 返回标准格式的检索结果
- [ ] 查询响应时间 ≤ 200ms

### 任务4：错误处理与日志记录 (30分钟)

#### 4.1 实现异常处理
```python
class KeywordSearchError(Exception):
    """关键词检索异常基类"""
    pass

class QueryProcessingError(KeywordSearchError):
    """查询处理异常"""
    pass

class DatabaseConnectionError(KeywordSearchError):
    """数据库连接异常"""
    pass

class SearchExecutionError(KeywordSearchError):
    """搜索执行异常"""
    pass

def handle_search_error(error: Exception, query: str) -> Dict[str, Any]:
    """
    处理搜索错误
    
    Args:
        error: 异常对象
        query: 查询字符串
    
    Returns:
        错误响应字典
    """
    # TODO: 实现错误处理逻辑
    pass
```

#### 4.2 实现日志记录
```python
import logging
from datetime import datetime

def setup_logging():
    """设置日志配置"""
    # TODO: 配置日志记录器
    pass

def log_search_request(query: str, keywords: List[str]):
    """记录搜索请求"""
    # TODO: 记录搜索请求日志
    pass

def log_search_result(query: str, result_count: int, query_time: float):
    """记录搜索结果"""
    # TODO: 记录搜索结果日志
    pass
```

**验收标准**：
- [ ] 定义完整的异常类型体系
- [ ] 实现有效的错误处理机制
- [ ] 配置合理的日志记录
- [ ] 错误信息清晰且有助于调试

### 任务5：单元测试编写 (45分钟)

#### 5.1 测试查询预处理
```python
import pytest
from keyword_search import preprocess_query, load_stop_words

class TestQueryPreprocessing:
    """查询预处理测试类"""
    
    def test_basic_segmentation(self):
        """测试基本分词功能"""
        # TODO: 实现基本分词测试
        pass
    
    def test_stop_words_filtering(self):
        """测试停用词过滤"""
        # TODO: 实现停用词过滤测试
        pass
    
    def test_word_length_filtering(self):
        """测试词长度过滤"""
        # TODO: 实现词长度过滤测试
        pass
    
    def test_special_characters(self):
        """测试特殊字符处理"""
        # TODO: 实现特殊字符处理测试
        pass
    
    def test_empty_query(self):
        """测试空查询处理"""
        # TODO: 实现空查询测试
        pass
```

#### 5.2 测试关键词检索
```python
class TestKeywordSearch:
    """关键词检索测试类"""
    
    def test_basic_search(self):
        """测试基本搜索功能"""
        # TODO: 实现基本搜索测试
        pass
    
    def test_search_with_limit(self):
        """测试限制结果数量"""
        # TODO: 实现结果数量限制测试
        pass
    
    def test_search_performance(self):
        """测试搜索性能"""
        # TODO: 实现性能测试
        pass
    
    def test_search_accuracy(self):
        """测试搜索准确性"""
        # TODO: 实现准确性测试
        pass
    
    def test_error_handling(self):
        """测试错误处理"""
        # TODO: 实现错误处理测试
        pass
```

**验收标准**：
- [ ] 测试覆盖率 ≥ 80%
- [ ] 所有测试用例通过
- [ ] 包含边界情况和异常测试
- [ ] 性能测试验证响应时间要求

### 任务6：集成测试与性能评估 (30分钟)

#### 6.1 集成测试
```python
def test_end_to_end_search():
    """端到端搜索测试"""
    test_queries = [
        "Python编程",
        "机器学习算法",
        "数据库设计",
        "Web开发框架",
        "人工智能应用"
    ]
    
    for query in test_queries:
        # TODO: 执行端到端测试
        pass

def test_search_consistency():
    """搜索一致性测试"""
    # TODO: 测试相同查询的结果一致性
    pass

def test_concurrent_search():
    """并发搜索测试"""
    # TODO: 测试并发搜索能力
    pass
```

#### 6.2 性能评估
```python
def benchmark_search_performance():
    """搜索性能基准测试"""
    # TODO: 实现性能基准测试
    pass

def analyze_search_accuracy():
    """搜索准确性分析"""
    # TODO: 分析搜索结果的准确性
    pass

def compare_with_vector_search():
    """与向量搜索对比"""
    # TODO: 对比关键词搜索和向量搜索的效果
    pass
```

**验收标准**：
- [ ] 端到端测试通过
- [ ] 并发搜索测试通过
- [ ] 平均响应时间 ≤ 200ms
- [ ] 搜索准确率 ≥ 85%

## 提交要求

### 文件结构
```
lab07_submission/
├── keyword_search.py          # 核心实现文件
├── test_keyword_search.py     # 单元测试文件
├── integration_test.py        # 集成测试文件
├── config.py                  # 配置文件
├── requirements.txt           # 依赖列表
├── README.md                  # 项目说明
├── data/
│   └── stop_words.txt        # 停用词文件
├── logs/
│   └── search.log            # 搜索日志
└── results/
    ├── test_results.txt      # 测试结果
    ├── performance_report.md # 性能报告
    └── comparison_analysis.md # 对比分析
```

### 核心文件要求

#### keyword_search.py
- [ ] 包含完整的preprocess_query函数实现
- [ ] 包含完整的keyword_search函数实现
- [ ] 包含错误处理和日志记录
- [ ] 代码结构清晰，注释完整
- [ ] 符合PEP 8代码规范

#### test_keyword_search.py
- [ ] 包含查询预处理测试
- [ ] 包含关键词检索测试
- [ ] 包含错误处理测试
- [ ] 测试覆盖率 ≥ 80%
- [ ] 所有测试用例通过

#### README.md
- [ ] 项目概述和功能说明
- [ ] 安装和配置说明
- [ ] 使用示例和API文档
- [ ] 测试结果和性能数据
- [ ] 已知问题和改进建议

### 性能要求
- [ ] 单次查询响应时间 ≤ 200ms
- [ ] 搜索准确率 ≥ 85%
- [ ] 支持并发查询 (≥ 5个并发)
- [ ] 内存使用 ≤ 512MB
- [ ] 错误率 ≤ 5%

### 质量要求
- [ ] 代码pylint评分 ≥ 8.0
- [ ] 函数文档字符串完整
- [ ] 类型提示正确
- [ ] 异常处理完善
- [ ] 日志记录合理

## 评分标准

### 功能实现 (40分)
- **查询预处理** (15分)
  - 正确使用jieba分词 (5分)
  - 有效过滤停用词和无效词汇 (5分)
  - 处理边界情况 (5分)

- **关键词检索** (25分)
  - 正确构建SQL查询 (10分)
  - 有效处理数据库操作 (8分)
  - 返回标准格式结果 (7分)

### 代码质量 (30分)
- **代码结构** (10分)
  - 模块化设计合理 (5分)
  - 函数职责单一 (5分)

- **代码规范** (10分)
  - 符合PEP 8规范 (5分)
  - 命名规范合理 (5分)

- **文档注释** (10分)
  - 函数文档字符串完整 (5分)
  - 关键逻辑注释清晰 (5分)

### 测试验证 (20分)
- **单元测试** (12分)
  - 测试用例完整 (6分)
  - 测试覆盖率达标 (6分)

- **集成测试** (8分)
  - 端到端测试通过 (4分)
  - 性能测试达标 (4分)

### 文档质量 (10分)
- **README文档** (5分)
  - 项目说明清晰 (3分)
  - 使用示例完整 (2分)

- **性能报告** (5分)
  - 测试数据真实 (3分)
  - 分析结论合理 (2分)

## 常见问题与解决方案

### Q1: jieba分词结果不理想怎么办？
**A1**: 
- 尝试不同的分词模式（精确模式、全模式、搜索引擎模式）
- 添加自定义词典提高分词准确性
- 调整词长度过滤参数
- 优化停用词列表

### Q2: PostgreSQL全文检索查询很慢？
**A2**:
- 确认GIN索引已正确创建
- 检查查询语句是否使用了索引
- 考虑调整PostgreSQL配置参数
- 优化查询条件，避免过于宽泛的搜索

### Q3: 搜索结果准确率不高？
**A3**:
- 检查分词质量，优化预处理逻辑
- 调整相关性得分计算方法
- 增加查询扩展和同义词处理
- 分析测试数据，找出准确率低的原因

### Q4: 并发测试失败？
**A4**:
- 检查数据库连接池配置
- 确认PostgreSQL支持的最大连接数
- 优化数据库连接管理
- 添加连接超时和重试机制

### Q5: 内存使用过高？
**A5**:
- 检查jieba词典加载方式
- 优化查询结果缓存策略
- 及时释放数据库连接
- 使用生成器而非列表存储大量数据

## 扩展挑战 (可选)

### 挑战1: 查询扩展
实现基于同义词的查询扩展功能，提高搜索召回率。

### 挑战2: 结果高亮
在搜索结果中高亮显示匹配的关键词。

### 挑战3: 搜索建议
实现搜索建议功能，当查询无结果时提供相关建议。

### 挑战4: 缓存优化
实现查询结果缓存，提高重复查询的响应速度。

### 挑战5: 多字段搜索
扩展搜索功能，支持在多个字段中进行关键词检索。

---

## 提交截止时间
请在课程结束后**7天内**完成实验并提交，逾期提交将影响成绩评定。

## 技术支持
如遇到技术问题，可以：
1. 查阅课程文档和FAQ
2. 在课程讨论区提问
3. 参加答疑时间
4. 联系助教获得帮助

祝你实验顺利！🚀