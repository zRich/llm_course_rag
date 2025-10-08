# Lesson 07 黑板/投屏操作步骤

## 操作前准备
- 确保PostgreSQL数据库运行正常
- 准备好包含中文文档的documents表
- 安装jieba库：`pip install jieba`
- 准备演示用的查询示例

## 步骤1：展示关键词检索vs向量检索对比（10分钟）

### 投屏内容1：概念对比表格
```
| 检索方式 | 匹配原理 | 优势 | 劣势 | 适用场景 |
|---------|---------|------|------|---------|
| 关键词检索 | 精确词汇匹配 | 精确、快速、可解释 | 无语义理解 | 法律文档、技术手册 |
| 向量检索 | 语义相似度 | 语义理解强 | 不够精确、成本高 | 推荐系统、模糊查询 |
```

### 现场演示对比
**操作步骤**：
1. 打开两个终端窗口
2. 左侧：向量检索演示
3. 右侧：关键词检索演示（待实现）

**演示查询**："人工智能医疗应用"

**讲解要点**：
- "注意看两种检索方式的结果差异"
- "向量检索可能返回包含'AI'、'机器学习'的文档"
- "关键词检索只返回包含确切词汇的文档"

## 步骤2：PostgreSQL全文检索配置演示（15分钟）

### 投屏内容2：数据库结构
```sql
-- 当前documents表结构
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 现场操作：添加全文检索支持
**操作步骤**：
1. 连接到PostgreSQL数据库
```bash
psql -h localhost -U postgres -d rag_system
```

2. 检查当前表结构
```sql
\d documents
```

3. 添加tsvector字段
```sql
ALTER TABLE documents ADD COLUMN content_tsvector tsvector;
```

4. 更新现有数据
```sql
UPDATE documents SET content_tsvector = to_tsvector('chinese', content);
```

5. 创建GIN索引
```sql
CREATE INDEX idx_documents_content_tsvector ON documents USING GIN(content_tsvector);
```

6. 验证索引创建
```sql
\d documents
SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'documents';
```

**讲解要点**：
- "tsvector字段存储预处理的检索向量"
- "GIN索引专门为全文检索优化"
- "chinese配置确保正确处理中文文本"

### 投屏内容3：全文检索函数说明
```sql
-- to_tsvector：文本转检索向量
SELECT to_tsvector('chinese', '人工智能技术发展');

-- to_tsquery：查询转检索查询  
SELECT to_tsquery('chinese', '人工智能 & 技术');

-- @@操作符：执行匹配
SELECT content FROM documents 
WHERE to_tsvector('chinese', content) @@ to_tsquery('chinese', '人工智能');
```

## 步骤3：jieba分词效果演示（10分钟）

### 投屏内容4：分词模式对比
```python
import jieba

text = '人工智能技术在医疗领域的应用研究'

# 三种分词模式对比
print("精确模式:", '/'.join(jieba.cut(text, cut_all=False)))
print("全模式:", '/'.join(jieba.cut(text, cut_all=True)))  
print("搜索模式:", '/'.join(jieba.cut_for_search(text)))
```

### 现场操作：分词测试
**操作步骤**：
1. 打开Python交互环境
```bash
python3
```

2. 导入jieba并测试
```python
import jieba

# 测试查询列表
test_queries = [
    "人工智能在医疗领域的应用",
    "机器学习算法优化方法", 
    "深度学习神经网络架构",
    "自然语言处理技术发展"
]

print("=== jieba分词效果测试 ===")
for query in test_queries:
    tokens = list(jieba.cut(query, cut_all=False))
    keywords = [token for token in tokens if len(token) > 1]
    print(f"原文: {query}")
    print(f"分词: {' | '.join(tokens)}")
    print(f"关键词: {keywords}")
    print("-" * 50)
```

**讲解要点**：
- "精确模式适合关键词检索，避免过度分词"
- "我们过滤掉单字符词，保留有意义的关键词"
- "分词质量直接影响检索效果"

## 步骤4：关键词检索功能实现（20分钟）

### 投屏内容5：函数架构图
```
用户查询 → preprocess_query() → 关键词列表 → keyword_search() → 检索结果
    ↓              ↓                    ↓              ↓
"人工智能医疗"  → jieba分词 → ['人工智能','医疗'] → PostgreSQL查询 → 相关文档
```

### 现场编码：预处理函数
**操作步骤**：
1. 创建新的Python文件
```bash
touch keyword_search.py
```

2. 实现预处理函数
```python
import jieba
import re
from typing import List

def preprocess_query(query: str) -> List[str]:
    """预处理查询文本，进行分词和清理"""
    if not query or not query.strip():
        return []
    
    # 使用jieba进行分词
    tokens = jieba.cut(query.strip(), cut_all=False)
    
    # 过滤和清理
    keywords = []
    for token in tokens:
        token = token.strip()
        # 保留长度大于1的中英文词汇
        if len(token) > 1 and re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9]+$', token):
            keywords.append(token)
    
    return keywords

# 测试预处理函数
test_query = "人工智能在医疗领域的应用研究"
result = preprocess_query(test_query)
print(f"查询: {test_query}")
print(f"关键词: {result}")
```

**讲解要点**：
- "正则表达式确保只保留中英文和数字"
- "过滤掉标点符号和无意义的短词"
- "返回清理后的关键词列表"

### 现场编码：检索函数
**操作步骤**：
1. 添加检索函数
```python
import psycopg2
from typing import Dict, Any
import os

# 数据库连接配置
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost/rag_system')

def keyword_search(query: str, limit: int = 10) -> Dict[str, Any]:
    """执行关键词检索"""
    # 预处理查询
    keywords = preprocess_query(query)
    
    if not keywords:
        return {
            "results": [], 
            "total": 0, 
            "message": "无有效关键词，请输入包含中英文的查询内容"
        }
    
    # 构建PostgreSQL全文检索查询
    search_query = ' & '.join(keywords)
    
    sql = """
    SELECT id, content, 
           ts_rank(content_tsvector, to_tsquery('chinese', %s)) as score
    FROM documents 
    WHERE content_tsvector @@ to_tsquery('chinese', %s)
    ORDER BY score DESC
    LIMIT %s
    """
    
    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (search_query, search_query, limit))
                results = cur.fetchall()
                
                return {
                    "results": [
                        {
                            "id": row[0],
                            "content": row[1][:200] + "..." if len(row[1]) > 200 else row[1],
                            "score": round(float(row[2]), 4),
                            "matched_keywords": keywords
                        }
                        for row in results
                    ],
                    "total": len(results),
                    "keywords": keywords,
                    "query_time": "< 0.1s"
                }
                
    except psycopg2.Error as e:
        return {
            "error": f"数据库错误: {str(e)}", 
            "results": [], 
            "total": 0
        }
    except Exception as e:
        return {
            "error": f"系统错误: {str(e)}", 
            "results": [], 
            "total": 0
        }
```

**讲解要点**：
- "使用&操作符要求所有关键词都匹配"
- "ts_rank函数计算相关性得分"
- "完善的错误处理确保系统稳定性"

## 步骤5：功能测试与验证（15分钟）

### 现场测试：多种查询场景
**操作步骤**：
1. 测试正常查询
```python
# 测试1：正常查询
result1 = keyword_search("人工智能医疗应用")
print("=== 测试1：正常查询 ===")
print(f"查询关键词: {result1['keywords']}")
print(f"结果数量: {result1['total']}")
if result1['results']:
    print(f"最高得分: {result1['results'][0]['score']}")
    print(f"内容预览: {result1['results'][0]['content']}")
```

2. 测试边界情况
```python
# 测试2：空查询
result2 = keyword_search("")
print("\n=== 测试2：空查询 ===")
print(f"结果: {result2}")

# 测试3：无匹配查询
result3 = keyword_search("不存在的内容xyz123")
print("\n=== 测试3：无匹配查询 ===")
print(f"结果数量: {result3['total']}")

# 测试4：单个关键词
result4 = keyword_search("技术")
print("\n=== 测试4：单个关键词 ===")
print(f"结果数量: {result4['total']}")
```

**讲解要点**：
- "测试覆盖正常情况和边界情况"
- "观察不同查询的得分差异"
- "验证错误处理的有效性"

### 投屏内容6：性能对比
```
检索方式对比测试结果：

查询："人工智能医疗应用"

关键词检索：
- 响应时间：0.05s
- 结果数量：15
- 精确匹配：100%

向量检索：
- 响应时间：0.12s  
- 结果数量：10
- 语义相关：85%
```

## 步骤6：一致性检查与提交模板演示（10分钟）

### 现场演示：检查清单验证
**操作步骤**：
1. 打开检查清单文件
```bash
cat checklist.md
```

2. 逐项验证实现
- ✅ preprocess_query函数实现正确
- ✅ keyword_search函数返回格式规范
- ✅ 错误处理完善
- ✅ 数据库查询优化
- ✅ 测试用例覆盖全面

### 现场填写：提交模板
**操作步骤**：
1. 复制提交模板
```bash
cp templates/lab07_submission_template.py lab07_keyword_search.py
```

2. 填写关键信息
```python
# 学生信息
STUDENT_NAME = "张三"
STUDENT_ID = "2024001"
SUBMISSION_DATE = "2024-03-15"

# 功能实现状态
IMPLEMENTATION_STATUS = {
    "preprocess_query": "已完成",
    "keyword_search": "已完成", 
    "database_setup": "已完成",
    "testing": "已完成"
}

# 性能指标
PERFORMANCE_METRICS = {
    "avg_response_time": "0.08s",
    "search_accuracy": "90%",
    "error_rate": "0%"
}
```

**讲解要点**：
- "提交模板确保格式统一"
- "性能指标需要实际测试获得"
- "实现状态如实填写，便于评估"

## 操作总结与检查点

### 关键检查点
1. **数据库配置**：tsvector字段和GIN索引创建成功
2. **分词效果**：jieba分词结果符合预期
3. **检索功能**：能够正确返回相关文档
4. **错误处理**：各种异常情况处理得当
5. **性能指标**：响应时间和准确率达标

### 常见问题排查
- **分词结果异常**：检查jieba库安装和中文编码
- **数据库连接失败**：验证连接字符串和权限
- **检索无结果**：确认tsvector字段更新和索引创建
- **性能较差**：检查GIN索引是否生效

### 下节课准备
- 保存今天实现的关键词检索代码
- 确保向量检索功能正常运行
- 准备混合检索策略的学习材料