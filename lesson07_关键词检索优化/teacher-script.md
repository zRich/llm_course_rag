# Lesson 07 教师讲稿：关键词检索优化

## 开场白（5分钟）

### 今天的目标与产物
"同学们好！今天我们要学习关键词检索优化。通过这节课，你们将掌握：
1. PostgreSQL全文检索的使用方法
2. jieba中文分词技术
3. 关键词检索功能的完整实现

**核心产物**：一个可运行的关键词检索系统，能够处理中文查询并返回相关文档。"

### 与上节课的连接
"上节课我们实现了向量检索系统，它基于语义相似度。今天我们要添加关键词检索能力，它基于精确匹配。这两种技术各有优势，下节课我们会学习如何结合使用。"

## 核心概念讲解（30分钟）

### 1. 关键词检索 vs 向量检索（10分钟）

**讲稿**：
"让我们先理解两种检索方式的区别：

**关键词检索**：
- 基于精确的词汇匹配
- 查询'人工智能'只会匹配包含这个词的文档
- 优势：精确、快速、可解释
- 劣势：无法理解语义，同义词匹配困难

**向量检索**：
- 基于语义相似度
- 查询'人工智能'可能匹配'AI'、'机器学习'等相关内容
- 优势：语义理解强
- 劣势：结果不够精确，计算成本高

**实际应用场景**：
- 关键词检索：法律文档、技术手册、精确查找
- 向量检索：推荐系统、语义搜索、模糊查询"

### 2. PostgreSQL全文检索（10分钟）

**讲稿**：
"PostgreSQL提供了强大的全文检索功能。核心组件包括：

**to_tsvector()函数**：
- 将文本转换为搜索向量
- 示例：`to_tsvector('chinese', '人工智能技术发展')`
- 结果：包含分词和位置信息的向量

**to_tsquery()函数**：
- 将查询转换为搜索查询
- 示例：`to_tsquery('chinese', '人工智能 & 技术')`
- 支持AND(&)、OR(|)、NOT(!)操作

**@@操作符**：
- 执行匹配操作
- 示例：`to_tsvector('chinese', content) @@ to_tsquery('chinese', '人工智能')`

**GIN索引**：
- 专门用于全文检索的索引类型
- 大幅提升检索性能"

### 3. jieba中文分词（10分钟）

**讲稿**：
"中文分词是关键词检索的基础，因为中文没有天然的词汇分隔符。

**jieba分词特点**：
- 基于统计和规则的混合方法
- 支持三种分词模式：精确模式、全模式、搜索引擎模式
- 可以添加自定义词典

**分词模式对比**：
```python
import jieba

text = '人工智能技术发展'

# 精确模式（推荐用于关键词检索）
precise = jieba.cut(text, cut_all=False)
print('精确模式:', '/'.join(precise))
# 输出: 人工智能/技术/发展

# 全模式
full = jieba.cut(text, cut_all=True)  
print('全模式:', '/'.join(full))
# 输出: 人工/人工智能/智能/技术/发展

# 搜索引擎模式
search = jieba.cut_for_search(text)
print('搜索模式:', '/'.join(search))
# 输出: 人工/智能/人工智能/技术/发展
```

**为什么选择精确模式**：
- 避免过度分词导致的噪音
- 保持与用户查询意图的一致性
- 提高检索精度"

## 实践演示（25分钟）

### 演示1：PostgreSQL全文检索配置（10分钟）

**操作步骤**：
```sql
-- 1. 为documents表添加全文检索字段
ALTER TABLE documents ADD COLUMN content_tsvector tsvector;

-- 2. 更新现有数据的检索向量
UPDATE documents SET content_tsvector = to_tsvector('chinese', content);

-- 3. 创建GIN索引
CREATE INDEX idx_documents_content_tsvector ON documents USING GIN(content_tsvector);

-- 4. 创建触发器自动更新检索向量
CREATE OR REPLACE FUNCTION update_content_tsvector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsvector = to_tsvector('chinese', NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trig_update_content_tsvector
    BEFORE INSERT OR UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_content_tsvector();
```

**讲解要点**：
- "我们添加一个专门的tsvector字段来存储预处理的检索向量"
- "GIN索引是专门为全文检索优化的，比普通B-tree索引快很多"
- "触发器确保新插入或更新的文档自动更新检索向量"

### 演示2：jieba分词测试（8分钟）

**操作步骤**：
```python
import jieba

# 测试不同类型的查询
test_queries = [
    "人工智能在医疗领域的应用",
    "机器学习算法优化",
    "深度学习神经网络",
    "自然语言处理技术"
]

print("=== jieba分词效果测试 ===")
for query in test_queries:
    tokens = list(jieba.cut(query, cut_all=False))
    print(f"原文: {query}")
    print(f"分词: {' | '.join(tokens)}")
    print(f"关键词: {[token for token in tokens if len(token) > 1]}")
    print("-" * 50)
```

**讲解要点**：
- "注意我们过滤掉了单字符的词，因为它们通常不是有效的关键词"
- "分词结果的质量直接影响检索效果"
- "可以根据领域需求添加自定义词典"

### 演示3：关键词检索实现（7分钟）

**操作步骤**：
```python
import jieba
import psycopg2
from typing import List, Dict, Any

def preprocess_query(query: str) -> List[str]:
    """预处理查询文本，进行分词"""
    # 使用jieba进行分词
    tokens = jieba.cut(query, cut_all=False)
    # 过滤掉长度小于2的词和标点符号
    keywords = [token.strip() for token in tokens 
                if len(token.strip()) > 1 and token.strip().isalnum()]
    return keywords

def keyword_search(query: str, limit: int = 10) -> Dict[str, Any]:
    """执行关键词检索"""
    # 预处理查询
    keywords = preprocess_query(query)
    
    if not keywords:
        return {"results": [], "total": 0, "message": "无有效关键词"}
    
    # 构建PostgreSQL查询
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
                            "score": float(row[2]),
                            "matched_keywords": keywords
                        }
                        for row in results
                    ],
                    "total": len(results),
                    "keywords": keywords
                }
    except Exception as e:
        return {"error": str(e), "results": [], "total": 0}

# 测试检索功能
test_query = "人工智能医疗应用"
result = keyword_search(test_query)
print(f"查询: {test_query}")
print(f"关键词: {result.get('keywords', [])}")
print(f"结果数量: {result.get('total', 0)}")
```

**讲解要点**：
- "preprocess_query函数负责将用户查询转换为关键词列表"
- "我们使用PostgreSQL的ts_rank函数来计算相关性得分"
- "错误处理确保系统的健壮性"

## 课堂提问与互动（15分钟）

### 提问1：概念理解
**问题**："关键词检索和向量检索的主要区别是什么？各自适用于什么场景？"

**参考答案**：
- 关键词检索基于精确匹配，向量检索基于语义相似度
- 关键词检索适用于精确查找、法律文档、技术手册
- 向量检索适用于语义搜索、推荐系统、模糊查询

### 提问2：技术细节
**问题**："为什么我们要使用GIN索引而不是普通的B-tree索引？"

**参考答案**：
- GIN索引专门为全文检索优化，支持倒排索引结构
- 对于tsvector类型的数据，GIN索引比B-tree索引快几个数量级
- GIN索引支持复杂的文本查询操作

### 提问3：实践应用
**问题**："如何评估jieba分词的效果？如果分词结果不理想怎么办？"

**参考答案**：
- 通过人工评估分词准确性，检查是否符合领域术语
- 可以添加自定义词典来改善特定领域的分词效果
- 可以使用其他分词工具如HanLP、LTP等进行对比

## 常见误区与应对话术

### 误区1：认为关键词检索已经过时
**应对话术**："关键词检索并没有过时，它在精确匹配场景下仍然是最佳选择。现代搜索系统通常结合使用多种检索技术。"

### 误区2：忽视中文分词的重要性
**应对话术**："中文分词质量直接影响检索效果。不同的分词结果会导致完全不同的检索结果，所以我们需要仔细调优分词策略。"

### 误区3：过度依赖默认配置
**应对话术**："PostgreSQL的中文全文检索需要正确配置。默认配置可能不适合中文文本，我们需要明确指定'chinese'配置。"

## 总结与下节课预告（10分钟）

### 本节课总结
"今天我们学习了：
1. 关键词检索的原理和应用场景
2. PostgreSQL全文检索的配置和使用
3. jieba中文分词技术
4. 完整的关键词检索系统实现

现在你们的RAG系统具备了两种检索能力：向量检索和关键词检索。"

### 下节课预告
"下节课我们将学习混合检索策略，把向量检索和关键词检索结合起来，发挥各自的优势，构建更强大的检索系统。请大家课后完成关键词检索功能的实现，下节课我们会在此基础上继续开发。"

### 课后任务提醒
"请完成以下任务：
1. 实现完整的关键词检索功能
2. 测试不同查询的检索效果
3. 优化分词和检索参数
4. 将功能集成到现有RAG系统中"