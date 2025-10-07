#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关键词搜索引擎
基于PostgreSQL全文检索和jieba中文分词
"""

import jieba
import psycopg2
from typing import List, Dict

# 数据库连接配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'rag_db',
    'user': 'rag_user',
    'password': 'rag_password'
}

def preprocess_query(query: str) -> str:
    """预处理查询文本"""
    # 使用jieba分词
    words = jieba.lcut_for_search(query)
    
    # 过滤空词和单字符
    filtered_words = [w.strip() for w in words if len(w.strip()) > 1]
    
    # 构建tsquery格式
    if not filtered_words:
        return ''
    
    # 使用 | 连接（OR操作）以提高匹配率
    tsquery_parts = [f"'{word}':*" for word in filtered_words]
    return ' | '.join(tsquery_parts)

def keyword_search(query: str, limit: int = 5) -> List[Dict]:
    """执行关键词搜索"""
    # 预处理查询
    processed_query = preprocess_query(query)
    if not processed_query:
        return []
    
    # 连接数据库
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        # 执行搜索
        sql = """
        SELECT 
            id, title, content,
            ts_rank(content_vector, to_tsquery('simple', %s)) as score
        FROM documents 
        WHERE content_vector @@ to_tsquery('simple', %s)
        ORDER BY score DESC
        LIMIT %s
        """
        
        cursor.execute(sql, (processed_query, processed_query, limit))
        results = cursor.fetchall()
        
        # 格式化结果
        formatted_results = []
        for row in results:
            formatted_results.append({
                'id': row[0],
                'title': row[1],
                'content': row[2][:200] + '...' if len(row[2]) > 200 else row[2],
                'score': round(row[3], 4)
            })
        
        return formatted_results
    
    finally:
        cursor.close()
        conn.close()

def test_search():
    """测试搜索功能"""
    test_queries = [
        "Python",
        "数据库",
        "机器学习"
    ]
    
    print("🔍 关键词搜索测试")
    print("=" * 40)
    
    for query in test_queries:
        print(f"\n搜索: {query}")
        print("-" * 20)
        
        # 显示预处理结果
        processed = preprocess_query(query)
        print(f"预处理: {processed}")
        
        results = keyword_search(query, limit=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. [{result['score']}] {result['title']}")
                print(f"   {result['content'][:100]}...")
        else:
            print("   无搜索结果")

if __name__ == "__main__":
    test_search()