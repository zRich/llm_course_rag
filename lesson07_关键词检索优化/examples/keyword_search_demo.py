#!/usr/bin/env python3
"""
关键词检索演示示例
演示PostgreSQL全文检索和jieba中文分词的使用
"""

import psycopg2
import jieba
from typing import List, Dict, Any

class KeywordSearchDemo:
    def __init__(self, db_config: Dict[str, str]):
        """初始化数据库连接"""
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()
        
    def setup_fulltext_search(self):
        """设置PostgreSQL全文检索"""
        # 创建GIN索引用于全文检索
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_fulltext 
            ON documents USING GIN(to_tsvector('english', content));
        """)
        
        # 创建中文全文检索索引
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_fulltext_chinese 
            ON documents USING GIN(to_tsvector('simple', content));
        """)
        
        self.conn.commit()
        print("✅ 全文检索索引创建完成")
    
    def jieba_segment_demo(self, text: str) -> List[str]:
        """jieba分词演示"""
        print(f"原文: {text}")
        
        # 精确模式
        seg_precise = jieba.cut(text, cut_all=False)
        precise_result = list(seg_precise)
        print(f"精确模式: {' / '.join(precise_result)}")
        
        # 全模式
        seg_full = jieba.cut(text, cut_all=True)
        full_result = list(seg_full)
        print(f"全模式: {' / '.join(full_result)}")
        
        # 搜索引擎模式
        seg_search = jieba.cut_for_search(text)
        search_result = list(seg_search)
        print(f"搜索模式: {' / '.join(search_result)}")
        
        return precise_result
    
    def keyword_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """关键词检索"""
        # 使用jieba分词处理查询
        keywords = list(jieba.cut(query))
        search_query = ' & '.join(keywords)
        
        print(f"查询词: {query}")
        print(f"分词结果: {keywords}")
        print(f"PostgreSQL查询: {search_query}")
        
        # 执行全文检索
        sql = """
            SELECT id, title, content, 
                   ts_rank(to_tsvector('simple', content), to_tsquery('simple', %s)) as rank
            FROM documents 
            WHERE to_tsvector('simple', content) @@ to_tsquery('simple', %s)
            ORDER BY rank DESC
            LIMIT %s;
        """
        
        self.cursor.execute(sql, (search_query, search_query, limit))
        results = self.cursor.fetchall()
        
        # 格式化结果
        formatted_results = []
        for row in results:
            formatted_results.append({
                'id': row[0],
                'title': row[1],
                'content': row[2][:200] + '...' if len(row[2]) > 200 else row[2],
                'rank': float(row[3])
            })
        
        return formatted_results
    
    def demo_run(self):
        """运行完整演示"""
        print("=== 关键词检索演示 ===\n")
        
        # 1. 分词演示
        print("1. jieba分词演示:")
        demo_texts = [
            "人工智能技术在自然语言处理中的应用",
            "机器学习算法优化与性能提升",
            "深度学习模型训练与调优"
        ]
        
        for text in demo_texts:
            self.jieba_segment_demo(text)
            print()
        
        # 2. 关键词检索演示
        print("2. 关键词检索演示:")
        demo_queries = [
            "人工智能",
            "机器学习算法",
            "深度学习模型"
        ]
        
        for query in demo_queries:
            print(f"\n--- 搜索: {query} ---")
            results = self.keyword_search(query)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"{i}. [{result['rank']:.4f}] {result['title']}")
                    print(f"   {result['content']}")
            else:
                print("未找到相关结果")
            print()
    
    def close(self):
        """关闭数据库连接"""
        self.cursor.close()
        self.conn.close()

if __name__ == "__main__":
    # 数据库配置
    db_config = {
        'host': 'localhost',
        'database': 'rag_system',
        'user': 'postgres',
        'password': 'password'
    }
    
    try:
        demo = KeywordSearchDemo(db_config)
        demo.setup_fulltext_search()
        demo.demo_run()
    except Exception as e:
        print(f"演示运行出错: {e}")
    finally:
        demo.close()