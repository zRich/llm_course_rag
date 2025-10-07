#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lesson07 功能测试脚本
测试关键词检索优化的所有功能
"""

import sys
import psycopg2
from keyword_search import keyword_search, preprocess_query
from test_jieba import test_segmentation

# 数据库连接配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'rag_db',
    'user': 'rag_user',
    'password': 'rag_password'
}

def test_database_connection():
    """测试数据库连接"""
    print("📊 测试数据库连接...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"✅ 数据库连接成功: {version[:50]}...")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return False

def test_database_schema():
    """测试数据库表结构"""
    print("\n🏗️ 测试数据库表结构...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 检查documents表是否存在content_vector字段
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'documents' AND column_name = 'content_vector'
        """)
        result = cursor.fetchone()
        
        if result:
            print(f"✅ content_vector字段存在: {result[1]}")
        else:
            print("❌ content_vector字段不存在")
            return False
        
        # 检查GIN索引是否存在
        cursor.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename = 'documents' AND indexname LIKE '%content_vector%'
        """)
        indexes = cursor.fetchall()
        
        if indexes:
            print(f"✅ GIN索引存在: {[idx[0] for idx in indexes]}")
        else:
            print("❌ GIN索引不存在")
            return False
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ 数据库表结构检查失败: {e}")
        return False

def test_data_content():
    """测试数据内容"""
    print("\n📄 测试数据内容...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 检查数据数量
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        print(f"✅ 文档数量: {count}")
        
        # 检查content_vector字段是否有数据
        cursor.execute("SELECT COUNT(*) FROM documents WHERE content_vector IS NOT NULL")
        vector_count = cursor.fetchone()[0]
        print(f"✅ 有向量数据的文档: {vector_count}")
        
        # 显示示例数据
        cursor.execute("SELECT title, LEFT(content, 50) FROM documents LIMIT 3")
        samples = cursor.fetchall()
        print("📋 示例数据:")
        for i, (title, content) in enumerate(samples, 1):
            print(f"  {i}. {title}: {content}...")
        
        cursor.close()
        conn.close()
        return count > 0 and vector_count > 0
    except Exception as e:
        print(f"❌ 数据内容检查失败: {e}")
        return False

def test_jieba_segmentation():
    """测试jieba分词"""
    print("\n🔤 测试jieba分词...")
    try:
        test_segmentation()
        print("✅ jieba分词测试通过")
        return True
    except Exception as e:
        print(f"❌ jieba分词测试失败: {e}")
        return False

def test_keyword_search_engine():
    """测试关键词搜索引擎"""
    print("\n🔍 测试关键词搜索引擎...")
    
    test_cases = [
        ("Python", "应该找到Python相关文档"),
        ("数据库", "应该找到数据库相关文档"),
        ("机器学习", "应该找到机器学习相关文档"),
        ("不存在的内容xyz123", "应该返回空结果")
    ]
    
    success_count = 0
    
    for query, expected in test_cases:
        try:
            print(f"\n  测试查询: '{query}' ({expected})")
            
            # 测试预处理
            processed = preprocess_query(query)
            print(f"    预处理结果: {processed}")
            
            # 测试搜索
            results = keyword_search(query, limit=2)
            print(f"    搜索结果数量: {len(results)}")
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"    {i}. [{result['score']}] {result['title']}")
            
            # 验证结果
            if query == "不存在的内容xyz123":
                if len(results) == 0:
                    print("    ✅ 空查询测试通过")
                    success_count += 1
                else:
                    print("    ❌ 空查询测试失败")
            else:
                if len(results) > 0:
                    print("    ✅ 搜索测试通过")
                    success_count += 1
                else:
                    print("    ❌ 搜索测试失败")
                    
        except Exception as e:
            print(f"    ❌ 搜索测试异常: {e}")
    
    print(f"\n搜索引擎测试结果: {success_count}/{len(test_cases)} 通过")
    return success_count == len(test_cases)

def run_all_tests():
    """运行所有测试"""
    print("🚀 Lesson07 功能测试开始")
    print("=" * 50)
    
    tests = [
        ("数据库连接", test_database_connection),
        ("数据库表结构", test_database_schema),
        ("数据内容", test_data_content),
        ("jieba分词", test_jieba_segmentation),
        ("关键词搜索引擎", test_keyword_search_engine)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试总结: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！Lesson07功能正常")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关功能")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)