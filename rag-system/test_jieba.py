#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文分词测试模块
演示jieba分词的基本用法
"""

import jieba

def test_segmentation():
    """测试中文分词功能"""
    # 测试文本
    test_texts = [
        "Python是一种高级编程语言",
        "数据库管理系统",
        "机器学习和人工智能"
    ]
    
    print("🔤 中文分词测试")
    print("=" * 40)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n测试 {i}: {text}")
        
        # 精确模式
        words1 = jieba.lcut(text)
        print(f"精确模式: {' / '.join(words1)}")
        
        # 搜索模式
        words2 = jieba.lcut_for_search(text)
        print(f"搜索模式: {' / '.join(words2)}")
        
        # 为PostgreSQL准备的文本
        search_ready = ' '.join(words2)
        print(f"搜索文本: {search_ready}")

if __name__ == "__main__":
    test_segmentation()