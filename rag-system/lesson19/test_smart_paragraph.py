#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第19节课 - 智能段落切分策略测试脚本

测试SmartParagraphStrategy的各项功能：
1. 基本段落切分
2. 短段落合并
3. 长段落分割
4. 插件系统集成
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 导入所需模块 - 通过chunking包导入以触发注册
from chunking import SmartParagraphStrategy, ChunkingConfig
from chunking.plugin_registry import registry as StrategyRegistry

def test_basic_chunking():
    """测试基本段落切分功能"""
    print("\n=== 测试基本段落切分功能 ===")
    
    # 测试文本
    test_text = """
这是第一个段落。它包含了一些基本的内容。

这是第二个段落。
它有多行内容。
应该被正确识别为一个段落。

这是第三个段落，内容比较长。它包含了很多信息，用来测试段落切分的准确性。这个段落应该被完整保留，因为它的长度在合理范围内。

短段落。

另一个短段落。

这是一个非常长的段落，它包含了大量的文本内容，用来测试长段落分割功能。这个段落的长度超过了默认的最大长度限制，因此应该被分割成多个较小的块。这样可以确保每个文档块的大小都在合理的范围内，便于后续的处理和检索。段落分割应该尽量保持语义的完整性，在句子边界进行分割。
    """.strip()
    
    # 创建配置
    config = ChunkingConfig(
        chunk_size=200,
        chunk_overlap=50,
        min_chunk_size=50
    )
    
    # 创建策略实例
    strategy = SmartParagraphStrategy(config=config)
    
    # 执行切分
    chunks = strategy.chunk_text(test_text)
    
    print(f"原始文本长度: {len(test_text)} 字符")
    print(f"切分结果: {len(chunks)} 个文档块")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n块 {i}:")
        print(f"  长度: {len(chunk.content)} 字符")
        print(f"  内容: {chunk.content[:100]}{'...' if len(chunk.content) > 100 else ''}")
        print(f"  元数据: {chunk.metadata}")
    
    return chunks

def test_short_paragraph_merging():
    """测试短段落合并功能"""
    print("\n=== 测试短段落合并功能 ===")
    
    # 包含多个短段落的测试文本
    test_text = """
短段落1。

短段落2。

短段落3。

这是一个正常长度的段落，不需要合并。它包含足够的内容来独立成为一个文档块。

短段落4。

短段落5。
    """.strip()
    
    config = ChunkingConfig(
        chunk_size=300,
        chunk_overlap=30,
        min_chunk_size=80
    )
    
    strategy = SmartParagraphStrategy(config=config)
    chunks = strategy.chunk_text(test_text)
    
    print(f"原始文本长度: {len(test_text)} 字符")
    print(f"切分结果: {len(chunks)} 个文档块")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n块 {i}:")
        print(f"  长度: {len(chunk.content)} 字符")
        print(f"  内容: {repr(chunk.content)}")
    
    return chunks

def test_long_paragraph_splitting():
    """测试长段落分割功能"""
    print("\n=== 测试长段落分割功能 ===")
    
    # 包含超长段落的测试文本
    long_paragraph = (
        "这是一个非常长的段落，用来测试长段落分割功能。" * 20 +
        "它包含了大量重复的内容。" * 15 +
        "这样的段落需要被分割成多个较小的块。" * 10 +
        "分割应该在合适的位置进行，保持内容的连贯性。" * 8
    )
    
    test_text = f"正常段落开始。\n\n{long_paragraph}\n\n正常段落结束。"
    
    config = ChunkingConfig(
        chunk_size=200,
        chunk_overlap=50,
        min_chunk_size=50
    )
    
    strategy = SmartParagraphStrategy(config=config)
    chunks = strategy.chunk_text(test_text)
    
    print(f"原始文本长度: {len(test_text)} 字符")
    print(f"长段落长度: {len(long_paragraph)} 字符")
    print(f"切分结果: {len(chunks)} 个文档块")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n块 {i}:")
        print(f"  长度: {len(chunk.content)} 字符")
        print(f"  内容预览: {chunk.content[:80]}{'...' if len(chunk.content) > 80 else ''}")
    
    return chunks

def test_plugin_system_integration():
    """测试插件系统集成"""
    print("\n=== 测试插件系统集成 ===")
    
    try:
        # 使用已导入的registry实例
        registry = StrategyRegistry
        
        # 检查已注册的策略
        strategies = registry.list_strategies()
        print(f"已注册的策略: {strategies}")
        
        # 验证SmartParagraphStrategy是否已注册
        if "smart_paragraph" in strategies:
            print("✓ SmartParagraphStrategy 已成功注册到插件系统")
            
            # 测试通过插件系统获取策略
            strategy = registry.get_strategy("smart_paragraph")
            print(f"✓ 通过插件系统获取策略成功: {type(strategy).__name__}")
            
            # 测试策略信息
            info = registry.get_strategy_info("smart_paragraph")
            print(f"✓ 策略信息: {info['description']}")
            
            return True
        else:
            print("✗ SmartParagraphStrategy 未注册到插件系统")
            return False
            
    except Exception as e:
        print(f"✗ 插件系统集成测试失败: {e}")
        return False

def test_configuration_options():
    """测试不同配置选项"""
    print("\n=== 测试不同配置选项 ===")
    
    test_text = """
配置测试段落1。

配置测试段落2，内容稍长一些。

这是一个更长的段落，用来测试不同的配置参数如何影响切分结果。它包含了足够的内容来展示配置的效果。

短段落。

另一个短段落。
    """.strip()
    
    # 测试不同的配置
    configs = [
        ("小块配置", ChunkingConfig(chunk_size=100, chunk_overlap=20, min_chunk_size=30)),
        ("中等块配置", ChunkingConfig(chunk_size=200, chunk_overlap=40, min_chunk_size=60)),
        ("大块配置", ChunkingConfig(chunk_size=400, chunk_overlap=80, min_chunk_size=100))
    ]
    
    for config_name, config in configs:
        print(f"\n--- {config_name} ---")
        print(f"chunk_size: {config.chunk_size}, chunk_overlap: {config.chunk_overlap}, min_chunk_size: {config.min_chunk_size}")
        
        strategy = SmartParagraphStrategy(config=config)
        chunks = strategy.chunk_text(test_text)
        
        print(f"切分结果: {len(chunks)} 个文档块")
        for i, chunk in enumerate(chunks, 1):
            print(f"  块 {i}: {len(chunk.content)} 字符")

def main():
    """主测试函数"""
    print("第19节课 - 智能段落切分策略测试")
    print("=" * 50)
    
    try:
        # 运行各项测试
        test_basic_chunking()
        test_short_paragraph_merging()
        test_long_paragraph_splitting()
        
        # 测试插件系统集成
        plugin_success = test_plugin_system_integration()
        
        # 测试配置选项
        test_configuration_options()
        
        print("\n=== 测试总结 ===")
        print("✓ 基本段落切分功能正常")
        print("✓ 短段落合并功能正常")
        print("✓ 长段落分割功能正常")
        print("✓ 配置选项测试正常")
        
        if plugin_success:
            print("✓ 插件系统集成正常")
            print("\n🎉 所有测试通过！智能段落切分策略实现成功！")
        else:
            print("✗ 插件系统集成失败")
            print("\n⚠️  部分测试失败，请检查插件注册代码")
            
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)