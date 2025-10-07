#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本清洗功能演示脚本
提供交互式的功能演示和测试
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from text_cleaner import TextCleaner
from noise_detector import NoiseDetector
from quality_evaluator import QualityEvaluator
from cleaning_pipeline import CleaningPipeline, CleaningConfig, create_pipeline


def print_separator(title: str = ""):
    """打印分隔线"""
    if title:
        print(f"\n{'='*20} {title} {'='*20}")
    else:
        print("="*60)


def demo_text_cleaner():
    """演示文本清洗功能"""
    print_separator("文本清洗演示")
    
    cleaner = TextCleaner()
    
    # 测试文本
    test_texts = [
        "  这是一个有多余空格的文本  ",
        "这是一个包含HTML<p>标签</p>的文本",
        "访问 https://www.example.com 获取更多信息",
        "这是包含@#$%特殊字符的文本",
        "这是一个\t\t有制表符\n\n和换行符的文本",
        "重复！！！标点符号？？？的文本"
    ]
    
    print("\n1. 基础清洗演示:")
    for i, text in enumerate(test_texts, 1):
        cleaned = cleaner.basic_clean(text)
        print(f"  {i}. 原文: {repr(text)}")
        print(f"     清洗后: {repr(cleaned)}")
        print()
    
    print("\n2. 高级清洗演示:")
    advanced_text = "  这是一个<div>包含HTML</div>标签和URL https://example.com 的复杂文本！！！  "
    print(f"原文: {repr(advanced_text)}")
    
    basic_result = cleaner.basic_clean(advanced_text)
    standard_result = cleaner.clean_text(advanced_text)
    advanced_result = cleaner.advanced_clean(advanced_text)
    
    print(f"基础清洗: {repr(basic_result)}")
    print(f"标准清洗: {repr(standard_result)}")
    print(f"高级清洗: {repr(advanced_result)}")
    
    # 清洗统计
    stats = cleaner.get_cleaning_stats(advanced_text, advanced_result)
    print(f"\n清洗统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")


def demo_noise_detector():
    """演示噪声检测功能"""
    print_separator("噪声检测演示")
    
    detector = NoiseDetector()
    
    # 测试文本
    test_cases = [
        ("正常文本", "这是一个质量良好的正常文本，内容完整且结构清晰。"),
        ("乱码文本", "这是包含乱码��������的文本"),
        ("重复内容", "重复重复重复重复重复的内容"),
        ("无意义行", "aaa\nbbb\nccc\n!!!\n???"),
        ("编码问题", "这是包含编码问题的文本：\x00\x01\x02"),
        ("格式噪声", "这是一个包含大量!!!???###符号的文本")
    ]
    
    for name, text in test_cases:
        print(f"\n{name}:")
        print(f"文本: {repr(text)}")
        
        result = detector.detect_noise(text)
        print(f"噪声检测结果:")
        print(f"  - 噪声比例: {result['noise_ratio']:.2f}")
        print(f"  - 乱码检测: {result['garbled_detection']['has_garbled']}")
        print(f"  - 重复内容: {result['repetitive_detection']['has_repetition']}")
        print(f"  - 无意义行: {result['meaningless_detection']['meaningless_count']}")
        print(f"  - 编码问题: {result['encoding_detection']['has_encoding_issues']}")
        print(f"  - 格式噪声: {result['format_detection']['noise_score']:.2f}")


def demo_quality_evaluator():
    """演示质量评估功能"""
    print_separator("质量评估演示")
    
    evaluator = QualityEvaluator()
    
    # 测试文本
    test_cases = [
        ("高质量文本", "这是一个高质量的文本示例。内容结构清晰，语言表达准确，信息完整详细。文本具有良好的可读性和一致性，能够有效传达预期的信息内容。"),
        ("中等质量文本", "这是一个中等质量的文本。内容基本完整，但表达不够清晰。"),
        ("低质量文本", "这是低质量文本"),
        ("空文本", ""),
        ("技术文档", "本文档介绍了机器学习中的文本预处理技术。文本预处理是自然语言处理的重要步骤，包括分词、去噪、标准化等操作。")
    ]
    
    for name, text in test_cases:
        print(f"\n{name}:")
        print(f"文本: {repr(text)}")
        
        if text:  # 非空文本才评估
            result = evaluator.evaluate_text(text)
            print(f"质量评估结果:")
            print(f"  - 总体分数: {result['overall_score']:.2f}")
            print(f"  - 可读性: {result['readability']['score']:.2f}")
            print(f"  - 完整性: {result['completeness']['score']:.2f}")
            print(f"  - 一致性: {result['consistency']['score']:.2f}")
            print(f"  - 信息密度: {result['information_density']['score']:.2f}")
            print(f"  - 结构质量: {result['structure']['score']:.2f}")
            
            if result['suggestions']:
                print(f"  - 改进建议: {', '.join(result['suggestions'])}")
        else:
            print("  - 空文本，无法评估")


def demo_cleaning_pipeline():
    """演示清洗流水线功能"""
    print_separator("清洗流水线演示")
    
    # 创建不同配置的流水线
    configs = [
        ("基础配置", CleaningConfig(cleaning_level="basic", min_quality_score=0.5)),
        ("标准配置", CleaningConfig(cleaning_level="standard", min_quality_score=0.6)),
        ("严格配置", CleaningConfig(cleaning_level="aggressive", min_quality_score=0.8))
    ]
    
    # 测试文本
    test_texts = [
        "这是一个正常的文本。",
        "  这是一个<p>包含HTML</p>标签和多余空格的文本  ",
        "这是包含URL https://example.com 和乱码��������的文本",
        "重复重复重复重复的内容！！！",
        "这是一个质量较好的文本，内容完整，表达清晰。"
    ]
    
    for config_name, config in configs:
        print(f"\n{config_name}:")
        pipeline = CleaningPipeline(config)
        
        results = pipeline.clean_batch(test_texts)
        
        for i, result in enumerate(results, 1):
            print(f"  文本{i}: 成功={result.success}, 质量={result.quality_score:.2f}, 噪声={result.noise_ratio:.2f}")
            if not result.success and result.error_message:
                print(f"    错误: {result.error_message}")
        
        # 显示统计信息
        stats = pipeline.get_pipeline_stats()
        print(f"  统计: 处理{stats['total_processed']}个, 成功率{stats['success_rate']:.1%}")


def demo_batch_processing():
    """演示批量处理功能"""
    print_separator("批量处理演示")
    
    # 创建测试数据
    test_data = [
        "这是第一个测试文本。",
        "  这是第二个有多余空格的文本  ",
        "这是第三个包含<b>HTML标签</b>的文本",
        "这是第四个包含URL https://example.com 的文本",
        "这是第五个包含乱码��������的文本",
        "重复重复重复重复的第六个文本",
        "这是第七个质量良好的文本，内容完整且表达清晰。",
        "短文本",
        "",  # 空文本
        "这是第十个正常的结束文本。"
    ]
    
    print(f"批量处理 {len(test_data)} 个文本...")
    
    # 使用标准配置
    pipeline = create_pipeline("standard", 0.6)
    results = pipeline.clean_batch(test_data)
    
    # 统计结果
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    avg_quality = sum(r.quality_score for r in results if r.success) / max(successful, 1)
    avg_noise = sum(r.noise_ratio for r in results) / len(results)
    
    print(f"\n批量处理结果:")
    print(f"  - 总文本数: {len(results)}")
    print(f"  - 成功处理: {successful}")
    print(f"  - 处理失败: {failed}")
    print(f"  - 成功率: {successful/len(results):.1%}")
    print(f"  - 平均质量分数: {avg_quality:.2f}")
    print(f"  - 平均噪声比例: {avg_noise:.2f}")
    
    # 显示详细结果
    print(f"\n详细结果:")
    for i, result in enumerate(results, 1):
        status = "✓" if result.success else "✗"
        print(f"  {i:2d}. {status} 质量:{result.quality_score:.2f} 噪声:{result.noise_ratio:.2f} 时间:{result.processing_time:.3f}s")
        if result.error_message:
            print(f"      错误: {result.error_message}")


def interactive_demo():
    """交互式演示"""
    print_separator("交互式文本清洗")
    
    pipeline = create_pipeline()
    
    print("输入文本进行清洗（输入 'quit' 退出）:")
    
    while True:
        try:
            text = input("\n请输入文本: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                print("请输入有效文本")
                continue
            
            # 清洗文本
            result = pipeline.clean_single_text(text)
            
            print(f"\n清洗结果:")
            print(f"  原文: {repr(text)}")
            print(f"  清洗后: {repr(result.cleaned_text)}")
            print(f"  质量分数: {result.quality_score:.2f}")
            print(f"  噪声比例: {result.noise_ratio:.2f}")
            print(f"  处理时间: {result.processing_time:.3f}s")
            print(f"  处理成功: {result.success}")
            
            if result.error_message:
                print(f"  错误信息: {result.error_message}")
            
        except KeyboardInterrupt:
            print("\n\n退出交互模式")
            break
        except Exception as e:
            print(f"处理出错: {str(e)}")


def main():
    """主函数"""
    print("文本清洗功能演示")
    print("==================")
    
    demos = [
        ("1", "文本清洗演示", demo_text_cleaner),
        ("2", "噪声检测演示", demo_noise_detector),
        ("3", "质量评估演示", demo_quality_evaluator),
        ("4", "清洗流水线演示", demo_cleaning_pipeline),
        ("5", "批量处理演示", demo_batch_processing),
        ("6", "交互式演示", interactive_demo),
        ("a", "运行所有演示", None),
        ("q", "退出", None)
    ]
    
    while True:
        print("\n请选择演示功能:")
        for key, name, _ in demos:
            print(f"  {key}. {name}")
        
        choice = input("\n请输入选择 (1-6, a, q): ").strip().lower()
        
        if choice == 'q':
            print("再见！")
            break
        elif choice == 'a':
            # 运行所有演示（除了交互式）
            for key, name, func in demos[:-3]:  # 排除交互式、全部、退出
                if func:
                    func()
        else:
            # 查找对应的演示函数
            demo_func = None
            for key, name, func in demos:
                if key == choice and func:
                    demo_func = func
                    break
            
            if demo_func:
                try:
                    demo_func()
                except Exception as e:
                    print(f"演示出错: {str(e)}")
            else:
                print("无效选择，请重新输入")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"程序出错: {str(e)}")