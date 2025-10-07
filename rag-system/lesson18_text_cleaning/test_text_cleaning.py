#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本清洗功能测试
测试各个模块的基础功能
"""

import unittest
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from text_cleaner import TextCleaner
from noise_detector import NoiseDetector
from quality_evaluator import QualityEvaluator
from cleaning_pipeline import CleaningPipeline, CleaningConfig


class TestTextCleaner(unittest.TestCase):
    """测试文本清洗器"""
    
    def setUp(self):
        self.cleaner = TextCleaner()
    
    def test_basic_clean(self):
        """测试基础清洗"""
        text = "  这是一个   有多余空格   的文本  "
        result = self.cleaner.basic_clean(text)
        self.assertEqual(result, "这是一个 有多余空格 的文本")
    
    def test_remove_extra_spaces(self):
        """测试去除多余空格"""
        text = "这是\t\t一个\n\n有多余\r\n空格的文本"
        result = self.cleaner.remove_extra_spaces(text)
        self.assertNotIn("\t\t", result)
        self.assertNotIn("\n\n", result)
    
    def test_remove_special_chars(self):
        """测试去除特殊字符"""
        text = "这是一个包含@#$%特殊字符的文本"
        result = self.cleaner.remove_special_chars(text)
        self.assertNotIn("@#$%", result)
    
    def test_remove_html_tags(self):
        """测试去除HTML标签"""
        text = "这是一个<div>包含HTML</div>标签的<p>文本</p>"
        result = self.cleaner.remove_html_tags(text)
        self.assertNotIn("<div>", result)
        self.assertNotIn("</div>", result)
        self.assertIn("包含HTML", result)
    
    def test_remove_urls(self):
        """测试去除URL"""
        text = "访问 https://www.example.com 获取更多信息"
        result = self.cleaner.remove_urls(text)
        self.assertNotIn("https://www.example.com", result)
        self.assertIn("访问", result)
        self.assertIn("获取更多信息", result)
    
    def test_batch_clean(self):
        """测试批量清洗"""
        texts = [
            "  文本1  ",
            "<p>文本2</p>",
            "文本3 https://example.com"
        ]
        results = self.cleaner.batch_clean(texts)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], "文本1")
        self.assertNotIn("<p>", results[1])
        self.assertNotIn("https://example.com", results[2])


class TestNoiseDetector(unittest.TestCase):
    """测试噪声检测器"""
    
    def setUp(self):
        self.detector = NoiseDetector()
    
    def test_detect_garbled_text(self):
        """测试检测乱码"""
        clean_text = "这是正常的中文文本"
        garbled_text = "这是包含乱码��������的文本"
        
        clean_result = self.detector.detect_garbled_text(clean_text)
        garbled_result = self.detector.detect_garbled_text(garbled_text)
        
        self.assertFalse(clean_result["has_garbled"])
        self.assertTrue(garbled_result["has_garbled"])
    
    def test_detect_repetitive_content(self):
        """测试检测重复内容"""
        normal_text = "这是一个正常的文本内容"
        repetitive_text = "重复重复重复重复重复的内容"
        
        normal_result = self.detector.detect_repetitive_content(normal_text)
        repetitive_result = self.detector.detect_repetitive_content(repetitive_text)
        
        self.assertFalse(normal_result["has_repetition"])
        self.assertTrue(repetitive_result["has_repetition"])
    
    def test_detect_meaningless_lines(self):
        """测试检测无意义行"""
        meaningful_text = "这是有意义的文本内容，包含完整的句子。"
        meaningless_text = "aaa\nbbb\nccc\n!!!"
        
        meaningful_result = self.detector.detect_meaningless_lines(meaningful_text)
        meaningless_result = self.detector.detect_meaningless_lines(meaningless_text)
        
        self.assertEqual(meaningful_result["meaningless_count"], 0)
        self.assertGreater(meaningless_result["meaningless_count"], 0)
    
    def test_comprehensive_noise_detection(self):
        """测试综合噪声检测"""
        clean_text = "这是一个质量良好的文本，内容完整且结构清晰。"
        noisy_text = "这是包含乱码��������和重复重复重复内容的文本"
        
        clean_result = self.detector.detect_noise(clean_text)
        noisy_result = self.detector.detect_noise(noisy_text)
        
        self.assertLess(clean_result["noise_ratio"], 0.3)
        self.assertGreater(noisy_result["noise_ratio"], 0.3)


class TestQualityEvaluator(unittest.TestCase):
    """测试质量评估器"""
    
    def setUp(self):
        self.evaluator = QualityEvaluator()
    
    def test_evaluate_readability(self):
        """测试可读性评估"""
        readable_text = "这是一个结构清晰、语言流畅的文本。内容完整，表达准确。"
        unreadable_text = "这是一个结构混乱语言不通的文本内容表达不清楚"
        
        readable_result = self.evaluator.evaluate_readability(readable_text)
        unreadable_result = self.evaluator.evaluate_readability(unreadable_text)
        
        self.assertGreater(readable_result["score"], unreadable_result["score"])
    
    def test_evaluate_completeness(self):
        """测试完整性评估"""
        complete_text = "这是一个完整的文本，包含开头、中间和结尾部分。内容详细，信息充分。"
        incomplete_text = "这是"
        
        complete_result = self.evaluator.evaluate_completeness(complete_text)
        incomplete_result = self.evaluator.evaluate_completeness(incomplete_text)
        
        self.assertGreater(complete_result["score"], incomplete_result["score"])
    
    def test_evaluate_consistency(self):
        """测试一致性评估"""
        consistent_text = "这是关于机器学习的文本。机器学习是人工智能的重要分支。"
        inconsistent_text = "这是关于机器学习的文本。今天天气很好。"
        
        consistent_result = self.evaluator.evaluate_consistency(consistent_text)
        inconsistent_result = self.evaluator.evaluate_consistency(inconsistent_text)
        
        self.assertGreater(consistent_result["score"], inconsistent_result["score"])
    
    def test_comprehensive_evaluation(self):
        """测试综合评估"""
        high_quality_text = "这是一个高质量的文本示例。内容结构清晰，语言表达准确，信息完整详细。文本具有良好的可读性和一致性，能够有效传达信息。"
        low_quality_text = "这是低质量文本"
        
        high_result = self.evaluator.evaluate_text(high_quality_text)
        low_result = self.evaluator.evaluate_text(low_quality_text)
        
        self.assertGreater(high_result["overall_score"], low_result["overall_score"])
        self.assertGreater(high_result["overall_score"], 0.7)
        self.assertLess(low_result["overall_score"], 0.5)


class TestCleaningPipeline(unittest.TestCase):
    """测试清洗流水线"""
    
    def setUp(self):
        self.config = CleaningConfig(
            cleaning_level="standard",
            min_quality_score=0.6,
            max_noise_ratio=0.3
        )
        self.pipeline = CleaningPipeline(self.config)
    
    def test_clean_single_text(self):
        """测试单文本清洗"""
        text = "  这是一个<p>包含HTML</p>标签的   文本  "
        result = self.pipeline.clean_single_text(text)
        
        self.assertTrue(result.success)
        self.assertNotIn("<p>", result.cleaned_text)
        self.assertNotIn("</p>", result.cleaned_text)
        self.assertGreater(result.quality_score, 0)
        self.assertLess(result.processing_time, 1.0)  # 应该很快完成
    
    def test_clean_batch(self):
        """测试批量清洗"""
        texts = [
            "正常文本",
            "  有多余空格的文本  ",
            "<div>包含HTML标签</div>的文本",
            "包含URL https://example.com 的文本"
        ]
        
        results = self.pipeline.clean_batch(texts)
        
        self.assertEqual(len(results), 4)
        for result in results:
            self.assertIsNotNone(result.cleaned_text)
            self.assertGreaterEqual(result.quality_score, 0)
            self.assertLessEqual(result.noise_ratio, 1.0)
    
    def test_pipeline_stats(self):
        """测试流水线统计"""
        texts = ["文本1", "文本2", "文本3"]
        self.pipeline.clean_batch(texts)
        
        stats = self.pipeline.get_pipeline_stats()
        
        self.assertEqual(stats["total_processed"], 3)
        self.assertGreaterEqual(stats["successful_cleanings"], 0)
        self.assertGreaterEqual(stats["success_rate"], 0)
        self.assertGreater(stats["avg_processing_time"], 0)
    
    def test_empty_text_handling(self):
        """测试空文本处理"""
        result = self.pipeline.clean_single_text("")
        
        self.assertFalse(result.success)
        self.assertEqual(result.cleaned_text, "")
        self.assertEqual(result.quality_score, 0.0)
        self.assertEqual(result.noise_ratio, 1.0)
        self.assertIsNotNone(result.error_message)


def run_basic_tests():
    """运行基础测试"""
    print("=== 运行文本清洗功能测试 ===")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_suite.addTest(unittest.makeSuite(TestTextCleaner))
    test_suite.addTest(unittest.makeSuite(TestNoiseDetector))
    test_suite.addTest(unittest.makeSuite(TestQualityEvaluator))
    test_suite.addTest(unittest.makeSuite(TestCleaningPipeline))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出结果
    print(f"\n=== 测试结果 ===")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"成功率: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result.wasSuccessful()


def run_quick_test():
    """快速功能测试"""
    print("=== 快速功能测试 ===")
    
    try:
        # 测试文本清洗
        cleaner = TextCleaner()
        test_text = "  这是一个<p>测试</p>文本  "
        cleaned = cleaner.clean_text(test_text)
        print(f"✓ 文本清洗: '{test_text}' -> '{cleaned}'")
        
        # 测试噪声检测
        detector = NoiseDetector()
        noise_result = detector.detect_noise("这是包含乱码��������的文本")
        print(f"✓ 噪声检测: 噪声比例 {noise_result['noise_ratio']:.2f}")
        
        # 测试质量评估
        evaluator = QualityEvaluator()
        quality_result = evaluator.evaluate_text("这是一个质量良好的测试文本。")
        print(f"✓ 质量评估: 总分 {quality_result['overall_score']:.2f}")
        
        # 测试流水线
        pipeline = CleaningPipeline()
        result = pipeline.clean_single_text(test_text)
        print(f"✓ 清洗流水线: 成功={result.success}, 质量分数={result.quality_score:.2f}")
        
        print("\n所有快速测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="文本清洗功能测试")
    parser.add_argument("--quick", action="store_true", help="运行快速测试")
    parser.add_argument("--full", action="store_true", help="运行完整测试")
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_test()
    elif args.full:
        success = run_basic_tests()
    else:
        # 默认运行快速测试
        success = run_quick_test()
        if success:
            print("\n如需运行完整测试，请使用: python test_text_cleaning.py --full")
    
    exit(0 if success else 1)