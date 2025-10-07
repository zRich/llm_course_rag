#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本清洗自动化流程
实现端到端的文本清洗管道，整合清洗、检测和评估功能
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import time

from .text_cleaner import TextCleaner
from .noise_detector import NoiseDetector
from .quality_evaluator import QualityEvaluator


@dataclass
class CleaningConfig:
    """清洗配置"""
    # 清洗级别
    cleaning_level: str = "standard"  # basic, standard, aggressive
    
    # 质量阈值
    min_quality_score: float = 0.6
    
    # 噪声阈值
    max_noise_ratio: float = 0.3
    
    # 是否保留原文
    keep_original: bool = True
    
    # 批处理大小
    batch_size: int = 100
    
    # 输出格式
    output_format: str = "json"  # json, txt, csv


@dataclass
class CleaningResult:
    """清洗结果"""
    original_text: str
    cleaned_text: str
    quality_score: float
    noise_ratio: float
    cleaning_stats: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class CleaningPipeline:
    """文本清洗流水线"""
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        
        # 初始化组件
        self.text_cleaner = TextCleaner()
        self.noise_detector = NoiseDetector()
        self.quality_evaluator = QualityEvaluator()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 统计信息
        self.stats = {
            "total_processed": 0,
            "successful_cleanings": 0,
            "failed_cleanings": 0,
            "total_processing_time": 0.0
        }
    
    def clean_single_text(self, text: str) -> CleaningResult:
        """清洗单个文本"""
        start_time = time.time()
        
        try:
            # 1. 预处理检查
            if not text or not text.strip():
                return CleaningResult(
                    original_text=text,
                    cleaned_text="",
                    quality_score=0.0,
                    noise_ratio=1.0,
                    cleaning_stats={},
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="Empty text"
                )
            
            # 2. 噪声检测
            noise_result = self.noise_detector.detect_noise(text)
            noise_ratio = noise_result["noise_ratio"]
            
            # 3. 根据配置选择清洗级别
            if self.config.cleaning_level == "basic":
                cleaned_text = self.text_cleaner.basic_clean(text)
            elif self.config.cleaning_level == "aggressive":
                cleaned_text = self.text_cleaner.advanced_clean(text)
            else:  # standard
                cleaned_text = self.text_cleaner.clean_text(text)
            
            # 4. 质量评估
            quality_result = self.quality_evaluator.evaluate_text(cleaned_text)
            quality_score = quality_result["overall_score"]
            
            # 5. 获取清洗统计
            cleaning_stats = self.text_cleaner.get_cleaning_stats(text, cleaned_text)
            
            # 6. 检查是否满足质量要求
            success = (
                quality_score >= self.config.min_quality_score and
                noise_ratio <= self.config.max_noise_ratio
            )
            
            processing_time = time.time() - start_time
            
            # 更新统计
            self.stats["total_processed"] += 1
            self.stats["total_processing_time"] += processing_time
            
            if success:
                self.stats["successful_cleanings"] += 1
            else:
                self.stats["failed_cleanings"] += 1
            
            return CleaningResult(
                original_text=text if self.config.keep_original else "",
                cleaned_text=cleaned_text,
                quality_score=quality_score,
                noise_ratio=noise_ratio,
                cleaning_stats=cleaning_stats,
                processing_time=processing_time,
                success=success
            )
            
        except Exception as e:
            self.logger.error(f"清洗文本时出错: {str(e)}")
            self.stats["total_processed"] += 1
            self.stats["failed_cleanings"] += 1
            
            return CleaningResult(
                original_text=text if self.config.keep_original else "",
                cleaned_text="",
                quality_score=0.0,
                noise_ratio=1.0,
                cleaning_stats={},
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def clean_batch(self, texts: List[str]) -> List[CleaningResult]:
        """批量清洗文本"""
        results = []
        
        # 分批处理
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            self.logger.info(f"处理批次 {i//self.config.batch_size + 1}, 大小: {len(batch)}")
            
            for text in batch:
                result = self.clean_single_text(text)
                results.append(result)
        
        return results
    
    def clean_from_file(self, input_file: str, output_file: Optional[str] = None) -> List[CleaningResult]:
        """从文件读取并清洗文本"""
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
        
        # 读取文本
        if input_path.suffix.lower() == '.json':
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    texts = [str(item) for item in data]
                elif isinstance(data, dict) and 'texts' in data:
                    texts = data['texts']
                else:
                    texts = [str(data)]
        else:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 按行分割或整体处理
                texts = content.split('\n') if '\n' in content else [content]
        
        # 清洗文本
        results = self.clean_batch(texts)
        
        # 保存结果
        if output_file:
            self.save_results(results, output_file)
        
        return results
    
    def save_results(self, results: List[CleaningResult], output_file: str):
        """保存清洗结果"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.output_format == "json":
            # JSON格式
            data = {
                "config": self.config.__dict__,
                "stats": self.stats,
                "results": [
                    {
                        "original_text": result.original_text,
                        "cleaned_text": result.cleaned_text,
                        "quality_score": result.quality_score,
                        "noise_ratio": result.noise_ratio,
                        "cleaning_stats": result.cleaning_stats,
                        "processing_time": result.processing_time,
                        "success": result.success,
                        "error_message": result.error_message
                    }
                    for result in results
                ]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        elif self.config.output_format == "txt":
            # 纯文本格式
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, result in enumerate(results):
                    f.write(f"=== 文本 {i+1} ===\n")
                    if result.success:
                        f.write(result.cleaned_text)
                    else:
                        f.write(f"清洗失败: {result.error_message}")
                    f.write("\n\n")
        
        elif self.config.output_format == "csv":
            # CSV格式
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'original_text', 'cleaned_text', 'quality_score', 
                    'noise_ratio', 'success', 'processing_time', 'error_message'
                ])
                
                for result in results:
                    writer.writerow([
                        result.original_text,
                        result.cleaned_text,
                        result.quality_score,
                        result.noise_ratio,
                        result.success,
                        result.processing_time,
                        result.error_message or ""
                    ])
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """获取流水线统计信息"""
        stats = self.stats.copy()
        
        if stats["total_processed"] > 0:
            stats["success_rate"] = stats["successful_cleanings"] / stats["total_processed"]
            stats["avg_processing_time"] = stats["total_processing_time"] / stats["total_processed"]
        else:
            stats["success_rate"] = 0.0
            stats["avg_processing_time"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_processed": 0,
            "successful_cleanings": 0,
            "failed_cleanings": 0,
            "total_processing_time": 0.0
        }


def create_pipeline(cleaning_level: str = "standard", 
                   min_quality: float = 0.6,
                   output_format: str = "json") -> CleaningPipeline:
    """创建清洗流水线的便捷函数"""
    config = CleaningConfig(
        cleaning_level=cleaning_level,
        min_quality_score=min_quality,
        output_format=output_format
    )
    return CleaningPipeline(config)


if __name__ == "__main__":
    # 示例用法
    pipeline = create_pipeline()
    
    # 测试文本
    test_texts = [
        "这是一个正常的文本。",
        "这是一个   有多余空格   的文本！！！",
        "这是一个包含HTML<tag>标签</tag>的文本。",
        "乱码测试：��������",
        "重复重复重复重复重复的内容"
    ]
    
    # 批量清洗
    results = pipeline.clean_batch(test_texts)
    
    # 显示结果
    for i, result in enumerate(results):
        print(f"\n=== 文本 {i+1} ===")
        print(f"原文: {result.original_text}")
        print(f"清洗后: {result.cleaned_text}")
        print(f"质量分数: {result.quality_score:.2f}")
        print(f"噪声比例: {result.noise_ratio:.2f}")
        print(f"处理时间: {result.processing_time:.3f}s")
        print(f"成功: {result.success}")
    
    # 显示统计信息
    print("\n=== 流水线统计 ===")
    stats = pipeline.get_pipeline_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")