#!/usr/bin/env python3
"""
智能Chunk拆分演示示例
展示不同拆分策略的实现和效果对比

作者: RAG课程组
日期: 2024年
用途: Lesson 04 课堂演示
"""

import re
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class ChunkStrategy(Enum):
    """Chunk拆分策略枚举"""
    FIXED_LENGTH = "fixed_length"
    SEMANTIC_BOUNDARY = "semantic_boundary"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class ChunkInfo:
    """Chunk信息数据类"""
    text: str
    start_pos: int
    end_pos: int
    length: int
    strategy: str
    overlap_with_previous: int = 0


class SmartChunker:
    """智能Chunk拆分器"""
    
    def __init__(self, max_chunk_size: int = 1000, overlap_size: int = 100):
        """
        初始化拆分器
        
        Args:
            max_chunk_size: 最大chunk长度
            overlap_size: 重叠长度
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
        # 语义边界模式（按优先级排序）
        self.boundary_patterns = [
            (r'\n\n+', '段落边界'),           # 段落边界（最高优先级）
            (r'[。！？]\s*\n', '句子换行边界'),  # 句子边界+换行
            (r'[。！？]\s+', '句子边界'),       # 句子边界
            (r'[，；]\s+', '子句边界'),         # 子句边界
            (r'\s+', '词汇边界'),              # 词汇边界（最低优先级）
        ]
    
    def split_by_fixed_length(self, text: str) -> List[ChunkInfo]:
        """
        固定长度拆分策略
        
        Args:
            text: 待拆分文本
            
        Returns:
            Chunk信息列表
        """
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            end_pos = min(current_pos + self.max_chunk_size, len(text))
            chunk_text = text[current_pos:end_pos].strip()
            
            if chunk_text:
                chunks.append(ChunkInfo(
                    text=chunk_text,
                    start_pos=current_pos,
                    end_pos=end_pos,
                    length=len(chunk_text),
                    strategy=ChunkStrategy.FIXED_LENGTH.value
                ))
            
            current_pos = end_pos
        
        return chunks
    
    def split_by_semantic_boundary(self, text: str) -> List[ChunkInfo]:
        """
        语义边界拆分策略
        
        Args:
            text: 待拆分文本
            
        Returns:
            Chunk信息列表
        """
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # 寻找最优分割点
            end_pos = self._find_optimal_split_point(
                text, current_pos, current_pos + self.max_chunk_size
            )
            
            chunk_text = text[current_pos:end_pos].strip()
            
            if chunk_text:
                chunks.append(ChunkInfo(
                    text=chunk_text,
                    start_pos=current_pos,
                    end_pos=end_pos,
                    length=len(chunk_text),
                    strategy=ChunkStrategy.SEMANTIC_BOUNDARY.value
                ))
            
            current_pos = end_pos
        
        return chunks
    
    def split_by_sliding_window(self, text: str) -> List[ChunkInfo]:
        """
        滑动窗口拆分策略
        
        Args:
            text: 待拆分文本
            
        Returns:
            Chunk信息列表
        """
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            end_pos = self._find_optimal_split_point(
                text, current_pos, current_pos + self.max_chunk_size
            )
            
            chunk_text = text[current_pos:end_pos].strip()
            
            if chunk_text:
                # 计算与前一个chunk的重叠
                overlap = 0
                if chunks:
                    prev_chunk = chunks[-1]
                    overlap_start = max(current_pos, prev_chunk.end_pos - self.overlap_size)
                    overlap = current_pos - overlap_start if current_pos > overlap_start else 0
                
                chunks.append(ChunkInfo(
                    text=chunk_text,
                    start_pos=current_pos,
                    end_pos=end_pos,
                    length=len(chunk_text),
                    strategy=ChunkStrategy.SLIDING_WINDOW.value,
                    overlap_with_previous=overlap
                ))
            
            # 滑动窗口：下一个chunk的起始位置考虑重叠
            next_start = max(end_pos - self.overlap_size, current_pos + 1)
            current_pos = next_start if next_start < len(text) else len(text)
        
        return chunks
    
    def _find_optimal_split_point(self, text: str, start: int, max_end: int) -> int:
        """
        寻找最优的分割点
        
        Args:
            text: 文本
            start: 起始位置
            max_end: 最大结束位置
            
        Returns:
            最优分割点位置
        """
        if max_end >= len(text):
            return len(text)
        
        # 在指定范围内搜索
        search_text = text[start:max_end]
        
        # 按优先级寻找边界
        for pattern, boundary_type in self.boundary_patterns:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                # 选择最接近最大长度的边界点
                best_match = matches[-1]
                split_point = start + best_match.end()
                return split_point
        
        # 如果没有找到合适的边界，在最大长度处强制分割
        return max_end
    
    def compare_strategies(self, text: str) -> Dict:
        """
        比较不同拆分策略的效果
        
        Args:
            text: 待拆分文本
            
        Returns:
            比较结果
        """
        results = {}
        
        # 测试三种策略
        strategies = [
            ("固定长度拆分", self.split_by_fixed_length),
            ("语义边界拆分", self.split_by_semantic_boundary),
            ("滑动窗口拆分", self.split_by_sliding_window)
        ]
        
        for strategy_name, strategy_func in strategies:
            chunks = strategy_func(text)
            
            # 计算统计信息
            lengths = [chunk.length for chunk in chunks]
            
            results[strategy_name] = {
                'chunk_count': len(chunks),
                'avg_length': sum(lengths) / len(lengths) if lengths else 0,
                'max_length': max(lengths) if lengths else 0,
                'min_length': min(lengths) if lengths else 0,
                'total_chars': sum(lengths),
                'chunks': chunks[:3]  # 只保留前3个chunk用于展示
            }
        
        return results


class ChunkQualityEvaluator:
    """Chunk质量评估器"""
    
    def __init__(self):
        # 质量评估权重
        self.weights = {
            'length_consistency': 0.3,    # 长度一致性
            'semantic_integrity': 0.4,    # 语义完整性
            'boundary_quality': 0.3       # 边界质量
        }
    
    def evaluate_chunks(self, chunks: List[ChunkInfo]) -> Dict:
        """
        评估chunk质量
        
        Args:
            chunks: chunk列表
            
        Returns:
            质量评估结果
        """
        if not chunks:
            return {'score': 0, 'details': {}}
        
        # 1. 长度一致性评估
        lengths = [chunk.length for chunk in chunks]
        length_std = self._calculate_std(lengths)
        length_score = max(0, 100 - length_std / 10)  # 标准差越小分数越高
        
        # 2. 语义完整性评估（简化版）
        semantic_score = self._evaluate_semantic_integrity(chunks)
        
        # 3. 边界质量评估
        boundary_score = self._evaluate_boundary_quality(chunks)
        
        # 综合评分
        total_score = (
            length_score * self.weights['length_consistency'] +
            semantic_score * self.weights['semantic_integrity'] +
            boundary_score * self.weights['boundary_quality']
        )
        
        return {
            'total_score': round(total_score, 2),
            'details': {
                'length_consistency': round(length_score, 2),
                'semantic_integrity': round(semantic_score, 2),
                'boundary_quality': round(boundary_score, 2)
            },
            'statistics': {
                'chunk_count': len(chunks),
                'avg_length': sum(lengths) / len(lengths),
                'length_std': length_std
            }
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _evaluate_semantic_integrity(self, chunks: List[ChunkInfo]) -> float:
        """评估语义完整性（简化版）"""
        score = 0
        for chunk in chunks:
            text = chunk.text.strip()
            
            # 检查是否以完整句子结尾
            if text.endswith(('。', '！', '？', '.', '!', '?')):
                score += 20
            elif text.endswith(('，', '；', ',', ';')):
                score += 10
            
            # 检查是否包含完整的语义单元
            if len(text.split('。')) > 1:  # 包含多个句子
                score += 15
            
            # 检查长度合理性
            if 200 <= len(text) <= 800:  # 合理的长度范围
                score += 10
        
        return min(100, score / len(chunks)) if chunks else 0
    
    def _evaluate_boundary_quality(self, chunks: List[ChunkInfo]) -> float:
        """评估边界质量"""
        if len(chunks) < 2:
            return 100
        
        good_boundaries = 0
        total_boundaries = len(chunks) - 1
        
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # 检查边界是否在合适的位置
            current_end = current_chunk.text.strip()
            next_start = next_chunk.text.strip()
            
            # 好的边界：当前chunk以句子结尾，下一个chunk以句子开头
            if (current_end.endswith(('。', '！', '？', '.', '!', '?')) and
                next_start and next_start[0].isupper()):
                good_boundaries += 1
            # 次好的边界：在标点符号处分割
            elif current_end.endswith(('，', '；', ',', ';')):
                good_boundaries += 0.5
        
        return (good_boundaries / total_boundaries) * 100 if total_boundaries > 0 else 100


def demo_chunking_strategies():
    """演示不同的拆分策略"""
    
    # 示例文本
    sample_text = """
    人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。

    自诞生以来，理论和技术日益成熟，应用领域也不断扩大。可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。人工智能可以对人的意识、思维的信息过程的模拟。

    人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。人工智能是一门极富挑战性的科学，从事这项工作的人必须懂得计算机知识，心理学和哲学。人工智能是包括十分广泛的科学，它由不同的领域组成。

    机器学习是人工智能的核心，是使计算机具有智能的根本途径。机器学习的应用已经遍及人工智能的各个分支，如专家系统、自动定理证明、自然语言理解、模式识别、计算机视觉等领域。

    深度学习是机器学习的一个分支，它基于人工神经网络的研究，特别是利用多层次的神经网络来进行学习和表示。深度学习的概念源于人工神经网络的研究，含多隐层的多层感知器就是一种深度学习结构。
    """
    
    print("=== 智能Chunk拆分策略演示 ===\n")
    
    # 创建拆分器
    chunker = SmartChunker(max_chunk_size=300, overlap_size=50)
    evaluator = ChunkQualityEvaluator()
    
    print(f"原始文本长度: {len(sample_text)} 字符")
    print(f"拆分参数: max_size={chunker.max_chunk_size}, overlap={chunker.overlap_size}\n")
    
    # 比较不同策略
    results = chunker.compare_strategies(sample_text)
    
    for strategy_name, result in results.items():
        print(f"=== {strategy_name} ===")
        print(f"Chunk数量: {result['chunk_count']}")
        print(f"平均长度: {result['avg_length']:.1f} 字符")
        print(f"长度范围: {result['min_length']} - {result['max_length']} 字符")
        
        # 质量评估
        quality = evaluator.evaluate_chunks(result['chunks'])
        print(f"质量评分: {quality['total_score']}/100")
        print(f"  - 长度一致性: {quality['details']['length_consistency']}")
        print(f"  - 语义完整性: {quality['details']['semantic_integrity']}")
        print(f"  - 边界质量: {quality['details']['boundary_quality']}")
        
        # 显示前2个chunk的内容
        print("\n前2个Chunk预览:")
        for i, chunk in enumerate(result['chunks'][:2]):
            print(f"  Chunk {i+1} ({chunk.length}字符): {chunk.text[:100]}...")
        
        print("\n" + "-" * 60 + "\n")


def main():
    """主演示函数"""
    try:
        demo_chunking_strategies()
        
        print("\n=== 自定义测试 ===")
        print("你可以修改 sample_text 变量来测试不同的文本内容")
        print("也可以调整 max_chunk_size 和 overlap_size 参数来观察效果变化")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")


if __name__ == "__main__":
    main()