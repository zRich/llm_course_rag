"""切分策略接口定义

定义插件化切分策略的统一接口，支持策略的动态注册和管理。
这是第19节课插件化架构的核心组件。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import time
import logging

from .chunker import DocumentChunk, ChunkMetadata, ChunkingConfig

logger = logging.getLogger(__name__)

@dataclass
class StrategyMetrics:
    """策略执行指标"""
    execution_time: float = 0.0  # 执行时间（秒）
    chunk_count: int = 0  # 生成的块数量
    avg_chunk_size: float = 0.0  # 平均块大小
    min_chunk_size: int = 0  # 最小块大小
    max_chunk_size: int = 0  # 最大块大小
    overlap_ratio: float = 0.0  # 重叠率
    memory_usage: float = 0.0  # 内存使用量（MB）
    quality_score: float = 0.0  # 质量评分

class ChunkingStrategy(ABC):
    """切分策略抽象基类
    
    所有切分策略插件都必须继承此类并实现相应方法。
    提供统一的策略接口和基础功能。
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None, **kwargs):
        """初始化策略
        
        Args:
            config: 切分配置
            **kwargs: 策略特定参数
        """
        self.config = config or ChunkingConfig()
        self.strategy_params = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics = StrategyMetrics()
        self._execution_count = 0
        
    @abstractmethod
    def get_strategy_name(self) -> str:
        """获取策略名称
        
        Returns:
            str: 策略名称
        """
        pass
    
    @abstractmethod
    def get_strategy_description(self) -> str:
        """获取策略描述
        
        Returns:
            str: 策略描述
        """
        pass
    
    @abstractmethod
    def chunk_text(self, text: str, source_file: str = "") -> List[DocumentChunk]:
        """执行文本切分
        
        Args:
            text: 要切分的文本
            source_file: 源文件路径
            
        Returns:
            List[DocumentChunk]: 切分后的文档块列表
        """
        pass
    
    def chunk_with_metrics(self, text: str, source_file: str = "") -> tuple[List[DocumentChunk], StrategyMetrics]:
        """执行文本切分并收集指标
        
        Args:
            text: 要切分的文本
            source_file: 源文件路径
            
        Returns:
            tuple: (切分结果, 执行指标)
        """
        import psutil
        import os
        
        # 记录开始状态
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # 执行切分
            chunks = self.chunk_text(text, source_file)
            
            # 计算指标
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            self.metrics.execution_time = end_time - start_time
            self.metrics.chunk_count = len(chunks)
            self.metrics.memory_usage = end_memory - start_memory
            
            if chunks:
                chunk_sizes = [len(chunk.content) for chunk in chunks]
                self.metrics.avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
                self.metrics.min_chunk_size = min(chunk_sizes)
                self.metrics.max_chunk_size = max(chunk_sizes)
                
                # 计算重叠率（简化计算）
                self.metrics.overlap_ratio = self._calculate_overlap_ratio(chunks)
                
                # 计算质量评分
                self.metrics.quality_score = self._calculate_quality_score(chunks, text)
            
            self._execution_count += 1
            
            return chunks, self.metrics
            
        except Exception as e:
            self.logger.error(f"策略 {self.get_strategy_name()} 执行失败: {e}")
            raise
    
    def _calculate_overlap_ratio(self, chunks: List[DocumentChunk]) -> float:
        """计算重叠率
        
        Args:
            chunks: 文档块列表
            
        Returns:
            float: 重叠率 (0-1)
        """
        if len(chunks) < 2:
            return 0.0
        
        total_overlap = 0
        total_adjacent_pairs = 0
        
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # 简化的重叠计算：基于位置信息
            if (hasattr(current_chunk.metadata, 'end_position') and 
                hasattr(next_chunk.metadata, 'start_position')):
                
                current_end = current_chunk.metadata.end_position
                next_start = next_chunk.metadata.start_position
                
                if current_end > next_start:
                    overlap = current_end - next_start
                    total_overlap += overlap
                
                total_adjacent_pairs += 1
        
        if total_adjacent_pairs == 0:
            return 0.0
        
        # 返回平均重叠率
        avg_chunk_size = self.metrics.avg_chunk_size or 1
        return min(total_overlap / (total_adjacent_pairs * avg_chunk_size), 1.0)
    
    def _calculate_quality_score(self, chunks: List[DocumentChunk], original_text: str) -> float:
        """计算切分质量评分
        
        Args:
            chunks: 文档块列表
            original_text: 原始文本
            
        Returns:
            float: 质量评分 (0-1)
        """
        if not chunks:
            return 0.0
        
        score = 0.0
        factors = 0
        
        # 因子1: 块大小一致性 (0-0.3)
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        if len(chunk_sizes) > 1:
            size_variance = sum((size - self.metrics.avg_chunk_size) ** 2 for size in chunk_sizes) / len(chunk_sizes)
            size_consistency = max(0, 1 - (size_variance / (self.metrics.avg_chunk_size ** 2)))
            score += size_consistency * 0.3
            factors += 0.3
        
        # 因子2: 内容完整性 (0-0.4)
        total_chunk_chars = sum(len(chunk.content) for chunk in chunks)
        content_coverage = min(total_chunk_chars / len(original_text), 1.0) if original_text else 0
        score += content_coverage * 0.4
        factors += 0.4
        
        # 因子3: 块数量合理性 (0-0.3)
        expected_chunks = max(1, len(original_text) // self.config.chunk_size)
        chunk_count_ratio = min(len(chunks) / expected_chunks, expected_chunks / len(chunks)) if expected_chunks > 0 else 0
        score += chunk_count_ratio * 0.3
        factors += 0.3
        
        return score / factors if factors > 0 else 0.0
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """获取策略信息
        
        Returns:
            Dict[str, Any]: 策略信息
        """
        return {
            'name': self.get_strategy_name(),
            'description': self.get_strategy_description(),
            'config': {
                'chunk_size': self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap,
                'min_chunk_size': self.config.min_chunk_size,
                'max_chunk_size': self.config.max_chunk_size,
                'preserve_sentences': self.config.preserve_sentences,
                'preserve_paragraphs': self.config.preserve_paragraphs
            },
            'strategy_params': self.strategy_params,
            'execution_count': self._execution_count,
            'last_metrics': {
                'execution_time': self.metrics.execution_time,
                'chunk_count': self.metrics.chunk_count,
                'avg_chunk_size': self.metrics.avg_chunk_size,
                'quality_score': self.metrics.quality_score
            }
        }
    
    def validate_config(self) -> bool:
        """验证策略配置
        
        Returns:
            bool: 配置是否有效
        """
        try:
            if self.config.chunk_size <= 0:
                raise ValueError("chunk_size 必须大于 0")
            
            if self.config.chunk_overlap < 0:
                raise ValueError("chunk_overlap 不能小于 0")
            
            if self.config.chunk_overlap >= self.config.chunk_size:
                raise ValueError("chunk_overlap 不能大于等于 chunk_size")
            
            return True
            
        except Exception as e:
            self.logger.error(f"策略配置验证失败: {e}")
            return False
    
    def reset_metrics(self) -> None:
        """重置执行指标"""
        self.metrics = StrategyMetrics()
        self._execution_count = 0
    
    def get_recommended_config(self, text_length: int) -> ChunkingConfig:
        """根据文本长度推荐配置
        
        Args:
            text_length: 文本长度
            
        Returns:
            ChunkingConfig: 推荐的配置
        """
        # 基础推荐逻辑，子类可以重写
        if text_length < 1000:
            chunk_size = 200
            overlap = 50
        elif text_length < 5000:
            chunk_size = 500
            overlap = 100
        elif text_length < 20000:
            chunk_size = 1000
            overlap = 200
        else:
            chunk_size = 1500
            overlap = 300
        
        return ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            min_chunk_size=max(50, chunk_size // 10),
            max_chunk_size=chunk_size * 2
        )

class StrategyError(Exception):
    """策略执行异常"""
    pass

class StrategyConfigError(Exception):
    """策略配置异常"""
    pass