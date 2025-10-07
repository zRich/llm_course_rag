#!/usr/bin/env python3
"""
智能段落切分策略模板

这是第19节课的核心实现文件，学生需要基于此模板完成智能段落切分策略。
本文件提供了完整的实现框架和关键方法的示例代码。

使用方法：
1. 将此文件复制到 src/chunking/smart_paragraph_chunker.py
2. 根据注释提示完成TODO部分的实现
3. 在 src/chunking/__init__.py 中注册策略
4. 运行测试验证功能
"""

import re
from typing import List, Optional, Tuple
import logging

# 导入基础类（需要确保路径正确）
try:
    from .strategy_interface import ChunkingStrategy, StrategyMetrics
    from .chunker import DocumentChunk, ChunkMetadata, ChunkingConfig
except ImportError:
    # 如果在lesson19目录下直接运行，使用相对导入
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.chunking.strategy_interface import ChunkingStrategy, StrategyMetrics
    from src.chunking.chunker import DocumentChunk, ChunkMetadata, ChunkingConfig

logger = logging.getLogger(__name__)

class SmartParagraphStrategy(ChunkingStrategy):
    """
    智能段落切分策略
    
    特点：
    1. 识别段落边界（双换行、列表项等）
    2. 智能合并短段落
    3. 分割过长段落
    4. 保持语义完整性
    
    适用场景：
    - 技术文档
    - 长篇文章
    - 结构化内容
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None, **kwargs):
        """
        初始化智能段落策略
        
        Args:
            config: 基础切分配置
            **kwargs: 策略特定参数
                - min_paragraph_length: 最小段落长度（默认50）
                - max_paragraph_length: 最大段落长度（默认为chunk_size）
                - merge_threshold: 合并阈值（默认0.3）
                - split_sentences: 是否按句子分割长段落（默认True）
        """
        super().__init__(config, **kwargs)
        
        # 策略特定参数
        self.min_paragraph_length = kwargs.get('min_paragraph_length', 50)
        self.max_paragraph_length = kwargs.get('max_paragraph_length', self.config.chunk_size)
        self.merge_threshold = kwargs.get('merge_threshold', 0.3)
        self.split_sentences = kwargs.get('split_sentences', True)
        
        # 段落识别模式
        self.paragraph_patterns = [
            r'\n\s*\n',              # 双换行
            r'\n\s*[-*•]\s+',        # 无序列表项
            r'\n\s*\d+[.).]\s+',     # 数字列表项
            r'\n\s*[a-zA-Z][.).]\s+', # 字母列表项
            r'\n\s*#{1,6}\s+',       # Markdown标题
        ]
        
        self.logger.info(f"智能段落策略初始化完成，参数: min_len={self.min_paragraph_length}, "
                        f"max_len={self.max_paragraph_length}, threshold={self.merge_threshold}")
    
    def get_strategy_name(self) -> str:
        """获取策略名称"""
        return "smart_paragraph"
    
    def get_strategy_description(self) -> str:
        """获取策略描述"""
        return "智能段落切分策略，结合段落结构和长度控制，适用于技术文档和长文本"
    
    def chunk_text(self, text: str, source_file: str = "") -> List[DocumentChunk]:
        """
        执行智能段落切分
        
        Args:
            text: 要切分的文本
            source_file: 源文件路径
            
        Returns:
            List[DocumentChunk]: 切分后的文档块列表
        """
        try:
            if not text.strip():
                return []
            
            # 步骤1: 识别段落边界
            paragraphs = self._identify_paragraphs(text)
            self.logger.debug(f"识别到 {len(paragraphs)} 个初始段落")
            
            # 步骤2: 智能合并短段落
            merged_paragraphs = self._merge_short_paragraphs(paragraphs)
            self.logger.debug(f"合并后剩余 {len(merged_paragraphs)} 个段落")
            
            # 步骤3: 分割过长段落
            final_paragraphs = self._split_long_paragraphs(merged_paragraphs)
            self.logger.debug(f"分割后最终 {len(final_paragraphs)} 个段落")
            
            # 步骤4: 创建文档块
            chunks = self._create_chunks_from_paragraphs(final_paragraphs, source_file)
            
            self.logger.info(f"智能段落切分完成: {len(paragraphs)} -> {len(merged_paragraphs)} -> "
                           f"{len(final_paragraphs)} -> {len(chunks)} 块")
            return chunks
            
        except Exception as e:
            self.logger.error(f"智能段落切分失败: {e}")
            raise
    
    def _identify_paragraphs(self, text: str) -> List[Tuple[str, int, int]]:
        """
        识别段落边界
        
        Args:
            text: 输入文本
            
        Returns:
            List[Tuple[str, int, int]]: (段落内容, 开始位置, 结束位置)
        """
        paragraphs = []
        
        # TODO: 学生实现 - 合并所有段落识别模式
        # 提示：使用 '|'.join() 合并正则表达式模式
        combined_pattern = '|'.join(f'({pattern})' for pattern in self.paragraph_patterns)
        
        # TODO: 学生实现 - 使用正则表达式分割文本
        # 提示：使用 re.split() 方法，保留分隔符
        parts = re.split(combined_pattern, text)
        
        current_pos = 0
        current_paragraph = ""
        
        for part in parts:
            if not part:
                continue
            
            # 检查是否是分隔符
            is_separator = any(re.match(pattern, part) for pattern in self.paragraph_patterns)
            
            if is_separator and current_paragraph.strip():
                # 结束当前段落
                start_pos = current_pos - len(current_paragraph)
                paragraphs.append((current_paragraph.strip(), start_pos, current_pos))
                current_paragraph = ""
            else:
                current_paragraph += part
            
            current_pos += len(part)
        
        # 添加最后一个段落
        if current_paragraph.strip():
            start_pos = current_pos - len(current_paragraph)
            paragraphs.append((current_paragraph.strip(), start_pos, current_pos))
        
        return paragraphs
    
    def _merge_short_paragraphs(self, paragraphs: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
        """
        智能合并过短的段落
        
        Args:
            paragraphs: 原始段落列表
            
        Returns:
            List[Tuple[str, int, int]]: 合并后的段落列表
        """
        if not paragraphs:
            return []
        
        merged = []
        current_group = [paragraphs[0]]
        current_length = len(paragraphs[0][0])
        
        for i in range(1, len(paragraphs)):
            para_content, para_start, para_end = paragraphs[i]
            para_length = len(para_content)
            
            # TODO: 学生实现 - 判断是否需要合并
            # 合并条件：
            # 1. 当前组长度小于最小段落长度
            # 2. 新段落长度小于最小段落长度
            # 3. 合并后长度在合理范围内
            should_merge = (
                current_length < self.min_paragraph_length or
                para_length < self.min_paragraph_length or
                (current_length + para_length) < self.max_paragraph_length * self.merge_threshold
            )
            
            if should_merge and (current_length + para_length) <= self.max_paragraph_length:
                # 合并到当前组
                current_group.append((para_content, para_start, para_end))
                current_length += para_length
            else:
                # 完成当前组，开始新组
                merged.append(self._merge_paragraph_group(current_group))
                current_group = [(para_content, para_start, para_end)]
                current_length = para_length
        
        # 添加最后一组
        if current_group:
            merged.append(self._merge_paragraph_group(current_group))
        
        return merged
    
    def _merge_paragraph_group(self, group: List[Tuple[str, int, int]]) -> Tuple[str, int, int]:
        """
        合并段落组
        
        Args:
            group: 要合并的段落组
            
        Returns:
            Tuple[str, int, int]: 合并后的段落
        """
        if len(group) == 1:
            return group[0]
        
        # TODO: 学生实现 - 合并段落内容
        # 提示：使用适当的分隔符连接段落
        contents = [para[0] for para in group]
        start_pos = group[0][1]
        end_pos = group[-1][2]
        
        # 使用双换行连接段落，保持结构
        merged_content = '\n\n'.join(contents)
        
        return (merged_content, start_pos, end_pos)
    
    def _split_long_paragraphs(self, paragraphs: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
        """
        分割过长的段落
        
        Args:
            paragraphs: 输入段落列表
            
        Returns:
            List[Tuple[str, int, int]]: 分割后的段落列表
        """
        result = []
        
        for content, start_pos, end_pos in paragraphs:
            if len(content) <= self.max_paragraph_length:
                result.append((content, start_pos, end_pos))
            else:
                # 分割长段落
                if self.split_sentences:
                    split_parts = self._split_paragraph_by_sentences(content, start_pos)
                else:
                    split_parts = self._split_by_length(content, start_pos)
                result.extend(split_parts)
        
        return result
    
    def _split_paragraph_by_sentences(self, content: str, start_pos: int) -> List[Tuple[str, int, int]]:
        """
        按句子分割长段落
        
        Args:
            content: 段落内容
            start_pos: 段落起始位置
            
        Returns:
            List[Tuple[str, int, int]]: 分割后的部分
        """
        # TODO: 学生实现 - 句子分割逻辑
        # 提示：使用正则表达式识别句子边界
        sentence_pattern = r'[.!?。！？]\s+'
        sentences = re.split(f'({sentence_pattern})', content)
        
        parts = []
        current_part = ""
        current_start = start_pos
        
        for sentence in sentences:
            if not sentence:
                continue
            
            if len(current_part + sentence) <= self.max_paragraph_length:
                current_part += sentence
            else:
                if current_part.strip():
                    parts.append((current_part.strip(), current_start, current_start + len(current_part)))
                    current_start += len(current_part)
                current_part = sentence
        
        # 添加最后一部分
        if current_part.strip():
            parts.append((current_part.strip(), current_start, current_start + len(current_part)))
        
        return parts if parts else [(content, start_pos, start_pos + len(content))]
    
    def _split_by_length(self, content: str, start_pos: int) -> List[Tuple[str, int, int]]:
        """
        按长度简单分割
        
        Args:
            content: 内容
            start_pos: 起始位置
            
        Returns:
            List[Tuple[str, int, int]]: 分割后的部分
        """
        parts = []
        current_pos = 0
        
        while current_pos < len(content):
            end_pos = min(current_pos + self.max_paragraph_length, len(content))
            part_content = content[current_pos:end_pos]
            
            parts.append((
                part_content,
                start_pos + current_pos,
                start_pos + end_pos
            ))
            
            current_pos = end_pos
        
        return parts
    
    def _create_chunks_from_paragraphs(self, paragraphs: List[Tuple[str, int, int]], 
                                     source_file: str) -> List[DocumentChunk]:
        """
        从段落创建文档块
        
        Args:
            paragraphs: 段落列表
            source_file: 源文件路径
            
        Returns:
            List[DocumentChunk]: 文档块列表
        """
        chunks = []
        
        for i, (content, start_pos, end_pos) in enumerate(paragraphs):
            # TODO: 学生实现 - 创建文档块元数据
            metadata = ChunkMetadata(
                chunk_id=f"smart_para_{i}",
                source_file=source_file,
                chunk_index=i,
                start_position=start_pos,
                end_position=end_pos,
                chunk_type="smart_paragraph",
                language=self.config.language,
                metadata={
                    'paragraph_length': len(content),
                    'strategy': self.get_strategy_name(),
                    'min_length': self.min_paragraph_length,
                    'max_length': self.max_paragraph_length,
                    'merge_threshold': self.merge_threshold
                }
            )
            
            chunk = DocumentChunk(
                content=content,
                metadata=metadata
            )
            
            chunks.append(chunk)
        
        return chunks

# 测试代码（仅在直接运行时执行）
if __name__ == "__main__":
    # 简单测试
    test_text = """
    # 人工智能简介
    
    人工智能（AI）是计算机科学的一个分支。
    
    ## 发展历史
    
    • 1950年代：图灵测试提出
    • 1956年：达特茅斯会议
    • 1980年代：专家系统兴起
    
    现在AI技术发展迅速，应用广泛。
    """
    
    config = ChunkingConfig(chunk_size=200, language="zh")
    strategy = SmartParagraphStrategy(config, min_paragraph_length=30)
    
    chunks = strategy.chunk_text(test_text, "test.md")
    
    print(f"切分结果: {len(chunks)} 个块")
    for i, chunk in enumerate(chunks):
        print(f"\n块 {i+1}: {len(chunk.content)} 字符")
        print(chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content)
        print(f"元数据: {chunk.metadata.metadata}")