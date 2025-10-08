"""
文本分块服务
实现文档的智能分块功能，支持多种分块策略
"""

import logging
import re
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.config.settings import settings

logger = logging.getLogger(__name__)


class SplitStrategy(Enum):
    """分块策略枚举"""
    FIXED_SIZE = "fixed_size"  # 固定大小分块
    SENTENCE = "sentence"      # 句子边界分块
    PARAGRAPH = "paragraph"    # 段落边界分块
    SEMANTIC = "semantic"      # 语义分块（基于内容相似性）
    # TODO(lab03-lesson11): 在本实验中，要求同学实现语义分块策略。
    # 需求要点：
    # - 结合embedding相似度，将相近语义的句子/段落聚为一个chunk；
    # - 支持阈值控制与最大chunk长度控制；
    # - 保留最小分块大小与句子边界优先的策略；
    # 验收提示：提供若干长文示例，语义相近的句子应聚类到同一分块，
    # 且平均分块大小与token数量在可控范围内。


@dataclass
class TextChunk:
    """文本分块数据类"""
    content: str
    chunk_index: int
    start_position: int
    end_position: int
    char_count: int
    word_count: int
    estimated_tokens: int
    content_hash: str
    metadata: Dict


class TextSplitter:
    """文本分块器"""
    
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.min_chunk_size = 50  # 最小分块大小
        
    def estimate_tokens(self, text: str) -> int:
        """
        估算文本的token数量
        
        Args:
            text: 输入文本
            
        Returns:
            估算的token数量
        """
        # 简单估算：中文1字符≈1token，英文1单词≈1.3token，平均4字符≈1token
        # TODO(lab03-lesson11): 使用真实分词器（如`tiktoken`或`transformers`的tokenizer）
        # 替换此估算逻辑，并在`settings`中增加可切换的估算/真实模式。
        if not text:
            return 0
        
        # 统计中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        # 统计英文单词
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
        # 其他字符
        other_chars = len(text) - chinese_chars - sum(len(word) for word in re.findall(r'\b[a-zA-Z]+\b', text))
        
        # 估算token数
        estimated_tokens = chinese_chars + int(english_words * 1.3) + other_chars // 4
        return max(estimated_tokens, len(text) // 4)  # 保底估算
    
    def calculate_chunk_hash(self, content: str) -> str:
        """
        计算分块内容的哈希值
        
        Args:
            content: 分块内容
            
        Returns:
            SHA256哈希值
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def clean_chunk_content(self, content: str) -> str:
        """
        清理分块内容
        
        Args:
            content: 原始内容
            
        Returns:
            清理后的内容
        """
        if not content:
            return ""
        
        # 移除多余的空白字符
        content = re.sub(r'\s+', ' ', content)
        
        # 移除行首行尾空白
        content = content.strip()
        
        return content
    
    def split_by_fixed_size(self, text: str, metadata: Dict = None) -> List[TextChunk]:
        """
        按固定大小分块
        
        Args:
            text: 输入文本
            metadata: 元数据
            
        Returns:
            分块列表
        """
        if not text:
            return []
        
        chunks = []
        chunk_index = 0
        start_pos = 0
        
        while start_pos < len(text):
            # 计算结束位置
            end_pos = min(start_pos + self.chunk_size, len(text))
            
            # 如果不是最后一个分块，尝试在句子边界分割
            if end_pos < len(text):
                # 寻找最近的句子结束符
                sentence_end = -1
                for i in range(end_pos - 1, start_pos + self.min_chunk_size - 1, -1):
                    if text[i] in '。！？.!?':
                        sentence_end = i + 1
                        break
                
                # 如果找到合适的句子边界，使用它
                if sentence_end > start_pos + self.min_chunk_size:
                    end_pos = sentence_end
            
            # 提取分块内容
            chunk_content = text[start_pos:end_pos]
            chunk_content = self.clean_chunk_content(chunk_content)
            # TODO(lab03-lesson11): 进一步优化固定大小策略：
            # - 支持窗口滑动与重叠的动态调整（基于标点/换行/标题）；
            # - 为不同语言设置不同的边界优先级（中文标点/英文句号）；
            
            # 跳过太短的分块
            if len(chunk_content) < self.min_chunk_size and chunk_index > 0:
                # 将短分块合并到前一个分块
                if chunks:
                    prev_chunk = chunks[-1]
                    merged_content = prev_chunk.content + " " + chunk_content
                    merged_content = self.clean_chunk_content(merged_content)
                    
                    # 更新前一个分块
                    chunks[-1] = TextChunk(
                        content=merged_content,
                        chunk_index=prev_chunk.chunk_index,
                        start_position=prev_chunk.start_position,
                        end_position=end_pos,
                        char_count=len(merged_content),
                        word_count=len(merged_content.split()),
                        estimated_tokens=self.estimate_tokens(merged_content),
                        content_hash=self.calculate_chunk_hash(merged_content),
                        metadata={**(metadata or {}), "merge_count": prev_chunk.metadata.get("merge_count", 0) + 1}
                    )
                break
            
            if chunk_content:
                chunk = TextChunk(
                    content=chunk_content,
                    chunk_index=chunk_index,
                    start_position=start_pos,
                    end_position=end_pos,
                    char_count=len(chunk_content),
                    word_count=len(chunk_content.split()),
                    estimated_tokens=self.estimate_tokens(chunk_content),
                    content_hash=self.calculate_chunk_hash(chunk_content),
                    metadata=metadata or {}
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # 计算下一个分块的起始位置（考虑重叠）
            start_pos = max(end_pos - self.chunk_overlap, start_pos + 1)
            
            # 避免无限循环
            if start_pos >= end_pos:
                break
        
        return chunks
    
    def split_by_sentences(self, text: str, metadata: Dict = None) -> List[TextChunk]:
        """
        按句子边界分块
        
        Args:
            text: 输入文本
            metadata: 元数据
            
        Returns:
            分块列表
        """
        if not text:
            return []
        
        # 分割句子
        sentence_pattern = r'[。！？.!?]+\s*'
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        chunk_index = 0
        current_chunk = ""
        start_pos = 0
        
        for sentence in sentences:
            # 检查添加当前句子是否会超过大小限制
            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if len(potential_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = potential_chunk
            else:
                # 保存当前分块
                if current_chunk:
                    chunk_content = self.clean_chunk_content(current_chunk)
                    if len(chunk_content) >= self.min_chunk_size:
                        chunk = TextChunk(
                            content=chunk_content,
                            chunk_index=chunk_index,
                            start_position=start_pos,
                            end_position=start_pos + len(chunk_content),
                            char_count=len(chunk_content),
                            word_count=len(chunk_content.split()),
                            estimated_tokens=self.estimate_tokens(chunk_content),
                            content_hash=self.calculate_chunk_hash(chunk_content),
                            metadata=metadata or {}
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                        start_pos += len(chunk_content)
                # TODO(lab03-lesson11): 在句子策略中加入“智能合并”与“标题锚点”处理：
                # - 对过短句子进行合并；
                # - 遇到标题/小节开头时，优先切分形成新块；
                
                # 开始新分块
                current_chunk = sentence
        
        # 处理最后一个分块
        if current_chunk:
            chunk_content = self.clean_chunk_content(current_chunk)
            if len(chunk_content) >= self.min_chunk_size:
                chunk = TextChunk(
                    content=chunk_content,
                    chunk_index=chunk_index,
                    start_position=start_pos,
                    end_position=start_pos + len(chunk_content),
                    char_count=len(chunk_content),
                    word_count=len(chunk_content.split()),
                    estimated_tokens=self.estimate_tokens(chunk_content),
                    content_hash=self.calculate_chunk_hash(chunk_content),
                    metadata=metadata or {}
                )
                chunks.append(chunk)
        
        return chunks
    
    def split_by_paragraphs(self, text: str, metadata: Dict = None) -> List[TextChunk]:
        """
        按段落边界分块
        
        Args:
            text: 输入文本
            metadata: 元数据
            
        Returns:
            分块列表
        """
        if not text:
            return []
        
        # 分割段落
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        chunk_index = 0
        current_chunk = ""
        start_pos = 0
        
        for paragraph in paragraphs:
            # 检查添加当前段落是否会超过大小限制
            potential_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
            
            if len(potential_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = potential_chunk
            else:
                # 保存当前分块
                if current_chunk:
                    chunk_content = self.clean_chunk_content(current_chunk)
                    if len(chunk_content) >= self.min_chunk_size:
                        chunk = TextChunk(
                            content=chunk_content,
                            chunk_index=chunk_index,
                            start_position=start_pos,
                            end_position=start_pos + len(chunk_content),
                            char_count=len(chunk_content),
                            word_count=len(chunk_content.split()),
                            estimated_tokens=self.estimate_tokens(chunk_content),
                            content_hash=self.calculate_chunk_hash(chunk_content),
                            metadata=metadata or {}
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                        start_pos += len(chunk_content)
                
                # 开始新分块
                current_chunk = paragraph
        
        # 处理最后一个分块
        if current_chunk:
            chunk_content = self.clean_chunk_content(current_chunk)
            if len(chunk_content) >= self.min_chunk_size:
                chunk = TextChunk(
                    content=chunk_content,
                    chunk_index=chunk_index,
                    start_position=start_pos,
                    end_position=start_pos + len(chunk_content),
                    char_count=len(chunk_content),
                    word_count=len(chunk_content.split()),
                    estimated_tokens=self.estimate_tokens(chunk_content),
                    content_hash=self.calculate_chunk_hash(chunk_content),
                    metadata=metadata or {}
                )
                chunks.append(chunk)
        
        return chunks
    
    def split_text(self, text: str, strategy: SplitStrategy = SplitStrategy.FIXED_SIZE, 
                   metadata: Dict = None) -> List[TextChunk]:
        """
        分割文本
        
        Args:
            text: 输入文本
            strategy: 分块策略
            metadata: 元数据
            
        Returns:
            分块列表
        """
        if not text or not text.strip():
            return []
        
        logger.info(f"开始文本分块，策略: {strategy.value}, 文本长度: {len(text)}")
        
        try:
            if strategy == SplitStrategy.FIXED_SIZE:
                chunks = self.split_by_fixed_size(text, metadata)
            elif strategy == SplitStrategy.SENTENCE:
                chunks = self.split_by_sentences(text, metadata)
            elif strategy == SplitStrategy.PARAGRAPH:
                chunks = self.split_by_paragraphs(text, metadata)
            elif strategy == SplitStrategy.SEMANTIC:
                chunks = self.split_by_semantic(text, metadata)
            else:
                # 默认使用固定大小分块
                chunks = self.split_by_fixed_size(text, metadata)
            
            logger.info(f"文本分块完成，生成 {len(chunks)} 个分块")
            
            # 记录分块统计信息
            if chunks:
                total_chars = sum(chunk.char_count for chunk in chunks)
                total_tokens = sum(chunk.estimated_tokens for chunk in chunks)
                avg_chunk_size = total_chars / len(chunks)
                
                logger.info(f"分块统计 - 总字符数: {total_chars}, "
                           f"总token数: {total_tokens}, "
                           f"平均分块大小: {avg_chunk_size:.1f}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"文本分块失败: {e}")
            raise
    
    def get_chunk_context(self, chunks: List[TextChunk], target_index: int, 
                         context_size: int = 1) -> Dict:
        """
        获取分块的上下文信息
        
        Args:
            chunks: 分块列表
            target_index: 目标分块索引
            context_size: 上下文大小（前后各几个分块）
            
        Returns:
            包含上下文的字典
        """
        if not chunks or target_index < 0 or target_index >= len(chunks):
            return {}
        
        target_chunk = chunks[target_index]
        
        # 获取前后文分块
        start_idx = max(0, target_index - context_size)
        end_idx = min(len(chunks), target_index + context_size + 1)
        
        context_chunks = chunks[start_idx:end_idx]
        
        return {
            "target_chunk": target_chunk,
            "context_chunks": context_chunks,
            "previous_chunks": chunks[start_idx:target_index],
            "next_chunks": chunks[target_index + 1:end_idx],
            "context_text": " ".join([chunk.content for chunk in context_chunks])
        }
    def split_by_semantic(self, text: str, metadata: Dict = None) -> List[TextChunk]:
        """
        语义分块（占位实现）

        TODO(lab03-lesson11): 请实现以下逻辑：
        - 使用Embedding对句子/段落进行编码，计算相邻单位的相似度；
        - 当相似度高于阈值时聚合为同一chunk，低于阈值时切分；
        - 控制最大chunk长度与最小分块大小；
        - 支持将相似度阈值、最大长度、最小长度从`settings`中读取；
        验收：提供示例文本与参数，输出的分块应与语义结构一致。
        """
        # 为保持学生版可运行，当前返回句子分块作为占位行为
        return self.split_by_sentences(text, metadata)