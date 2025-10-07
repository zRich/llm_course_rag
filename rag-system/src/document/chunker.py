"""文本分块器"""

import re
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class TextChunker:
    """文本分块器"""
    
    def __init__(self, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 separators: Optional[List[str]] = None):
        """
        初始化文本分块器
        
        Args:
            chunk_size: 文本块大小（字符数）
            chunk_overlap: 文本块重叠大小（字符数）
            separators: 分割符列表，按优先级排序
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 默认分割符，按优先级排序
        if separators is None:
            self.separators = [
                "\n\n",  # 段落分割
                "\n",    # 行分割
                "。",     # 中文句号
                "！",     # 中文感叹号
                "？",     # 中文问号
                ".",     # 英文句号
                "!",     # 英文感叹号
                "?",     # 英文问号
                ";",     # 分号
                ",",     # 逗号
                " ",     # 空格
                "",      # 字符级分割
            ]
        else:
            self.separators = separators
    
    def chunk_text(self, text: str) -> List[str]:
        """
        将文本分割成块
        
        Args:
            text: 输入文本
            
        Returns:
            文本块列表
        """
        if not text or not text.strip():
            return []
        
        # 清理文本
        text = self._clean_text(text)
        
        # 如果文本长度小于chunk_size，直接返回
        if len(text) <= self.chunk_size:
            return [text]
        
        # 递归分割文本
        chunks = self._split_text_recursive(text, self.separators)
        
        # 过滤空块并去除首尾空白
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        logger.info(f"文本分块完成: {len(chunks)} 个块")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        清理文本
        
        Args:
            text: 输入文本
            
        Returns:
            清理后的文本
        """
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除首尾空白
        text = text.strip()
        return text
    
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        递归分割文本
        
        Args:
            text: 输入文本
            separators: 分割符列表
            
        Returns:
            文本块列表
        """
        final_chunks = []
        
        # 当前分割符
        separator = separators[0] if separators else ""
        new_separators = separators[1:] if len(separators) > 1 else []
        
        # 按当前分割符分割
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)  # 字符级分割
        
        # 处理分割后的文本
        current_chunk = ""
        
        for i, split in enumerate(splits):
            # 重新添加分割符（除了最后一个）
            if separator and i < len(splits) - 1:
                split += separator
            
            # 如果当前块加上新分割后的文本仍在大小限制内
            if len(current_chunk + split) <= self.chunk_size:
                current_chunk += split
            else:
                # 如果当前块不为空，添加到结果中
                if current_chunk:
                    final_chunks.append(current_chunk)
                
                # 如果单个分割就超过了chunk_size，需要进一步分割
                if len(split) > self.chunk_size:
                    if new_separators:
                        # 使用下一级分割符
                        sub_chunks = self._split_text_recursive(split, new_separators)
                        final_chunks.extend(sub_chunks)
                    else:
                        # 强制按字符分割
                        for j in range(0, len(split), self.chunk_size):
                            final_chunks.append(split[j:j + self.chunk_size])
                    current_chunk = ""
                else:
                    current_chunk = split
        
        # 添加最后一个块
        if current_chunk:
            final_chunks.append(current_chunk)
        
        # 处理重叠
        if self.chunk_overlap > 0 and len(final_chunks) > 1:
            final_chunks = self._add_overlap(final_chunks)
        
        return final_chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        为文本块添加重叠
        
        Args:
            chunks: 文本块列表
            
        Returns:
            添加重叠后的文本块列表
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]  # 第一个块保持不变
        
        for i in range(1, len(chunks)):
            current_chunk = chunks[i]
            previous_chunk = chunks[i - 1]
            
            # 从前一个块的末尾取重叠部分
            if len(previous_chunk) >= self.chunk_overlap:
                overlap = previous_chunk[-self.chunk_overlap:]
                current_chunk = overlap + current_chunk
            
            overlapped_chunks.append(current_chunk)
        
        return overlapped_chunks
    
    def get_chunk_info(self, chunks: List[str]) -> dict:
        """
        获取分块信息统计
        
        Args:
            chunks: 文本块列表
            
        Returns:
            分块信息字典
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'total_length': 0,
                'avg_chunk_length': 0,
                'min_chunk_length': 0,
                'max_chunk_length': 0
            }
        
        chunk_lengths = [len(chunk) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'total_length': sum(chunk_lengths),
            'avg_chunk_length': sum(chunk_lengths) / len(chunks),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths)
        }