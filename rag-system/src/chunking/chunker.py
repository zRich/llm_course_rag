from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """文档块元数据"""
    chunk_id: str = ""
    source_file: str = ""
    chunk_index: int = 0
    start_position: int = 0
    end_position: int = 0
    chunk_type: str = "text"
    language: str = "unknown"
    word_count: int = 0
    char_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    parent_chunk_id: Optional[str] = None
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    confidence_score: float = 1.0
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentChunk:
    """文档块数据结构"""
    content: str
    metadata: ChunkMetadata
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.metadata.chunk_id:
            self.metadata.chunk_id = self._generate_chunk_id()
        
        if not self.metadata.word_count:
            self.metadata.word_count = len(self.content.split())
        
        if not self.metadata.char_count:
            self.metadata.char_count = len(self.content)
    
    def _generate_chunk_id(self) -> str:
        """生成块ID"""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        timestamp = int(datetime.now().timestamp() * 1000)
        return f"chunk_{timestamp}_{content_hash}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'content': self.content,
            'metadata': {
                'chunk_id': self.metadata.chunk_id,
                'source_file': self.metadata.source_file,
                'chunk_index': self.metadata.chunk_index,
                'start_position': self.metadata.start_position,
                'end_position': self.metadata.end_position,
                'chunk_type': self.metadata.chunk_type,
                'language': self.metadata.language,
                'word_count': self.metadata.word_count,
                'char_count': self.metadata.char_count,
                'created_at': self.metadata.created_at.isoformat(),
                'parent_chunk_id': self.metadata.parent_chunk_id,
                'section_title': self.metadata.section_title,
                'page_number': self.metadata.page_number,
                'confidence_score': self.metadata.confidence_score,
                'tags': self.metadata.tags,
                'custom_metadata': self.metadata.custom_metadata
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """从字典创建实例"""
        metadata_data = data['metadata']
        metadata = ChunkMetadata(
            chunk_id=metadata_data.get('chunk_id', ''),
            source_file=metadata_data.get('source_file', ''),
            chunk_index=metadata_data.get('chunk_index', 0),
            start_position=metadata_data.get('start_position', 0),
            end_position=metadata_data.get('end_position', 0),
            chunk_type=metadata_data.get('chunk_type', 'text'),
            language=metadata_data.get('language', 'unknown'),
            word_count=metadata_data.get('word_count', 0),
            char_count=metadata_data.get('char_count', 0),
            created_at=datetime.fromisoformat(metadata_data.get('created_at', datetime.now().isoformat())),
            parent_chunk_id=metadata_data.get('parent_chunk_id'),
            section_title=metadata_data.get('section_title'),
            page_number=metadata_data.get('page_number'),
            confidence_score=metadata_data.get('confidence_score', 1.0),
            tags=metadata_data.get('tags', []),
            custom_metadata=metadata_data.get('custom_metadata', {})
        )
        
        return cls(
            content=data['content'],
            metadata=metadata
        )

@dataclass
class ChunkingConfig:
    """分块配置"""
    chunk_size: int = 1000  # 块大小（字符数）
    chunk_overlap: int = 200  # 块重叠（字符数）
    min_chunk_size: int = 100  # 最小块大小
    max_chunk_size: int = 2000  # 最大块大小
    preserve_sentences: bool = True  # 保持句子完整性
    preserve_paragraphs: bool = False  # 保持段落完整性
    language: str = "auto"  # 语言设置
    custom_separators: List[str] = field(default_factory=list)  # 自定义分隔符
    metadata_extraction: bool = True  # 是否提取元数据
    filter_empty_chunks: bool = True  # 过滤空块
    normalize_whitespace: bool = True  # 标准化空白字符
    custom_config: Dict[str, Any] = field(default_factory=dict)  # 自定义配置

class DocumentChunker(ABC):
    """文档分块器抽象基类
    
    定义了文档分块的通用接口和基础功能
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def chunk_text(self, text: str, source_file: str = "") -> List[DocumentChunk]:
        """将文本分割成块
        
        Args:
            text: 要分割的文本
            source_file: 源文件路径
            
        Returns:
            List[DocumentChunk]: 分割后的文档块列表
        """
        pass
    
    @abstractmethod
    def get_chunker_type(self) -> str:
        """获取分块器类型
        
        Returns:
            str: 分块器类型标识
        """
        pass
    
    def chunk_document(self, content: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """分割文档
        
        Args:
            content: 文档内容
            metadata: 文档元数据
            
        Returns:
            List[DocumentChunk]: 分割后的文档块列表
        """
        try:
            source_file = metadata.get('source_file', '') if metadata else ''
            chunks = self.chunk_text(content, source_file)
            
            # 更新块的元数据
            if metadata:
                for chunk in chunks:
                    self._update_chunk_metadata(chunk, metadata)
            
            # 后处理
            chunks = self._post_process_chunks(chunks)
            
            self.logger.info(f"文档分块完成，生成 {len(chunks)} 个块")
            return chunks
            
        except Exception as e:
            self.logger.error(f"文档分块失败: {e}")
            raise
    
    def _update_chunk_metadata(self, chunk: DocumentChunk, document_metadata: Dict[str, Any]):
        """更新块的元数据
        
        Args:
            chunk: 文档块
            document_metadata: 文档元数据
        """
        # 更新源文件信息
        if 'source_file' in document_metadata:
            chunk.metadata.source_file = document_metadata['source_file']
        
        # 更新语言信息
        if 'language' in document_metadata:
            chunk.metadata.language = document_metadata['language']
        
        # 更新页面信息
        if 'page_number' in document_metadata:
            chunk.metadata.page_number = document_metadata['page_number']
        
        # 更新自定义元数据
        if 'custom_metadata' in document_metadata:
            chunk.metadata.custom_metadata.update(document_metadata['custom_metadata'])
    
    def _post_process_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """后处理块列表
        
        Args:
            chunks: 原始块列表
            
        Returns:
            List[DocumentChunk]: 处理后的块列表
        """
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            # 更新块索引
            chunk.metadata.chunk_index = i
            
            # 过滤空块
            if self.config.filter_empty_chunks and not chunk.content.strip():
                continue
            
            # 标准化空白字符
            if self.config.normalize_whitespace:
                chunk.content = self._normalize_whitespace(chunk.content)
            
            # 检查块大小
            if len(chunk.content) < self.config.min_chunk_size:
                self.logger.warning(f"块 {chunk.metadata.chunk_id} 小于最小大小 {self.config.min_chunk_size}")
            
            if len(chunk.content) > self.config.max_chunk_size:
                self.logger.warning(f"块 {chunk.metadata.chunk_id} 超过最大大小 {self.config.max_chunk_size}")
            
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _normalize_whitespace(self, text: str) -> str:
        """标准化空白字符
        
        Args:
            text: 原始文本
            
        Returns:
            str: 标准化后的文本
        """
        import re
        
        # 替换多个空白字符为单个空格
        text = re.sub(r'\s+', ' ', text)
        
        # 移除首尾空白
        text = text.strip()
        
        return text
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言
        
        Args:
            text: 文本内容
            
        Returns:
            str: 语言代码
        """
        try:
            from langdetect import detect
            return detect(text)
        except Exception:
            return "unknown"
    
    def _create_chunk(self, content: str, start_pos: int, end_pos: int, 
                     source_file: str = "", chunk_index: int = 0) -> DocumentChunk:
        """创建文档块
        
        Args:
            content: 块内容
            start_pos: 开始位置
            end_pos: 结束位置
            source_file: 源文件
            chunk_index: 块索引
            
        Returns:
            DocumentChunk: 文档块
        """
        metadata = ChunkMetadata(
            source_file=source_file,
            chunk_index=chunk_index,
            start_position=start_pos,
            end_position=end_pos,
            chunk_type=self.get_chunker_type(),
            language=self._detect_language(content) if self.config.language == "auto" else self.config.language
        )
        
        return DocumentChunk(content=content, metadata=metadata)
    
    def validate_config(self) -> bool:
        """验证配置
        
        Returns:
            bool: 配置是否有效
        """
        try:
            # 检查块大小配置
            if self.config.chunk_size <= 0:
                raise ValueError("chunk_size 必须大于 0")
            
            if self.config.chunk_overlap < 0:
                raise ValueError("chunk_overlap 不能小于 0")
            
            if self.config.chunk_overlap >= self.config.chunk_size:
                raise ValueError("chunk_overlap 不能大于等于 chunk_size")
            
            if self.config.min_chunk_size <= 0:
                raise ValueError("min_chunk_size 必须大于 0")
            
            if self.config.max_chunk_size <= self.config.min_chunk_size:
                raise ValueError("max_chunk_size 必须大于 min_chunk_size")
            
            return True
            
        except Exception as e:
            self.logger.error(f"配置验证失败: {e}")
            return False
    
    def get_config_info(self) -> Dict[str, Any]:
        """获取配置信息
        
        Returns:
            Dict[str, Any]: 配置信息
        """
        return {
            'chunker_type': self.get_chunker_type(),
            'chunk_size': self.config.chunk_size,
            'chunk_overlap': self.config.chunk_overlap,
            'min_chunk_size': self.config.min_chunk_size,
            'max_chunk_size': self.config.max_chunk_size,
            'preserve_sentences': self.config.preserve_sentences,
            'preserve_paragraphs': self.config.preserve_paragraphs,
            'language': self.config.language,
            'custom_separators': self.config.custom_separators,
            'metadata_extraction': self.config.metadata_extraction,
            'filter_empty_chunks': self.config.filter_empty_chunks,
            'normalize_whitespace': self.config.normalize_whitespace
        }