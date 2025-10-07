"""分块器模块

提供多种文档分块策略：
- 基于句子的分块器
- 基于语义的分块器  
- 基于结构的分块器
- 统一的分块管理器
"""

from .chunker import (
    DocumentChunker,
    DocumentChunk,
    ChunkMetadata,
    ChunkingConfig
)

from .sentence_chunker import SentenceChunker
from .semantic_chunker import SemanticChunker
from .structure_chunker import StructureChunker
from .chunk_manager import ChunkManager, chunk_manager

__all__ = [
    # 基础类
    'DocumentChunker',
    'DocumentChunk', 
    'ChunkMetadata',
    'ChunkingConfig',
    
    # 分块器实现
    'SentenceChunker',
    'SemanticChunker',
    'StructureChunker',
    
    # 管理器
    'ChunkManager',
    'chunk_manager'
]