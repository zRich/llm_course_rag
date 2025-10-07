from typing import List, Dict, Any, Optional, Union, Type
import logging
from pathlib import Path

from .chunker import DocumentChunker, DocumentChunk, ChunkingConfig
from .sentence_chunker import SentenceChunker
from .semantic_chunker import SemanticChunker
from .structure_chunker import StructureChunker

logger = logging.getLogger(__name__)

class ChunkManager:
    """分块管理器
    
    统一管理所有分块器，提供统一的分块接口
    """
    
    def __init__(self):
        self.chunkers: Dict[str, DocumentChunker] = {}
        self.default_config = ChunkingConfig()
        
        # 注册默认分块器
        self._register_default_chunkers()
    
    def _register_default_chunkers(self) -> None:
        """注册默认分块器"""
        try:
            # 注册句子分块器
            self.register_chunker("sentence", SentenceChunker(self.default_config))
            
            # 注册语义分块器
            self.register_chunker("semantic", SemanticChunker(self.default_config))
            
            # 注册结构分块器
            self.register_chunker("structure", StructureChunker(self.default_config))
            
            logger.info("默认分块器注册完成")
            
        except Exception as e:
            logger.error(f"注册默认分块器失败: {e}")
    
    def register_chunker(self, name: str, chunker: DocumentChunker) -> None:
        """注册分块器
        
        Args:
            name: 分块器名称
            chunker: 分块器实例
        """
        if not isinstance(chunker, DocumentChunker):
            raise ValueError(f"分块器必须继承自DocumentChunker: {type(chunker)}")
        
        self.chunkers[name] = chunker
        logger.info(f"分块器 '{name}' 注册成功")
    
    def get_chunker(self, name: str) -> Optional[DocumentChunker]:
        """获取分块器
        
        Args:
            name: 分块器名称
            
        Returns:
            Optional[DocumentChunker]: 分块器实例，如果不存在则返回None
        """
        return self.chunkers.get(name)
    
    def list_chunkers(self) -> List[str]:
        """列出所有可用的分块器
        
        Returns:
            List[str]: 分块器名称列表
        """
        return list(self.chunkers.keys())
    
    def chunk_text(self, text: str, chunker_type: str = "sentence", 
                   source_file: str = "", config: Optional[ChunkingConfig] = None) -> List[DocumentChunk]:
        """使用指定分块器分割文本
        
        Args:
            text: 要分割的文本
            chunker_type: 分块器类型
            source_file: 源文件路径
            config: 分块配置（可选）
            
        Returns:
            List[DocumentChunk]: 分割后的文档块列表
            
        Raises:
            ValueError: 如果分块器不存在
        """
        chunker = self.get_chunker(chunker_type)
        if not chunker:
            raise ValueError(f"分块器 '{chunker_type}' 不存在。可用分块器: {self.list_chunkers()}")
        
        # 如果提供了配置，临时更新分块器配置
        original_config = None
        if config:
            original_config = chunker.config
            chunker.config = config
        
        try:
            chunks = chunker.chunk_text(text, source_file)
            logger.info(f"使用 '{chunker_type}' 分块器处理文本: {len(text)} 字符 -> {len(chunks)} 个块")
            return chunks
        finally:
            # 恢复原始配置
            if original_config:
                chunker.config = original_config
    
    def chunk_file(self, file_path: Union[str, Path], chunker_type: str = "sentence",
                   config: Optional[ChunkingConfig] = None) -> List[DocumentChunk]:
        """分割文件内容
        
        Args:
            file_path: 文件路径
            chunker_type: 分块器类型
            config: 分块配置（可选）
            
        Returns:
            List[DocumentChunk]: 分割后的文档块列表
            
        Raises:
            FileNotFoundError: 如果文件不存在
            ValueError: 如果分块器不存在
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            return self.chunk_text(text, chunker_type, str(file_path), config)
            
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    text = f.read()
                return self.chunk_text(text, chunker_type, str(file_path), config)
            except Exception as e:
                logger.error(f"读取文件失败: {file_path}, 错误: {e}")
                raise
    
    def batch_chunk_files(self, file_paths: List[Union[str, Path]], 
                         chunker_type: str = "sentence",
                         config: Optional[ChunkingConfig] = None) -> Dict[str, List[DocumentChunk]]:
        """批量分割多个文件
        
        Args:
            file_paths: 文件路径列表
            chunker_type: 分块器类型
            config: 分块配置（可选）
            
        Returns:
            Dict[str, List[DocumentChunk]]: 文件路径到分块结果的映射
        """
        results = {}
        
        for file_path in file_paths:
            try:
                chunks = self.chunk_file(file_path, chunker_type, config)
                results[str(file_path)] = chunks
                logger.info(f"文件 {file_path} 分块完成: {len(chunks)} 个块")
            except Exception as e:
                logger.error(f"文件 {file_path} 分块失败: {e}")
                results[str(file_path)] = []
        
        return results
    
    def compare_chunkers(self, text: str, chunker_types: List[str], 
                        source_file: str = "") -> Dict[str, Any]:
        """比较不同分块器的效果
        
        Args:
            text: 要分割的文本
            chunker_types: 要比较的分块器类型列表
            source_file: 源文件路径
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        results = {}
        
        for chunker_type in chunker_types:
            try:
                chunks = self.chunk_text(text, chunker_type, source_file)
                
                # 计算统计信息
                chunk_lengths = [len(chunk.content) for chunk in chunks]
                word_counts = [chunk.metadata.word_count for chunk in chunks]
                
                results[chunker_type] = {
                    'chunk_count': len(chunks),
                    'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0,
                    'min_chunk_length': min(chunk_lengths) if chunk_lengths else 0,
                    'max_chunk_length': max(chunk_lengths) if chunk_lengths else 0,
                    'avg_word_count': sum(word_counts) / len(word_counts) if word_counts else 0,
                    'total_chunks': len(chunks),
                    'chunks': chunks
                }
                
            except Exception as e:
                logger.error(f"分块器 '{chunker_type}' 比较失败: {e}")
                results[chunker_type] = {'error': str(e)}
        
        return results
    
    def get_chunker_info(self, chunker_type: str) -> Dict[str, Any]:
        """获取分块器信息
        
        Args:
            chunker_type: 分块器类型
            
        Returns:
            Dict[str, Any]: 分块器信息
        """
        chunker = self.get_chunker(chunker_type)
        if not chunker:
            return {'error': f'分块器 {chunker_type} 不存在'}
        
        return {
            'type': chunker.get_chunker_type(),
            'class_name': chunker.__class__.__name__,
            'config': {
                'chunk_size': chunker.config.chunk_size,
                'overlap_size': chunker.config.overlap_size,
                'min_chunk_size': chunker.config.min_chunk_size,
                'preserve_formatting': chunker.config.preserve_formatting,
                'language': chunker.config.language
            }
        }
    
    def create_chunker(self, chunker_type: str, config: Optional[ChunkingConfig] = None) -> DocumentChunker:
        """创建新的分块器实例
        
        Args:
            chunker_type: 分块器类型
            config: 分块配置（可选）
            
        Returns:
            DocumentChunker: 分块器实例
            
        Raises:
            ValueError: 如果分块器类型不支持
        """
        config = config or self.default_config
        
        chunker_classes = {
            'sentence': SentenceChunker,
            'semantic': SemanticChunker,
            'structure': StructureChunker
        }
        
        if chunker_type not in chunker_classes:
            raise ValueError(f"不支持的分块器类型: {chunker_type}。支持的类型: {list(chunker_classes.keys())}")
        
        chunker_class = chunker_classes[chunker_type]
        return chunker_class(config)
    
    def optimize_chunking_strategy(self, text: str, target_chunk_count: Optional[int] = None,
                                 max_chunk_size: Optional[int] = None) -> Dict[str, Any]:
        """优化分块策略
        
        Args:
            text: 文本内容
            target_chunk_count: 目标块数量（可选）
            max_chunk_size: 最大块大小（可选）
            
        Returns:
            Dict[str, Any]: 优化建议
        """
        text_length = len(text)
        word_count = len(text.split())
        
        # 分析文本特征
        has_structure = bool(re.search(r'^#{1,6}\s+.+$', text, re.MULTILINE))
        has_lists = bool(re.search(r'^[•·*-]\s+.+$', text, re.MULTILINE))
        has_numbers = bool(re.search(r'^\d+\.\s+.+$', text, re.MULTILINE))
        
        # 计算段落数量
        paragraphs = re.split(r'\n\s*\n', text)
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        recommendations = []
        
        # 基于文本特征推荐分块器
        if has_structure:
            recommendations.append({
                'chunker': 'structure',
                'reason': '文本包含明确的结构标记（标题、列表等）',
                'priority': 1
            })
        
        if word_count > 1000:
            recommendations.append({
                'chunker': 'semantic',
                'reason': '文本较长，语义分块可能效果更好',
                'priority': 2
            })
        
        recommendations.append({
            'chunker': 'sentence',
            'reason': '通用的句子分块器，适用于大多数情况',
            'priority': 3
        })
        
        # 配置建议
        config_suggestions = {}
        
        if target_chunk_count:
            suggested_chunk_size = max(100, text_length // target_chunk_count)
            config_suggestions['chunk_size'] = suggested_chunk_size
        
        if max_chunk_size:
            config_suggestions['chunk_size'] = min(
                config_suggestions.get('chunk_size', self.default_config.chunk_size),
                max_chunk_size
            )
        
        return {
            'text_stats': {
                'length': text_length,
                'word_count': word_count,
                'paragraph_count': paragraph_count,
                'has_structure': has_structure,
                'has_lists': has_lists,
                'has_numbers': has_numbers
            },
            'recommendations': sorted(recommendations, key=lambda x: x['priority']),
            'config_suggestions': config_suggestions
        }
    
    def export_chunks(self, chunks: List[DocumentChunk], format_type: str = "json") -> str:
        """导出分块结果
        
        Args:
            chunks: 文档块列表
            format_type: 导出格式（json, csv, txt）
            
        Returns:
            str: 导出的字符串内容
        """
        if format_type == "json":
            import json
            chunk_data = []
            for chunk in chunks:
                chunk_data.append({
                    'id': chunk.chunk_id,
                    'content': chunk.content,
                    'start_position': chunk.start_position,
                    'end_position': chunk.end_position,
                    'source_file': chunk.source_file,
                    'metadata': {
                        'word_count': chunk.metadata.word_count,
                        'char_count': chunk.metadata.char_count,
                        'chunk_type': chunk.metadata.chunk_type,
                        'language': chunk.metadata.language,
                        'custom_metadata': chunk.metadata.custom_metadata
                    }
                })
            return json.dumps(chunk_data, ensure_ascii=False, indent=2)
        
        elif format_type == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # 写入标题行
            writer.writerow(['chunk_id', 'content', 'word_count', 'char_count', 'source_file', 'chunk_type'])
            
            # 写入数据行
            for chunk in chunks:
                writer.writerow([
                    chunk.chunk_id,
                    chunk.content.replace('\n', ' '),  # 替换换行符
                    chunk.metadata.word_count,
                    chunk.metadata.char_count,
                    chunk.source_file,
                    chunk.metadata.chunk_type
                ])
            
            return output.getvalue()
        
        elif format_type == "txt":
            lines = []
            for i, chunk in enumerate(chunks):
                lines.append(f"=== 块 {i+1} (ID: {chunk.chunk_id}) ===")
                lines.append(f"来源: {chunk.source_file}")
                lines.append(f"位置: {chunk.start_position}-{chunk.end_position}")
                lines.append(f"字数: {chunk.metadata.word_count}, 字符数: {chunk.metadata.char_count}")
                lines.append(f"类型: {chunk.metadata.chunk_type}")
                lines.append("")
                lines.append(chunk.content)
                lines.append("")
                lines.append("-" * 50)
                lines.append("")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"不支持的导出格式: {format_type}。支持的格式: json, csv, txt")

# 创建全局分块管理器实例
chunk_manager = ChunkManager()