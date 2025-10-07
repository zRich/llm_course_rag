import re
from typing import List, Optional, Dict, Any, Tuple, Set
import logging
from dataclasses import dataclass

from .chunker import DocumentChunker, DocumentChunk, ChunkingConfig

logger = logging.getLogger(__name__)

@dataclass
class StructurePattern:
    """结构模式定义"""
    name: str
    pattern: str
    priority: int
    chunk_boundary: bool = True  # 是否作为块边界
    
class StructureChunker(DocumentChunker):
    """基于文档结构的分块器
    
    根据标题、段落、列表等结构特征进行智能分块
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        super().__init__(config)
        
        # 结构识别模式
        self.structure_patterns = self._init_structure_patterns()
        
        # 分块配置
        self.respect_paragraph_boundaries = True
        self.merge_short_sections = True
        self.min_section_length = 100
        self.max_section_length = self.config.chunk_size * 2
    
    def get_chunker_type(self) -> str:
        """获取分块器类型"""
        return "structure"
    
    def _init_structure_patterns(self) -> List[StructurePattern]:
        """初始化结构识别模式"""
        patterns = [
            # 标题模式（按优先级排序）
            StructurePattern("h1", r'^#{1}\s+.+$', 1, True),  # Markdown H1
            StructurePattern("h2", r'^#{2}\s+.+$', 2, True),  # Markdown H2
            StructurePattern("h3", r'^#{3}\s+.+$', 3, True),  # Markdown H3
            StructurePattern("h4", r'^#{4,6}\s+.+$', 4, True),  # Markdown H4-H6
            
            # 数字标题
            StructurePattern("numbered_title", r'^\d+\.\s+[^\n]{10,}$', 2, True),
            StructurePattern("numbered_subtitle", r'^\d+\.\d+\.?\s+[^\n]{5,}$', 3, True),
            
            # 中文标题模式
            StructurePattern("chinese_title", r'^[一二三四五六七八九十]+[、．.]\s*[^\n]{5,}$', 2, True),
            StructurePattern("chinese_subtitle", r'^\([一二三四五六七八九十]+\)\s*[^\n]{3,}$', 3, True),
            
            # 列表模式
            StructurePattern("bullet_list", r'^[•·*-]\s+.+$', 5, False),
            StructurePattern("numbered_list", r'^\d+[.).]\s+.+$', 5, False),
            StructurePattern("letter_list", r'^[a-zA-Z][.).]\s+.+$', 5, False),
            
            # 段落分隔
            StructurePattern("paragraph_break", r'^\s*$', 6, False),
            
            # 特殊结构
            StructurePattern("code_block", r'^```[\s\S]*?```$', 4, True),
            StructurePattern("quote_block", r'^>\s+.+$', 5, False),
            StructurePattern("horizontal_rule", r'^[-*_]{3,}\s*$', 3, True),
        ]
        
        return patterns
    
    def chunk_text(self, text: str, source_file: str = "") -> List[DocumentChunk]:
        """基于文档结构分割文本
        
        Args:
            text: 要分割的文本
            source_file: 源文件路径
            
        Returns:
            List[DocumentChunk]: 分割后的文档块列表
        """
        try:
            # 分析文档结构
            structure_info = self._analyze_document_structure(text)
            
            # 基于结构创建分块
            chunks = self._create_structure_based_chunks(text, structure_info, source_file)
            
            # 后处理分块
            chunks = self._post_process_chunks(chunks)
            
            self.logger.info(f"结构分块完成: {len(structure_info['sections'])} 个结构段 -> {len(chunks)} 个块")
            return chunks
            
        except Exception as e:
            self.logger.error(f"结构分块失败: {e}")
            # 回退到简单段落分块
            return self._fallback_paragraph_chunking(text, source_file)
    
    def _analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """分析文档结构
        
        Args:
            text: 文档文本
            
        Returns:
            Dict[str, Any]: 结构分析结果
        """
        lines = text.split('\n')
        structure_elements = []
        sections = []
        
        current_section = {
            'start_line': 0,
            'end_line': 0,
            'title': '',
            'level': 0,
            'content': '',
            'type': 'content'
        }
        
        for line_idx, line in enumerate(lines):
            # 检查每种结构模式
            matched_pattern = self._match_structure_pattern(line)
            
            if matched_pattern and matched_pattern.chunk_boundary:
                # 结束当前段落
                if current_section['content'].strip():
                    current_section['end_line'] = line_idx - 1
                    sections.append(current_section.copy())
                
                # 开始新段落
                current_section = {
                    'start_line': line_idx,
                    'end_line': line_idx,
                    'title': line.strip(),
                    'level': matched_pattern.priority,
                    'content': line,
                    'type': matched_pattern.name
                }
                
                structure_elements.append({
                    'line': line_idx,
                    'content': line,
                    'pattern': matched_pattern.name,
                    'priority': matched_pattern.priority
                })
            else:
                # 添加到当前段落
                if current_section['content']:
                    current_section['content'] += '\n' + line
                else:
                    current_section['content'] = line
                current_section['end_line'] = line_idx
        
        # 添加最后一个段落
        if current_section['content'].strip():
            sections.append(current_section)
        
        return {
            'total_lines': len(lines),
            'structure_elements': structure_elements,
            'sections': sections,
            'has_clear_structure': len(structure_elements) > 0
        }
    
    def _match_structure_pattern(self, line: str) -> Optional[StructurePattern]:
        """匹配结构模式
        
        Args:
            line: 文本行
            
        Returns:
            Optional[StructurePattern]: 匹配的模式，如果没有匹配则返回None
        """
        line = line.strip()
        
        for pattern in self.structure_patterns:
            if re.match(pattern.pattern, line, re.MULTILINE):
                return pattern
        
        return None
    
    def _create_structure_based_chunks(self, text: str, structure_info: Dict[str, Any], 
                                     source_file: str) -> List[DocumentChunk]:
        """基于结构信息创建分块
        
        Args:
            text: 原始文本
            structure_info: 结构分析结果
            source_file: 源文件路径
            
        Returns:
            List[DocumentChunk]: 文档块列表
        """
        chunks = []
        sections = structure_info['sections']
        
        if not sections:
            # 没有明确结构，按段落分块
            return self._fallback_paragraph_chunking(text, source_file)
        
        lines = text.split('\n')
        
        for section_idx, section in enumerate(sections):
            # 获取段落内容
            section_lines = lines[section['start_line']:section['end_line'] + 1]
            section_content = '\n'.join(section_lines)
            
            # 计算在原文中的位置
            start_pos = self._calculate_text_position(text, section['start_line'])
            end_pos = start_pos + len(section_content)
            
            # 检查段落长度
            if len(section_content) > self.max_section_length:
                # 段落太长，需要进一步分割
                sub_chunks = self._split_long_section(section_content, start_pos, source_file, section_idx)
                chunks.extend(sub_chunks)
            elif len(section_content.strip()) >= self.min_section_length or not self.merge_short_sections:
                # 创建单个块
                chunk = self._create_structure_chunk(
                    section_content, start_pos, end_pos, source_file, section_idx, section
                )
                chunks.append(chunk)
            else:
                # 段落太短，尝试与前一个块合并
                if chunks and self._can_merge_with_previous(chunks[-1], section_content):
                    self._merge_with_previous_chunk(chunks[-1], section_content, end_pos, section)
                else:
                    # 创建独立的小块
                    chunk = self._create_structure_chunk(
                        section_content, start_pos, end_pos, source_file, section_idx, section
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def _calculate_text_position(self, text: str, line_number: int) -> int:
        """计算指定行在文本中的字符位置
        
        Args:
            text: 完整文本
            line_number: 行号（从0开始）
            
        Returns:
            int: 字符位置
        """
        lines = text.split('\n')
        position = 0
        
        for i in range(min(line_number, len(lines))):
            if i > 0:
                position += 1  # 换行符
            position += len(lines[i])
        
        return position
    
    def _split_long_section(self, content: str, start_pos: int, source_file: str, 
                          section_idx: int) -> List[DocumentChunk]:
        """分割过长的段落
        
        Args:
            content: 段落内容
            start_pos: 起始位置
            source_file: 源文件
            section_idx: 段落索引
            
        Returns:
            List[DocumentChunk]: 分割后的块列表
        """
        chunks = []
        
        # 尝试按段落分割
        paragraphs = self._split_by_paragraphs(content)
        
        current_chunk_content = ""
        current_start_pos = start_pos
        chunk_count = 0
        
        for paragraph in paragraphs:
            # 检查添加当前段落是否会超过限制
            potential_content = current_chunk_content
            if potential_content:
                potential_content += "\n\n" + paragraph
            else:
                potential_content = paragraph
            
            if len(potential_content) <= self.config.chunk_size or not current_chunk_content:
                # 可以添加到当前块
                current_chunk_content = potential_content
            else:
                # 创建当前块
                if current_chunk_content:
                    chunk = self._create_chunk(
                        current_chunk_content, 
                        current_start_pos, 
                        current_start_pos + len(current_chunk_content),
                        source_file, 
                        f"{section_idx}_{chunk_count}"
                    )
                    chunk.metadata.chunk_type = "structure_split"
                    chunks.append(chunk)
                    
                    current_start_pos += len(current_chunk_content) + 2  # +2 for \n\n
                    chunk_count += 1
                
                # 开始新块
                current_chunk_content = paragraph
        
        # 添加最后一个块
        if current_chunk_content:
            chunk = self._create_chunk(
                current_chunk_content,
                current_start_pos,
                current_start_pos + len(current_chunk_content),
                source_file,
                f"{section_idx}_{chunk_count}"
            )
            chunk.metadata.chunk_type = "structure_split"
            chunks.append(chunk)
        
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """按段落分割文本
        
        Args:
            text: 文本内容
            
        Returns:
            List[str]: 段落列表
        """
        # 按双换行符分割
        paragraphs = re.split(r'\n\s*\n', text)
        
        # 清理和过滤
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para:
                cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
    def _create_structure_chunk(self, content: str, start_pos: int, end_pos: int,
                              source_file: str, chunk_idx: int, 
                              section_info: Dict[str, Any]) -> DocumentChunk:
        """创建结构化块
        
        Args:
            content: 块内容
            start_pos: 起始位置
            end_pos: 结束位置
            source_file: 源文件
            chunk_idx: 块索引
            section_info: 段落信息
            
        Returns:
            DocumentChunk: 文档块
        """
        chunk = self._create_chunk(content, start_pos, end_pos, source_file, chunk_idx)
        
        # 添加结构相关的元数据
        chunk.metadata.chunk_type = "structure"
        chunk.metadata.custom_metadata.update({
            'section_title': section_info.get('title', ''),
            'section_level': section_info.get('level', 0),
            'section_type': section_info.get('type', 'content'),
            'start_line': section_info.get('start_line', 0),
            'end_line': section_info.get('end_line', 0)
        })
        
        return chunk
    
    def _can_merge_with_previous(self, previous_chunk: DocumentChunk, new_content: str) -> bool:
        """检查是否可以与前一个块合并
        
        Args:
            previous_chunk: 前一个块
            new_content: 新内容
            
        Returns:
            bool: 是否可以合并
        """
        if not self.merge_short_sections:
            return False
        
        combined_length = len(previous_chunk.content) + len(new_content) + 2  # +2 for separator
        return combined_length <= self.config.chunk_size
    
    def _merge_with_previous_chunk(self, previous_chunk: DocumentChunk, new_content: str,
                                 new_end_pos: int, section_info: Dict[str, Any]) -> None:
        """与前一个块合并
        
        Args:
            previous_chunk: 前一个块
            new_content: 新内容
            new_end_pos: 新的结束位置
            section_info: 段落信息
        """
        # 更新内容
        previous_chunk.content += "\n\n" + new_content
        
        # 更新位置
        previous_chunk.end_position = new_end_pos
        
        # 更新元数据
        previous_chunk.metadata.word_count = len(previous_chunk.content.split())
        previous_chunk.metadata.char_count = len(previous_chunk.content)
        
        # 更新自定义元数据
        if 'merged_sections' not in previous_chunk.metadata.custom_metadata:
            previous_chunk.metadata.custom_metadata['merged_sections'] = []
        
        previous_chunk.metadata.custom_metadata['merged_sections'].append({
            'title': section_info.get('title', ''),
            'type': section_info.get('type', 'content'),
            'level': section_info.get('level', 0)
        })
    
    def _post_process_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """后处理分块结果
        
        Args:
            chunks: 原始块列表
            
        Returns:
            List[DocumentChunk]: 处理后的块列表
        """
        if not chunks:
            return chunks
        
        processed_chunks = []
        
        for chunk in chunks:
            # 清理内容
            chunk.content = self._clean_chunk_content(chunk.content)
            
            # 过滤空块
            if chunk.content.strip():
                processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _clean_chunk_content(self, content: str) -> str:
        """清理块内容
        
        Args:
            content: 原始内容
            
        Returns:
            str: 清理后的内容
        """
        # 标准化换行符
        content = re.sub(r'\r\n', '\n', content)
        content = re.sub(r'\r', '\n', content)
        
        # 移除多余的空行
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # 清理行首行尾空白
        lines = content.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        content = '\n'.join(cleaned_lines)
        
        return content.strip()
    
    def _fallback_paragraph_chunking(self, text: str, source_file: str) -> List[DocumentChunk]:
        """回退的段落分块方法
        
        Args:
            text: 文本内容
            source_file: 源文件路径
            
        Returns:
            List[DocumentChunk]: 文档块列表
        """
        chunks = []
        paragraphs = self._split_by_paragraphs(text)
        
        current_chunk_content = ""
        current_start_pos = 0
        chunk_count = 0
        
        for paragraph in paragraphs:
            # 检查添加当前段落是否会超过限制
            potential_content = current_chunk_content
            if potential_content:
                potential_content += "\n\n" + paragraph
            else:
                potential_content = paragraph
            
            if len(potential_content) <= self.config.chunk_size or not current_chunk_content:
                current_chunk_content = potential_content
            else:
                # 创建当前块
                if current_chunk_content:
                    chunk = self._create_chunk(
                        current_chunk_content,
                        current_start_pos,
                        current_start_pos + len(current_chunk_content),
                        source_file,
                        chunk_count
                    )
                    chunk.metadata.chunk_type = "paragraph"
                    chunks.append(chunk)
                    
                    current_start_pos += len(current_chunk_content) + 2
                    chunk_count += 1
                
                current_chunk_content = paragraph
        
        # 添加最后一个块
        if current_chunk_content:
            chunk = self._create_chunk(
                current_chunk_content,
                current_start_pos,
                current_start_pos + len(current_chunk_content),
                source_file,
                chunk_count
            )
            chunk.metadata.chunk_type = "paragraph"
            chunks.append(chunk)
        
        return chunks
    
    def analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """分析文档结构（公共接口）
        
        Args:
            text: 文档文本
            
        Returns:
            Dict[str, Any]: 结构分析结果
        """
        structure_info = self._analyze_document_structure(text)
        
        # 添加统计信息
        sections = structure_info['sections']
        structure_elements = structure_info['structure_elements']
        
        # 计算结构层次
        levels = [elem['priority'] for elem in structure_elements]
        level_counts = {}
        for level in levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # 计算段落长度统计
        section_lengths = [len(section['content']) for section in sections]
        
        analysis_result = {
            'total_sections': len(sections),
            'structure_elements': len(structure_elements),
            'has_clear_structure': structure_info['has_clear_structure'],
            'structure_levels': level_counts,
            'avg_section_length': sum(section_lengths) / len(section_lengths) if section_lengths else 0,
            'min_section_length': min(section_lengths) if section_lengths else 0,
            'max_section_length': max(section_lengths) if section_lengths else 0,
            'sections_detail': [
                {
                    'index': i,
                    'title': section['title'],
                    'type': section['type'],
                    'level': section['level'],
                    'length': len(section['content']),
                    'start_line': section['start_line'],
                    'end_line': section['end_line']
                }
                for i, section in enumerate(sections)
            ]
        }
        
        return analysis_result