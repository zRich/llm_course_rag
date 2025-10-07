import chardet
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import logging

from .parser import DocumentParser, DocumentMetadata, ParsedDocument

logger = logging.getLogger(__name__)

class TxtParser(DocumentParser):
    """文本文档解析器"""
    
    SUPPORTED_EXTENSIONS = ['.txt', '.md', '.rst', '.log']
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def can_parse(self, file_path: str) -> bool:
        """检查是否可以解析文本文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否可以解析
        """
        try:
            path = Path(file_path)
            return path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        except Exception as e:
            self.logger.error(f"检查文件类型失败: {e}")
            return False
    
    def parse(self, file_path: str) -> ParsedDocument:
        """解析文本文档
        
        Args:
            file_path: 文本文件路径
            
        Returns:
            ParsedDocument: 解析后的文档数据
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持
            Exception: 解析过程中的其他错误
        """
        self.validate_file(file_path)
        
        if not self.can_parse(file_path):
            raise ValueError(f"不支持的文件格式: {file_path}")
        
        try:
            # 检测文件编码
            encoding = self._detect_encoding(file_path)
            
            # 读取文件内容
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            # 提取元数据
            metadata = self._extract_metadata_from_content(content, file_path)
            
            # 清理文本
            content = self.clean_text(content)
            
            # 检测语言
            if not metadata.language:
                metadata.language = self.detect_language(content)
            
            return ParsedDocument(
                content=content,
                metadata=metadata,
                file_path=file_path,
                file_type='txt'
            )
            
        except Exception as e:
            self.logger.error(f"解析文本文件失败 {file_path}: {e}")
            raise Exception(f"文本解析失败: {e}")
    
    def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """提取文本文档元数据
        
        Args:
            file_path: 文本文件路径
            
        Returns:
            DocumentMetadata: 文档元数据
        """
        self.validate_file(file_path)
        
        try:
            # 检测文件编码
            encoding = self._detect_encoding(file_path)
            
            # 读取文件内容
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            return self._extract_metadata_from_content(content, file_path)
            
        except Exception as e:
            self.logger.error(f"提取文本元数据失败 {file_path}: {e}")
            # 返回基本元数据
            file_info = self.get_file_info(file_path)
            return DocumentMetadata(
                title=file_info['file_name'],
                file_size=file_info['file_size'],
                creation_date=file_info['creation_time'],
                modification_date=file_info['modification_time']
            )
    
    def _detect_encoding(self, file_path: str) -> str:
        """检测文件编码
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 文件编码
        """
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # 读取前10KB用于检测编码
            
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            
            # 如果检测结果不可靠，使用常见编码
            if result.get('confidence', 0) < 0.7:
                # 尝试常见编码
                for enc in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
                    try:
                        with open(file_path, 'r', encoding=enc) as f:
                            f.read(1000)
                        encoding = enc
                        break
                    except UnicodeDecodeError:
                        continue
            
            self.logger.debug(f"检测到文件编码: {encoding}")
            return encoding
            
        except Exception as e:
            self.logger.warning(f"编码检测失败，使用UTF-8: {e}")
            return 'utf-8'
    
    def _extract_metadata_from_content(self, content: str, file_path: str) -> DocumentMetadata:
        """从文本内容中提取元数据
        
        Args:
            content: 文本内容
            file_path: 文件路径
            
        Returns:
            DocumentMetadata: 文档元数据
        """
        try:
            # 获取文件信息
            file_info = self.get_file_info(file_path)
            
            # 分析文本内容
            lines = content.split('\n')
            line_count = len(lines)
            word_count = len(content.split())
            char_count = len(content)
            
            # 尝试从内容中提取标题（第一行非空行）
            title = file_info['file_name']
            for line in lines:
                line = line.strip()
                if line and len(line) < 100:  # 假设标题不会太长
                    title = line
                    break
            
            # 估算页面数（每页约500字符）
            page_count = max(1, char_count // 500) if char_count > 0 else 1
            
            return DocumentMetadata(
                title=title,
                creation_date=file_info['creation_time'],
                modification_date=file_info['modification_time'],
                page_count=page_count,
                file_size=file_info['file_size']
            )
            
        except Exception as e:
            self.logger.warning(f"从内容提取元数据失败: {e}")
            # 返回基本元数据
            file_info = self.get_file_info(file_path)
            return DocumentMetadata(
                title=file_info['file_name'],
                file_size=file_info['file_size'],
                creation_date=file_info['creation_time'],
                modification_date=file_info['modification_time']
            )
    
    def extract_lines(self, file_path: str, line_range: Optional[tuple] = None) -> List[str]:
        """提取指定行范围的文本
        
        Args:
            file_path: 文本文件路径
            line_range: 行范围 (start, end)，None表示所有行
            
        Returns:
            List[str]: 行文本列表
        """
        self.validate_file(file_path)
        
        try:
            encoding = self._detect_encoding(file_path)
            
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            
            if line_range:
                start_line = max(0, line_range[0])
                end_line = min(len(lines), line_range[1])
                lines = lines[start_line:end_line]
            
            # 清理每行文本
            return [self.clean_text(line.rstrip('\n\r')) for line in lines]
            
        except Exception as e:
            self.logger.error(f"提取文本行失败 {file_path}: {e}")
            raise Exception(f"提取文本行失败: {e}")
    
    def get_line_count(self, file_path: str) -> int:
        """获取文件行数
        
        Args:
            file_path: 文本文件路径
            
        Returns:
            int: 行数
        """
        self.validate_file(file_path)
        
        try:
            encoding = self._detect_encoding(file_path)
            
            with open(file_path, 'r', encoding=encoding) as f:
                return sum(1 for _ in f)
                
        except Exception as e:
            self.logger.error(f"获取文件行数失败 {file_path}: {e}")
            return 0
    
    def get_word_count(self, file_path: str) -> int:
        """获取文件词数
        
        Args:
            file_path: 文本文件路径
            
        Returns:
            int: 词数
        """
        self.validate_file(file_path)
        
        try:
            encoding = self._detect_encoding(file_path)
            
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            return len(content.split())
            
        except Exception as e:
            self.logger.error(f"获取文件词数失败 {file_path}: {e}")
            return 0
    
    def extract_paragraphs(self, file_path: str) -> List[str]:
        """提取文档段落（以空行分隔）
        
        Args:
            file_path: 文本文件路径
            
        Returns:
            List[str]: 段落文本列表
        """
        self.validate_file(file_path)
        
        try:
            encoding = self._detect_encoding(file_path)
            
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            # 按空行分割段落
            paragraphs = content.split('\n\n')
            
            # 清理段落文本
            cleaned_paragraphs = []
            for paragraph in paragraphs:
                cleaned = self.clean_text(paragraph.replace('\n', ' '))
                if cleaned:
                    cleaned_paragraphs.append(cleaned)
            
            return cleaned_paragraphs
            
        except Exception as e:
            self.logger.error(f"提取文本段落失败 {file_path}: {e}")
            raise Exception(f"提取文本段落失败: {e}")