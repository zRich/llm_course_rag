from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime

# 配置日志
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """文档元数据类"""
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    page_count: Optional[int] = None
    file_size: Optional[int] = None
    language: Optional[str] = None
    keywords: Optional[List[str]] = None
    subject: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'title': self.title,
            'author': self.author,
            'creation_date': self.creation_date.isoformat() if self.creation_date else None,
            'modification_date': self.modification_date.isoformat() if self.modification_date else None,
            'page_count': self.page_count,
            'file_size': self.file_size,
            'language': self.language,
            'keywords': self.keywords,
            'subject': self.subject
        }

@dataclass
class ParsedDocument:
    """解析后的文档数据类"""
    content: str
    metadata: DocumentMetadata
    file_path: str
    file_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'content': self.content,
            'metadata': self.metadata.to_dict(),
            'file_path': self.file_path,
            'file_type': self.file_type
        }

class DocumentParser(ABC):
    """文档解析器基类"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def can_parse(self, file_path: str) -> bool:
        """检查是否可以解析指定文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否可以解析
        """
        pass
    
    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        """解析文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            ParsedDocument: 解析后的文档数据
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持
            Exception: 解析过程中的其他错误
        """
        pass
    
    @abstractmethod
    def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """提取文档元数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            DocumentMetadata: 文档元数据
        """
        pass
    
    def validate_file(self, file_path: str) -> None:
        """验证文件是否存在且可读
        
        Args:
            file_path: 文件路径
            
        Raises:
            FileNotFoundError: 文件不存在
            PermissionError: 文件无法读取
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"路径不是文件: {file_path}")
        
        if not path.stat().st_size > 0:
            raise ValueError(f"文件为空: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                f.read(1)
        except PermissionError:
            raise PermissionError(f"无法读取文件: {file_path}")
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """获取文件基本信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict[str, Any]: 文件信息
        """
        path = Path(file_path)
        stat = path.stat()
        
        return {
            'file_name': path.name,
            'file_size': stat.st_size,
            'file_extension': path.suffix.lower(),
            'creation_time': datetime.fromtimestamp(stat.st_ctime),
            'modification_time': datetime.fromtimestamp(stat.st_mtime)
        }
    
    def clean_text(self, text: str) -> str:
        """清理文本内容
        
        Args:
            text: 原始文本
            
        Returns:
            str: 清理后的文本
        """
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = ' '.join(text.split())
        
        # 移除特殊字符（保留基本标点）
        import re
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
        return text.strip()
    
    def detect_language(self, text: str) -> Optional[str]:
        """检测文本语言
        
        Args:
            text: 文本内容
            
        Returns:
            Optional[str]: 语言代码
        """
        try:
            from langdetect import detect
            if len(text.strip()) < 10:
                return None
            return detect(text)
        except Exception as e:
            self.logger.warning(f"语言检测失败: {e}")
            return None