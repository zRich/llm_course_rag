import fitz  # PyMuPDF
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import logging

from .parser import DocumentParser, DocumentMetadata, ParsedDocument

logger = logging.getLogger(__name__)

class PDFParser(DocumentParser):
    """PDF文档解析器"""
    
    SUPPORTED_EXTENSIONS = ['.pdf']
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def can_parse(self, file_path: str) -> bool:
        """检查是否可以解析PDF文件
        
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
        """解析PDF文档
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            ParsedDocument: 解析后的文档数据
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持或损坏
            Exception: 解析过程中的其他错误
        """
        self.validate_file(file_path)
        
        if not self.can_parse(file_path):
            raise ValueError(f"不支持的文件格式: {file_path}")
        
        try:
            # 打开PDF文档
            doc = fitz.open(file_path)
            
            # 提取文本内容
            content = self._extract_text(doc)
            
            # 提取元数据
            metadata = self._extract_metadata_from_doc(doc, file_path)
            
            # 关闭文档
            doc.close()
            
            # 清理文本
            content = self.clean_text(content)
            
            # 检测语言
            if not metadata.language:
                metadata.language = self.detect_language(content)
            
            return ParsedDocument(
                content=content,
                metadata=metadata,
                file_path=file_path,
                file_type='pdf'
            )
            
        except Exception as e:
            self.logger.error(f"解析PDF文件失败 {file_path}: {e}")
            raise Exception(f"PDF解析失败: {e}")
    
    def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """提取PDF文档元数据
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            DocumentMetadata: 文档元数据
        """
        self.validate_file(file_path)
        
        try:
            doc = fitz.open(file_path)
            metadata = self._extract_metadata_from_doc(doc, file_path)
            doc.close()
            return metadata
        except Exception as e:
            self.logger.error(f"提取PDF元数据失败 {file_path}: {e}")
            # 返回基本元数据
            file_info = self.get_file_info(file_path)
            return DocumentMetadata(
                file_size=file_info['file_size'],
                creation_date=file_info['creation_time'],
                modification_date=file_info['modification_time']
            )
    
    def _extract_text(self, doc: fitz.Document) -> str:
        """从PDF文档中提取文本
        
        Args:
            doc: PyMuPDF文档对象
            
        Returns:
            str: 提取的文本内容
        """
        text_content = []
        
        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text.strip():
                    text_content.append(text)
                    
            except Exception as e:
                self.logger.warning(f"提取第{page_num + 1}页文本失败: {e}")
                continue
        
        return '\n\n'.join(text_content)
    
    def _extract_metadata_from_doc(self, doc: fitz.Document, file_path: str) -> DocumentMetadata:
        """从PDF文档对象中提取元数据
        
        Args:
            doc: PyMuPDF文档对象
            file_path: 文件路径
            
        Returns:
            DocumentMetadata: 文档元数据
        """
        try:
            # 获取PDF元数据
            pdf_metadata = doc.metadata
            
            # 获取文件信息
            file_info = self.get_file_info(file_path)
            
            # 解析日期
            creation_date = self._parse_pdf_date(pdf_metadata.get('creationDate'))
            modification_date = self._parse_pdf_date(pdf_metadata.get('modDate'))
            
            # 提取关键词
            keywords = None
            if pdf_metadata.get('keywords'):
                keywords = [kw.strip() for kw in pdf_metadata['keywords'].split(',') if kw.strip()]
            
            return DocumentMetadata(
                title=pdf_metadata.get('title') or file_info['file_name'],
                author=pdf_metadata.get('author'),
                creation_date=creation_date or file_info['creation_time'],
                modification_date=modification_date or file_info['modification_time'],
                page_count=len(doc),
                file_size=file_info['file_size'],
                keywords=keywords,
                subject=pdf_metadata.get('subject')
            )
            
        except Exception as e:
            self.logger.warning(f"提取PDF元数据失败: {e}")
            # 返回基本元数据
            file_info = self.get_file_info(file_path)
            return DocumentMetadata(
                title=file_info['file_name'],
                page_count=len(doc) if doc else None,
                file_size=file_info['file_size'],
                creation_date=file_info['creation_time'],
                modification_date=file_info['modification_time']
            )
    
    def _parse_pdf_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """解析PDF日期格式
        
        Args:
            date_str: PDF日期字符串
            
        Returns:
            Optional[datetime]: 解析后的日期对象
        """
        if not date_str:
            return None
        
        try:
            # PDF日期格式通常为: D:YYYYMMDDHHmmSSOHH'mm'
            if date_str.startswith('D:'):
                date_str = date_str[2:]
            
            # 提取基本日期部分
            if len(date_str) >= 14:
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                hour = int(date_str[8:10])
                minute = int(date_str[10:12])
                second = int(date_str[12:14])
                
                return datetime(year, month, day, hour, minute, second)
            
        except (ValueError, IndexError) as e:
            self.logger.warning(f"解析PDF日期失败 {date_str}: {e}")
        
        return None
    
    def extract_pages(self, file_path: str, page_range: Optional[tuple] = None) -> List[str]:
        """提取指定页面的文本
        
        Args:
            file_path: PDF文件路径
            page_range: 页面范围 (start, end)，None表示所有页面
            
        Returns:
            List[str]: 每页的文本内容
        """
        self.validate_file(file_path)
        
        try:
            doc = fitz.open(file_path)
            pages_text = []
            
            start_page = page_range[0] if page_range else 0
            end_page = page_range[1] if page_range else len(doc)
            
            for page_num in range(start_page, min(end_page, len(doc))):
                try:
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    pages_text.append(self.clean_text(text))
                except Exception as e:
                    self.logger.warning(f"提取第{page_num + 1}页失败: {e}")
                    pages_text.append("")
            
            doc.close()
            return pages_text
            
        except Exception as e:
            self.logger.error(f"提取PDF页面失败 {file_path}: {e}")
            raise Exception(f"提取PDF页面失败: {e}")
    
    def get_page_count(self, file_path: str) -> int:
        """获取PDF页面数
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            int: 页面数
        """
        self.validate_file(file_path)
        
        try:
            doc = fitz.open(file_path)
            page_count = len(doc)
            doc.close()
            return page_count
        except Exception as e:
            self.logger.error(f"获取PDF页面数失败 {file_path}: {e}")
            return 0