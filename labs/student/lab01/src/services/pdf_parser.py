"""
PDF解析服务
实现PDF文档的文本提取和预处理功能
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import re

import PyPDF2
import pdfplumber
from pdfplumber.page import Page

from src.config.settings import settings

logger = logging.getLogger(__name__)


class PDFParser:
    """PDF解析器"""
    
    def __init__(self):
        self.max_file_size = settings.max_file_size  # 已经是字节单位
        
    def validate_pdf(self, file_path: Path) -> Tuple[bool, str]:
        """
        验证PDF文件
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            (is_valid, error_message)
        """
        try:
            # 检查文件是否存在
            if not file_path.exists():
                return False, "文件不存在"
                
            # 检查文件大小
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                return False, f"文件大小超过限制 ({self.max_file_size / 1024 / 1024:.1f}MB)"
                
            # 检查文件扩展名
            if file_path.suffix.lower() != '.pdf':
                return False, "文件格式不是PDF"
                
            # 尝试打开PDF文件
            with open(file_path, 'rb') as file:
                try:
                    PyPDF2.PdfReader(file)
                except Exception as e:
                    return False, f"PDF文件损坏或无法读取: {str(e)}"
                    
            return True, ""
            
        except Exception as e:
            logger.error(f"PDF验证失败: {e}")
            return False, f"验证过程出错: {str(e)}"
    
    def extract_text_pypdf2(self, file_path: Path) -> Tuple[str, Dict]:
        """
        使用PyPDF2提取PDF文本
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            (extracted_text, metadata)
        """
        text_content = ""
        metadata = {
            "total_pages": 0,
            "extraction_method": "pypdf2",
            "has_images": False,
            "has_tables": False
        }
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata["total_pages"] = len(pdf_reader.pages)
                
                # 提取PDF元数据
                if pdf_reader.metadata:
                    metadata.update({
                        "title": pdf_reader.metadata.get('/Title', ''),
                        "author": pdf_reader.metadata.get('/Author', ''),
                        "subject": pdf_reader.metadata.get('/Subject', ''),
                        "creator": pdf_reader.metadata.get('/Creator', ''),
                        "producer": pdf_reader.metadata.get('/Producer', ''),
                        "creation_date": str(pdf_reader.metadata.get('/CreationDate', '')),
                        "modification_date": str(pdf_reader.metadata.get('/ModDate', ''))
                    })
                
                # 逐页提取文本
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += f"\n--- 第{page_num}页 ---\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"第{page_num}页文本提取失败: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"PyPDF2文本提取失败: {e}")
            raise
            
        return text_content, metadata
    
    def extract_text_pdfplumber(self, file_path: Path) -> Tuple[str, Dict]:
        """
        使用pdfplumber提取PDF文本（更好的表格和布局处理）
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            (extracted_text, metadata)
        """
        text_content = ""
        metadata = {
            "total_pages": 0,
            "extraction_method": "pdfplumber",
            "has_images": False,
            "has_tables": False,
            "tables_count": 0,
            "images_count": 0
        }
        
        try:
            with pdfplumber.open(file_path) as pdf:
                metadata["total_pages"] = len(pdf.pages)
                
                # 提取PDF元数据
                if pdf.metadata:
                    metadata.update({
                        "title": pdf.metadata.get('Title', ''),
                        "author": pdf.metadata.get('Author', ''),
                        "subject": pdf.metadata.get('Subject', ''),
                        "creator": pdf.metadata.get('Creator', ''),
                        "producer": pdf.metadata.get('Producer', ''),
                        "creation_date": str(pdf.metadata.get('CreationDate', '')),
                        "modification_date": str(pdf.metadata.get('ModDate', ''))
                    })
                
                # 逐页提取文本和结构化内容
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = ""
                        
                        # 检查是否有表格
                        tables = page.find_tables()
                        if tables:
                            metadata["has_tables"] = True
                            metadata["tables_count"] += len(tables)
                            
                            # 提取表格内容
                            for table_idx, table in enumerate(tables):
                                try:
                                    table_data = table.extract()
                                    if table_data:
                                        page_text += f"\n[表格 {table_idx + 1}]\n"
                                        for row in table_data:
                                            if row:
                                                page_text += " | ".join([cell or "" for cell in row]) + "\n"
                                        page_text += "\n"
                                except Exception as e:
                                    logger.warning(f"第{page_num}页表格{table_idx + 1}提取失败: {e}")
                        
                        # 提取常规文本（排除表格区域）
                        regular_text = page.extract_text()
                        if regular_text:
                            page_text += regular_text
                        
                        # 检查是否有图片
                        if hasattr(page, 'images') and page.images:
                            metadata["has_images"] = True
                            metadata["images_count"] += len(page.images)
                            page_text += f"\n[页面包含 {len(page.images)} 个图片]\n"
                        
                        if page_text.strip():
                            text_content += f"\n--- 第{page_num}页 ---\n{page_text}\n"
                            
                    except Exception as e:
                        logger.warning(f"第{page_num}页内容提取失败: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"pdfplumber文本提取失败: {e}")
            raise
            
        return text_content, metadata
    
    def clean_text(self, text: str) -> str:
        """
        清理提取的文本
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
            
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符和控制字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # 规范化换行符
        text = re.sub(r'\r\n|\r', '\n', text)
        
        # 移除过多的连续换行
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 移除行首行尾空白
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def calculate_text_hash(self, text: str) -> str:
        """
        计算文本内容的哈希值
        
        Args:
            text: 文本内容
            
        Returns:
            SHA256哈希值
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def parse_pdf(self, file_path: Path, use_pdfplumber: bool = True) -> Dict:
        """
        解析PDF文件
        
        Args:
            file_path: PDF文件路径
            use_pdfplumber: 是否使用pdfplumber（默认True，更好的表格处理）
            
        Returns:
            解析结果字典
        """
        logger.info(f"开始解析PDF文件: {file_path}")
        
        # 验证PDF文件
        is_valid, error_msg = self.validate_pdf(file_path)
        if not is_valid:
            raise ValueError(f"PDF文件验证失败: {error_msg}")
        
        try:
            # 选择提取方法
            if use_pdfplumber:
                raw_text, metadata = self.extract_text_pdfplumber(file_path)
            else:
                raw_text, metadata = self.extract_text_pypdf2(file_path)
            
            # 清理文本
            cleaned_text = self.clean_text(raw_text)
            
            # 计算统计信息
            char_count = len(cleaned_text)
            word_count = len(cleaned_text.split()) if cleaned_text else 0
            
            # 估算token数量（粗略估算：1 token ≈ 4 字符）
            estimated_tokens = char_count // 4
            
            # 计算内容哈希
            content_hash = self.calculate_text_hash(cleaned_text)
            
            result = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "raw_text": raw_text,
                "cleaned_text": cleaned_text,
                "content_hash": content_hash,
                "char_count": char_count,
                "word_count": word_count,
                "estimated_tokens": estimated_tokens,
                "metadata": metadata,
                "extraction_success": True,
                "extraction_method": metadata.get("extraction_method", "unknown")
            }
            
            logger.info(f"PDF解析完成: {file_path.name}, "
                       f"页数: {metadata.get('total_pages', 0)}, "
                       f"字符数: {char_count}, "
                       f"预估tokens: {estimated_tokens}")
            
            return result
            
        except Exception as e:
            logger.error(f"PDF解析失败: {file_path}, 错误: {e}")
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size if file_path.exists() else 0,
                "raw_text": "",
                "cleaned_text": "",
                "content_hash": "",
                "char_count": 0,
                "word_count": 0,
                "estimated_tokens": 0,
                "metadata": {},
                "extraction_success": False,
                "extraction_method": "failed",
                "error_message": str(e)
            }