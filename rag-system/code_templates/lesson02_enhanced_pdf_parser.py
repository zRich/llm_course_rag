#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lesson02 增强PDF解析器实现模板
解决PDF解析优化、文档预处理和错误处理机制缺失问题

功能特性：
1. 布局感知的PDF解析
2. 复杂表格处理
3. 文本清洗和预处理
4. 完善的错误处理
5. 结构化日志记录
"""

import logging
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import fitz  # PyMuPDF
import pandas as pd
from dataclasses import dataclass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DocumentSection:
    """文档段落结构"""
    content: str
    page_num: int
    section_type: str  # 'text', 'table', 'header', 'footer'
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    confidence: float = 1.0

class DocumentPreprocessor:
    """文档预处理器"""
    
    def __init__(self):
        self.text_patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}-\d{3}-\d{4}\b|\b\d{10}\b'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'extra_spaces': re.compile(r'\s+'),
            'line_breaks': re.compile(r'\n+'),
        }
    
    def clean_text(self, text: str) -> str:
        """文本清洗"""
        if not text:
            return ""
        
        # 移除多余空格和换行
        text = self.text_patterns['extra_spaces'].sub(' ', text)
        text = self.text_patterns['line_breaks'].sub('\n', text)
        
        # 移除特殊字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        # 标准化引号
        text = text.replace('"', '\"').replace('"', '\"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def normalize_encoding(self, text: str) -> str:
        """编码标准化"""
        try:
            # 尝试UTF-8编码
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')
            return text
        except Exception as e:
            logger.warning(f"编码标准化失败: {e}")
            return str(text)
    
    def extract_metadata(self, text: str) -> Dict[str, List[str]]:
        """提取文档元数据"""
        metadata = {
            'emails': self.text_patterns['email'].findall(text),
            'phones': self.text_patterns['phone'].findall(text),
            'urls': self.text_patterns['url'].findall(text)
        }
        return metadata

class EnhancedPDFParser:
    """增强PDF解析器"""
    
    def __init__(self):
        self.preprocessor = DocumentPreprocessor()
        self.min_text_length = 10  # 最小文本长度
        self.table_detection_threshold = 0.7  # 表格检测阈值
    
    def extract_with_layout(self, file_path: str) -> List[DocumentSection]:
        """布局感知的PDF解析"""
        sections = []
        
        try:
            doc = fitz.open(file_path)
            logger.info(f"开始解析PDF: {file_path}, 共{len(doc)}页")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_sections = self._extract_page_sections(page, page_num)
                sections.extend(page_sections)
            
            doc.close()
            logger.info(f"PDF解析完成，共提取{len(sections)}个段落")
            
        except Exception as e:
            logger.error(f"PDF解析失败: {e}")
            raise
        
        return sections
    
    def _extract_page_sections(self, page, page_num: int) -> List[DocumentSection]:
        """提取页面段落"""
        sections = []
        
        # 获取文本块
        text_blocks = page.get_text("dict")
        
        for block in text_blocks["blocks"]:
            if "lines" in block:
                # 文本块
                text_content = self._extract_block_text(block)
                if len(text_content) >= self.min_text_length:
                    section = DocumentSection(
                        content=self.preprocessor.clean_text(text_content),
                        page_num=page_num + 1,
                        section_type="text",
                        bbox=tuple(block["bbox"]),
                        confidence=0.9
                    )
                    sections.append(section)
        
        # 检测和提取表格
        tables = self._detect_tables(page, page_num)
        sections.extend(tables)
        
        return sections"}]}}}