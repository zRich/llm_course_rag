"""文档解析模块

提供各种文档格式的解析功能，包括PDF、Word、文本等格式的解析器。
"""

from .parser import DocumentParser, ParsedDocument, DocumentMetadata
from .pdf_parser import PDFParser
from .docx_parser import DocxParser
from .txt_parser import TxtParser
from .document_manager import DocumentManager, document_manager
from .chunker import TextChunker

__all__ = [
    'DocumentParser',
    'ParsedDocument', 
    'DocumentMetadata',
    'PDFParser',
    'DocxParser',
    'TxtParser',
    'DocumentManager',
    'document_manager',
    'TextChunker'
]