from typing import Dict, List, Optional, Type, Union
from pathlib import Path
import logging

from .parser import DocumentParser, ParsedDocument, DocumentMetadata
from .pdf_parser import PDFParser
from .docx_parser import DocxParser
from .txt_parser import TxtParser

logger = logging.getLogger(__name__)

class DocumentManager:
    """文档解析管理器
    
    统一管理所有类型的文档解析器，提供统一的文档解析接口
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._parsers: Dict[str, DocumentParser] = {}
        self._extension_mapping: Dict[str, str] = {}
        
        # 注册默认解析器
        self._register_default_parsers()
    
    def _register_default_parsers(self):
        """注册默认的文档解析器"""
        # 注册PDF解析器
        pdf_parser = PDFParser()
        self.register_parser('pdf', pdf_parser)
        
        # 注册Word解析器
        docx_parser = DocxParser()
        self.register_parser('docx', docx_parser)
        
        # 注册文本解析器
        txt_parser = TxtParser()
        self.register_parser('txt', txt_parser)
    
    def register_parser(self, parser_type: str, parser: DocumentParser):
        """注册文档解析器
        
        Args:
            parser_type: 解析器类型标识
            parser: 解析器实例
        """
        self._parsers[parser_type] = parser
        
        # 更新扩展名映射
        if hasattr(parser, 'SUPPORTED_EXTENSIONS'):
            for ext in parser.SUPPORTED_EXTENSIONS:
                self._extension_mapping[ext.lower()] = parser_type
        
        self.logger.info(f"注册解析器: {parser_type}")
    
    def get_parser(self, file_path: str) -> Optional[DocumentParser]:
        """根据文件路径获取合适的解析器
        
        Args:
            file_path: 文件路径
            
        Returns:
            Optional[DocumentParser]: 解析器实例，如果没有合适的解析器则返回None
        """
        try:
            path = Path(file_path)
            extension = path.suffix.lower()
            
            parser_type = self._extension_mapping.get(extension)
            if parser_type:
                return self._parsers.get(parser_type)
            
            # 尝试通过解析器的can_parse方法查找
            for parser in self._parsers.values():
                if parser.can_parse(file_path):
                    return parser
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取解析器失败 {file_path}: {e}")
            return None
    
    def can_parse(self, file_path: str) -> bool:
        """检查是否可以解析指定文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否可以解析
        """
        return self.get_parser(file_path) is not None
    
    def parse_document(self, file_path: str) -> ParsedDocument:
        """解析文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            ParsedDocument: 解析后的文档数据
            
        Raises:
            ValueError: 不支持的文件格式
            Exception: 解析过程中的错误
        """
        parser = self.get_parser(file_path)
        if not parser:
            raise ValueError(f"不支持的文件格式: {file_path}")
        
        try:
            self.logger.info(f"开始解析文档: {file_path}")
            result = parser.parse(file_path)
            self.logger.info(f"文档解析完成: {file_path}")
            return result
        except Exception as e:
            self.logger.error(f"文档解析失败 {file_path}: {e}")
            raise
    
    def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """提取文档元数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            DocumentMetadata: 文档元数据
            
        Raises:
            ValueError: 不支持的文件格式
            Exception: 提取过程中的错误
        """
        parser = self.get_parser(file_path)
        if not parser:
            raise ValueError(f"不支持的文件格式: {file_path}")
        
        try:
            return parser.extract_metadata(file_path)
        except Exception as e:
            self.logger.error(f"提取元数据失败 {file_path}: {e}")
            raise
    
    def parse_batch(self, file_paths: List[str], 
                   ignore_errors: bool = True) -> List[Optional[ParsedDocument]]:
        """批量解析文档
        
        Args:
            file_paths: 文件路径列表
            ignore_errors: 是否忽略单个文件的解析错误
            
        Returns:
            List[Optional[ParsedDocument]]: 解析结果列表，失败的文件返回None
        """
        results = []
        
        for file_path in file_paths:
            try:
                result = self.parse_document(file_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"批量解析失败 {file_path}: {e}")
                if ignore_errors:
                    results.append(None)
                else:
                    raise
        
        return results
    
    def get_supported_extensions(self) -> List[str]:
        """获取所有支持的文件扩展名
        
        Returns:
            List[str]: 支持的文件扩展名列表
        """
        return list(self._extension_mapping.keys())
    
    def get_parser_info(self) -> Dict[str, Dict[str, any]]:
        """获取所有解析器信息
        
        Returns:
            Dict[str, Dict[str, any]]: 解析器信息
        """
        info = {}
        
        for parser_type, parser in self._parsers.items():
            extensions = []
            if hasattr(parser, 'SUPPORTED_EXTENSIONS'):
                extensions = parser.SUPPORTED_EXTENSIONS
            
            info[parser_type] = {
                'class_name': parser.__class__.__name__,
                'supported_extensions': extensions,
                'description': parser.__class__.__doc__ or ''
            }
        
        return info
    
    def validate_files(self, file_paths: List[str]) -> Dict[str, Dict[str, any]]:
        """验证文件列表
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            Dict[str, Dict[str, any]]: 验证结果
        """
        results = {
            'valid': [],
            'invalid': [],
            'unsupported': [],
            'summary': {
                'total': len(file_paths),
                'valid': 0,
                'invalid': 0,
                'unsupported': 0
            }
        }
        
        for file_path in file_paths:
            try:
                # 检查文件是否存在
                path = Path(file_path)
                if not path.exists():
                    results['invalid'].append({
                        'file_path': file_path,
                        'error': '文件不存在'
                    })
                    continue
                
                if not path.is_file():
                    results['invalid'].append({
                        'file_path': file_path,
                        'error': '不是文件'
                    })
                    continue
                
                # 检查是否支持解析
                parser = self.get_parser(file_path)
                if not parser:
                    results['unsupported'].append({
                        'file_path': file_path,
                        'extension': path.suffix.lower()
                    })
                    continue
                
                # 验证文件
                parser.validate_file(file_path)
                results['valid'].append({
                    'file_path': file_path,
                    'parser_type': type(parser).__name__,
                    'file_size': path.stat().st_size
                })
                
            except Exception as e:
                results['invalid'].append({
                    'file_path': file_path,
                    'error': str(e)
                })
        
        # 更新统计信息
        results['summary']['valid'] = len(results['valid'])
        results['summary']['invalid'] = len(results['invalid'])
        results['summary']['unsupported'] = len(results['unsupported'])
        
        return results
    
    def find_documents(self, directory: str, 
                      recursive: bool = True,
                      include_extensions: Optional[List[str]] = None) -> List[str]:
        """在目录中查找支持的文档文件
        
        Args:
            directory: 搜索目录
            recursive: 是否递归搜索子目录
            include_extensions: 包含的文件扩展名，None表示所有支持的扩展名
            
        Returns:
            List[str]: 找到的文档文件路径列表
        """
        try:
            directory_path = Path(directory)
            if not directory_path.exists() or not directory_path.is_dir():
                raise ValueError(f"目录不存在或不是目录: {directory}")
            
            # 确定要搜索的扩展名
            if include_extensions:
                extensions = [ext.lower() for ext in include_extensions]
            else:
                extensions = self.get_supported_extensions()
            
            found_files = []
            
            # 搜索文件
            pattern = '**/*' if recursive else '*'
            for file_path in directory_path.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in extensions:
                    found_files.append(str(file_path))
            
            self.logger.info(f"在目录 {directory} 中找到 {len(found_files)} 个文档文件")
            return sorted(found_files)
            
        except Exception as e:
            self.logger.error(f"搜索文档文件失败 {directory}: {e}")
            raise

# 创建全局文档管理器实例
document_manager = DocumentManager()