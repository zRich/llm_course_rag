"""
TXT文本解析服务
实现TXT文档的文本提取和预处理功能
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import re
import chardet

from src.config.settings import settings

logger = logging.getLogger(__name__)


class TXTParser:
    """TXT解析器"""
    
    def __init__(self):
        self.max_file_size = settings.max_file_size  # 已经是字节单位
        
    def validate_txt(self, file_path: Path) -> Tuple[bool, str]:
        """
        验证TXT文件
        
        Args:
            file_path: TXT文件路径
            
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
            if file_path.suffix.lower() != '.txt':
                return False, "文件格式不是TXT"
                
            # 尝试读取文件内容（检测编码）
            try:
                with open(file_path, 'rb') as file:
                    raw_data = file.read(1024)  # 读取前1KB检测编码
                    if not raw_data:
                        return False, "文件为空"
                    
                    # 检测编码
                    encoding_result = chardet.detect(raw_data)
                    if encoding_result['confidence'] < 0.7:
                        logger.warning(f"文件编码检测置信度较低: {encoding_result}")
                        
            except Exception as e:
                return False, f"TXT文件无法读取: {str(e)}"
                    
            return True, ""
            
        except Exception as e:
            logger.error(f"TXT验证失败: {e}")
            return False, f"验证过程出错: {str(e)}"
    
    def detect_encoding(self, file_path: Path) -> str:
        """
        检测文件编码
        
        Args:
            file_path: 文件路径
            
        Returns:
            编码名称
        """
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                logger.debug(f"检测到编码: {encoding}, 置信度: {confidence}")
                
                # 如果置信度太低，使用默认编码
                if confidence < 0.7:
                    logger.warning(f"编码检测置信度较低({confidence})，使用UTF-8")
                    return 'utf-8'
                    
                return encoding or 'utf-8'
                
        except Exception as e:
            logger.error(f"编码检测失败: {e}")
            return 'utf-8'
    
    def extract_text(self, file_path: Path) -> Tuple[str, Dict]:
        """
        提取TXT文件文本内容
        
        Args:
            file_path: TXT文件路径
            
        Returns:
            (文本内容, 元数据)
        """
        logger.info(f"开始提取TXT文件内容: {file_path}")
        
        try:
            # 检测编码
            encoding = self.detect_encoding(file_path)
            
            # 读取文件内容
            with open(file_path, 'r', encoding=encoding, errors='replace') as file:
                text_content = file.read()
            
            # 统计信息
            file_size = file_path.stat().st_size
            line_count = len(text_content.splitlines())
            char_count = len(text_content)
            
            # 构建元数据
            metadata = {
                "extraction_method": "txt_parser",
                "file_size": file_size,
                "encoding": encoding,
                "line_count": line_count,
                "char_count": char_count,
                "has_content": bool(text_content.strip()),
                "extraction_time": None  # 可以添加时间戳
            }
            
            logger.info(f"TXT文件内容提取完成: {file_path.name}, "
                       f"行数: {line_count}, 字符数: {char_count}")
            
            return text_content, metadata
            
        except Exception as e:
            logger.error(f"TXT文件内容提取失败: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        清理文本内容
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\r\n', '\n', text)  # 统一换行符
        text = re.sub(r'\r', '\n', text)    # 处理Mac格式换行符
        text = re.sub(r'\n{3,}', '\n\n', text)  # 限制连续空行
        text = re.sub(r'[ \t]+', ' ', text)  # 合并多个空格和制表符
        text = re.sub(r' +\n', '\n', text)   # 移除行尾空格
        
        # 移除文件开头和结尾的空白
        text = text.strip()
        
        return text
    
    def calculate_text_hash(self, text: str) -> str:
        """
        计算文本内容的哈希值
        
        Args:
            text: 文本内容
            
        Returns:
            SHA256哈希值
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def parse_txt(self, file_path: Path) -> Dict:
        """
        解析TXT文件
        
        Args:
            file_path: TXT文件路径
            
        Returns:
            解析结果字典
        """
        logger.info(f"开始解析TXT文件: {file_path}")
        
        # 验证TXT文件
        is_valid, error_msg = self.validate_txt(file_path)
        if not is_valid:
            raise ValueError(f"TXT文件验证失败: {error_msg}")
        
        try:
            # 提取文本内容
            raw_text, metadata = self.extract_text(file_path)
            
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
                "extraction_method": "txt_parser"
            }
            
            logger.info(f"TXT解析完成: {file_path.name}, "
                       f"行数: {metadata.get('line_count', 0)}, "
                       f"字符数: {char_count}, "
                       f"预估tokens: {estimated_tokens}")
            
            return result
            
        except Exception as e:
            logger.error(f"TXT解析失败: {file_path}, 错误: {e}")
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