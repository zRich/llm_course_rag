#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本清洗模块
提供基础的文本清洗功能，适合45分钟课程演示
"""

import re
import unicodedata
from typing import List, Dict, Any


class TextCleaner:
    """文本清洗器
    
    提供常用的文本清洗功能：
    - 去除多余空格
    - 清理特殊字符
    - 标准化文本格式
    """
    
    def __init__(self):
        # 常见的噪声字符模式
        self.noise_patterns = {
            'extra_spaces': r'\s+',  # 多余空格
            'special_chars': r'[^\w\s\u4e00-\u9fff.,!?;:()\[\]{}"\'-]',  # 特殊字符（保留中文）
            'repeated_punctuation': r'([.,!?;:]){2,}',  # 重复标点
            'html_tags': r'<[^>]+>',  # HTML标签
            'urls': r'https?://[^\s]+',  # URL链接
            'emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # 邮箱
        }
    
    def clean_whitespace(self, text: str) -> str:
        """清理多余空格
        
        Args:
            text: 输入文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        # 替换多个空格为单个空格
        text = re.sub(self.noise_patterns['extra_spaces'], ' ', text)
        # 去除首尾空格
        text = text.strip()
        
        return text
    
    def remove_special_chars(self, text: str, keep_punctuation: bool = True) -> str:
        """移除特殊字符
        
        Args:
            text: 输入文本
            keep_punctuation: 是否保留标点符号
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        if keep_punctuation:
            # 只移除非常规字符，保留标点
            text = re.sub(self.noise_patterns['special_chars'], '', text)
        else:
            # 只保留字母、数字、中文和空格
            text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        
        return text
    
    def clean_repeated_punctuation(self, text: str) -> str:
        """清理重复标点符号
        
        Args:
            text: 输入文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        # 将重复标点替换为单个
        text = re.sub(self.noise_patterns['repeated_punctuation'], r'\1', text)
        
        return text
    
    def remove_html_tags(self, text: str) -> str:
        """移除HTML标签
        
        Args:
            text: 输入文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        text = re.sub(self.noise_patterns['html_tags'], '', text)
        
        return text
    
    def remove_urls_and_emails(self, text: str) -> str:
        """移除URL和邮箱地址
        
        Args:
            text: 输入文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        # 移除URL
        text = re.sub(self.noise_patterns['urls'], '', text)
        # 移除邮箱
        text = re.sub(self.noise_patterns['emails'], '', text)
        
        return text
    
    def normalize_unicode(self, text: str) -> str:
        """Unicode标准化
        
        Args:
            text: 输入文本
            
        Returns:
            标准化后的文本
        """
        if not text:
            return ""
        
        # 使用NFKC标准化（兼容性分解后重组）
        text = unicodedata.normalize('NFKC', text)
        
        return text
    
    def basic_clean(self, text: str) -> str:
        """基础清洗流程
        
        Args:
            text: 输入文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        # 执行基础清洗步骤
        text = self.normalize_unicode(text)
        text = self.remove_html_tags(text)
        text = self.remove_urls_and_emails(text)
        text = self.clean_repeated_punctuation(text)
        text = self.remove_special_chars(text)
        text = self.clean_whitespace(text)
        
        return text
    
    def advanced_clean(self, text: str, options: Dict[str, Any] = None) -> str:
        """高级清洗流程
        
        Args:
            text: 输入文本
            options: 清洗选项
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        if options is None:
            options = {}
        
        # 默认选项
        default_options = {
            'normalize_unicode': True,
            'remove_html': True,
            'remove_urls': True,
            'clean_punctuation': True,
            'remove_special_chars': True,
            'keep_punctuation': True
        }
        
        # 合并选项
        clean_options = {**default_options, **options}
        
        # 根据选项执行清洗
        if clean_options['normalize_unicode']:
            text = self.normalize_unicode(text)
        
        if clean_options['remove_html']:
            text = self.remove_html_tags(text)
        
        if clean_options['remove_urls']:
            text = self.remove_urls_and_emails(text)
        
        if clean_options['clean_punctuation']:
            text = self.clean_repeated_punctuation(text)
        
        if clean_options['remove_special_chars']:
            text = self.remove_special_chars(text, clean_options['keep_punctuation'])
        
        text = self.clean_whitespace(text)
        
        return text
    
    def batch_clean(self, texts: List[str], method: str = 'basic') -> List[str]:
        """批量清洗文本
        
        Args:
            texts: 文本列表
            method: 清洗方法 ('basic' 或 'advanced')
            
        Returns:
            清洗后的文本列表
        """
        if not texts:
            return []
        
        cleaned_texts = []
        
        for text in texts:
            if method == 'basic':
                cleaned_text = self.basic_clean(text)
            else:
                cleaned_text = self.advanced_clean(text)
            
            cleaned_texts.append(cleaned_text)
        
        return cleaned_texts
    
    def get_cleaning_stats(self, original_text: str, cleaned_text: str) -> Dict[str, Any]:
        """获取清洗统计信息
        
        Args:
            original_text: 原始文本
            cleaned_text: 清洗后文本
            
        Returns:
            统计信息字典
        """
        stats = {
            'original_length': len(original_text) if original_text else 0,
            'cleaned_length': len(cleaned_text) if cleaned_text else 0,
            'chars_removed': 0,
            'reduction_ratio': 0.0
        }
        
        if stats['original_length'] > 0:
            stats['chars_removed'] = stats['original_length'] - stats['cleaned_length']
            stats['reduction_ratio'] = stats['chars_removed'] / stats['original_length']
        
        return stats


if __name__ == "__main__":
    # 简单测试
    cleaner = TextCleaner()
    
    test_text = """
    <p>这是一个测试文本！！！   包含多余空格和HTML标签。</p>
    访问 https://example.com 或联系 test@example.com
    特殊字符：@#$%^&*()_+{}|:<>?[]
    """
    
    print("原始文本:")
    print(repr(test_text))
    print("\n基础清洗结果:")
    cleaned = cleaner.basic_clean(test_text)
    print(repr(cleaned))
    
    print("\n清洗统计:")
    stats = cleaner.get_cleaning_stats(test_text, cleaned)
    print(stats)