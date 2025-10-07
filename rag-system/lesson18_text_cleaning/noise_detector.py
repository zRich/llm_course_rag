#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
噪声检测模块
检测文本中的各种噪声类型，适合45分钟课程演示
"""

import re
import string
from typing import List, Dict, Any, Tuple
from collections import Counter


class NoiseDetector:
    """噪声检测器
    
    检测文本中的各种噪声：
    - 乱码字符
    - 重复内容
    - 格式问题
    - 无意义字符
    """
    
    def __init__(self):
        # 常见乱码模式
        self.garbled_patterns = [
            r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]',  # 控制字符
            r'[\uFFFD\uFEFF]',  # 替换字符和BOM
            r'â€™|â€œ|â€\x9d|â€¦',  # 常见编码错误
            r'\?{2,}',  # 连续问号（通常是乱码）
        ]
        
        # 无意义字符模式
        self.meaningless_patterns = [
            r'^[\s\-_=+*#@!~`]{3,}$',  # 纯符号行
            r'^[0-9\s\-.,]{10,}$',  # 纯数字序列
            r'[a-zA-Z]{1}\s[a-zA-Z]{1}\s[a-zA-Z]{1}',  # 单字母间隔
        ]
        
        # 重复模式阈值
        self.repeat_threshold = 3
        self.similarity_threshold = 0.8
    
    def detect_garbled_text(self, text: str) -> Dict[str, Any]:
        """检测乱码文本
        
        Args:
            text: 输入文本
            
        Returns:
            检测结果字典
        """
        if not text:
            return {'has_garbled': False, 'garbled_count': 0, 'patterns': []}
        
        garbled_matches = []
        total_garbled = 0
        
        for pattern in self.garbled_patterns:
            matches = re.findall(pattern, text)
            if matches:
                garbled_matches.append({
                    'pattern': pattern,
                    'matches': matches,
                    'count': len(matches)
                })
                total_garbled += len(matches)
        
        return {
            'has_garbled': total_garbled > 0,
            'garbled_count': total_garbled,
            'patterns': garbled_matches,
            'garbled_ratio': total_garbled / len(text) if text else 0
        }
    
    def detect_repeated_content(self, text: str, min_length: int = 10) -> Dict[str, Any]:
        """检测重复内容
        
        Args:
            text: 输入文本
            min_length: 最小重复长度
            
        Returns:
            检测结果字典
        """
        if not text or len(text) < min_length * 2:
            return {'has_repeats': False, 'repeats': [], 'repeat_ratio': 0}
        
        repeats = []
        text_length = len(text)
        
        # 检测连续重复的子字符串
        for length in range(min_length, text_length // 2 + 1):
            for start in range(text_length - length * 2 + 1):
                substring = text[start:start + length]
                next_substring = text[start + length:start + length * 2]
                
                if substring == next_substring:
                    # 检查重复次数
                    repeat_count = 1
                    pos = start + length * 2
                    
                    while pos + length <= text_length:
                        if text[pos:pos + length] == substring:
                            repeat_count += 1
                            pos += length
                        else:
                            break
                    
                    if repeat_count >= self.repeat_threshold:
                        repeats.append({
                            'content': substring,
                            'start_pos': start,
                            'repeat_count': repeat_count,
                            'total_length': length * repeat_count
                        })
        
        # 计算重复内容比例
        total_repeat_length = sum(r['total_length'] for r in repeats)
        repeat_ratio = total_repeat_length / text_length if text_length > 0 else 0
        
        return {
            'has_repeats': len(repeats) > 0,
            'repeats': repeats,
            'repeat_ratio': repeat_ratio
        }
    
    def detect_meaningless_lines(self, text: str) -> Dict[str, Any]:
        """检测无意义行
        
        Args:
            text: 输入文本
            
        Returns:
            检测结果字典
        """
        if not text:
            return {'has_meaningless': False, 'meaningless_lines': [], 'meaningless_ratio': 0}
        
        lines = text.split('\n')
        meaningless_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:  # 空行
                continue
            
            # 检查是否匹配无意义模式
            for pattern in self.meaningless_patterns:
                if re.match(pattern, line):
                    meaningless_lines.append({
                        'line_number': i + 1,
                        'content': line,
                        'pattern': pattern
                    })
                    break
        
        meaningless_ratio = len(meaningless_lines) / len(lines) if lines else 0
        
        return {
            'has_meaningless': len(meaningless_lines) > 0,
            'meaningless_lines': meaningless_lines,
            'meaningless_ratio': meaningless_ratio
        }
    
    def detect_encoding_issues(self, text: str) -> Dict[str, Any]:
        """检测编码问题
        
        Args:
            text: 输入文本
            
        Returns:
            检测结果字典
        """
        if not text:
            return {'has_encoding_issues': False, 'issues': []}
        
        issues = []
        
        # 检测常见编码问题
        encoding_problems = [
            (r'\\x[0-9a-fA-F]{2}', '十六进制转义序列'),
            (r'\\u[0-9a-fA-F]{4}', 'Unicode转义序列'),
            (r'[\x80-\xFF]{2,}', '可能的字节序列'),
            (r'\?{3,}', '连续问号（解码失败）')
        ]
        
        for pattern, description in encoding_problems:
            matches = re.findall(pattern, text)
            if matches:
                issues.append({
                    'type': description,
                    'pattern': pattern,
                    'matches': matches,
                    'count': len(matches)
                })
        
        return {
            'has_encoding_issues': len(issues) > 0,
            'issues': issues
        }
    
    def detect_format_noise(self, text: str) -> Dict[str, Any]:
        """检测格式噪声
        
        Args:
            text: 输入文本
            
        Returns:
            检测结果字典
        """
        if not text:
            return {'has_format_noise': False, 'noise_types': []}
        
        noise_types = []
        
        # 检测过多的空白字符
        excessive_whitespace = re.findall(r'\s{5,}', text)
        if excessive_whitespace:
            noise_types.append({
                'type': '过多空白字符',
                'count': len(excessive_whitespace),
                'examples': excessive_whitespace[:3]
            })
        
        # 检测过多的换行符
        excessive_newlines = re.findall(r'\n{3,}', text)
        if excessive_newlines:
            noise_types.append({
                'type': '过多换行符',
                'count': len(excessive_newlines),
                'examples': [f'{len(match)}个连续换行' for match in excessive_newlines[:3]]
            })
        
        # 检测混乱的标点符号
        messy_punctuation = re.findall(r'[.,!?;:]{3,}', text)
        if messy_punctuation:
            noise_types.append({
                'type': '混乱标点符号',
                'count': len(messy_punctuation),
                'examples': messy_punctuation[:3]
            })
        
        return {
            'has_format_noise': len(noise_types) > 0,
            'noise_types': noise_types
        }
    
    def comprehensive_noise_detection(self, text: str) -> Dict[str, Any]:
        """综合噪声检测
        
        Args:
            text: 输入文本
            
        Returns:
            综合检测结果
        """
        if not text:
            return {
                'text_length': 0,
                'has_noise': False,
                'noise_score': 0.0,
                'detections': {}
            }
        
        # 执行各种检测
        detections = {
            'garbled': self.detect_garbled_text(text),
            'repeated': self.detect_repeated_content(text),
            'meaningless': self.detect_meaningless_lines(text),
            'encoding': self.detect_encoding_issues(text),
            'format': self.detect_format_noise(text)
        }
        
        # 计算噪声评分
        noise_score = 0.0
        weights = {
            'garbled': 0.3,
            'repeated': 0.25,
            'meaningless': 0.2,
            'encoding': 0.15,
            'format': 0.1
        }
        
        for detection_type, result in detections.items():
            if detection_type == 'garbled' and result['has_garbled']:
                noise_score += weights[detection_type] * min(result['garbled_ratio'] * 10, 1.0)
            elif detection_type == 'repeated' and result['has_repeats']:
                noise_score += weights[detection_type] * min(result['repeat_ratio'] * 2, 1.0)
            elif detection_type == 'meaningless' and result['has_meaningless']:
                noise_score += weights[detection_type] * min(result['meaningless_ratio'] * 3, 1.0)
            elif detection_type == 'encoding' and result['has_encoding_issues']:
                noise_score += weights[detection_type] * min(len(result['issues']) * 0.2, 1.0)
            elif detection_type == 'format' and result['has_format_noise']:
                noise_score += weights[detection_type] * min(len(result['noise_types']) * 0.3, 1.0)
        
        # 检查是否有噪声
        has_noise = any([
            detections['garbled']['has_garbled'],
            detections['repeated']['has_repeats'],
            detections['meaningless']['has_meaningless'],
            detections['encoding']['has_encoding_issues'],
            detections['format']['has_format_noise']
        ])
        
        return {
            'text_length': len(text),
            'has_noise': has_noise,
            'noise_score': min(noise_score, 1.0),  # 限制在0-1之间
            'detections': detections
        }
    
    def get_noise_summary(self, detection_result: Dict[str, Any]) -> str:
        """获取噪声检测摘要
        
        Args:
            detection_result: 检测结果
            
        Returns:
            摘要字符串
        """
        if not detection_result['has_noise']:
            return "文本质量良好，未检测到明显噪声。"
        
        summary_parts = []
        detections = detection_result['detections']
        
        if detections['garbled']['has_garbled']:
            summary_parts.append(f"检测到{detections['garbled']['garbled_count']}个乱码字符")
        
        if detections['repeated']['has_repeats']:
            summary_parts.append(f"检测到{len(detections['repeated']['repeats'])}处重复内容")
        
        if detections['meaningless']['has_meaningless']:
            summary_parts.append(f"检测到{len(detections['meaningless']['meaningless_lines'])}行无意义内容")
        
        if detections['encoding']['has_encoding_issues']:
            summary_parts.append(f"检测到{len(detections['encoding']['issues'])}个编码问题")
        
        if detections['format']['has_format_noise']:
            summary_parts.append(f"检测到{len(detections['format']['noise_types'])}种格式噪声")
        
        noise_level = "低" if detection_result['noise_score'] < 0.3 else "中" if detection_result['noise_score'] < 0.7 else "高"
        
        summary = f"噪声等级：{noise_level}（{detection_result['noise_score']:.2f}）。" + "；".join(summary_parts) + "。"
        
        return summary


if __name__ == "__main__":
    # 简单测试
    detector = NoiseDetector()
    
    test_text = """
    这是正常文本。
    这是正常文本。这是正常文本。这是正常文本。
    \x00\x01乱码字符测试
    !!!!!!!!!!!!
    â€™â€œâ€\x9d编码错误测试
    123456789012345678901234567890
    """
    
    print("测试文本:")
    print(repr(test_text))
    print("\n噪声检测结果:")
    result = detector.comprehensive_noise_detection(test_text)
    print(f"噪声评分: {result['noise_score']:.2f}")
    print(f"摘要: {detector.get_noise_summary(result)}")