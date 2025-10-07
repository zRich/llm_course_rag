import re
import math
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class QualityMetrics:
    """文本质量评估指标"""
    readability_score: float  # 可读性评分
    completeness_score: float  # 完整性评分
    consistency_score: float  # 一致性评分
    information_density: float  # 信息密度
    structure_score: float  # 结构评分
    overall_score: float  # 综合评分
    issues: List[str]  # 发现的问题
    suggestions: List[str]  # 改进建议

class QualityEvaluator:
    """文本质量评估器"""
    
    def __init__(self):
        # 常见的停用词（简化版）
        self.stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be'
        }
        
        # 质量评估权重
        self.weights = {
            'readability': 0.25,
            'completeness': 0.20,
            'consistency': 0.20,
            'information_density': 0.20,
            'structure': 0.15
        }
    
    def evaluate(self, text: str) -> QualityMetrics:
        """评估文本质量"""
        if not text or not text.strip():
            return QualityMetrics(
                readability_score=0.0,
                completeness_score=0.0,
                consistency_score=0.0,
                information_density=0.0,
                structure_score=0.0,
                overall_score=0.0,
                issues=['文本为空'],
                suggestions=['请提供有效的文本内容']
            )
        
        # 计算各项指标
        readability = self._calculate_readability(text)
        completeness = self._calculate_completeness(text)
        consistency = self._calculate_consistency(text)
        info_density = self._calculate_information_density(text)
        structure = self._calculate_structure_score(text)
        
        # 计算综合评分
        overall = (
            readability * self.weights['readability'] +
            completeness * self.weights['completeness'] +
            consistency * self.weights['consistency'] +
            info_density * self.weights['information_density'] +
            structure * self.weights['structure']
        )
        
        # 收集问题和建议
        issues, suggestions = self._analyze_issues(text, {
            'readability': readability,
            'completeness': completeness,
            'consistency': consistency,
            'information_density': info_density,
            'structure': structure
        })
        
        return QualityMetrics(
            readability_score=readability,
            completeness_score=completeness,
            consistency_score=consistency,
            information_density=info_density,
            structure_score=structure,
            overall_score=overall,
            issues=issues,
            suggestions=suggestions
        )
    
    def _calculate_readability(self, text: str) -> float:
        """计算可读性评分"""
        sentences = re.split(r'[.!?。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # 平均句子长度
        words = re.findall(r'\b\w+\b', text)
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # 复杂词汇比例（长度>6的词）
        complex_words = [w for w in words if len(w) > 6]
        complex_ratio = len(complex_words) / len(words) if words else 0
        
        # 标点符号使用情况
        punctuation_count = len(re.findall(r'[,.!?;:。，！？；：]', text))
        punctuation_ratio = punctuation_count / len(text) if text else 0
        
        # 可读性评分（基于句子长度和复杂度）
        if avg_sentence_length == 0:
            return 0.0
        
        # 理想句子长度为15-25词
        length_score = 1.0 - abs(avg_sentence_length - 20) / 20
        length_score = max(0, min(1, length_score))
        
        # 复杂词汇不应超过20%
        complexity_score = 1.0 - max(0, complex_ratio - 0.2) * 2
        
        # 标点符号使用合理性
        punct_score = min(1.0, punctuation_ratio * 50)
        
        return (length_score * 0.4 + complexity_score * 0.4 + punct_score * 0.2)
    
    def _calculate_completeness(self, text: str) -> float:
        """计算完整性评分"""
        # 检查文本长度
        length_score = min(1.0, len(text) / 500)  # 500字符为基准
        
        # 检查句子完整性
        sentences = re.split(r'[.!?。！？]', text)
        complete_sentences = [s for s in sentences if s.strip() and len(s.strip()) > 5]
        sentence_score = len(complete_sentences) / max(1, len(sentences))
        
        # 检查段落结构
        paragraphs = text.split('\n')
        paragraph_score = min(1.0, len(paragraphs) / 3)  # 至少3个段落
        
        return (length_score * 0.4 + sentence_score * 0.4 + paragraph_score * 0.2)
    
    def _calculate_consistency(self, text: str) -> float:
        """计算一致性评分"""
        # 检查标点符号一致性
        chinese_punct = len(re.findall(r'[。，！？；：]', text))
        english_punct = len(re.findall(r'[.,:;!?]', text))
        total_punct = chinese_punct + english_punct
        
        punct_consistency = 1.0
        if total_punct > 0:
            # 如果混用中英文标点，降低评分
            if chinese_punct > 0 and english_punct > 0:
                punct_consistency = 0.7
        
        # 简化的引号一致性检查
        quote_consistency = 1.0
        if '"' in text and '