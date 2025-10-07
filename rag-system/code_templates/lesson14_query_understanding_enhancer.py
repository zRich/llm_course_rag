#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lesson14 查询理解增强器实现模板
解决查询扩展、意图识别和上下文理解缺失问题

功能特性：
1. 查询意图识别和分类
2. 查询扩展和重写
3. 上下文理解和会话管理
4. 多语言查询处理
5. 查询质量评估和优化
"""

import logging
import re
import json
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
from collections import defaultdict, deque
import threading

# NLP相关库
try:
    import spacy
except ImportError:
    spacy = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
except ImportError:
    pipeline = None
    AutoTokenizer = None
    AutoModel = None

try:
    import jieba
    import jieba.analyse
except ImportError:
    jieba = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """查询意图枚举"""
    FACTUAL = "factual"  # 事实性查询
    PROCEDURAL = "procedural"  # 程序性查询
    CONCEPTUAL = "conceptual"  # 概念性查询
    COMPARATIVE = "comparative"  # 比较性查询
    CAUSAL = "causal"  # 因果性查询
    TEMPORAL = "temporal"  # 时间性查询
    SPATIAL = "spatial"  # 空间性查询
    PERSONAL = "personal"  # 个人化查询
    UNKNOWN = "unknown"  # 未知意图

class QueryType(Enum):
    """查询类型枚举"""
    QUESTION = "question"  # 问题
    COMMAND = "command"  # 命令
    STATEMENT = "statement"  # 陈述
    KEYWORD = "keyword"  # 关键词

@dataclass
class QueryAnalysis:
    """查询分析结果"""
    original_query: str
    intent: QueryIntent
    query_type: QueryType
    entities: List[Dict] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    language: str = "zh"
    confidence: float = 0.0
    complexity: float = 0.0
    ambiguity: float = 0.0
    temporal_info: Dict = field(default_factory=dict)
    spatial_info: Dict = field(default_factory=dict)
    
@dataclass
class ExpandedQuery:
    """扩展查询结果"""
    original_query: str
    expanded_queries: List[str] = field(default_factory=list)
    synonyms: Dict[str, List[str]] = field(default_factory=dict)
    related_terms: List[str] = field(default_factory=list)
    reformulated_queries: List[str] = field(default_factory=list)
    weighted_terms: Dict[str, float] = field(default_factory=dict)

@dataclass
class ConversationContext:
    """会话上下文"""
    session_id: str
    user_id: str
    conversation_history: List[Dict] = field(default_factory=list)
    current_topic: str = ""
    user_preferences: Dict = field(default_factory=dict)
    context_entities: List[Dict] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

class EntityExtractor:
    """实体提取器"""
    
    def __init__(self):
        self.nlp = None
        self._load_models()
        
        # 预定义实体模式
        self.entity_patterns = {
            'date': [
                r'\d{4}年\d{1,2}月\d{1,2}日',
                r'\d{4}-\d{1,2}-\d{1,2}',
                r'\d{1,2}月\d{1,2}日',
                r'今天|明天|昨天|前天|后天'
            ],
            'time': [
                r'\d{1,2}点\d{1,2}分',
                r'\d{1,2}:\d{1,2}',
                r'上午|下午|晚上|凌晨'
            ],
            'number': [
                r'\d+\.\d+',
                r'\d+',
                r'[一二三四五六七八九十百千万亿]+'
            ],
            'location': [
                r'[北京|上海|广州|深圳|杭州|南京|武汉|成都|重庆|天津|青岛|大连|厦门|苏州|无锡|宁波|长沙|郑州|西安|沈阳|哈尔滨|长春|石家庄|太原|呼和浩特|兰州|西宁|银川|乌鲁木齐|拉萨|昆明|贵阳|南宁|海口|福州|南昌|合肥|济南]',
                r'\w+市|\w+省|\w+区|\w+县'
            ]
        }
    
    def _load_models(self):
        """加载NLP模型"""
        try:
            if spacy:
                # 尝试加载中文模型
                try:
                    self.nlp = spacy.load("zh_core_web_sm")
                except OSError:
                    logger.warning("未找到中文spaCy模型，使用英文模型")
                    try:
                        self.nlp = spacy.load("en_core_web_sm")
                    except OSError:
                        logger.warning("未找到spaCy模型")
        except Exception as e:
            logger.error(f"加载NLP模型失败: {e}")
    
    def extract_entities(self, text: str) -> List[Dict]:
        """提取实体"""
        entities = []
        
        # 使用spaCy提取实体
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.8
                })
        
        # 使用正则表达式提取实体
        regex_entities = self._extract_with_regex(text)
        entities.extend(regex_entities)
        
        # 去重和合并
        entities = self._merge_entities(entities)
        
        return entities
    
    def _extract_with_regex(self, text: str) -> List[Dict]:
        """使用正则表达式提取实体"""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entities.append({
                        'text': match.group(),
                        'label': entity_type,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.6
                    })
        
        return entities
    
    def _merge_entities(self, entities: List[Dict]) -> List[Dict]:
        """合并重叠的实体"""
        if not entities:
            return entities
        
        # 按位置排序
        entities.sort(key=lambda x: (x['start'], x['end']))
        
        merged = [entities[0]]
        
        for entity in entities[1:]:
            last = merged[-1]
            
            # 检查是否重叠
            if entity['start'] < last['end']:
                # 选择置信度更高的实体
                if entity['confidence'] > last['confidence']:
                    merged[-1] = entity
            else:
                merged.append(entity)
        
        return merged

class IntentClassifier:
    """意图分类器"""
    
    def __init__(self):
        self.intent_patterns = {
            QueryIntent.FACTUAL: [
                r'什么是|是什么|定义|含义',
                r'who|what|where|when',
                r'介绍|解释|说明'
            ],
            QueryIntent.PROCEDURAL: [
                r'如何|怎么|怎样|方法|步骤',
                r'how to|how can|steps',
                r'教程|指南|操作'
            ],
            QueryIntent.CONCEPTUAL: [
                r'为什么|原理|机制|概念',
                r'why|principle|concept',
                r'理论|思想|观点'
            ],
            QueryIntent.COMPARATIVE: [
                r'比较|对比|区别|差异',
                r'compare|difference|versus',
                r'哪个更好|优缺点'
            ],
            QueryIntent.CAUSAL: [
                r'因为|由于|导致|原因|结果',
                r'because|cause|result|lead to',
                r'影响|作用|效果'
            ],
            QueryIntent.TEMPORAL: [
                r'什么时候|何时|时间|日期',
                r'when|time|date|schedule',
                r'历史|发展|演变'
            ],
            QueryIntent.SPATIAL: [
                r'在哪里|位置|地点|地址',
                r'where|location|place|address',
                r'分布|范围|区域'
            ],
            QueryIntent.PERSONAL: [
                r'我的|个人|推荐|建议',
                r'my|personal|recommend|suggest',
                r'适合|喜欢|偏好'
            ]
        }
        
        self.question_words = {
            'zh': ['什么', '为什么', '如何', '怎么', '哪个', '谁', '何时', '在哪'],
            'en': ['what', 'why', 'how', 'which', 'who', 'when', 'where']
        }
    
    def classify_intent(self, query: str, language: str = 'zh') -> Tuple[QueryIntent, float]:
        """分类查询意图"""
        query_lower = query.lower()
        intent_scores = defaultdict(float)
        
        # 基于模式匹配计算分数
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    intent_scores[intent] += 1.0
        
        # 基于问词判断
        question_words = self.question_words.get(language, [])
        for word in question_words:
            if word in query_lower:
                if word in ['什么', 'what']:
                    intent_scores[QueryIntent.FACTUAL] += 0.8
                elif word in ['为什么', 'why']:
                    intent_scores[QueryIntent.CAUSAL] += 0.8
                elif word in ['如何', '怎么', 'how']:
                    intent_scores[QueryIntent.PROCEDURAL] += 0.8
                elif word in ['何时', 'when']:
                    intent_scores[QueryIntent.TEMPORAL] += 0.8
                elif word in ['在哪', 'where']:
                    intent_scores[QueryIntent.SPATIAL] += 0.8
        
        # 返回最高分数的意图
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            confidence = min(best_intent[1] / 2.0, 1.0)  # 归一化置信度
            return best_intent[0], confidence
        
        return QueryIntent.UNKNOWN, 0.0
    
    def classify_query_type(self, query: str) -> QueryType:
        """分类查询类型"""
        query = query.strip()
        
        # 检查是否为问句
        if query.endswith('?') or query.endswith('？'):
            return QueryType.QUESTION
        
        # 检查是否包含问词
        question_indicators = ['什么', '为什么', '如何', '怎么', '哪个', '谁', '何时', '在哪',
                             'what', 'why', 'how', 'which', 'who', 'when', 'where']
        for indicator in question_indicators:
            if indicator in query.lower():
                return QueryType.QUESTION
        
        # 检查是否为命令
        command_indicators = ['请', '帮我', '给我', '显示', '查找', '搜索',
                            'please', 'help', 'show', 'find', 'search']
        for indicator in command_indicators:
            if query.lower().startswith(indicator):
                return QueryType.COMMAND
        
        # 检查是否为关键词（短且无动词）
        if len(query.split()) <= 3 and not any(word in query for word in ['是', '有', '能', 'is', 'are', 'can']):
            return QueryType.KEYWORD
        
        return QueryType.STATEMENT

class QueryExpander:
    """查询扩展器"""
    
    def __init__(self):
        self.synonym_dict = self._load_synonyms()
        self.stop_words = self._load_stop_words()
        
    def _load_synonyms(self) -> Dict[str, List[str]]:
        """加载同义词词典"""
        # 简化的同义词词典
        return {
            '机器学习': ['ML', '人工智能', 'AI', '深度学习'],
            '深度学习': ['DL', '神经网络', '机器学习'],
            '自然语言处理': ['NLP', '文本处理', '语言模型'],
            '计算机视觉': ['CV', '图像处理', '视觉识别'],
            '数据科学': ['数据分析', '数据挖掘', '统计分析'],
            '算法': ['方法', '技术', '策略'],
            '模型': ['网络', '架构', '框架'],
            '训练': ['学习', '优化', '拟合'],
            '预测': ['推理', '预估', '判断'],
            '分类': ['分组', '归类', '识别']
        }
    
    def _load_stop_words(self) -> set:
        """加载停用词"""
        chinese_stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '里', '就是', '可以', '什么', '如果', '没', '多', '然后'
        }
        
        english_stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        return chinese_stop_words.union(english_stop_words)
    
    def expand_query(self, query: str, analysis: QueryAnalysis) -> ExpandedQuery:
        """扩展查询"""
        expanded = ExpandedQuery(original_query=query)
        
        # 提取关键词
        keywords = self._extract_keywords(query)
        
        # 生成同义词扩展
        synonyms = self._generate_synonyms(keywords)
        expanded.synonyms = synonyms
        
        # 生成相关词
        related_terms = self._generate_related_terms(keywords, analysis)
        expanded.related_terms = related_terms
        
        # 生成扩展查询
        expanded_queries = self._generate_expanded_queries(query, synonyms, related_terms)
        expanded.expanded_queries = expanded_queries
        
        # 生成重构查询
        reformulated = self._reformulate_queries(query, analysis)
        expanded.reformulated_queries = reformulated
        
        # 计算词权重
        weighted_terms = self._calculate_term_weights(keywords, analysis)
        expanded.weighted_terms = weighted_terms
        
        return expanded
    
    def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        # 使用jieba分词（如果可用）
        if jieba:
            # 提取关键词
            keywords = jieba.analyse.extract_tags(query, topK=10, withWeight=False)
            # 过滤停用词
            keywords = [kw for kw in keywords if kw not in self.stop_words]
        else:
            # 简单的分词
            words = re.findall(r'\b\w+\b', query.lower())
            keywords = [w for w in words if w not in self.stop_words and len(w) > 1]
        
        return keywords
    
    def _generate_synonyms(self, keywords: List[str]) -> Dict[str, List[str]]:
        """生成同义词"""
        synonyms = {}
        
        for keyword in keywords:
            if keyword in self.synonym_dict:
                synonyms[keyword] = self.synonym_dict[keyword]
            else:
                # 简单的同义词生成逻辑
                synonyms[keyword] = []
        
        return synonyms
    
    def _generate_related_terms(self, keywords: List[str], analysis: QueryAnalysis) -> List[str]:
        """生成相关词"""
        related_terms = []
        
        # 基于意图生成相关词
        if analysis.intent == QueryIntent.PROCEDURAL:
            related_terms.extend(['方法', '步骤', '流程', '操作', '指南'])
        elif analysis.intent == QueryIntent.FACTUAL:
            related_terms.extend(['定义', '概念', '介绍', '说明', '解释'])
        elif analysis.intent == QueryIntent.COMPARATIVE:
            related_terms.extend(['比较', '对比', '区别', '优缺点', '差异'])
        
        # 基于实体生成相关词
        for entity in analysis.entities:
            if entity['label'] == 'date':
                related_terms.extend(['时间', '日期', '历史', '发展'])
            elif entity['label'] == 'location':
                related_terms.extend(['地点', '位置', '区域', '分布'])
        
        return list(set(related_terms))  # 去重
    
    def _generate_expanded_queries(self, original_query: str, synonyms: Dict[str, List[str]], 
                                 related_terms: List[str]) -> List[str]:
        """生成扩展查询"""
        expanded_queries = []
        
        # 同义词替换
        for word, syns in synonyms.items():
            for syn in syns[:2]:  # 限制同义词数量
                expanded_query = original_query.replace(word, syn)
                if expanded_query != original_query:
                    expanded_queries.append(expanded_query)
        
        # 添加相关词
        for term in related_terms[:3]:  # 限制相关词数量
            expanded_queries.append(f"{original_query} {term}")
        
        return expanded_queries
    
    def _reformulate_queries(self, query: str, analysis: QueryAnalysis) -> List[str]:
        """重构查询"""
        reformulated = []
        
        # 基于意图重构
        if analysis.intent == QueryIntent.PROCEDURAL:
            if not any(word in query for word in ['如何', '怎么', 'how']):
                reformulated.append(f"如何{query}")
                reformulated.append(f"{query}的方法")
        
        elif analysis.intent == QueryIntent.FACTUAL:
            if not any(word in query for word in ['什么是', 'what is']):
                reformulated.append(f"什么是{query}")
                reformulated.append(f"{query}的定义")
        
        elif analysis.intent == QueryIntent.COMPARATIVE:
            if '和' in query or 'vs' in query.lower():
                parts = re.split(r'和|vs|versus', query, flags=re.IGNORECASE)
                if len(parts) == 2:
                    reformulated.append(f"{parts[0].strip()}与{parts[1].strip()}的区别")
                    reformulated.append(f"比较{parts[0].strip()}和{parts[1].strip()}")
        
        return reformulated
    
    def _calculate_term_weights(self, keywords: List[str], analysis: QueryAnalysis) -> Dict[str, float]:
        """计算词权重"""
        weights = {}
        
        # 基础权重
        for keyword in keywords:
            weights[keyword] = 1.0
        
        # 基于实体调整权重
        for entity in analysis.entities:
            entity_text = entity['text']
            if entity_text in weights:
                weights[entity_text] *= 1.5  # 实体权重加成
        
        # 基于意图调整权重
        intent_keywords = {
            QueryIntent.PROCEDURAL: ['方法', '步骤', '如何', '怎么'],
            QueryIntent.FACTUAL: ['什么', '定义', '概念'],
            QueryIntent.COMPARATIVE: ['比较', '对比', '区别']
        }
        
        if analysis.intent in intent_keywords:
            for keyword in intent_keywords[analysis.intent]:
                if keyword in weights:
                    weights[keyword] *= 1.3
        
        return weights

class ContextManager:
    """上下文管理器"""
    
    def __init__(self, max_history: int = 10):
        self.contexts = {}  # session_id -> ConversationContext
        self.max_history = max_history
        self._lock = threading.RLock()
    
    def get_context(self, session_id: str, user_id: str = None) -> ConversationContext:
        """获取会话上下文"""
        with self._lock:
            if session_id not in self.contexts:
                self.contexts[session_id] = ConversationContext(
                    session_id=session_id,
                    user_id=user_id or "anonymous"
                )
            return self.contexts[session_id]
    
    def update_context(self, session_id: str, query: str, analysis: QueryAnalysis, 
                      response: str = None) -> None:
        """更新会话上下文"""
        with self._lock:
            context = self.get_context(session_id)
            
            # 添加到历史记录
            history_item = {
                'timestamp': datetime.now(),
                'query': query,
                'analysis': analysis,
                'response': response
            }
            context.conversation_history.append(history_item)
            
            # 限制历史记录长度
            if len(context.conversation_history) > self.max_history:
                context.conversation_history = context.conversation_history[-self.max_history:]
            
            # 更新当前主题
            context.current_topic = self._extract_topic(analysis)
            
            # 更新上下文实体
            self._update_context_entities(context, analysis.entities)
            
            context.last_updated = datetime.now()
    
    def _extract_topic(self, analysis: QueryAnalysis) -> str:
        """提取查询主题"""
        if analysis.topics:
            return analysis.topics[0]
        elif analysis.keywords:
            return analysis.keywords[0]
        else:
            return "general"
    
    def _update_context_entities(self, context: ConversationContext, entities: List[Dict]) -> None:
        """更新上下文实体"""
        # 添加新实体
        for entity in entities:
            # 检查是否已存在
            existing = False
            for ctx_entity in context.context_entities:
                if (ctx_entity['text'] == entity['text'] and 
                    ctx_entity['label'] == entity['label']):
                    ctx_entity['count'] = ctx_entity.get('count', 1) + 1
                    existing = True
                    break
            
            if not existing:
                entity_copy = entity.copy()
                entity_copy['count'] = 1
                entity_copy['first_seen'] = datetime.now()
                context.context_entities.append(entity_copy)
        
        # 限制实体数量
        if len(context.context_entities) > 50:
            # 按出现次数排序，保留前50个
            context.context_entities.sort(key=lambda x: x.get('count', 1), reverse=True)
            context.context_entities = context.context_entities[:50]
    
    def get_contextual_keywords(self, session_id: str) -> List[str]:
        """获取上下文关键词"""
        with self._lock:
            if session_id not in self.contexts:
                return []
            
            context = self.contexts[session_id]
            keywords = []
            
            # 从历史查询中提取关键词
            for item in context.conversation_history[-3:]:  # 最近3次对话
                if hasattr(item['analysis'], 'keywords'):
                    keywords.extend(item['analysis'].keywords)
            
            # 从上下文实体中提取
            for entity in context.context_entities:
                if entity.get('count', 1) > 1:  # 多次出现的实体
                    keywords.append(entity['text'])
            
            return list(set(keywords))  # 去重

class QueryUnderstandingEnhancer:
    """查询理解增强器"""
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.intent_classifier = IntentClassifier()
        self.query_expander = QueryExpander()
        self.context_manager = ContextManager()
        self._lock = threading.RLock()
    
    def analyze_query(self, query: str, session_id: str = None, 
                     language: str = 'zh') -> QueryAnalysis:
        """分析查询"""
        with self._lock:
            # 基础分析
            analysis = QueryAnalysis(original_query=query, language=language)
            
            # 实体提取
            analysis.entities = self.entity_extractor.extract_entities(query)
            
            # 意图分类
            analysis.intent, analysis.confidence = self.intent_classifier.classify_intent(query, language)
            
            # 查询类型分类
            analysis.query_type = self.intent_classifier.classify_query_type(query)
            
            # 关键词提取
            analysis.keywords = self.query_expander._extract_keywords(query)
            
            # 复杂度评估
            analysis.complexity = self._assess_complexity(query, analysis)
            
            # 歧义性评估
            analysis.ambiguity = self._assess_ambiguity(query, analysis)
            
            # 时间信息提取
            analysis.temporal_info = self._extract_temporal_info(analysis.entities)
            
            # 空间信息提取
            analysis.spatial_info = self._extract_spatial_info(analysis.entities)
            
            # 主题提取
            analysis.topics = self._extract_topics(query, analysis)
            
            # 更新上下文
            if session_id:
                self.context_manager.update_context(session_id, query, analysis)
            
            return analysis
    
    def enhance_query(self, query: str, session_id: str = None, 
                     language: str = 'zh') -> Tuple[QueryAnalysis, ExpandedQuery]:
        """增强查询理解"""
        # 分析查询
        analysis = self.analyze_query(query, session_id, language)
        
        # 扩展查询
        expanded = self.query_expander.expand_query(query, analysis)
        
        # 结合上下文信息
        if session_id:
            contextual_keywords = self.context_manager.get_contextual_keywords(session_id)
            expanded.related_terms.extend(contextual_keywords)
            expanded.related_terms = list(set(expanded.related_terms))  # 去重
        
        return analysis, expanded
    
    def _assess_complexity(self, query: str, analysis: QueryAnalysis) -> float:
        """评估查询复杂度"""
        complexity = 0.0
        
        # 基于长度
        complexity += min(len(query) / 100.0, 0.3)
        
        # 基于实体数量
        complexity += min(len(analysis.entities) / 10.0, 0.2)
        
        # 基于关键词数量
        complexity += min(len(analysis.keywords) / 10.0, 0.2)
        
        # 基于意图类型
        if analysis.intent in [QueryIntent.COMPARATIVE, QueryIntent.CAUSAL]:
            complexity += 0.2
        
        # 基于查询类型
        if analysis.query_type == QueryType.QUESTION:
            complexity += 0.1
        
        return min(complexity, 1.0)
    
    def _assess_ambiguity(self, query: str, analysis: QueryAnalysis) -> float:
        """评估查询歧义性"""
        ambiguity = 0.0
        
        # 基于代词
        pronouns = ['它', '他', '她', '这个', '那个', 'it', 'this', 'that']
        for pronoun in pronouns:
            if pronoun in query.lower():
                ambiguity += 0.2
        
        # 基于模糊词汇
        vague_words = ['一些', '某些', '大概', '可能', 'some', 'maybe', 'probably']
        for word in vague_words:
            if word in query.lower():
                ambiguity += 0.1
        
        # 基于实体歧义
        entity_texts = [e['text'] for e in analysis.entities]
        if len(set(entity_texts)) < len(entity_texts):  # 有重复实体
            ambiguity += 0.1
        
        # 基于意图置信度
        if analysis.confidence < 0.5:
            ambiguity += 0.3
        
        return min(ambiguity, 1.0)
    
    def _extract_temporal_info(self, entities: List[Dict]) -> Dict:
        """提取时间信息"""
        temporal_info = {
            'has_temporal': False,
            'temporal_entities': [],
            'time_range': None,
            'relative_time': None
        }
        
        for entity in entities:
            if entity['label'] in ['date', 'time']:
                temporal_info['has_temporal'] = True
                temporal_info['temporal_entities'].append(entity)
                
                # 检查相对时间
                relative_words = ['今天', '明天', '昨天', '前天', '后天', 'today', 'tomorrow', 'yesterday']
                if any(word in entity['text'] for word in relative_words):
                    temporal_info['relative_time'] = entity['text']
        
        return temporal_info
    
    def _extract_spatial_info(self, entities: List[Dict]) -> Dict:
        """提取空间信息"""
        spatial_info = {
            'has_spatial': False,
            'spatial_entities': [],
            'locations': [],
            'spatial_relations': []
        }
        
        for entity in entities:
            if entity['label'] == 'location':
                spatial_info['has_spatial'] = True
                spatial_info['spatial_entities'].append(entity)
                spatial_info['locations'].append(entity['text'])
        
        return spatial_info
    
    def _extract_topics(self, query: str, analysis: QueryAnalysis) -> List[str]:
        """提取主题"""
        topics = []
        
        # 基于关键词
        if analysis.keywords:
            topics.extend(analysis.keywords[:3])  # 取前3个关键词作为主题
        
        # 基于实体
        for entity in analysis.entities:
            if entity['label'] not in ['date', 'time', 'number']:
                topics.append(entity['text'])
        
        # 去重并限制数量
        topics = list(set(topics))[:5]
        
        return topics
    
    def get_enhancement_report(self, session_id: str = None) -> Dict:
        """获取增强报告"""
        report = {
            'entity_extractor_status': 'active' if self.entity_extractor.nlp else 'limited',
            'supported_languages': ['zh', 'en'],
            'available_intents': [intent.value for intent in QueryIntent],
            'available_query_types': [qtype.value for qtype in QueryType]
        }
        
        if session_id and session_id in self.context_manager.contexts:
            context = self.context_manager.contexts[session_id]
            report['session_info'] = {
                'conversation_length': len(context.conversation_history),
                'current_topic': context.current_topic,
                'context_entities_count': len(context.context_entities),
                'last_updated': context.last_updated.isoformat()
            }
        
        return report

def main():
    """示例用法"""
    # 创建查询理解增强器
    enhancer = QueryUnderstandingEnhancer()
    
    # 测试查询
    test_queries = [
        "什么是机器学习？",
        "如何训练深度学习模型？",
        "比较CNN和RNN的区别",
        "北京有哪些AI公司？",
        "明天的会议在哪里举行？",
        "推荐一些自然语言处理的书籍"
    ]
    
    session_id = "test_session_001"
    
    print("查询理解增强器测试\n" + "="*50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n查询 {i}: {query}")
        print("-" * 30)
        
        # 分析和增强查询
        analysis, expanded = enhancer.enhance_query(query, session_id)
        
        # 显示分析结果
        print(f"意图: {analysis.intent.value} (置信度: {analysis.confidence:.2f})")
        print(f"查询类型: {analysis.query_type.value}")
        print(f"复杂度: {analysis.complexity:.2f}")
        print(f"歧义性: {analysis.ambiguity:.2f}")
        
        if analysis.entities:
            print(f"实体: {[f\"{e['text']}({e['label']})\" for e in analysis.entities]}")
        
        if analysis.keywords:
            print(f"关键词: {analysis.keywords}")
        
        if expanded.expanded_queries:
            print(f"扩展查询: {expanded.expanded_queries[:2]}")
        
        if expanded.related_terms:
            print(f"相关词: {expanded.related_terms[:3]}")
    
    # 显示增强报告
    print(f"\n\n增强报告\n" + "="*30)
    report = enhancer.get_enhancement_report(session_id)
    for key, value in report.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()