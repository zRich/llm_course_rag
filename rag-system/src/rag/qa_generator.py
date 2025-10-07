"""问答生成器模块"""

from typing import List, Dict, Any, Optional
import logging
import json
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class QAResponse:
    """问答响应"""
    answer: str
    confidence: float
    sources: List[str]
    processing_time: float
    metadata: Dict[str, Any]

class QAGenerator:
    """问答生成器
    
    基于检索到的上下文生成答案
    """
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.1,
                 max_tokens: int = 500):
        """
        初始化问答生成器
        
        Args:
            model_name: 使用的模型名称
            temperature: 生成温度
            max_tokens: 最大token数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 简化版本：使用模板生成答案
        self.logger.info(f"问答生成器初始化完成: {model_name}")
    
    def generate_answer(self, 
                       question: str,
                       context: str,
                       conversation_history: Optional[List[Dict[str, str]]] = None) -> QAResponse:
        """
        生成答案
        
        Args:
            question: 用户问题
            context: 检索到的上下文
            conversation_history: 对话历史
            
        Returns:
            问答响应
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"开始生成答案: {question[:50]}...")
            
            # 简化版本：基于模板生成答案
            answer = self._generate_template_answer(question, context)
            
            # 计算置信度（简化版本）
            confidence = self._calculate_confidence(question, context, answer)
            
            # 提取来源
            sources = self._extract_sources(context)
            
            processing_time = time.time() - start_time
            
            response = QAResponse(
                answer=answer,
                confidence=confidence,
                sources=sources,
                processing_time=processing_time,
                metadata={
                    "model_name": self.model_name,
                    "temperature": self.temperature,
                    "context_length": len(context),
                    "question_length": len(question)
                }
            )
            
            self.logger.info(f"答案生成完成: 耗时 {processing_time:.2f}s")
            return response
            
        except Exception as e:
            self.logger.error(f"答案生成失败: {e}")
            raise
    
    def _generate_template_answer(self, question: str, context: str) -> str:
        """
        基于模板生成答案（简化版本）
        
        Args:
            question: 问题
            context: 上下文
            
        Returns:
            生成的答案
        """
        if not context.strip():
            return "抱歉，我没有找到相关的信息来回答您的问题。请尝试重新表述您的问题或提供更多上下文。"
        
        # 简单的关键词匹配和答案生成
        question_lower = question.lower()
        context_lower = context.lower()
        
        # 检查问题类型
        if any(word in question_lower for word in ['什么是', 'what is', '定义', 'definition']):
            answer_template = "根据提供的文档，{concept}是指：\n\n{relevant_info}\n\n以上信息来源于相关文档。"
        elif any(word in question_lower for word in ['如何', 'how to', '怎么', '方法']):
            answer_template = "根据文档内容，关于{topic}的方法如下：\n\n{relevant_info}\n\n请参考上述步骤进行操作。"
        elif any(word in question_lower for word in ['为什么', 'why', '原因']):
            answer_template = "根据相关文档，{topic}的原因包括：\n\n{relevant_info}\n\n以上是基于文档内容的分析。"
        else:
            answer_template = "根据相关文档，关于您的问题：\n\n{relevant_info}\n\n希望这些信息对您有帮助。"
        
        # 提取相关信息（简化版本：取前500字符）
        relevant_info = context[:500] + ("..." if len(context) > 500 else "")
        
        # 提取主题/概念（简化版本）
        topic = self._extract_topic(question)
        
        # 格式化答案
        try:
            if '{concept}' in answer_template:
                answer = answer_template.format(concept=topic, relevant_info=relevant_info)
            elif '{topic}' in answer_template:
                answer = answer_template.format(topic=topic, relevant_info=relevant_info)
            else:
                answer = answer_template.format(relevant_info=relevant_info)
        except:
            answer = f"根据相关文档：\n\n{relevant_info}\n\n希望这些信息对您有帮助。"
        
        return answer
    
    def _extract_topic(self, question: str) -> str:
        """
        从问题中提取主题（简化版本）
        
        Args:
            question: 问题
            
        Returns:
            提取的主题
        """
        # 移除常见的疑问词
        stop_words = ['什么是', 'what is', '如何', 'how to', '怎么', '为什么', 'why', '的', 'is', 'are', '？', '?']
        
        topic = question
        for stop_word in stop_words:
            topic = topic.replace(stop_word, '')
        
        return topic.strip()[:50]  # 限制长度
    
    def _calculate_confidence(self, question: str, context: str, answer: str) -> float:
        """
        计算答案置信度（简化版本）
        
        Args:
            question: 问题
            context: 上下文
            answer: 答案
            
        Returns:
            置信度分数 (0-1)
        """
        try:
            # 简化的置信度计算
            confidence = 0.5  # 基础分数
            
            # 基于上下文长度调整
            if len(context) > 100:
                confidence += 0.2
            if len(context) > 500:
                confidence += 0.1
            
            # 基于关键词匹配调整
            question_words = set(question.lower().split())
            context_words = set(context.lower().split())
            
            if question_words & context_words:  # 有交集
                overlap_ratio = len(question_words & context_words) / len(question_words)
                confidence += overlap_ratio * 0.2
            
            # 确保在0-1范围内
            confidence = max(0.0, min(1.0, confidence))
            
            return confidence
            
        except Exception as e:
            self.logger.warning(f"置信度计算失败: {e}")
            return 0.5
    
    def _extract_sources(self, context: str) -> List[str]:
        """
        从上下文中提取来源信息
        
        Args:
            context: 上下文
            
        Returns:
            来源列表
        """
        sources = []
        
        # 查找文档标记
        import re
        doc_pattern = r'\[文档(\d+)\]'
        matches = re.findall(doc_pattern, context)
        
        for match in matches:
            sources.append(f"文档{match}")
        
        return list(set(sources))  # 去重
    
    def generate_followup_questions(self, 
                                  question: str, 
                                  answer: str, 
                                  context: str) -> List[str]:
        """
        生成后续问题建议
        
        Args:
            question: 原问题
            answer: 生成的答案
            context: 上下文
            
        Returns:
            后续问题列表
        """
        try:
            followup_questions = []
            
            # 基于问题类型生成后续问题
            question_lower = question.lower()
            
            if '什么是' in question_lower or 'what is' in question_lower:
                followup_questions.extend([
                    f"如何使用{self._extract_topic(question)}？",
                    f"{self._extract_topic(question)}有什么优势？",
                    f"{self._extract_topic(question)}的应用场景有哪些？"
                ])
            elif '如何' in question_lower or 'how to' in question_lower:
                followup_questions.extend([
                    "有什么注意事项吗？",
                    "还有其他方法吗？",
                    "这种方法的效果如何？"
                ])
            else:
                followup_questions.extend([
                    "能详细解释一下吗？",
                    "有相关的例子吗？",
                    "还有其他相关信息吗？"
                ])
            
            return followup_questions[:3]  # 返回前3个
            
        except Exception as e:
            self.logger.warning(f"生成后续问题失败: {e}")
            return []
    
    def validate_answer(self, answer: str, context: str) -> Dict[str, Any]:
        """
        验证答案质量
        
        Args:
            answer: 生成的答案
            context: 上下文
            
        Returns:
            验证结果
        """
        validation_result = {
            "is_valid": True,
            "issues": [],
            "suggestions": []
        }
        
        try:
            # 检查答案长度
            if len(answer) < 10:
                validation_result["issues"].append("答案过短")
                validation_result["is_valid"] = False
            
            if len(answer) > 2000:
                validation_result["issues"].append("答案过长")
                validation_result["suggestions"].append("建议缩短答案")
            
            # 检查是否包含上下文信息
            if context and not any(word in answer.lower() for word in context.lower().split()[:10]):
                validation_result["issues"].append("答案与上下文相关性较低")
                validation_result["suggestions"].append("建议增强答案与上下文的关联性")
            
            return validation_result
            
        except Exception as e:
            self.logger.warning(f"答案验证失败: {e}")
            return {"is_valid": True, "issues": [], "suggestions": []}