"""
问答服务
实现基于检索增强生成(RAG)的问答功能
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
import os

import openai
from sqlalchemy.orm import Session

from src.models.database import get_db
from src.services.vector_service import VectorService
from src.config.settings import settings

logger = logging.getLogger(__name__)


class QAService:
    """问答服务"""
    
    def __init__(self):
        self.vector_service = VectorService()
        # 使用官方代码推荐的配置
        self.client = openai.OpenAI(
            # 直接使用settings中的火山引擎API Key
            api_key=settings.volcengine_api_key,
            base_url=settings.volcengine_base_url,
            # 深度思考模型耗费时间会较长，设置30分钟超时
            timeout=1800
        )
        
        # 配置参数
        self.model_name = settings.volcengine_model
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature
        self.top_k = settings.top_k
        self.score_threshold = settings.similarity_threshold
        
    def answer_question(self, question: str, 
                       document_ids: List[int] = None,
                       context_size: int = 2,
                       db: Session = None) -> Dict:
        """
        回答问题
        
        Args:
            question: 用户问题
            document_ids: 限制搜索的文档ID列表
            context_size: 上下文大小
            db: 数据库会话
            
        Returns:
            问答结果字典
        """
        if db is None:
            db = next(get_db())
        
        try:
            start_time = datetime.utcnow()
            
            # 1. 检索相关文档片段
            logger.info(f"开始检索相关文档: 问题='{question[:100]}...'")
            
            search_results = self.vector_service.search_similar_chunks(
                query=question,
                limit=self.top_k,
                score_threshold=self.score_threshold,
                document_ids=document_ids,
                db=db
            )
            
            if not search_results:
                return {
                    "question": question,
                    "answer": "抱歉，我没有找到相关的信息来回答您的问题。",
                    "sources": [],
                    "retrieval_results": [],
                    "processing_time": (datetime.utcnow() - start_time).total_seconds(),
                    "status": "no_results"
                }
            
            # 2. 构建上下文
            logger.info(f"找到 {len(search_results)} 个相关片段，开始构建上下文")
            
            context_chunks = []
            sources = []
            
            for result in search_results:
                # 获取分块的上下文
                chunk_context = self.vector_service.get_chunk_context(
                    chunk_id=result["chunk_id"],
                    context_size=context_size,
                    db=db
                )
                
                if chunk_context:
                    context_chunks.append({
                        "content": chunk_context.get("full_context", result["content"]),
                        "source": f"{result['document_filename']} (第{result['chunk_index']+1}段)",
                        "score": result["score"],
                        "chunk_id": result["chunk_id"],
                        "document_id": result["document_id"]
                    })
                    
                    # 添加来源信息
                    source_info = {
                        "document_id": result["document_id"],
                        "document_filename": result["document_filename"],
                        "document_title": result.get("document_title", ""),
                        "chunk_id": result["chunk_id"],
                        "chunk_index": result["chunk_index"],
                        "score": result["score"],
                        "content_preview": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]
                    }
                    sources.append(source_info)
            
            # 3. 生成回答
            logger.info("开始生成回答")
            
            answer_result = self._generate_answer(question, context_chunks)
            
            # 4. 构建最终结果
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            result = {
                "question": question,
                "answer": answer_result["answer"],
                "sources": sources,
                "retrieval_results": search_results,
                "context_used": len(context_chunks),
                "processing_time": processing_time,
                "model_used": self.model_name,
                "generation_info": answer_result.get("generation_info", {}),
                "status": "success"
            }
            
            logger.info(f"问答完成: 处理时间 {processing_time:.2f}s, 使用 {len(context_chunks)} 个上下文片段")
            return result
            
        except Exception as e:
            logger.error(f"问答处理失败: {e}")
            return {
                "question": question,
                "answer": "抱歉，处理您的问题时出现了错误，请稍后再试。",
                "sources": [],
                "retrieval_results": [],
                "context_used": 0,
                "processing_time": 0,
                "model_used": self.model_name,
                "generation_info": {},
                "status": "error",
                "error": str(e)
            }
    
    def _generate_answer(self, question: str, context_chunks: List[Dict]) -> Dict:
        """
        基于上下文生成回答
        
        Args:
            question: 用户问题
            context_chunks: 上下文片段列表
            
        Returns:
            生成结果字典
        """
        try:
            # 构建上下文文本
            context_text = "\n\n".join([
                f"文档片段 {i+1} (来源: {chunk['source']}, 相似度: {chunk['score']:.3f}):\n{chunk['content']}"
                for i, chunk in enumerate(context_chunks)
            ])
            
            # 构建提示词
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(question, context_text)
            
            # 调用LLM生成回答
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False
            )
            
            # 当触发深度思考时，打印思维链内容
            if hasattr(response.choices[0].message, 'reasoning_content'):
                logger.info(f"深度思考内容: {response.choices[0].message.reasoning_content}")
            
            answer = response.choices[0].message.content.strip()
            
            # 生成信息
            generation_info = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return {
                "answer": answer,
                "generation_info": generation_info
            }
            
        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return {
                "answer": "抱歉，生成回答时出现错误。",
                "generation_info": {},
                "error": str(e)
            }
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """你是一个专业的AI助手，专门基于提供的文档内容回答用户问题。

请遵循以下规则：
1. 仅基于提供的文档内容回答问题，不要添加文档中没有的信息
2. 如果文档内容不足以回答问题，请明确说明
3. 回答要准确、简洁、有条理
4. 如果可能，请引用具体的文档片段来支持你的回答
5. 使用中文回答问题
6. 保持客观和专业的语调

如果文档内容与问题不相关，请礼貌地说明无法基于提供的内容回答该问题。"""
    
    def _build_user_prompt(self, question: str, context: str) -> str:
        """构建用户提示词"""
        return f"""基于以下文档内容，请回答用户的问题。

文档内容：
{context}

用户问题：{question}

请基于上述文档内容提供准确的回答："""
    
    def batch_answer_questions(self, questions: List[str], 
                             document_ids: List[int] = None,
                             db: Session = None) -> List[Dict]:
        """
        批量回答问题
        
        Args:
            questions: 问题列表
            document_ids: 限制搜索的文档ID列表
            db: 数据库会话
            
        Returns:
            回答结果列表
        """
        results = []
        
        for i, question in enumerate(questions):
            logger.info(f"处理问题 {i+1}/{len(questions)}: {question[:50]}...")
            
            result = self.answer_question(
                question=question,
                document_ids=document_ids,
                db=db
            )
            
            results.append(result)
        
        return results
    
    def get_conversation_context(self, conversation_history: List[Dict]) -> str:
        """
        构建对话上下文
        
        Args:
            conversation_history: 对话历史列表
            
        Returns:
            对话上下文字符串
        """
        context_parts = []
        
        for turn in conversation_history[-5:]:  # 只保留最近5轮对话
            if turn.get("question"):
                context_parts.append(f"用户: {turn['question']}")
            if turn.get("answer"):
                context_parts.append(f"助手: {turn['answer']}")
        
        return "\n".join(context_parts)
    
    def answer_with_conversation_context(self, question: str,
                                       conversation_history: List[Dict] = None,
                                       document_ids: List[int] = None,
                                       db: Session = None) -> Dict:
        """
        基于对话上下文回答问题
        
        Args:
            question: 当前问题
            conversation_history: 对话历史
            document_ids: 限制搜索的文档ID列表
            db: 数据库会话
            
        Returns:
            问答结果字典
        """
        # 如果有对话历史，构建增强的查询
        enhanced_question = question
        if conversation_history:
            context = self.get_conversation_context(conversation_history)
            if context:
                enhanced_question = f"对话上下文：\n{context}\n\n当前问题：{question}"
        
        # 使用增强的问题进行检索和回答
        result = self.answer_question(
            question=enhanced_question,
            document_ids=document_ids,
            db=db
        )
        
        # 保持原始问题在结果中
        result["original_question"] = question
        result["enhanced_question"] = enhanced_question if enhanced_question != question else None
        
        return result
    
    def evaluate_answer_quality(self, question: str, answer: str, 
                              sources: List[Dict]) -> Dict:
        """
        评估回答质量
        
        Args:
            question: 问题
            answer: 回答
            sources: 来源信息
            
        Returns:
            质量评估结果
        """
        try:
            evaluation = {
                "relevance_score": 0.0,
                "completeness_score": 0.0,
                "accuracy_score": 0.0,
                "source_quality": 0.0,
                "overall_score": 0.0,
                "feedback": []
            }
            
            # 相关性评估（基于检索分数）
            if sources:
                avg_retrieval_score = sum(s.get("score", 0) for s in sources) / len(sources)
                evaluation["relevance_score"] = min(avg_retrieval_score * 100, 100)
            
            # 来源质量评估
            if sources:
                high_quality_sources = sum(1 for s in sources if s.get("score", 0) > 0.8)
                evaluation["source_quality"] = (high_quality_sources / len(sources)) * 100
            
            # 完整性评估（基于回答长度和结构）
            answer_length = len(answer)
            if answer_length > 100:
                evaluation["completeness_score"] = min(answer_length / 500 * 100, 100)
            else:
                evaluation["completeness_score"] = answer_length / 100 * 50
            
            # 简单的准确性检查（基于是否包含错误信息的关键词）
            error_keywords = ["抱歉", "无法", "不知道", "没有找到", "错误"]
            has_error = any(keyword in answer for keyword in error_keywords)
            evaluation["accuracy_score"] = 50 if has_error else 85
            
            # 计算总体分数
            scores = [
                evaluation["relevance_score"],
                evaluation["completeness_score"],
                evaluation["accuracy_score"],
                evaluation["source_quality"]
            ]
            evaluation["overall_score"] = sum(scores) / len(scores)
            
            # 生成反馈
            if evaluation["overall_score"] >= 80:
                evaluation["feedback"].append("回答质量良好")
            elif evaluation["overall_score"] >= 60:
                evaluation["feedback"].append("回答质量一般，可以改进")
            else:
                evaluation["feedback"].append("回答质量较差，需要改进")
            
            if evaluation["relevance_score"] < 60:
                evaluation["feedback"].append("检索到的内容相关性较低")
            
            if evaluation["source_quality"] < 70:
                evaluation["feedback"].append("来源质量有待提高")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"评估回答质量失败: {e}")
            return {
                "relevance_score": 0.0,
                "completeness_score": 0.0,
                "accuracy_score": 0.0,
                "source_quality": 0.0,
                "overall_score": 0.0,
                "feedback": ["评估失败"],
                "error": str(e)
            }
    
    def get_service_stats(self) -> Dict:
        """
        获取服务统计信息
        
        Returns:
            统计信息字典
        """
        try:
            # 获取向量服务统计
            vector_stats = self.vector_service.get_vectorization_stats()
            
            stats = {
                "service_name": "QA Service",
                "model_name": self.model_name,
                "configuration": {
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_k": self.top_k,
                    "score_threshold": self.score_threshold
                },
                "vector_service_stats": vector_stats,
                "status": "healthy"
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取服务统计失败: {e}")
            return {
                "service_name": "QA Service",
                "status": "error",
                "error": str(e)
            }