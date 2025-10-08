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
            
            # TODO(lab01-task4): 实现文档检索逻辑
            # 任务说明：根据用户问题检索相关的文档片段
            # 实现要点：
            # 1. 记录开始时间和日志：
            #    logger.info(f"开始检索相关文档: 问题='{question[:100]}...'")
            # 2. 调用向量服务搜索相似分块：
            #    search_results = self.vector_service.search_similar_chunks(
            #        query=question,
            #        limit=self.top_k,
            #        score_threshold=self.score_threshold,
            #        document_ids=document_ids,
            #        db=db
            #    )
            # 3. 检查搜索结果，如果为空则返回无结果响应
            # 期望返回：search_results列表
            
            # 示例代码：
            # logger.info(f"开始检索相关文档: 问题='{question[:100]}...'")
            # 
            # search_results = self.vector_service.search_similar_chunks(
            #     query=question,
            #     limit=self.top_k,
            #     score_threshold=self.score_threshold,
            #     document_ids=document_ids,
            #     db=db
            # )
            # 
            # if not search_results:
            #     return {
            #         "question": question,
            #         "answer": "抱歉，我没有找到相关的信息来回答您的问题。",
            #         "sources": [],
            #         "retrieval_results": [],
            #         "processing_time": (datetime.utcnow() - start_time).total_seconds(),
            #         "status": "no_results"
            #     }
            
            raise NotImplementedError("请实现文档检索逻辑")
            
            # TODO(lab01-task4): 构建RAG上下文
            # 任务说明：基于检索结果构建问答上下文
            # 实现要点：
            # 1. 记录日志：logger.info(f"找到 {len(search_results)} 个相关片段，开始构建上下文")
            # 2. 初始化上下文和来源列表：context_chunks = [], sources = []
            # 3. 遍历搜索结果：for result in search_results:
            # 4. 获取分块上下文：
            #    chunk_context = self.vector_service.get_chunk_context(
            #        chunk_id=result["chunk_id"],
            #        context_size=context_size,
            #        db=db
            #    )
            # 5. 处理上下文信息，添加到context_chunks和sources列表
            # 期望返回：context_chunks和sources列表
            
            # 示例代码结构：
            # logger.info(f"找到 {len(search_results)} 个相关片段，开始构建上下文")
            # 
            # context_chunks = []
            # sources = []
            # 
            # for result in search_results:
            #     chunk_context = self.vector_service.get_chunk_context(
            #         chunk_id=result["chunk_id"],
            #         context_size=context_size,
            #         db=db
            #     )
            #     
            #     if chunk_context:
            #         context_chunks.append({
            #             "content": chunk_context["content"],
            #             "document_filename": result["document_filename"],
            #             "chunk_index": result["chunk_index"],
            #             "score": result["score"]
            #         })
            #         
            #         sources.append({
            #             "document_id": result["document_id"],
            #             "document_filename": result["document_filename"],
            #             "document_title": result.get("document_title"),
            #             "chunk_id": result["chunk_id"],
            #             "chunk_index": result["chunk_index"],
            #             "score": result["score"],
            #             "start_position": result.get("start_pos"),
            #             "end_position": result.get("end_pos")
            #         })
            
            # TODO(lab01-task4): 生成AI回答
            # 任务说明：使用LLM基于上下文生成答案
            # 实现要点：
            # 1. 调用答案生成方法：
            #    answer_result = self._generate_answer(question, context_chunks)
            # 2. 检查生成结果，处理可能的错误
            # 3. 计算处理时间：processing_time = (datetime.utcnow() - start_time).total_seconds()
            # 4. 构建完整的响应字典，包含问题、答案、来源、检索结果等信息
            # 期望返回：完整的问答结果字典
            
            # 示例代码结构：
            # answer_result = self._generate_answer(question, context_chunks)
            # 
            # if not answer_result.get("success", False):
            #     return {
            #         "question": question,
            #         "answer": "抱歉，生成答案时出现错误。",
            #         "sources": sources,
            #         "retrieval_results": search_results,
            #         "processing_time": (datetime.utcnow() - start_time).total_seconds(),
            #         "status": "generation_error",
            #         "error": answer_result.get("error")
            #     }
            # 
            # processing_time = (datetime.utcnow() - start_time).total_seconds()
            # 
            # return {
            #     "question": question,
            #     "answer": answer_result["answer"],
            #     "sources": sources,
            #     "retrieval_results": search_results,
            #     "processing_time": processing_time,
            #     "status": "success",
            #     "model_info": {
            #         "model_name": self.model_name,
            #         "temperature": self.temperature,
            #         "max_tokens": self.max_tokens
            #     },
            #     "retrieval_info": {
            #         "chunks_found": len(search_results),
            #         "chunks_used": len(context_chunks),
            #         "score_threshold": self.score_threshold
            #     }
            # }
            
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
            # TODO(lab01-task4): 构建上下文文本
            # 任务说明：将多个上下文片段合并为单一的上下文文本
            # 实现要点：
            # 1. 遍历上下文片段：for i, chunk in enumerate(context_chunks):
            # 2. 格式化每个片段：包含片段编号、来源、相似度和内容
            # 3. 使用换行符连接：context_text = "\n\n".join([...])
            # 期望返回：格式化的上下文文本字符串
            
            # 示例代码：
            # context_text = "\n\n".join([
            #     f"文档片段 {i+1} (来源: {chunk['source']}, 相似度: {chunk['score']:.3f}):\n{chunk['content']}"
            #     for i, chunk in enumerate(context_chunks)
            # ])
            
            raise NotImplementedError("请实现上下文文本构建逻辑")
            
            # TODO(lab01-task4): 构建提示词
            # 任务说明：构建系统提示词和用户提示词
            # 实现要点：
            # 1. 构建系统提示词：system_prompt = self._build_system_prompt()
            # 2. 构建用户提示词：user_prompt = self._build_user_prompt(question, context_text)
            # 3. 准备消息列表：messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            # 期望返回：messages列表
            
            # 示例代码：
            # system_prompt = self._build_system_prompt()
            # user_prompt = self._build_user_prompt(question, context_text)
            # 
            # messages = [
            #     {"role": "system", "content": system_prompt},
            #     {"role": "user", "content": user_prompt}
            # ]
            
            # TODO(lab01-task4): 调用LLM生成回答
            # 任务说明：使用OpenAI客户端调用LLM生成回答
            # 实现要点：
            # 1. 调用客户端：
            #    response = self.client.chat.completions.create(
            #        model=self.model_name,
            #        messages=messages,
            #        max_tokens=self.max_tokens,
            #        temperature=self.temperature
            #    )
            # 2. 提取回答内容：answer = response.choices[0].message.content.strip()
            # 3. 构建成功响应：包含answer、success=True等信息
            # 期望返回：包含生成结果的字典
            
            # 示例代码结构：
            # response = self.client.chat.completions.create(
            #     model=self.model_name,
            #     messages=messages,
            #     max_tokens=self.max_tokens,
            #     temperature=self.temperature
            # )
            # 
            # answer = response.choices[0].message.content.strip()
            # 
            # return {
            #     "success": True,
            #     "answer": answer,
            #     "generation_info": {
            #         "model": self.model_name,
            #         "tokens_used": response.usage.total_tokens if response.usage else 0,
            #         "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            #         "completion_tokens": response.usage.completion_tokens if response.usage else 0
            #     }
            # }
            
        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return {
                "success": False,
                "answer": "抱歉，生成回答时出现错误。",
                "error": str(e),
                "generation_info": {}
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