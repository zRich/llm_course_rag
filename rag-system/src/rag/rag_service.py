"""RAG服务模块"""

from typing import List, Dict, Any, Optional
import logging
import time
from dataclasses import dataclass, asdict

from .retriever import DocumentRetriever
from .qa_generator import QAGenerator, QAResponse
from ..embedding.embedder import TextEmbedder
from ..vector_store.qdrant_client import QdrantVectorStore

logger = logging.getLogger(__name__)

@dataclass
class RAGRequest:
    """RAG请求"""
    question: str
    collection_name: str = "documents"
    top_k: int = 5
    score_threshold: float = 0.7
    conversation_history: Optional[List[Dict[str, str]]] = None
    include_metadata: bool = True

@dataclass
class RAGResponse:
    """RAG响应"""
    question: str
    answer: str
    confidence: float
    sources: List[str]
    retrieved_documents: List[Dict[str, Any]]
    processing_time: float
    metadata: Dict[str, Any]
    followup_questions: List[str]

class RAGService:
    """RAG服务
    
    整合文档检索和问答生成的完整RAG流水线
    """
    
    def __init__(self,
                 embedder: Optional[TextEmbedder] = None,
                 vector_store: Optional[QdrantVectorStore] = None,
                 retriever: Optional[DocumentRetriever] = None,
                 qa_generator: Optional[QAGenerator] = None):
        """
        初始化RAG服务
        
        Args:
            embedder: 文本嵌入器
            vector_store: 向量存储
            retriever: 文档检索器
            qa_generator: 问答生成器
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化组件
        self.embedder = embedder or TextEmbedder()
        self.vector_store = vector_store or QdrantVectorStore()
        self.retriever = retriever or DocumentRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store
        )
        self.qa_generator = qa_generator or QAGenerator()
        
        self.logger.info("RAG服务初始化完成")
    
    async def query(self, request: RAGRequest) -> RAGResponse:
        """
        执行RAG查询
        
        Args:
            request: RAG请求
            
        Returns:
            RAG响应
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"开始RAG查询: {request.question[:50]}...")
            
            # 1. 文档检索
            retrieved_docs = await self.retriever.retrieve(
                query=request.question,
                collection_name=request.collection_name,
                top_k=request.top_k,
                score_threshold=request.score_threshold
            )
            
            self.logger.info(f"检索到 {len(retrieved_docs)} 个相关文档")
            
            # 2. 构建上下文
            context = self.retriever.format_context(retrieved_docs)
            
            # 3. 生成答案
            qa_response = self.qa_generator.generate_answer(
                question=request.question,
                context=context,
                conversation_history=request.conversation_history
            )
            
            # 4. 生成后续问题建议
            followup_questions = self.qa_generator.generate_followup_questions(
                question=request.question,
                answer=qa_response.answer,
                context=context
            )
            
            # 5. 构建响应
            processing_time = time.time() - start_time
            
            response = RAGResponse(
                question=request.question,
                answer=qa_response.answer,
                confidence=qa_response.confidence,
                sources=qa_response.sources,
                retrieved_documents=retrieved_docs if request.include_metadata else [],
                processing_time=processing_time,
                metadata={
                    "retrieval_time": processing_time - qa_response.processing_time,
                    "generation_time": qa_response.processing_time,
                    "total_documents": len(retrieved_docs),
                    "context_length": len(context),
                    **qa_response.metadata
                },
                followup_questions=followup_questions
            )
            
            self.logger.info(f"RAG查询完成: 耗时 {processing_time:.2f}s")
            return response
            
        except Exception as e:
            self.logger.error(f"RAG查询失败: {e}")
            raise
    
    def query_sync(self, request: RAGRequest) -> RAGResponse:
        """
        同步执行RAG查询
        
        Args:
            request: RAG请求
            
        Returns:
            RAG响应
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"开始同步RAG查询: {request.question[:50]}...")
            
            # 1. 文档检索
            retrieved_docs = self.retriever.retrieve_sync(
                query=request.question,
                collection_name=request.collection_name,
                top_k=request.top_k,
                score_threshold=request.score_threshold
            )
            
            self.logger.info(f"检索到 {len(retrieved_docs)} 个相关文档")
            
            # 2. 构建上下文
            context = self.retriever.format_context(retrieved_docs)
            
            # 3. 生成答案
            qa_response = self.qa_generator.generate_answer(
                question=request.question,
                context=context,
                conversation_history=request.conversation_history
            )
            
            # 4. 生成后续问题建议
            followup_questions = self.qa_generator.generate_followup_questions(
                question=request.question,
                answer=qa_response.answer,
                context=context
            )
            
            # 5. 构建响应
            processing_time = time.time() - start_time
            
            response = RAGResponse(
                question=request.question,
                answer=qa_response.answer,
                confidence=qa_response.confidence,
                sources=qa_response.sources,
                retrieved_documents=retrieved_docs if request.include_metadata else [],
                processing_time=processing_time,
                metadata={
                    "retrieval_time": processing_time - qa_response.processing_time,
                    "generation_time": qa_response.processing_time,
                    "total_documents": len(retrieved_docs),
                    "context_length": len(context),
                    **qa_response.metadata
                },
                followup_questions=followup_questions
            )
            
            self.logger.info(f"同步RAG查询完成: 耗时 {processing_time:.2f}s")
            return response
            
        except Exception as e:
            self.logger.error(f"同步RAG查询失败: {e}")
            raise
    
    def batch_query(self, requests: List[RAGRequest]) -> List[RAGResponse]:
        """
        批量执行RAG查询
        
        Args:
            requests: RAG请求列表
            
        Returns:
            RAG响应列表
        """
        responses = []
        
        for i, request in enumerate(requests):
            try:
                self.logger.info(f"处理批量查询 {i+1}/{len(requests)}")
                response = self.query_sync(request)
                responses.append(response)
            except Exception as e:
                self.logger.error(f"批量查询第 {i+1} 个请求失败: {e}")
                # 创建错误响应
                error_response = RAGResponse(
                    question=request.question,
                    answer=f"处理请求时发生错误: {str(e)}",
                    confidence=0.0,
                    sources=[],
                    retrieved_documents=[],
                    processing_time=0.0,
                    metadata={"error": str(e)},
                    followup_questions=[]
                )
                responses.append(error_response)
        
        return responses
    
    def get_collection_stats(self, collection_name: str = "documents") -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Args:
            collection_name: 集合名称
            
        Returns:
            统计信息
        """
        try:
            return self.retriever.get_collection_stats(collection_name)
        except Exception as e:
            self.logger.error(f"获取集合统计信息失败: {e}")
            return {"error": str(e)}
    
    def validate_query(self, question: str) -> Dict[str, Any]:
        """
        验证查询问题
        
        Args:
            question: 问题
            
        Returns:
            验证结果
        """
        validation_result = {
            "is_valid": True,
            "issues": [],
            "suggestions": []
        }
        
        try:
            # 检查问题长度
            if not question or not question.strip():
                validation_result["is_valid"] = False
                validation_result["issues"].append("问题不能为空")
                return validation_result
            
            if len(question.strip()) < 3:
                validation_result["issues"].append("问题过短")
                validation_result["suggestions"].append("请提供更详细的问题描述")
            
            if len(question) > 1000:
                validation_result["issues"].append("问题过长")
                validation_result["suggestions"].append("请简化问题描述")
            
            # 检查问题类型
            question_lower = question.lower().strip()
            if not any(char in question_lower for char in ['?', '？', '什么', '如何', '为什么', 'what', 'how', 'why', 'when', 'where']):
                validation_result["suggestions"].append("建议使用疑问句形式提问")
            
            return validation_result
            
        except Exception as e:
            self.logger.warning(f"问题验证失败: {e}")
            return {"is_valid": True, "issues": [], "suggestions": []}
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态
        
        Returns:
            系统状态信息
        """
        try:
            status = {
                "service_status": "running",
                "components": {
                    "embedder": "initialized" if self.embedder else "not_initialized",
                    "vector_store": "initialized" if self.vector_store else "not_initialized",
                    "retriever": "initialized" if self.retriever else "not_initialized",
                    "qa_generator": "initialized" if self.qa_generator else "not_initialized"
                },
                "timestamp": time.time()
            }
            
            # 检查向量存储连接
            try:
                collections = self.vector_store.list_collections()
                status["vector_store_collections"] = len(collections)
                status["vector_store_status"] = "connected"
            except Exception as e:
                status["vector_store_status"] = f"error: {str(e)}"
            
            return status
            
        except Exception as e:
            self.logger.error(f"获取系统状态失败: {e}")
            return {
                "service_status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def to_dict(self, response: RAGResponse) -> Dict[str, Any]:
        """
        将RAG响应转换为字典
        
        Args:
            response: RAG响应
            
        Returns:
            字典格式的响应
        """
        return asdict(response)