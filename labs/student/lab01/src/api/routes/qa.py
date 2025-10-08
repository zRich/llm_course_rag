"""
问答API路由
处理基于RAG的问答功能
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from api.schemas import (
    QuestionRequest, QuestionResponse, BatchQuestionRequest, BatchQuestionResponse,
    SourceInfo, BaseResponse
)
from api.dependencies import (
    get_db, get_qa_service, validate_qa_params
)
from services.qa_service import QAService
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/qa", tags=["问答"])


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    qa_service: QAService = Depends(get_qa_service)
):
    """
    回答用户问题
    
    基于检索增强生成（RAG）技术回答问题
    """
    try:
        # TODO(lab01-task4): 实现问答参数验证
        # 任务说明：验证问答请求参数的有效性
        # 实现要点：
        # 1. 调用参数验证函数：
        #    validated_params = validate_qa_params(
        #        question=request.question,
        #        top_k=request.top_k,
        #        score_threshold=request.score_threshold,
        #        context_size=request.context_size
        #    )
        # 2. 验证问题不为空，长度合理
        # 3. 验证top_k在合理范围内（1-50）
        # 4. 验证score_threshold在0.0-1.0范围内
        # 5. 验证context_size在合理范围内（0-5）
        # 期望返回：validated_params字典
        
        # 示例代码：
        # validated_params = validate_qa_params(
        #     question=request.question,
        #     top_k=request.top_k,
        #     score_threshold=request.score_threshold,
        #     context_size=request.context_size
        # )
        
        raise NotImplementedError("请实现问答参数验证逻辑")
        
        # TODO(lab01-task4): 执行RAG问答
        # 任务说明：调用问答服务进行RAG问答
        # 实现要点：
        # 1. 调用问答服务：
        #    result = qa_service.answer_question(
        #        question=validated_params["question"],
        #        document_ids=request.document_ids,
        #        context_size=validated_params["context_size"]
        #    )
        # 2. 处理问答结果，确保返回格式正确
        # 期望返回：result字典，包含问答结果
        
        # 示例代码：
        # result = qa_service.answer_question(
        #     question=validated_params["question"],
        #     document_ids=request.document_ids,
        #     context_size=validated_params["context_size"]
        # )
        
        # TODO(lab01-task4): 转换来源信息格式
        # 任务说明：将问答结果中的来源信息转换为API响应格式
        # 实现要点：
        # 1. 初始化来源列表：sources = []
        # 2. 遍历结果中的来源：for source in result["sources"]:
        # 3. 创建SourceInfo对象：
        #    source_info = SourceInfo(
        #        document_id=source["document_id"],
        #        document_filename=source["document_filename"],
        #        document_title=source.get("document_title"),
        #        chunk_id=source["chunk_id"],
        #        chunk_index=source["chunk_index"],
        #        score=source["score"],
        #        content_preview=source["content_preview"]
        #    )
        # 4. 添加到来源列表：sources.append(source_info)
        # 期望返回：sources列表
        
        # 示例代码结构：
        # sources = []
        # for source in result["sources"]:
        #     source_info = SourceInfo(
        #         document_id=source["document_id"],
        #         document_filename=source["document_filename"],
        #         document_title=source.get("document_title"),
        #         chunk_id=source["chunk_id"],
        #         chunk_index=source["chunk_index"],
        #         score=source["score"],
        #         content_preview=source["content_preview"]
        #     )
        #     sources.append(source_info)
        
        # TODO(lab01-task4): 构建问答响应
        # 任务说明：构建标准的问答响应对象
        # 实现要点：
        # 1. 创建QuestionResponse对象：
        #    return QuestionResponse(
        #        success=True,
        #        message="问答完成",
        #        question=validated_params["question"],
        #        answer=result["answer"],
        #        sources=sources,
        #        context_used=result["context_used"],
        #        processing_time=result["processing_time"],
        #        model_used=result["model_used"],
        #        generation_info=result.get("generation_info")
        #    )
        # 期望返回：QuestionResponse对象
        
        # 示例代码：
        # return QuestionResponse(
        #     success=True,
        #     message="问答完成",
        #     question=validated_params["question"],
        #     answer=result["answer"],
        #     sources=sources,
        #     context_used=result["context_used"],
        #     processing_time=result["processing_time"],
        #     model_used=result["model_used"],
        #     generation_info=result.get("generation_info")
        # )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"问答处理失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"问答处理失败: {str(e)}"
        )


@router.get("/ask", response_model=QuestionResponse)
async def ask_question_get(
    question: str = Query(..., description="用户问题"),
    document_ids: Optional[List[int]] = Query(None, description="限制搜索的文档ID列表"),
    top_k: int = Query(10, ge=1, le=50, description="检索结果数量"),
    score_threshold: float = Query(0.7, ge=0.0, le=1.0, description="相似度阈值"),
    context_size: int = Query(2, ge=0, le=5, description="上下文大小"),
    qa_service: QAService = Depends(get_qa_service)
):
    """
    回答用户问题（GET方法）
    
    基于检索增强生成（RAG）技术回答问题
    """
    try:
        # 验证问答参数
        validated_params = validate_qa_params(
            question=question,
            top_k=top_k,
            score_threshold=score_threshold,
            context_size=context_size
        )
        
        # 执行问答
        result = qa_service.answer_question(
            question=validated_params["question"],
            document_ids=document_ids,
            context_size=validated_params["context_size"]
        )
        
        # 转换来源信息
        sources = []
        for source in result["sources"]:
            source_info = SourceInfo(
                document_id=source["document_id"],
                document_filename=source["document_filename"],
                document_title=source.get("document_title"),
                chunk_id=source["chunk_id"],
                chunk_index=source["chunk_index"],
                score=source["score"],
                content_preview=source["content_preview"]
            )
            sources.append(source_info)
        
        return QuestionResponse(
            success=True,
            message="问答完成",
            question=validated_params["question"],
            answer=result["answer"],
            sources=sources,
            context_used=result["context_used"],
            processing_time=result["processing_time"],
            model_used=result["model_used"],
            generation_info=result.get("generation_info")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"问答处理失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"问答处理失败: {str(e)}"
        )


@router.post("/batch", response_model=BatchQuestionResponse)
async def ask_batch_questions(
    request: BatchQuestionRequest,
    qa_service: QAService = Depends(get_qa_service)
):
    """
    批量回答问题
    
    一次性回答多个问题
    """
    try:
        # 执行批量问答
        results = await qa_service.answer_questions_batch(
            questions=request.questions,
            document_ids=request.document_ids,
            top_k=request.top_k,
            score_threshold=request.score_threshold
        )
        
        # 转换结果
        question_responses = []
        total_processing_time = 0
        
        for result in results["results"]:
            # 转换来源信息
            sources = []
            for source in result["sources"]:
                source_info = SourceInfo(
                    document_id=source["document_id"],
                    document_filename=source["document_filename"],
                    document_title=source.get("document_title"),
                    chunk_id=source["chunk_id"],
                    chunk_index=source["chunk_index"],
                    score=source["score"],
                    content_preview=source["content_preview"]
                )
                sources.append(source_info)
            
            question_response = QuestionResponse(
                success=True,
                message="问答完成",
                question=result["question"],
                answer=result["answer"],
                sources=sources,
                context_used=result["context_used"],
                processing_time=result["processing_time"],
                model_used=result["model_used"],
                generation_info=result.get("generation_info")
            )
            question_responses.append(question_response)
            total_processing_time += result["processing_time"]
        
        return BatchQuestionResponse(
            success=True,
            message=f"批量问答完成，处理了 {len(question_responses)} 个问题",
            results=question_responses,
            total_processing_time=total_processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量问答处理失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量问答处理失败: {str(e)}"
        )


@router.post("/conversation", response_model=QuestionResponse)
async def ask_with_conversation(
    request: QuestionRequest,
    qa_service: QAService = Depends(get_qa_service)
):
    """
    基于对话历史回答问题
    
    考虑对话上下文的问答功能
    """
    try:
        # 验证问答参数
        validated_params = validate_qa_params(
            question=request.question,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            context_size=request.context_size
        )
        
        # 执行对话问答
        result = await qa_service.answer_with_conversation_context(
            question=validated_params["question"],
            conversation_history=request.conversation_history or [],
            document_ids=request.document_ids,
            top_k=validated_params["top_k"],
            score_threshold=validated_params["score_threshold"],
            context_size=validated_params["context_size"]
        )
        
        # 转换来源信息
        sources = []
        for source in result["sources"]:
            source_info = SourceInfo(
                document_id=source["document_id"],
                document_filename=source["document_filename"],
                document_title=source.get("document_title"),
                chunk_id=source["chunk_id"],
                chunk_index=source["chunk_index"],
                score=source["score"],
                content_preview=source["content_preview"]
            )
            sources.append(source_info)
        
        return QuestionResponse(
            success=True,
            message="对话问答完成",
            question=validated_params["question"],
            answer=result["answer"],
            sources=sources,
            context_used=result["context_used"],
            processing_time=result["processing_time"],
            model_used=result["model_used"],
            generation_info=result.get("generation_info")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"对话问答处理失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"对话问答处理失败: {str(e)}"
        )


@router.post("/evaluate", response_model=BaseResponse)
async def evaluate_answer(
    question: str = Query(..., description="问题"),
    answer: str = Query(..., description="回答"),
    expected_answer: Optional[str] = Query(None, description="期望回答"),
    qa_service: QAService = Depends(get_qa_service)
):
    """
    评估回答质量
    
    对生成的回答进行质量评估
    """
    try:
        result = await qa_service.evaluate_answer(
            question=question,
            answer=answer,
            expected_answer=expected_answer
        )
        
        return BaseResponse(
            success=True,
            message=f"回答评估完成，质量分数: {result.get('quality_score', 'N/A')}"
        )
        
    except Exception as e:
        logger.error(f"回答评估失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"回答评估失败: {str(e)}"
        )


@router.get("/stats", response_model=BaseResponse)
async def get_qa_stats(
    qa_service: QAService = Depends(get_qa_service)
):
    """
    获取问答服务统计信息
    """
    try:
        stats = await qa_service.get_service_stats()
        
        return BaseResponse(
            success=True,
            message="获取问答服务统计信息成功"
        )
        
    except Exception as e:
        logger.error(f"获取问答服务统计信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取问答服务统计信息失败: {str(e)}"
        )