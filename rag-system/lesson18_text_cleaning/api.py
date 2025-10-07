#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 文本清洗接口
提供简单的Web API服务用于文本清洗
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import logging
from pathlib import Path

from .cleaning_pipeline import CleaningPipeline, CleaningConfig, create_pipeline
from .text_cleaner import TextCleaner
from .noise_detector import NoiseDetector
from .quality_evaluator import QualityEvaluator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="文本清洗API",
    description="提供文本清洗、噪声检测和质量评估服务",
    version="1.0.0"
)

# 全局组件实例
text_cleaner = TextCleaner()
noise_detector = NoiseDetector()
quality_evaluator = QualityEvaluator()
default_pipeline = create_pipeline()


# 请求模型
class TextCleanRequest(BaseModel):
    """文本清洗请求"""
    text: str = Field(..., description="要清洗的文本")
    cleaning_level: str = Field("standard", description="清洗级别: basic, standard, aggressive")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "这是一个   有多余空格   的文本！！！",
                "cleaning_level": "standard"
            }
        }


class BatchCleanRequest(BaseModel):
    """批量清洗请求"""
    texts: List[str] = Field(..., description="要清洗的文本列表")
    cleaning_level: str = Field("standard", description="清洗级别")
    min_quality_score: float = Field(0.6, description="最低质量分数")
    
    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "这是第一个文本。",
                    "这是第二个   有问题   的文本！！！"
                ],
                "cleaning_level": "standard",
                "min_quality_score": 0.6
            }
        }


class NoiseDetectionRequest(BaseModel):
    """噪声检测请求"""
    text: str = Field(..., description="要检测的文本")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "这是一个包含乱码��������的文本"
            }
        }


class QualityEvaluationRequest(BaseModel):
    """质量评估请求"""
    text: str = Field(..., description="要评估的文本")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "这是一个质量良好的文本，内容完整且结构清晰。"
            }
        }


# API端点
@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "文本清洗API服务",
        "version": "1.0.0",
        "endpoints": {
            "clean": "/clean - 单文本清洗",
            "batch_clean": "/batch_clean - 批量文本清洗",
            "detect_noise": "/detect_noise - 噪声检测",
            "evaluate_quality": "/evaluate_quality - 质量评估",
            "pipeline_stats": "/pipeline_stats - 流水线统计"
        }
    }


@app.post("/clean")
async def clean_text(request: TextCleanRequest):
    """清洗单个文本"""
    try:
        if request.cleaning_level == "basic":
            cleaned_text = text_cleaner.basic_clean(request.text)
        elif request.cleaning_level == "aggressive":
            cleaned_text = text_cleaner.advanced_clean(request.text)
        else:
            cleaned_text = text_cleaner.clean_text(request.text)
        
        # 获取清洗统计
        stats = text_cleaner.get_cleaning_stats(request.text, cleaned_text)
        
        # 质量评估
        quality_result = quality_evaluator.evaluate_text(cleaned_text)
        
        return {
            "success": True,
            "original_text": request.text,
            "cleaned_text": cleaned_text,
            "cleaning_level": request.cleaning_level,
            "cleaning_stats": stats,
            "quality_score": quality_result["overall_score"],
            "quality_details": quality_result
        }
        
    except Exception as e:
        logger.error(f"文本清洗失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清洗失败: {str(e)}")


@app.post("/batch_clean")
async def batch_clean_texts(request: BatchCleanRequest):
    """批量清洗文本"""
    try:
        # 创建配置
        config = CleaningConfig(
            cleaning_level=request.cleaning_level,
            min_quality_score=request.min_quality_score
        )
        
        # 创建流水线
        pipeline = CleaningPipeline(config)
        
        # 批量处理
        results = pipeline.clean_batch(request.texts)
        
        # 格式化结果
        formatted_results = []
        for result in results:
            formatted_results.append({
                "original_text": result.original_text,
                "cleaned_text": result.cleaned_text,
                "quality_score": result.quality_score,
                "noise_ratio": result.noise_ratio,
                "success": result.success,
                "processing_time": result.processing_time,
                "error_message": result.error_message
            })
        
        # 获取统计信息
        stats = pipeline.get_pipeline_stats()
        
        return {
            "success": True,
            "total_texts": len(request.texts),
            "results": formatted_results,
            "pipeline_stats": stats
        }
        
    except Exception as e:
        logger.error(f"批量清洗失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量清洗失败: {str(e)}")


@app.post("/detect_noise")
async def detect_noise(request: NoiseDetectionRequest):
    """检测文本噪声"""
    try:
        result = noise_detector.detect_noise(request.text)
        
        return {
            "success": True,
            "text": request.text,
            "noise_detection": result
        }
        
    except Exception as e:
        logger.error(f"噪声检测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"噪声检测失败: {str(e)}")


@app.post("/evaluate_quality")
async def evaluate_quality(request: QualityEvaluationRequest):
    """评估文本质量"""
    try:
        result = quality_evaluator.evaluate_text(request.text)
        
        return {
            "success": True,
            "text": request.text,
            "quality_evaluation": result
        }
        
    except Exception as e:
        logger.error(f"质量评估失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"质量评估失败: {str(e)}")


@app.post("/upload_file")
async def upload_and_clean_file(file: UploadFile = File(...), 
                               cleaning_level: str = "standard"):
    """上传文件并清洗"""
    try:
        # 检查文件类型
        if not file.filename.endswith(('.txt', '.json')):
            raise HTTPException(status_code=400, detail="只支持.txt和.json文件")
        
        # 读取文件内容
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # 处理JSON文件
        if file.filename.endswith('.json'):
            try:
                data = json.loads(text_content)
                if isinstance(data, list):
                    texts = [str(item) for item in data]
                elif isinstance(data, dict) and 'texts' in data:
                    texts = data['texts']
                else:
                    texts = [str(data)]
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="无效的JSON格式")
        else:
            # 处理文本文件
            texts = text_content.split('\n') if '\n' in text_content else [text_content]
        
        # 批量清洗
        config = CleaningConfig(cleaning_level=cleaning_level)
        pipeline = CleaningPipeline(config)
        results = pipeline.clean_batch(texts)
        
        # 格式化结果
        formatted_results = []
        for result in results:
            formatted_results.append({
                "original_text": result.original_text,
                "cleaned_text": result.cleaned_text,
                "quality_score": result.quality_score,
                "success": result.success
            })
        
        return {
            "success": True,
            "filename": file.filename,
            "total_texts": len(texts),
            "results": formatted_results,
            "pipeline_stats": pipeline.get_pipeline_stats()
        }
        
    except Exception as e:
        logger.error(f"文件处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")


@app.get("/pipeline_stats")
async def get_pipeline_stats():
    """获取流水线统计信息"""
    try:
        stats = default_pipeline.get_pipeline_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@app.post("/reset_stats")
async def reset_pipeline_stats():
    """重置流水线统计信息"""
    try:
        default_pipeline.reset_stats()
        return {
            "success": True,
            "message": "统计信息已重置"
        }
    except Exception as e:
        logger.error(f"重置统计信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重置统计信息失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "文本清洗API",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    
    # 启动服务器
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )