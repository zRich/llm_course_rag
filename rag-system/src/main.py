from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

# 创建FastAPI应用实例
app = FastAPI(
    title="RAG System API",
    description="一个基于FastAPI的RAG（检索增强生成）系统",
    version="1.0.0"
)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义响应模型
class HealthResponse(BaseModel):
    status: str
    message: str
    version: str

class InfoResponse(BaseModel):
    project: str
    description: str
    features: list[str]

# 根路径
@app.get("/")
async def root() -> Dict[str, Any]:
    """根路径，返回API基本信息"""
    return {
        "message": "欢迎使用RAG系统API",
        "docs": "/docs",
        "health": "/health"
    }

# 健康检查接口
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """健康检查接口"""
    return HealthResponse(
        status="healthy",
        message="RAG系统运行正常",
        version="1.0.0"
    )

# 系统信息接口
@app.get("/info", response_model=InfoResponse)
async def get_info() -> InfoResponse:
    """获取系统信息"""
    return InfoResponse(
        project="RAG System",
        description="基于FastAPI的检索增强生成系统",
        features=[
            "文档上传与解析",
            "向量化存储",
            "语义检索",
            "智能问答"
        ]
    )

# 如果直接运行此文件，启动服务器
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )