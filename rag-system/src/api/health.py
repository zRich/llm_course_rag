from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import sys
import platform

# 导入路由
from .embedding import router as embedding_router

app = FastAPI(
    title="RAG System API",
    description="Enterprise RAG System with Embedding Support",
    version="0.1.0"
)

# 注册路由
app.include_router(embedding_router)

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    python_version: str
    platform: str

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="0.1.0",
        python_version=sys.version,
        platform=platform.platform()
    )

@app.get("/")
async def root():
    """根路径"""
    return {"message": "Welcome to RAG System API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)