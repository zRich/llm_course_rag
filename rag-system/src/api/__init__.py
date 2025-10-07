"""API模块初始化"""

from .health import app
from .embedding import router as embedding_router

__all__ = ['app', 'embedding_router']