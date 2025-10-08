"""
向量化服务
实现文本的向量化功能，支持本地和API两种embedding模型
"""

import logging
import os
from typing import List, Dict, Optional, Tuple
import asyncio
import time
from datetime import datetime

import numpy as np

# 条件导入
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from src.config.settings import settings
from src.utils.cache import get_cache, make_key, canonicalize_text

logger = logging.getLogger(__name__)


class EmbeddingService:
    """向量化服务，支持本地和API两种模式"""
    
    def __init__(self):
        self.provider = settings.embedding_provider
        self.rate_limit_delay = 1.0  # 请求间隔（秒）
        self.batch_size = 32  # 默认批处理大小
        self.cache = get_cache()
        
        if self.provider == "local":
            self._init_local_model()
        elif self.provider == "api":
            self._init_api_client()
        else:
            raise ValueError(f"不支持的嵌入模型提供者: {self.provider}")
    
    def _init_local_model(self):
        """初始化本地模型"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "使用本地模型需要安装 sentence-transformers: "
                "pip install sentence-transformers"
            )
        
        self.model_name = settings.local_embedding_model
        # 维度以后以实际加载的模型为准，避免与备选模型不一致
        self.dimension = settings.local_embedding_dimension
        self.batch_size = 32  # 添加批处理大小
        
        # 验证模型是否在支持列表中
        if self.model_name not in settings.supported_local_models:
            logger.warning(f"模型 {self.model_name} 不在推荐列表中，可能需要手动下载")
        
        # 优先使用环境变量中的缓存目录，提升命中率
        cache_folder = os.environ.get("HF_HOME")

        try:
            logger.info(f"正在加载本地嵌入模型: {self.model_name}")
            # 若提供缓存目录则传入，避免重复下载
            if cache_folder:
                self.model = SentenceTransformer(self.model_name, cache_folder=cache_folder)
            else:
                self.model = SentenceTransformer(self.model_name)

            # 以实际模型维度为准
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"本地嵌入模型加载成功，维度: {self.dimension}")
        except Exception as e:
            logger.warning(f"加载本地模型失败: {e}. 尝试使用备选模型 sentence-transformers/all-MiniLM-L6-v2")
            # 失败时回退到轻量模型，避免启动阻塞
            fallback_name = "sentence-transformers/all-MiniLM-L6-v2"
            try:
                if cache_folder:
                    self.model = SentenceTransformer(fallback_name, cache_folder=cache_folder)
                else:
                    self.model = SentenceTransformer(fallback_name)
                self.model_name = fallback_name
                self.dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"备选模型加载成功: {self.model_name}，维度: {self.dimension}")
            except Exception as e2:
                logger.error(f"备选模型加载失败: {e2}")
                raise
    
    def _init_api_client(self):
        """初始化API客户端"""
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "使用API模式需要安装 openai: "
                "pip install openai"
            )
        
        self.client = OpenAI(
            api_key=settings.volcengine_api_key,
            base_url=settings.volcengine_base_url,
            timeout=60.0
        )
        self.model_name = settings.volcengine_embedding_model
        self.dimension = 1536  # 火山引擎嵌入模型维度
        self.batch_size = 16  # 添加批处理大小
        
    def _truncate_text(self, text: str, max_tokens: int = None) -> str:
        """
        截断文本以适应模型的token限制
        
        Args:
            text: 输入文本
            max_tokens: 最大token数（默认使用配置值）
            
        Returns:
            截断后的文本
        """
        if max_tokens is None:
            max_tokens = getattr(settings, 'embedding_max_tokens', 8000)
        
        # 简单的字符级截断（粗略估算：4字符≈1token）
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        
        # 在句子边界截断
        truncated = text[:max_chars]
        last_sentence = max(
            truncated.rfind('。'),
            truncated.rfind('！'),
            truncated.rfind('？'),
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )
        
        if last_sentence > max_chars * 0.8:  # 如果句子边界在合理范围内
            return truncated[:last_sentence + 1]
        else:
            return truncated
    
    def get_embedding(self, text: str, model: str = None) -> Tuple[List[float], Dict]:
        """
        获取单个文本的向量表示
        
        Args:
            text: 输入文本
            model: 模型名称（可选）
            
        Returns:
            (向量, 元数据)
        """
        # 缓存包装：按 provider+model+规范化文本生成键
        ttl = int(getattr(settings, 'ttl_embedding', 604800))
        canon = canonicalize_text(text)
        model_ = model or getattr(self, 'model_name', '')
        key = make_key({
            "layer": "embedding",
            "provider": self.provider,
            "model": model_,
            "text": canon,
        })

        def _supplier():
            if self.provider == "local":
                return self._get_local_embedding(text)
            else:
                return self._get_api_embedding(text, model)

        vec_meta = self.cache.get_or_set(key, ttl, _supplier)
        # 兼容 redis 返回 list 的情况
        if isinstance(vec_meta, list) and len(vec_meta) == 2:
            return vec_meta[0], vec_meta[1]
        return vec_meta
    
    def _get_local_embedding(self, text: str) -> Tuple[List[float], Dict]:
        """使用本地模型获取嵌入向量"""
        if not text or not text.strip():
            raise ValueError("输入文本不能为空")
        
        try:
            start_time = time.time()
            
            # 使用本地模型生成嵌入向量
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            # 确保返回的是Python列表
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            elif isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            end_time = time.time()
            
            # 构建元数据
            metadata = {
                "model": self.model_name,
                "provider": "local",
                "dimensions": len(embedding),
                "processing_time": end_time - start_time,
                "text_length": len(text),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.debug(f"本地模型生成向量成功，维度: {len(embedding)}, "
                        f"耗时: {metadata['processing_time']:.3f}s")
            
            return embedding, metadata
            
        except Exception as e:
            logger.error(f"本地模型向量化失败: {e}")
            raise
    
    def _get_api_embedding(self, text: str, model: str = None) -> Tuple[List[float], Dict]:
        """使用API获取嵌入向量"""
        if not text or not text.strip():
            raise ValueError("输入文本不能为空")
        
        model = model or self.model_name
        
        # 截断文本（API模式需要考虑token限制）
        max_tokens = getattr(settings, 'embedding_max_tokens', 8000)
        truncated_text = self._truncate_text(text, max_tokens)
        
        try:
            start_time = time.time()
            logger.debug(f"开始API向量化，文本长度: {len(text)}, 模型: {model}")
            
            response = self.client.embeddings.create(
                input=truncated_text,
                model=model
            )
            
            end_time = time.time()
            
            # 提取向量
            embedding = response.data[0].embedding
            
            # 构建元数据
            metadata = {
                "model": model,
                "provider": "api",
                "dimensions": len(embedding),
                "input_tokens": response.usage.total_tokens,
                "processing_time": end_time - start_time,
                "text_length": len(text),
                "truncated_length": len(truncated_text),
                "was_truncated": len(text) != len(truncated_text),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.debug(f"API生成向量成功，维度: {len(embedding)}, "
                        f"tokens: {response.usage.total_tokens}, "
                        f"耗时: {metadata['processing_time']:.2f}s")
            
            return embedding, metadata
            
        except Exception as e:
            logger.error(f"API向量化失败: {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str], model: str = None) -> Tuple[List[List[float]], List[Dict]]:
        """
        批量获取文本的向量表示
        
        Args:
            texts: 文本列表
            model: 模型名称（可选）
            
        Returns:
            (向量列表, 元数据列表)
        """
        if not texts:
            return [], []
        
        model = model or self.model
        embeddings = []
        metadata_list = []
        
        # 分批处理
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            # TODO(lab03-lesson14): 为批量嵌入接入缓存与预热：
            # - 逐条键：{"layer":"embedding","provider":self.provider,"model":model,"text":canonical}
            # - 批量键（可选）：文本列表hash；命中返回对应向量集合；
            # - 预热：导入常用词库/FAQ，提升命中率；
            batch_embeddings, batch_metadata = self._process_batch(batch_texts, model)
            
            embeddings.extend(batch_embeddings)
            metadata_list.extend(batch_metadata)
            
            # 添加延迟以避免触发速率限制
            if i + self.batch_size < len(texts):
                time.sleep(self.rate_limit_delay)
        
        logger.info(f"批量向量化完成，处理 {len(texts)} 个文本")
        return embeddings, metadata_list
    
    def _process_batch(self, texts: List[str], model: str) -> Tuple[List[List[float]], List[Dict]]:
        """
        处理单个批次的文本
        
        Args:
            texts: 文本列表
            model: 模型名称
            
        Returns:
            (向量列表, 元数据列表)
        """
        # 根据provider选择不同的处理方式
        if self.provider == "local":
            return self._process_batch_local(texts, model)
        else:
            return self._process_batch_api(texts, model)
    
    def _process_batch_local(self, texts: List[str], model: str) -> Tuple[List[List[float]], List[Dict]]:
        """使用本地模型处理批次"""
        try:
            start_time = time.time()
            
            # 使用本地模型批量生成嵌入向量
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            
            # 确保返回的是Python列表
            if hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()
            elif isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            end_time = time.time()
            
            # 构建元数据
            metadata_list = []
            for i, text in enumerate(texts):
                metadata = {
                    "model": model,
                    "provider": "local",
                    "dimensions": len(embeddings[i]),
                    "processing_time": (end_time - start_time) / len(texts),
                    "text_length": len(text),
                    "batch_size": len(texts),
                    "timestamp": datetime.utcnow().isoformat()
                }
                metadata_list.append(metadata)
            
            return embeddings, metadata_list
            
        except Exception as e:
            logger.error(f"本地模型批量向量化失败: {e}")
            raise
    
    def _process_batch_api(self, texts: List[str], model: str) -> Tuple[List[List[float]], List[Dict]]:
        """使用API处理批次"""
        # 截断文本
        truncated_texts = [self._truncate_text(text) for text in texts]
        
        try:
            start_time = time.time()
            
            response = self.client.embeddings.create(
                input=truncated_texts,
                model=model
            )
            
            end_time = time.time()
            
            # 提取向量
            embeddings = [data.embedding for data in response.data]
            
            # 构建元数据
            metadata_list = []
            for i, (original_text, truncated_text) in enumerate(zip(texts, truncated_texts)):
                metadata = {
                    "model": model,
                    "provider": "api",
                    "dimensions": len(embeddings[i]),
                    "input_tokens": response.usage.total_tokens // len(texts),  # 平均分配
                    "processing_time": (end_time - start_time) / len(texts),  # 平均分配
                    "text_length": len(original_text),
                    "truncated_length": len(truncated_text),
                    "was_truncated": len(original_text) != len(truncated_text),
                    "batch_size": len(texts),
                    "timestamp": datetime.utcnow().isoformat()
                }
                metadata_list.append(metadata)
            
            logger.debug(f"批次处理完成，{len(texts)} 个文本，"
                        f"总tokens: {response.usage.total_tokens}, "
                        f"耗时: {end_time - start_time:.2f}s")
            
            return embeddings, metadata_list
            
        except Exception as e:
            logger.error(f"批次向量化失败: {e}")
            raise
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            embedding1: 向量1
            embedding2: 向量2
            
        Returns:
            余弦相似度 (-1 到 1)
        """
        if len(embedding1) != len(embedding2):
            raise ValueError("向量维度不匹配")
        
        # 转换为numpy数组
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # 计算余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def find_most_similar(self, query_embedding: List[float], 
                         candidate_embeddings: List[List[float]], 
                         top_k: int = 5) -> List[Tuple[int, float]]:
        """
        找到最相似的向量
        
        Args:
            query_embedding: 查询向量
            candidate_embeddings: 候选向量列表
            top_k: 返回前k个最相似的
            
        Returns:
            [(索引, 相似度)] 列表，按相似度降序排列
        """
        similarities = []
        
        for i, candidate_embedding in enumerate(candidate_embeddings):
            similarity = self.calculate_similarity(query_embedding, candidate_embedding)
            similarities.append((i, similarity))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        验证向量的有效性
        
        Args:
            embedding: 向量
            
        Returns:
            是否有效
        """
        if not embedding:
            return False
        
        # 检查是否包含NaN或无穷大
        vec = np.array(embedding)
        if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
            return False
        
        # 检查向量长度
        norm = np.linalg.norm(vec)
        if norm == 0:
            return False
        
        return True
    
    def normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        归一化向量
        
        Args:
            embedding: 原始向量
            
        Returns:
            归一化后的向量
        """
        vec = np.array(embedding)
        norm = np.linalg.norm(vec)
        
        if norm == 0:
            return embedding
        
        normalized = vec / norm
        return normalized.tolist()
    
    async def get_embedding_async(self, text: str, model: str = None) -> Tuple[List[float], Dict]:
        """
        异步获取文本向量
        
        Args:
            text: 输入文本
            model: 模型名称（可选）
            
        Returns:
            (向量, 元数据)
        """
        # 在实际应用中，这里应该使用异步的火山引擎客户端
        # 目前使用同步方法的异步包装
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_embedding, text, model)
    
    def get_model_info(self) -> Dict:
        """
        获取当前模型信息
        
        Returns:
            模型信息字典
        """
        base_info = {
            "provider": self.provider,
            "model": self.model_name,
            "dimensions": self.dimension,
            "rate_limit_delay": self.rate_limit_delay,
        }
        
        if self.provider == "local":
            base_info.update({
                "model_type": "sentence-transformers",
                "supported_models": list(settings.supported_local_models.keys()),
                "current_model_info": settings.supported_local_models.get(
                    self.model_name, 
                    {"description": "自定义模型"}
                )
            })
        else:
            base_info.update({
                "model_type": "api",
                "api_base": settings.volcengine_base_url,
                "max_tokens": getattr(settings, 'embedding_max_tokens', 8000),
                "batch_size": getattr(settings, 'embedding_batch_size', 10)
            })
        
        return base_info