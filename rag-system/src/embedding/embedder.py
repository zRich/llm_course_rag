"""文本向量化模块"""

import os
import json
import pickle
from typing import List, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path

# 简化版本，使用基础的向量化实现
import hashlib
import re
from collections import Counter
import math

import logging
logger = logging.getLogger(__name__)

class TextEmbedder:
    """文本向量化器 - 简化版本使用TF-IDF"""
    
    def __init__(self, model_name: str = "tfidf", device: str = "cpu"):
        """
        初始化文本向量化器
        
        Args:
            model_name: 模型名称 (简化版本固定为tfidf)
            device: 设备类型 (简化版本固定为cpu)
        """
        self.model_name = model_name
        self.device = device
        self.vocabulary = {}
        self.idf_scores = {}
        self.embedding_dimension = 384  # 固定维度
        logger.info(f"初始化简化版向量化器: {self.model_name}")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """文本预处理"""
        # 转小写，移除标点，分词
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        return [word for word in words if len(word) > 1]
    
    def _build_vocabulary(self, texts: List[str]):
        """构建词汇表"""
        word_counts = Counter()
        doc_counts = Counter()
        
        for text in texts:
            words = self._preprocess_text(text)
            word_counts.update(words)
            doc_counts.update(set(words))
        
        # 选择最常见的词作为特征
        most_common = word_counts.most_common(self.embedding_dimension)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}
        
        # 计算IDF分数
        total_docs = len(texts)
        for word in self.vocabulary:
            self.idf_scores[word] = math.log(total_docs / (doc_counts[word] + 1))
    
    def _text_to_vector(self, text: str) -> np.ndarray:
        """
        将文本转换为TF-IDF向量
        
        Args:
            text: 输入文本
            
        Returns:
            TF-IDF向量
        """
        if not self.vocabulary:
             return np.zeros(self.embedding_dimension, dtype=np.float32)
        
        # 预处理文本
        words = self._preprocess_text(text)
        word_count = Counter(words)
        
        # 计算TF-IDF向量
        vector = np.zeros(len(self.vocabulary), dtype=np.float32)
        total_words = len(words)
        
        if total_words == 0:
            return np.zeros(self.embedding_dimension, dtype=np.float32)
        
        # 创建TF-IDF向量
        vector = np.zeros(len(self.vocabulary), dtype=np.float32)
        for word, count in word_count.items():
            if word in self.vocabulary:
                word_idx = self.vocabulary[word]
                tf = np.float32(count) / np.float32(total_words)  # 确保为float32
                idf = np.float32(self.idf_scores.get(word, 0))  # 确保为float32
                vector[word_idx] = tf * idf
        
        # 如果向量维度不匹配，调整到指定维度
        if len(vector) != self.embedding_dimension:
            if len(vector) > self.embedding_dimension:
                vector = vector[:self.embedding_dimension]
            else:
                padded_vector = np.zeros(self.embedding_dimension, dtype=np.float32)
                padded_vector[:len(vector)] = vector
                vector = padded_vector
        
        return vector
    
    def encode(self, 
               texts: Union[str, List[str]], 
               normalize_embeddings: bool = True,
               show_progress: bool = True) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            texts: 文本或文本列表
            normalize_embeddings: 是否归一化向量 (简化版本忽略)
            show_progress: 是否显示进度 (简化版本忽略)
            
        Returns:
            向量数组
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        # 如果词汇表为空，先构建词汇表
        if not self.vocabulary:
            self._build_vocabulary(texts)
        
        try:
            embeddings = np.array([self._text_to_vector(text) for text in texts])
            logger.debug(f"编码完成: {len(texts)} 个文本 -> {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            raise
    
    def encode_batch(self, 
                    texts: List[str], 
                    batch_size: Optional[int] = None,
                    normalize_embeddings: bool = True) -> np.ndarray:
        """
        批量编码文本
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小 (简化版本忽略)
            normalize_embeddings: 是否归一化向量 (简化版本忽略)
            
        Returns:
            向量数组
        """
        # 简化版本直接调用encode方法
        return self.encode(texts, normalize_embeddings=normalize_embeddings)
    
    def similarity(self, 
                  embeddings1: np.ndarray, 
                  embeddings2: np.ndarray) -> np.ndarray:
        """
        计算向量相似度
        
        Args:
            embeddings1: 向量组1
            embeddings2: 向量组2
            
        Returns:
            相似度矩阵
        """
        try:
            # 确保向量已归一化
            if embeddings1.ndim == 1:
                embeddings1 = embeddings1.reshape(1, -1)
            if embeddings2.ndim == 1:
                embeddings2 = embeddings2.reshape(1, -1)
            
            # 计算余弦相似度
            similarities = np.dot(embeddings1, embeddings2.T)
            
            return similarities
            
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            raise
    
    def save_embeddings(self, 
                         embeddings: np.ndarray, 
                         texts: List[str],
                         save_path: Union[str, Path],
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        保存向量到文件
        
        Args:
            embeddings: 向量数组
            texts: 对应的文本列表
            save_path: 保存路径
            metadata: 元数据
            
        Returns:
            是否保存成功
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 准备保存数据
            save_data = {
                "embeddings": embeddings,
                "texts": texts,
                "vocabulary": self.vocabulary,
                "idf_scores": self.idf_scores,
                "model_info": self.get_model_info(),
                "metadata": metadata or {},
                "timestamp": str(Path().cwd())
            }
            
            # 保存为pickle文件
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"向量保存成功: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"向量保存失败: {e}")
            return False
    
    def load_embeddings(self, load_path: Union[str, Path]) -> tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        加载向量化结果
        
        Args:
            load_path: 输入文件路径
            
        Returns:
            (向量数组, 文本列表, 元数据)
        """
        try:
            load_path = Path(load_path)
            
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
            
            embeddings = data['embeddings']
            texts = data['texts']
            metadata = data.get('metadata', {})
            
            # 恢复词汇表和IDF分数
            if 'vocabulary' in data:
                self.vocabulary = data['vocabulary']
            if 'idf_scores' in data:
                self.idf_scores = data['idf_scores']
            
            logger.info(f"向量化结果已加载: {load_path}")
            logger.info(f"向量维度: {embeddings.shape}")
            
            return embeddings, texts, metadata
            
        except Exception as e:
            logger.error(f"加载向量化结果失败: {e}")
            raise
    
    def compute_similarity(self, input1: Union[str, np.ndarray], input2: Union[str, np.ndarray]) -> float:
        """
        计算两个向量或文本的余弦相似度
        
        Args:
            input1: 第一个向量或文本
            input2: 第二个向量或文本
            
        Returns:
            余弦相似度值
        """
        try:
            # 如果输入是字符串，先转换为向量
            if isinstance(input1, str):
                embedding1 = self.encode(input1).flatten()
            else:
                embedding1 = input1.flatten() if input1.ndim > 1 else input1
                
            if isinstance(input2, str):
                embedding2 = self.encode(input2).flatten()
            else:
                embedding2 = input2.flatten() if input2.ndim > 1 else input2
            
            # 计算余弦相似度
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"计算相似度失败: {e}")
            return 0.0
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        计算向量矩阵的相似度矩阵
        
        Args:
            embeddings: 向量矩阵 (n, d)
            
        Returns:
            相似度矩阵 (n, n)
        """
        try:
            n = embeddings.shape[0]
            similarity_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    similarity_matrix[i, j] = self.compute_similarity(embeddings[i], embeddings[j])
            
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"计算相似度矩阵失败: {e}")
            return np.zeros((embeddings.shape[0], embeddings.shape[0]))
    
    def get_vector_dimension(self) -> int:
        """
        获取向量维度
        
        Returns:
            向量维度
        """
        return self.embedding_dimension
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        try:
            return {
                "model_name": self.model_name,
                "device": self.device,
                "embedding_dimension": self.embedding_dimension,
                "vocabulary_size": len(self.vocabulary),
                "model_type": "TF-IDF"
            }
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            return {}