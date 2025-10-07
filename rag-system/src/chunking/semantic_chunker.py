import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import re

from .chunker import DocumentChunker, DocumentChunk, ChunkingConfig
from .sentence_chunker import SentenceChunker

logger = logging.getLogger(__name__)

class SemanticChunker(DocumentChunker):
    """基于语义的文档分块器
    
    使用机器学习方法分析文本语义相似性，进行智能分块
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        super().__init__(config)
        
        # 语义分析配置
        self.similarity_threshold = 0.3  # 语义相似度阈值
        self.min_sentences_per_chunk = 2  # 每个块最少句子数
        self.max_sentences_per_chunk = 10  # 每个块最多句子数
        
        # TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        # 句子分块器（用于预处理）
        self.sentence_chunker = SentenceChunker(config)
    
    def get_chunker_type(self) -> str:
        """获取分块器类型"""
        return "semantic"
    
    def chunk_text(self, text: str, source_file: str = "") -> List[DocumentChunk]:
        """基于语义相似性分割文本
        
        Args:
            text: 要分割的文本
            source_file: 源文件路径
            
        Returns:
            List[DocumentChunk]: 分割后的文档块列表
        """
        try:
            # 首先按句子分割
            sentences = self._extract_sentences(text)
            
            if len(sentences) < 2:
                # 句子太少，直接返回单个块
                return [self._create_chunk(text, 0, len(text), source_file, 0)]
            
            # 计算句子向量
            sentence_vectors = self._compute_sentence_vectors(sentences)
            
            if sentence_vectors is None or len(sentence_vectors) == 0:
                # 向量化失败，回退到句子分块
                self.logger.warning("语义向量化失败，回退到句子分块")
                return self.sentence_chunker.chunk_text(text, source_file)
            
            # 基于语义相似性分组
            sentence_groups = self._group_sentences_by_similarity(
                sentences, sentence_vectors
            )
            
            # 创建语义块
            chunks = self._create_semantic_chunks(
                sentence_groups, text, source_file
            )
            
            self.logger.info(f"语义分块完成: {len(sentences)} 个句子 -> {len(chunks)} 个语义块")
            return chunks
            
        except Exception as e:
            self.logger.error(f"语义分块失败: {e}")
            # 回退到句子分块
            return self.sentence_chunker.chunk_text(text, source_file)
    
    def _extract_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """提取句子及其位置信息
        
        Args:
            text: 文本内容
            
        Returns:
            List[Tuple[str, int, int]]: 句子列表，包含(内容, 开始位置, 结束位置)
        """
        language = self.sentence_chunker._detect_text_language(text)
        return self.sentence_chunker._split_sentences(text, language)
    
    def _compute_sentence_vectors(self, sentences: List[Tuple[str, int, int]]) -> Optional[np.ndarray]:
        """计算句子的TF-IDF向量
        
        Args:
            sentences: 句子列表
            
        Returns:
            Optional[np.ndarray]: 句子向量矩阵
        """
        try:
            # 提取句子文本
            sentence_texts = [sentence[0] for sentence in sentences]
            
            # 预处理句子
            processed_sentences = [self._preprocess_sentence(s) for s in sentence_texts]
            
            # 过滤空句子
            valid_sentences = [s for s in processed_sentences if s.strip()]
            
            if len(valid_sentences) < 2:
                return None
            
            # 计算TF-IDF向量
            tfidf_matrix = self.vectorizer.fit_transform(valid_sentences)
            
            return tfidf_matrix.toarray()
            
        except Exception as e:
            self.logger.error(f"计算句子向量失败: {e}")
            return None
    
    def _preprocess_sentence(self, sentence: str) -> str:
        """预处理句子
        
        Args:
            sentence: 原始句子
            
        Returns:
            str: 预处理后的句子
        """
        # 转换为小写
        sentence = sentence.lower()
        
        # 移除特殊字符（保留中文字符）
        sentence = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', sentence)
        
        # 标准化空白字符
        sentence = re.sub(r'\s+', ' ', sentence)
        
        return sentence.strip()
    
    def _group_sentences_by_similarity(self, sentences: List[Tuple[str, int, int]], 
                                     vectors: np.ndarray) -> List[List[int]]:
        """基于语义相似性对句子分组
        
        Args:
            sentences: 句子列表
            vectors: 句子向量矩阵
            
        Returns:
            List[List[int]]: 句子分组，每个组包含句子索引列表
        """
        if len(vectors) < 2:
            return [[0]] if len(vectors) == 1 else []
        
        try:
            # 方法1: 基于相似度阈值的贪心分组
            groups = self._greedy_similarity_grouping(vectors)
            
            # 方法2: 如果分组效果不好，尝试聚类
            if self._should_use_clustering(groups, len(sentences)):
                groups = self._cluster_based_grouping(vectors)
            
            # 后处理分组
            groups = self._post_process_groups(groups, sentences)
            
            return groups
            
        except Exception as e:
            self.logger.error(f"句子分组失败: {e}")
            # 回退到顺序分组
            return self._sequential_grouping(len(sentences))
    
    def _greedy_similarity_grouping(self, vectors: np.ndarray) -> List[List[int]]:
        """基于相似度阈值的贪心分组
        
        Args:
            vectors: 句子向量矩阵
            
        Returns:
            List[List[int]]: 句子分组
        """
        n_sentences = len(vectors)
        groups = []
        used = set()
        
        for i in range(n_sentences):
            if i in used:
                continue
            
            current_group = [i]
            used.add(i)
            
            # 查找相似的句子
            for j in range(i + 1, n_sentences):
                if j in used:
                    continue
                
                # 计算与当前组的平均相似度
                similarities = []
                for group_idx in current_group:
                    sim = cosine_similarity([vectors[group_idx]], [vectors[j]])[0][0]
                    similarities.append(sim)
                
                avg_similarity = np.mean(similarities)
                
                # 如果相似度超过阈值且组大小未超限，加入当前组
                if (avg_similarity >= self.similarity_threshold and 
                    len(current_group) < self.max_sentences_per_chunk):
                    current_group.append(j)
                    used.add(j)
            
            groups.append(current_group)
        
        return groups
    
    def _cluster_based_grouping(self, vectors: np.ndarray) -> List[List[int]]:
        """基于聚类的分组
        
        Args:
            vectors: 句子向量矩阵
            
        Returns:
            List[List[int]]: 句子分组
        """
        try:
            n_sentences = len(vectors)
            
            # 估算聚类数量
            n_clusters = max(2, min(n_sentences // 3, n_sentences // self.min_sentences_per_chunk))
            
            # K-means聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(vectors)
            
            # 将聚类结果转换为分组
            groups = [[] for _ in range(n_clusters)]
            for i, label in enumerate(cluster_labels):
                groups[label].append(i)
            
            # 过滤空组
            groups = [group for group in groups if group]
            
            return groups
            
        except Exception as e:
            self.logger.error(f"聚类分组失败: {e}")
            return self._sequential_grouping(len(vectors))
    
    def _should_use_clustering(self, groups: List[List[int]], total_sentences: int) -> bool:
        """判断是否应该使用聚类方法
        
        Args:
            groups: 当前分组结果
            total_sentences: 总句子数
            
        Returns:
            bool: 是否使用聚类
        """
        if not groups:
            return True
        
        # 如果分组数量太少或太多，考虑使用聚类
        if len(groups) < 2 or len(groups) > total_sentences // 2:
            return True
        
        # 如果有太多单句子组，考虑使用聚类
        single_sentence_groups = sum(1 for group in groups if len(group) == 1)
        if single_sentence_groups > len(groups) // 2:
            return True
        
        return False
    
    def _sequential_grouping(self, n_sentences: int) -> List[List[int]]:
        """顺序分组（回退方法）
        
        Args:
            n_sentences: 句子总数
            
        Returns:
            List[List[int]]: 句子分组
        """
        groups = []
        current_group = []
        
        for i in range(n_sentences):
            current_group.append(i)
            
            if len(current_group) >= self.max_sentences_per_chunk:
                groups.append(current_group)
                current_group = []
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _post_process_groups(self, groups: List[List[int]], 
                           sentences: List[Tuple[str, int, int]]) -> List[List[int]]:
        """后处理分组结果
        
        Args:
            groups: 原始分组
            sentences: 句子列表
            
        Returns:
            List[List[int]]: 处理后的分组
        """
        processed_groups = []
        
        for group in groups:
            if not group:
                continue
            
            # 确保组内句子按顺序排列
            group.sort()
            
            # 如果组太小，尝试合并到相邻组
            if len(group) < self.min_sentences_per_chunk and processed_groups:
                # 合并到前一个组
                processed_groups[-1].extend(group)
                processed_groups[-1].sort()
            else:
                processed_groups.append(group)
        
        # 检查最后一个组是否太小
        if (len(processed_groups) > 1 and 
            len(processed_groups[-1]) < self.min_sentences_per_chunk):
            # 合并到前一个组
            processed_groups[-2].extend(processed_groups[-1])
            processed_groups[-2].sort()
            processed_groups.pop()
        
        return processed_groups
    
    def _create_semantic_chunks(self, sentence_groups: List[List[int]], 
                              original_text: str, source_file: str) -> List[DocumentChunk]:
        """创建语义块
        
        Args:
            sentence_groups: 句子分组
            original_text: 原始文本
            source_file: 源文件路径
            
        Returns:
            List[DocumentChunk]: 语义块列表
        """
        chunks = []
        sentences = self._extract_sentences(original_text)
        
        for chunk_index, group in enumerate(sentence_groups):
            if not group:
                continue
            
            # 获取组内句子
            group_sentences = [sentences[i] for i in group]
            
            # 计算块的位置范围
            start_pos = min(s[1] for s in group_sentences)
            end_pos = max(s[2] for s in group_sentences)
            
            # 合并句子内容
            content_parts = [s[0] for s in group_sentences]
            content = ' '.join(content_parts)
            
            # 创建块
            chunk = self._create_chunk(content, start_pos, end_pos, source_file, chunk_index)
            
            # 添加语义相关的元数据
            chunk.metadata.chunk_type = "semantic"
            chunk.metadata.custom_metadata.update({
                'sentence_count': len(group),
                'sentence_indices': group,
                'semantic_coherence': self._calculate_coherence_score(group_sentences)
            })
            
            chunks.append(chunk)
        
        return chunks
    
    def _calculate_coherence_score(self, sentences: List[Tuple[str, int, int]]) -> float:
        """计算句子组的语义连贯性分数
        
        Args:
            sentences: 句子列表
            
        Returns:
            float: 连贯性分数 (0-1)
        """
        try:
            if len(sentences) < 2:
                return 1.0
            
            sentence_texts = [s[0] for s in sentences]
            processed_sentences = [self._preprocess_sentence(s) for s in sentence_texts]
            
            # 计算句子向量
            vectors = self.vectorizer.transform(processed_sentences).toarray()
            
            # 计算平均相似度
            similarities = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    sim = cosine_similarity([vectors[i]], [vectors[j]])[0][0]
                    similarities.append(sim)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            self.logger.error(f"计算连贯性分数失败: {e}")
            return 0.0
    
    def analyze_semantic_structure(self, text: str) -> Dict[str, Any]:
        """分析文本的语义结构
        
        Args:
            text: 文本内容
            
        Returns:
            Dict[str, Any]: 语义结构分析结果
        """
        try:
            sentences = self._extract_sentences(text)
            
            if len(sentences) < 2:
                return {
                    'total_sentences': len(sentences),
                    'semantic_groups': 1,
                    'avg_group_size': len(sentences),
                    'coherence_score': 1.0
                }
            
            # 计算句子向量
            vectors = self._compute_sentence_vectors(sentences)
            
            if vectors is None:
                return {'error': '无法计算语义向量'}
            
            # 分组分析
            groups = self._group_sentences_by_similarity(sentences, vectors)
            
            # 计算统计信息
            group_sizes = [len(group) for group in groups]
            
            # 计算整体连贯性
            overall_coherence = self._calculate_overall_coherence(vectors)
            
            return {
                'total_sentences': len(sentences),
                'semantic_groups': len(groups),
                'avg_group_size': np.mean(group_sizes) if group_sizes else 0,
                'min_group_size': min(group_sizes) if group_sizes else 0,
                'max_group_size': max(group_sizes) if group_sizes else 0,
                'coherence_score': overall_coherence,
                'group_details': [
                    {
                        'group_id': i,
                        'sentence_count': len(group),
                        'sentence_indices': group,
                        'coherence': self._calculate_coherence_score([sentences[j] for j in group])
                    }
                    for i, group in enumerate(groups)
                ]
            }
            
        except Exception as e:
            self.logger.error(f"语义结构分析失败: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_coherence(self, vectors: np.ndarray) -> float:
        """计算整体语义连贯性
        
        Args:
            vectors: 句子向量矩阵
            
        Returns:
            float: 整体连贯性分数
        """
        try:
            if len(vectors) < 2:
                return 1.0
            
            # 计算所有句子对的相似度
            similarities = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    sim = cosine_similarity([vectors[i]], [vectors[j]])[0][0]
                    similarities.append(sim)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            self.logger.error(f"计算整体连贯性失败: {e}")
            return 0.0