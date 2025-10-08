#!/usr/bin/env python3
"""
Lesson 06 - RAG系统实现模板
学生需要完成的RAG系统核心组件实现
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import time


@dataclass
class Document:
    """文档数据结构"""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any] = None


@dataclass
class SearchQuery:
    """搜索查询数据结构"""
    text: str
    top_k: int = 5
    similarity_threshold: float = 0.0
    filters: Dict[str, Any] = None


@dataclass
class SearchResult:
    """搜索结果数据结构"""
    document: Document
    score: float
    rank: int


@dataclass
class RAGResponse:
    """RAG响应数据结构"""
    question: str
    answer: str
    sources: List[SearchResult]
    context_length: int
    processing_time: float


# ============================================================================
# Exercise 1: 实现向量化服务
# ============================================================================

class EmbeddingService(ABC):
    """向量化服务抽象基类"""
    
    @abstractmethod
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """将文本列表转换为向量矩阵"""
        pass
    
    @abstractmethod
    def encode_single(self, text: str) -> np.ndarray:
        """将单个文本转换为向量"""
        pass


class MockEmbeddingService(EmbeddingService):
    """模拟向量化服务实现"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        # TODO: 初始化必要的参数
        pass
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Exercise 1.1: 实现批量文本向量化
        
        要求:
        1. 将文本列表转换为向量矩阵
        2. 确保向量已归一化
        3. 返回形状为 (len(texts), dimension) 的矩阵
        
        提示:
        - 可以使用文本的hash值作为随机种子
        - 使用numpy生成随机向量后进行归一化
        """
        # TODO: 实现批量向量化逻辑
        pass
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Exercise 1.2: 实现单个文本向量化
        
        要求:
        1. 将单个文本转换为向量
        2. 确保向量已归一化
        3. 返回形状为 (dimension,) 的向量
        """
        # TODO: 实现单个文本向量化逻辑
        pass


# ============================================================================
# Exercise 2: 实现相似度计算
# ============================================================================

class SimilarityCalculator(ABC):
    """相似度计算抽象基类"""
    
    @abstractmethod
    def calculate_similarity(self, query_vector: np.ndarray, 
                           doc_vectors: np.ndarray) -> np.ndarray:
        """计算查询向量与文档向量的相似度"""
        pass


class CosineSimilarityCalculator(SimilarityCalculator):
    """余弦相似度计算器"""
    
    def calculate_similarity(self, query_vector: np.ndarray, 
                           doc_vectors: np.ndarray) -> np.ndarray:
        """
        Exercise 2.1: 实现余弦相似度计算
        
        要求:
        1. 计算查询向量与所有文档向量的余弦相似度
        2. 返回相似度数组，长度等于文档数量
        3. 相似度值范围在[-1, 1]之间
        
        提示:
        - 余弦相似度 = dot(a, b) / (norm(a) * norm(b))
        - 如果向量已归一化，则余弦相似度 = dot(a, b)
        """
        # TODO: 实现余弦相似度计算
        pass


class EuclideanDistanceCalculator(SimilarityCalculator):
    """欧几里得距离计算器（转换为相似度）"""
    
    def calculate_similarity(self, query_vector: np.ndarray, 
                           doc_vectors: np.ndarray) -> np.ndarray:
        """
        Exercise 2.2: 实现基于欧几里得距离的相似度计算
        
        要求:
        1. 计算查询向量与所有文档向量的欧几里得距离
        2. 将距离转换为相似度（距离越小，相似度越高）
        3. 返回相似度数组
        
        提示:
        - 欧几里得距离 = sqrt(sum((a - b)^2))
        - 相似度转换: similarity = 1 / (1 + distance)
        """
        # TODO: 实现欧几里得距离计算和相似度转换
        pass


# ============================================================================
# Exercise 3: 实现向量检索器
# ============================================================================

class VectorRetriever:
    """向量检索器"""
    
    def __init__(self, embedding_service: EmbeddingService,
                 similarity_calculator: SimilarityCalculator):
        self.embedding_service = embedding_service
        self.similarity_calculator = similarity_calculator
        self.documents = []
        self.document_vectors = None
    
    def add_documents(self, documents: List[Document]):
        """
        Exercise 3.1: 实现文档添加功能
        
        要求:
        1. 将文档添加到内部存储
        2. 对文档内容进行向量化
        3. 更新文档向量矩阵
        
        提示:
        - 需要处理首次添加和追加添加两种情况
        - 使用np.vstack合并向量矩阵
        """
        # TODO: 实现文档添加逻辑
        pass
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Exercise 3.2: 实现向量检索功能
        
        要求:
        1. 对查询文本进行向量化
        2. 计算与所有文档的相似度
        3. 根据相似度阈值过滤结果
        4. 按相似度排序并返回TopK结果
        5. 构建SearchResult对象列表
        
        提示:
        - 使用similarity_calculator计算相似度
        - 注意处理空结果的情况
        - 确保返回结果按相似度降序排列
        """
        # TODO: 实现向量检索逻辑
        pass


# ============================================================================
# Exercise 4: 实现上下文构建器
# ============================================================================

class ContextBuilder:
    """上下文构建器"""
    
    def __init__(self, max_context_length: int = 2000,
                 context_template: str = None):
        self.max_context_length = max_context_length
        self.context_template = context_template or self._default_template()
    
    def _default_template(self) -> str:
        """默认上下文模板"""
        return "参考文档 {rank}: {title}\n内容: {content}\n相似度: {score:.3f}\n"
    
    def build_context(self, search_results: List[SearchResult]) -> str:
        """
        Exercise 4.1: 实现上下文构建功能
        
        要求:
        1. 根据搜索结果构建上下文字符串
        2. 控制上下文总长度不超过max_context_length
        3. 使用模板格式化每个文档片段
        4. 如果超长需要截断，优先保留高相似度的文档
        
        提示:
        - 按相似度顺序处理搜索结果
        - 累计计算上下文长度
        - 如果添加下一个文档会超长，考虑截断策略
        """
        # TODO: 实现上下文构建逻辑
        pass
    
    def estimate_context_length(self, search_results: List[SearchResult]) -> int:
        """
        Exercise 4.2: 实现上下文长度估算
        
        要求:
        1. 估算给定搜索结果构建的上下文总长度
        2. 不实际构建上下文，只计算长度
        3. 用于优化上下文构建策略
        """
        # TODO: 实现长度估算逻辑
        pass


# ============================================================================
# Exercise 5: 实现LLM生成服务
# ============================================================================

class LLMService(ABC):
    """大语言模型服务抽象基类"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """生成回答"""
        pass


class MockLLMService(LLMService):
    """模拟LLM服务"""
    
    def __init__(self, model_name: str = "mock-llm"):
        self.model_name = model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Exercise 5.1: 实现模拟LLM生成
        
        要求:
        1. 基于prompt内容生成合理的回答
        2. 模拟真实LLM的响应延迟
        3. 根据prompt中的问题类型生成不同风格的回答
        
        提示:
        - 可以分析prompt中的关键词来判断问题类型
        - 使用time.sleep模拟生成延迟
        - 生成的回答应该看起来像是基于上下文的
        """
        # TODO: 实现模拟LLM生成逻辑
        pass


class PromptBuilder:
    """Prompt构建器"""
    
    def __init__(self):
        self.system_prompt = self._default_system_prompt()
        self.user_template = self._default_user_template()
    
    def _default_system_prompt(self) -> str:
        """默认系统提示"""
        return """你是一个专业的AI助手，请根据提供的参考文档回答用户问题。

要求:
1. 只基于提供的参考文档回答问题
2. 如果文档中没有相关信息，请明确说明
3. 回答要准确、简洁、有条理
4. 可以引用具体的文档片段支持你的回答"""
    
    def _default_user_template(self) -> str:
        """默认用户提示模板"""
        return """参考文档:
{context}

用户问题: {question}

请基于上述参考文档回答问题:"""
    
    def build_prompt(self, question: str, context: str) -> str:
        """
        Exercise 5.2: 实现Prompt构建
        
        要求:
        1. 结合系统提示、上下文和用户问题构建完整prompt
        2. 确保prompt格式清晰、结构合理
        3. 适当控制prompt总长度
        
        提示:
        - 可以根据上下文长度调整prompt结构
        - 考虑添加必要的格式化和分隔符
        """
        # TODO: 实现Prompt构建逻辑
        pass


# ============================================================================
# Exercise 6: 实现完整RAG系统
# ============================================================================

class RAGSystem:
    """完整的RAG系统"""
    
    def __init__(self, 
                 embedding_service: EmbeddingService,
                 similarity_calculator: SimilarityCalculator,
                 llm_service: LLMService):
        self.retriever = VectorRetriever(embedding_service, similarity_calculator)
        self.context_builder = ContextBuilder()
        self.prompt_builder = PromptBuilder()
        self.llm_service = llm_service
    
    def add_documents(self, documents: List[Document]):
        """添加文档到知识库"""
        self.retriever.add_documents(documents)
    
    def query(self, question: str, top_k: int = 5, 
              similarity_threshold: float = 0.0) -> RAGResponse:
        """
        Exercise 6.1: 实现完整RAG查询流程
        
        要求:
        1. 创建搜索查询对象
        2. 执行向量检索
        3. 构建上下文
        4. 构建prompt
        5. 调用LLM生成回答
        6. 构建并返回RAGResponse对象
        7. 记录处理时间
        
        提示:
        - 使用time.time()记录开始和结束时间
        - 处理检索结果为空的情况
        - 确保所有组件正确协作
        """
        # TODO: 实现完整RAG查询流程
        pass
    
    def batch_query(self, questions: List[str], **kwargs) -> List[RAGResponse]:
        """
        Exercise 6.2: 实现批量查询功能
        
        要求:
        1. 对问题列表进行批量处理
        2. 返回对应的回答列表
        3. 可以考虑并行处理优化（可选）
        
        提示:
        - 可以简单地循环调用单个查询
        - 考虑错误处理和异常情况
        """
        # TODO: 实现批量查询逻辑
        pass


# ============================================================================
# Exercise 7: 系统评估与测试
# ============================================================================

class RAGEvaluator:
    """RAG系统评估器"""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
    
    def evaluate_retrieval_quality(self, test_queries: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Exercise 7.1: 实现检索质量评估
        
        要求:
        1. 评估检索结果的相关性
        2. 计算平均相似度分数
        3. 统计检索成功率（返回结果数 > 0）
        4. 返回评估指标字典
        
        测试查询格式:
        {
            "question": "问题文本",
            "expected_doc_ids": ["相关文档ID列表"],  # 可选
            "top_k": 5
        }
        """
        # TODO: 实现检索质量评估逻辑
        pass
    
    def evaluate_generation_quality(self, test_queries: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Exercise 7.2: 实现生成质量评估
        
        要求:
        1. 评估生成回答的质量
        2. 计算平均回答长度
        3. 统计回答完整性（非空回答比例）
        4. 返回评估指标字典
        
        测试查询格式:
        {
            "question": "问题文本",
            "expected_answer": "期望答案",  # 可选
            "top_k": 5
        }
        """
        # TODO: 实现生成质量评估逻辑
        pass
    
    def benchmark_performance(self, test_queries: List[str], 
                            iterations: int = 3) -> Dict[str, float]:
        """
        Exercise 7.3: 实现性能基准测试
        
        要求:
        1. 多次运行测试查询
        2. 统计平均响应时间
        3. 计算吞吐量（查询/秒）
        4. 分析性能瓶颈
        5. 返回性能指标字典
        """
        # TODO: 实现性能基准测试逻辑
        pass


# ============================================================================
# 测试和演示代码
# ============================================================================

def create_test_documents() -> List[Document]:
    """创建测试文档集合"""
    return [
        Document(
            id="doc1",
            title="机器学习基础",
            content="机器学习是人工智能的一个重要分支，通过算法让计算机从数据中学习模式。主要包括监督学习、无监督学习和强化学习三大类。"
        ),
        Document(
            id="doc2", 
            title="深度学习介绍",
            content="深度学习是机器学习的子领域，使用多层神经网络来学习数据的复杂表示。在图像识别、自然语言处理等领域取得了突破性进展。"
        ),
        Document(
            id="doc3",
            title="自然语言处理",
            content="自然语言处理（NLP）是计算机科学和人工智能的交叉领域，致力于让计算机理解和生成人类语言。包括文本分析、语言生成、机器翻译等任务。"
        )
    ]


def test_embedding_service():
    """测试向量化服务"""
    print("=== 测试向量化服务 ===")
    
    # TODO: 创建MockEmbeddingService实例并测试
    # embedding_service = MockEmbeddingService()
    # texts = ["测试文本1", "测试文本2"]
    # vectors = embedding_service.encode_texts(texts)
    # print(f"向量形状: {vectors.shape}")
    pass


def test_similarity_calculator():
    """测试相似度计算"""
    print("=== 测试相似度计算 ===")
    
    # TODO: 创建相似度计算器并测试
    # calculator = CosineSimilarityCalculator()
    # query_vector = np.random.random(384)
    # doc_vectors = np.random.random((3, 384))
    # similarities = calculator.calculate_similarity(query_vector, doc_vectors)
    # print(f"相似度: {similarities}")
    pass


def test_vector_retriever():
    """测试向量检索器"""
    print("=== 测试向量检索器 ===")
    
    # TODO: 创建检索器并测试
    # embedding_service = MockEmbeddingService()
    # similarity_calculator = CosineSimilarityCalculator()
    # retriever = VectorRetriever(embedding_service, similarity_calculator)
    # 
    # documents = create_test_documents()
    # retriever.add_documents(documents)
    # 
    # query = SearchQuery("什么是机器学习？", top_k=2)
    # results = retriever.search(query)
    # print(f"检索到 {len(results)} 个结果")
    pass


def test_rag_system():
    """测试完整RAG系统"""
    print("=== 测试完整RAG系统 ===")
    
    # TODO: 创建完整RAG系统并测试
    # embedding_service = MockEmbeddingService()
    # similarity_calculator = CosineSimilarityCalculator()
    # llm_service = MockLLMService()
    # 
    # rag = RAGSystem(embedding_service, similarity_calculator, llm_service)
    # rag.add_documents(create_test_documents())
    # 
    # response = rag.query("什么是深度学习？")
    # print(f"问题: {response.question}")
    # print(f"回答: {response.answer}")
    # print(f"处理时间: {response.processing_time:.2f}秒")
    pass


if __name__ == "__main__":
    print("Lesson 06 - RAG系统实现模板")
    print("=" * 50)
    print("请完成以下Exercise:")
    print("1. 实现向量化服务 (MockEmbeddingService)")
    print("2. 实现相似度计算 (CosineSimilarityCalculator, EuclideanDistanceCalculator)")
    print("3. 实现向量检索器 (VectorRetriever)")
    print("4. 实现上下文构建器 (ContextBuilder)")
    print("5. 实现LLM生成服务 (MockLLMService, PromptBuilder)")
    print("6. 实现完整RAG系统 (RAGSystem)")
    print("7. 实现系统评估与测试 (RAGEvaluator)")
    print("\n完成后运行测试函数验证实现:")
    
    # 运行测试（需要先完成实现）
    # test_embedding_service()
    # test_similarity_calculator()
    # test_vector_retriever()
    # test_rag_system()