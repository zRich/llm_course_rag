#!/usr/bin/env python3
"""
Lesson 06 - MVP RAG系统演示
演示完整的检索增强生成（RAG）系统实现
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class Document:
    """文档数据结构"""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any] = None


@dataclass
class RAGQuery:
    """RAG查询数据结构"""
    question: str
    top_k: int = 5
    similarity_threshold: float = 0.0
    max_context_length: int = 2000


@dataclass
class RAGResult:
    """RAG结果数据结构"""
    question: str
    answer: str
    retrieved_documents: List[Document]
    retrieval_scores: List[float]
    generation_time: float
    total_time: float
    context_used: str


class MockEmbeddingModel:
    """模拟向量化模型"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        np.random.seed(42)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """文本向量化"""
        vectors = []
        for text in texts:
            hash_value = hash(text) % 1000000
            np.random.seed(hash_value)
            vector = np.random.normal(0, 1, self.dimension)
            vector = vector / np.linalg.norm(vector)
            vectors.append(vector)
        return np.array(vectors)


class MockLLM:
    """模拟大语言模型"""
    
    def __init__(self, model_name: str = "mock-gpt-3.5"):
        self.model_name = model_name
    
    def generate(self, prompt: str) -> str:
        """生成回答"""
        # 模拟生成延迟
        time.sleep(0.1)
        
        # 基于prompt内容生成不同类型的回答
        if "什么是" in prompt or "介绍" in prompt:
            return self._generate_definition_answer(prompt)
        elif "如何" in prompt or "怎么" in prompt:
            return self._generate_howto_answer(prompt)
        elif "优势" in prompt or "好处" in prompt:
            return self._generate_advantage_answer(prompt)
        elif "区别" in prompt or "比较" in prompt:
            return self._generate_comparison_answer(prompt)
        else:
            return self._generate_general_answer(prompt)
    
    def _generate_definition_answer(self, prompt: str) -> str:
        return """根据提供的参考文档，我可以为您详细解释这个概念：

这是一个重要的技术概念，具有以下核心特征：
1. **基本定义**：从文档中可以看出，这个概念涉及多个技术层面
2. **主要特点**：具有高效性、可扩展性和实用性
3. **应用场景**：广泛应用于现代AI和数据处理系统中
4. **技术优势**：能够显著提升系统性能和用户体验

基于文档内容，这个概念在实际应用中具有重要价值。"""
    
    def _generate_howto_answer(self, prompt: str) -> str:
        return """基于参考文档，我为您提供详细的实施指导：

**实施步骤：**
1. **准备阶段**：
   - 确保所需的技术环境和依赖项已就绪
   - 准备必要的数据和配置文件

2. **配置阶段**：
   - 根据文档说明设置相关参数
   - 验证配置的正确性

3. **执行阶段**：
   - 按照最佳实践进行具体操作
   - 监控执行过程中的关键指标

4. **验证阶段**：
   - 检查结果是否符合预期
   - 进行必要的调整和优化

**注意事项**：
- 严格遵循文档中的安全建议
- 在生产环境部署前先在测试环境验证"""
    
    def _generate_advantage_answer(self, prompt: str) -> str:
        return """根据参考文档分析，主要优势包括：

**核心优势：**
1. **性能提升**：显著改善系统响应速度和处理能力
2. **准确性增强**：通过先进算法提高结果的准确性
3. **可扩展性**：支持大规模数据处理和高并发访问
4. **易用性**：提供友好的接口和完善的文档支持

**实际价值：**
- 降低开发和维护成本
- 提升用户体验和满意度
- 增强系统的可靠性和稳定性
- 支持业务快速发展和扩展

这些优势使其成为现代技术架构中的重要组成部分。"""
    
    def _generate_comparison_answer(self, prompt: str) -> str:
        return """基于参考文档，我为您分析主要区别：

**技术对比：**

**方案A的特点：**
- 实现相对简单，学习成本较低
- 适合小到中等规模的应用场景
- 资源消耗较少，部署成本低

**方案B的特点：**
- 功能更加强大和完善
- 适合大规模、高性能要求的场景
- 需要更多的技术投入和维护

**选择建议：**
- 如果是初期项目或资源有限，建议选择方案A
- 如果需要高性能和完整功能，建议选择方案B
- 可以考虑分阶段实施，先用方案A验证，再升级到方案B

具体选择应根据实际需求和资源情况决定。"""
    
    def _generate_general_answer(self, prompt: str) -> str:
        return """根据提供的参考文档，我理解您的问题并提供以下分析：

**核心观点：**
基于文档内容，这个问题涉及多个重要方面，需要综合考虑技术、业务和实施等因素。

**详细分析：**
1. **技术层面**：文档中提到的技术方案具有良好的可行性
2. **实施层面**：需要考虑资源投入和时间安排
3. **风险层面**：存在一些潜在风险，但都有相应的解决方案
4. **收益层面**：预期能够带来显著的价值提升

**建议：**
- 建议按照文档中的最佳实践进行实施
- 注意关键风险点的控制和监控
- 保持与相关团队的密切沟通

如需更具体的指导，建议参考文档中的详细技术说明。"""


class VectorRetriever:
    """向量检索器"""
    
    def __init__(self):
        self.embedding_model = MockEmbeddingModel()
        self.documents = []
        self.vectors = None
    
    def add_documents(self, documents: List[Document]):
        """添加文档"""
        self.documents.extend(documents)
        
        texts = [doc.content for doc in documents]
        new_vectors = self.embedding_model.encode(texts)
        
        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])
    
    def search(self, query: str, top_k: int = 5, 
               threshold: float = 0.0) -> tuple[List[Document], List[float]]:
        """执行向量检索"""
        if not self.documents:
            return [], []
        
        # 向量化查询
        query_vector = self.embedding_model.encode([query])[0]
        
        # 计算相似度
        similarities = []
        for i, doc_vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, doc_vector)
            if similarity >= threshold:
                similarities.append((i, similarity))
        
        # 排序并取TopK
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        # 构建结果
        retrieved_docs = []
        scores = []
        for doc_idx, score in top_results:
            retrieved_docs.append(self.documents[doc_idx])
            scores.append(score)
        
        return retrieved_docs, scores


class ContextBuilder:
    """上下文构建器"""
    
    def __init__(self, max_length: int = 2000):
        self.max_length = max_length
    
    def build_context(self, documents: List[Document], 
                     scores: List[float]) -> str:
        """构建上下文"""
        context_parts = []
        current_length = 0
        
        for i, (doc, score) in enumerate(zip(documents, scores)):
            # 格式化文档片段
            doc_text = f"文档{i+1} (相似度: {score:.3f}): {doc.content}"
            
            # 检查长度限制
            if current_length + len(doc_text) > self.max_length:
                # 截断最后一个文档
                remaining_length = self.max_length - current_length
                if remaining_length > 100:  # 至少保留100字符
                    doc_text = doc_text[:remaining_length] + "..."
                    context_parts.append(doc_text)
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n\n".join(context_parts)


class RAGPipeline:
    """RAG流水线"""
    
    def __init__(self):
        self.retriever = VectorRetriever()
        self.context_builder = ContextBuilder()
        self.llm = MockLLM()
        
        # Prompt模板
        self.prompt_template = """你是一个专业的AI助手，请根据以下参考文档回答用户问题。

参考文档：
{context}

用户问题：{question}

请注意：
1. 只基于提供的参考文档回答问题
2. 如果文档中没有相关信息，请明确说明
3. 回答要准确、简洁、有条理
4. 可以引用具体的文档片段支持你的回答

回答："""
    
    def add_documents(self, documents: List[Document]):
        """添加文档到知识库"""
        self.retriever.add_documents(documents)
        print(f"已添加 {len(documents)} 个文档到知识库")
    
    def query(self, rag_query: RAGQuery) -> RAGResult:
        """执行RAG查询"""
        start_time = time.time()
        
        # 1. 向量检索
        retrieved_docs, scores = self.retriever.search(
            rag_query.question,
            top_k=rag_query.top_k,
            threshold=rag_query.similarity_threshold
        )
        
        if not retrieved_docs:
            return RAGResult(
                question=rag_query.question,
                answer="抱歉，我在知识库中没有找到相关信息来回答您的问题。",
                retrieved_documents=[],
                retrieval_scores=[],
                generation_time=0.0,
                total_time=time.time() - start_time,
                context_used=""
            )
        
        # 2. 构建上下文
        context = self.context_builder.build_context(retrieved_docs, scores)
        
        # 3. 生成Prompt
        prompt = self.prompt_template.format(
            context=context,
            question=rag_query.question
        )
        
        # 4. LLM生成
        generation_start = time.time()
        answer = self.llm.generate(prompt)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        return RAGResult(
            question=rag_query.question,
            answer=answer,
            retrieved_documents=retrieved_docs,
            retrieval_scores=scores,
            generation_time=generation_time,
            total_time=total_time,
            context_used=context
        )


def create_sample_knowledge_base() -> List[Document]:
    """创建示例知识库"""
    documents = [
        Document(
            id="doc1",
            title="RAG系统介绍",
            content="RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI架构。它通过从外部知识库检索相关信息，然后将这些信息作为上下文提供给大语言模型，从而生成更准确、更有根据的回答。RAG系统的核心优势在于能够利用最新的、特定领域的知识，而不需要重新训练模型。"
        ),
        Document(
            id="doc2",
            title="向量检索技术",
            content="向量检索是RAG系统的核心组件之一。它将文本转换为高维向量表示，然后通过计算向量间的相似度来找到最相关的文档片段。常用的相似度计算方法包括余弦相似度、点积相似度和欧几里得距离。向量检索的质量直接影响RAG系统的最终性能。"
        ),
        Document(
            id="doc3",
            title="大语言模型集成",
            content="在RAG系统中，大语言模型负责基于检索到的上下文生成最终答案。常用的模型包括GPT系列、Claude、以及各种开源模型如Llama。模型的选择需要考虑性能、成本、延迟等因素。Prompt工程在这个环节特别重要，好的Prompt设计能显著提升生成质量。"
        ),
        Document(
            id="doc4",
            title="系统架构设计",
            content="一个完整的RAG系统通常包括文档预处理、向量化、向量存储、检索服务、上下文构建和生成服务等组件。系统需要考虑可扩展性、可靠性和性能优化。常见的架构模式包括微服务架构、事件驱动架构等。缓存机制对于提升系统性能也很重要。"
        ),
        Document(
            id="doc5",
            title="性能优化策略",
            content="RAG系统的性能优化可以从多个角度进行：1）检索优化：使用更好的向量化模型、优化索引结构、实施重排序；2）生成优化：选择合适的模型、优化Prompt、使用流式生成；3）系统优化：实施缓存、负载均衡、异步处理。监控和日志对于识别性能瓶颈也很重要。"
        ),
        Document(
            id="doc6",
            title="评估与测试",
            content="RAG系统的评估需要考虑检索质量和生成质量两个方面。检索质量可以用召回率、精确率、NDCG等指标衡量。生成质量可以用BLEU、ROUGE、人工评估等方法评估。A/B测试在实际部署中也很重要。建立完善的测试集和评估流程是确保系统质量的关键。"
        )
    ]
    return documents


def demo_basic_rag():
    """演示基础RAG功能"""
    print("=== 基础RAG系统演示 ===")
    
    # 创建RAG系统
    rag = RAGPipeline()
    
    # 添加知识库
    documents = create_sample_knowledge_base()
    rag.add_documents(documents)
    
    # 测试查询
    test_queries = [
        "什么是RAG系统？",
        "向量检索是如何工作的？",
        "如何优化RAG系统的性能？",
        "RAG系统有什么优势？"
    ]
    
    for question in test_queries:
        print(f"\n问题: {question}")
        print("-" * 60)
        
        query = RAGQuery(question=question, top_k=3)
        result = rag.query(query)
        
        print(f"回答: {result.answer}")
        print(f"\n检索到的文档:")
        for i, (doc, score) in enumerate(zip(result.retrieved_documents, result.retrieval_scores)):
            print(f"  {i+1}. {doc.title} (相似度: {score:.3f})")
        
        print(f"\n性能指标:")
        print(f"  生成时间: {result.generation_time:.2f}秒")
        print(f"  总时间: {result.total_time:.2f}秒")


def demo_advanced_queries():
    """演示高级查询功能"""
    print("\n=== 高级查询演示 ===")
    
    rag = RAGPipeline()
    documents = create_sample_knowledge_base()
    rag.add_documents(documents)
    
    # 测试不同参数配置
    test_configs = [
        ("标准配置", RAGQuery("RAG系统的核心组件有哪些？", top_k=3)),
        ("高召回配置", RAGQuery("RAG系统的核心组件有哪些？", top_k=5, similarity_threshold=0.1)),
        ("高精度配置", RAGQuery("RAG系统的核心组件有哪些？", top_k=2, similarity_threshold=0.3))
    ]
    
    for config_name, query in test_configs:
        print(f"\n--- {config_name} ---")
        print(f"参数: top_k={query.top_k}, threshold={query.similarity_threshold}")
        
        result = rag.query(query)
        
        print(f"检索到 {len(result.retrieved_documents)} 个文档")
        print(f"回答长度: {len(result.answer)} 字符")
        print(f"总时间: {result.total_time:.2f}秒")


def demo_context_management():
    """演示上下文管理"""
    print("\n=== 上下文管理演示 ===")
    
    rag = RAGPipeline()
    documents = create_sample_knowledge_base()
    rag.add_documents(documents)
    
    # 测试长查询
    long_question = "请详细介绍RAG系统的完整架构，包括各个组件的作用、技术实现、性能优化策略，以及在实际部署中需要注意的问题。"
    
    query = RAGQuery(question=long_question, top_k=5, max_context_length=1000)
    result = rag.query(query)
    
    print(f"问题: {long_question}")
    print(f"\n上下文长度: {len(result.context_used)} 字符")
    print(f"检索文档数: {len(result.retrieved_documents)}")
    print(f"\n回答: {result.answer}")


def demo_performance_analysis():
    """演示性能分析"""
    print("\n=== 性能分析演示 ===")
    
    rag = RAGPipeline()
    documents = create_sample_knowledge_base()
    rag.add_documents(documents)
    
    # 批量测试
    test_questions = [
        "RAG是什么？",
        "如何实现向量检索？",
        "系统架构设计要点",
        "性能优化方法",
        "评估指标有哪些？"
    ]
    
    total_time = 0
    total_generation_time = 0
    
    print("批量查询性能测试:")
    for i, question in enumerate(test_questions, 1):
        query = RAGQuery(question=question, top_k=3)
        result = rag.query(query)
        
        total_time += result.total_time
        total_generation_time += result.generation_time
        
        print(f"  查询 {i}: {result.total_time:.2f}秒 (生成: {result.generation_time:.2f}秒)")
    
    print(f"\n性能统计:")
    print(f"  平均总时间: {total_time/len(test_questions):.2f}秒")
    print(f"  平均生成时间: {total_generation_time/len(test_questions):.2f}秒")
    print(f"  检索时间占比: {((total_time-total_generation_time)/total_time)*100:.1f}%")


if __name__ == "__main__":
    print("Lesson 06 - MVP RAG系统演示")
    print("=" * 60)
    
    # 运行演示
    demo_basic_rag()
    demo_advanced_queries()
    demo_context_management()
    demo_performance_analysis()
    
    print("\n演示完成！")
    print("\n关键要点:")
    print("1. RAG系统将检索和生成有机结合，提供准确的知识问答")
    print("2. 向量检索质量直接影响最终答案的准确性")
    print("3. 上下文管理需要平衡信息完整性和长度限制")
    print("4. 系统性能优化需要从检索、生成、架构等多个维度考虑")
    print("5. 完善的评估体系是确保系统质量的基础")