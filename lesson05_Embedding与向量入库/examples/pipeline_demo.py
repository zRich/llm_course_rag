#!/usr/bin/env python3
"""
Lesson 05: Embedding与向量入库 - 完整向量化流水线演示

本文件演示完整的向量化流水线，包括：
1. 文档预处理和分块
2. 批量向量化处理
3. 向量数据库入库
4. 质量检查和验证
5. 性能监控和优化

作者: RAG课程团队
日期: 2024-01-01
用途: Lesson 05课堂演示
"""

import time
import uuid
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Generator
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import numpy as np


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """文档分块数据结构"""
    id: str
    text: str
    source: str
    chunk_index: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VectorData:
    """向量数据结构"""
    id: str
    vector: List[float]
    chunk: DocumentChunk
    
    def to_point(self) -> PointStruct:
        """转换为Qdrant Point结构"""
        return PointStruct(
            id=self.id,
            vector=self.vector,
            payload={
                "text": self.chunk.text,
                "source": self.chunk.source,
                "chunk_index": self.chunk.chunk_index,
                "text_length": len(self.chunk.text),
                "text_hash": hashlib.md5(self.chunk.text.encode()).hexdigest(),
                **self.chunk.metadata
            }
        )


@dataclass
class PipelineStats:
    """流水线统计信息"""
    total_documents: int = 0
    total_chunks: int = 0
    total_vectors: int = 0
    processing_time: float = 0.0
    vectorization_time: float = 0.0
    insertion_time: float = 0.0
    average_chunk_length: float = 0.0
    success_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_text(self, text: str, source: str = "unknown") -> List[DocumentChunk]:
        """处理文本并分块"""
        logger.info(f"处理文档: {source}, 长度: {len(text)}字符")
        
        # 简单的分块策略（按句子分割）
        sentences = self._split_sentences(text)
        chunks = self._create_chunks(sentences, source)
        
        logger.info(f"文档分块完成: {len(chunks)}个块")
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        # 简单的句子分割（可以使用更复杂的NLP工具）
        import re
        sentences = re.split(r'[.!?。！？]\s*', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunks(self, sentences: List[str], source: str) -> List[DocumentChunk]:
        """创建文档块"""
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            # 检查是否需要创建新块
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # 创建当前块
                chunk = DocumentChunk(
                    id=str(uuid.uuid4()),
                    text=current_chunk.strip(),
                    source=source,
                    chunk_index=chunk_index,
                    metadata={
                        "sentence_count": len(current_chunk.split('.')),
                        "created_at": time.time(),
                        "language": self._detect_language(current_chunk)
                    }
                )
                chunks.append(chunk)
                
                # 准备下一个块（考虑重叠）
                if self.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
                
                chunk_index += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # 处理最后一个块
        if current_chunk.strip():
            chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                text=current_chunk.strip(),
                source=source,
                chunk_index=chunk_index,
                metadata={
                    "sentence_count": len(current_chunk.split('.')),
                    "created_at": time.time(),
                    "language": self._detect_language(current_chunk)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _detect_language(self, text: str) -> str:
        """简单的语言检测"""
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        return "zh" if chinese_chars > len(text) * 0.3 else "en"


class VectorPipeline:
    """向量化流水线"""
    
    def __init__(self, 
                 model_name: str = "BAAI/bge-m3",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 collection_name: str = "pipeline_demo",
                 batch_size: int = 32,
                 max_workers: int = 4):
        
        self.model_name = model_name
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # 初始化组件
        self.model = None
        self.client = None
        self.doc_processor = DocumentProcessor()
        self.stats = PipelineStats()
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """初始化流水线组件"""
        logger.info("初始化向量化流水线...")
        
        # 加载向量化模型
        logger.info(f"加载向量化模型: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # 连接Qdrant数据库
        logger.info(f"连接Qdrant数据库: {self.qdrant_host}:{self.qdrant_port}")
        self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        
        # 创建或重置集合
        self._setup_collection()
        
        logger.info("流水线初始化完成")
    
    def _setup_collection(self) -> None:
        """设置Qdrant集合"""
        logger.info(f"设置集合: {self.collection_name}")
        
        # 检查集合是否存在
        collections = self.client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if self.collection_name in collection_names:
            logger.info(f"删除现有集合: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
        
        # 创建新集合
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=768,  # bge-m3模型维度
                distance=Distance.COSINE
            )
        )
        
        logger.info(f"集合创建成功: {self.collection_name}")
    
    def process_documents(self, documents: List[Tuple[str, str]]) -> List[DocumentChunk]:
        """处理多个文档"""
        logger.info(f"开始处理 {len(documents)} 个文档")
        
        all_chunks = []
        start_time = time.time()
        
        for i, (text, source) in enumerate(documents, 1):
            logger.info(f"处理文档 {i}/{len(documents)}: {source}")
            chunks = self.doc_processor.process_text(text, source)
            all_chunks.extend(chunks)
        
        processing_time = time.time() - start_time
        
        # 更新统计信息
        self.stats.total_documents = len(documents)
        self.stats.total_chunks = len(all_chunks)
        self.stats.processing_time = processing_time
        self.stats.average_chunk_length = np.mean([len(chunk.text) for chunk in all_chunks])
        
        logger.info(f"文档处理完成: {len(all_chunks)}个块, 耗时: {processing_time:.2f}秒")
        return all_chunks
    
    def vectorize_chunks(self, chunks: List[DocumentChunk]) -> List[VectorData]:
        """批量向量化文档块"""
        logger.info(f"开始向量化 {len(chunks)} 个文档块")
        
        start_time = time.time()
        vector_data_list = []
        
        # 批量处理
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            batch_texts = [chunk.text for chunk in batch_chunks]
            
            logger.info(f"向量化批次 {i//self.batch_size + 1}/{(len(chunks)-1)//self.batch_size + 1}")
            
            # 批量向量化
            batch_vectors = self.model.encode(batch_texts, show_progress_bar=False)
            
            # 创建向量数据
            for chunk, vector in zip(batch_chunks, batch_vectors):
                vector_data = VectorData(
                    id=chunk.id,
                    vector=vector.tolist(),
                    chunk=chunk
                )
                vector_data_list.append(vector_data)
        
        vectorization_time = time.time() - start_time
        
        # 更新统计信息
        self.stats.total_vectors = len(vector_data_list)
        self.stats.vectorization_time = vectorization_time
        
        logger.info(f"向量化完成: {len(vector_data_list)}个向量, 耗时: {vectorization_time:.2f}秒")
        return vector_data_list
    
    def insert_vectors(self, vector_data_list: List[VectorData]) -> bool:
        """批量插入向量到数据库"""
        logger.info(f"开始插入 {len(vector_data_list)} 个向量")
        
        start_time = time.time()
        success_count = 0
        
        try:
            # 转换为Qdrant Points
            points = [vd.to_point() for vd in vector_data_list]
            
            # 批量插入
            for i in range(0, len(points), self.batch_size):
                batch_points = points[i:i + self.batch_size]
                
                logger.info(f"插入批次 {i//self.batch_size + 1}/{(len(points)-1)//self.batch_size + 1}")
                
                result = self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
                
                if result.status == "completed":
                    success_count += len(batch_points)
                else:
                    logger.warning(f"批次插入失败: {result.status}")
            
            insertion_time = time.time() - start_time
            
            # 更新统计信息
            self.stats.insertion_time = insertion_time
            self.stats.success_rate = success_count / len(vector_data_list)
            
            logger.info(f"向量插入完成: {success_count}/{len(vector_data_list)}, 耗时: {insertion_time:.2f}秒")
            return success_count == len(vector_data_list)
            
        except Exception as e:
            logger.error(f"向量插入失败: {e}")
            return False
    
    def validate_pipeline(self, sample_queries: List[str] = None) -> Dict[str, Any]:
        """验证流水线结果"""
        logger.info("开始验证流水线结果")
        
        validation_results = {
            "collection_info": {},
            "sample_search": [],
            "data_quality": {}
        }
        
        try:
            # 1. 检查集合信息
            collection_info = self.client.get_collection(self.collection_name)
            validation_results["collection_info"] = {
                "name": self.collection_name,
                "status": collection_info.status,
                "points_count": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size
            }
            
            logger.info(f"集合验证: {collection_info.points_count}个向量")
            
            # 2. 样本搜索测试
            if sample_queries:
                for query in sample_queries[:3]:  # 限制查询数量
                    query_vector = self.model.encode([query])[0]
                    search_results = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=query_vector.tolist(),
                        limit=3,
                        with_payload=True
                    )
                    
                    validation_results["sample_search"].append({
                        "query": query,
                        "results_count": len(search_results),
                        "top_score": search_results[0].score if search_results else 0,
                        "top_result": search_results[0].payload.get("text", "")[:100] if search_results else ""
                    })
            
            # 3. 数据质量检查
            sample_points = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True
            )[0]
            
            if sample_points:
                text_lengths = [len(point.payload.get("text", "")) for point in sample_points]
                sources = [point.payload.get("source", "unknown") for point in sample_points]
                languages = [point.payload.get("language", "unknown") for point in sample_points]
                
                validation_results["data_quality"] = {
                    "sample_size": len(sample_points),
                    "avg_text_length": np.mean(text_lengths),
                    "text_length_std": np.std(text_lengths),
                    "unique_sources": len(set(sources)),
                    "language_distribution": dict(zip(*np.unique(languages, return_counts=True)))
                }
            
            logger.info("流水线验证完成")
            return validation_results
            
        except Exception as e:
            logger.error(f"流水线验证失败: {e}")
            return validation_results
    
    def run_pipeline(self, documents: List[Tuple[str, str]], 
                    sample_queries: List[str] = None) -> Dict[str, Any]:
        """运行完整的向量化流水线"""
        logger.info("🚀 启动完整向量化流水线")
        
        pipeline_start_time = time.time()
        
        try:
            # 1. 文档处理和分块
            logger.info("📄 步骤1: 文档处理和分块")
            chunks = self.process_documents(documents)
            
            # 2. 向量化
            logger.info("🧠 步骤2: 批量向量化")
            vector_data_list = self.vectorize_chunks(chunks)
            
            # 3. 向量入库
            logger.info("💾 步骤3: 向量数据库入库")
            insertion_success = self.insert_vectors(vector_data_list)
            
            # 4. 验证结果
            logger.info("✅ 步骤4: 结果验证")
            validation_results = self.validate_pipeline(sample_queries)
            
            # 计算总耗时
            total_time = time.time() - pipeline_start_time
            
            # 汇总结果
            pipeline_results = {
                "success": insertion_success,
                "total_time": total_time,
                "stats": self.stats.to_dict(),
                "validation": validation_results,
                "performance_metrics": {
                    "documents_per_second": self.stats.total_documents / total_time,
                    "chunks_per_second": self.stats.total_chunks / total_time,
                    "vectors_per_second": self.stats.total_vectors / total_time,
                    "vectorization_efficiency": self.stats.total_vectors / self.stats.vectorization_time,
                    "insertion_efficiency": self.stats.total_vectors / self.stats.insertion_time
                }
            }
            
            logger.info(f"🎉 流水线执行完成! 总耗时: {total_time:.2f}秒")
            return pipeline_results
            
        except Exception as e:
            logger.error(f"❌ 流水线执行失败: {e}")
            raise
    
    def cleanup(self) -> None:
        """清理资源"""
        logger.info("清理流水线资源")
        try:
            if self.client:
                self.client.delete_collection(self.collection_name)
                logger.info(f"集合 {self.collection_name} 已删除")
        except Exception as e:
            logger.warning(f"清理失败: {e}")


def main():
    """主演示函数"""
    print("=" * 80)
    print("🚀 Lesson 05: 完整向量化流水线演示")
    print("=" * 80)
    
    # 准备演示文档
    demo_documents = [
        (
            "人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的智能机器。"
            "AI的主要目标是开发能够学习、推理、感知、理解自然语言和解决问题的系统。现代AI技术包括机器学习、深度学习、自然语言处理、"
            "计算机视觉和机器人技术等多个领域。这些技术正在改变我们的生活方式，从智能手机的语音助手到自动驾驶汽车，"
            "AI正在各个行业中发挥越来越重要的作用。",
            "AI_introduction.txt"
        ),
        (
            "机器学习是人工智能的一个重要分支，它使计算机系统能够从数据中自动学习和改进，而无需明确编程。"
            "机器学习算法通过分析大量数据来识别模式，并使用这些模式对新数据进行预测或决策。"
            "主要的机器学习类型包括监督学习、无监督学习和强化学习。监督学习使用标记的训练数据来学习输入和输出之间的映射关系。"
            "无监督学习在没有标记数据的情况下发现数据中的隐藏模式。强化学习通过与环境交互并接收奖励或惩罚来学习最优行为。",
            "machine_learning.txt"
        ),
        (
            "深度学习是机器学习的一个子领域，它使用人工神经网络来模拟人脑的学习过程。"
            "深度神经网络由多个层组成，每一层都能学习数据的不同特征表示。这种分层的特征学习使深度学习在图像识别、"
            "语音识别、自然语言处理等任务中取得了突破性进展。卷积神经网络（CNN）特别适合处理图像数据，"
            "而循环神经网络（RNN）和Transformer架构则在序列数据处理方面表现出色。深度学习的成功很大程度上依赖于大量的训练数据、"
            "强大的计算能力和先进的优化算法。",
            "deep_learning.txt"
        ),
        (
            "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. "
            "NLP combines computational linguistics with machine learning and deep learning to enable computers to understand, interpret, and generate human language. "
            "Key NLP tasks include text classification, sentiment analysis, named entity recognition, machine translation, and question answering. "
            "Modern NLP systems use transformer-based models like BERT, GPT, and T5, which have achieved remarkable performance on various language understanding tasks. "
            "These models are pre-trained on large text corpora and can be fine-tuned for specific applications.",
            "nlp_overview.txt"
        ),
        (
            "Computer vision is a field of artificial intelligence that enables computers to interpret and understand visual information from the world. "
            "It involves developing algorithms and techniques to extract meaningful information from digital images and videos. "
            "Computer vision applications include object detection, image classification, facial recognition, medical image analysis, and autonomous vehicles. "
            "Deep learning has revolutionized computer vision, with convolutional neural networks (CNNs) becoming the standard approach for most vision tasks. "
            "Recent advances include attention mechanisms, vision transformers, and multi-modal models that combine vision with language understanding.",
            "computer_vision.txt"
        )
    ]
    
    # 准备测试查询
    sample_queries = [
        "什么是人工智能？",
        "机器学习的主要类型有哪些？",
        "深度学习和传统机器学习的区别",
        "What is natural language processing?",
        "Computer vision applications"
    ]
    
    # 初始化流水线
    pipeline = VectorPipeline(
        collection_name="lesson05_pipeline_demo",
        batch_size=16,
        max_workers=2
    )
    
    try:
        # 运行完整流水线
        results = pipeline.run_pipeline(demo_documents, sample_queries)
        
        # 展示结果
        print("\n" + "=" * 80)
        print("📊 流水线执行结果")
        print("=" * 80)
        
        print(f"✅ 执行状态: {'成功' if results['success'] else '失败'}")
        print(f"⏱️  总耗时: {results['total_time']:.2f}秒")
        print(f"📄 处理文档: {results['stats']['total_documents']}个")
        print(f"📝 生成块数: {results['stats']['total_chunks']}个")
        print(f"🧠 向量数量: {results['stats']['total_vectors']}个")
        print(f"📊 成功率: {results['stats']['success_rate']:.2%}")
        print(f"📏 平均块长度: {results['stats']['average_chunk_length']:.1f}字符")
        
        print(f"\n⚡ 性能指标:")
        metrics = results['performance_metrics']
        print(f"   - 文档处理速度: {metrics['documents_per_second']:.2f}文档/秒")
        print(f"   - 分块处理速度: {metrics['chunks_per_second']:.2f}块/秒")
        print(f"   - 向量化速度: {metrics['vectorization_efficiency']:.2f}向量/秒")
        print(f"   - 入库速度: {metrics['insertion_efficiency']:.2f}向量/秒")
        
        print(f"\n🔍 验证结果:")
        validation = results['validation']
        collection_info = validation['collection_info']
        print(f"   - 集合状态: {collection_info['status']}")
        print(f"   - 存储向量数: {collection_info['points_count']}")
        print(f"   - 向量维度: {collection_info['vector_size']}")
        
        if validation['sample_search']:
            print(f"\n🎯 搜索测试:")
            for i, search in enumerate(validation['sample_search'], 1):
                print(f"   查询{i}: '{search['query']}'")
                print(f"   - 结果数: {search['results_count']}")
                print(f"   - 最高分: {search['top_score']:.4f}")
                print(f"   - 最佳匹配: '{search['top_result']}...'")
                print()
        
        if validation['data_quality']:
            quality = validation['data_quality']
            print(f"📈 数据质量:")
            print(f"   - 样本大小: {quality['sample_size']}")
            print(f"   - 平均文本长度: {quality['avg_text_length']:.1f}±{quality['text_length_std']:.1f}")
            print(f"   - 数据源数量: {quality['unique_sources']}")
            print(f"   - 语言分布: {quality['language_distribution']}")
        
        print("\n" + "=" * 80)
        print("🎯 关键学习要点")
        print("=" * 80)
        print("1. 流水线设计: 模块化、可扩展的架构设计")
        print("2. 批量处理: 提高向量化和入库效率")
        print("3. 错误处理: 完善的异常处理和日志记录")
        print("4. 性能监控: 实时统计和性能指标")
        print("5. 质量验证: 自动化的结果验证机制")
        print("6. 资源管理: 合理的内存和计算资源使用")
        
        # 询问是否进行交互式搜索测试
        print("\n" + "=" * 80)
        interactive = input("是否进行交互式搜索测试？(y/N): ").lower().strip()
        if interactive == 'y':
            print("输入查询文本进行搜索测试 (输入'quit'退出):")
            while True:
                query = input("\n🔍 查询: ").strip()
                if query.lower() in ['quit', 'exit', '退出']:
                    break
                
                if query:
                    try:
                        query_vector = pipeline.model.encode([query])[0]
                        search_results = pipeline.client.search(
                            collection_name=pipeline.collection_name,
                            query_vector=query_vector.tolist(),
                            limit=3,
                            with_payload=True
                        )
                        
                        print(f"\n📋 搜索结果 (共{len(search_results)}个):")
                        for i, result in enumerate(search_results, 1):
                            print(f"  {i}. 相似度: {result.score:.4f}")
                            print(f"     来源: {result.payload.get('source', 'unknown')}")
                            print(f"     文本: '{result.payload.get('text', '')[:150]}...'")
                            print()
                    except Exception as e:
                        print(f"❌ 搜索失败: {e}")
        
    except Exception as e:
        print(f"❌ 流水线执行失败: {e}")
        logger.error(f"Pipeline execution failed: {e}")
    
    finally:
        # 询问是否清理数据
        cleanup = input("\n是否清理演示数据？(y/N): ").lower().strip()
        if cleanup == 'y':
            pipeline.cleanup()
        else:
            print(f"演示数据保留在集合 '{pipeline.collection_name}' 中")
        
        print("\n🎉 演示完成！")


if __name__ == "__main__":
    main()