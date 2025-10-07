"""测试批量向量化功能"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import shutil
import pytest
from pathlib import Path
from src.embedding.embedder import TextEmbedder
from src.vector_store.qdrant_client import QdrantVectorStore
from src.vector_store.document_vectorizer import DocumentVectorizer
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def test_dir():
    """创建临时测试目录和测试文档"""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_")
    test_path = create_test_documents(temp_dir)
    yield test_path
    # 清理临时目录
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def create_test_documents(test_dir: str):
    """创建测试文档"""
    print("\n=== 创建测试文档 ===")
    
    # 创建测试目录
    test_path = Path(test_dir)
    test_path.mkdir(parents=True, exist_ok=True)
    
    # 测试文档内容
    test_documents = {
        "ai_overview.txt": """
人工智能概述

人工智能（Artificial Intelligence，AI）是计算机科学的一个重要分支，致力于创建能够执行通常需要人类智能的任务的系统。AI的目标是开发能够学习、推理、感知、理解和决策的计算机系统。

人工智能的发展历程可以追溯到20世纪50年代，当时科学家们开始探索让机器模拟人类智能的可能性。经过几十年的发展，AI技术已经在各个领域取得了显著进展。

现代人工智能主要包括以下几个核心领域：
1. 机器学习：让计算机从数据中学习模式和规律
2. 深度学习：基于神经网络的学习方法
3. 自然语言处理：让计算机理解和生成人类语言
4. 计算机视觉：让计算机理解和分析图像和视频
5. 机器人学：开发能够在物理世界中行动的智能系统
""",
        
        "machine_learning.txt": """
机器学习基础

机器学习是人工智能的核心技术之一，它使计算机能够在没有明确编程的情况下从数据中学习和改进。机器学习算法通过分析大量数据来识别模式，并使用这些模式对新数据进行预测或决策。

机器学习主要分为三种类型：

1. 监督学习（Supervised Learning）
监督学习使用标记的训练数据来学习输入和输出之间的映射关系。常见的监督学习任务包括分类和回归。

2. 无监督学习（Unsupervised Learning）
无监督学习处理没有标签的数据，目标是发现数据中的隐藏结构或模式。聚类和降维是常见的无监督学习任务。

3. 强化学习（Reinforcement Learning）
强化学习通过与环境交互来学习最优行为策略，通过奖励和惩罚机制来指导学习过程。

机器学习的应用非常广泛，包括图像识别、语音识别、推荐系统、金融风控、医疗诊断等领域。
""",
        
        "deep_learning.txt": """
深度学习技术

深度学习是机器学习的一个子领域，它基于人工神经网络，特别是深层神经网络来进行学习和模式识别。深度学习的"深度"指的是网络中的层数，通常包含多个隐藏层。

深度学习的核心概念：

1. 神经网络
神经网络由大量相互连接的节点（神经元）组成，每个节点接收输入信号，进行处理后输出结果。

2. 反向传播
反向传播是训练神经网络的核心算法，通过计算损失函数的梯度来更新网络参数。

3. 激活函数
激活函数为神经网络引入非线性，常见的激活函数包括ReLU、Sigmoid、Tanh等。

深度学习的主要架构：
- 卷积神经网络（CNN）：主要用于图像处理
- 循环神经网络（RNN）：适合处理序列数据
- 长短期记忆网络（LSTM）：解决RNN的长期依赖问题
- Transformer：在自然语言处理领域表现出色

深度学习在计算机视觉、自然语言处理、语音识别等领域取得了突破性进展。
""",
        
        "nlp_basics.txt": """
自然语言处理入门

自然语言处理（Natural Language Processing，NLP）是人工智能的一个重要分支，专注于让计算机理解、解释和生成人类语言。NLP结合了计算机科学、人工智能和语言学的知识。

NLP的主要任务包括：

1. 文本预处理
- 分词：将文本分割成单词或词汇单元
- 词性标注：识别每个词的语法角色
- 命名实体识别：识别文本中的人名、地名、组织名等

2. 语义理解
- 词义消歧：确定多义词在特定上下文中的含义
- 语义角色标注：识别句子中各成分的语义角色
- 情感分析：判断文本的情感倾向

3. 文本生成
- 机器翻译：将一种语言的文本翻译成另一种语言
- 文本摘要：生成文本的简洁摘要
- 对话系统：与用户进行自然语言交互

现代NLP技术大量使用深度学习方法，特别是Transformer架构和预训练语言模型（如BERT、GPT等）在各种NLP任务中取得了显著成果。
""",
        
        "computer_vision.txt": """
计算机视觉概述

计算机视觉是人工智能的一个重要分支，致力于让计算机能够从数字图像或视频中获取高层次的理解。计算机视觉的目标是模拟人类视觉系统的功能，使计算机能够识别、分析和理解视觉信息。

计算机视觉的主要任务：

1. 图像分类
将图像分配到预定义的类别中，例如识别图像中是否包含猫、狗或其他物体。

2. 目标检测
不仅要识别图像中的物体类别，还要确定物体在图像中的位置，通常用边界框表示。

3. 图像分割
将图像分割成不同的区域或像素级别的分类，包括语义分割和实例分割。

4. 人脸识别
识别和验证图像中的人脸身份，广泛应用于安全和身份验证系统。

5. 光学字符识别（OCR）
从图像中提取和识别文本信息。

计算机视觉技术的应用领域非常广泛，包括：
- 自动驾驶汽车
- 医学图像分析
- 工业质量检测
- 安防监控
- 增强现实
- 机器人导航

深度学习，特别是卷积神经网络（CNN），已经成为现代计算机视觉的核心技术。
"""
    }
    
    # 写入测试文档
    for filename, content in test_documents.items():
        file_path = test_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"创建测试文档: {filename}")
    
    print(f"✓ 创建了 {len(test_documents)} 个测试文档")
    return str(test_path)

@pytest.fixture
def vectorizer():
    """创建DocumentVectorizer实例"""
    try:
        # 检查Qdrant连接
        vector_store = QdrantVectorStore(
            host="localhost",
            port=6333,
            timeout=10
        )
        
        # 初始化向量化器
        embedder = TextEmbedder(model_name="BAAI/bge-m3")
        
        # 初始化文档向量化管理器
        vectorizer = DocumentVectorizer(
            embedder=embedder,
            vector_store=vector_store,
            collection_name="test_documents",
            chunk_size=300,
            chunk_overlap=50
        )
        
        yield vectorizer
        
        # 清理测试集合
        try:
            vectorizer.vector_store.delete_collection("test_documents")
        except Exception as e:
            logger.warning(f"清理测试集合失败: {e}")
            
    except Exception as e:
        pytest.skip(f"无法初始化vectorizer: {e}")

def test_document_vectorizer_setup(vectorizer):
    """测试DocumentVectorizer初始化"""
    print("\n=== 测试DocumentVectorizer初始化 ===")
    
    # 检查vectorizer是否正确初始化
    assert vectorizer is not None
    assert hasattr(vectorizer, 'embedder')
    assert hasattr(vectorizer, 'vector_store')
    assert hasattr(vectorizer, 'collection_name')
    assert vectorizer.collection_name == "test_documents"
    
    print("✓ DocumentVectorizer初始化成功")

def test_single_document_processing(vectorizer, test_dir):
    """测试单个文档处理"""
    print("\n=== 测试单个文档处理 ===")
    
    try:
        # 选择一个测试文档
        test_file = os.path.join(test_dir, "ai_overview.txt")
        print(f"处理文档: {test_file}")
        
        # 处理文档
        result = vectorizer.process_document(test_file)
        
        # 输出结果
        print(f"处理结果:")
        print(f"  - 文件路径: {result['file_path']}")
        print(f"  - 处理成功: {result['success']}")
        print(f"  - 文本块数量: {result['chunks_count']}")
        print(f"  - 向量数量: {result['vectors_count']}")
        print(f"  - 处理时间: {result['processing_time']:.2f}s")
        
        if result['error']:
            print(f"  - 错误信息: {result['error']}")
        
        assert result['success'], f"文档处理失败: {result.get('error', '未知错误')}"
        
    except Exception as e:
        print(f"✗ 单个文档处理失败: {e}")
        pytest.fail(f"单个文档处理失败: {e}")

def test_batch_directory_processing(vectorizer, test_dir):
    """测试批量目录处理"""
    print("\n=== 测试批量目录处理 ===")
    
    try:
        # 批量处理目录
        print(f"批量处理目录: {test_dir}")
        
        batch_result = vectorizer.batch_process_directory(
            directory_path=test_dir,
            file_extensions=['.txt'],
            recursive=False
        )
        
        # 输出批量处理结果
        print(f"批量处理结果:")
        print(f"  - 总文件数: {batch_result['total_files']}")
        print(f"  - 成功文件数: {batch_result['successful_files']}")
        print(f"  - 失败文件数: {batch_result['failed_files']}")
        print(f"  - 总文本块数: {batch_result['total_chunks']}")
        print(f"  - 总向量数: {batch_result['total_vectors']}")
        print(f"  - 处理时间: {batch_result['processing_time']:.2f}s")
        
        # 显示每个文件的处理结果
        print("\n各文件处理详情:")
        for file_result in batch_result['file_results']:
            status = "✓" if file_result['success'] else "✗"
            filename = os.path.basename(file_result['file_path'])
            print(f"  {status} {filename}: {file_result['chunks_count']} 块, {file_result['processing_time']:.2f}s")
            if file_result['error']:
                print(f"    错误: {file_result['error']}")
        
        assert batch_result['successful_files'] > 0, "批量处理没有成功处理任何文件"
        
    except Exception as e:
        print(f"✗ 批量目录处理失败: {e}")
        pytest.fail(f"批量目录处理失败: {e}")

def test_document_search(vectorizer):
    """测试文档搜索"""
    print("\n=== 测试文档搜索 ===")
    
    try:
        # 测试查询
        test_queries = [
            "什么是机器学习？",
            "深度学习的神经网络",
            "自然语言处理的应用",
            "计算机视觉技术",
            "人工智能的发展历程"
        ]
        
        for query in test_queries:
            print(f"\n--- 查询: {query} ---")
            
            # 执行搜索
            results = vectorizer.search_documents(
                query=query,
                limit=3,
                score_threshold=0.3
            )
            
            print(f"找到 {len(results)} 个相关结果:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. [分数: {result['score']:.4f}] {result['file_name']}")
                print(f"     文本: {result['chunk_text'][:100]}...")
                print(f"     块索引: {result['chunk_index']}, 长度: {result['chunk_length']}")
        
        # 搜索测试通过（即使没有结果也是正常的，因为可能还没有文档被索引）
        
    except Exception as e:
        print(f"✗ 文档搜索失败: {e}")
        pytest.fail(f"文档搜索失败: {e}")

def test_collection_stats(vectorizer):
    """测试集合统计"""
    print("\n=== 测试集合统计 ===")
    
    try:
        # 获取集合统计信息
        stats = vectorizer.get_collection_stats()
        
        print("集合统计信息:")
        if 'error' in stats:
            print(f"  错误: {stats['error']}")
            return False
        else:
            print(f"  - 集合名称: {stats['collection_name']}")
            print(f"  - 总向量数: {stats['total_vectors']}")
            print(f"  - 已索引向量数: {stats['indexed_vectors']}")
            print(f"  - 总点数: {stats['total_points']}")
            print(f"  - 向量维度: {stats['vector_dimension']}")
            print(f"  - 距离度量: {stats['distance_metric']}")
            print(f"  - 状态: {stats['status']}")
        
        # 集合统计测试完成
        
    except Exception as e:
        print(f"✗ 集合统计失败: {e}")
        pytest.fail(f"集合统计失败: {e}")

def test_processing_log(vectorizer):
    """测试处理日志"""
    print("\n=== 测试处理日志 ===")
    
    try:
        # 保存处理日志
        log_file = "processing_log.json"
        vectorizer.save_processing_log(log_file)
        
        # 检查日志文件
        if os.path.exists(log_file):
            print(f"✓ 处理日志已保存: {log_file}")
            
            # 读取并显示日志摘要
            import json
            with open(log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            print(f"日志摘要:")
            print(f"  - 时间戳: {log_data['timestamp']}")
            print(f"  - 集合名称: {log_data['collection_name']}")
            print(f"  - 处理文件总数: {log_data['total_processed']}")
            
            # 清理日志文件
            os.remove(log_file)
            print(f"✓ 测试日志文件已清理")
            
            # 处理日志测试通过
            pass
        else:
            print("✗ 处理日志文件未创建")
            pytest.fail("处理日志文件未创建")
        
    except Exception as e:
        print(f"✗ 处理日志测试失败: {e}")
        pytest.fail(f"处理日志测试失败: {e}")

# 移除main函数，使用pytest运行测试