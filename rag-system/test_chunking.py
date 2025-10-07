#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分块器测试脚本

测试各种文档分块策略，包括：
- 基于句子的分块器
- 基于语义的分块器
- 基于结构的分块器
- 分块管理器
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.chunking.sentence_chunker import SentenceChunker
from src.chunking.semantic_chunker import SemanticChunker
from src.chunking.structure_chunker import StructureChunker
from src.chunking.chunk_manager import chunk_manager
from src.chunking.chunker import ChunkingConfig
from src.document.document_manager import document_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 测试文本
SAMPLE_TEXT = """
# 人工智能技术发展报告

## 1. 引言

人工智能（Artificial Intelligence，AI）是计算机科学的一个重要分支。它致力于研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统。

人工智能的发展历程可以追溯到20世纪50年代。从那时起，AI技术经历了多次起伏，但总体趋势是不断向前发展的。

## 2. 主要技术领域

### 2.1 机器学习

机器学习是人工智能的核心技术之一。它使计算机能够在没有明确编程的情况下学习和改进。主要包括以下几种类型：

- 监督学习：使用标记数据进行训练
- 无监督学习：从未标记数据中发现模式
- 强化学习：通过与环境交互来学习最优策略

### 2.2 深度学习

深度学习是机器学习的一个子集，它模仿人脑神经网络的结构和功能。深度学习在图像识别、自然语言处理、语音识别等领域取得了突破性进展。

### 2.3 自然语言处理

自然语言处理（NLP）是人工智能的重要应用领域。它研究如何让计算机理解、处理和生成人类语言。

## 3. 应用场景

人工智能技术在各个行业都有广泛应用：

1. 医疗健康：疾病诊断、药物研发、个性化治疗
2. 金融服务：风险评估、算法交易、反欺诈检测
3. 交通运输：自动驾驶、智能交通管理、路径优化
4. 教育培训：个性化学习、智能辅导、自动评分
5. 制造业：质量控制、预测性维护、供应链优化

## 4. 发展趋势

未来人工智能的发展将呈现以下趋势：

- 技术融合：AI与其他技术的深度融合
- 边缘计算：AI计算向边缘设备迁移
- 可解释性：提高AI决策的透明度和可解释性
- 伦理规范：建立AI发展的伦理框架和规范

## 5. 结论

人工智能技术正在快速发展，并对社会各个方面产生深远影响。我们需要在推动技术进步的同时，也要关注其带来的挑战和风险，确保AI技术能够造福人类社会。
"""

def test_sentence_chunker():
    """测试基于句子的分块器"""
    print("\n=== 测试基于句子的分块器 ===")
    
    # 创建分块器
    config = ChunkingConfig(
        chunk_size=500,
        min_chunk_size=100,
        chunk_overlap=50
    )
    
    chunker = SentenceChunker(config)
    
    try:
        # 执行分块
        chunks = chunker.chunk_text(SAMPLE_TEXT)
        
        print(f"分块完成! 共生成 {len(chunks)} 个块")
        print(f"分块器类型: {chunker.get_chunker_type()}")
        
        # 显示分块结果
        for i, chunk in enumerate(chunks[:3]):  # 只显示前3个块
            print(f"\n块 {i+1}:")
            print(f"  ID: {chunk.metadata.chunk_id}")
            print(f"  长度: {len(chunk.content)} 字符")
            print(f"  内容预览: {chunk.content[:100]}...")
            print(f"  元数据: {chunk.metadata.__dict__}")
        
        if len(chunks) > 3:
            print(f"\n... 还有 {len(chunks) - 3} 个块")
        
        # 测试统计信息
        total_chars = sum(len(chunk.content) for chunk in chunks)
        print(f"\n统计信息:")
        print(f"  总字符数: {total_chars}")
        print(f"  平均块大小: {total_chars // len(chunks)} 字符")
        
    except Exception as e:
        print(f"句子分块测试失败: {e}")
        logger.error(f"句子分块错误: {e}", exc_info=True)

def test_semantic_chunker():
    """测试基于语义的分块器"""
    print("\n=== 测试基于语义的分块器 ===")
    
    # 创建分块器
    config = ChunkingConfig(
        chunk_size=800,
        min_chunk_size=200,
        chunk_overlap=100
    )
    
    chunker = SemanticChunker(config)
    
    try:
        # 执行分块
        chunks = chunker.chunk_text(SAMPLE_TEXT)
        
        print(f"语义分块完成! 共生成 {len(chunks)} 个块")
        print(f"分块器类型: {chunker.get_chunker_type()}")
        
        # 显示分块结果
        for i, chunk in enumerate(chunks[:3]):  # 只显示前3个块
            print(f"\n语义块 {i+1}:")
            print(f"  ID: {chunk.chunk_id}")
            print(f"  长度: {len(chunk.content)} 字符")
            print(f"  内容预览: {chunk.content[:150]}...")
            
            # 显示语义相关的元数据
            if hasattr(chunk.metadata, 'semantic_score'):
                print(f"  语义得分: {chunk.metadata.semantic_score}")
            if hasattr(chunk.metadata, 'sentence_count'):
                print(f"  句子数量: {chunk.metadata.sentence_count}")
        
        if len(chunks) > 3:
            print(f"\n... 还有 {len(chunks) - 3} 个语义块")
        
    except Exception as e:
        print(f"语义分块测试失败: {e}")
        logger.error(f"语义分块错误: {e}", exc_info=True)
        
        # 如果语义分块失败，测试回退机制
        print("测试回退到句子分块...")
        try:
            fallback_chunks = chunker.chunk_text("简单的测试文本。这是第二句话。")
            print(f"回退机制工作正常，生成 {len(fallback_chunks)} 个块")
        except Exception as fallback_e:
            print(f"回退机制也失败: {fallback_e}")

def test_structure_chunker():
    """测试基于结构的分块器"""
    print("\n=== 测试基于结构的分块器 ===")
    
    # 创建分块器
    config = ChunkingConfig(
        chunk_size=1000,
        min_chunk_size=200,
        chunk_overlap=50
    )
    
    chunker = StructureChunker(config)
    
    try:
        # 执行分块
        chunks = chunker.chunk_text(SAMPLE_TEXT)
        
        print(f"结构分块完成! 共生成 {len(chunks)} 个块")
        print(f"分块器类型: {chunker.get_chunker_type()}")
        
        # 显示分块结果
        for i, chunk in enumerate(chunks[:3]):  # 只显示前3个块
            print(f"\n结构块 {i+1}:")
            print(f"  ID: {chunk.chunk_id}")
            print(f"  长度: {len(chunk.content)} 字符")
            print(f"  内容预览: {chunk.content[:150]}...")
            
            # 显示结构相关的元数据
            if hasattr(chunk.metadata, 'structure_type'):
                print(f"  结构类型: {chunk.metadata.structure_type}")
            if hasattr(chunk.metadata, 'heading_level'):
                print(f"  标题级别: {chunk.metadata.heading_level}")
        
        if len(chunks) > 3:
            print(f"\n... 还有 {len(chunks) - 3} 个结构块")
        
        # 测试文档结构分析
        print("\n=== 文档结构分析 ===")
        structure_info = chunker.analyze_document_structure(SAMPLE_TEXT)
        print(f"结构统计: {structure_info}")
        
    except Exception as e:
        print(f"结构分块测试失败: {e}")
        logger.error(f"结构分块错误: {e}", exc_info=True)

def test_chunk_manager():
    """测试分块管理器"""
    print("\n=== 测试分块管理器 ===")
    
    try:
        # 测试可用的分块器
        available_chunkers = chunk_manager.list_chunkers()
        print(f"可用的分块器: {available_chunkers}")
        
        # 测试不同分块器的比较
        print("\n=== 分块器比较 ===")
        
        chunker_types = ['sentence', 'semantic', 'structure']
        comparison_results = {}
        
        for chunker_type in chunker_types:
            try:
                chunks = chunk_manager.chunk_text(SAMPLE_TEXT, chunker_type=chunker_type)
                comparison_results[chunker_type] = {
                    'chunk_count': len(chunks),
                    'avg_length': sum(len(c.content) for c in chunks) // len(chunks) if chunks else 0,
                    'total_length': sum(len(c.content) for c in chunks)
                }
                print(f"{chunker_type}分块器: {len(chunks)} 个块, 平均长度 {comparison_results[chunker_type]['avg_length']} 字符")
            except Exception as e:
                print(f"{chunker_type}分块器失败: {e}")
                comparison_results[chunker_type] = None
        
        # 测试分块策略推荐
        print("\n=== 分块策略推荐 ===")
        recommendation = chunk_manager.recommend_chunker(SAMPLE_TEXT)
        print(f"推荐的分块器: {recommendation}")
        
        # 测试批量分块
        print("\n=== 批量分块测试 ===")
        test_texts = [
            "这是第一个测试文档。包含简单的句子。",
            "# 标题\n\n这是第二个测试文档。\n\n## 子标题\n\n包含结构化内容。",
            "长文本测试。" * 50  # 创建一个较长的文本
        ]
        
        batch_results = chunk_manager.chunk_texts(test_texts, chunker_type='sentence')
        print(f"批量分块完成: {len(batch_results)} 个文档")
        
        for i, chunks in enumerate(batch_results):
            print(f"  文档 {i+1}: {len(chunks)} 个块")
        
    except Exception as e:
        print(f"分块管理器测试失败: {e}")
        logger.error(f"分块管理器错误: {e}", exc_info=True)

def test_file_chunking():
    """测试文件分块功能"""
    print("\n=== 测试文件分块功能 ===")
    
    test_docs_dir = project_root / "test_documents"
    
    if not test_docs_dir.exists():
        print(f"测试文档目录不存在: {test_docs_dir}")
        return
    
    # 查找测试文件
    test_files = []
    for file_path in test_docs_dir.iterdir():
        if file_path.is_file() and document_manager.can_parse(str(file_path)):
            test_files.append(str(file_path))
    
    if not test_files:
        print("没有找到支持的测试文档")
        return
    
    print(f"找到 {len(test_files)} 个测试文件")
    
    for file_path in test_files[:2]:  # 只测试前2个文件
        try:
            filename = Path(file_path).name
            print(f"\n正在分块文件: {filename}")
            
            # 使用不同的分块器测试
            for chunker_type in ['sentence', 'structure']:
                try:
                    chunks = chunk_manager.chunk_file(file_path, chunker_type=chunker_type)
                    print(f"  {chunker_type}分块: {len(chunks)} 个块")
                    
                    if chunks:
                        avg_length = sum(len(c.content) for c in chunks) // len(chunks)
                        print(f"    平均块长度: {avg_length} 字符")
                
                except Exception as e:
                    print(f"  {chunker_type}分块失败: {e}")
        
        except Exception as e:
            print(f"文件分块失败 {filename}: {e}")

def test_chunk_export():
    """测试分块结果导出"""
    print("\n=== 测试分块结果导出 ===")
    
    try:
        # 生成一些测试分块
        chunks = chunk_manager.chunk_text(SAMPLE_TEXT, chunker_type='sentence')
        
        if not chunks:
            print("没有分块结果可导出")
            return
        
        # 测试不同格式的导出
        export_formats = ['json', 'csv', 'txt']
        
        for format_type in export_formats:
            try:
                output_file = project_root / f"test_chunks.{format_type}"
                chunk_manager.export_chunks(chunks, str(output_file), format_type)
                
                if output_file.exists():
                    file_size = output_file.stat().st_size
                    print(f"导出 {format_type.upper()} 格式成功: {output_file} ({file_size} 字节)")
                else:
                    print(f"导出 {format_type.upper()} 格式失败: 文件未创建")
            
            except Exception as e:
                print(f"导出 {format_type.upper()} 格式失败: {e}")
    
    except Exception as e:
        print(f"分块导出测试失败: {e}")
        logger.error(f"分块导出错误: {e}", exc_info=True)

def test_chunking_config():
    """测试分块配置"""
    print("\n=== 测试分块配置 ===")
    
    # 测试不同的配置
    configs = [
        ChunkingConfig(chunk_size=300, min_chunk_size=50, chunk_overlap=30),
        ChunkingConfig(chunk_size=800, min_chunk_size=200, chunk_overlap=100),
        ChunkingConfig(chunk_size=1500, min_chunk_size=300, chunk_overlap=150)
    ]
    
    for i, config in enumerate(configs):
        print(f"\n配置 {i+1}: 块大小={config.chunk_size}, 最小块={config.min_chunk_size}, 重叠={config.chunk_overlap}")
        
        try:
            chunker = SentenceChunker(config)
            chunks = chunker.chunk_text(SAMPLE_TEXT)
            
            avg_length = sum(len(c.content) for c in chunks) // len(chunks) if chunks else 0
            print(f"  结果: {len(chunks)} 个块, 平均长度 {avg_length} 字符")
            
            # 检查重叠
            if len(chunks) > 1:
                overlap_found = False
                for j in range(len(chunks) - 1):
                    current_end = chunks[j].content[-50:]
                    next_start = chunks[j+1].content[:50]
                    if any(word in next_start for word in current_end.split()[-5:]):
                        overlap_found = True
                        break
                print(f"  重叠检测: {'发现重叠' if overlap_found else '未发现重叠'}")
        
        except Exception as e:
            print(f"  配置测试失败: {e}")

def create_test_environment():
    """创建测试环境"""
    print("\n=== 创建测试环境 ===")
    
    # 创建测试文档目录
    test_docs_dir = project_root / "test_documents"
    test_docs_dir.mkdir(exist_ok=True)
    print(f"测试文档目录: {test_docs_dir}")
    
    # 创建长文本测试文件
    long_text_file = test_docs_dir / "long_text.txt"
    if not long_text_file.exists():
        long_content = SAMPLE_TEXT * 3  # 创建更长的文本
        long_text_file.write_text(long_content, encoding='utf-8')
        print(f"创建长文本测试文件: {long_text_file}")
    
    # 创建结构化文档测试文件
    structured_file = test_docs_dir / "structured.md"
    if not structured_file.exists():
        structured_file.write_text(SAMPLE_TEXT, encoding='utf-8')
        print(f"创建结构化测试文件: {structured_file}")
    
    return test_docs_dir

def main():
    """主测试函数"""
    print("分块器测试开始")
    print("=" * 50)
    
    # 创建测试环境
    create_test_environment()
    
    # 运行测试
    test_sentence_chunker()
    test_semantic_chunker()
    test_structure_chunker()
    test_chunk_manager()
    test_file_chunking()
    test_chunk_export()
    test_chunking_config()
    
    print("\n=== 测试完成 ===")
    print("如果看到分块成功的消息，说明分块器工作正常")
    print("如果有错误，请检查依赖安装（特别是nltk和scikit-learn）")
    print("\n注意：语义分块器需要较多的计算资源，可能在某些环境下失败")

if __name__ == "__main__":
    main()