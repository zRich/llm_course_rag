"""测试向量化功能"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.embedding.embedder import TextEmbedder
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_embedding():
    """测试基础向量化功能"""
    print("\n=== 测试基础向量化功能 ===")
    
    try:
        # 初始化向量化器
        embedder = TextEmbedder(model_name="BAAI/bge-m3")
        
        # 测试文本
        test_texts = [
            "人工智能是计算机科学的一个分支",
            "机器学习是人工智能的核心技术",
            "深度学习基于神经网络",
            "自然语言处理处理人类语言",
            "今天天气很好，适合出门散步"
        ]
        
        print(f"测试文本数量: {len(test_texts)}")
        
        # 单个文本向量化
        print("\n--- 单个文本向量化 ---")
        for i, text in enumerate(test_texts[:2]):
            vector = embedder.encode(text)
            print(f"文本 {i+1}: {text[:20]}...")
            print(f"向量维度: {vector.shape}")
            print(f"向量范围: [{vector.min():.4f}, {vector.max():.4f}]")
            print(f"向量模长: {np.linalg.norm(vector):.4f}")
            print()
        
        return True
        
    except Exception as e:
        print(f"基础向量化测试失败: {e}")
        return False

def test_batch_embedding():
    """测试批量向量化功能"""
    print("\n=== 测试批量向量化功能 ===")
    
    try:
        embedder = TextEmbedder(model_name="BAAI/bge-m3")
        
        # 批量测试文本
        batch_texts = [
            "人工智能技术发展迅速",
            "机器学习算法不断优化",
            "深度学习模型越来越复杂",
            "自然语言处理应用广泛",
            "计算机视觉识别准确率提高"
        ]
        
        print(f"批量处理文本数量: {len(batch_texts)}")
        
        # 批量向量化
        vectors = embedder.encode_batch(batch_texts)
        print(f"生成向量数量: {len(vectors)}")
        print(f"向量矩阵形状: {vectors.shape}")
        
        # 计算相似度矩阵
        print("\n--- 文本相似度矩阵 ---")
        similarity_matrix = embedder.compute_similarity_matrix(vectors)
        
        for i, text1 in enumerate(batch_texts):
            for j, text2 in enumerate(batch_texts):
                if i < j:  # 只显示上三角
                    similarity = similarity_matrix[i][j]
                    print(f"'{text1[:15]}...' vs '{text2[:15]}...': {similarity:.4f}")
        
        return True
        
    except Exception as e:
        print(f"批量向量化测试失败: {e}")
        return False

def test_different_models():
    """测试不同模型"""
    print("\n=== 测试不同模型 ===")
    
    # 可用的模型列表
    models_to_test = [
        "BAAI/bge-m3",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]
    
    test_text = "人工智能改变世界"
    
    for model_name in models_to_test:
        try:
            print(f"\n--- 测试模型: {model_name} ---")
            embedder = TextEmbedder(model_name=model_name)
            
            # 获取模型信息
            model_info = embedder.get_model_info()
            print(f"模型信息: {model_info}")
            
            # 向量化测试
            vector = embedder.encode(test_text)
            print(f"测试文本: {test_text}")
            print(f"向量维度: {vector.shape[0]}")
            print(f"向量模长: {np.linalg.norm(vector):.4f}")
            
        except Exception as e:
            print(f"模型 {model_name} 测试失败: {e}")
            continue
    
    return True

def test_vector_operations():
    """测试向量操作"""
    print("\n=== 测试向量操作 ===")
    
    try:
        embedder = TextEmbedder(model_name="BAAI/bge-m3")
        
        # 测试文本对
        text_pairs = [
            ("机器学习是人工智能的分支", "人工智能包含机器学习技术"),
            ("今天天气很好", "机器学习算法优化"),
            ("深度学习使用神经网络", "神经网络是深度学习基础")
        ]
        
        print("--- 文本相似度计算 ---")
        for text1, text2 in text_pairs:
            similarity = embedder.compute_similarity(text1, text2)
            print(f"文本1: {text1}")
            print(f"文本2: {text2}")
            print(f"相似度: {similarity:.4f}")
            print()
        
        # 测试向量保存和加载
        print("--- 向量保存和加载测试 ---")
        test_texts = ["测试文本1", "测试文本2", "测试文本3"]
        vectors = embedder.encode_batch(test_texts)
        
        # 保存向量
        save_path = "test_vectors.npy"
        embedder.save_embeddings(vectors, test_texts, save_path)
        print(f"向量已保存到: {save_path}")
        
        # 加载向量
        loaded_vectors, loaded_texts, loaded_metadata = embedder.load_embeddings(save_path)
        print(f"加载向量数量: {len(loaded_vectors)}")
        print(f"加载文本数量: {len(loaded_texts)}")
        
        # 验证一致性
        if np.allclose(vectors, loaded_vectors) and test_texts == loaded_texts:
            print("向量保存和加载验证成功！")
        else:
            print("向量保存和加载验证失败！")
        
        # 清理测试文件
        if os.path.exists(save_path):
            os.remove(save_path)
        if os.path.exists(save_path.replace('.npy', '_texts.json')):
            os.remove(save_path.replace('.npy', '_texts.json'))
        
        return True
        
    except Exception as e:
        print(f"向量操作测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始向量化功能测试...")
    
    test_results = []
    
    # 运行所有测试
    test_functions = [
        ("基础向量化", test_basic_embedding),
        ("批量向量化", test_batch_embedding),
        ("不同模型", test_different_models),
        ("向量操作", test_vector_operations)
    ]
    
    for test_name, test_func in test_functions:
        print(f"\n{'='*50}")
        print(f"运行测试: {test_name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"测试 {test_name} 出现异常: {e}")
            test_results.append((test_name, False))
    
    # 输出测试结果汇总
    print(f"\n{'='*50}")
    print("测试结果汇总")
    print(f"{'='*50}")
    
    passed = 0
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(test_results)} 个测试通过")
    
    if passed == len(test_results):
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败，请检查错误信息")

if __name__ == "__main__":
    main()