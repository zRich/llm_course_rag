#!/usr/bin/env python3
"""
课程要求分析脚本
根据课程讲义内容，分析每个lesson应该实现的具体功能和代码变更
"""

import json
from typing import Dict, List, Any

def analyze_lesson_requirements() -> Dict[str, Any]:
    """
    根据课程讲义分析每个lesson的具体开发要求
    """
    
    lesson_requirements = {
        "lesson01": {
            "module": "A",
            "title": "课程导入与环境准备",
            "expected_changes": [
                "创建基础项目结构",
                "配置Python环境和依赖管理(uv)",
                "创建最小FastAPI应用",
                "配置开发环境"
            ],
            "key_files": [
                "main.py",
                "pyproject.toml",
                "README.md"
            ],
            "dependencies": ["fastapi", "uvicorn"],
            "code_estimate": "基础项目结构，约100-200行代码"
        },
        
        "lesson02": {
            "module": "A",
            "title": "容器化与依赖服务启动",
            "expected_changes": [
                "添加Docker Compose配置",
                "配置PostgreSQL、Redis、Qdrant、MinIO服务",
                "添加容器化相关依赖"
            ],
            "key_files": [
                "docker-compose.yml",
                "pyproject.toml"
            ],
            "dependencies": ["beautifulsoup4", "lxml", "requests"],
            "code_estimate": "主要是配置文件，少量代码变更"
        },
        
        "lesson03": {
            "module": "A",
            "title": "数据模型与迁移",
            "expected_changes": [
                "设计RAG系统核心数据模型",
                "实现SQLModel数据模型",
                "配置数据库连接",
                "实现数据迁移脚本"
            ],
            "key_files": [
                "app/data/models.py",
                "app/core/db.py",
                "pyproject.toml"
            ],
            "dependencies": ["chromadb", "sentence-transformers"],
            "code_estimate": "数据模型和数据库配置，约200-400行代码"
        },
        
        "lesson04": {
            "module": "A",
            "title": "PDF解析与Chunk拆分",
            "expected_changes": [
                "实现PDF文档解析功能",
                "实现文本分块(Chunk)处理",
                "添加文档结构识别",
                "实现元数据提取"
            ],
            "key_files": [
                "src/chunking/",
                "src/data_connectors/"
            ],
            "dependencies": ["PyMuPDF", "fitz"],
            "code_estimate": "文档解析和分块功能，约300-600行代码"
        },
        
        "lesson05": {
            "module": "A",
            "title": "Embedding与向量入库",
            "expected_changes": [
                "实现文本向量化功能",
                "集成sentence-transformers和bge-m3模型",
                "实现Qdrant向量数据库操作",
                "实现批量向量化处理"
            ],
            "key_files": [
                "src/embedding/",
                "src/vector_store/"
            ],
            "dependencies": ["sentence-transformers", "qdrant-client"],
            "code_estimate": "向量化和向量存储功能，约400-800行代码"
        },
        
        "lesson06": {
            "module": "A",
            "title": "最小检索与生成（MVP RAG）",
            "expected_changes": [
                "实现基础向量检索功能",
                "集成LLM调用接口",
                "实现基础RAG流程",
                "创建API接口"
            ],
            "key_files": [
                "src/retrieval/",
                "src/generation/",
                "api/"
            ],
            "dependencies": ["openai"],
            "code_estimate": "基础RAG系统，约300-500行代码"
        },
        
        "lesson07": {
            "module": "B",
            "title": "关键词检索",
            "expected_changes": [
                "实现PostgreSQL全文检索功能",
                "添加中文分词支持",
                "实现关键词检索和排序",
                "优化查询性能"
            ],
            "key_files": [
                "src/retrieval/keyword_search.py",
                "src/text_processing/"
            ],
            "dependencies": ["jieba"],
            "code_estimate": "关键词检索功能，约200-400行代码"
        },
        
        "lesson08": {
            "module": "B",
            "title": "混合检索融合策略",
            "expected_changes": [
                "实现向量检索与关键词检索融合",
                "实现权重调节算法",
                "实现去重和排序策略",
                "添加A/B测试框架"
            ],
            "key_files": [
                "src/retrieval/hybrid_search.py",
                "src/fusion/"
            ],
            "dependencies": [],
            "code_estimate": "混合检索功能，约300-500行代码"
        },
        
        "lesson09": {
            "module": "B",
            "title": "元数据过滤",
            "expected_changes": [
                "实现基于元数据的过滤功能",
                "支持动态复合查询条件",
                "优化元数据索引",
                "添加查询性能监控"
            ],
            "key_files": [
                "src/retrieval/metadata_filter.py",
                "src/indexing/"
            ],
            "dependencies": [],
            "code_estimate": "元数据过滤功能，约200-300行代码"
        },
        
        "lesson10": {
            "module": "B",
            "title": "重排序(Rerank)接入",
            "expected_changes": [
                "集成bge-reranker-v2-m3模型",
                "实现重排序功能",
                "添加缓存策略",
                "提升检索精度"
            ],
            "key_files": [
                "src/reranking/",
                "src/caching/"
            ],
            "dependencies": ["sentence-transformers"],
            "code_estimate": "重排序功能，约200-400行代码"
        },
        
        "lesson11": {
            "module": "B",
            "title": "Chunk尺寸与重叠实验",
            "expected_changes": [
                "实现动态分块算法",
                "优化重叠策略",
                "添加分块质量评估",
                "实现参数调优实验"
            ],
            "key_files": [
                "src/chunking/dynamic_chunking.py",
                "src/evaluation/"
            ],
            "dependencies": [],
            "code_estimate": "分块优化和实验功能，约300-600行代码"
        },
        
        "lesson12": {
            "module": "B",
            "title": "多文档源处理",
            "expected_changes": [
                "支持PDF、Word、TXT、HTML文档解析",
                "实现批量处理优化",
                "添加错误处理机制",
                "统一文档处理接口"
            ],
            "key_files": [
                "src/data_connectors/multi_format.py",
                "src/batch_processing/"
            ],
            "dependencies": ["python-docx", "beautifulsoup4"],
            "code_estimate": "多格式文档处理，约400-700行代码"
        },
        
        "lesson13": {
            "module": "B",
            "title": "引用与可溯源输出",
            "expected_changes": [
                "实现引用链追踪",
                "添加置信度计算",
                "实现可溯源信息展示",
                "与生成结果结合"
            ],
            "key_files": [
                "src/citation/",
                "src/traceability/"
            ],
            "dependencies": [],
            "code_estimate": "引用和溯源功能，约200-400行代码"
        },
        
        "lesson14": {
            "module": "B",
            "title": "缓存策略",
            "expected_changes": [
                "实现Redis多级缓存",
                "添加缓存失效与更新策略",
                "优化缓存命中率",
                "设计分布式缓存"
            ],
            "key_files": [
                "src/caching/redis_cache.py",
                "src/caching/cache_manager.py"
            ],
            "dependencies": ["redis"],
            "code_estimate": "缓存系统，约300-500行代码"
        },
        
        "lesson15": {
            "module": "C",
            "title": "批量/断点续传",
            "expected_changes": [
                "实现异步批量处理",
                "添加断点续传机制",
                "实现处理进度跟踪",
                "添加错误恢复策略"
            ],
            "key_files": [
                "src/batch_processing/async_batch.py",
                "src/recovery/"
            ],
            "dependencies": ["asyncio"],
            "code_estimate": "批量处理和断点续传，约400-600行代码"
        },
        
        "lesson16": {
            "module": "C",
            "title": "增量更新",
            "expected_changes": [
                "实现变更检测算法",
                "添加增量索引更新",
                "实现文档版本控制",
                "添加冲突解决策略"
            ],
            "key_files": [
                "src/incremental/update_manager.py",
                "src/versioning/"
            ],
            "dependencies": [],
            "code_estimate": "增量更新功能，约300-500行代码"
        },
        
        "lesson17": {
            "module": "C",
            "title": "结构化数据接入",
            "expected_changes": [
                "实现数据库连接器",
                "添加REST/GraphQL API集成",
                "实现数据格式转换",
                "添加实时数据同步"
            ],
            "key_files": [
                "src/data_connectors/structured_data.py",
                "src/sync/"
            ],
            "dependencies": ["pandas", "requests"],
            "code_estimate": "结构化数据接入，约400-700行代码"
        },
        
        "lesson18": {
            "module": "C",
            "title": "文本清洗与去噪",
            "expected_changes": [
                "实现文本预处理算法",
                "添加噪声检测与清理",
                "实现文本质量评估",
                "添加自动化清洗流程"
            ],
            "key_files": [
                "src/text_processing/cleaning.py",
                "src/quality_assessment/"
            ],
            "dependencies": ["re"],
            "code_estimate": "文本清洗功能，约300-500行代码"
        },
        
        "lesson19": {
            "module": "C",
            "title": "切分策略插件化",
            "expected_changes": [
                "实现插件化架构设计",
                "添加策略注册机制",
                "实现动态策略选择",
                "添加性能基准测试"
            ],
            "key_files": [
                "src/chunking/plugin_system.py",
                "src/chunking/strategies/",
                "src/benchmarking/"
            ],
            "dependencies": [],
            "code_estimate": "插件化切分系统，约400-600行代码"
        },
        
        "lesson20": {
            "module": "C",
            "title": "故障注入与恢复演练",
            "expected_changes": [
                "实现故障注入框架",
                "添加自动恢复机制",
                "实现监控与告警",
                "添加灾难恢复演练"
            ],
            "key_files": [
                "src/fault_injection/",
                "src/recovery/auto_recovery.py",
                "src/monitoring/"
            ],
            "dependencies": ["random"],
            "code_estimate": "故障注入和恢复系统，约500-800行代码"
        }
    }
    
    return lesson_requirements

def save_requirements_analysis(requirements: Dict[str, Any], filename: str = "lesson_requirements.json"):
    """
    保存课程要求分析结果到JSON文件
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(requirements, f, ensure_ascii=False, indent=2)
    print(f"课程要求分析结果已保存到 {filename}")

def print_summary(requirements: Dict[str, Any]):
    """
    打印课程要求分析摘要
    """
    print("\n=== 课程要求分析摘要 ===")
    
    modules = {"A": [], "B": [], "C": []}
    for lesson_id, info in requirements.items():
        modules[info["module"]].append(lesson_id)
    
    for module, lessons in modules.items():
        print(f"\n模块{module}: {len(lessons)}个课程")
        for lesson in lessons:
            info = requirements[lesson]
            print(f"  {lesson}: {info['title']}")
            print(f"    预期变更: {len(info['expected_changes'])}项")
            print(f"    关键文件: {len(info['key_files'])}个")
            print(f"    新增依赖: {len(info['dependencies'])}个")

if __name__ == "__main__":
    # 分析课程要求
    requirements = analyze_lesson_requirements()
    
    # 保存结果
    save_requirements_analysis(requirements)
    
    # 打印摘要
    print_summary(requirements)