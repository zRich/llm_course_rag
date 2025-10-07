#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档管理器测试脚本

测试文档管理器的统一文档解析功能，包括：
- 多种文档格式解析
- 批量文档处理
- 元数据提取
- 解析器管理
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.document.document_manager import document_manager
from src.document.parser import DocumentParser
from src.document.pdf_parser import PDFParser
from src.document.docx_parser import DocxParser
from src.document.txt_parser import TxtParser

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_document_manager_basic():
    """测试文档管理器基本功能"""
    print("\n=== 测试文档管理器基本功能 ===")
    
    # 测试支持的文件扩展名
    supported_extensions = document_manager.get_supported_extensions()
    print(f"支持的文件扩展名: {supported_extensions}")
    
    # 获取解析器信息
    parser_info = document_manager.get_parser_info()
    print(f"所有解析器信息: {parser_info}")
    
    # 测试文件支持检查
    test_files = [
        'test.pdf', 'test.docx', 'test.doc', 
        'test.txt', 'test.md', 'test.rst',
        'test.xlsx', 'test.pptx'  # 不支持的格式
    ]
    
    for file in test_files:
        supported = document_manager.can_parse(file)
        print(f"文件 {file} 是否支持: {supported}")

def test_single_document_parsing():
    """测试单个文档解析"""
    print("\n=== 测试单个文档解析 ===")
    
    test_docs_dir = project_root / "test_documents"
    
    # 测试不同格式的文档
    test_files = [
        "sample.pdf",
        "sample.docx", 
        "sample.txt",
        "sample.md"
    ]
    
    for filename in test_files:
        file_path = test_docs_dir / filename
        
        if not file_path.exists():
            print(f"跳过测试文件（不存在）: {file_path}")
            continue
        
        try:
            print(f"\n正在解析: {filename}")
            
            # 解析文档内容
            content = document_manager.parse_document(str(file_path))
            print(f"解析成功! 内容长度: {len(content.content)} 字符")
            print(f"内容预览: {content.content[:150]}...")
            
            # 提取元数据
            metadata = document_manager.extract_metadata(str(file_path))
            print(f"元数据:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
            
        except Exception as e:
            print(f"解析失败: {e}")
            logger.error(f"文档解析错误 {filename}: {e}", exc_info=True)

def test_batch_document_parsing():
    """测试批量文档解析"""
    print("\n=== 测试批量文档解析 ===")
    
    test_docs_dir = project_root / "test_documents"
    
    if not test_docs_dir.exists():
        print(f"测试文档目录不存在: {test_docs_dir}")
        return
    
    # 获取所有支持的文档文件
    supported_files = []
    for file_path in test_docs_dir.iterdir():
        if file_path.is_file() and document_manager.can_parse(str(file_path)):
            supported_files.append(str(file_path))
    
    if not supported_files:
        print("没有找到支持的测试文档")
        return
    
    print(f"找到 {len(supported_files)} 个支持的文档文件")
    
    try:
        # 批量解析文档
        results = document_manager.parse_batch(supported_files)
        
        print(f"批量解析完成! 成功解析 {len(results)} 个文档")
        
        for file_path, content in results.items():
            filename = Path(file_path).name
            print(f"  {filename}: {len(content)} 字符")
        
    except Exception as e:
        print(f"批量解析失败: {e}")
        logger.error(f"批量解析错误: {e}", exc_info=True)

def test_document_search():
    """测试文档搜索功能（简化版本）"""
    print("\n=== 测试文档搜索功能 ===")
    
    test_docs_dir = project_root / "test_documents"
    
    if not test_docs_dir.exists():
        print(f"测试文档目录不存在: {test_docs_dir}")
        return
    
    # 由于DocumentManager没有search_documents方法，这里实现简单的文件名搜索
    search_keywords = ["测试", "文档", "PDF", "示例"]
    
    for keyword in search_keywords:
        try:
            print(f"\n搜索关键词: '{keyword}'")
            # 查找包含关键词的文件
            all_files = list(test_docs_dir.iterdir())
            results = [f for f in all_files if f.is_file() and keyword.lower() in f.name.lower()]
            
            if results:
                print(f"找到 {len(results)} 个匹配的文档:")
                for file_path in results:
                    filename = file_path.name
                    print(f"  {filename}: 文件名匹配")
                    if document_manager.can_parse(str(file_path)):
                        try:
                            content = document_manager.parse_document(str(file_path))
                            preview = content.content[:100] if hasattr(content, 'content') else str(content)[:100]
                            print(f"    - {preview}...")
                        except:
                            print(f"    - 无法预览内容")
            else:
                print("没有找到匹配的文档")
        
        except Exception as e:
            print(f"搜索失败: {e}")
            logger.error(f"文档搜索错误: {e}", exc_info=True)

def test_parser_registration():
    """测试解析器注册功能"""
    print("\n=== 测试解析器注册功能 ===")
    
    try:
        # 创建自定义解析器
        class CustomParser(DocumentParser):
            def __init__(self):
                super().__init__()
                self.supported_extensions = ['.custom']
            
            def can_parse(self, file_path: str) -> bool:
                """检查是否可以解析指定文件"""
                return Path(file_path).suffix.lower() in [ext.lower() for ext in self.supported_extensions]
            
            def parse(self, file_path: str) -> str:
                return "Custom parser content"
            
            def extract_metadata(self, file_path: str) -> dict:
                return {'parser': 'custom', 'type': 'custom'}
        
        # 注册自定义解析器
        custom_parser = CustomParser()
        
        # 检查document_manager是否有register_parser方法
        if hasattr(document_manager, 'register_parser'):
            document_manager.register_parser('.custom', custom_parser)
            print("✓ 自定义解析器注册成功")
            
            # 测试注册后的功能
            supported_extensions = document_manager.get_supported_extensions()
            print(f"注册后支持的扩展名: {supported_extensions}")
            
            # 测试自定义解析器
            if document_manager.can_parse('test.custom'):
                print("✓ 自定义文件格式支持检查通过")
            else:
                print("✗ 自定义文件格式支持检查失败")
            
            parser_info = document_manager.get_parser_info()
            print(f"解析器信息: {parser_info}")
        else:
            print("✗ DocumentManager不支持动态注册解析器")
            print("当前支持的解析器:")
            parser_info = document_manager.get_parser_info()
            for parser_name, info in parser_info.items():
                print(f"  {parser_name}: {info}")
    
    except Exception as e:
        print(f"解析器注册测试失败: {e}")
        logger.error(f"解析器注册测试错误: {e}", exc_info=True)

def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    # 测试不存在的文件
    try:
        document_manager.parse_document("nonexistent.pdf")
    except Exception as e:
        print(f"不存在文件的错误处理: {e}")
    
    # 测试不支持的文件格式
    try:
        document_manager.parse_document("test.unsupported")
    except Exception as e:
        print(f"不支持格式的错误处理: {e}")
    
    # 测试空文件路径
    try:
        document_manager.parse_document("")
    except Exception as e:
        print(f"空文件路径的错误处理: {e}")
    
    # 测试批量解析中的错误
    try:
        invalid_files = ["nonexistent1.pdf", "nonexistent2.txt"]
        results = document_manager.parse_batch(invalid_files)
        print(f"批量解析错误处理: 返回 {len(results)} 个结果")
    except Exception as e:
        print(f"批量解析错误处理: {e}")

def create_test_environment():
    """创建测试环境"""
    print("\n=== 创建测试环境 ===")
    
    # 创建测试文档目录
    test_docs_dir = project_root / "test_documents"
    test_docs_dir.mkdir(exist_ok=True)
    print(f"测试文档目录: {test_docs_dir}")
    
    # 创建示例文本文件
    sample_txt = test_docs_dir / "sample.txt"
    if not sample_txt.exists():
        sample_content = """这是一个示例文本文档。

用于测试文档管理器的文本解析功能。

包含多个段落和不同的内容类型：
- 列表项1
- 列表项2
- 列表项3

数字编号：
1. 第一项
2. 第二项
3. 第三项

这是最后一个段落，包含一些测试关键词：PDF、文档、解析、管理器。"""
        
        sample_txt.write_text(sample_content, encoding='utf-8')
        print(f"创建示例文本文件: {sample_txt}")
    
    # 创建示例Markdown文件
    sample_md = test_docs_dir / "sample.md"
    if not sample_md.exists():
        md_content = """# 示例Markdown文档

这是一个**示例**Markdown文档，用于测试文档管理器。

## 功能特性

- 支持多种文档格式
- 统一的解析接口
- 批量处理能力
- 元数据提取

## 代码示例

```python
from src.document.document_manager import document_manager

# 解析文档
content = document_manager.parse_document('sample.pdf')
print(content)
```

## 总结

文档管理器提供了强大的文档处理能力。"""
        
        sample_md.write_text(md_content, encoding='utf-8')
        print(f"创建示例Markdown文件: {sample_md}")
    
    # 提示用户添加其他格式的文件
    other_files = {
        "sample.pdf": "PDF文档",
        "sample.docx": "Word文档"
    }
    
    for filename, description in other_files.items():
        file_path = test_docs_dir / filename
        if not file_path.exists():
            print(f"请手动添加{description}到: {file_path}")
    
    return test_docs_dir

def main():
    """主测试函数"""
    print("文档管理器测试开始")
    print("=" * 50)
    
    # 创建测试环境
    create_test_environment()
    
    # 运行测试
    test_document_manager_basic()
    test_single_document_parsing()
    test_batch_document_parsing()
    test_document_search()
    test_parser_registration()
    test_error_handling()
    
    print("\n=== 测试完成 ===")
    print("如果看到解析成功的消息，说明文档管理器工作正常")
    print("如果有错误，请检查依赖安装和测试文件")

if __name__ == "__main__":
    main()