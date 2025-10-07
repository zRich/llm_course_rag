#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF解析器测试脚本

测试PDF文档解析功能，包括：
- 文档内容解析
- 元数据提取
- 页面提取
- 错误处理
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.document.pdf_parser import PDFParser
from src.document.document_manager import document_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pdf_parser_basic():
    """测试PDF解析器基本功能"""
    print("\n=== 测试PDF解析器基本功能 ===")
    
    parser = PDFParser()
    
    # 测试支持的文件扩展名
    print(f"支持的文件扩展名: {parser.SUPPORTED_EXTENSIONS}")
    
    # 测试文件扩展名检查
    test_files = ['test.pdf', 'test.txt', 'test.docx']
    for file in test_files:
        supported = parser.can_parse(file)
        print(f"文件 {file} 是否支持: {supported}")

def test_pdf_parsing():
    """测试PDF文档解析"""
    print("\n=== 测试PDF文档解析 ===")
    
    # 创建测试PDF文件路径
    test_pdf_path = project_root / "test_documents" / "sample.pdf"
    
    if not test_pdf_path.exists():
        print(f"警告: 测试PDF文件不存在: {test_pdf_path}")
        print("请创建test_documents目录并添加sample.pdf文件")
        return
    
    parser = PDFParser()
    
    try:
        # 解析PDF文档
        print(f"正在解析PDF文件: {test_pdf_path}")
        content = parser.parse(str(test_pdf_path))
        
        print(f"解析成功!")
        print(f"文档内容长度: {len(content)} 字符")
        print(f"文档内容预览: {content[:200]}...")
        
        # 提取元数据
        metadata = parser.extract_metadata(str(test_pdf_path))
        print(f"\n文档元数据:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        # 提取指定页面
        if metadata.get('page_count', 0) > 0:
            page_content = parser.extract_pages(str(test_pdf_path), [1])
            print(f"\n第1页内容长度: {len(page_content)} 字符")
            print(f"第1页内容预览: {page_content[:200]}...")
        
    except Exception as e:
        print(f"解析失败: {e}")
        logger.error(f"PDF解析错误: {e}", exc_info=True)

def test_document_manager_pdf():
    """测试文档管理器的PDF解析功能"""
    print("\n=== 测试文档管理器PDF解析 ===")
    
    test_pdf_path = project_root / "test_documents" / "sample.pdf"
    
    if not test_pdf_path.exists():
        print(f"警告: 测试PDF文件不存在: {test_pdf_path}")
        return
    
    try:
        # 使用文档管理器解析PDF
        print(f"使用文档管理器解析: {test_pdf_path}")
        content = document_manager.parse_document(str(test_pdf_path))
        
        print(f"解析成功!")
        print(f"文档内容长度: {len(content)} 字符")
        
        # 获取元数据
        metadata = document_manager.extract_metadata(str(test_pdf_path))
        print(f"\n元数据:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        # 检查解析器信息
        parser_info = document_manager.get_parser_info('.pdf')
        print(f"\nPDF解析器信息: {parser_info}")
        
    except Exception as e:
        print(f"文档管理器解析失败: {e}")
        logger.error(f"文档管理器PDF解析错误: {e}", exc_info=True)

def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    parser = PDFParser()
    
    # 测试不存在的文件
    try:
        parser.parse("nonexistent.pdf")
    except Exception as e:
        print(f"不存在文件的错误处理: {e}")
    
    # 测试无效的PDF文件
    invalid_pdf_path = project_root / "test_documents" / "invalid.pdf"
    if invalid_pdf_path.exists():
        try:
            parser.parse(str(invalid_pdf_path))
        except Exception as e:
            print(f"无效PDF文件的错误处理: {e}")
    else:
        print("跳过无效PDF文件测试（文件不存在）")

def create_test_environment():
    """创建测试环境"""
    print("\n=== 创建测试环境 ===")
    
    # 创建测试文档目录
    test_docs_dir = project_root / "test_documents"
    test_docs_dir.mkdir(exist_ok=True)
    print(f"测试文档目录: {test_docs_dir}")
    
    # 创建示例PDF文件（如果不存在）
    sample_pdf = test_docs_dir / "sample.pdf"
    if not sample_pdf.exists():
        print(f"请手动添加示例PDF文件到: {sample_pdf}")
        print("或者使用任何PDF文件重命名为sample.pdf")
    
    return test_docs_dir

def main():
    """主测试函数"""
    print("PDF解析器测试开始")
    print("=" * 50)
    
    # 创建测试环境
    create_test_environment()
    
    # 运行测试
    test_pdf_parser_basic()
    test_pdf_parsing()
    test_document_manager_pdf()
    test_error_handling()
    
    print("\n=== 测试完成 ===")
    print("如果看到解析成功的消息，说明PDF解析器工作正常")
    print("如果有错误，请检查依赖安装和测试文件")

if __name__ == "__main__":
    main()