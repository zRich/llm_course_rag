#!/usr/bin/env python3
"""
PDF解析演示示例
演示PyMuPDF的基础用法和常见操作

作者: RAG课程组
日期: 2024年
用途: Lesson 04 课堂演示
"""

import fitz  # PyMuPDF
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


class PDFParserDemo:
    """PDF解析演示类"""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def parse_basic_info(self, pdf_path: str) -> Dict:
        """
        解析PDF基本信息
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            包含基本信息的字典
        """
        try:
            doc = fitz.open(pdf_path)
            
            # 获取基本信息
            info = {
                'file_path': pdf_path,
                'page_count': doc.page_count,
                'metadata': doc.metadata,
                'is_encrypted': doc.is_encrypted,
                'is_pdf': doc.is_pdf,
                'file_size': os.path.getsize(pdf_path)
            }
            
            doc.close()
            return info
            
        except Exception as e:
            return {'error': str(e)}
    
    def extract_text_simple(self, pdf_path: str, page_num: int = 0) -> str:
        """
        简单文本提取演示
        
        Args:
            pdf_path: PDF文件路径
            page_num: 页面编号（从0开始）
            
        Returns:
            提取的文本内容
        """
        try:
            doc = fitz.open(pdf_path)
            
            if page_num >= doc.page_count:
                return f"页面编号超出范围，总页数: {doc.page_count}"
            
            page = doc[page_num]
            text = page.get_text()
            
            doc.close()
            return text
            
        except Exception as e:
            return f"提取失败: {str(e)}"
    
    def extract_text_structured(self, pdf_path: str, page_num: int = 0) -> Dict:
        """
        结构化文本提取演示
        
        Args:
            pdf_path: PDF文件路径
            page_num: 页面编号
            
        Returns:
            结构化的文本信息
        """
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            
            # 获取结构化文本信息
            text_dict = page.get_text("dict")
            
            # 解析文本块
            blocks_info = []
            for block in text_dict["blocks"]:
                if "lines" in block:  # 文本块
                    block_info = {
                        'type': 'text',
                        'bbox': block['bbox'],  # 边界框
                        'lines_count': len(block['lines']),
                        'text_preview': ''
                    }
                    
                    # 提取前100个字符作为预览
                    full_text = ""
                    for line in block['lines']:
                        for span in line['spans']:
                            full_text += span['text']
                    
                    block_info['text_preview'] = full_text[:100] + "..." if len(full_text) > 100 else full_text
                    blocks_info.append(block_info)
                    
                else:  # 图像块
                    blocks_info.append({
                        'type': 'image',
                        'bbox': block['bbox']
                    })
            
            doc.close()
            
            return {
                'page_info': {
                    'width': text_dict['width'],
                    'height': text_dict['height']
                },
                'blocks': blocks_info,
                'total_blocks': len(blocks_info)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def extract_images_info(self, pdf_path: str, page_num: int = 0) -> List[Dict]:
        """
        提取图像信息演示
        
        Args:
            pdf_path: PDF文件路径
            page_num: 页面编号
            
        Returns:
            图像信息列表
        """
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            
            images = page.get_images()
            images_info = []
            
            for img_index, img in enumerate(images):
                # img是一个元组，包含图像的各种信息
                img_info = {
                    'index': img_index,
                    'xref': img[0],  # 图像的交叉引用编号
                    'smask': img[1],  # 软掩码
                    'width': img[2],
                    'height': img[3],
                    'bpc': img[4],   # 每个颜色分量的位数
                    'colorspace': img[5],
                    'alt': img[6],   # 替代文本
                    'name': img[7],  # 图像名称
                    'filter': img[8]  # 过滤器
                }
                images_info.append(img_info)
            
            doc.close()
            return images_info
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def demo_all_features(self, pdf_path: str) -> Dict:
        """
        综合演示所有功能
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            完整的解析结果
        """
        if not os.path.exists(pdf_path):
            return {'error': f'文件不存在: {pdf_path}'}
        
        print(f"=== PDF解析演示：{pdf_path} ===\n")
        
        # 1. 基本信息
        print("1. 基本信息解析...")
        basic_info = self.parse_basic_info(pdf_path)
        print(f"   页数: {basic_info.get('page_count', 'N/A')}")
        print(f"   文件大小: {basic_info.get('file_size', 0)} bytes")
        print(f"   是否加密: {basic_info.get('is_encrypted', 'N/A')}")
        
        if 'metadata' in basic_info:
            metadata = basic_info['metadata']
            print(f"   标题: {metadata.get('title', '未知')}")
            print(f"   作者: {metadata.get('author', '未知')}")
        
        # 2. 文本提取
        print("\n2. 文本提取演示...")
        text = self.extract_text_simple(pdf_path, 0)
        print(f"   第一页文本长度: {len(text)} 字符")
        print(f"   文本预览: {text[:200]}..." if len(text) > 200 else f"   文本内容: {text}")
        
        # 3. 结构化信息
        print("\n3. 结构化信息解析...")
        structured = self.extract_text_structured(pdf_path, 0)
        if 'error' not in structured:
            print(f"   页面尺寸: {structured['page_info']['width']} x {structured['page_info']['height']}")
            print(f"   文本块数量: {structured['total_blocks']}")
            
            # 显示前3个文本块的预览
            text_blocks = [b for b in structured['blocks'] if b['type'] == 'text']
            for i, block in enumerate(text_blocks[:3]):
                print(f"   文本块{i+1}: {block['text_preview']}")
        
        # 4. 图像信息
        print("\n4. 图像信息提取...")
        images = self.extract_images_info(pdf_path, 0)
        if images and 'error' not in images[0]:
            print(f"   图像数量: {len(images)}")
            for i, img in enumerate(images[:3]):  # 只显示前3个
                print(f"   图像{i+1}: {img['width']}x{img['height']}, 颜色空间: {img['colorspace']}")
        else:
            print("   未发现图像或提取失败")
        
        print("\n=== 演示完成 ===")
        
        return {
            'basic_info': basic_info,
            'text_length': len(text),
            'structured_info': structured,
            'images_count': len(images) if images and 'error' not in images[0] else 0
        }


def main():
    """主演示函数"""
    demo = PDFParserDemo()
    
    # 检查示例文件
    sample_files = [
        "sample_document.pdf",
        "exercise_document.pdf"
    ]
    
    current_dir = Path(__file__).parent
    
    for filename in sample_files:
        file_path = current_dir / filename
        if file_path.exists():
            print(f"\n{'='*60}")
            print(f"演示文件: {filename}")
            print('='*60)
            demo.demo_all_features(str(file_path))
        else:
            print(f"\n注意: 示例文件 {filename} 不存在")
            print("请准备PDF文件用于演示")


if __name__ == "__main__":
    # 检查PyMuPDF是否安装
    try:
        import fitz
        print("PyMuPDF (fitz) 已正确安装")
        print(f"版本: {fitz.version}")
        main()
    except ImportError:
        print("错误: PyMuPDF 未安装")
        print("请运行: pip install PyMuPDF")