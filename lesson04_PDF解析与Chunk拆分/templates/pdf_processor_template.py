#!/usr/bin/env python3
"""
PDF处理器模板
供学生完成Exercise使用

任务：实现完整的PDF文档处理流水线
包括：文档解析、文本提取、Chunk拆分、质量评估

作者: [学生姓名]
学号: [学生学号]
日期: [完成日期]
"""

import fitz  # PyMuPDF
import re
import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentInfo:
    """文档信息数据类"""
    file_path: str
    page_count: int
    total_chars: int
    metadata: Dict
    is_encrypted: bool
    
    # TODO: 添加更多文档属性
    # 提示：可以添加文件大小、创建时间、修改时间等


@dataclass
class ChunkData:
    """Chunk数据类"""
    text: str
    start_pos: int
    end_pos: int
    length: int
    page_num: Optional[int] = None
    
    # TODO: 添加更多Chunk属性
    # 提示：可以添加chunk_id、质量评分、语义标签等


class PDFProcessor:
    """PDF处理器主类"""
    
    def __init__(self, max_chunk_size: int = 1000, overlap_size: int = 100):
        """
        初始化PDF处理器
        
        Args:
            max_chunk_size: 最大chunk长度
            overlap_size: chunk重叠长度
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
        # TODO: 初始化语义边界模式
        # 提示：参考课堂演示的boundary_patterns
        self.boundary_patterns = [
            # 在这里添加正则表达式模式
        ]
    
    def extract_document_info(self, pdf_path: str) -> DocumentInfo:
        """
        提取文档基本信息
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            DocumentInfo对象
            
        TODO: 实现文档信息提取
        提示：
        1. 使用fitz.open()打开文档
        2. 获取页数、元数据、加密状态
        3. 计算总字符数（需要遍历所有页面）
        4. 记得关闭文档
        """
        # 检查文件是否存在
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"文件不存在: {pdf_path}")
        
        # TODO: 实现文档信息提取逻辑
        doc = None
        try:
            # 1. 打开文档
            # doc = fitz.open(pdf_path)
            
            # 2. 获取基本信息
            # page_count = ?
            # metadata = ?
            # is_encrypted = ?
            
            # 3. 计算总字符数
            # total_chars = 0
            # for page_num in range(page_count):
            #     page = doc[page_num]
            #     text = page.get_text()
            #     total_chars += len(text)
            
            # 4. 创建DocumentInfo对象
            # return DocumentInfo(...)
            
            pass  # 删除这行，实现上述逻辑
            
        except Exception as e:
            raise Exception(f"文档信息提取失败: {str(e)}")
        finally:
            if doc:
                doc.close()
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        从PDF提取所有文本
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            提取的完整文本
            
        TODO: 实现文本提取
        提示：
        1. 遍历所有页面
        2. 使用page.get_text()提取文本
        3. 合并所有页面的文本
        4. 处理可能的编码问题
        """
        # TODO: 实现文本提取逻辑
        pass
    
    def split_text_into_chunks(self, text: str) -> List[ChunkData]:
        """
        将文本拆分为chunks
        
        Args:
            text: 待拆分的文本
            
        Returns:
            ChunkData对象列表
            
        TODO: 实现智能文本拆分
        提示：
        1. 实现语义边界识别
        2. 考虑重叠策略
        3. 处理边界情况
        4. 创建ChunkData对象
        """
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # TODO: 实现拆分逻辑
            # 1. 确定当前chunk的结束位置
            # end_pos = self._find_split_point(text, current_pos, current_pos + self.max_chunk_size)
            
            # 2. 提取chunk文本
            # chunk_text = text[current_pos:end_pos].strip()
            
            # 3. 创建ChunkData对象
            # if chunk_text:
            #     chunk = ChunkData(
            #         text=chunk_text,
            #         start_pos=current_pos,
            #         end_pos=end_pos,
            #         length=len(chunk_text)
            #     )
            #     chunks.append(chunk)
            
            # 4. 更新位置（考虑重叠）
            # current_pos = ?
            
            pass  # 删除这行，实现上述逻辑
        
        return chunks
    
    def _find_split_point(self, text: str, start: int, max_end: int) -> int:
        """
        寻找最优的文本分割点
        
        Args:
            text: 文本
            start: 起始位置
            max_end: 最大结束位置
            
        Returns:
            最优分割点位置
            
        TODO: 实现分割点查找算法
        提示：
        1. 在指定范围内搜索边界模式
        2. 按优先级选择最佳边界
        3. 如果没有找到边界，在最大长度处分割
        """
        if max_end >= len(text):
            return len(text)
        
        # TODO: 实现分割点查找逻辑
        # 搜索范围
        search_text = text[start:max_end]
        
        # 按优先级寻找边界
        for pattern in self.boundary_patterns:
            # matches = re.finditer(pattern, search_text)
            # if matches:
            #     # 选择最接近最大长度的边界
            #     pass
            pass
        
        # 如果没有找到合适边界，强制分割
        return max_end
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """
        完整的PDF处理流水线
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            处理结果字典
            
        TODO: 整合所有处理步骤
        提示：
        1. 提取文档信息
        2. 提取文本内容
        3. 进行chunk拆分
        4. 计算统计信息
        5. 返回完整结果
        """
        try:
            print(f"开始处理PDF文档: {pdf_path}")
            
            # TODO: 实现完整处理流程
            # 1. 提取文档信息
            # doc_info = self.extract_document_info(pdf_path)
            # print(f"文档页数: {doc_info.page_count}")
            
            # 2. 提取文本
            # text = self.extract_text_from_pdf(pdf_path)
            # print(f"文本长度: {len(text)} 字符")
            
            # 3. 拆分chunks
            # chunks = self.split_text_into_chunks(text)
            # print(f"生成chunks: {len(chunks)} 个")
            
            # 4. 计算统计信息
            # stats = self._calculate_statistics(chunks)
            
            # 5. 返回结果
            # return {
            #     'document_info': doc_info,
            #     'text_length': len(text),
            #     'chunks': chunks,
            #     'statistics': stats,
            #     'success': True
            # }
            
            pass  # 删除这行，实现上述逻辑
            
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }
    
    def _calculate_statistics(self, chunks: List[ChunkData]) -> Dict:
        """
        计算chunk统计信息
        
        Args:
            chunks: chunk列表
            
        Returns:
            统计信息字典
            
        TODO: 实现统计信息计算
        提示：计算平均长度、最大最小长度、总字符数等
        """
        if not chunks:
            return {}
        
        # TODO: 实现统计计算
        lengths = [chunk.length for chunk in chunks]
        
        return {
            'chunk_count': len(chunks),
            'avg_length': sum(lengths) / len(lengths),
            'max_length': max(lengths),
            'min_length': min(lengths),
            'total_chars': sum(lengths)
        }


class ChunkQualityAnalyzer:
    """Chunk质量分析器"""
    
    def __init__(self):
        """初始化质量分析器"""
        # TODO: 定义质量评估标准
        pass
    
    def analyze_chunk_quality(self, chunks: List[ChunkData]) -> Dict:
        """
        分析chunk质量
        
        Args:
            chunks: chunk列表
            
        Returns:
            质量分析结果
            
        TODO: 实现质量分析
        提示：
        1. 长度分布分析
        2. 语义完整性检查
        3. 边界质量评估
        4. 重叠效果分析
        """
        # TODO: 实现质量分析逻辑
        pass


def exercise_main():
    """
    Exercise主函数
    
    任务：
    1. 实现PDFProcessor类的所有TODO方法
    2. 处理examples/exercise_document.pdf文件
    3. 输出处理结果和统计信息
    4. 分析chunk质量
    
    评分标准：
    - PDF解析正确性 (25分)
    - 文本提取完整性 (25分)
    - Chunk拆分合理性 (25分)
    - 统计信息准确性 (15分)
    - 代码质量和注释 (10分)
    """
    
    # 初始化处理器
    processor = PDFProcessor(max_chunk_size=500, overlap_size=50)
    
    # 处理示例文档
    pdf_path = "examples/exercise_document.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"错误: 找不到文件 {pdf_path}")
        print("请确保examples目录下有exercise_document.pdf文件")
        return
    
    # TODO: 调用处理方法并输出结果
    # result = processor.process_pdf(pdf_path)
    
    # if result.get('success'):
    #     print("=== PDF处理结果 ===")
    #     # 输出文档信息
    #     # 输出统计信息
    #     # 显示前几个chunk的内容
    # else:
    #     print(f"处理失败: {result.get('error')}")
    
    print("TODO: 完成exercise_main函数的实现")


if __name__ == "__main__":
    print("=== PDF处理器Exercise ===")
    print("请完成所有TODO标记的方法实现")
    print()
    
    # 检查PyMuPDF是否安装
    try:
        import fitz
        print(f"✓ PyMuPDF已安装，版本: {fitz.version}")
    except ImportError:
        print("✗ PyMuPDF未安装，请运行: pip install PyMuPDF")
        exit(1)
    
    # 运行Exercise
    exercise_main()
    
    print("\n=== Exercise完成检查清单 ===")
    print("□ extract_document_info方法已实现")
    print("□ extract_text_from_pdf方法已实现")
    print("□ split_text_into_chunks方法已实现")
    print("□ _find_split_point方法已实现")
    print("□ process_pdf方法已实现")
    print("□ _calculate_statistics方法已实现")
    print("□ exercise_main函数已完成")
    print("□ 输出结果格式正确")
    print("□ 代码注释完整")
    print("□ 异常处理完善")