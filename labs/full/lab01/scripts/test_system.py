#!/usr/bin/env python3
"""
RAG系统测试脚本
用于验证系统各个组件的功能
"""

import os
import sys
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

console = Console()

class RAGSystemTester:
    """RAG系统测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1"
        self.client = httpx.AsyncClient(timeout=30.0)
        self.test_results = []
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def log_test(self, test_name: str, success: bool, message: str = "", data: Any = None):
        """记录测试结果"""
        result = {
            "test_name": test_name,
            "success": success,
            "message": message,
            "data": data,
            "timestamp": time.time()
        }
        self.test_results.append(result)
        
        status = "✅" if success else "❌"
        console.print(f"{status} {test_name}: {message}")
        
        if data and isinstance(data, dict):
            for key, value in data.items():
                console.print(f"   {key}: {value}")
    
    async def test_health_check(self):
        """测试健康检查"""
        try:
            response = await self.client.get(f"{self.api_base}/system/health")
            if response.status_code == 200:
                data = response.json()
                self.log_test(
                    "健康检查",
                    True,
                    "系统健康状态正常",
                    {
                        "数据库": "正常" if data.get("database") else "异常",
                        "向量存储": "正常" if data.get("vector_store") else "异常",
                        "嵌入服务": "正常" if data.get("embedding_service") else "异常"
                    }
                )
                return True
            else:
                self.log_test("健康检查", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("健康检查", False, f"连接失败: {str(e)}")
            return False
    
    async def test_system_stats(self):
        """测试系统统计"""
        try:
            response = await self.client.get(f"{self.api_base}/system/stats")
            if response.status_code == 200:
                data = response.json()["data"]
                self.log_test(
                    "系统统计",
                    True,
                    "获取统计信息成功",
                    {
                        "文档数量": data.get("documents", {}).get("total", 0),
                        "分块数量": data.get("chunks", {}).get("total", 0),
                        "向量数量": data.get("vectors", {}).get("total", 0)
                    }
                )
                return True
            else:
                self.log_test("系统统计", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("系统统计", False, f"请求失败: {str(e)}")
            return False
    
    async def test_document_upload(self, pdf_path: str = None):
        """测试文档上传"""
        if not pdf_path:
            # 创建测试PDF文件
            pdf_path = await self.create_test_pdf()
        
        try:
            with open(pdf_path, "rb") as f:
                files = {"file": ("test.pdf", f, "application/pdf")}
                response = await self.client.post(
                    f"{self.api_base}/documents/upload",
                    files=files
                )
            
            if response.status_code == 200:
                data = response.json()["data"]
                self.log_test(
                    "文档上传",
                    True,
                    "文档上传成功",
                    {
                        "文档ID": data.get("document_id"),
                        "文件名": data.get("filename"),
                        "文件大小": f"{data.get('file_size', 0)} bytes",
                        "分块数量": data.get("chunks_count", 0)
                    }
                )
                return data.get("document_id")
            else:
                self.log_test("文档上传", False, f"HTTP {response.status_code}")
                return None
        except Exception as e:
            self.log_test("文档上传", False, f"上传失败: {str(e)}")
            return None
    
    async def test_document_list(self):
        """测试文档列表"""
        try:
            response = await self.client.get(f"{self.api_base}/documents")
            if response.status_code == 200:
                data = response.json()["data"]
                self.log_test(
                    "文档列表",
                    True,
                    "获取文档列表成功",
                    {
                        "文档数量": len(data.get("documents", [])),
                        "总数": data.get("total", 0)
                    }
                )
                return data.get("documents", [])
            else:
                self.log_test("文档列表", False, f"HTTP {response.status_code}")
                return []
        except Exception as e:
            self.log_test("文档列表", False, f"请求失败: {str(e)}")
            return []
    
    async def test_vectorization(self, document_id: str):
        """测试文档向量化"""
        try:
            response = await self.client.post(
                f"{self.api_base}/vectors/vectorize",
                json={"document_ids": [document_id]}
            )
            
            if response.status_code == 200:
                data = response.json()["data"]
                self.log_test(
                    "文档向量化",
                    True,
                    "向量化处理成功",
                    {
                        "处理文档数": data.get("processed_documents", 0),
                        "向量数量": data.get("total_vectors", 0),
                        "处理时间": f"{data.get('processing_time', 0):.2f}s"
                    }
                )
                return True
            else:
                self.log_test("文档向量化", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("文档向量化", False, f"向量化失败: {str(e)}")
            return False
    
    async def test_vector_search(self, query: str = "测试查询"):
        """测试向量搜索"""
        try:
            response = await self.client.post(
                f"{self.api_base}/vectors/search",
                json={
                    "query": query,
                    "top_k": 5,
                    "similarity_threshold": 0.5
                }
            )
            
            if response.status_code == 200:
                data = response.json()["data"]
                results = data.get("results", [])
                self.log_test(
                    "向量搜索",
                    True,
                    "搜索执行成功",
                    {
                        "查询": query,
                        "结果数量": len(results),
                        "最高相似度": f"{max([r.get('similarity', 0) for r in results], default=0):.3f}" if results else "0.000"
                    }
                )
                return results
            else:
                self.log_test("向量搜索", False, f"HTTP {response.status_code}")
                return []
        except Exception as e:
            self.log_test("向量搜索", False, f"搜索失败: {str(e)}")
            return []
    
    async def test_qa_system(self, question: str = "这个文档讲了什么？"):
        """测试问答系统"""
        try:
            response = await self.client.post(
                f"{self.api_base}/qa/ask",
                json={
                    "question": question,
                    "top_k": 3,
                    "similarity_threshold": 0.5
                }
            )
            
            if response.status_code == 200:
                data = response.json()["data"]
                self.log_test(
                    "问答系统",
                    True,
                    "问答执行成功",
                    {
                        "问题": question,
                        "回答长度": len(data.get("answer", "")),
                        "引用数量": len(data.get("sources", [])),
                        "置信度": f"{data.get('confidence', 0):.3f}"
                    }
                )
                return data
            else:
                self.log_test("问答系统", False, f"HTTP {response.status_code}")
                return None
        except Exception as e:
            self.log_test("问答系统", False, f"问答失败: {str(e)}")
            return None
    
    async def create_test_pdf(self) -> str:
        """创建测试PDF文件"""
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            pdf_path = "/tmp/test_document.pdf"
            c = canvas.Canvas(pdf_path, pagesize=letter)
            
            # 添加测试内容
            c.drawString(100, 750, "RAG系统测试文档")
            c.drawString(100, 700, "这是一个用于测试RAG系统的示例文档。")
            c.drawString(100, 650, "文档包含了多种类型的内容，用于验证系统的处理能力。")
            c.drawString(100, 600, "")
            c.drawString(100, 550, "第一章：系统介绍")
            c.drawString(100, 500, "RAG（检索增强生成）是一种结合了信息检索和文本生成的技术。")
            c.drawString(100, 450, "它能够根据用户的问题，从知识库中检索相关信息，")
            c.drawString(100, 400, "然后基于这些信息生成准确的回答。")
            c.drawString(100, 350, "")
            c.drawString(100, 300, "第二章：技术特点")
            c.drawString(100, 250, "1. 高精度：基于向量相似度的精确检索")
            c.drawString(100, 200, "2. 实时性：支持实时问答和动态更新")
            c.drawString(100, 150, "3. 可扩展：支持大规模文档库")
            
            c.save()
            return pdf_path
            
        except ImportError:
            # 如果没有reportlab，创建一个简单的文本文件
            txt_path = "/tmp/test_document.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("""RAG系统测试文档

这是一个用于测试RAG系统的示例文档。
文档包含了多种类型的内容，用于验证系统的处理能力。

第一章：系统介绍
RAG（检索增强生成）是一种结合了信息检索和文本生成的技术。
它能够根据用户的问题，从知识库中检索相关信息，
然后基于这些信息生成准确的回答。

第二章：技术特点
1. 高精度：基于向量相似度的精确检索
2. 实时性：支持实时问答和动态更新
3. 可扩展：支持大规模文档库
""")
            return txt_path
    
    async def run_all_tests(self):
        """运行所有测试"""
        console.print(Panel.fit("🚀 开始RAG系统测试", style="bold blue"))
        
        # 1. 健康检查
        console.print("\n[bold]1. 系统健康检查[/bold]")
        health_ok = await self.test_health_check()
        
        if not health_ok:
            console.print("[red]系统健康检查失败，停止测试[/red]")
            return
        
        # 2. 系统统计
        console.print("\n[bold]2. 系统统计信息[/bold]")
        await self.test_system_stats()
        
        # 3. 文档管理测试
        console.print("\n[bold]3. 文档管理测试[/bold]")
        document_id = await self.test_document_upload()
        await self.test_document_list()
        
        if document_id:
            # 4. 向量化测试
            console.print("\n[bold]4. 向量化测试[/bold]")
            vectorization_ok = await self.test_vectorization(document_id)
            
            if vectorization_ok:
                # 5. 向量搜索测试
                console.print("\n[bold]5. 向量搜索测试[/bold]")
                await self.test_vector_search("RAG系统")
                await self.test_vector_search("技术特点")
                
                # 6. 问答系统测试
                console.print("\n[bold]6. 问答系统测试[/bold]")
                await self.test_qa_system("RAG系统是什么？")
                await self.test_qa_system("这个系统有什么特点？")
        
        # 显示测试结果摘要
        self.show_test_summary()
    
    def show_test_summary(self):
        """显示测试结果摘要"""
        console.print("\n" + "="*60)
        console.print("[bold]测试结果摘要[/bold]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("测试项目", style="cyan")
        table.add_column("状态", justify="center")
        table.add_column("说明", style="dim")
        
        success_count = 0
        for result in self.test_results:
            status = "✅ 通过" if result["success"] else "❌ 失败"
            if result["success"]:
                success_count += 1
            
            table.add_row(
                result["test_name"],
                status,
                result["message"]
            )
        
        console.print(table)
        
        total_tests = len(self.test_results)
        success_rate = (success_count / total_tests * 100) if total_tests > 0 else 0
        
        console.print(f"\n[bold]总计: {success_count}/{total_tests} 通过 ({success_rate:.1f}%)[/bold]")
        
        if success_rate == 100:
            console.print("[green]🎉 所有测试通过！系统运行正常。[/green]")
        elif success_rate >= 80:
            console.print("[yellow]⚠️  大部分测试通过，系统基本正常。[/yellow]")
        else:
            console.print("[red]❌ 多个测试失败，请检查系统配置。[/red]")


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG系统测试脚本")
    parser.add_argument("--url", default="http://localhost:8000", help="API服务地址")
    parser.add_argument("--pdf", help="测试PDF文件路径")
    args = parser.parse_args()
    
    async with RAGSystemTester(args.url) as tester:
        await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())