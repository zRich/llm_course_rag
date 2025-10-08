#!/usr/bin/env python3
"""
RAG系统服务启动脚本
提供一键启动所有必要服务的功能
"""

import os
import sys
import time
import subprocess
import signal
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import psutil
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

class ServiceManager:
    """服务管理器"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.services = {}
        self.running_processes = []
        
    def check_dependencies(self) -> bool:
        """检查系统依赖"""
        console.print("[bold blue]检查系统依赖...[/bold blue]")
        
        dependencies = {
            "python": "python3 --version",
            "uv": "uv --version", 
            "docker": "docker --version",
            "docker-compose": "docker-compose --version"
        }
        
        missing_deps = []
        
        for name, cmd in dependencies.items():
            try:
                result = subprocess.run(cmd.split(), capture_output=True, text=True)
                if result.returncode == 0:
                    console.print(f"✅ {name}: {result.stdout.strip()}")
                else:
                    missing_deps.append(name)
                    console.print(f"❌ {name}: 未安装或不可用")
            except FileNotFoundError:
                missing_deps.append(name)
                console.print(f"❌ {name}: 未找到命令")
        
        if missing_deps:
            console.print(f"\n[bold red]缺少依赖: {', '.join(missing_deps)}[/bold red]")
            console.print("请安装缺少的依赖后重试")
            return False
        
        console.print("\n[bold green]所有依赖检查通过！[/bold green]")
        return True
    
    def start_infrastructure(self) -> bool:
        """启动基础设施服务（PostgreSQL, Redis, Qdrant）"""
        console.print("\n[bold blue]启动基础设施服务...[/bold blue]")
        
        # 检查是否存在 docker-compose.yml
        compose_file = self.project_root / "docker-compose.yml"
        if not compose_file.exists():
            self.create_docker_compose()
        
        try:
            # 启动 Docker Compose 服务
            cmd = ["docker-compose", "up", "-d", "postgres", "redis", "qdrant"]
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode != 0:
                console.print(f"[bold red]启动基础设施失败: {result.stderr}[/bold red]")
                return False
            
            console.print("✅ 基础设施服务启动成功")
            
            # 等待服务就绪
            self.wait_for_services()
            return True
            
        except Exception as e:
            console.print(f"[bold red]启动基础设施失败: {e}[/bold red]")
            return False
    
    def create_docker_compose(self):
        """创建 docker-compose.yml 文件"""
        compose_content = """version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: rag_system
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: rag_password
    ports:
      - "15432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag_user -d rag_system"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "16379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT__SERVICE__HTTP_PORT: 6333
      QDRANT__SERVICE__GRPC_PORT: 6334
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
"""
        
        compose_file = self.project_root / "docker-compose.yml"
        with open(compose_file, 'w', encoding='utf-8') as f:
            f.write(compose_content)
        
        console.print("✅ 创建 docker-compose.yml 文件")
    
    def wait_for_services(self):
        """等待服务就绪"""
        services = {
            "PostgreSQL": ("localhost", 15432),
        "Redis": ("localhost", 16379),
            "Qdrant": ("localhost", 6333)
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for service_name, (host, port) in services.items():
                task = progress.add_task(f"等待 {service_name} 就绪...", total=None)
                
                max_retries = 30
                for i in range(max_retries):
                    try:
                        if service_name == "Qdrant":
                            # Qdrant 使用 HTTP 健康检查
                            response = requests.get(f"http://{host}:{port}/health", timeout=2)
                            if response.status_code == 200:
                                break
                        else:
                            # PostgreSQL 和 Redis 使用端口检查
                            import socket
                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            sock.settimeout(2)
                            result = sock.connect_ex((host, port))
                            sock.close()
                            if result == 0:
                                break
                    except:
                        pass
                    
                    time.sleep(1)
                else:
                    console.print(f"[bold red]⚠️  {service_name} 启动超时[/bold red]")
                    continue
                
                progress.update(task, description=f"✅ {service_name} 已就绪")
                time.sleep(0.5)
    
    def setup_environment(self) -> bool:
        """设置环境"""
        console.print("\n[bold blue]设置环境...[/bold blue]")
        
        try:
            # 复制环境变量文件
            env_example = self.project_root / ".env.example"
            env_file = self.project_root / ".env"
            
            if not env_file.exists() and env_example.exists():
                import shutil
                shutil.copy(env_example, env_file)
                console.print("✅ 创建 .env 文件")
            
            # 安装依赖
            console.print("安装 Python 依赖...")
            result = subprocess.run(
                ["uv", "sync", "--extra", "dev"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                console.print(f"[bold red]依赖安装失败: {result.stderr}[/bold red]")
                return False
            
            console.print("✅ Python 依赖安装完成")
            
            # 运行数据库迁移
            console.print("运行数据库迁移...")
            result = subprocess.run(
                ["uv", "run", "alembic", "upgrade", "head"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                console.print(f"[bold yellow]数据库迁移警告: {result.stderr}[/bold yellow]")
            else:
                console.print("✅ 数据库迁移完成")
            
            return True
            
        except Exception as e:
            console.print(f"[bold red]环境设置失败: {e}[/bold red]")
            return False
    
    def start_rag_service(self) -> bool:
        """启动 RAG 服务"""
        console.print("\n[bold blue]启动 RAG 服务...[/bold blue]")
        
        try:
            # 启动 FastAPI 应用
            cmd = [
                "uv", "run", "uvicorn", "src.main:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ]
            
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.running_processes.append(process)
            
            # 等待服务启动
            max_retries = 30
            for i in range(max_retries):
                try:
                    response = requests.get("http://localhost:8000/api/v1/system/health", timeout=2)
                    if response.status_code == 200:
                        console.print("✅ RAG 服务启动成功")
                        return True
                except:
                    pass
                time.sleep(1)
            
            console.print("[bold red]RAG 服务启动超时[/bold red]")
            return False
            
        except Exception as e:
            console.print(f"[bold red]RAG 服务启动失败: {e}[/bold red]")
            return False
    
    def show_service_status(self):
        """显示服务状态"""
        console.print("\n[bold green]🎉 RAG 系统启动完成！[/bold green]")
        
        table = Table(title="服务状态")
        table.add_column("服务", style="cyan")
        table.add_column("地址", style="magenta")
        table.add_column("状态", style="green")
        
        services = [
            ("PostgreSQL", "localhost:15432", "✅ 运行中"),
            ("Redis", "localhost:16379", "✅ 运行中"),
            ("Qdrant", "localhost:6333", "✅ 运行中"),
            ("RAG API", "http://localhost:8000", "✅ 运行中"),
            ("API 文档", "http://localhost:8000/docs", "✅ 可访问"),
        ]
        
        for service, address, status in services:
            table.add_row(service, address, status)
        
        console.print(table)
        
        console.print("\n[bold blue]快速链接:[/bold blue]")
        console.print("• API 文档: http://localhost:8000/docs")
        console.print("• ReDoc: http://localhost:8000/redoc")
        console.print("• 健康检查: http://localhost:8000/api/v1/system/health")
        console.print("• 系统统计: http://localhost:8000/api/v1/system/stats")
        
        console.print("\n[bold yellow]使用 Ctrl+C 停止服务[/bold yellow]")
    
    def stop_services(self):
        """停止所有服务"""
        console.print("\n[bold yellow]正在停止服务...[/bold yellow]")
        
        # 停止 RAG 服务
        for process in self.running_processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        
        # 停止基础设施服务
        try:
            subprocess.run(
                ["docker-compose", "down"],
                cwd=self.project_root,
                capture_output=True
            )
            console.print("✅ 所有服务已停止")
        except:
            console.print("⚠️  停止基础设施服务时出现问题")
    
    def run_tests(self):
        """运行系统测试"""
        console.print("\n[bold blue]运行系统测试...[/bold blue]")
        
        try:
            result = subprocess.run(
                ["uv", "run", "python", "scripts/test_system.py"],
                cwd=self.project_root
            )
            return result.returncode == 0
        except Exception as e:
            console.print(f"[bold red]测试运行失败: {e}[/bold red]")
            return False


def main():
    parser = argparse.ArgumentParser(description="RAG系统服务管理")
    parser.add_argument("--skip-deps", action="store_true", help="跳过依赖检查")
    parser.add_argument("--skip-infra", action="store_true", help="跳过基础设施启动")
    parser.add_argument("--test-only", action="store_true", help="仅运行测试")
    parser.add_argument("--stop", action="store_true", help="停止所有服务")
    
    args = parser.parse_args()
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    
    # 创建服务管理器
    manager = ServiceManager(project_root)
    
    # 显示欢迎信息
    console.print(Panel.fit(
        "[bold blue]RAG 系统服务管理器[/bold blue]\n"
        "自动启动和管理 RAG 系统的所有服务",
        title="🚀 欢迎"
    ))
    
    try:
        if args.stop:
            manager.stop_services()
            return
        
        if args.test_only:
            success = manager.run_tests()
            sys.exit(0 if success else 1)
        
        # 检查依赖
        if not args.skip_deps and not manager.check_dependencies():
            sys.exit(1)
        
        # 启动基础设施
        if not args.skip_infra and not manager.start_infrastructure():
            sys.exit(1)
        
        # 设置环境
        if not manager.setup_environment():
            sys.exit(1)
        
        # 启动 RAG 服务
        if not manager.start_rag_service():
            sys.exit(1)
        
        # 显示服务状态
        manager.show_service_status()
        
        # 等待用户中断
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_services()


if __name__ == "__main__":
    main()