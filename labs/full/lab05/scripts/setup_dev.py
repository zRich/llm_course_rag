#!/usr/bin/env python3
"""
开发环境设置脚本
用于快速设置RAG系统的开发环境
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

console = Console()

class DevEnvironmentSetup:
    """开发环境设置器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.src_dir = self.project_root / "src"
        self.pyproject_file = self.project_root / "pyproject.toml"
        self.env_file = self.project_root / ".env"
        
    def check_python_version(self) -> bool:
        """检查Python版本"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 9:
            console.print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            console.print(f"❌ Python版本过低: {version.major}.{version.minor}.{version.micro}")
            console.print("   需要Python 3.9或更高版本")
            return False
    
    def check_system_dependencies(self) -> List[str]:
        """检查系统依赖"""
        dependencies = ["git", "curl", "docker-compose"]
        missing = []
        
        for dep in dependencies:
            if shutil.which(dep):
                console.print(f"✅ {dep}: 已安装")
            else:
                console.print(f"❌ {dep}: 未安装")
                missing.append(dep)
        
        return missing
    
    def create_virtual_environment(self) -> bool:
        """创建虚拟环境"""
        venv_path = self.project_root / "venv"
        
        if venv_path.exists():
            if Confirm.ask("虚拟环境已存在，是否重新创建？"):
                shutil.rmtree(venv_path)
            else:
                console.print("✅ 使用现有虚拟环境")
                return True
        
        try:
            console.print("🔄 创建虚拟环境...")
            subprocess.run([
                sys.executable, "-m", "venv", str(venv_path)
            ], check=True, capture_output=True)
            
            console.print("✅ 虚拟环境创建成功")
            return True
            
        except subprocess.CalledProcessError as e:
            console.print(f"❌ 虚拟环境创建失败: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """安装Python依赖"""
        if not self.pyproject_file.exists():
            console.print("❌ pyproject.toml文件不存在")
            return False
        
        venv_path = self.project_root / "venv"
        if sys.platform == "win32":
            pip_path = venv_path / "Scripts" / "pip"
        else:
            pip_path = venv_path / "bin" / "pip"
        
        try:
            console.print("🔄 安装Python依赖...")
            subprocess.run([
                str(pip_path), "install", "--upgrade", "pip"
            ], check=True, capture_output=True)
            
            subprocess.run([
                str(pip_path), "install", "-e", "."
            ], check=True, capture_output=True)
            
            console.print("✅ Python依赖安装成功")
            return True
            
        except subprocess.CalledProcessError as e:
            console.print(f"❌ 依赖安装失败: {e}")
            return False
    
    def create_env_file(self) -> bool:
        """创建环境变量文件"""
        if self.env_file.exists():
            if not Confirm.ask(".env文件已存在，是否覆盖？"):
                console.print("✅ 使用现有.env文件")
                return True
        
        # 获取用户输入
        console.print("\n[bold]请输入配置信息：[/bold]")
        
        volcengine_api_key = Prompt.ask("火山引擎API Key", password=True)
        volcengine_base_url = Prompt.ask(
            "火山引擎Base URL", 
            default="https://ark.cn-beijing.volces.com/api/v3"
        )
        
        database_url = Prompt.ask(
            "数据库URL",
            default="postgresql://rag_user:rag_password@localhost:15432/rag_db"
        )
        
        redis_url = Prompt.ask(
            "Redis URL",
            default="redis://localhost:16379/0"
        )
        
        qdrant_host = Prompt.ask("Qdrant主机", default="localhost")
        qdrant_port = Prompt.ask("Qdrant端口", default="6333")
        
        # 创建.env文件内容
        env_content = f"""# RAG系统环境变量配置

# 应用配置
APP_NAME=RAG System
APP_VERSION=1.0.0
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8000
WORKERS=1

# 数据库配置
DATABASE_URL={database_url}

# Redis配置
REDIS_URL={redis_url}

# Qdrant配置
QDRANT_HOST={qdrant_host}
QDRANT_PORT={qdrant_port}
QDRANT_COLLECTION_NAME=rag_documents

# 火山引擎豆包大模型配置
VOLCENGINE_API_KEY={volcengine_api_key}
VOLCENGINE_BASE_URL={volcengine_base_url}
VOLCENGINE_MODEL=doubao-seed-1-6-250615
VOLCENGINE_EMBEDDING_MODEL=doubao-embedding
VOLCENGINE_MAX_TOKENS=2000
VOLCENGINE_TEMPERATURE=0.7

# 文件上传配置
UPLOAD_DIR=uploads
MAX_FILE_SIZE=10485760
ALLOWED_EXTENSIONS=pdf

# 日志配置
LOG_LEVEL=INFO
LOG_DIR=logs
LOG_FILE=rag_system.log

# 安全配置
SECRET_KEY=your-secret-key-change-in-production
ALLOWED_HOSTS=localhost,127.0.0.1

# 文本处理配置
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CHUNKS_PER_DOCUMENT=1000

# 向量搜索配置
DEFAULT_TOP_K=5
DEFAULT_SIMILARITY_THRESHOLD=0.7
MAX_CONTEXT_LENGTH=4000

# 缓存配置
CACHE_TTL=3600
ENABLE_CACHE=true
"""
        
        try:
            with open(self.env_file, "w", encoding="utf-8") as f:
                f.write(env_content)
            
            console.print("✅ .env文件创建成功")
            return True
            
        except Exception as e:
            console.print(f"❌ .env文件创建失败: {e}")
            return False
    
    def setup_database(self) -> bool:
        """设置数据库"""
        if not Confirm.ask("是否启动数据库服务？"):
            return True
        
        try:
            console.print("🔄 启动数据库服务...")
            subprocess.run([
                "docker-compose", "-f", "../../docker-compose.yml", "up", "-d", "postgres", "redis", "qdrant"
            ], check=True, cwd=self.project_root, capture_output=True)
            
            console.print("✅ 数据库服务启动成功")
            
            # 等待数据库启动
            import time
            console.print("⏳ 等待数据库启动...")
            time.sleep(10)
            
            # 运行数据库迁移
            console.print("🔄 运行数据库迁移...")
            venv_path = self.project_root / "venv"
            if sys.platform == "win32":
                python_path = venv_path / "Scripts" / "python"
            else:
                python_path = venv_path / "bin" / "python"
            
            subprocess.run([
                str(python_path), "-m", "alembic", "upgrade", "head"
            ], cwd=self.project_root, check=True)
            
            console.print("✅ 数据库迁移完成")
            return True
            
        except subprocess.CalledProcessError as e:
            console.print(f"❌ 数据库设置失败: {e}")
            return False
    
    def create_directories(self) -> bool:
        """创建必要的目录"""
        directories = [
            self.project_root / "uploads",
            self.project_root / "logs",
            self.project_root / "data",
            self.project_root / "tests" / "data"
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                console.print(f"✅ 目录创建: {directory.name}")
            except Exception as e:
                console.print(f"❌ 目录创建失败 {directory}: {e}")
                return False
        
        return True
    
    def show_next_steps(self):
        """显示后续步骤"""
        console.print("\n" + "="*60)
        console.print("[bold green]🎉 开发环境设置完成！[/bold green]")
        
        console.print("\n[bold]后续步骤：[/bold]")
        console.print("1. 激活虚拟环境：")
        if sys.platform == "win32":
            console.print("   [cyan]venv\\Scripts\\activate[/cyan]")
        else:
            console.print("   [cyan]source venv/bin/activate[/cyan]")
        
        console.print("\n2. 启动开发服务器：")
        console.print("   [cyan]python src/main.py[/cyan]")
        
        console.print("\n3. 访问API文档：")
        console.print("   [cyan]http://localhost:8000/docs[/cyan]")
        
        console.print("\n4. 运行测试：")
        console.print("   [cyan]python scripts/test_system.py[/cyan]")
        
        console.print("\n5. 查看日志：")
        console.print("   [cyan]tail -f logs/rag_system.log[/cyan]")
    
    def run_setup(self):
        """运行完整设置流程"""
        console.print(Panel.fit("🚀 RAG系统开发环境设置", style="bold blue"))
        
        # 1. 检查Python版本
        console.print("\n[bold]1. 检查Python版本[/bold]")
        if not self.check_python_version():
            return False
        
        # 2. 检查系统依赖
        console.print("\n[bold]2. 检查系统依赖[/bold]")
        missing_deps = self.check_system_dependencies()
        if missing_deps:
            console.print(f"[red]请先安装缺失的依赖: {', '.join(missing_deps)}[/red]")
            if not Confirm.ask("是否继续设置？"):
                return False
        
        # 3. 创建虚拟环境
        console.print("\n[bold]3. 创建虚拟环境[/bold]")
        if not self.create_virtual_environment():
            return False
        
        # 4. 安装依赖
        console.print("\n[bold]4. 安装Python依赖[/bold]")
        if not self.install_dependencies():
            return False
        
        # 5. 创建环境变量文件
        console.print("\n[bold]5. 配置环境变量[/bold]")
        if not self.create_env_file():
            return False
        
        # 6. 创建目录
        console.print("\n[bold]6. 创建项目目录[/bold]")
        if not self.create_directories():
            return False
        
        # 7. 设置数据库
        console.print("\n[bold]7. 设置数据库[/bold]")
        if not self.setup_database():
            return False
        
        # 8. 显示后续步骤
        self.show_next_steps()
        
        return True


def main():
    """主函数"""
    setup = DevEnvironmentSetup()
    success = setup.run_setup()
    
    if success:
        console.print("\n[green]✅ 开发环境设置成功！[/green]")
    else:
        console.print("\n[red]❌ 开发环境设置失败！[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()