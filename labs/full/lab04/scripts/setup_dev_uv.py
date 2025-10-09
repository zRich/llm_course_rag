#!/usr/bin/env python3
"""
开发环境设置脚本 - 使用uv管理依赖
用于快速设置RAG系统的开发环境
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    console = Console()
except ImportError:
    # 如果rich未安装，使用基本的print
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    console = Console()
    
    def Confirm(prompt):
        return input(f"{prompt} (y/n): ").lower().startswith('y')
    
    def Prompt(prompt, default=None):
        if default:
            return input(f"{prompt} [{default}]: ") or default
        return input(f"{prompt}: ")

class DevEnvironmentSetup:
    """开发环境设置器 - 使用uv"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.src_dir = self.project_root / "src"
        self.pyproject_file = self.project_root / "pyproject.toml"
        self.env_file = self.project_root / ".env"
        
    def check_python_version(self) -> bool:
        """检查Python版本"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 12:
            console.print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            console.print(f"❌ Python版本过低: {version.major}.{version.minor}.{version.micro}")
            console.print("   需要Python 3.12或更高版本")
            return False
    
    def check_uv_installation(self) -> bool:
        """检查uv是否已安装"""
        if shutil.which("uv"):
            try:
                result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
                console.print(f"✅ uv已安装: {result.stdout.strip()}")
                return True
            except Exception:
                console.print("❌ uv安装异常")
                return False
        else:
            console.print("❌ uv未安装")
            return False
    
    def install_uv(self) -> bool:
        """安装uv"""
        console.print("正在安装uv...")
        try:
            # 使用官方安装脚本
            subprocess.run([
                "curl", "-LsSf", "https://astral.sh/uv/install.sh"
            ], check=True, stdout=subprocess.PIPE)
            
            # 或者使用pip安装
            subprocess.run([
                sys.executable, "-m", "pip", "install", "uv"
            ], check=True)
            
            console.print("✅ uv安装成功")
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"❌ uv安装失败: {e}")
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
    
    def sync_dependencies(self) -> bool:
        """使用uv同步依赖"""
        console.print("正在使用uv同步依赖...")
        try:
            # 切换到项目目录
            os.chdir(self.project_root)
            
            # 同步依赖
            subprocess.run(["uv", "sync"], check=True)
            console.print("✅ 依赖同步成功")
            
            # 安装开发依赖
            if Confirm("是否安装开发依赖？"):
                subprocess.run(["uv", "sync", "--extra", "dev"], check=True)
                console.print("✅ 开发依赖安装成功")
            
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"❌ 依赖同步失败: {e}")
            return False
    
    def create_env_file(self) -> bool:
        """创建.env配置文件"""
        if self.env_file.exists():
            if not Confirm("发现已存在的.env文件，是否覆盖？"):
                console.print("跳过.env文件创建")
                return True
        
        console.print("\\n创建.env配置文件...")
        
        # 获取用户输入
        volcengine_api_key = Prompt("请输入火山引擎API Key")
        if not volcengine_api_key:
            console.print("❌ 火山引擎API Key是必需的")
            return False
        
        database_url = Prompt(
            "请输入数据库URL", 
            "postgresql://rag_user:rag_password@localhost:15432/rag_db"
        )
        
        redis_url = Prompt("请输入Redis URL", "redis://localhost:16379/0")
        qdrant_url = Prompt("请输入Qdrant URL", "http://localhost:6333")
        
        # 创建.env文件内容
        env_content = f\"\"\"# 应用配置
APP_NAME=RAG System
APP_VERSION=1.0.0
APP_DEBUG=true
APP_HOST=0.0.0.0
APP_PORT=8000

# 数据库配置
DATABASE_URL={database_url}
DATABASE_ECHO=false

# Redis配置
REDIS_URL={redis_url}
REDIS_DECODE_RESPONSES=true

# Qdrant配置
QDRANT_URL={qdrant_url}
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=documents

# 火山引擎豆包大模型配置
VOLCENGINE_API_KEY={volcengine_api_key}
VOLCENGINE_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
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
LOG_FILE=logs/app.log
LOG_ROTATION=1 day
LOG_RETENTION=30 days

# 安全配置
SECRET_KEY=your-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# 文本处理配置
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CHUNKS_PER_DOCUMENT=1000

# 向量搜索配置
VECTOR_SEARCH_TOP_K=5
VECTOR_SEARCH_SCORE_THRESHOLD=0.7

# 缓存配置
CACHE_TTL=3600
CACHE_MAX_SIZE=1000
\"\"\"
        
        try:
            with open(self.env_file, 'w', encoding='utf-8') as f:
                f.write(env_content)
            console.print("✅ .env文件创建成功")
            return True
        except Exception as e:
            console.print(f"❌ .env文件创建失败: {e}")
            return False
    
    def setup_database(self) -> bool:
        """设置数据库"""
        console.print("\\n设置数据库...")
        
        try:
            # 启动数据库服务
            console.print("启动数据库服务...")
            subprocess.run([
                "docker-compose", "-f", "../../docker-compose.yml", "up", "-d", "postgres", "redis", "qdrant"
            ], check=True, cwd=self.project_root)
            
            console.print("等待数据库启动...")
            import time
            time.sleep(10)
            
            # 运行数据库迁移
            console.print("运行数据库迁移...")
            subprocess.run([
                "uv", "run", "alembic", "upgrade", "head"
            ], check=True, cwd=self.project_root)
            
            console.print("✅ 数据库设置成功")
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
                console.print(f"✅ 创建目录: {directory}")
            except Exception as e:
                console.print(f"❌ 创建目录失败 {directory}: {e}")
                return False
        
        return True
    
    def show_next_steps(self):
        """显示后续步骤"""
        console.print("\\n" + "="*50)
        console.print("🎉 开发环境设置完成！")
        console.print("="*50)
        console.print("\\n后续步骤:")
        console.print("1. 启动应用:")
        console.print("   uv run python src/main.py")
        console.print("\\n2. 或者使用脚本命令:")
        console.print("   uv run rag-server")
        console.print("\\n3. 访问API文档:")
        console.print("   http://localhost:8000/docs")
        console.print("\\n4. 运行测试:")
        console.print("   uv run python scripts/test_system.py")
        console.print("\\n5. 开发工具:")
        console.print("   uv run black src/  # 代码格式化")
        console.print("   uv run flake8 src/  # 代码检查")
        console.print("   uv run pytest  # 运行测试")
        console.print("\\n6. 管理依赖:")
        console.print("   uv add <package>  # 添加依赖")
        console.print("   uv remove <package>  # 移除依赖")
        console.print("   uv sync  # 同步依赖")
        console.print("   uv lock  # 更新锁文件")
    
    def run_setup(self):
        """运行完整的设置流程"""
        console.print("🚀 开始设置RAG系统开发环境 (使用uv)")
        console.print("="*50)
        
        # 检查Python版本
        if not self.check_python_version():
            return False
        
        # 检查uv安装
        if not self.check_uv_installation():
            if Confirm("是否安装uv？"):
                if not self.install_uv():
                    return False
            else:
                console.print("❌ 需要安装uv才能继续")
                return False
        
        # 检查系统依赖
        missing_deps = self.check_system_dependencies()
        if missing_deps:
            console.print(f"\\n⚠️  缺少系统依赖: {', '.join(missing_deps)}")
            console.print("请先安装这些依赖后再继续")
            if not Confirm("是否继续？"):
                return False
        
        # 同步依赖
        if not self.sync_dependencies():
            return False
        
        # 创建.env文件
        if not self.create_env_file():
            return False
        
        # 创建目录
        if not self.create_directories():
            return False
        
        # 设置数据库
        if Confirm("是否设置数据库？"):
            self.setup_database()
        
        # 显示后续步骤
        self.show_next_steps()
        
        return True

def main():
    """主函数"""
    setup = DevEnvironmentSetup()
    
    try:
        success = setup.run_setup()
        if success:
            console.print("\\n✅ 设置完成！")
        else:
            console.print("\\n❌ 设置失败！")
            sys.exit(1)
    except KeyboardInterrupt:
        console.print("\\n\\n⚠️  设置被用户中断")
        sys.exit(1)
    except Exception as e:
        console.print(f"\\n❌ 设置过程中出现错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()