#!/usr/bin/env python3
"""
RAG系统启动脚本
支持开发模式和生产模式启动
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def get_project_root():
    """获取项目根目录"""
    return Path(__file__).parent.parent


def check_uv_installed():
    """检查uv是否已安装"""
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_uv():
    """安装uv包管理器"""
    print("正在安装uv包管理器...")
    try:
        subprocess.run([
            "curl", "-LsSf", "https://astral.sh/uv/install.sh"
        ], check=True, stdout=subprocess.PIPE)
        subprocess.run(["sh"], input=b"", check=True)
        print("uv安装完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"uv安装失败: {e}")
        return False


def sync_dependencies():
    """同步项目依赖"""
    print("正在同步项目依赖...")
    try:
        subprocess.run(["uv", "sync", "--extra", "dev"], check=True)
        print("依赖同步完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"依赖同步失败: {e}")
        return False


def check_env_file():
    """检查环境配置文件"""
    project_root = get_project_root()
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"
    
    if not env_file.exists():
        if env_example.exists():
            print("正在创建环境配置文件...")
            env_file.write_text(env_example.read_text())
            print(f"已创建 {env_file}")
            print("请编辑 .env 文件配置相关参数")
        else:
            print("警告: 未找到环境配置文件")
    
    return env_file.exists()


def start_server(reload=False, host="0.0.0.0", port=8000):
    """启动RAG服务器"""
    project_root = get_project_root()
    os.chdir(project_root)
    
    # 设置Python路径
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    
    cmd = [
        "uv", "run", "uvicorn", "src.main:app",
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    print(f"正在启动RAG服务器...")
    print(f"服务地址: http://{host}:{port}")
    print(f"API文档: http://{host}:{port}/docs")
    print("按 Ctrl+C 停止服务")
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n服务已停止")
    except subprocess.CalledProcessError as e:
        print(f"服务启动失败: {e}")
        return False
    
    return True


def run_tests():
    """运行测试"""
    print("正在运行测试...")
    try:
        subprocess.run(["uv", "run", "pytest", "-v"], check=True)
        print("测试完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"测试失败: {e}")
        return False


def check_services():
    """检查依赖服务状态"""
    print("正在检查依赖服务...")
    
    services = {
        "PostgreSQL": ("localhost", 15432),
        "Redis": ("localhost", 16379),
        "Qdrant": ("localhost", 6333)
    }
    
    import socket
    
    for service, (host, port) in services.items():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                print(f"✓ {service} ({host}:{port}) - 运行中")
            else:
                print(f"✗ {service} ({host}:{port}) - 未运行")
        except Exception as e:
            print(f"✗ {service} ({host}:{port}) - 检查失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="RAG系统启动脚本")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 服务器启动命令
    server_parser = subparsers.add_parser("server", help="启动RAG服务器")
    server_parser.add_argument("--reload", action="store_true", help="启用热重载（开发模式）")
    server_parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")
    server_parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    
    # 测试命令
    subparsers.add_parser("test", help="运行测试")
    
    # 检查命令
    subparsers.add_parser("check", help="检查系统状态")
    
    # 设置命令
    subparsers.add_parser("setup", help="设置开发环境")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    project_root = get_project_root()
    os.chdir(project_root)
    
    if args.command == "setup":
        print("正在设置开发环境...")
        
        if not check_uv_installed():
            if not install_uv():
                sys.exit(1)
        
        if not sync_dependencies():
            sys.exit(1)
        
        check_env_file()
        print("开发环境设置完成！")
        
    elif args.command == "server":
        # 检查基本环境
        if not check_uv_installed():
            print("错误: uv未安装，请先运行 'python scripts/run_with_uv.py setup'")
            sys.exit(1)
        
        check_env_file()
        start_server(reload=args.reload, host=args.host, port=args.port)
        
    elif args.command == "test":
        if not check_uv_installed():
            print("错误: uv未安装，请先运行 'python scripts/run_with_uv.py setup'")
            sys.exit(1)
        
        run_tests()
        
    elif args.command == "check":
        check_services()


if __name__ == "__main__":
    main()