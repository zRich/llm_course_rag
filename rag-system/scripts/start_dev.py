#!/usr/bin/env python3
"""
开发环境启动脚本
用于启动RAG系统的开发服务器
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    import uvicorn
    from src.config import settings, validate_config
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所有依赖: pip install fastapi uvicorn pydantic-settings")
    sys.exit(1)

def main():
    """主函数"""
    print("=== RAG系统开发服务器启动脚本 ===")
    print(f"项目根目录: {project_root}")
    print(f"Python路径: {sys.path[:3]}...")
    
    # 验证配置
    print("\n=== 配置验证 ===")
    if not validate_config():
        print("配置验证失败，请检查配置文件")
        sys.exit(1)
    
    # 显示启动信息
    print(f"\n=== 启动信息 ===")
    print(f"应用名称: {settings.app_name}")
    print(f"版本: {settings.app_version}")
    print(f"主机: {settings.host}")
    print(f"端口: {settings.port}")
    print(f"调试模式: {settings.debug}")
    print(f"热重载: {settings.reload}")
    
    # 检查端口是否被占用
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((settings.host, settings.port))
    sock.close()
    
    if result == 0:
        print(f"\n警告: 端口 {settings.port} 已被占用")
        response = input("是否继续启动？(y/N): ")
        if response.lower() != 'y':
            print("启动已取消")
            sys.exit(0)
    
    print(f"\n=== 启动服务器 ===")
    print(f"访问地址: http://{settings.host}:{settings.port}")
    print(f"API文档: http://{settings.host}:{settings.port}/docs")
    print(f"交互式文档: http://{settings.host}:{settings.port}/redoc")
    print("\n按 Ctrl+C 停止服务器")
    print("=" * 50)
    
    try:
        # 启动uvicorn服务器
        uvicorn.run(
            "src.main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.reload,
            log_level=settings.log_level.lower(),
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n\n=== 服务器已停止 ===")
        print("感谢使用RAG系统！")
    except Exception as e:
        print(f"\n启动失败: {e}")
        print("请检查配置和依赖是否正确安装")
        sys.exit(1)

if __name__ == "__main__":
    main()