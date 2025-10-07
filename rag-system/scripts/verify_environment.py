#!/usr/bin/env python3
"""
环境验证脚本
验证所有必需的技术组件是否正确安装和配置
"""

import sys
import subprocess
import importlib
from typing import List, Tuple

def check_python_version() -> Tuple[bool, str]:
    """检查Python版本"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 12:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python版本过低: {version.major}.{version.minor}.{version.micro}"

def check_command(command: str) -> Tuple[bool, str]:
    """检查命令是否可用"""
    try:
        result = subprocess.run(
            [command, "--version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode == 0:
            return True, result.stdout.strip().split('\n')[0]
        return False, f"{command} 命令执行失败"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, f"{command} 命令未找到"

def check_python_package(package: str) -> Tuple[bool, str]:
    """检查Python包是否安装"""
    try:
        module = importlib.import_module(package)
        version = getattr(module, '__version__', 'unknown')
        return True, f"{package} {version}"
    except ImportError:
        return False, f"{package} 未安装"

def main():
    """主验证函数"""
    print("🔍 RAG系统环境验证")
    print("=" * 50)
    
    checks = [
        ("Python版本", lambda: check_python_version()),
        ("Docker", lambda: check_command("docker")),
        ("Docker Compose", lambda: check_command("docker-compose")),
        ("uv", lambda: check_command("uv")),
        ("Git", lambda: check_command("git")),
    ]
    
    # 检查Python包
    packages = [
        "fastapi", "uvicorn", "pydantic", "sqlalchemy", 
        "redis", "qdrant_client", "sentence_transformers"
    ]
    
    for package in packages:
        checks.append((f"Python包: {package}", lambda p=package: check_python_package(p)))
    
    # 执行检查
    passed = 0
    failed = 0
    
    for name, check_func in checks:
        try:
            success, message = check_func()
            status = "✅" if success else "❌"
            print(f"{status} {name}: {message}")
            if success:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {name}: 检查失败 - {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 验证结果: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("🎉 环境验证完全通过！可以开始RAG系统开发。")
        return 0
    else:
        print("⚠️  存在环境问题，请根据上述信息进行修复。")
        return 1

if __name__ == "__main__":
    sys.exit(main())