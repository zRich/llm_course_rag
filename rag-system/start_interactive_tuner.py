#!/usr/bin/env python3
"""启动交互式Chunk参数调优工具"""

import subprocess
import sys
from pathlib import Path

def main():
    """启动Streamlit应用"""
    # 获取交互式调优工具的路径
    tuner_path = Path(__file__).parent / "experiments" / "chunk_optimization" / "interactive_tuner.py"
    
    if not tuner_path.exists():
        print(f"❌ 找不到交互式调优工具: {tuner_path}")
        sys.exit(1)
    
    print("🚀 正在启动交互式Chunk参数调优工具...")
    print(f"📁 工具路径: {tuner_path}")
    print("\n🌐 浏览器将自动打开，如果没有请手动访问显示的URL")
    print("⏹️  按 Ctrl+C 停止服务\n")
    
    try:
        # 启动Streamlit应用
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(tuner_path),
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ], check=True)
    
    except KeyboardInterrupt:
        print("\n👋 交互式调优工具已停止")
    
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
        print("\n💡 请确保已安装streamlit: pip install streamlit")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()