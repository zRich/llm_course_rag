#!/usr/bin/env python3
"""
测试配置加载脚本
验证.env文件是否正确加载以及Settings类配置
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_env_file_loading():
    """测试.env文件加载"""
    print("=== 测试.env文件加载 ===")
    
    # 检查.env文件是否存在
    env_file = project_root / ".env"
    if env_file.exists():
        print(f"✓ .env文件存在: {env_file}")
        
        # 读取.env文件内容
        with open(env_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"✓ .env文件内容长度: {len(content)} 字符")
            
            # 检查关键配置项
            key_configs = [
                "VOLCENGINE_API_KEY",
                "VOLCENGINE_BASE_URL", 
                "VOLCENGINE_MODEL",
                "DATABASE_URL",
                "QDRANT_URL"
            ]
            
            for key in key_configs:
                if key in content:
                    print(f"✓ 找到配置项: {key}")
                else:
                    print(f"✗ 缺少配置项: {key}")
    else:
        print(f"✗ .env文件不存在: {env_file}")

def test_environment_variables():
    """测试环境变量"""
    print("\n=== 测试环境变量 ===")
    
    key_vars = [
        "VOLCENGINE_API_KEY",
        "VOLCENGINE_BASE_URL",
        "VOLCENGINE_MODEL",
        "DATABASE_URL",
        "QDRANT_URL"
    ]
    
    for var in key_vars:
        value = os.environ.get(var)
        if value:
            # 对于API密钥，只显示前几位和后几位
            if "API_KEY" in var or "PASSWORD" in var:
                masked_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
                print(f"✓ {var} = {masked_value}")
            else:
                print(f"✓ {var} = {value}")
        else:
            print(f"✗ {var} = None")

def test_settings_loading():
    """测试Settings类加载"""
    print("\n=== 测试Settings类加载 ===")
    
    try:
        from src.config.settings import settings
        print("✓ Settings类导入成功")
        
        # 检查关键配置项
        config_items = [
            ("volcengine_api_key", "火山引擎API密钥"),
            ("volcengine_base_url", "火山引擎基础URL"),
            ("volcengine_model", "火山引擎模型"),
            ("database_url", "数据库URL"),
            ("qdrant_url", "Qdrant URL")
        ]
        
        for attr, desc in config_items:
            if hasattr(settings, attr):
                value = getattr(settings, attr)
                if value:
                    # 对于敏感信息进行脱敏
                    if "api_key" in attr or "password" in attr:
                        masked_value = f"{str(value)[:8]}...{str(value)[-4:]}" if len(str(value)) > 12 else "***"
                        print(f"✓ {desc} ({attr}) = {masked_value}")
                    else:
                        print(f"✓ {desc} ({attr}) = {value}")
                else:
                    print(f"✗ {desc} ({attr}) = None 或空值")
            else:
                print(f"✗ Settings类缺少属性: {attr}")
                
    except Exception as e:
        print(f"✗ Settings类加载失败: {e}")
        import traceback
        traceback.print_exc()

def test_pydantic_settings():
    """测试Pydantic Settings配置"""
    print("\n=== 测试Pydantic Settings配置 ===")
    
    try:
        from src.config.settings import Settings
        
        # 创建新的Settings实例来测试
        test_settings = Settings()
        print("✓ 创建Settings实例成功")
        
        # 检查字段定义
        fields = test_settings.__fields__ if hasattr(test_settings, '__fields__') else test_settings.model_fields
        print(f"✓ Settings类定义了 {len(fields)} 个字段")
        
        # 检查关键字段
        key_fields = [
            "volcengine_api_key",
            "volcengine_base_url", 
            "volcengine_model"
        ]
        
        for field in key_fields:
            if field in fields:
                field_info = fields[field]
                print(f"✓ 字段 {field} 已定义")
                # 检查字段默认值
                if hasattr(field_info, 'default'):
                    print(f"  - 默认值: {field_info.default}")
            else:
                print(f"✗ 字段 {field} 未定义")
                
    except Exception as e:
        print(f"✗ Pydantic Settings测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    print("开始配置加载测试...\n")
    
    # 设置工作目录
    os.chdir(project_root)
    
    test_env_file_loading()
    test_environment_variables()
    test_settings_loading()
    test_pydantic_settings()
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    main()