#!/usr/bin/env python3
"""
RAG系统依赖服务连接测试脚本

这个脚本用于测试所有依赖服务的连接状态，包括：
- PostgreSQL 数据库
- Qdrant 向量数据库
- Redis 缓存
- MinIO 对象存储

使用方法：
    python test_connections.py
"""

import sys
import time
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_postgres() -> bool:
    """测试PostgreSQL连接"""
    try:
        import psycopg2
        from psycopg2 import sql
        
        # 从环境变量获取连接参数
        conn_params = {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB", "rag_db"),
            "user": os.getenv("POSTGRES_USER", "rag_user"),
            "password": os.getenv("POSTGRES_PASSWORD", "rag_password")
        }
        
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        
        # 测试基本连接
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        
        # 测试表是否存在
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        print("✅ PostgreSQL连接成功")
        print(f"   版本: {version.split(',')[0]}")
        print(f"   数据库: {conn_params['database']}")
        print(f"   表数量: {len(tables)}")
        if tables:
            table_names = [table[0] for table in tables]
            print(f"   表列表: {', '.join(table_names)}")
        
        return True
        
    except ImportError:
        print("❌ PostgreSQL连接失败: 缺少 psycopg2 依赖")
        print("   请运行: uv add psycopg2-binary")
        return False
    except Exception as e:
        print(f"❌ PostgreSQL连接失败: {e}")
        return False

def test_qdrant() -> bool:
    """测试Qdrant连接"""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
        
        # 从环境变量获取连接参数
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        
        client = QdrantClient(host=host, port=port)
        
        # 测试连接和获取信息
        cluster_info = client.get_cluster_info()
        collections = client.get_collections()
        
        print("✅ Qdrant连接成功")
        print(f"   地址: {host}:{port}")
        print(f"   状态: {cluster_info.status}")
        print(f"   集合数量: {len(collections.collections)}")
        
        if collections.collections:
            collection_names = [col.name for col in collections.collections]
            print(f"   集合列表: {', '.join(collection_names)}")
        
        return True
        
    except ImportError:
        print("❌ Qdrant连接失败: 缺少 qdrant-client 依赖")
        print("   请运行: uv add qdrant-client")
        return False
    except Exception as e:
        print(f"❌ Qdrant连接失败: {e}")
        return False

def test_redis() -> bool:
    """测试Redis连接"""
    try:
        import redis
        
        # 从环境变量获取连接参数
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", 6379))
        password = os.getenv("REDIS_PASSWORD", "redis_password")
        db = int(os.getenv("REDIS_DB", 0))
        
        r = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=True
        )
        
        # 测试连接
        r.ping()
        
        # 测试基本操作
        test_key = "test_connection_" + str(int(time.time()))
        r.set(test_key, "success", ex=60)  # 60秒过期
        result = r.get(test_key)
        r.delete(test_key)
        
        # 获取Redis信息
        info = r.info()
        
        print("✅ Redis连接成功")
        print(f"   地址: {host}:{port}")
        print(f"   数据库: {db}")
        print(f"   版本: {info.get('redis_version', 'Unknown')}")
        print(f"   测试结果: {result}")
        print(f"   内存使用: {info.get('used_memory_human', 'Unknown')}")
        
        return True
        
    except ImportError:
        print("❌ Redis连接失败: 缺少 redis 依赖")
        print("   请运行: uv add redis")
        return False
    except Exception as e:
        print(f"❌ Redis连接失败: {e}")
        return False

def test_minio() -> bool:
    """测试MinIO连接"""
    try:
        from minio import Minio
        from minio.error import S3Error
        
        # 从环境变量获取连接参数
        endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY", "minio_admin")
        secret_key = os.getenv("MINIO_SECRET_KEY", "minio_password")
        secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        
        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        
        # 测试连接
        buckets = client.list_buckets()
        
        print("✅ MinIO连接成功")
        print(f"   地址: {endpoint}")
        print(f"   安全连接: {secure}")
        print(f"   存储桶数量: {len(buckets)}")
        
        if buckets:
            bucket_names = [bucket.name for bucket in buckets]
            print(f"   存储桶列表: {', '.join(bucket_names)}")
        
        # 测试创建存储桶（如果不存在）
        bucket_name = os.getenv("MINIO_BUCKET_NAME", "rag-documents")
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"   已创建存储桶: {bucket_name}")
        else:
            print(f"   存储桶已存在: {bucket_name}")
        
        return True
        
    except ImportError:
        print("❌ MinIO连接失败: 缺少 minio 依赖")
        print("   请运行: uv add minio")
        return False
    except Exception as e:
        print(f"❌ MinIO连接失败: {e}")
        return False

def check_docker_services() -> Dict[str, bool]:
    """检查Docker服务状态"""
    try:
        import subprocess
        
        # 检查docker-compose服务状态
        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            import json
            services = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    try:
                        service = json.loads(line)
                        services.append(service)
                    except json.JSONDecodeError:
                        continue
            
            print("🐳 Docker服务状态:")
            service_status = {}
            for service in services:
                name = service.get('Service', 'Unknown')
                state = service.get('State', 'Unknown')
                status = service.get('Status', 'Unknown')
                is_healthy = 'healthy' in status.lower() or state.lower() == 'running'
                
                status_icon = "✅" if is_healthy else "❌"
                print(f"   {status_icon} {name}: {state} ({status})")
                service_status[name] = is_healthy
            
            return service_status
        else:
            print("⚠️  无法获取Docker服务状态")
            return {}
            
    except Exception as e:
        print(f"⚠️  检查Docker服务状态失败: {e}")
        return {}

def main():
    """主测试函数"""
    print("🚀 开始测试RAG系统依赖服务连接...\n")
    
    # 检查Docker服务状态
    print("📋 检查Docker服务状态...")
    docker_status = check_docker_services()
    print()
    
    # 等待服务启动
    if docker_status:
        print("⏳ 等待服务完全启动...")
        time.sleep(5)
        print()
    
    # 定义测试项目
    tests = [
        ("PostgreSQL", test_postgres),
        ("Qdrant", test_qdrant),
        ("Redis", test_redis),
        ("MinIO", test_minio)
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"📋 测试 {name}...")
        results[name] = test_func()
        print()
        time.sleep(1)
    
    # 汇总结果
    print("📊 测试结果汇总:")
    print("=" * 40)
    success_count = 0
    for name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{name:12}: {status}")
        if success:
            success_count += 1
    
    print(f"\n总计: {success_count}/{len(tests)} 个服务连接成功")
    
    if success_count == len(tests):
        print("🎉 所有服务连接测试通过！")
        print("\n💡 提示:")
        print("   - 可以开始开发RAG应用了")
        print("   - MinIO控制台: http://localhost:9001")
        print("   - Qdrant仪表板: http://localhost:6333/dashboard")
        return 0
    else:
        print("⚠️  部分服务连接失败，请检查:")
        print("   1. Docker服务是否正常运行")
        print("   2. 端口是否被占用")
        print("   3. 环境变量配置是否正确")
        print("   4. 防火墙设置")
        return 1

if __name__ == "__main__":
    sys.exit(main())