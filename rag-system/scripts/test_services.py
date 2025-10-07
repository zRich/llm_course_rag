#!/usr/bin/env python3
"""
RAG系统服务连接测试脚本
用于测试FastAPI、PostgreSQL、Redis、Qdrant、MinIO等服务的连接状态
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import httpx
import psycopg2
import redis
from qdrant_client import QdrantClient
from minio import Minio
from src.config import settings

class ServiceTester:
    """服务测试类"""
    
    def __init__(self):
        self.results = {}
    
    async def test_fastapi(self):
        """测试FastAPI服务"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://{settings.host}:{settings.port}/health")
                if response.status_code == 200:
                    self.results['FastAPI'] = {'status': 'OK', 'details': 'Health check passed'}
                else:
                    self.results['FastAPI'] = {'status': 'ERROR', 'details': f'HTTP {response.status_code}'}
        except Exception as e:
            self.results['FastAPI'] = {'status': 'ERROR', 'details': str(e)}
    
    def test_postgresql(self):
        """测试PostgreSQL连接"""
        try:
            # 解析数据库URL
            db_url = settings.database_url
            conn = psycopg2.connect(db_url)
            cursor = conn.cursor()
            
            # 测试查询
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            
            # 测试表是否存在
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public';
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            cursor.close()
            conn.close()
            
            self.results['PostgreSQL'] = {
                'status': 'OK', 
                'details': f'Connected. Tables: {len(tables)}',
                'version': version[:50] + '...' if len(version) > 50 else version
            }
        except Exception as e:
            self.results['PostgreSQL'] = {'status': 'ERROR', 'details': str(e)}
    
    def test_redis(self):
        """测试Redis连接"""
        try:
            r = redis.from_url(settings.redis_url)
            
            # 测试连接
            r.ping()
            
            # 测试读写
            test_key = "test:connection"
            r.set(test_key, "test_value", ex=60)
            value = r.get(test_key)
            r.delete(test_key)
            
            # 获取Redis信息
            info = r.info()
            
            self.results['Redis'] = {
                'status': 'OK',
                'details': f'Connected. Version: {info.get("redis_version", "unknown")}',
                'memory_usage': info.get('used_memory_human', 'unknown')
            }
        except Exception as e:
            self.results['Redis'] = {'status': 'ERROR', 'details': str(e)}
    
    def test_qdrant(self):
        """测试Qdrant连接"""
        try:
            client = QdrantClient(url=settings.qdrant_url)
            
            # 获取集合列表
            collections = client.get_collections()
            
            # 测试创建临时集合
            test_collection = "test_connection"
            try:
                client.delete_collection(test_collection)
            except:
                pass
            
            from qdrant_client.models import Distance, VectorParams
            client.create_collection(
                collection_name=test_collection,
                vectors_config=VectorParams(size=4, distance=Distance.COSINE)
            )
            
            # 删除测试集合
            client.delete_collection(test_collection)
            
            self.results['Qdrant'] = {
                'status': 'OK',
                'details': f'Connected. Collections: {len(collections.collections)}'
            }
        except Exception as e:
            self.results['Qdrant'] = {'status': 'ERROR', 'details': str(e)}
    
    def test_minio(self):
        """测试MinIO连接"""
        try:
            client = Minio(
                settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=settings.minio_secure
            )
            
            # 测试连接
            buckets = client.list_buckets()
            
            # 确保存储桶存在
            bucket_name = settings.minio_bucket_name
            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
                bucket_created = True
            else:
                bucket_created = False
            
            # 测试上传和下载
            import io
            test_data = b"test data for connection"
            test_object = "test/connection.txt"
            
            client.put_object(
                bucket_name,
                test_object,
                io.BytesIO(test_data),
                len(test_data)
            )
            
            # 下载测试
            response = client.get_object(bucket_name, test_object)
            downloaded_data = response.read()
            response.close()
            response.release_conn()
            
            # 删除测试对象
            client.remove_object(bucket_name, test_object)
            
            self.results['MinIO'] = {
                'status': 'OK',
                'details': f'Connected. Buckets: {len(buckets)}',
                'bucket_created': bucket_created,
                'test_upload': len(downloaded_data) == len(test_data)
            }
        except Exception as e:
            self.results['MinIO'] = {'status': 'ERROR', 'details': str(e)}
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("=== RAG系统服务连接测试 ===")
        print(f"测试环境: {settings.app_name} v{settings.app_version}")
        print()
        
        # 测试FastAPI
        print("1. 测试FastAPI服务...")
        await self.test_fastapi()
        
        # 测试PostgreSQL
        print("2. 测试PostgreSQL连接...")
        self.test_postgresql()
        
        # 测试Redis
        print("3. 测试Redis连接...")
        self.test_redis()
        
        # 测试Qdrant
        print("4. 测试Qdrant连接...")
        self.test_qdrant()
        
        # 测试MinIO
        print("5. 测试MinIO连接...")
        self.test_minio()
        
        # 输出结果
        print("\n=== 测试结果 ===")
        all_ok = True
        for service, result in self.results.items():
            status_icon = "✅" if result['status'] == 'OK' else "❌"
            print(f"{status_icon} {service}: {result['status']}")
            print(f"   详情: {result['details']}")
            if 'version' in result:
                print(f"   版本: {result['version']}")
            if 'memory_usage' in result:
                print(f"   内存使用: {result['memory_usage']}")
            if 'bucket_created' in result:
                print(f"   存储桶创建: {result['bucket_created']}")
            if 'test_upload' in result:
                print(f"   上传测试: {result['test_upload']}")
            print()
            
            if result['status'] != 'OK':
                all_ok = False
        
        if all_ok:
            print("🎉 所有服务连接正常！")
            return 0
        else:
            print("⚠️  部分服务连接失败，请检查配置和服务状态")
            return 1

async def main():
    """主函数"""
    tester = ServiceTester()
    exit_code = await tester.run_all_tests()
    sys.exit(exit_code)

if __name__ == "__main__":
    asyncio.run(main())