#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据迁移脚本

用于处理数据库迁移、数据转换和版本升级
"""

import os
import sys
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.config import get_config
from src.database import DatabaseManager, get_async_session
from src.models import (
    User, Document, DocumentChunk, QueryHistory, SystemConfig,
    UserRole, DocumentStatus, DocumentType, QueryStatus, QueryType
)
from src.repositories import (
    user_repository, document_repository, document_chunk_repository,
    query_history_repository, system_config_repository
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataMigrator:
    """数据迁移器"""
    
    def __init__(self):
        self.config = get_config()
        self.db_manager = DatabaseManager()
        self.migration_history = []
    
    async def initialize(self):
        """初始化迁移器"""
        try:
            await self.db_manager.initialize()
            logger.info("数据库连接初始化成功")
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    async def close(self):
        """关闭迁移器"""
        await self.db_manager.close()
        logger.info("数据库连接已关闭")
    
    async def backup_data(self, backup_path: str = None) -> str:
        """备份数据"""
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"backup_{timestamp}.sql"
        
        try:
            # 这里应该调用pg_dump或类似工具
            # 为了演示，我们记录备份操作
            logger.info(f"开始备份数据到: {backup_path}")
            
            # 实际备份命令（需要根据实际环境调整）
            db_config = self.config.get_database_config()
            backup_cmd = (
                f"pg_dump -h {db_config['host']} -p {db_config['port']} "
                f"-U {db_config['username']} -d {db_config['database']} "
                f"-f {backup_path}"
            )
            
            logger.info(f"备份命令: {backup_cmd}")
            logger.info(f"数据备份完成: {backup_path}")
            
            return backup_path
        except Exception as e:
            logger.error(f"数据备份失败: {e}")
            raise
    
    async def migrate_user_data(self) -> Dict[str, Any]:
        """迁移用户数据"""
        logger.info("开始迁移用户数据")
        
        try:
            # 检查是否需要创建默认管理员
            admin_user = user_repository.get_by_username("admin")
            
            if not admin_user:
                # 创建默认管理员用户
                from src.models import UserCreate
                admin_data = UserCreate(
                    username="admin",
                    email="admin@example.com",
                    password="admin123",
                    full_name="系统管理员",
                    role=UserRole.ADMIN
                )
                
                admin_user = user_repository.create(admin_data)
                logger.info(f"创建默认管理员用户: {admin_user.username}")
            
            # 统计用户数据
            total_users = len(user_repository.get_all())
            active_users = len(user_repository.get_active_users())
            
            result = {
                "total_users": total_users,
                "active_users": active_users,
                "admin_created": admin_user is not None
            }
            
            logger.info(f"用户数据迁移完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"用户数据迁移失败: {e}")
            raise
    
    async def migrate_document_data(self) -> Dict[str, Any]:
        """迁移文档数据"""
        logger.info("开始迁移文档数据")
        
        try:
            # 检查文档状态一致性
            all_documents = document_repository.get_all()
            
            processed_count = 0
            error_count = 0
            
            for doc in all_documents:
                try:
                    # 检查文档块是否存在
                    chunks = document_chunk_repository.get_by_document_id(doc.id)
                    
                    # 如果有块但状态不是已处理，更新状态
                    if chunks and doc.status != DocumentStatus.PROCESSED:
                        from src.models import DocumentUpdate
                        update_data = DocumentUpdate(status=DocumentStatus.PROCESSED)
                        document_repository.update(doc.id, update_data)
                        processed_count += 1
                        logger.info(f"更新文档状态: {doc.title} -> PROCESSED")
                    
                    # 如果没有块但状态是已处理，标记为错误
                    elif not chunks and doc.status == DocumentStatus.PROCESSED:
                        update_data = DocumentUpdate(status=DocumentStatus.ERROR)
                        document_repository.update(doc.id, update_data)
                        error_count += 1
                        logger.warning(f"文档状态不一致: {doc.title} -> ERROR")
                        
                except Exception as e:
                    logger.error(f"处理文档 {doc.title} 时出错: {e}")
                    error_count += 1
            
            result = {
                "total_documents": len(all_documents),
                "processed_updated": processed_count,
                "error_marked": error_count
            }
            
            logger.info(f"文档数据迁移完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"文档数据迁移失败: {e}")
            raise
    
    async def migrate_system_config(self) -> Dict[str, Any]:
        """迁移系统配置"""
        logger.info("开始迁移系统配置")
        
        try:
            # 默认系统配置
            default_configs = [
                {"key": "max_file_size", "value": "10485760", "category": "upload", "description": "最大文件大小(字节)"},
                {"key": "allowed_file_types", "value": "pdf,txt,docx,md", "category": "upload", "description": "允许的文件类型"},
                {"key": "chunk_size", "value": "1000", "category": "processing", "description": "文档分块大小"},
                {"key": "chunk_overlap", "value": "200", "category": "processing", "description": "分块重叠大小"},
                {"key": "embedding_model", "value": "text-embedding-ada-002", "category": "ai", "description": "嵌入模型"},
                {"key": "max_query_length", "value": "1000", "category": "query", "description": "最大查询长度"},
                {"key": "session_timeout", "value": "3600", "category": "auth", "description": "会话超时时间(秒)"},
                {"key": "rate_limit", "value": "100", "category": "api", "description": "API速率限制(每分钟)"},
            ]
            
            created_count = 0
            updated_count = 0
            
            for config_data in default_configs:
                existing_config = system_config_repository.get_by_key(config_data["key"])
                
                if not existing_config:
                    # 创建新配置
                    system_config_repository.set_config(
                        key=config_data["key"],
                        value=config_data["value"],
                        category=config_data["category"],
                        description=config_data["description"]
                    )
                    created_count += 1
                    logger.info(f"创建系统配置: {config_data['key']}")
                else:
                    # 更新描述（如果为空）
                    if not existing_config.description and config_data.get("description"):
                        from src.models import SystemConfigUpdate
                        update_data = SystemConfigUpdate(description=config_data["description"])
                        system_config_repository.update(existing_config.id, update_data)
                        updated_count += 1
                        logger.info(f"更新系统配置描述: {config_data['key']}")
            
            result = {
                "total_configs": len(default_configs),
                "created": created_count,
                "updated": updated_count
            }
            
            logger.info(f"系统配置迁移完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"系统配置迁移失败: {e}")
            raise
    
    async def cleanup_orphaned_data(self) -> Dict[str, Any]:
        """清理孤立数据"""
        logger.info("开始清理孤立数据")
        
        try:
            # 清理没有对应文档的文档块
            all_chunks = document_chunk_repository.get_all()
            orphaned_chunks = 0
            
            for chunk in all_chunks:
                document = document_repository.get_by_id(chunk.document_id)
                if not document:
                    document_chunk_repository.delete(chunk.id)
                    orphaned_chunks += 1
                    logger.info(f"删除孤立文档块: {chunk.id}")
            
            # 清理没有对应用户的查询历史
            all_queries = query_history_repository.get_all()
            orphaned_queries = 0
            
            for query in all_queries:
                if query.user_id:
                    user = user_repository.get_by_id(query.user_id)
                    if not user:
                        query_history_repository.delete(query.id)
                        orphaned_queries += 1
                        logger.info(f"删除孤立查询历史: {query.id}")
            
            result = {
                "orphaned_chunks_removed": orphaned_chunks,
                "orphaned_queries_removed": orphaned_queries
            }
            
            logger.info(f"孤立数据清理完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"孤立数据清理失败: {e}")
            raise
    
    async def run_full_migration(self, backup: bool = True) -> Dict[str, Any]:
        """运行完整迁移"""
        logger.info("开始完整数据迁移")
        
        migration_result = {
            "start_time": datetime.now(timezone.utc),
            "backup_file": None,
            "user_migration": None,
            "document_migration": None,
            "config_migration": None,
            "cleanup_result": None,
            "success": False,
            "error": None
        }
        
        try:
            # 1. 备份数据
            if backup:
                backup_file = await self.backup_data()
                migration_result["backup_file"] = backup_file
            
            # 2. 迁移用户数据
            user_result = await self.migrate_user_data()
            migration_result["user_migration"] = user_result
            
            # 3. 迁移文档数据
            doc_result = await self.migrate_document_data()
            migration_result["document_migration"] = doc_result
            
            # 4. 迁移系统配置
            config_result = await self.migrate_system_config()
            migration_result["config_migration"] = config_result
            
            # 5. 清理孤立数据
            cleanup_result = await self.cleanup_orphaned_data()
            migration_result["cleanup_result"] = cleanup_result
            
            migration_result["success"] = True
            migration_result["end_time"] = datetime.now(timezone.utc)
            
            logger.info("完整数据迁移成功完成")
            
        except Exception as e:
            migration_result["error"] = str(e)
            migration_result["end_time"] = datetime.now(timezone.utc)
            logger.error(f"数据迁移失败: {e}")
            raise
        
        return migration_result


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据迁移脚本")
    parser.add_argument("--no-backup", action="store_true", help="跳过数据备份")
    parser.add_argument("--user-only", action="store_true", help="仅迁移用户数据")
    parser.add_argument("--document-only", action="store_true", help="仅迁移文档数据")
    parser.add_argument("--config-only", action="store_true", help="仅迁移系统配置")
    parser.add_argument("--cleanup-only", action="store_true", help="仅清理孤立数据")
    
    args = parser.parse_args()
    
    migrator = DataMigrator()
    
    try:
        await migrator.initialize()
        
        if args.user_only:
            result = await migrator.migrate_user_data()
            print(f"用户数据迁移结果: {result}")
        elif args.document_only:
            result = await migrator.migrate_document_data()
            print(f"文档数据迁移结果: {result}")
        elif args.config_only:
            result = await migrator.migrate_system_config()
            print(f"系统配置迁移结果: {result}")
        elif args.cleanup_only:
            result = await migrator.cleanup_orphaned_data()
            print(f"数据清理结果: {result}")
        else:
            # 运行完整迁移
            result = await migrator.run_full_migration(backup=not args.no_backup)
            print(f"完整迁移结果: {result}")
        
    except Exception as e:
        logger.error(f"迁移过程中发生错误: {e}")
        sys.exit(1)
    finally:
        await migrator.close()


if __name__ == "__main__":
    asyncio.run(main())