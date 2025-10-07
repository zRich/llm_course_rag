#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库优化脚本

用于数据库性能优化、索引管理和维护任务
"""

import os
import sys
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.config import get_config
from src.database import DatabaseManager, get_async_session
from sqlalchemy import text, inspect
from sqlalchemy.engine import Engine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DatabaseOptimizer:
    """数据库优化器"""
    
    def __init__(self):
        self.config = get_config()
        self.db_manager = DatabaseManager()
        self.optimization_history = []
    
    async def initialize(self):
        """初始化优化器"""
        try:
            await self.db_manager.initialize()
            logger.info("数据库连接初始化成功")
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    async def close(self):
        """关闭优化器"""
        await self.db_manager.close()
        logger.info("数据库连接已关闭")
    
    async def analyze_table_stats(self) -> Dict[str, Any]:
        """分析表统计信息"""
        logger.info("开始分析表统计信息")
        
        try:
            async with get_async_session() as session:
                # 获取所有表的统计信息
                stats_query = text("""
                    SELECT 
                        schemaname,
                        tablename,
                        attname,
                        n_distinct,
                        correlation,
                        most_common_vals,
                        most_common_freqs
                    FROM pg_stats 
                    WHERE schemaname = 'public'
                    ORDER BY tablename, attname;
                """)
                
                result = await session.execute(stats_query)
                stats_data = result.fetchall()
                
                # 获取表大小信息
                size_query = text("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
                """)
                
                size_result = await session.execute(size_query)
                size_data = size_result.fetchall()
                
                # 获取索引信息
                index_query = text("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        indexdef,
                        pg_size_pretty(pg_relation_size(indexname)) as index_size
                    FROM pg_indexes 
                    WHERE schemaname = 'public'
                    ORDER BY tablename, indexname;
                """)
                
                index_result = await session.execute(index_query)
                index_data = index_result.fetchall()
                
                analysis_result = {
                    "table_stats": [dict(row._mapping) for row in stats_data],
                    "table_sizes": [dict(row._mapping) for row in size_data],
                    "indexes": [dict(row._mapping) for row in index_data],
                    "analysis_time": datetime.now(timezone.utc)
                }
                
                logger.info(f"表统计分析完成，共分析 {len(size_data)} 个表")
                return analysis_result
                
        except Exception as e:
            logger.error(f"表统计分析失败: {e}")
            raise
    
    async def create_performance_indexes(self) -> Dict[str, Any]:
        """创建性能优化索引"""
        logger.info("开始创建性能优化索引")
        
        try:
            async with get_async_session() as session:
                # 定义需要创建的索引
                indexes_to_create = [
                    # 用户表索引
                    {
                        "name": "idx_users_username_active",
                        "table": "users",
                        "columns": "username, is_active",
                        "description": "用户名和活跃状态复合索引"
                    },
                    {
                        "name": "idx_users_email_active",
                        "table": "users",
                        "columns": "email, is_active",
                        "description": "邮箱和活跃状态复合索引"
                    },
                    {
                        "name": "idx_users_role",
                        "table": "users",
                        "columns": "role",
                        "description": "用户角色索引"
                    },
                    {
                        "name": "idx_users_created_at",
                        "table": "users",
                        "columns": "created_at DESC",
                        "description": "用户创建时间降序索引"
                    },
                    
                    # 文档表索引
                    {
                        "name": "idx_documents_owner_status",
                        "table": "documents",
                        "columns": "owner_id, status",
                        "description": "文档所有者和状态复合索引"
                    },
                    {
                        "name": "idx_documents_title_gin",
                        "table": "documents",
                        "columns": "title gin_trgm_ops",
                        "type": "GIN",
                        "description": "文档标题全文搜索索引"
                    },
                    {
                        "name": "idx_documents_file_hash",
                        "table": "documents",
                        "columns": "file_hash",
                        "description": "文档哈希值索引"
                    },
                    {
                        "name": "idx_documents_type_status",
                        "table": "documents",
                        "columns": "document_type, status",
                        "description": "文档类型和状态复合索引"
                    },
                    {
                        "name": "idx_documents_created_at",
                        "table": "documents",
                        "columns": "created_at DESC",
                        "description": "文档创建时间降序索引"
                    },
                    
                    # 文档块表索引
                    {
                        "name": "idx_document_chunks_document_id",
                        "table": "document_chunks",
                        "columns": "document_id, chunk_index",
                        "description": "文档ID和块索引复合索引"
                    },
                    {
                        "name": "idx_document_chunks_vector_id",
                        "table": "document_chunks",
                        "columns": "vector_id",
                        "description": "向量ID索引"
                    },
                    {
                        "name": "idx_document_chunks_content_gin",
                        "table": "document_chunks",
                        "columns": "content gin_trgm_ops",
                        "type": "GIN",
                        "description": "文档块内容全文搜索索引"
                    },
                    
                    # 查询历史表索引
                    {
                        "name": "idx_query_history_user_id",
                        "table": "query_history",
                        "columns": "user_id, created_at DESC",
                        "description": "用户ID和创建时间复合索引"
                    },
                    {
                        "name": "idx_query_history_session_id",
                        "table": "query_history",
                        "columns": "session_id, created_at DESC",
                        "description": "会话ID和创建时间复合索引"
                    },
                    {
                        "name": "idx_query_history_status",
                        "table": "query_history",
                        "columns": "status",
                        "description": "查询状态索引"
                    },
                    {
                        "name": "idx_query_history_query_gin",
                        "table": "query_history",
                        "columns": "query gin_trgm_ops",
                        "type": "GIN",
                        "description": "查询内容全文搜索索引"
                    },
                    
                    # 系统配置表索引
                    {
                        "name": "idx_system_config_key",
                        "table": "system_config",
                        "columns": "key",
                        "unique": True,
                        "description": "配置键唯一索引"
                    },
                    {
                        "name": "idx_system_config_category",
                        "table": "system_config",
                        "columns": "category",
                        "description": "配置分类索引"
                    }
                ]
                
                created_indexes = []
                skipped_indexes = []
                
                for index_def in indexes_to_create:
                    try:
                        # 检查索引是否已存在
                        check_query = text("""
                            SELECT indexname 
                            FROM pg_indexes 
                            WHERE schemaname = 'public' AND indexname = :index_name
                        """)
                        
                        check_result = await session.execute(
                            check_query, 
                            {"index_name": index_def["name"]}
                        )
                        
                        if check_result.fetchone():
                            skipped_indexes.append(index_def["name"])
                            logger.info(f"索引已存在，跳过: {index_def['name']}")
                            continue
                        
                        # 构建创建索引的SQL
                        index_type = index_def.get("type", "BTREE")
                        unique_clause = "UNIQUE " if index_def.get("unique", False) else ""
                        
                        if index_type == "GIN":
                            # 先确保pg_trgm扩展已安装
                            await session.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
                        
                        create_sql = f"""
                            CREATE {unique_clause}INDEX {index_def['name']} 
                            ON {index_def['table']} 
                            USING {index_type} ({index_def['columns']})
                        """
                        
                        await session.execute(text(create_sql))
                        await session.commit()
                        
                        created_indexes.append({
                            "name": index_def["name"],
                            "table": index_def["table"],
                            "description": index_def["description"]
                        })
                        
                        logger.info(f"创建索引成功: {index_def['name']}")
                        
                    except Exception as e:
                        logger.error(f"创建索引失败 {index_def['name']}: {e}")
                        await session.rollback()
                        continue
                
                result = {
                    "created_indexes": created_indexes,
                    "skipped_indexes": skipped_indexes,
                    "total_attempted": len(indexes_to_create),
                    "created_count": len(created_indexes),
                    "skipped_count": len(skipped_indexes)
                }
                
                logger.info(f"索引创建完成: 创建 {len(created_indexes)} 个，跳过 {len(skipped_indexes)} 个")
                return result
                
        except Exception as e:
            logger.error(f"创建性能索引失败: {e}")
            raise
    
    async def analyze_slow_queries(self) -> Dict[str, Any]:
        """分析慢查询"""
        logger.info("开始分析慢查询")
        
        try:
            async with get_async_session() as session:
                # 启用pg_stat_statements扩展（如果可用）
                try:
                    await session.execute(text("CREATE EXTENSION IF NOT EXISTS pg_stat_statements"))
                    await session.commit()
                    
                    # 获取慢查询统计
                    slow_query_sql = text("""
                        SELECT 
                            query,
                            calls,
                            total_time,
                            mean_time,
                            rows,
                            100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
                        FROM pg_stat_statements 
                        WHERE mean_time > 100  -- 平均执行时间超过100ms的查询
                        ORDER BY mean_time DESC 
                        LIMIT 20;
                    """)
                    
                    result = await session.execute(slow_query_sql)
                    slow_queries = [dict(row._mapping) for row in result.fetchall()]
                    
                except Exception as e:
                    logger.warning(f"pg_stat_statements扩展不可用: {e}")
                    slow_queries = []
                
                # 获取当前活跃查询
                active_query_sql = text("""
                    SELECT 
                        pid,
                        now() - pg_stat_activity.query_start AS duration,
                        query,
                        state
                    FROM pg_stat_activity 
                    WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes'
                    AND state = 'active';
                """)
                
                active_result = await session.execute(active_query_sql)
                long_running_queries = [dict(row._mapping) for row in active_result.fetchall()]
                
                analysis_result = {
                    "slow_queries": slow_queries,
                    "long_running_queries": long_running_queries,
                    "analysis_time": datetime.now(timezone.utc)
                }
                
                logger.info(f"慢查询分析完成: 发现 {len(slow_queries)} 个慢查询，{len(long_running_queries)} 个长时间运行查询")
                return analysis_result
                
        except Exception as e:
            logger.error(f"慢查询分析失败: {e}")
            raise
    
    async def vacuum_and_analyze(self) -> Dict[str, Any]:
        """执行VACUUM和ANALYZE操作"""
        logger.info("开始执行VACUUM和ANALYZE操作")
        
        try:
            # 获取所有用户表
            tables = ['users', 'documents', 'document_chunks', 'query_history', 'system_config']
            
            vacuum_results = []
            
            # 对每个表执行VACUUM ANALYZE
            for table in tables:
                try:
                    async with get_async_session() as session:
                        # 注意：VACUUM不能在事务中执行，需要autocommit
                        await session.execute(text(f"VACUUM ANALYZE {table}"))
                        await session.commit()
                        
                        vacuum_results.append({
                            "table": table,
                            "status": "success",
                            "timestamp": datetime.now(timezone.utc)
                        })
                        
                        logger.info(f"VACUUM ANALYZE完成: {table}")
                        
                except Exception as e:
                    vacuum_results.append({
                        "table": table,
                        "status": "failed",
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc)
                    })
                    logger.error(f"VACUUM ANALYZE失败 {table}: {e}")
            
            result = {
                "vacuum_results": vacuum_results,
                "total_tables": len(tables),
                "successful_tables": len([r for r in vacuum_results if r["status"] == "success"]),
                "failed_tables": len([r for r in vacuum_results if r["status"] == "failed"])
            }
            
            logger.info(f"VACUUM操作完成: 成功 {result['successful_tables']} 个表，失败 {result['failed_tables']} 个表")
            return result
            
        except Exception as e:
            logger.error(f"VACUUM操作失败: {e}")
            raise
    
    async def optimize_database_settings(self) -> Dict[str, Any]:
        """优化数据库设置"""
        logger.info("开始优化数据库设置")
        
        try:
            async with get_async_session() as session:
                # 获取当前数据库设置
                settings_query = text("""
                    SELECT name, setting, unit, context, short_desc
                    FROM pg_settings 
                    WHERE name IN (
                        'shared_buffers',
                        'effective_cache_size',
                        'maintenance_work_mem',
                        'checkpoint_completion_target',
                        'wal_buffers',
                        'default_statistics_target',
                        'random_page_cost',
                        'effective_io_concurrency',
                        'work_mem',
                        'max_connections'
                    )
                    ORDER BY name;
                """)
                
                result = await session.execute(settings_query)
                current_settings = [dict(row._mapping) for row in result.fetchall()]
                
                # 推荐的优化设置（基于常见的生产环境配置）
                recommended_settings = {
                    'shared_buffers': '256MB',  # 通常设置为系统内存的25%
                    'effective_cache_size': '1GB',  # 通常设置为系统内存的75%
                    'maintenance_work_mem': '64MB',
                    'checkpoint_completion_target': '0.9',
                    'wal_buffers': '16MB',
                    'default_statistics_target': '100',
                    'random_page_cost': '1.1',  # SSD存储
                    'effective_io_concurrency': '200',  # SSD存储
                    'work_mem': '4MB'
                }
                
                optimization_suggestions = []
                
                for setting in current_settings:
                    name = setting['name']
                    current_value = setting['setting']
                    recommended_value = recommended_settings.get(name)
                    
                    if recommended_value and current_value != recommended_value:
                        optimization_suggestions.append({
                            'setting': name,
                            'current_value': current_value,
                            'recommended_value': recommended_value,
                            'description': setting['short_desc'],
                            'context': setting['context']
                        })
                
                result = {
                    'current_settings': current_settings,
                    'optimization_suggestions': optimization_suggestions,
                    'analysis_time': datetime.now(timezone.utc)
                }
                
                logger.info(f"数据库设置分析完成: 发现 {len(optimization_suggestions)} 个优化建议")
                return result
                
        except Exception as e:
            logger.error(f"数据库设置优化失败: {e}")
            raise
    
    async def run_full_optimization(self) -> Dict[str, Any]:
        """运行完整优化"""
        logger.info("开始完整数据库优化")
        
        optimization_result = {
            "start_time": datetime.now(timezone.utc),
            "table_analysis": None,
            "index_creation": None,
            "slow_query_analysis": None,
            "vacuum_results": None,
            "settings_analysis": None,
            "success": False,
            "error": None
        }
        
        try:
            # 1. 分析表统计信息
            table_analysis = await self.analyze_table_stats()
            optimization_result["table_analysis"] = table_analysis
            
            # 2. 创建性能索引
            index_creation = await self.create_performance_indexes()
            optimization_result["index_creation"] = index_creation
            
            # 3. 分析慢查询
            slow_query_analysis = await self.analyze_slow_queries()
            optimization_result["slow_query_analysis"] = slow_query_analysis
            
            # 4. 执行VACUUM和ANALYZE
            vacuum_results = await self.vacuum_and_analyze()
            optimization_result["vacuum_results"] = vacuum_results
            
            # 5. 分析数据库设置
            settings_analysis = await self.optimize_database_settings()
            optimization_result["settings_analysis"] = settings_analysis
            
            optimization_result["success"] = True
            optimization_result["end_time"] = datetime.now(timezone.utc)
            
            logger.info("完整数据库优化成功完成")
            
        except Exception as e:
            optimization_result["error"] = str(e)
            optimization_result["end_time"] = datetime.now(timezone.utc)
            logger.error(f"数据库优化失败: {e}")
            raise
        
        return optimization_result


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据库优化脚本")
    parser.add_argument("--analyze-only", action="store_true", help="仅分析表统计信息")
    parser.add_argument("--indexes-only", action="store_true", help="仅创建性能索引")
    parser.add_argument("--slow-queries-only", action="store_true", help="仅分析慢查询")
    parser.add_argument("--vacuum-only", action="store_true", help="仅执行VACUUM操作")
    parser.add_argument("--settings-only", action="store_true", help="仅分析数据库设置")
    
    args = parser.parse_args()
    
    optimizer = DatabaseOptimizer()
    
    try:
        await optimizer.initialize()
        
        if args.analyze_only:
            result = await optimizer.analyze_table_stats()
            print(f"表分析结果: {result}")
        elif args.indexes_only:
            result = await optimizer.create_performance_indexes()
            print(f"索引创建结果: {result}")
        elif args.slow_queries_only:
            result = await optimizer.analyze_slow_queries()
            print(f"慢查询分析结果: {result}")
        elif args.vacuum_only:
            result = await optimizer.vacuum_and_analyze()
            print(f"VACUUM结果: {result}")
        elif args.settings_only:
            result = await optimizer.optimize_database_settings()
            print(f"设置分析结果: {result}")
        else:
            # 运行完整优化
            result = await optimizer.run_full_optimization()
            print(f"完整优化结果: {result}")
        
    except Exception as e:
        logger.error(f"优化过程中发生错误: {e}")
        sys.exit(1)
    finally:
        await optimizer.close()


if __name__ == "__main__":
    asyncio.run(main())