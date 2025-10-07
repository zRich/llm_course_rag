#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库测试文件

测试数据库连接、配置和初始化功能
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.database import (
    DatabaseConfig, db_config,
    DatabaseManager, db_manager,
    get_sync_session, get_async_session,
    init_database, close_database, check_database_health
)
from src.config import settings


class TestDatabaseConfig:
    """数据库配置测试"""
    
    def test_config_initialization(self):
        """测试配置初始化"""
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.user == "rag_user"
        assert config.database == "rag_db"
        assert config.pool_size == 5
        assert config.max_overflow == 10
    
    def test_sync_url_generation(self):
        """测试同步连接URL生成"""
        config = DatabaseConfig(
            host="testhost",
            port=5433,
            user="testuser",
            password="testpass",
            database="testdb"
        )
        url = config.get_sync_url()
        expected = "postgresql://testuser:testpass@testhost:5433/testdb"
        assert url == expected
    
    def test_async_url_generation(self):
        """测试异步连接URL生成"""
        config = DatabaseConfig(
            host="testhost",
            port=5433,
            user="testuser",
            password="testpass",
            database="testdb"
        )
        url = config.get_async_url()
        expected = "postgresql+asyncpg://testuser:testpass@testhost:5433/testdb"
        assert url == expected
    
    def test_alembic_url_generation(self):
        """测试Alembic连接URL生成"""
        config = DatabaseConfig()
        url = config.get_alembic_url()
        assert url.startswith("postgresql://")
        assert "rag_user" in url
        assert "rag_db" in url
    
    def test_connection_params(self):
        """测试连接参数"""
        config = DatabaseConfig()
        params = config.get_connection_params()
        
        assert "sslmode" in params
        assert "connect_timeout" in params
        assert "application_name" in params
        assert params["application_name"] == "rag-system"
    
    def test_engine_params(self):
        """测试引擎参数"""
        config = DatabaseConfig()
        params = config.get_engine_params()
        
        assert "pool_size" in params
        assert "max_overflow" in params
        assert "pool_timeout" in params
        assert "pool_recycle" in params
        assert "pool_pre_ping" in params
        assert params["pool_size"] == 5
        assert params["max_overflow"] == 10


class TestDatabaseManager:
    """数据库管理器测试"""
    
    def test_manager_initialization(self):
        """测试管理器初始化"""
        manager = DatabaseManager()
        assert manager.config is not None
        assert manager.sync_engine is None
        assert manager.async_engine is None
        assert manager.sync_session_factory is None
        assert manager.async_session_factory is None
    
    @patch('src.database.connection.create_engine')
    def test_init_sync_engine(self, mock_create_engine):
        """测试同步引擎初始化"""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        manager = DatabaseManager()
        manager.init_sync_engine()
        
        assert manager.sync_engine == mock_engine
        assert manager.sync_session_factory is not None
        mock_create_engine.assert_called_once()
    
    @patch('src.database.connection.create_async_engine')
    def test_init_async_engine(self, mock_create_async_engine):
        """测试异步引擎初始化"""
        mock_engine = MagicMock()
        mock_create_async_engine.return_value = mock_engine
        
        manager = DatabaseManager()
        manager.init_async_engine()
        
        assert manager.async_engine == mock_engine
        assert manager.async_session_factory is not None
        mock_create_async_engine.assert_called_once()
    
    @patch('src.database.connection.create_engine')
    def test_get_sync_session(self, mock_create_engine):
        """测试获取同步会话"""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        manager = DatabaseManager()
        manager.init_sync_engine()
        
        session_gen = manager.get_sync_session()
        session = next(session_gen)
        
        assert session is not None
    
    @patch('src.database.connection.create_async_engine')
    @pytest.mark.asyncio
    async def test_get_async_session(self, mock_create_async_engine):
        """测试获取异步会话"""
        mock_engine = MagicMock()
        mock_create_async_engine.return_value = mock_engine
        
        manager = DatabaseManager()
        manager.init_async_engine()
        
        session_gen = manager.get_async_session()
        session = await session_gen.__anext__()
        
        assert session is not None


class TestDatabaseOperations:
    """数据库操作测试"""
    
    @patch('src.database.connection.db_manager')
    def test_init_database(self, mock_manager):
        """测试数据库初始化"""
        mock_manager.init_sync_engine.return_value = None
        mock_manager.init_async_engine.return_value = None
        
        result = init_database()
        
        assert result is True
        mock_manager.init_sync_engine.assert_called_once()
        mock_manager.init_async_engine.assert_called_once()
    
    @patch('src.database.connection.db_manager')
    def test_close_database(self, mock_manager):
        """测试数据库关闭"""
        mock_sync_engine = MagicMock()
        mock_async_engine = MagicMock()
        mock_manager.sync_engine = mock_sync_engine
        mock_manager.async_engine = mock_async_engine
        
        close_database()
        
        mock_sync_engine.dispose.assert_called_once()
    
    @patch('src.database.connection.db_manager')
    def test_check_database_health_success(self, mock_manager):
        """测试数据库健康检查成功"""
        mock_session = MagicMock()
        mock_session.execute.return_value = None
        mock_manager.get_sync_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_manager.get_sync_session.return_value.__exit__ = MagicMock(return_value=None)
        
        result = check_database_health()
        
        assert result is True
    
    @patch('src.database.connection.db_manager')
    def test_check_database_health_failure(self, mock_manager):
        """测试数据库健康检查失败"""
        mock_manager.get_sync_session.side_effect = SQLAlchemyError("Connection failed")
        
        result = check_database_health()
        
        assert result is False


class TestSessionManagement:
    """会话管理测试"""
    
    @patch('src.database.connection.db_manager')
    def test_get_sync_session_function(self, mock_manager):
        """测试同步会话获取函数"""
        mock_session = MagicMock()
        mock_manager.get_sync_session.return_value = iter([mock_session])
        
        session_gen = get_sync_session()
        session = next(session_gen)
        
        assert session == mock_session
    
    @patch('src.database.connection.db_manager')
    @pytest.mark.asyncio
    async def test_get_async_session_function(self, mock_manager):
        """测试异步会话获取函数"""
        mock_session = MagicMock()
        
        async def mock_async_gen():
            yield mock_session
        
        mock_manager.get_async_session.return_value = mock_async_gen()
        
        session_gen = get_async_session()
        session = await session_gen.__anext__()
        
        assert session == mock_session


class TestDatabaseInitialization:
    """数据库初始化测试"""
    
    @patch('src.database.init_db.psycopg2.connect')
    def test_create_database_if_not_exists(self, mock_connect):
        """测试数据库创建"""
        from src.database.init_db import create_database_if_not_exists
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        # 模拟数据库不存在
        mock_cursor.fetchone.return_value = None
        
        result = create_database_if_not_exists()
        
        assert result is True
        mock_cursor.execute.assert_called()
    
    @patch('src.database.init_db.get_sync_session')
    def test_create_extensions(self, mock_get_session):
        """测试扩展创建"""
        from src.database.init_db import create_extensions
        
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        result = create_extensions()
        
        assert result is True
        # 验证执行了扩展创建SQL
        assert mock_session.execute.call_count >= 3  # uuid-ossp, pg_trgm, btree_gin
    
    @patch('src.database.init_db.get_sync_session')
    def test_create_indexes(self, mock_get_session):
        """测试索引创建"""
        from src.database.init_db import create_indexes
        
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        result = create_indexes()
        
        assert result is True
        # 验证执行了索引创建SQL
        assert mock_session.execute.call_count > 0
    
    @patch('src.database.init_db.get_sync_session')
    @patch('src.database.init_db.user_repository')
    def test_create_default_admin(self, mock_user_repo, mock_get_session):
        """测试默认管理员创建"""
        from src.database.init_db import create_default_admin
        
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        # 模拟管理员不存在
        mock_user_repo.get_by_username.return_value = None
        mock_user_repo.create.return_value = MagicMock()
        
        result = create_default_admin()
        
        assert result is True
        mock_user_repo.create.assert_called_once()


class TestConfigIntegration:
    """配置集成测试"""
    
    def test_global_config_instance(self):
        """测试全局配置实例"""
        assert db_config is not None
        assert isinstance(db_config, DatabaseConfig)
    
    def test_global_manager_instance(self):
        """测试全局管理器实例"""
        assert db_manager is not None
        assert isinstance(db_manager, DatabaseManager)
    
    def test_config_from_settings(self):
        """测试从设置加载配置"""
        config = DatabaseConfig()
        
        # 验证配置值来自settings
        assert config.host == settings.db_host
        assert config.port == settings.db_port
        assert config.user == settings.db_user
        assert config.database == settings.db_name


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])