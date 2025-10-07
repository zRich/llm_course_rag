#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仓库测试文件

测试所有仓库类的CRUD操作和业务逻辑
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from uuid import uuid4
from decimal import Decimal

from src.repositories import (
    BaseRepository,
    UserRepository, user_repository,
    DocumentRepository, DocumentChunkRepository,
    document_repository, document_chunk_repository,
    QueryHistoryRepository, SystemConfigRepository,
    query_history_repository, system_config_repository
)
from src.models import (
    User, UserCreate, UserUpdate,
    Document, DocumentCreate, DocumentUpdate,
    DocumentChunk, DocumentChunkCreate, DocumentChunkUpdate,
    QueryHistory, QueryHistoryCreate, QueryHistoryUpdate,
    SystemConfig, SystemConfigCreate, SystemConfigUpdate
)
from src.models.base import UserRole, DocumentStatus, DocumentType, QueryStatus, QueryType


class TestBaseRepository:
    """基础仓库测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.mock_session = MagicMock()
        self.repository = BaseRepository(User)
    
    def test_repository_initialization(self):
        """测试仓库初始化"""
        repo = BaseRepository(User)
        assert repo.model == User
    
    @patch('src.repositories.base.get_sync_session')
    def test_create_sync(self, mock_get_session):
        """测试同步创建"""
        mock_get_session.return_value.__enter__ = MagicMock(return_value=self.mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        user_data = UserCreate(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        
        mock_user = User(id=uuid4(), username="testuser", email="test@example.com")
        self.mock_session.add.return_value = None
        self.mock_session.commit.return_value = None
        self.mock_session.refresh.return_value = None
        
        # 模拟User.model_validate
        with patch.object(User, 'model_validate', return_value=mock_user):
            result = self.repository.create(user_data)
        
        assert result == mock_user
        self.mock_session.add.assert_called_once()
        self.mock_session.commit.assert_called_once()
    
    @patch('src.repositories.base.get_sync_session')
    def test_get_by_id_sync(self, mock_get_session):
        """测试同步按ID获取"""
        mock_get_session.return_value.__enter__ = MagicMock(return_value=self.mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        user_id = uuid4()
        mock_user = User(id=user_id, username="testuser", email="test@example.com")
        self.mock_session.get.return_value = mock_user
        
        result = self.repository.get_by_id(user_id)
        
        assert result == mock_user
        self.mock_session.get.assert_called_once_with(User, user_id)
    
    @patch('src.repositories.base.get_sync_session')
    def test_get_all_sync(self, mock_get_session):
        """测试同步获取所有记录"""
        mock_get_session.return_value.__enter__ = MagicMock(return_value=self.mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        mock_query = MagicMock()
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        
        self.mock_session.query.return_value = mock_query
        
        result = self.repository.get_all(skip=0, limit=10)
        
        assert result == []
        self.mock_session.query.assert_called_once_with(User)
    
    @patch('src.repositories.base.get_sync_session')
    def test_update_sync(self, mock_get_session):
        """测试同步更新"""
        mock_get_session.return_value.__enter__ = MagicMock(return_value=self.mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        user_id = uuid4()
        mock_user = User(id=user_id, username="testuser", email="test@example.com")
        self.mock_session.get.return_value = mock_user
        
        update_data = UserUpdate(full_name="Updated Name")
        
        result = self.repository.update(user_id, update_data)
        
        assert result == mock_user
        self.mock_session.commit.assert_called_once()
    
    @patch('src.repositories.base.get_sync_session')
    def test_delete_sync(self, mock_get_session):
        """测试同步删除"""
        mock_get_session.return_value.__enter__ = MagicMock(return_value=self.mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        user_id = uuid4()
        mock_user = User(id=user_id, username="testuser", email="test@example.com")
        self.mock_session.get.return_value = mock_user
        
        result = self.repository.delete(user_id)
        
        assert result is True
        self.mock_session.delete.assert_called_once_with(mock_user)
        self.mock_session.commit.assert_called_once()


class TestUserRepository:
    """用户仓库测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.mock_session = MagicMock()
        self.repository = UserRepository()
    
    @patch('src.repositories.user.get_sync_session')
    def test_get_by_username(self, mock_get_session):
        """测试按用户名获取用户"""
        mock_get_session.return_value.__enter__ = MagicMock(return_value=self.mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None
        
        self.mock_session.query.return_value = mock_query
        
        result = self.repository.get_by_username("testuser")
        
        assert result is None
        self.mock_session.query.assert_called_once_with(User)
    
    @patch('src.repositories.user.get_sync_session')
    def test_get_by_email(self, mock_get_session):
        """测试按邮箱获取用户"""
        mock_get_session.return_value.__enter__ = MagicMock(return_value=self.mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None
        
        self.mock_session.query.return_value = mock_query
        
        result = self.repository.get_by_email("test@example.com")
        
        assert result is None
    
    def test_hash_password(self):
        """测试密码哈希"""
        password = "password123"
        hashed = self.repository._hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 0
    
    def test_verify_password(self):
        """测试密码验证"""
        password = "password123"
        hashed = self.repository._hash_password(password)
        
        assert self.repository._verify_password(password, hashed) is True
        assert self.repository._verify_password("wrongpassword", hashed) is False
    
    @patch('src.repositories.user.get_sync_session')
    def test_authenticate_user(self, mock_get_session):
        """测试用户认证"""
        mock_get_session.return_value.__enter__ = MagicMock(return_value=self.mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        password = "password123"
        hashed_password = self.repository._hash_password(password)
        
        mock_user = User(
            id=uuid4(),
            username="testuser",
            email="test@example.com",
            password_hash=hashed_password,
            is_active=True
        )
        
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = mock_user
        
        self.mock_session.query.return_value = mock_query
        
        result = self.repository.authenticate_user("testuser", password)
        
        assert result == mock_user
    
    @patch('src.repositories.user.get_sync_session')
    def test_get_active_users(self, mock_get_session):
        """测试获取活跃用户"""
        mock_get_session.return_value.__enter__ = MagicMock(return_value=self.mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        
        self.mock_session.query.return_value = mock_query
        
        result = self.repository.get_active_users()
        
        assert result == []


class TestDocumentRepository:
    """文档仓库测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.mock_session = MagicMock()
        self.repository = DocumentRepository()
    
    @patch('src.repositories.document.get_sync_session')
    def test_get_by_title(self, mock_get_session):
        """测试按标题获取文档"""
        mock_get_session.return_value.__enter__ = MagicMock(return_value=self.mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None
        
        self.mock_session.query.return_value = mock_query
        
        result = self.repository.get_by_title("Test Document")
        
        assert result is None
    
    @patch('src.repositories.document.get_sync_session')
    def test_get_by_hash(self, mock_get_session):
        """测试按哈希获取文档"""
        mock_get_session.return_value.__enter__ = MagicMock(return_value=self.mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None
        
        self.mock_session.query.return_value = mock_query
        
        result = self.repository.get_by_hash("abc123")
        
        assert result is None
    
    @patch('src.repositories.document.get_sync_session')
    def test_get_by_owner(self, mock_get_session):
        """测试按所有者获取文档"""
        mock_get_session.return_value.__enter__ = MagicMock(return_value=self.mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        
        self.mock_session.query.return_value = mock_query
        
        owner_id = uuid4()
        result = self.repository.get_by_owner(owner_id)
        
        assert result == []
    
    @patch('src.repositories.document.get_sync_session')
    def test_get_by_status(self, mock_get_session):
        """测试按状态获取文档"""
        mock_get_session.return_value.__enter__ = MagicMock(return_value=self.mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        
        self.mock_session.query.return_value = mock_query
        
        result = self.repository.get_by_status(DocumentStatus.PROCESSED)
        
        assert result == []


class TestDocumentChunkRepository:
    """文档块仓库测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.mock_session = MagicMock()
        self.repository = DocumentChunkRepository()
    
    @patch('src.repositories.document.get_sync_session')
    def test_get_by_document_id(self, mock_get_session):
        """测试按文档ID获取块"""
        mock_get_session.return_value.__enter__ = MagicMock(return_value=self.mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        
        self.mock_session.query.return_value = mock_query
        
        document_id = uuid4()
        result = self.repository.get_by_document_id(document_id)
        
        assert result == []
    
    @patch('src.repositories.document.get_sync_session')
    def test_get_by_vector_id(self, mock_get_session):
        """测试按向量ID获取块"""
        mock_get_session.return_value.__enter__ = MagicMock(return_value=self.mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None
        
        self.mock_session.query.return_value = mock_query
        
        result = self.repository.get_by_vector_id("vector_123")
        
        assert result is None


class TestQueryHistoryRepository:
    """查询历史仓库测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.mock_session = MagicMock()
        self.repository = QueryHistoryRepository()
    
    @patch('src.repositories.query.get_sync_session')
    def test_get_by_user_id(self, mock_get_session):
        """测试按用户ID获取查询历史"""
        mock_get_session.return_value.__enter__ = MagicMock(return_value=self.mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        
        self.mock_session.query.return_value = mock_query
        
        user_id = uuid4()
        result = self.repository.get_by_user_id(user_id)
        
        assert result == []
    
    @patch('src.repositories.query.get_sync_session')
    def test_get_by_session_id(self, mock_get_session):
        """测试按会话ID获取查询历史"""
        mock_get_session.return_value.__enter__ = MagicMock(return_value=self.mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        
        self.mock_session.query.return_value = mock_query
        
        result = self.repository.get_by_session_id("session_123")
        
        assert result == []


class TestSystemConfigRepository:
    """系统配置仓库测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.mock_session = MagicMock()
        self.repository = SystemConfigRepository()
    
    @patch('src.repositories.query.get_sync_session')
    def test_get_by_key(self, mock_get_session):
        """测试按键获取配置"""
        mock_get_session.return_value.__enter__ = MagicMock(return_value=self.mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None
        
        self.mock_session.query.return_value = mock_query
        
        result = self.repository.get_by_key("max_file_size")
        
        assert result is None
    
    @patch('src.repositories.query.get_sync_session')
    def test_get_by_category(self, mock_get_session):
        """测试按分类获取配置"""
        mock_get_session.return_value.__enter__ = MagicMock(return_value=self.mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        
        self.mock_session.query.return_value = mock_query
        
        result = self.repository.get_by_category("upload")
        
        assert result == []
    
    @patch('src.repositories.query.get_sync_session')
    def test_set_config(self, mock_get_session):
        """测试设置配置"""
        mock_get_session.return_value.__enter__ = MagicMock(return_value=self.mock_session)
        mock_get_session.return_value.__exit__ = MagicMock(return_value=None)
        
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None
        
        self.mock_session.query.return_value = mock_query
        
        # 模拟SystemConfig.model_validate
        mock_config = SystemConfig(
            id=uuid4(),
            key="test_key",
            value="test_value",
            category="test"
        )
        
        with patch.object(SystemConfig, 'model_validate', return_value=mock_config):
            result = self.repository.set_config("test_key", "test_value", "test")
        
        assert result == mock_config


class TestRepositoryInstances:
    """仓库实例测试"""
    
    def test_global_instances_exist(self):
        """测试全局实例存在"""
        assert user_repository is not None
        assert isinstance(user_repository, UserRepository)
        
        assert document_repository is not None
        assert isinstance(document_repository, DocumentRepository)
        
        assert document_chunk_repository is not None
        assert isinstance(document_chunk_repository, DocumentChunkRepository)
        
        assert query_history_repository is not None
        assert isinstance(query_history_repository, QueryHistoryRepository)
        
        assert system_config_repository is not None
        assert isinstance(system_config_repository, SystemConfigRepository)


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])