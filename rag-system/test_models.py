#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据模型测试文件

测试所有数据模型的创建、验证和序列化功能
"""

import pytest
from datetime import datetime, timezone
from uuid import uuid4
from decimal import Decimal

from src.models import (
    User, UserCreate, UserUpdate, UserResponse,
    Document, DocumentCreate, DocumentUpdate, DocumentResponse,
    DocumentChunk, DocumentChunkCreate, DocumentChunkUpdate, DocumentChunkResponse,
    QueryHistory, QueryHistoryCreate, QueryHistoryUpdate, QueryHistoryResponse,
    SystemConfig, SystemConfigCreate, SystemConfigUpdate, SystemConfigResponse
)
from src.models.base import UserRole, DocumentStatus, DocumentType, QueryStatus, QueryType


class TestUserModel:
    """用户模型测试"""
    
    def test_user_create_valid(self):
        """测试创建有效用户"""
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "password123",
            "full_name": "Test User"
        }
        user_create = UserCreate(**user_data)
        assert user_create.username == "testuser"
        assert user_create.email == "test@example.com"
        assert user_create.password == "password123"
        assert user_create.full_name == "Test User"
        assert user_create.role == UserRole.USER  # 默认角色
        assert user_create.is_active is True  # 默认激活
    
    def test_user_create_admin(self):
        """测试创建管理员用户"""
        user_data = {
            "username": "admin",
            "email": "admin@example.com",
            "password": "admin123",
            "role": UserRole.ADMIN
        }
        user_create = UserCreate(**user_data)
        assert user_create.role == UserRole.ADMIN
    
    def test_user_update(self):
        """测试用户更新"""
        user_update = UserUpdate(
            full_name="Updated Name",
            email="updated@example.com"
        )
        assert user_update.full_name == "Updated Name"
        assert user_update.email == "updated@example.com"
        assert user_update.password is None  # 可选字段
    
    def test_user_response(self):
        """测试用户响应模型"""
        user_id = uuid4()
        now = datetime.now(timezone.utc)
        
        user_response = UserResponse(
            id=user_id,
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            role=UserRole.USER,
            is_active=True,
            created_at=now,
            updated_at=now
        )
        
        assert user_response.id == user_id
        assert user_response.username == "testuser"
        # 密码不应该在响应中
        assert not hasattr(user_response, 'password_hash')


class TestDocumentModel:
    """文档模型测试"""
    
    def test_document_create(self):
        """测试创建文档"""
        doc_data = {
            "title": "Test Document",
            "content": "This is test content",
            "file_path": "/uploads/test.txt",
            "file_hash": "abc123",
            "file_size": 1024,
            "document_type": DocumentType.TEXT,
            "owner_id": uuid4()
        }
        doc_create = DocumentCreate(**doc_data)
        assert doc_create.title == "Test Document"
        assert doc_create.document_type == DocumentType.TEXT
        assert doc_create.status == DocumentStatus.PENDING  # 默认状态
    
    def test_document_update(self):
        """测试文档更新"""
        doc_update = DocumentUpdate(
            title="Updated Title",
            status=DocumentStatus.PROCESSED
        )
        assert doc_update.title == "Updated Title"
        assert doc_update.status == DocumentStatus.PROCESSED
    
    def test_document_response(self):
        """测试文档响应模型"""
        doc_id = uuid4()
        owner_id = uuid4()
        now = datetime.now(timezone.utc)
        
        doc_response = DocumentResponse(
            id=doc_id,
            title="Test Document",
            file_path="/uploads/test.txt",
            file_hash="abc123",
            file_size=1024,
            document_type=DocumentType.TEXT,
            status=DocumentStatus.PROCESSED,
            owner_id=owner_id,
            created_at=now,
            updated_at=now,
            processed_at=now
        )
        
        assert doc_response.id == doc_id
        assert doc_response.owner_id == owner_id
        # 内容不应该在响应中（太大）
        assert not hasattr(doc_response, 'content')


class TestDocumentChunkModel:
    """文档块模型测试"""
    
    def test_chunk_create(self):
        """测试创建文档块"""
        chunk_data = {
            "document_id": uuid4(),
            "content": "This is a chunk of text",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 100,
            "token_count": 20
        }
        chunk_create = DocumentChunkCreate(**chunk_data)
        assert chunk_create.content == "This is a chunk of text"
        assert chunk_create.chunk_index == 0
        assert chunk_create.token_count == 20
    
    def test_chunk_update(self):
        """测试文档块更新"""
        chunk_update = DocumentChunkUpdate(
            vector_id="vector_123",
            embedding_model="text-embedding-ada-002"
        )
        assert chunk_update.vector_id == "vector_123"
        assert chunk_update.embedding_model == "text-embedding-ada-002"


class TestQueryHistoryModel:
    """查询历史模型测试"""
    
    def test_query_create(self):
        """测试创建查询历史"""
        query_data = {
            "user_id": uuid4(),
            "query_text": "What is machine learning?",
            "query_type": QueryType.SEARCH,
            "session_id": "session_123"
        }
        query_create = QueryHistoryCreate(**query_data)
        assert query_create.query_text == "What is machine learning?"
        assert query_create.query_type == QueryType.SEARCH
        assert query_create.status == QueryStatus.PENDING  # 默认状态
    
    def test_query_update(self):
        """测试查询历史更新"""
        query_update = QueryHistoryUpdate(
            response_text="Machine learning is...",
            status=QueryStatus.COMPLETED,
            response_time=Decimal("1.5")
        )
        assert query_update.response_text == "Machine learning is..."
        assert query_update.status == QueryStatus.COMPLETED
        assert query_update.response_time == Decimal("1.5")


class TestSystemConfigModel:
    """系统配置模型测试"""
    
    def test_config_create(self):
        """测试创建系统配置"""
        config_data = {
            "key": "max_file_size",
            "value": "10485760",
            "category": "upload",
            "description": "Maximum file size in bytes"
        }
        config_create = SystemConfigCreate(**config_data)
        assert config_create.key == "max_file_size"
        assert config_create.value == "10485760"
        assert config_create.category == "upload"
    
    def test_config_update(self):
        """测试系统配置更新"""
        config_update = SystemConfigUpdate(
            value="20971520",
            description="Updated maximum file size"
        )
        assert config_update.value == "20971520"
        assert config_update.description == "Updated maximum file size"


class TestModelValidation:
    """模型验证测试"""
    
    def test_user_email_validation(self):
        """测试用户邮箱验证"""
        with pytest.raises(ValueError):
            UserCreate(
                username="test",
                email="invalid-email",  # 无效邮箱
                password="password123"
            )
    
    def test_document_file_size_validation(self):
        """测试文档文件大小验证"""
        with pytest.raises(ValueError):
            DocumentCreate(
                title="Test",
                content="Content",
                file_path="/test.txt",
                file_hash="hash",
                file_size=-1,  # 负数文件大小
                document_type=DocumentType.TEXT,
                owner_id=uuid4()
            )
    
    def test_chunk_index_validation(self):
        """测试文档块索引验证"""
        with pytest.raises(ValueError):
            DocumentChunkCreate(
                document_id=uuid4(),
                content="Content",
                chunk_index=-1,  # 负数索引
                start_char=0,
                end_char=100
            )


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])