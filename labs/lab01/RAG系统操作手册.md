# RAG系统操作手册

## 概述

本手册详细说明了RAG（检索增强生成）系统的所有功能模块，包括API接口、输入输出格式、使用示例和最佳实践。系统基于FastAPI构建，提供完整的文档处理、向量化、语义搜索和智能问答功能。

**系统基础信息**：
- 基础URL: `http://localhost:8000`
- API版本: `v1`
- 支持文件类型: PDF, TXT
- 最大文件大小: 10MB
- 向量维度: 1024

---

## 1. 系统健康检查

### 接口信息
- **路径**: `/api/v1/system/health`
- **方法**: `GET`
- **描述**: 检查系统各组件的健康状态

### 请求参数
无需参数

### 响应格式
```json
{
  "success": true,
  "message": "系统健康检查完成",
  "data": {
    "status": "healthy",
    "components": {
      "database": "healthy",
      "vector_store": "healthy", 
      "embedding_service": "healthy",
      "volcengine_api": "healthy"
    },
    "uptime": 237,
    "timestamp": "2025-01-08T07:46:15"
  }
}
```

### 字段说明
- `status`: 系统整体状态 (healthy/unhealthy)
- `components`: 各组件状态详情
- `uptime`: 系统运行时间（秒）
- `timestamp`: 检查时间戳

### 使用示例
```bash
curl -X GET "http://localhost:8000/api/v1/system/health"
```

### 错误处理
- `503 Service Unavailable`: 系统组件异常

---

## 2. 文档上传功能

### 接口信息
- **路径**: `/api/v1/documents/upload`
- **方法**: `POST`
- **描述**: 上传并处理文档文件

### 请求参数
- **Content-Type**: `multipart/form-data`
- **file** (必需): 文档文件
- **title** (可选): 文档标题
- **description** (可选): 文档描述

### 支持的文件类型
- `.pdf`: PDF文档
- `.txt`: 纯文本文件

### 响应格式
```json
{
  "success": true,
  "message": "文档上传成功",
  "data": {
    "document": {
      "id": "2fc7a2dc-76d3-4e79-837a-9ddd15912d04",
      "filename": "test_document.txt",
      "title": "测试文档",
      "description": "这是一个测试文档",
      "file_size": 263,
      "file_type": "text/plain",
      "char_count": 103,
      "word_count": 11,
      "estimated_tokens": 26,
      "chunk_count": 1,
      "is_processed": true,
      "is_vectorized": false,
      "created_at": "2025-01-08T07:47:12",
      "updated_at": "2025-01-08T07:47:12"
    },
    "processing_info": {
      "chunks_created": 1,
      "processing_time": 0.15
    }
  }
}
```

### 字段说明
- `id`: 文档唯一标识符
- `file_size`: 文件大小（字节）
- `char_count`: 字符数量
- `word_count`: 词语数量
- `estimated_tokens`: 预估Token数量
- `chunk_count`: 分块数量
- `is_processed`: 是否已处理
- `is_vectorized`: 是否已向量化

### 使用示例
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@test_document.txt" \
  -F "title=测试文档" \
  -F "description=这是一个测试文档"
```

### 错误处理
- `400 Bad Request`: 文件格式不支持或文件过大
- `413 Payload Too Large`: 文件超过大小限制
- `500 Internal Server Error`: 文档处理失败

---

## 3. 文档管理功能

### 3.1 获取文档列表

#### 接口信息
- **路径**: `/api/v1/documents/`
- **方法**: `GET`
- **描述**: 获取文档列表，支持分页和过滤

#### 请求参数
- `page` (可选): 页码，默认1
- `size` (可选): 每页数量，默认10
- `is_processed` (可选): 过滤已处理文档 (true/false)
- `is_vectorized` (可选): 过滤已向量化文档 (true/false)
- `search` (可选): 搜索关键词（标题或描述）

#### 响应格式
```json
{
  "success": true,
  "message": "获取文档列表成功",
  "data": {
    "documents": [
      {
        "id": "doc-id-1",
        "filename": "document1.pdf",
        "title": "文档1",
        "description": "描述1",
        "file_size": 1024,
        "file_type": "application/pdf",
        "is_processed": true,
        "is_vectorized": true,
        "created_at": "2025-01-08T07:00:00"
      }
    ],
    "pagination": {
      "page": 1,
      "size": 10,
      "total": 9,
      "pages": 1
    }
  }
}
```

#### 使用示例
```bash
# 获取所有文档
curl -X GET "http://localhost:8000/api/v1/documents/"

# 获取已向量化的文档
curl -X GET "http://localhost:8000/api/v1/documents/?is_vectorized=true"

# 分页获取
curl -X GET "http://localhost:8000/api/v1/documents/?page=1&size=5"
```

### 3.2 获取文档详情

#### 接口信息
- **路径**: `/api/v1/documents/{document_id}`
- **方法**: `GET`
- **描述**: 获取指定文档的详细信息

#### 请求参数
- `document_id` (路径参数): 文档ID

#### 响应格式
```json
{
  "success": true,
  "message": "获取文档详情成功",
  "data": {
    "id": "doc-id",
    "filename": "document.pdf",
    "title": "文档标题",
    "description": "文档描述",
    "file_size": 2048,
    "file_type": "application/pdf",
    "content_hash": "abc123",
    "char_count": 500,
    "word_count": 100,
    "estimated_tokens": 125,
    "chunk_count": 3,
    "is_processed": true,
    "is_vectorized": true,
    "processed_at": "2025-01-08T07:30:00",
    "vectorized_at": "2025-01-08T07:31:00",
    "created_at": "2025-01-08T07:29:00",
    "updated_at": "2025-01-08T07:31:00",
    "metadata": "{\"source\": \"upload\"}"
  }
}
```

#### 使用示例
```bash
curl -X GET "http://localhost:8000/api/v1/documents/doc-id-123"
```

#### 错误处理
- `404 Not Found`: 文档不存在

---

## 4. 文档向量化

### 接口信息
- **路径**: `/api/v1/vectors/vectorize`
- **方法**: `POST`
- **描述**: 对指定文档进行向量化处理

### 请求参数
```json
{
  "document_id": "doc-id-123"
}
```

### 响应格式
```json
{
  "success": true,
  "message": "向量化处理完成",
  "data": {
    "document_id": "doc-id-123",
    "processed_documents": 1,
    "processed_chunks": 3,
    "processing_time": 0.18,
    "failed_documents": 0,
    "results": [
      {
        "chunk_id": "chunk-1",
        "vector_id": "vector-1",
        "embedding_dimension": 1024,
        "success": true
      }
    ]
  }
}
```

### 字段说明
- `processed_documents`: 处理的文档数量
- `processed_chunks`: 处理的分块数量
- `processing_time`: 处理耗时（秒）
- `failed_documents`: 失败的文档数量
- `embedding_dimension`: 向量维度

### 使用示例
```bash
curl -X POST "http://localhost:8000/api/v1/vectors/vectorize" \
  -H "Content-Type: application/json" \
  -d '{"document_id": "doc-id-123"}'
```

### 错误处理
- `404 Not Found`: 文档不存在
- `400 Bad Request`: 文档未处理或已向量化
- `500 Internal Server Error`: 向量化处理失败

---

## 5. 语义搜索功能

### 接口信息
- **路径**: `/api/v1/vectors/search`
- **方法**: `POST`
- **描述**: 基于语义相似度搜索相关文档片段

### 请求参数
```json
{
  "query": "搜索关键词",
  "top_k": 5,
  "score_threshold": 0.5,
  "document_ids": ["doc-1", "doc-2"]
}
```

### 参数说明
- `query` (必需): 搜索查询文本
- `top_k` (可选): 返回结果数量，默认5
- `score_threshold` (可选): 相似度阈值，默认0.5
- `document_ids` (可选): 限制搜索的文档ID列表

### 响应格式
```json
{
  "success": true,
  "message": "搜索完成，找到 3 个相关结果",
  "data": {
    "query": "搜索关键词",
    "results": [
      {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "content": "相关文本内容...",
        "score": 0.78,
        "start_position": 0,
        "end_position": 100,
        "document_info": {
          "filename": "document.pdf",
          "title": "文档标题"
        }
      }
    ],
    "search_params": {
      "top_k": 5,
      "score_threshold": 0.5,
      "total_found": 3
    }
  }
}
```

### 字段说明
- `score`: 相似度分数 (0-1)
- `start_position`: 文本起始位置
- `end_position`: 文本结束位置
- `document_info`: 文档基本信息

### 使用示例
```bash
curl -X POST "http://localhost:8000/api/v1/vectors/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "人工智能技术",
    "top_k": 3,
    "score_threshold": 0.6
  }'
```

### 错误处理
- `400 Bad Request`: 查询参数无效
- `500 Internal Server Error`: 搜索服务异常

---

## 6. 智能问答功能

### 接口信息
- **路径**: `/api/v1/qa/ask`
- **方法**: `POST`
- **描述**: 基于文档内容进行智能问答

### 请求参数
```json
{
  "question": "用户问题",
  "max_tokens": 500,
  "temperature": 0.7,
  "top_k": 5,
  "score_threshold": 0.5,
  "document_ids": ["doc-1", "doc-2"]
}
```

### 参数说明
- `question` (必需): 用户问题
- `max_tokens` (可选): 最大生成Token数，默认500
- `temperature` (可选): 生成温度，默认0.7
- `top_k` (可选): 检索结果数量，默认5
- `score_threshold` (可选):