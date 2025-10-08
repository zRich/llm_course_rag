# RAG系统实验1 - 完整答案解析

## 📋 实验概述

本文档提供RAG系统实验1的完整答案解析，包含所有5个实验任务的详细实现方案、关键技术点解析和测试验证方法。学生可以参考本文档理解每个任务的实现原理和技术细节。

## 🎯 实验任务总览

| 任务编号 | 任务名称 | 核心技术 | 难度等级 |
|---------|---------|---------|---------|
| 任务1 | 文档上传和处理 | 文件处理、PDF解析、文本分块 | ⭐⭐ |
| 任务2 | 文档向量化 | 嵌入模型、向量生成、批处理 | ⭐⭐⭐ |
| 任务3 | 语义搜索实现 | 向量搜索、相似度计算、结果排序 | ⭐⭐⭐ |
| 任务4 | RAG问答系统 | 检索增强生成、提示工程、LLM调用 | ⭐⭐⭐⭐ |
| 任务5 | 系统监控和健康检查 | 系统监控、状态检查、性能统计 | ⭐⭐ |

---

## 📄 任务1：文档上传和处理

### 🎯 任务目标

实现PDF文档的上传、解析和文本分块功能，为后续的向量化和检索奠定基础。

### 🔧 核心实现

#### 1.1 文档处理服务 (`src/services/document_processor.py`)

```python
import os
import uuid
from typing import List, Optional
from pathlib import Path
import logging
from sqlalchemy.orm import Session

from ..models.document import Document
from ..models.chunk import Chunk
from ..services.pdf_parser import PDFParser
from ..services.txt_parser import TxtParser
from ..services.text_splitter import TextSplitter
from ..utils.logger import get_logger

logger = get_logger(__name__)

class DocumentProcessor:
    """文档处理服务 - 负责文档上传、解析和分块"""
    
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        
        # 初始化解析器
        self.pdf_parser = PDFParser()
        self.txt_parser = TxtParser()
        self.text_splitter = TextSplitter()
        
        # 支持的文件类型
        self.supported_types = {
            '.pdf': self.pdf_parser,
            '.txt': self.txt_parser
        }
    
    async def process_document(
        self, 
        file_content: bytes, 
        filename: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        db: Session = None
    ) -> Document:
        """
        处理上传的文档
        
        Args:
            file_content: 文件内容
            filename: 文件名
            title: 文档标题
            description: 文档描述
            db: 数据库会话
            
        Returns:
            Document: 创建的文档对象
        """
        try:
            # TODO(lab01-task1): 实现文档处理逻辑
            # 1. 验证文件类型和大小
            file_ext = Path(filename).suffix.lower()
            if file_ext not in self.supported_types:
                raise ValueError(f"不支持的文件类型: {file_ext}")
            
            # 检查文件大小 (限制为10MB)
            max_size = 10 * 1024 * 1024  # 10MB
            if len(file_content) > max_size:
                raise ValueError(f"文件大小超过限制: {len(file_content)} bytes")
            
            # 2. 保存文件到本地
            file_id = str(uuid.uuid4())
            file_path = self.upload_dir / f"{file_id}{file_ext}"
            
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # 3. 创建文档记录
            document = Document(
                id=file_id,
                title=title or filename,
                description=description or "",
                filename=filename,
                file_path=str(file_path),
                file_size=len(file_content),
                status="processing"
            )
            
            # 保存到数据库
            if db:
                db.add(document)
                db.commit()
                db.refresh(document)
            
            # 4. 解析文档内容
            parser = self.supported_types[file_ext]
            text_content = await parser.parse(file_path)
            
            if not text_content.strip():
                raise ValueError("文档内容为空或解析失败")
            
            # 5. 文本分块处理
            chunks = await self._split_text_into_chunks(
                text_content, 
                document.id, 
                db
            )
            
            # 6. 更新文档状态
            document.status = "completed"
            document.chunk_count = len(chunks)
            
            if db:
                db.commit()
            
            logger.info(f"文档处理完成: {filename}, 生成 {len(chunks)} 个文本块")
            return document
            
        except Exception as e:
            logger.error(f"文档处理失败: {filename}, 错误: {str(e)}")
            if db and 'document' in locals():
                document.status = "failed"
                db.commit()
            raise
    
    async def _split_text_into_chunks(
        self, 
        text: str, 
        document_id: str, 
        db: Session
    ) -> List[Chunk]:
        """
        将文本分割成块
        
        Args:
            text: 原始文本
            document_id: 文档ID
            db: 数据库会话
            
        Returns:
            List[Chunk]: 文本块列表
        """
        # TODO(lab01-task1): 实现文本分块逻辑
        # 使用TextSplitter进行智能分块
        text_chunks = self.text_splitter.split_text(text)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = Chunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                content=chunk_text,
                chunk_index=i,
                start_char=text.find(chunk_text),
                end_char=text.find(chunk_text) + len(chunk_text),
                token_count=len(chunk_text.split())  # 简单的token计数
            )
            chunks.append(chunk)
            
            # 保存到数据库
            if db:
                db.add(chunk)
        
        if db:
            db.commit()
        
        return chunks
    
    def get_supported_types(self) -> List[str]:
        """获取支持的文件类型"""
        return list(self.supported_types.keys())
    
    async def delete_document(self, document_id: str, db: Session) -> bool:
        """
        删除文档及其相关文件
        
        Args:
            document_id: 文档ID
            db: 数据库会话
            
        Returns:
            bool: 删除是否成功
        """
        try:
            # 查找文档
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                return False
            
            # 删除物理文件
            if document.file_path and os.path.exists(document.file_path):
                os.remove(document.file_path)
            
            # 删除数据库记录（级联删除chunks）
            db.delete(document)
            db.commit()
            
            logger.info(f"文档删除成功: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"文档删除失败: {document_id}, 错误: {str(e)}")
            return False
```

#### 1.2 文本分割器 (`src/services/text_splitter.py`)

```python
import re
from typing import List, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)

class TextSplitter:
    """智能文本分割器 - 保持语义完整性"""
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 默认分隔符，按优先级排序
        self.separators = separators or [
            "\n\n",  # 段落分隔
            "\n",    # 行分隔
            "。",    # 中文句号
            "！",    # 中文感叹号
            "？",    # 中文问号
            ".",     # 英文句号
            "!",     # 英文感叹号
            "?",     # 英文问号
            ";",     # 分号
            ",",     # 逗号
            " ",     # 空格
            ""       # 字符级别分割
        ]
    
    def split_text(self, text: str) -> List[str]:
        """
        智能分割文本，保持语义完整性
        
        Args:
            text: 待分割的文本
            
        Returns:
            List[str]: 分割后的文本块列表
        """
        # TODO(lab01-task1): 实现智能文本分割算法
        if not text.strip():
            return []
        
        # 清理文本
        text = self._clean_text(text)
        
        # 如果文本长度小于chunk_size，直接返回
        if len(text) <= self.chunk_size:
            return [text]
        
        # 递归分割
        chunks = self._split_text_recursive(text, self.separators)
        
        # 合并过小的块
        chunks = self._merge_small_chunks(chunks)
        
        logger.info(f"文本分割完成: 原长度 {len(text)}, 分割为 {len(chunks)} 块")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """清理文本，去除多余空白字符"""
        # 去除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 去除首尾空白
        text = text.strip()
        return text
    
    def _split_text_recursive(
        self, 
        text: str, 
        separators: List[str]
    ) -> List[str]:
        """递归分割文本"""
        if not separators:
            # 没有分隔符时，强制按字符分割
            return [text[i:i+self.chunk_size] 
                   for i in range(0, len(text), self.chunk_size)]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            # 字符级别分割
            return [text[i:i+self.chunk_size] 
                   for i in range(0, len(text), self.chunk_size)]
        
        # 按当前分隔符分割
        splits = text.split(separator)
        
        # 重新组合分隔符
        if len(splits) > 1:
            splits = [splits[0]] + [separator + s for s in splits[1:]]
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # 如果单个split就超过chunk_size，需要进一步分割
            if len(split) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # 递归分割大的split
                sub_chunks = self._split_text_recursive(split, remaining_separators)
                chunks.extend(sub_chunks)
            else:
                # 检查是否可以添加到当前chunk
                if len(current_chunk) + len(split) <= self.chunk_size:
                    current_chunk += split
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = split
        
        # 添加最后一个chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """合并过小的文本块"""
        if not chunks:
            return chunks
        
        merged_chunks = []
        current_chunk = chunks[0]
        
        for i in range(1, len(chunks)):
            next_chunk = chunks[i]
            
            # 如果当前块太小，尝试与下一块合并
            if (len(current_chunk) < self.chunk_size // 2 and 
                len(current_chunk) + len(next_chunk) <= self.chunk_size):
                current_chunk += " " + next_chunk
            else:
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk
        
        # 添加最后一个chunk
        merged_chunks.append(current_chunk)
        
        return merged_chunks
```

### 🧪 测试验证

#### 1.3 测试用例

```python
# test_document_processor.py
import pytest
import tempfile
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.services.document_processor import DocumentProcessor
from src.models.database import Base

@pytest.fixture
def db_session():
    """创建测试数据库会话"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()

@pytest.fixture
def document_processor():
    """创建文档处理器实例"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield DocumentProcessor(upload_dir=temp_dir)

@pytest.mark.asyncio
async def test_process_pdf_document(document_processor, db_session):
    """测试PDF文档处理"""
    # 创建测试PDF内容
    test_content = b"Test PDF content"
    filename = "test.pdf"
    
    # 处理文档
    document = await document_processor.process_document(
        file_content=test_content,
        filename=filename,
        title="测试文档",
        description="这是一个测试文档",
        db=db_session
    )
    
    # 验证结果
    assert document.title == "测试文档"
    assert document.filename == filename
    assert document.status == "completed"
    assert document.chunk_count > 0

@pytest.mark.asyncio
async def test_text_splitting(document_processor):
    """测试文本分割功能"""
    long_text = "这是一个很长的文本。" * 100
    
    chunks = document_processor.text_splitter.split_text(long_text)
    
    # 验证分割结果
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= document_processor.text_splitter.chunk_size
```

### 💡 关键技术点解析

1. **文件类型验证**：通过文件扩展名判断文件类型，确保只处理支持的格式
2. **文件大小限制**：防止过大文件影响系统性能
3. **智能文本分块**：按语义边界分割，保持文本完整性
4. **异常处理**：完善的错误处理和状态管理
5. **数据库事务**：确保数据一致性

### 🔍 常见问题和解决方案

**Q1: PDF解析失败怎么办？**
- 检查PDF文件是否损坏
- 确认PDF不是扫描版（需要OCR）
- 尝试使用其他PDF解析库

**Q2: 文本分块效果不好？**
- 调整chunk_size和chunk_overlap参数
- 优化分隔符优先级
- 针对特定文档类型定制分割策略

---

## 🔢 任务2：文档向量化

### 🎯 任务目标

将文档文本转换为向量表示，支持语义搜索和相似度计算。

### 🔧 核心实现

#### 2.1 嵌入服务 (`src/services/embedding_service.py`)

```python
import asyncio
import numpy as np
from typing import List, Optional, Union
import logging
from sentence_transformers import SentenceTransformer
import torch
from concurrent.futures import ThreadPoolExecutor

from ..config.settings import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

class EmbeddingService:
    """嵌入向量生成服务"""
    
    def __init__(self):
        self.model = None
        self.model_name = settings.LOCAL_EMBEDDING_MODEL
        self.dimension = settings.LOCAL_EMBEDDING_DIMENSION
        self.batch_size = getattr(settings, 'EMBEDDING_BATCH_SIZE', 32)
        self.max_length = getattr(settings, 'MAX_TEXT_LENGTH', 512)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self):
        """初始化嵌入模型"""
        if self.model is None:
            try:
                logger.info(f"正在加载嵌入模型: {self.model_name}")
                
                # TODO(lab01-task2): 实现模型初始化逻辑
                # 在线程池中加载模型，避免阻塞主线程
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(
                    self.executor, 
                    self._load_model
                )
                
                logger.info(f"嵌入模型加载完成: {self.model_name}")
                
                # 验证模型维度
                test_embedding = await self.embed_text("测试文本")
                actual_dim = len(test_embedding)
                
                if actual_dim != self.dimension:
                    logger.warning(
                        f"模型维度不匹配: 期望 {self.dimension}, 实际 {actual_dim}"
                    )
                    self.dimension = actual_dim
                
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}")
                raise
    
    def _load_model(self) -> SentenceTransformer:
        """在线程池中加载模型"""
        # 设置设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备: {device}")
        
        # 加载模型
        model = SentenceTransformer(self.model_name, device=device)
        
        # 设置模型为评估模式
        model.eval()
        
        return model
    
    async def embed_text(self, text: str) -> List[float]:
        """
        生成单个文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 嵌入向量
        """
        if not self.model:
            await self.initialize()
        
        # TODO(lab01-task2): 实现单文本向量化
        try:
            # 文本预处理
            processed_text = self._preprocess_text(text)
            
            # 在线程池中生成嵌入
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self.executor,
                self._generate_embedding,
                processed_text
            )
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"文本向量化失败: {str(e)}")
            raise
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文本嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not self.model:
            await self.initialize()
        
        if not texts:
            return []
        
        # TODO(lab01-task2): 实现批量向量化
        try:
            # 文本预处理
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # 分批处理
            all_embeddings = []
            for i in range(0, len(processed_texts), self.batch_size):
                batch_texts = processed_texts[i:i + self.batch_size]
                
                # 在线程池中生成嵌入
                loop = asyncio.get_event_loop()
                batch_embeddings = await loop.run_in_executor(
                    self.executor,
                    self._generate_batch_embeddings,
                    batch_texts
                )
                
                all_embeddings.extend(batch_embeddings)
                
                # 记录进度
                logger.info(f"批量向量化进度: {min(i + self.batch_size, len(texts))}/{len(texts)}")
            
            return [embedding.tolist() for embedding in all_embeddings]
            
        except Exception as e:
            logger.error(f"批量向量化失败: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if not text or not text.strip():
            return ""
        
        # 去除多余空白字符
        text = ' '.join(text.split())
        
        # 截断过长文本
        if len(text) > self.max_length:
            text = text[:self.max_length]
            logger.warning(f"文本被截断到 {self.max_length} 字符")
        
        return text
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """生成单个文本的嵌入向量"""
        with torch.no_grad():
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
    
    def _generate_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """批量生成嵌入向量"""
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return [embedding for embedding in embeddings]
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.dimension
    
    def is_initialized(self) -> bool:
        """检查模型是否已初始化"""
        return self.model is not None
```

#### 2.2 向量服务 (`src/services/vector_service.py`)

```python
import asyncio
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

from ..models.chunk import Chunk
from ..services.embedding_service import EmbeddingService
from ..services.vector_store import VectorStore
from ..utils.logger import get_logger

logger = get_logger(__name__)

class VectorService:
    """向量化服务 - 管理文档向量化和存储"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        
    async def initialize(self):
        """初始化服务"""
        await self.embedding_service.initialize()
        await self.vector_store.initialize()
    
    async def vectorize_document(
        self, 
        document_id: str, 
        db: Session
    ) -> bool:
        """
        向量化文档的所有文本块
        
        Args:
            document_id: 文档ID
            db: 数据库会话
            
        Returns:
            bool: 向量化是否成功
        """
        try:
            # TODO(lab01-task2): 实现文档向量化逻辑
            # 1. 获取文档的所有文本块
            chunks = db.query(Chunk).filter(
                Chunk.document_id == document_id,
                Chunk.is_vector_stored == False
            ).all()
            
            if not chunks:
                logger.info(f"文档 {document_id} 没有需要向量化的文本块")
                return True
            
            # 2. 提取文本内容
            texts = [chunk.content for chunk in chunks]
            chunk_ids = [chunk.id for chunk in chunks]
            
            logger.info(f"开始向量化文档 {document_id}, 共 {len(texts)} 个文本块")
            
            # 3. 批量生成嵌入向量
            embeddings = await self.embedding_service.embed_texts(texts)
            
            if len(embeddings) != len(chunks):
                raise ValueError(f"向量数量不匹配: {len(embeddings)} vs {len(chunks)}")
            
            # 4. 存储向量到向量数据库
            success = await self.vector_store.add_vectors(
                vectors=embeddings,
                chunk_ids=chunk_ids,
                metadata=[{
                    'document_id': chunk.document_id,
                    'chunk_index': chunk.chunk_index,
                    'content': chunk.content[:200]  # 存储前200字符作为元数据
                } for chunk in chunks]
            )
            
            if not success:
                raise RuntimeError("向量存储失败")
            
            # 5. 更新数据库中的向量化状态
            for chunk in chunks:
                chunk.is_vector_stored = True
                chunk.vector_dimension = self.embedding_service.get_dimension()
            
            db.commit()
            
            logger.info(f"文档 {document_id} 向量化完成")
            return True
            
        except Exception as e:
            logger.error(f"文档向量化失败: {document_id}, 错误: {str(e)}")
            db.rollback()
            return False
    
    async def vectorize_text(self, text: str) -> Optional[List[float]]:
        """
        向量化单个文本
        
        Args:
            text: 输入文本
            
        Returns:
            Optional[List[float]]: 嵌入向量，失败时返回None
        """
        try:
            # TODO(lab01-task2): 实现单文本向量化
            if not text.strip():
                return None
            
            embedding = await self.embedding_service.embed_text(text)
            return embedding
            
        except Exception as e:
            logger.error(f"文本向量化失败: {str(e)}")
            return None
    
    async def search_similar_chunks(
        self, 
        query_text: str, 
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似的文本块
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            document_ids: 限制搜索的文档ID列表
            
        Returns:
            List[Dict[str, Any]]: 搜索结果列表
        """
        try:
            # TODO(lab01-task2): 实现相似度搜索
            # 1. 向量化查询文本
            query_embedding = await self.vectorize_text(query_text)
            if not query_embedding:
                return []
            
            # 2. 在向量数据库中搜索
            search_results = await self.vector_store.search(
                query_vector=query_embedding,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                filter_metadata={'document_id': document_ids} if document_ids else None
            )
            
            return search_results
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {str(e)}")
            return []
    
    async def delete_document_vectors(self, document_id: str, db: Session) -> bool:
        """
        删除文档的所有向量
        
        Args:
            document_id: 文档ID
            db: 数据库会话
            
        Returns:
            bool: 删除是否成功
        """
        try:
            # 获取文档的所有chunk ID
            chunks = db.query(Chunk).filter(
                Chunk.document_id == document_id
            ).all()
            
            chunk_ids = [chunk.id for chunk in chunks]
            
            # 从向量数据库中删除
            success = await self.vector_store.delete_vectors(chunk_ids)
            
            if success:
                # 更新数据库状态
                for chunk in chunks:
                    chunk.is_vector_stored = False
                db.commit()
            
            return success
            
        except Exception as e:
            logger.error(f"删除文档向量失败: {document_id}, 错误: {str(e)}")
            return False
    
    async def get_vector_stats(self) -> Dict[str, Any]:
        """获取向量化统计信息"""
        try:
            stats = await self.vector_store.get_collection_info()
            return {
                'total_vectors': stats.get('vectors_count', 0),
                'dimension': self.embedding_service.get_dimension(),
                'model_name': self.embedding_service.model_name,
                'is_initialized': self.embedding_service.is_initialized()
            }
        except Exception as e:
            logger.error(f"获取向量统计失败: {str(e)}")
            return {}
```

### 🧪 测试验证

#### 2.3 测试用例

```python
# test_embedding_service.py
import pytest
import asyncio

from src.services.embedding_service import EmbeddingService

@pytest.fixture
async def embedding_service():
    """创建嵌入服务实例"""
    service = EmbeddingService()
    await service.initialize()
    return service

@pytest.mark.asyncio
async def test_embed_single_text(embedding_service):
    """测试单文本向量化"""
    text = "这是一个测试文本"
    
    embedding = await embedding_service.embed_text(text)
    
    # 验证结果
    assert isinstance(embedding, list)
    assert len(embedding) == embedding_service.dimension
    assert all(isinstance(x, float) for x in embedding)

@pytest.mark.asyncio
async def test_embed_batch_texts(embedding_service):
    """测试批量文本向量化"""
    texts = [
        "第一个测试文本",
        "第二个测试文本",
        "第三个测试文本"
    ]
    
    embeddings = await embedding_service.embed_texts(texts)
    
    # 验证结果
    assert len(embeddings) == len(texts)
    for embedding in embeddings:
        assert len(embedding) == embedding_service.dimension

@pytest.mark.asyncio
async def test_similarity_calculation():
    """测试相似度计算"""
    service = EmbeddingService()
    await service.initialize()
    
    # 相似文本
    text1 = "人工智能是计算机科学的一个分支"
    text2 = "AI是计算机科学的重要领域"
    
    # 不相似文本
    text3 = "今天天气很好"
    
    embedding1 = await service.embed_text(text1)
    embedding2 = await service.embed_text(text2)
    embedding3 = await service.embed_text(text3)
    
    # 计算余弦相似度
    import numpy as np
    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    sim_12 = cosine_similarity(embedding1, embedding2)
    sim_13 = cosine_similarity(embedding1, embedding3)
    
    # 相似文本的相似度应该更高
    assert sim_12 > sim_13
    assert sim_12 > 0.5  # 相似文本相似度应该较高
```

### 💡 关键技术点解析

1. **异步处理**：使用线程池避免模型推理阻塞主线程
2. **批量处理**：提高向量化效率，减少模型调用次数
3. **内存管理**：合理控制批处理大小，避免内存溢出
4. **错误处理**：完善的异常处理和重试机制
5. **性能优化**：模型预热、设备选择、推理优化

### 🔍 常见问题和解决方案

**Q1: 模型加载慢怎么办？**
- 使用模型缓存
- 选择更小的模型
- 预热模型

**Q2: 向量化内存不足？**
- 减小批处理大小
- 使用CPU推理
- 清理不必要的变量

---

## 🔍 任务3：语义搜索实现

### 🎯 任务目标

实现基于向量相似度的语义搜索功能，支持多种搜索策略和结果排序。

### 🔧 核心实现

#### 3.1 向量存储服务 (`src/services/vector_store.py`)

```python
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from ..config.settings import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

class VectorStore:
    """向量存储服务 - 基于Qdrant实现"""
    
    def __init__(self):
        self.client = None
        self.collection_name = "documents"
        self.qdrant_url = settings.QDRANT_URL
        
    async def initialize(self):
        """初始化Qdrant客户端和集合"""
        try:
            # TODO(lab01-task3): 实现向量数据库初始化
            # 1. 创建Qdrant客户端
            self.client = QdrantClient(url=self.qdrant_url)
            
            # 2. 检查集合是否存在
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # 3. 创建集合
                await self._create_collection()
                logger.info(f"创建向量集合: {self.collection_name}")
            else:
                logger.info(f"向量集合已存在: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"向量数据库初始化失败: {str(e)}")
            raise
    
    async def _create_collection(self):
        """创建向量集合"""
        # TODO(lab01-task3): 实现集合创建逻辑
        # 获取向量维度（从配置或默认值）
        dimension = getattr(settings, 'LOCAL_EMBEDDING_DIMENSION', 384)
        
        # 创建集合配置
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE  # 使用余弦距离
            )
        )
    
    async def add_vectors(
        self, 
        vectors: List[List[float]], 
        chunk_ids: List[str],
        metadata: List[Dict[str, Any]]
    ) -> bool:
        """
        添加向量到存储
        
        Args:
            vectors: 向量列表
            chunk_ids: 对应的chunk ID列表
            metadata: 元数据列表
            
        Returns:
            bool: 添加是否成功
        """
        try:
            # TODO(lab01-task3): 实现向量添加逻辑
            if len(vectors) != len(chunk_ids) or len(vectors) != len(metadata):
                raise ValueError("向量、ID和元数据数量不匹配")
            
            # 构建点数据
            points = []
            for i, (vector, chunk_id, meta) in enumerate(zip(vectors, chunk_ids, metadata)):
                point = PointStruct(
                    id=chunk_id,  # 使用chunk_id作为点ID
                    vector=vector,
                    payload=meta  # 存储元数据
                )
                points.append(point)
            
            # 批量插入向量
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"成功添加 {len(points)} 个向量到集合")
            return True
            
        except Exception as e:
            logger.error(f"向量添加失败: {str(e)}")
            return False
    
    async def search(
        self, 
        query_vector: List[float], 
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        向量相似度搜索
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            filter_metadata: 过滤条件
            
        Returns:
            List[Dict[str, Any]]: 搜索结果
        """
        try:
            # TODO(lab01-task3): 实现向量搜索逻辑
            # 构建搜索过滤器
            search_filter = None
            if filter_metadata:
                conditions = []
                for key, value in filter_metadata.items():
                    if isinstance(value, list):
                        # 多值匹配
                        conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchAny(any=value)
                            )
                        )
                    else:
                        # 单值匹配
                        conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )
                
                if conditions:
                    search_filter = models.Filter(must=conditions)
            
            # 执行搜索
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False  # 不返回向量数据，节省带宽
            )
            
            # 处理搜索结果
            results = []
            for result in search_results:
                # 检查相似度阈值
                if result.score < similarity_threshold:
                    continue
                
                result_dict = {
                    'chunk_id': result.id,
                    'similarity_score': result.score,
                    'metadata': result.payload
                }
                results.append(result_dict)
            
            logger.info(f"搜索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
            return []
    
    async def delete_vectors(self, chunk_ids: List[str]) -> bool:
        """
        删除指定的向量
        
        Args:
            chunk_ids: 要删除的chunk ID列表
            
        Returns:
            bool: 删除是否成功
        """
        try:
            # TODO(lab01-task3): 实现向量删除逻辑
            if not chunk_ids:
                return True
            
            # 批量删除向量
            operation_info = self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=chunk_ids
                )
            )
            
            logger.info(f"成功删除 {len(chunk_ids)} 个向量")
            return True
            
        except Exception as e:
            logger.error(f"向量删除失败: {str(e)}")
            return False
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        try:
            # TODO(lab01-task3): 实现集合信息获取
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                'vectors_count': collection_info.vectors_count,
                'indexed_vectors_count': collection_info.indexed_vectors_count,
                'points_count': collection_info.points_count,
                'status': collection_info.status,
                'optimizer_status': collection_info.optimizer_status
            }
            
        except Exception as e:
            logger.error(f"获取集合信息失败: {str(e)}")
            return {}
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            # 检查客户端连接
            collections = self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"向量数据库健康检查失败: {str(e)}")
            return False
```

#### 3.2 搜索API路由 (`src/api/routes/vectors.py`)

```python
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from sqlalchemy.orm import Session

from ...api.dependencies import get_db
from ...api.schemas import SearchRequest, SearchResponse, SearchResult
from ...services.vector_service import VectorService
from ...models.chunk import Chunk
from ...utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# 全局向量服务实例
vector_service = VectorService()

@router.on_event("startup")
async def startup_event():
    """启动时初始化向量服务"""
    await vector_service.initialize()

@router.post("/search", response_model=SearchResponse)
async def search_vectors(
    request: SearchRequest,
    db: Session = Depends(get_db)
):
    """
    向量语义搜索
    
    Args:
        request: 搜索请求
        db: 数据库会话
        
    Returns:
        SearchResponse: 搜索结果
    """
    try:
        # TODO(lab01-task3): 实现搜索API逻辑
        # 1. 参数验证
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="查询文本不能为空")
        
        if request.top_k <= 0 or request.top_k > 50:
            raise HTTPException(status_code=400, detail="top_k必须在1-50之间")
        
        # 2. 执行向量搜索
        search_results = await vector_service.search_similar_chunks(
            query_text=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            document_ids=request.document_ids
        )
        
        # 3. 获取详细的chunk信息
        results = []
        for result in search_results:
            chunk_id = result['chunk_id']
            
            # 从数据库获取完整的chunk信息
            chunk = db.query(Chunk).filter(Chunk.id == chunk_id).first()
            if not chunk:
                continue
            
            search_result = SearchResult(
                chunk_id=chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                similarity_score=result['similarity_score'],
                metadata={
                    'chunk_index': chunk.chunk_index,
                    'token_count': chunk.token_count,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char
                }
            )
            results.append(search_result)
        
        # 4. 构建响应
        response = SearchResponse(
            results=results,
            total_results=len(results),
            processing_time=0.0  # 可以添加实际的处理时间统计
        )
        
        logger.info(f"搜索完成: 查询='{request.query}', 结果数={len(results)}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail="搜索服务内部错误")

@router.get("/search", response_model=SearchResponse)
async def search_vectors_get(
    query: str = Query(..., description="搜索查询文本"),
    top_k: int = Query(5, ge=1, le=50, description="返回结果数量"),
    similarity_threshold: float = Query(0.7, ge=0.0, le=1.0, description="相似度阈值"),
    document_ids: Optional[str] = Query(None, description="文档ID列表，逗号分隔"),
    db: Session = Depends(get_db)
):
    """
    GET方式的向量搜索（便于测试）
    """
    # TODO(lab01-task3): 实现GET搜索接口
    # 解析document_ids
    doc_ids = None
    if document_ids:
        doc_ids = [doc_id.strip() for doc_id in document_ids.split(',') if doc_id.strip()]
    
    # 构建搜索请求
    request = SearchRequest(
        query=query,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        document_ids=doc_ids
    )
    
    # 调用POST接口逻辑
    return await search_vectors(request, db)

@router.post("/vectorize/{document_id}")
async def vectorize_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    手动触发文档向量化
    
    Args:
        document_id: 文档ID
        db: 数据库会话
        
    Returns:
        dict: 向量化结果
    """
    try:
        # TODO(lab01-task3): 实现手动向量化接口
        # 检查文档是否存在
        from ...models.document import Document
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="文档不存在")
        
        # 执行向量化
        success = await vector_service.vectorize_document(document_id, db)
        
        if success:
            return {
                "message": f"文档 {document_id} 向量化成功",
                "document_id": document_id,
                "status": "completed"
            }
        else:
            raise HTTPException(status_code=500, detail="向量化失败")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"向量化失败: {str(e)}")
        raise HTTPException(status_code=500, detail="向量化服务内部错误")

@router.get("/stats")
async def get_vector_stats():
    """获取向量化统计信息"""
    try:
        # TODO(lab01-task3): 实现统计信息接口
        stats = await vector_service.get_vector_stats()
        return stats
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取统计信息失败")
```

### 🧪 测试验证

#### 3.3 测试用例

```python
# test_vector_search.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.main import app
from src.models.database import Base
from src.api.dependencies import get_db

# 创建测试数据库
engine = create_engine("sqlite:///:memory:")
Base.metadata.create_all(engine)
TestingSessionLocal = sessionmaker(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

def test_search_vectors():
    """测试向量搜索API"""
    # 准备搜索请求
    search_data = {
        "query": "人工智能",
        "top_k": 5,
        "similarity_threshold": 0.7
    }
    
    # 发送搜索请求
    response = client.post("/api/v1/search", json=search_data)
    
    # 验证响应
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "total_results" in data
    assert isinstance(data["results"], list)

def test_search_with_filters():
    """测试带过滤条件的搜索"""
    search_data = {
        "query": "机器学习",
        "top_k": 3,
        "similarity_threshold": 0.6,
        "document_ids": ["doc_123", "doc_456"]
    }
    
    response = client.post("/api/v1/search", json=search_data)
    assert response.status_code == 200

def test_get_search():
    """测试GET方式搜索"""
    response = client.get(
        "/api/v1/search",
        params={
            "query": "深度学习",
            "top_k": 3,
            "similarity_threshold": 0.8
        }
    )
    assert response.status_code == 200

def test_vector_stats():
    """测试向量统计信息"""
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    
    data = response.json()
    expected_keys = ["total_vectors", "dimension", "model_name", "is_initialized"]
    for key in expected_keys:
        assert key in data
```

### 💡 关键技术点解析

1. **向量数据库选择**：Qdrant提供高性能的向量搜索能力
2. **相似度计算**：余弦相似度适合文本向量比较
3. **搜索优化**：支持过滤条件和阈值设置
4. **批量操作**：提高向量插入和删除效率
5. **API设计**：RESTful接口，支持多种查询方式

### 🔍 常见问题和解决方案

**Q1: 搜索结果不准确？**
- 调整相似度阈值
- 优化文本预处理
- 使用更好的嵌入模型

**Q2: 搜索速度慢？**
- 建立向量索引
- 减少返回结果数量
- 优化向量维度

---

## 🤖 任务4：RAG问答系统

### 🎯 任务目标

实现检索增强生成（RAG）问答系统，结合语义搜索和大语言模型生成准确回答。

### 🔧 核心实现

#### 4.1 问答服务 (`src/services/qa_service.py`)

```python
import asyncio
import time
from typing import List, Dict, Any, Optional
import httpx
from sqlalchemy.orm import Session

from ..config.settings import get_settings
from ..services.vector_service import VectorService
from ..models.chunk import Chunk
from ..utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

class QAService:
    """问答服务 - 实现RAG问答逻辑"""
    
    def __init__(self):
        self.vector_service = VectorService()
        self.llm_client = None
        
        # LLM配置
        self.api_key = getattr(settings, 'VOLCENGINE_API_KEY', '')
        self.base_url = getattr(settings, 'VOLCENGINE_BASE_URL', '')
        self.model_name = getattr(settings, 'VOLCENGINE_MODEL', 'doubao-seed-1-6-250615')
        
        # 问答配置
        self.max_context_length = getattr(settings, 'MAX_CONTEXT_LENGTH', 4000)
        self.max_tokens = getattr(settings, 'MAX_TOKENS', 1000)
        self.temperature = getattr(settings, 'TEMPERATURE', 0.7)
        
    async def initialize(self):
        """初始化问答服务"""
        await self.vector_service.initialize()
        
        # 初始化HTTP客户端
        self.llm_client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            },
            timeout=30.0
        )
    
    async def answer_question(
        self,
        question: str,
        document_ids: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        db: Session = None
    ) -> Dict[str, Any]:
        """
        回答问题
        
        Args:
            question: 用户问题
            document_ids: 限制搜索的文档ID列表
            max_tokens: 最大生成token数
            temperature: 生成温度
            db: 数据库会话
            
        Returns:
            Dict[str, Any]: 问答结果
        """
        start_time = time.time()
        
        try:
            # TODO(lab01-task4): 实现RAG问答逻辑
            # 1. 参数验证和预处理
            if not question.strip():
                raise ValueError("问题不能为空")
            
            question = question.strip()
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature
            
            logger.info(f"开始处理问题: {question}")
            
            # 2. 检索相关文档片段
            relevant_chunks = await self._retrieve_relevant_chunks(
                question, 
                document_ids,
                top_k=5,
                similarity_threshold=0.6
            )
            
            if not relevant_chunks:
                return {
                    'answer': '抱歉，我没有找到相关的信息来回答您的问题。',
                    'sources': [],
                    'processing_time': time.time() - start_time,
                    'model_used': self.model_name
                }
            
            # 3. 构建上下文
            context = self._build_context(relevant_chunks)
            
            # 4. 生成提示词
            prompt = self._build_prompt(question, context)
            
            # 5. 调用LLM生成回答
            answer = await self._generate_answer(
                prompt, 
                max_tokens, 
                temperature
            )
            
            # 6. 后处理和验证
            answer = self._post_process_answer(answer, question)
            
            # 7. 构建响应
            processing_time = time.time() - start_time
            
            response = {
                'answer': answer,
                'sources': self._format_sources(relevant_chunks, db),
                'processing_time': processing_time,
                'model_used': self.model_name,
                'context_length': len(context),
                'chunks_used': len(relevant_chunks)
            }
            
            logger.info(f"问答完成: 耗时 {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"问答失败: {str(e)}")
            return {
                'answer': f'抱歉，处理您的问题时出现了错误: {str(e)}',
                'sources': [],
                'processing_time': time.time() - start_time,
                'model_used': self.model_name,
                'error': str(e)
            }
    
    async def _retrieve_relevant_chunks(
        self,
        question: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """检索相关文档片段"""
        # TODO(lab01-task4): 实现文档检索逻辑
        try:
            # 使用向量服务搜索相关片段
            search_results = await self.vector_service.search_similar_chunks(
                query_text=question,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                document_ids=document_ids
            )
            
            # 按相似度排序
            search_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return search_results
            
        except Exception as e:
            logger.error(f"文档检索失败: {str(e)}")
            return []
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """构建上下文文本"""
        # TODO(lab01-task4): 实现上下文构建逻辑
        if not chunks:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks):
            # 获取chunk内容
            content = chunk.get('metadata', {}).get('content', '')
            if not content:
                continue
            
            # 检查长度限制
            if current_length + len(content) > self.max_context_length:
                break
            
            # 添加到上下文
            context_parts.append(f"[文档片段 {i+1}]\n{content}\n")
            current_length += len(content)
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """构建LLM提示词"""
        # TODO(lab01-task4): 实现提示词构建逻辑
        if not context:
            prompt = f"""请回答以下问题：

问题：{question}

请基于您的知识回答问题。如果不确定答案，请诚实地说明。"""
        else:
            prompt = f"""请基于以下文档内容回答问题：

文档内容：
{context}

问题：{question}

请仔细阅读文档内容，并基于文档信息回答问题。如果文档中没有相关信息，请明确说明。回答要准确、简洁、有条理。"""
        
        return prompt
    
    async def _generate_answer(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float
    ) -> str:
        """调用LLM生成回答"""
        # TODO(lab01-task4): 实现LLM调用逻辑
        try:
            # 构建请求数据
            request_data = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            # 发送请求
            response = await self.llm_client.post(
                "/v1/chat/completions",
                json=request_data
            )
            
            if response.status_code != 200:
                raise Exception(f"LLM API调用失败: {response.status_code}")
            
            # 解析响应
            result = response.json()
            
            if 'choices' not in result or not result['choices']:
                raise Exception("LLM响应格式错误")
            
            answer = result['choices'][0]['message']['content']
            return answer.strip()
            
        except Exception as e:
            logger.error(f"LLM生成失败: {str(e)}")
            raise
    
    def _post_process_answer(self, answer: str, question: str) -> str:
        """后处理生成的回答"""
        # TODO(lab01-task4): 实现回答后处理逻辑
        if not answer:
            return "抱歉，我无法生成回答。"
        
        # 去除多余空白
        answer = answer.strip()
        
        # 确保回答完整（不以句号结尾的添加句号）
        if answer and not answer.endswith(('.', '!', '?', '。', '！', '？')):
            answer += '。'
        
        return answer
    
    def _format_sources(
        self, 
        chunks: List[Dict[str, Any]], 
        db: Session
    ) -> List[Dict[str, Any]]:
        """格式化引用来源"""
        # TODO(lab01-task4): 实现来源格式化逻辑
        sources = []
        
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id')
            if not chunk_id or not db:
                continue
            
            # 从数据库获取详细信息
            chunk_obj = db.query(Chunk).filter(Chunk.id == chunk_id).first()
            if not chunk_obj:
                continue
            
            # 获取文档信息
            from ..models.document import Document
            document = db.query(Document).filter(
                Document.id == chunk_obj.document_id
            ).first()
            
            source = {
                'chunk_id': chunk_id,
                'document_id': chunk_obj.document_id,
                'document_title': document.title if document else '未知文档',
                'similarity_score': chunk.get('similarity_score', 0.0),
                'content_preview': chunk_obj.content[:200] + '...' if len(chunk_obj.content) > 200 else chunk_obj.content
            }
            sources.append(source)
        
        return sources
    
    async def batch_answer_questions(
        self,
        questions: List[str],
        document_ids: Optional[List[str]] = None,
        db: Session = None
    ) -> List[Dict[str, Any]]:
        """批量回答问题"""
        # TODO(lab01-task4): 实现批量问答逻辑
        results = []
        
        for i, question in enumerate(questions):
            try:
                logger.info(f"处理问题 {i+1}/{len(questions)}: {question}")
                
                result = await self.answer_question(
                    question=question,
                    document_ids=document_ids,
                    db=db
                )
                
                results.append({
                    'question': question,
                    'result': result,
                    'status': 'success'
                })
                
            except Exception as e:
                logger.error(f"批量问答失败: 问题 {i+1}, 错误: {str(e)}")
                results.append({
                    'question': question,
                    'result': {
                        'answer': f'处理问题时出现错误: {str(e)}',
                        'sources': [],
                        'error': str(e)
                    },
                    'status': 'error'
                })
        
        return results
    
    async def check_answer_accuracy(
        self,
        question: str,
        answer: str,
        expected_keywords: List[str] = None
    ) -> Dict[str, Any]:
        """检查回答准确性"""
        # TODO(lab01-task4): 实现准确性检查逻辑
        accuracy_score = 0.0
        feedback = []
        
        # 基本检查
        if not answer or len(answer.strip()) < 10:
            feedback.append("回答过于简短")
        else:
            accuracy_score += 0.3
        
        # 关键词检查
        if expected_keywords:
            found_keywords = []
            answer_lower = answer.lower()
            
            for keyword in expected_keywords:
                if keyword.lower() in answer_lower:
                    found_keywords.append(keyword)
            
            keyword_score = len(found_keywords) / len(expected_keywords)
            accuracy_score += keyword_score * 0.7
            
            feedback.append(f"包含关键词: {found_keywords}")
            if len(found_keywords) < len(expected_keywords):
                missing = set(expected_keywords) - set(found_keywords)
                feedback.append(f"缺少关键词: {list(missing)}")
        
        return {
             'accuracy_score': min(accuracy_score, 1.0),
             'feedback': feedback,
             'answer_length': len(answer),
             'has_sources': 'sources' in answer.lower()
         }
```

#### 4.2 问答API路由 (`src/api/routes/qa.py`)

```python
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from sqlalchemy.orm import Session

from ...api.dependencies import get_db
from ...api.schemas import QuestionRequest, QuestionResponse, BatchQuestionRequest, BatchQuestionResponse
from ...services.qa_service import QAService
from ...utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# 全局问答服务实例
qa_service = QAService()

@router.on_event("startup")
async def startup_event():
    """启动时初始化问答服务"""
    await qa_service.initialize()

@router.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    db: Session = Depends(get_db)
):
    """
    单个问题问答
    
    Args:
        request: 问题请求
        db: 数据库会话
        
    Returns:
        QuestionResponse: 问答结果
    """
    try:
        # TODO(lab01-task4): 实现问答API逻辑
        # 参数验证
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="问题不能为空")
        
        # 调用问答服务
        result = await qa_service.answer_question(
            question=request.question,
            document_ids=request.document_ids,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            db=db
        )
        
        # 构建响应
        response = QuestionResponse(
            question=request.question,
            answer=result['answer'],
            sources=result.get('sources', []),
            processing_time=result.get('processing_time', 0.0),
            model_used=result.get('model_used', ''),
            context_length=result.get('context_length', 0),
            chunks_used=result.get('chunks_used', 0)
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"问答API失败: {str(e)}")
        raise HTTPException(status_code=500, detail="问答服务内部错误")

@router.post("/batch-ask", response_model=BatchQuestionResponse)
async def batch_ask_questions(
    request: BatchQuestionRequest,
    db: Session = Depends(get_db)
):
    """
    批量问题问答
    
    Args:
        request: 批量问题请求
        db: 数据库会话
        
    Returns:
        BatchQuestionResponse: 批量问答结果
    """
    try:
        # TODO(lab01-task4): 实现批量问答API逻辑
        # 参数验证
        if not request.questions:
            raise HTTPException(status_code=400, detail="问题列表不能为空")
        
        if len(request.questions) > 10:
            raise HTTPException(status_code=400, detail="批量问题数量不能超过10个")
        
        # 调用批量问答服务
        results = await qa_service.batch_answer_questions(
            questions=request.questions,
            document_ids=request.document_ids,
            db=db
        )
        
        # 构建响应
        responses = []
        for result in results:
            if result['status'] == 'success':
                qa_result = result['result']
                response = QuestionResponse(
                    question=result['question'],
                    answer=qa_result['answer'],
                    sources=qa_result.get('sources', []),
                    processing_time=qa_result.get('processing_time', 0.0),
                    model_used=qa_result.get('model_used', ''),
                    context_length=qa_result.get('context_length', 0),
                    chunks_used=qa_result.get('chunks_used', 0)
                )
            else:
                response = QuestionResponse(
                    question=result['question'],
                    answer=result['result']['answer'],
                    sources=[],
                    processing_time=0.0,
                    model_used='',
                    error=result['result'].get('error', '')
                )
            responses.append(response)
        
        batch_response = BatchQuestionResponse(
            results=responses,
            total_questions=len(request.questions),
            successful_answers=sum(1 for r in results if r['status'] == 'success')
        )
        
        return batch_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量问答API失败: {str(e)}")
        raise HTTPException(status_code=500, detail="批量问答服务内部错误")
```

### 🧪 测试验证

#### 4.3 测试用例

```python
# test_qa_service.py
import pytest
from unittest.mock import AsyncMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.services.qa_service import QAService
from src.models.database import Base

@pytest.fixture
def db_session():
    """创建测试数据库会话"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()

@pytest.fixture
async def qa_service():
    """创建问答服务实例"""
    service = QAService()
    
    # Mock LLM客户端
    service.llm_client = AsyncMock()
    service.llm_client.post.return_value.status_code = 200
    service.llm_client.post.return_value.json.return_value = {
        'choices': [{
            'message': {
                'content': '这是一个测试回答。'
            }
        }]
    }
    
    # Mock向量服务
    service.vector_service = AsyncMock()
    service.vector_service.search_similar_chunks.return_value = [
        {
            'chunk_id': 'chunk_123',
            'similarity_score': 0.85,
            'metadata': {
                'content': '这是相关的文档内容。'
            }
        }
    ]
    
    return service

@pytest.mark.asyncio
async def test_answer_question(qa_service, db_session):
    """测试单问题回答"""
    question = "什么是人工智能？"
    
    result = await qa_service.answer_question(
        question=question,
        db=db_session
    )
    
    # 验证结果
    assert 'answer' in result
    assert 'sources' in result
    assert 'processing_time' in result
    assert result['answer'] != ''

@pytest.mark.asyncio
async def test_batch_answer_questions(qa_service, db_session):
    """测试批量问题回答"""
    questions = [
        "什么是机器学习？",
        "深度学习的应用有哪些？"
    ]
    
    results = await qa_service.batch_answer_questions(
        questions=questions,
        db=db_session
    )
    
    # 验证结果
    assert len(results) == len(questions)
    for result in results:
        assert 'question' in result
        assert 'result' in result
        assert 'status' in result

@pytest.mark.asyncio
async def test_check_answer_accuracy(qa_service):
    """测试回答准确性检查"""
    question = "什么是深度学习？"
    answer = "深度学习是机器学习的一个分支，使用神经网络进行学习。"
    keywords = ["深度学习", "机器学习", "神经网络"]
    
    accuracy = await qa_service.check_answer_accuracy(
        question=question,
        answer=answer,
        expected_keywords=keywords
    )
    
    # 验证结果
    assert 'accuracy_score' in accuracy
    assert 'feedback' in accuracy
    assert accuracy['accuracy_score'] > 0.5
```

### 💡 关键技术点解析

1. **RAG架构**：检索-增强-生成的完整流程
2. **上下文管理**：智能选择和组织相关文档片段
3. **提示工程**：设计有效的LLM提示词模板
4. **API集成**：与外部LLM服务的稳定集成
5. **质量控制**：回答后处理和准确性验证

### 🔍 常见问题和解决方案

**Q1: LLM回答不准确？**
- 优化提示词模板
- 提高检索质量
- 调整生成参数

**Q2: API调用失败？**
- 检查API密钥和配置
- 实现重试机制
- 添加降级策略

---

## 📊 任务5：系统监控和健康检查

### 🎯 任务目标

实现系统监控、健康检查和性能统计功能，确保RAG系统稳定运行。

### 🔧 核心实现

#### 5.1 健康检查服务 (`src/services/health_service.py`)

```python
import asyncio
import time
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import text

from ..config.settings import get_settings
from ..services.vector_service import VectorService
from ..services.qa_service import QAService
from ..utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

class HealthService:
    """系统健康检查服务"""
    
    def __init__(self):
        self.vector_service = VectorService()
        self.qa_service = QAService()
        
    async def check_system_health(self, db: Session) -> Dict[str, Any]:
        """
        全面的系统健康检查
        
        Args:
            db: 数据库会话
            
        Returns:
            Dict[str, Any]: 健康检查结果
        """
        start_time = time.time()
        
        # TODO(lab01-task5): 实现系统健康检查逻辑
        health_status = {
            'overall_status': 'healthy',
            'timestamp': time.time(),
            'checks': {},
            'performance': {},
            'errors': []
        }
        
        try:
            # 1. 数据库健康检查
            db_health = await self._check_database_health(db)
            health_status['checks']['database'] = db_health
            
            # 2. 向量数据库健康检查
            vector_health = await self._check_vector_store_health()
            health_status['checks']['vector_store'] = vector_health
            
            # 3. 嵌入服务健康检查
            embedding_health = await self._check_embedding_service_health()
            health_status['checks']['embedding_service'] = embedding_health
            
            # 4. LLM服务健康检查
            llm_health = await self._check_llm_service_health()
            health_status['checks']['llm_service'] = llm_health
            
            # 5. 系统资源检查
            resource_health = await self._check_system_resources()
            health_status['checks']['system_resources'] = resource_health
            
            # 6. 计算总体状态
            failed_checks = [
                name for name, check in health_status['checks'].items()
                if not check.get('status', False)
            ]
            
            if failed_checks:
                health_status['overall_status'] = 'unhealthy'
                health_status['errors'] = [
                    f"{check}服务异常" for check in failed_checks
                ]
            elif any(
                check.get('warning', False) 
                for check in health_status['checks'].values()
            ):
                health_status['overall_status'] = 'warning'
            
            # 7. 性能统计
            health_status['performance'] = {
                'check_duration': time.time() - start_time,
                'total_checks': len(health_status['checks']),
                'passed_checks': len(health_status['checks']) - len(failed_checks),
                'failed_checks': len(failed_checks)
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"系统健康检查失败: {str(e)}")
            health_status['overall_status'] = 'error'
            health_status['errors'].append(f"健康检查异常: {str(e)}")
            return health_status
    
    async def _check_database_health(self, db: Session) -> Dict[str, Any]:
        """检查数据库健康状态"""
        # TODO(lab01-task5): 实现数据库健康检查
        try:
            start_time = time.time()
            
            # 执行简单查询测试连接
            result = db.execute(text("SELECT 1"))
            result.fetchone()
            
            # 检查表是否存在
            from ..models.document import Document
            from ..models.chunk import Chunk
            
            doc_count = db.query(Document).count()
            chunk_count = db.query(Chunk).count()
            
            response_time = time.time() - start_time
            
            return {
                'status': True,
                'response_time': response_time,
                'document_count': doc_count,
                'chunk_count': chunk_count,
                'warning': response_time > 1.0  # 响应时间超过1秒警告
            }
            
        except Exception as e:
            logger.error(f"数据库健康检查失败: {str(e)}")
            return {
                'status': False,
                'error': str(e),
                'response_time': 0
            }
    
    async def _check_vector_store_health(self) -> Dict[str, Any]:
        """检查向量数据库健康状态"""
        # TODO(lab01-task5): 实现向量数据库健康检查
        try:
            start_time = time.time()
            
            # 检查向量存储连接
            health_ok = await self.vector_service.vector_store.health_check()
            
            if health_ok:
                # 获取集合信息
                collection_info = await self.vector_service.vector_store.get_collection_info()
                response_time = time.time() - start_time
                
                return {
                    'status': True,
                    'response_time': response_time,
                    'vectors_count': collection_info.get('vectors_count', 0),
                    'collection_status': collection_info.get('status', 'unknown'),
                    'warning': response_time > 2.0
                }
            else:
                return {
                    'status': False,
                    'error': '向量数据库连接失败',
                    'response_time': time.time() - start_time
                }
                
        except Exception as e:
            logger.error(f"向量数据库健康检查失败: {str(e)}")
            return {
                'status': False,
                'error': str(e),
                'response_time': 0
            }
    
    async def _check_embedding_service_health(self) -> Dict[str, Any]:
        """检查嵌入服务健康状态"""
        # TODO(lab01-task5): 实现嵌入服务健康检查
        try:
            start_time = time.time()
            
            # 测试嵌入生成
            test_text = "健康检查测试文本"
            embedding = await self.vector_service.embedding_service.embed_text(test_text)
            
            response_time = time.time() - start_time
            
            if embedding and len(embedding) > 0:
                return {
                    'status': True,
                    'response_time': response_time,
                    'model_name': self.vector_service.embedding_service.model_name,
                    'dimension': len(embedding),
                    'is_initialized': self.vector_service.embedding_service.is_initialized(),
                    'warning': response_time > 5.0  # 嵌入生成超过5秒警告
                }
            else:
                return {
                    'status': False,
                    'error': '嵌入生成失败',
                    'response_time': response_time
                }
                
        except Exception as e:
            logger.error(f"嵌入服务健康检查失败: {str(e)}")
            return {
                'status': False,
                'error': str(e),
                'response_time': 0
            }
    
    async def _check_llm_service_health(self) -> Dict[str, Any]:
        """检查LLM服务健康状态"""
        # TODO(lab01-task5): 实现LLM服务健康检查
        try:
            start_time = time.time()
            
            # 测试LLM调用
            test_prompt = "请回答：1+1等于多少？"
            
            if not self.qa_service.llm_client:
                await self.qa_service.initialize()
            
            # 发送测试请求
            response = await self.qa_service.llm_client.post(
                "/v1/chat/completions",
                json={
                    "model": self.qa_service.model_name,
                    "messages": [{"role": "user", "content": test_prompt}],
                    "max_tokens": 10,
                    "temperature": 0.1
                }
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and result['choices']:
                    return {
                        'status': True,
                        'response_time': response_time,
                        'model_name': self.qa_service.model_name,
                        'api_status': 'connected',
                        'warning': response_time > 10.0  # LLM调用超过10秒警告
                    }
            
            return {
                'status': False,
                'error': f'LLM API返回错误: {response.status_code}',
                'response_time': response_time
            }
            
        except Exception as e:
            logger.error(f"LLM服务健康检查失败: {str(e)}")
            return {
                'status': False,
                'error': str(e),
                'response_time': 0
            }
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """检查系统资源状态"""
        # TODO(lab01-task5): 实现系统资源检查
        try:
            import psutil
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # 判断警告状态
            warning = (
                cpu_percent > 80 or 
                memory_percent > 85 or 
                disk_percent > 90
            )
            
            return {
                'status': True,
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_percent': disk_percent,
                'warning': warning,
                'details': {
                    'memory_total': memory.total,
                    'memory_available': memory.available,
                    'disk_total': disk.total,
                    'disk_free': disk.free
                }
            }
            
        except ImportError:
            # psutil未安装时的简化检查
            return {
                'status': True,
                'message': 'psutil未安装，无法获取详细资源信息',
                'warning': False
            }
        except Exception as e:
            logger.error(f"系统资源检查失败: {str(e)}")
            return {
                'status': False,
                'error': str(e)
            }
    
    async def get_system_stats(self, db: Session) -> Dict[str, Any]:
        """获取系统统计信息"""
        # TODO(lab01-task5): 实现系统统计信息获取
        try:
            from ..models.document import Document
            from ..models.chunk import Chunk
            
            # 数据库统计
            total_documents = db.query(Document).count()
            total_chunks = db.query(Chunk).count()
            vectorized_chunks = db.query(Chunk).filter(
                Chunk.is_vector_stored == True
            ).count()
            
            # 向量数据库统计
            vector_stats = await self.vector_service.get_vector_stats()
            
            # 系统运行时间（简化实现）
            import psutil
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time
            
            return {
                'database_stats': {
                    'total_documents': total_documents,
                    'total_chunks': total_chunks,
                    'vectorized_chunks': vectorized_chunks,
                    'vectorization_rate': vectorized_chunks / total_chunks if total_chunks > 0 else 0
                },
                'vector_stats': vector_stats,
                'system_stats': {
                    'uptime_seconds': uptime,
                    'uptime_hours': uptime / 3600,
                    'timestamp': time.time()
                }
            }
            
        except Exception as e:
            logger.error(f"获取系统统计失败: {str(e)}")
            return {
                'error': str(e),
                'timestamp': time.time()
            }