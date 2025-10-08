"""
文档处理服务
整合PDF解析和文本分块功能，提供完整的文档处理流程
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid

from sqlalchemy.orm import Session

from src.models.document import Document
from src.models.chunk import Chunk
from src.services.pdf_parser import PDFParser
from src.services.txt_parser import TXTParser
from src.services.text_splitter import TextSplitter, SplitStrategy, TextChunk
from src.config.settings import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.txt_parser = TXTParser()
        self.text_splitter = TextSplitter()
        # TODO(lab03-lesson12): 支持可插拔的解析/预处理流水线。
        # - 允许通过settings选择解析器与清洗器；
        # - 为批处理场景预留并发控制参数（max_workers、batch_size）。
        
    def process_document(self, file_path: Path, db: Session, 
                        split_strategy: SplitStrategy = SplitStrategy.FIXED_SIZE) -> Tuple[Document, List[Chunk]]:
        """
        处理文档：解析PDF + 文本分块 + 数据库存储
        
        Args:
            file_path: 文档文件路径
            db: 数据库会话
            split_strategy: 分块策略
            
        Returns:
            (Document对象, Chunk对象列表)
        """
        logger.info(f"开始处理文档: {file_path}")
        
        try:
            # 1. 根据文件类型选择解析器
            file_extension = file_path.suffix.lower()
            if file_extension == '.pdf':
                parse_result = self.pdf_parser.parse_pdf(file_path)
            elif file_extension == '.txt':
                parse_result = self.txt_parser.parse_txt(file_path)
            else:
                raise ValueError(f"不支持的文件类型: {file_extension}")
            
            if not parse_result["extraction_success"]:
                raise ValueError(f"文档解析失败: {parse_result.get('error_message', '未知错误')}")
            
            # 2. 创建Document记录
            document = self._create_document_record(parse_result, db)
            
            # 3. 文本分块
            chunks_data = self.text_splitter.split_text(
                text=parse_result["cleaned_text"],
                strategy=split_strategy,
                metadata={
                    "document_id": document.id,
                    "file_name": document.filename,
                    "split_strategy": split_strategy.value
                }
            )
            # TODO(lab03-lesson12): 为长文档启用并发分块（批量句子/段落处理）。
            # - 设计分片策略，保证上下文边界不被跨线程破坏；
            # - 控制最大并发数与内存占用，避免OOM。
            
            # 4. 创建Chunk记录
            chunk_objects = self._create_chunk_records(chunks_data, document.id, db)
            
            # 5. 更新Document统计信息
            self._update_document_stats(document, chunk_objects, db)
            
            logger.info(f"文档处理完成: {file_path.name}, "
                       f"生成 {len(chunk_objects)} 个分块")
            
            return document, chunk_objects
            
        except Exception as e:
            logger.error(f"文档处理失败: {file_path}, 错误: {e}")
            # TODO(lab03-lesson12): 增加重试与降级策略。
            # - 对解析失败的页或段进行重试（指数退避）；
            # - 提供兜底清洗与简单分块，保证流程不中断；
            # 如果已创建Document记录，标记为失败状态
            if 'document' in locals():
                document.processing_status = "failed"
                document.error_message = str(e)
                db.commit()
            raise
    
    def _create_document_record(self, parse_result: Dict, db: Session) -> Document:
        """
        创建Document数据库记录
        
        Args:
            parse_result: PDF解析结果
            db: 数据库会话
            
        Returns:
            Document对象
        """
        # 根据文件扩展名确定文件类型
        file_path = Path(parse_result["file_path"])
        file_type = file_path.suffix.lower().lstrip('.')
        
        document = Document(
            filename=parse_result["file_name"],
            original_filename=parse_result["file_name"],
            file_path=parse_result["file_path"],
            file_size=parse_result["file_size"],
            file_type=file_type,
            content_hash=parse_result["content_hash"],
            content=parse_result["cleaned_text"],
            status="processing",
            total_tokens=parse_result["estimated_tokens"],
            metadata_=str(parse_result["metadata"])
        )
        
        db.add(document)
        db.commit()
        db.refresh(document)
        
        logger.info(f"创建Document记录: {document.id}")
        return document
    
    def _create_chunk_records(self, chunks_data: List[TextChunk], 
                            document_id: str, db: Session) -> List[Chunk]:
        """
        创建Chunk数据库记录
        
        Args:
            chunks_data: 分块数据列表
            document_id: 文档ID
            db: 数据库会话
            
        Returns:
            Chunk对象列表
        """
        chunk_objects = []
        
        for chunk_data in chunks_data:
            chunk = Chunk(
                document_id=document_id,
                chunk_index=chunk_data.chunk_index,
                content=chunk_data.content,
                content_hash=chunk_data.content_hash,
                start_pos=chunk_data.start_position,
                end_pos=chunk_data.end_position,
                char_count=chunk_data.char_count,
                token_count=chunk_data.estimated_tokens,
                metadata_=str(chunk_data.metadata)
            )
            
            chunk_objects.append(chunk)
        
        # 批量插入
        db.add_all(chunk_objects)
        db.commit()
        
        # 刷新对象
        for chunk in chunk_objects:
            db.refresh(chunk)
        
        logger.info(f"创建 {len(chunk_objects)} 个Chunk记录")
        return chunk_objects
    
    def _update_document_stats(self, document: Document, 
                             chunks: List[Chunk], db: Session) -> None:
        """
        更新Document统计信息
        
        Args:
            document: Document对象
            chunks: Chunk对象列表
            db: 数据库会话
        """
        document.total_chunks = len(chunks)
        document.status = "completed"
        
        # 计算分块统计
        if chunks:
            total_chunk_chars = sum(chunk.char_count for chunk in chunks)
            total_chunk_tokens = sum(chunk.token_count for chunk in chunks)
            avg_chunk_size = total_chunk_chars / len(chunks)
            
            # 更新总token数
            document.total_tokens = total_chunk_tokens
        
        db.commit()
        logger.info(f"更新Document统计信息: {document.id}")

    # TODO(lab03-lesson12): 批处理接口，占位实现，便于并发与容错改造
    def process_documents_batch(self, file_paths: List[Path], db: Session,
                                split_strategy: SplitStrategy = SplitStrategy.FIXED_SIZE) -> Dict:
        """
        批量处理文档（占位实现）

        目标：
        - 使用线程池或协程并发处理多个文档；
        - 对失败任务进行重试与错误聚合；
        - 输出结构化统计信息（成功/失败/耗时/分块数）。

        验收：
        - 提供N>50的文档批次，整体处理时间优于串行；
        - 错误不影响整体收敛，统计信息完整。
        """
        results = []
        start = datetime.utcnow()
        for fp in file_paths:
            try:
                doc, chunks = self.process_document(fp, db, split_strategy)
                results.append({
                    "file": fp.name,
                    "document_id": str(doc.id),
                    "chunks": len(chunks),
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "file": fp.name,
                    "status": "failed",
                    "error": str(e)
                })
        elapsed = (datetime.utcnow() - start).total_seconds()
        return {
            "count": len(results),
            "elapsed": elapsed,
            "results": results
        }
    
    def reprocess_document(self, document_id: str, db: Session,
                          split_strategy: SplitStrategy = None) -> Tuple[Document, List[Chunk]]:
        """
        重新处理文档
        
        Args:
            document_id: 文档ID
            db: 数据库会话
            split_strategy: 新的分块策略（可选）
            
        Returns:
            (Document对象, Chunk对象列表)
        """
        logger.info(f"重新处理文档: {document_id}")
        
        # 获取现有文档记录
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise ValueError(f"文档不存在: {document_id}")
        
        # 删除现有分块
        db.query(Chunk).filter(Chunk.document_id == document_id).delete()
        db.commit()
        
        # 重新分块
        if split_strategy is None:
            split_strategy = SplitStrategy.FIXED_SIZE
        
        chunks_data = self.text_splitter.split_text(
                text=document.content,
                strategy=split_strategy,
                metadata={
                    "document_id": document.id,
                    "file_name": document.filename,
                    "split_strategy": split_strategy.value,
                    "reprocessed": True
                }
            )
        
        # 创建新的分块记录
        chunk_objects = self._create_chunk_records(chunks_data, document.id, db)
        
        # 更新文档统计
        self._update_document_stats(document, chunk_objects, db)
        
        logger.info(f"文档重新处理完成: {document_id}, "
                   f"生成 {len(chunk_objects)} 个分块")
        
        return document, chunk_objects
    
    def get_document_with_chunks(self, document_id: str, db: Session) -> Optional[Tuple[Document, List[Chunk]]]:
        """
        获取文档及其分块
        
        Args:
            document_id: 文档ID
            db: 数据库会话
            
        Returns:
            (Document对象, Chunk对象列表) 或 None
        """
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            return None
        
        chunks = db.query(Chunk).filter(
            Chunk.document_id == document_id
        ).order_by(Chunk.chunk_index).all()
        
        return document, chunks
    
    def delete_document(self, document_id: str, db: Session) -> bool:
        """
        删除文档及其分块
        
        Args:
            document_id: 文档ID
            db: 数据库会话
            
        Returns:
            是否删除成功
        """
        try:
            # 删除分块
            deleted_chunks = db.query(Chunk).filter(Chunk.document_id == document_id).delete()
            
            # 删除文档
            deleted_docs = db.query(Document).filter(Document.id == document_id).delete()
            
            db.commit()
            
            logger.info(f"删除文档: {document_id}, "
                       f"删除分块: {deleted_chunks}, "
                       f"删除文档: {deleted_docs}")
            
            return deleted_docs > 0
            
        except Exception as e:
            logger.error(f"删除文档失败: {document_id}, 错误: {e}")
            db.rollback()
            return False
    
    def get_processing_stats(self, db: Session) -> Dict:
        """
        获取处理统计信息
        
        Args:
            db: 数据库会话
            
        Returns:
            统计信息字典
        """
        total_docs = db.query(Document).count()
        completed_docs = db.query(Document).filter(Document.processing_status == "completed").count()
        failed_docs = db.query(Document).filter(Document.processing_status == "failed").count()
        processing_docs = db.query(Document).filter(Document.processing_status == "processing").count()
        
        total_chunks = db.query(Chunk).count()
        pending_chunks = db.query(Chunk).filter(Chunk.processing_status == "pending").count()
        processed_chunks = db.query(Chunk).filter(Chunk.processing_status == "completed").count()
        
        return {
            "documents": {
                "total": total_docs,
                "completed": completed_docs,
                "failed": failed_docs,
                "processing": processing_docs
            },
            "chunks": {
                "total": total_chunks,
                "pending": pending_chunks,
                "processed": processed_chunks
            }
        }