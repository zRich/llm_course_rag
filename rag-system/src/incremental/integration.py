"""增量更新系统与RAG系统集成模块"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from config import get_config, IncrementalConfig

# 添加父目录到Python路径，以便导入RAG系统模块
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.config import get_settings
    from src.database.connection import get_database_session
    from src.embedding.embedder import TextEmbedder
    from src.vector_store.qdrant_client import QdrantVectorStore
    from src.document.document_manager import DocumentManager
except ImportError as e:
    logging.warning(f"无法导入RAG系统模块: {e}")
    # 提供备用实现
    get_settings = None
    get_database_session = None
    TextEmbedder = None
    QdrantVectorStore = None
    DocumentManager = None

from .change_detector import ChangeDetector
from .version_manager import VersionManager
from .incremental_indexer import IncrementalIndexer
from .conflict_resolver import ConflictResolver
from .monitoring import get_monitoring_manager

logger = logging.getLogger(__name__)

class RAGIncrementalIntegration:
    """RAG增量更新集成服务"""
    
    def __init__(self, config: IncrementalConfig):
        """初始化集成服务"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.monitoring = get_monitoring_manager()
        
        # 设置日志
        self._setup_logging()
        
        # 初始化增量更新组件
        self.change_detector = ChangeDetector(
            data_dir=config.data_dir,
            metadata_dir=config.metadata_dir,
            hash_algorithm=config.hash_algorithm
        )
        
        self.version_manager = VersionManager(
            versions_dir=config.metadata_dir / "versions",
            max_versions_per_document=config.max_versions_per_document
        )
        
        self.incremental_indexer = IncrementalIndexer(
            index_dir=config.metadata_dir / "index",
            batch_size=config.batch_size,
            enable_parallel_processing=config.enable_parallel_processing
        )
        
        self.conflict_resolver = ConflictResolver(
            conflicts_dir=config.metadata_dir / "conflicts",
            default_strategy=config.default_conflict_strategy,
            enable_auto_resolution=config.enable_auto_resolution
        )
        
        # RAG系统组件（可选）
        self.embedder = None
        self.vector_store = None
        self.document_manager = None
        
        # 初始化RAG集成
        if config.enable_rag_integration:
            self._initialize_rag_components()
        
        self.logger.info("RAG增量更新集成服务初始化完成")
    
    def _setup_logging(self):
        """设置日志配置"""
        try:
            # 设置日志级别
            log_level = getattr(logging, self.config.log_level.upper())
            self.logger.setLevel(log_level)
            
            # 如果配置了日志文件，添加文件处理器
            if self.config.log_file:
                log_file = Path(self.config.log_file)
                log_file.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(log_level)
                
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(formatter)
                
                self.logger.addHandler(file_handler)
                
        except Exception as e:
            print(f"设置日志配置失败: {e}")
    
    def _initialize_rag_components(self):
        """初始化RAG系统组件"""
        try:
            # 这里应该根据实际的RAG系统实现来初始化组件
            # 由于我们不知道具体的RAG系统结构，这里只是示例
            self.logger.info("初始化RAG系统组件...")
            
            # 示例：初始化嵌入器
            # self.embedder = SomeEmbedder()
            
            # 示例：初始化向量存储
            # self.vector_store = SomeVectorStore()
            
            # 示例：初始化文档管理器
            # self.document_manager = SomeDocumentManager()
            
            self.logger.info("RAG系统组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化RAG系统组件失败: {e}")
            if self.monitoring:
                self.monitoring.handle_error(e, {
                    "operation": "initialize_rag_components"
                })
    
    async def scan_and_update(
        self,
        directory: Optional[Path] = None,
        collection_name: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """扫描目录并执行增量更新"""
        try:
            with self.monitoring.timer("scan_and_update"):
                self.logger.info("开始扫描和更新操作")
                
                # 使用配置中的默认值
                if directory is None:
                    directory = self.config.data_dir
                if collection_name is None:
                    collection_name = self.config.default_collection_name
                if batch_size is None:
                    batch_size = self.config.batch_size
                
                # 检测变更
                changes = self.change_detector.detect_changes(directory)
                
                if not changes:
                    self.logger.info("未检测到文件变更")
                    return {
                        "status": "success",
                        "message": "未检测到文件变更",
                        "changes": [],
                        "processed_count": 0
                    }
                
                self.logger.info(f"检测到 {len(changes)} 个文件变更")
                
                # 处理变更
                results = await self._process_changes(
                    changes, collection_name, batch_size
                )
                
                # 记录监控指标
                if self.monitoring:
                    self.monitoring.record_metric("files_processed", len(changes))
                    self.monitoring.record_metric("scan_and_update_success", 1)
                
                self.logger.info(f"增量更新完成，处理了 {len(changes)} 个文件")
                
                return {
                    "status": "success",
                    "message": f"成功处理 {len(changes)} 个文件变更",
                    "changes": [change.to_dict() for change in changes],
                    "processed_count": len(changes),
                    "results": results
                }
                
        except Exception as e:
            self.logger.error(f"扫描和更新操作失败: {e}")
            if self.monitoring:
                self.monitoring.handle_error(e, {
                    "operation": "scan_and_update",
                    "directory": str(directory),
                    "collection_name": collection_name
                })
                self.monitoring.record_metric("scan_and_update_error", 1)
            
            return {
                "status": "error",
                "message": f"扫描和更新操作失败: {str(e)}",
                "changes": [],
                "processed_count": 0
            }
    
    async def _process_changes(
        self,
        changes: List[Any],
        collection_name: str,
        batch_size: int
    ) -> Dict[str, Any]:
        """处理文件变更"""
        results = {
            "added": [],
            "modified": [],
            "deleted": [],
            "conflicts": [],
            "errors": []
        }
        
        try:
            for change in changes:
                try:
                    # 创建版本记录
                    version = self.version_manager.create_version(
                        file_path=change.file_path,
                        content_hash=change.new_hash,
                        metadata={
                            "change_type": change.change_type,
                            "timestamp": change.timestamp,
                            "file_size": change.file_size
                        }
                    )
                    
                    # 检测冲突
                    conflict = self.conflict_resolver.detect_conflict(
                        file_path=change.file_path,
                        conflict_type="version_conflict",
                        description=f"处理 {change.change_type} 操作时的潜在冲突",
                        metadata={
                            "change_type": change.change_type,
                            "version_id": version.version_id if version else None
                        }
                    )
                    
                    if conflict:
                        # 尝试自动解决冲突
                        resolved = self.conflict_resolver.resolve_conflict(
                            conflict.conflict_id
                        )
                        
                        if not resolved:
                            results["conflicts"].append({
                                "file_path": change.file_path,
                                "conflict_id": conflict.conflict_id,
                                "description": conflict.description
                            })
                            continue
                    
                    # 处理增量索引
                    index_result = await self.incremental_indexer.process_changes(
                        [change], collection_name
                    )
                    
                    # 更新RAG系统（如果启用）
                    if self.config.enable_rag_integration:
                        await self._update_rag_system(change, collection_name)
                    
                    # 记录结果
                    change_type = change.change_type.lower()
                    if change_type in results:
                        results[change_type].append({
                            "file_path": change.file_path,
                            "version_id": version.version_id if version else None,
                            "index_result": index_result
                        })
                    
                except Exception as e:
                    self.logger.error(f"处理文件变更失败 {change.file_path}: {e}")
                    results["errors"].append({
                        "file_path": change.file_path,
                        "error": str(e)
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"批量处理变更失败: {e}")
            if self.monitoring:
                self.monitoring.handle_error(e, {
                    "operation": "process_changes",
                    "changes_count": len(changes)
                })
            raise
    
    async def _update_rag_system(self, change: Any, collection_name: str):
        """更新RAG系统"""
        try:
            # 这里应该根据实际的RAG系统实现来更新
            # 由于我们不知道具体的RAG系统结构，这里只是示例
            
            if change.change_type.lower() == "added" or change.change_type.lower() == "modified":
                # 添加或更新文档到向量存储
                # if self.vector_store:
                #     await self.vector_store.upsert_document(change.file_path, collection_name)
                pass
            
            elif change.change_type.lower() == "deleted":
                # 从向量存储中删除文档
                # if self.vector_store:
                #     await self.vector_store.delete_document(change.file_path, collection_name)
                pass
            
            self.logger.debug(f"RAG系统更新完成: {change.file_path}")
            
        except Exception as e:
            self.logger.error(f"更新RAG系统失败 {change.file_path}: {e}")
            if self.monitoring:
                self.monitoring.handle_error(e, {
                    "operation": "update_rag_system",
                    "file_path": change.file_path,
                    "change_type": change.change_type
                })
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            with self.monitoring.timer("get_system_status"):
                # 获取各组件状态
                change_detector_stats = self.change_detector.get_stats()
                version_manager_stats = self.version_manager.get_stats()
                indexer_stats = self.incremental_indexer.get_stats()
                conflict_stats = self.conflict_resolver.get_stats()
                
                # 获取监控信息
                monitoring_stats = {}
                if self.monitoring:
                    monitoring_stats = {
                        "performance_metrics": self.monitoring.get_performance_metrics().to_dict(),
                        "error_stats": self.monitoring.get_error_stats(),
                        "health_status": self.monitoring.get_health_status()
                    }
                
                return {
                    "status": "running",
                    "timestamp": datetime.now().isoformat(),
                    "config": {
                        "data_dir": str(self.config.data_dir),
                        "metadata_dir": str(self.config.metadata_dir),
                        "enable_rag_integration": self.config.enable_rag_integration,
                        "batch_size": self.config.batch_size,
                        "log_level": self.config.log_level
                    },
                    "components": {
                        "change_detector": change_detector_stats,
                        "version_manager": version_manager_stats,
                        "incremental_indexer": indexer_stats,
                        "conflict_resolver": conflict_stats.to_dict()
                    },
                    "monitoring": monitoring_stats
                }
                
        except Exception as e:
            self.logger.error(f"获取系统状态失败: {e}")
            if self.monitoring:
                self.monitoring.handle_error(e, {
                    "operation": "get_system_status"
                })
            
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """获取集成统计信息"""
        try:
            stats = {
                "total_files_processed": 0,
                "total_versions_created": 0,
                "total_conflicts_detected": 0,
                "total_conflicts_resolved": 0,
                "rag_integration_enabled": self.config.enable_rag_integration,
                "last_update": datetime.now().isoformat()
            }
            
            # 聚合各组件统计信息
            try:
                change_stats = self.change_detector.get_stats()
                stats["total_files_processed"] = change_stats.get("total_files_scanned", 0)
            except:
                pass
            
            try:
                version_stats = self.version_manager.get_stats()
                stats["total_versions_created"] = version_stats.get("total_versions", 0)
            except:
                pass
            
            try:
                conflict_stats = self.conflict_resolver.get_stats()
                stats["total_conflicts_detected"] = conflict_stats.total_conflicts
                stats["total_conflicts_resolved"] = conflict_stats.resolved_conflicts
            except:
                pass
            
            return stats
            
        except Exception as e:
            self.logger.error(f"获取集成统计信息失败: {e}")
            if self.monitoring:
                self.monitoring.handle_error(e, {
                    "operation": "get_integration_stats"
                })
            
            return {
                "error": str(e),
                "last_update": datetime.now().isoformat()
            }


# 单例模式
_integration_instance = None

def get_integration_instance(config: Optional[IncrementalConfig] = None) -> RAGIncrementalIntegration:
    """获取集成服务实例（单例模式）"""
    global _integration_instance
    
    if _integration_instance is None:
        if config is None:
            # 使用默认配置
            config = IncrementalConfig()
        _integration_instance = RAGIncrementalIntegration(config)
    
    return _integration_instance

if __name__ == "__main__":
    # 测试集成功能
    import asyncio
    
    async def test_integration():
        integration = get_integration_instance()
        
        # 获取系统状态
        status = integration.get_system_status()
        print("系统状态:", status)
        
        # 获取统计信息
        stats = integration.get_integration_stats()
        print("统计信息:", stats)
    
    asyncio.run(test_integration())