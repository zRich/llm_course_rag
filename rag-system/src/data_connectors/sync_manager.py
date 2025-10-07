from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import logging
import json
import asyncio
from enum import Enum
from dataclasses import dataclass, asdict
import pandas as pd

from data_connector import DataConnector
from database_connector import DatabaseConnector
from api_connector import APIConnector

logger = logging.getLogger(__name__)

class SyncType(Enum):
    """同步类型枚举"""
    FULL = "full"
    INCREMENTAL = "incremental"

class SyncStatus(Enum):
    """同步状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class SyncResult:
    """同步结果数据类"""
    sync_id: str
    sync_type: SyncType
    status: SyncStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    total_records: int = 0
    processed_records: int = 0
    failed_records: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['sync_type'] = self.sync_type.value
        result['status'] = self.status.value
        result['start_time'] = self.start_time.isoformat()
        if self.end_time:
            result['end_time'] = self.end_time.isoformat()
        return result

class DataTransformer:
    """数据转换器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.field_mappings = config.get('field_mappings', {})
        self.data_types = config.get('data_types', {})
        self.filters = config.get('filters', [])
        self.transformations = config.get('transformations', [])
    
    def transform_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        转换单条记录
        
        Args:
            record: 原始记录
            
        Returns:
            Optional[Dict[str, Any]]: 转换后的记录，如果过滤掉则返回None
        """
        try:
            # 应用过滤器
            if not self._apply_filters(record):
                return None
            
            # 字段映射
            transformed = self._apply_field_mappings(record)
            
            # 数据类型转换
            transformed = self._apply_data_type_conversions(transformed)
            
            # 自定义转换
            transformed = self._apply_custom_transformations(transformed)
            
            return transformed
            
        except Exception as e:
            logger.error(f"Error transforming record: {str(e)}")
            return None
    
    def _apply_filters(self, record: Dict[str, Any]) -> bool:
        """
        应用过滤器
        
        Args:
            record: 记录
            
        Returns:
            bool: 是否通过过滤
        """
        for filter_config in self.filters:
            field = filter_config.get('field')
            operator = filter_config.get('operator', 'eq')
            value = filter_config.get('value')
            
            if field not in record:
                continue
            
            record_value = record[field]
            
            if operator == 'eq' and record_value != value:
                return False
            elif operator == 'ne' and record_value == value:
                return False
            elif operator == 'gt' and record_value <= value:
                return False
            elif operator == 'gte' and record_value < value:
                return False
            elif operator == 'lt' and record_value >= value:
                return False
            elif operator == 'lte' and record_value > value:
                return False
            elif operator == 'in' and record_value not in value:
                return False
            elif operator == 'not_in' and record_value in value:
                return False
        
        return True
    
    def _apply_field_mappings(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用字段映射
        
        Args:
            record: 原始记录
            
        Returns:
            Dict[str, Any]: 映射后的记录
        """
        if not self.field_mappings:
            return record
        
        mapped_record = {}
        for source_field, target_field in self.field_mappings.items():
            if source_field in record:
                mapped_record[target_field] = record[source_field]
        
        # 保留未映射的字段
        for field, value in record.items():
            if field not in self.field_mappings:
                mapped_record[field] = value
        
        return mapped_record
    
    def _apply_data_type_conversions(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用数据类型转换
        
        Args:
            record: 记录
            
        Returns:
            Dict[str, Any]: 转换后的记录
        """
        for field, target_type in self.data_types.items():
            if field in record and record[field] is not None:
                try:
                    if target_type == 'int':
                        record[field] = int(record[field])
                    elif target_type == 'float':
                        record[field] = float(record[field])
                    elif target_type == 'str':
                        record[field] = str(record[field])
                    elif target_type == 'bool':
                        record[field] = bool(record[field])
                    elif target_type == 'datetime':
                        if isinstance(record[field], str):
                            record[field] = datetime.fromisoformat(record[field].replace('Z', '+00:00'))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to convert field {field} to {target_type}: {str(e)}")
        
        return record
    
    def _apply_custom_transformations(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用自定义转换
        
        Args:
            record: 记录
            
        Returns:
            Dict[str, Any]: 转换后的记录
        """
        for transformation in self.transformations:
            transform_type = transformation.get('type')
            field = transformation.get('field')
            
            if field not in record:
                continue
            
            if transform_type == 'uppercase':
                record[field] = str(record[field]).upper()
            elif transform_type == 'lowercase':
                record[field] = str(record[field]).lower()
            elif transform_type == 'trim':
                record[field] = str(record[field]).strip()
            elif transform_type == 'replace':
                old_value = transformation.get('old_value', '')
                new_value = transformation.get('new_value', '')
                record[field] = str(record[field]).replace(old_value, new_value)
        
        return record

class SyncManager:
    """
    同步管理器
    负责管理数据源的全量和增量同步
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化同步管理器
        
        Args:
            config: 同步配置
                - connectors: 数据连接器配置列表
                - transformers: 数据转换器配置
                - sync_interval: 同步间隔（秒）
                - batch_size: 批处理大小
                - max_retries: 最大重试次数
                - retry_delay: 重试延迟（秒）
        """
        self.config = config
        self.connectors: Dict[str, DataConnector] = {}
        self.transformers: Dict[str, DataTransformer] = {}
        self.sync_history: Dict[str, SyncResult] = {}
        self.active_syncs: Dict[str, SyncResult] = {}
        self.sync_callbacks: List[Callable] = []
        
        self._initialize_connectors()
        self._initialize_transformers()
    
    def _initialize_connectors(self):
        """
        初始化数据连接器
        """
        connector_configs = self.config.get('connectors', [])
        
        for connector_config in connector_configs:
            try:
                connector_id = connector_config['id']
                connector_type = connector_config['type']
                config = connector_config.get('config', {})
                
                if connector_type == 'database':
                    connector = DatabaseConnector(connector_id, config)
                elif connector_type == 'api':
                    connector = APIConnector(connector_id, config)
                else:
                    logger.error(f"Unsupported connector type: {connector_type}")
                    continue
                
                self.connectors[connector_id] = connector
                logger.info(f"Initialized {connector_type} connector: {connector_id}")
            except Exception as e:
                logger.error(f"Failed to initialize connector {connector_config.get('id', 'unknown')}: {str(e)}")
                continue
    
    def _initialize_transformers(self):
        """
        初始化数据转换器
        """
        transformer_configs = self.config.get('transformers', {})
        
        for transformer_id, transformer_config in transformer_configs.items():
            self.transformers[transformer_id] = DataTransformer(transformer_config)
            logger.info(f"Initialized transformer: {transformer_id}")
    
    def add_sync_callback(self, callback: Callable[[SyncResult], None]):
        """
        添加同步回调函数
        
        Args:
            callback: 回调函数，接收SyncResult参数
        """
        self.sync_callbacks.append(callback)
    
    def start_full_sync(self, 
                       connector_id: str,
                       transformer_id: Optional[str] = None,
                       query: Optional[str] = None) -> str:
        """
        启动全量同步
        
        Args:
            connector_id: 连接器ID
            transformer_id: 转换器ID
            query: 查询条件
            
        Returns:
            str: 同步任务ID
        """
        sync_id = f"full_{connector_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        sync_result = SyncResult(
            sync_id=sync_id,
            sync_type=SyncType.FULL,
            status=SyncStatus.PENDING,
            start_time=datetime.now(),
            metadata={'connector_id': connector_id, 'transformer_id': transformer_id, 'query': query}
        )
        
        self.active_syncs[sync_id] = sync_result
        
        # 异步执行同步任务
        try:
            asyncio.create_task(self._execute_sync(sync_result, query))
        except RuntimeError:
            # 如果没有运行的事件循环（测试环境），同步执行
            asyncio.run(self._execute_sync(sync_result, query))
        
        logger.info(f"Started full sync: {sync_id}")
        return sync_id
    
    def start_incremental_sync(self, 
                              connector_id: str,
                              last_sync_time: Optional[datetime] = None,
                              transformer_id: Optional[str] = None,
                              query: Optional[str] = None) -> str:
        """
        启动增量同步
        
        Args:
            connector_id: 连接器ID
            last_sync_time: 上次同步时间，如果为None则使用连接器的最后同步时间
            transformer_id: 转换器ID
            query: 查询条件
            
        Returns:
            str: 同步任务ID
        """
        connector = self.connectors.get(connector_id)
        if not connector:
            raise ValueError(f"Connector not found: {connector_id}")
        
        if last_sync_time is None:
            last_sync_time = connector.last_sync_time
            if last_sync_time is None:
                # 如果没有上次同步时间，使用24小时前
                last_sync_time = datetime.now() - timedelta(hours=24)
        
        sync_id = f"incr_{connector_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        sync_result = SyncResult(
            sync_id=sync_id,
            sync_type=SyncType.INCREMENTAL,
            status=SyncStatus.PENDING,
            start_time=datetime.now(),
            metadata={
                'connector_id': connector_id, 
                'transformer_id': transformer_id, 
                'query': query,
                'last_sync_time': last_sync_time.isoformat()
            }
        )
        
        self.active_syncs[sync_id] = sync_result
        
        # 异步执行同步任务
        try:
            asyncio.create_task(self._execute_incremental_sync(sync_result, last_sync_time, query))
        except RuntimeError:
            # 如果没有运行的事件循环（测试环境），同步执行
            asyncio.run(self._execute_incremental_sync(sync_result, last_sync_time, query))
        
        logger.info(f"Started incremental sync: {sync_id}")
        return sync_id
    
    async def _execute_sync(self, sync_result: SyncResult, query: Optional[str] = None):
        """
        执行全量同步
        
        Args:
            sync_result: 同步结果对象
            query: 查询条件
        """
        connector_id = sync_result.metadata['connector_id']
        transformer_id = sync_result.metadata.get('transformer_id')
        
        try:
            sync_result.status = SyncStatus.RUNNING
            self._notify_callbacks(sync_result)
            
            connector = self.connectors[connector_id]
            transformer = self.transformers.get(transformer_id) if transformer_id else None
            
            # 连接数据源
            if not connector.connect():
                raise Exception(f"Failed to connect to data source: {connector_id}")
            
            try:
                # 获取总数
                total_count = connector.get_total_count(query)
                sync_result.total_records = total_count
                
                batch_size = self.config.get('batch_size', 1000)
                processed = 0
                
                # 分批处理数据
                for offset in range(0, total_count, batch_size):
                    current_batch_size = min(batch_size, total_count - offset)
                    
                    # 获取数据
                    records = list(connector.fetch_data(
                        query=query,
                        limit=current_batch_size,
                        offset=offset
                    ))
                    
                    # 处理数据
                    processed_batch = await self._process_batch(records, transformer)
                    processed += len(processed_batch)
                    
                    sync_result.processed_records = processed
                    
                    logger.info(f"Processed {processed}/{total_count} records for sync {sync_result.sync_id}")
                
                # 更新连接器的最后同步时间
                connector.update_last_sync_time()
                
                sync_result.status = SyncStatus.COMPLETED
                sync_result.end_time = datetime.now()
                
            finally:
                connector.disconnect()
            
        except Exception as e:
            logger.error(f"Sync failed: {str(e)}")
            sync_result.status = SyncStatus.FAILED
            sync_result.error_message = str(e)
            sync_result.end_time = datetime.now()
        
        finally:
            # 移动到历史记录
            self.sync_history[sync_result.sync_id] = sync_result
            if sync_result.sync_id in self.active_syncs:
                del self.active_syncs[sync_result.sync_id]
            
            self._notify_callbacks(sync_result)
    
    async def _execute_incremental_sync(self, 
                                       sync_result: SyncResult, 
                                       last_sync_time: datetime,
                                       query: Optional[str] = None):
        """
        执行增量同步
        
        Args:
            sync_result: 同步结果对象
            last_sync_time: 上次同步时间
            query: 查询条件
        """
        connector_id = sync_result.metadata['connector_id']
        transformer_id = sync_result.metadata.get('transformer_id')
        
        try:
            sync_result.status = SyncStatus.RUNNING
            self._notify_callbacks(sync_result)
            
            connector = self.connectors[connector_id]
            transformer = self.transformers.get(transformer_id) if transformer_id else None
            
            # 连接数据源
            if not connector.connect():
                raise Exception(f"Failed to connect to data source: {connector_id}")
            
            try:
                # 获取增量数据
                records = list(connector.fetch_incremental_data(last_sync_time, query))
                sync_result.total_records = len(records)
                
                # 处理数据
                processed_records = await self._process_batch(records, transformer)
                sync_result.processed_records = len(processed_records)
                
                # 更新连接器的最后同步时间
                connector.update_last_sync_time()
                
                sync_result.status = SyncStatus.COMPLETED
                sync_result.end_time = datetime.now()
                
                logger.info(f"Incremental sync completed: {sync_result.processed_records} records processed")
                
            finally:
                connector.disconnect()
            
        except Exception as e:
            logger.error(f"Incremental sync failed: {str(e)}")
            sync_result.status = SyncStatus.FAILED
            sync_result.error_message = str(e)
            sync_result.end_time = datetime.now()
        
        finally:
            # 移动到历史记录
            self.sync_history[sync_result.sync_id] = sync_result
            if sync_result.sync_id in self.active_syncs:
                del self.active_syncs[sync_result.sync_id]
            
            self._notify_callbacks(sync_result)
    
    async def _process_batch(self, 
                           records: List[Dict[str, Any]], 
                           transformer: Optional[DataTransformer] = None) -> List[Dict[str, Any]]:
        """
        处理数据批次
        
        Args:
            records: 原始记录列表
            transformer: 数据转换器
            
        Returns:
            List[Dict[str, Any]]: 处理后的记录列表
        """
        processed_records = []
        
        for record in records:
            try:
                if transformer:
                    transformed_record = transformer.transform_record(record)
                    if transformed_record is not None:
                        processed_records.append(transformed_record)
                else:
                    processed_records.append(record)
            except Exception as e:
                logger.error(f"Error processing record: {str(e)}")
                continue
        
        return processed_records
    
    def _notify_callbacks(self, sync_result: SyncResult):
        """
        通知回调函数
        
        Args:
            sync_result: 同步结果
        """
        for callback in self.sync_callbacks:
            try:
                callback(sync_result)
            except Exception as e:
                logger.error(f"Error in sync callback: {str(e)}")
    
    def get_sync_status(self, sync_id: str) -> Optional[Dict[str, Any]]:
        """
        获取同步状态
        
        Args:
            sync_id: 同步任务ID
            
        Returns:
            Optional[Dict[str, Any]]: 同步状态信息
        """
        # 检查活跃同步
        if sync_id in self.active_syncs:
            return self.active_syncs[sync_id].to_dict()
        
        # 检查历史记录
        if sync_id in self.sync_history:
            return self.sync_history[sync_id].to_dict()
        
        return None
    
    def get_all_sync_status(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取所有同步状态
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: 所有同步状态信息
        """
        return {
            'active': [sync.to_dict() for sync in self.active_syncs.values()],
            'history': [sync.to_dict() for sync in list(self.sync_history.values())[-50:]]  # 最近50条历史记录
        }
    
    def cancel_sync(self, sync_id: str) -> bool:
        """
        取消同步任务
        
        Args:
            sync_id: 同步任务ID
            
        Returns:
            bool: 是否成功取消
        """
        if sync_id in self.active_syncs:
            sync_result = self.active_syncs[sync_id]
            sync_result.status = SyncStatus.CANCELLED
            sync_result.end_time = datetime.now()
            
            # 移动到历史记录
            self.sync_history[sync_id] = sync_result
            del self.active_syncs[sync_id]
            
            self._notify_callbacks(sync_result)
            
            logger.info(f"Cancelled sync: {sync_id}")
            return True
        
        return False
    
    def cleanup_history(self, days: int = 30):
        """
        清理历史记录
        
        Args:
            days: 保留天数
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        
        original_count = len(self.sync_history)
        self.sync_history = {
            sync_id: sync for sync_id, sync in self.sync_history.items() 
            if sync.start_time > cutoff_time
        }
        
        cleaned_count = original_count - len(self.sync_history)
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old sync history records")
    
    def get_sync_history(self) -> List[Dict[str, Any]]:
        """
        获取同步历史记录
        
        Returns:
            List[Dict[str, Any]]: 同步历史记录列表
        """
        return [sync.to_dict() for sync in self.sync_history.values()]
    
    def cleanup_old_history(self, days: int = 30) -> int:
        """
        清理旧的历史记录
        
        Args:
            days: 保留天数
            
        Returns:
            int: 清理的记录数量
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        
        original_count = len(self.sync_history)
        self.sync_history = {
            sync_id: sync for sync_id, sync in self.sync_history.items() 
            if sync.start_time > cutoff_time
        }
        
        cleaned_count = original_count - len(self.sync_history)
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old sync history records")
        
        return cleaned_count
    
    def add_connector(self, connector_id: str, connector: DataConnector) -> bool:
        """
        添加数据连接器
        
        Args:
            connector_id: 连接器ID
            connector: 数据连接器实例
            
        Returns:
            bool: 是否成功添加
        """
        if connector_id in self.connectors:
            logger.warning(f"Connector {connector_id} already exists")
            return False
        
        try:
            # 测试连接
            if not connector.test_connection():
                logger.error(f"Failed to test connection for connector {connector_id}")
                return False
            
            self.connectors[connector_id] = connector
            logger.info(f"Added connector: {connector_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding connector {connector_id}: {str(e)}")
            return False
    
    def remove_connector(self, connector_id: str) -> bool:
        """
        移除数据连接器
        
        Args:
            connector_id: 连接器ID
            
        Returns:
            bool: 是否成功移除
        """
        if connector_id not in self.connectors:
            logger.warning(f"Connector {connector_id} not found")
            return False
        
        # 检查是否有活跃的同步任务使用此连接器
        for sync_result in self.active_syncs.values():
            if sync_result.metadata.get('connector_id') == connector_id:
                logger.error(f"Cannot remove connector {connector_id}: active sync in progress")
                return False
        
        try:
            connector = self.connectors[connector_id]
            connector.disconnect()
            del self.connectors[connector_id]
            logger.info(f"Removed connector: {connector_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing connector {connector_id}: {str(e)}")
            return False
    
    def get_connector_info(self, connector_id: str) -> Optional[Dict[str, Any]]:
        """
        获取连接器信息
        
        Args:
            connector_id: 连接器ID
            
        Returns:
            Optional[Dict[str, Any]]: 连接器信息
        """
        if connector_id not in self.connectors:
            return None
        
        connector = self.connectors[connector_id]
        
        # 安全地获取配置信息，避免递归
        try:
            config = getattr(connector, 'config', {})
            if hasattr(config, 'copy'):
                config = config.copy()
            elif isinstance(config, dict):
                config = dict(config)
            else:
                config = {}
        except Exception:
            config = {}
        
        # 安全地获取类型信息
        try:
            connector_type = type(connector).__name__
            if 'DatabaseConnector' in connector_type:
                connector_type = 'database'
            elif 'APIConnector' in connector_type:
                connector_type = 'api'
            else:
                connector_type = 'unknown'
        except Exception:
            connector_type = 'unknown'
        
        return {
            'id': connector_id,
            'type': connector_type,
            'config': config,
            'last_sync_time': getattr(connector, 'last_sync_time', None),
            'is_connected': getattr(connector, 'is_connected', False)
        }
    
    def list_connectors(self) -> List[Dict[str, Any]]:
        """
        列出所有连接器
        
        Returns:
            List[Dict[str, Any]]: 连接器信息列表
        """
        return [
            self.get_connector_info(connector_id)
            for connector_id in self.connectors.keys()
        ]
    
    def add_transformer(self, transformer_id: str, transformer: DataTransformer) -> bool:
        """
        添加数据转换器
        
        Args:
            transformer_id: 转换器ID
            transformer: 数据转换器实例
            
        Returns:
            bool: 是否成功添加
        """
        if transformer_id in self.transformers:
            logger.warning(f"Transformer {transformer_id} already exists")
            return False
        
        try:
            self.transformers[transformer_id] = transformer
            logger.info(f"Added transformer: {transformer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding transformer {transformer_id}: {str(e)}")
            return False
    
    def remove_transformer(self, transformer_id: str) -> bool:
        """
        移除数据转换器
        
        Args:
            transformer_id: 转换器ID
            
        Returns:
            bool: 是否成功移除
        """
        if transformer_id not in self.transformers:
            logger.warning(f"Transformer {transformer_id} not found")
            return False
        
        # 检查是否有活跃的同步任务使用此转换器
        for sync_result in self.active_syncs.values():
            if sync_result.metadata.get('transformer_id') == transformer_id:
                logger.error(f"Cannot remove transformer {transformer_id}: active sync in progress")
                return False
        
        try:
            del self.transformers[transformer_id]
            logger.info(f"Removed transformer: {transformer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing transformer {transformer_id}: {str(e)}")
            return False
    
    def get_transformer_info(self, transformer_id: str) -> Optional[Dict[str, Any]]:
        """
        获取转换器信息
        
        Args:
            transformer_id: 转换器ID
            
        Returns:
            Optional[Dict[str, Any]]: 转换器信息
        """
        if transformer_id not in self.transformers:
            return None
        
        transformer = self.transformers[transformer_id]
        return {
            'id': transformer_id,
            'type': transformer.__class__.__name__,
            'config': transformer.config
        }
    
    def list_transformers(self) -> List[Dict[str, Any]]:
        """
        列出所有转换器
        
        Returns:
            List[Dict[str, Any]]: 转换器信息列表
        """
        return [
            self.get_transformer_info(transformer_id)
            for transformer_id in self.transformers.keys()
        ]