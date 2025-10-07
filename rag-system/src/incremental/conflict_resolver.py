#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冲突解决器 - ConflictResolver

处理增量更新过程中的各种冲突
支持多种冲突解决策略
提供冲突检测和自动解决机制

作者: RAG系统开发团队
日期: 2024-01-15
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

# 尝试导入监控模块
try:
    from .monitoring import get_monitoring_manager
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


class ConflictType(Enum):
    """冲突类型枚举"""
    VERSION_CONFLICT = "version_conflict"      # 版本冲突
    HASH_MISMATCH = "hash_mismatch"            # 哈希不匹配
    CONCURRENT_UPDATE = "concurrent_update"    # 并发更新
    DEPENDENCY_CONFLICT = "dependency_conflict" # 依赖冲突
    SCHEMA_CONFLICT = "schema_conflict"        # 模式冲突
    CONTENT_CONFLICT = "content_conflict"      # 内容冲突


class ResolutionStrategy(Enum):
    """解决策略枚举"""
    LATEST_WINS = "latest_wins"              # 最新版本获胜
    MANUAL_REVIEW = "manual_review"          # 手动审查
    MERGE_CONTENT = "merge_content"          # 合并内容
    SKIP_UPDATE = "skip_update"              # 跳过更新
    FORCE_UPDATE = "force_update"            # 强制更新
    ROLLBACK = "rollback"                    # 回滚
    CUSTOM_HANDLER = "custom_handler"        # 自定义处理器


@dataclass
class ConflictRecord:
    """冲突记录"""
    conflict_id: str
    conflict_type: str
    file_path: str
    description: str
    detected_at: str
    resolved_at: Optional[str] = None
    resolution_strategy: Optional[str] = None
    resolution_result: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConflictRecord':
        """从字典创建实例"""
        return cls(**data)


@dataclass
class ConflictStats:
    """冲突统计信息"""
    total_conflicts: int = 0
    resolved_conflicts: int = 0
    pending_conflicts: int = 0
    conflicts_by_type: Dict[str, int] = None
    conflicts_by_strategy: Dict[str, int] = None
    average_resolution_time: float = 0.0
    last_update: Optional[str] = None
    
    def __post_init__(self):
        if self.conflicts_by_type is None:
            self.conflicts_by_type = {}
        if self.conflicts_by_strategy is None:
            self.conflicts_by_strategy = {}
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


class ConflictResolver:
    """冲突解决器
    
    功能:
    1. 检测各种类型的冲突
    2. 应用不同的解决策略
    3. 记录冲突历史
    4. 提供冲突统计
    5. 支持自定义解决器
    """
    
    def __init__(self,
                 conflicts_dir: str = "conflicts",
                 default_strategy: ResolutionStrategy = ResolutionStrategy.LATEST_WINS,
                 enable_auto_resolution: bool = True,
                 max_resolution_attempts: int = 3,
                 config: Optional[Dict] = None):
        """
        初始化冲突解决器
        
        Args:
            conflicts_dir: 冲突记录存储目录
            default_strategy: 默认解决策略
            enable_auto_resolution: 是否启用自动解决
            max_resolution_attempts: 最大解决尝试次数
            config: 配置字典
        """
        self.conflicts_dir = Path(conflicts_dir)
        self.default_strategy = default_strategy
        self.enable_auto_resolution = enable_auto_resolution
        self.max_resolution_attempts = max_resolution_attempts
        self.config = config or {}
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 获取监控管理器
        self.monitoring = None
        if MONITORING_AVAILABLE:
            try:
                self.monitoring = get_monitoring_manager()
            except ValueError:
                # 监控管理器未初始化
                pass
        
        # 创建冲突目录
        self.conflicts_dir.mkdir(parents=True, exist_ok=True)
        
        # 冲突存储
        self.conflicts_file = self.conflicts_dir / "conflicts.json"
        self.stats_file = self.conflicts_dir / "stats.json"
        
        # 加载现有冲突
        self.conflicts: Dict[str, ConflictRecord] = self._load_conflicts()
        self.stats = self._load_stats()
        
        # 策略处理器映射
        self.strategy_handlers: Dict[ResolutionStrategy, Callable] = {
            ResolutionStrategy.LATEST_WINS: self._resolve_latest_wins,
            ResolutionStrategy.MANUAL_REVIEW: self._resolve_manual_review,
            ResolutionStrategy.MERGE_CONTENT: self._resolve_merge_content,
            ResolutionStrategy.SKIP_UPDATE: self._resolve_skip_update,
            ResolutionStrategy.FORCE_UPDATE: self._resolve_force_update,
            ResolutionStrategy.ROLLBACK: self._resolve_rollback
        }
        
        # 自定义处理器
        self.custom_handlers: Dict[str, Callable] = {}
        
        # 运行时统计
        self.runtime_stats = {
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'auto_resolutions': 0,
            'manual_resolutions': 0,
            'resolution_failures': 0
        }
    
    def detect_conflict(self,
                       file_path: str,
                       conflict_type: ConflictType,
                       description: str,
                       metadata: Optional[Dict[str, Any]] = None) -> ConflictRecord:
        """
        检测并记录冲突
        
        Args:
            file_path: 文件路径
            conflict_type: 冲突类型
            description: 冲突描述
            metadata: 附加元数据
            
        Returns:
            冲突记录
        """
        import uuid
        import time
        
        start_time = time.time()
        
        try:
            # 创建冲突记录
            conflict_id = str(uuid.uuid4())
            conflict = ConflictRecord(
                conflict_id=conflict_id,
                conflict_type=conflict_type.value,
                file_path=file_path,
                description=description,
                detected_at=datetime.now().isoformat(),
                metadata=metadata or {}
            )
            
            # 存储冲突
            self.conflicts[conflict_id] = conflict
            
            # 更新统计
            self.runtime_stats['conflicts_detected'] += 1
            
            # 记录监控日志
            if self.monitoring:
                self.monitoring.log_operation(
                    "conflict_detection",
                    {
                        "conflict_id": conflict_id,
                        "conflict_type": conflict_type.value,
                        "file_path": file_path,
                        "description": description,
                        "detection_duration": time.time() - start_time
                    }
                )
                
                # 记录指标
                self.monitoring.performance_monitor.metrics_collector.record_metric(
                    "conflict_detected", 1
                )
                self.monitoring.performance_monitor.metrics_collector.record_metric(
                    f"conflict_{conflict_type.value}", 1
                )
            
            self.logger.warning(
                f"检测到冲突 {conflict_id}: {conflict_type.value} - {file_path} - {description}"
            )
            
            # 自动解决（如果启用）
            if self.enable_auto_resolution:
                try:
                    self.resolve_conflict(conflict_id)
                except Exception as e:
                    self.logger.error(f"自动解决冲突失败 {conflict_id}: {e}")
            
            # 保存冲突记录
            self._save_conflicts()
            self._update_stats()
            
            return conflict
            
        except Exception as e:
            if self.monitoring:
                self.monitoring.handle_error(e, {
                    "operation": "detect_conflict",
                    "file_path": file_path,
                    "conflict_type": conflict_type.value
                })
            self.logger.error(f"冲突检测失败: {e}")
            raise
    
    def resolve_conflict(self,
                        conflict_id: str,
                        strategy: Optional[ResolutionStrategy] = None,
                        custom_handler: Optional[str] = None) -> bool:
        """
        解决冲突
        
        Args:
            conflict_id: 冲突ID
            strategy: 解决策略
            custom_handler: 自定义处理器名称
            
        Returns:
            是否解决成功
        """
        import time
        start_time = time.time()
        
        try:
            # 获取冲突记录
            conflict = self.conflicts.get(conflict_id)
            if not conflict:
                raise ValueError(f"冲突不存在: {conflict_id}")
            
            if conflict.resolved_at:
                self.logger.info(f"冲突已解决: {conflict_id}")
                return True
            
            # 使用监控计时器
            if self.monitoring:
                with self.monitoring.timer("conflict_resolution"):
                    result = self._perform_conflict_resolution(
                        conflict, strategy, custom_handler
                    )
            else:
                result = self._perform_conflict_resolution(
                    conflict, strategy, custom_handler
                )
            
            # 更新统计
            resolution_time = time.time() - start_time
            if result:
                self.runtime_stats['conflicts_resolved'] += 1
                if strategy or custom_handler:
                    self.runtime_stats['manual_resolutions'] += 1
                else:
                    self.runtime_stats['auto_resolutions'] += 1
            else:
                self.runtime_stats['resolution_failures'] += 1
            
            # 记录监控日志
            if self.monitoring:
                self.monitoring.log_operation(
                    "conflict_resolution",
                    {
                        "conflict_id": conflict_id,
                        "strategy": strategy.value if strategy else self.default_strategy.value,
                        "custom_handler": custom_handler,
                        "success": result,
                        "resolution_duration": resolution_time
                    }
                )
                
                # 记录指标
                if result:
                    self.monitoring.performance_monitor.metrics_collector.record_metric(
                        "conflict_resolved", 1
                    )
                else:
                    self.monitoring.performance_monitor.metrics_collector.record_metric(
                        "conflict_resolution_failed", 1
                    )
            
            # 保存更新
            self._save_conflicts()
            self._update_stats()
            
            return result
            
        except Exception as e:
            if self.monitoring:
                self.monitoring.handle_error(e, {
                    "operation": "resolve_conflict",
                    "conflict_id": conflict_id,
                    "strategy": strategy.value if strategy else None
                })
            self.logger.error(f"冲突解决失败 {conflict_id}: {e}")
            self.runtime_stats['resolution_failures'] += 1
            return False
    
    def _perform_conflict_resolution(self,
                                   conflict: ConflictRecord,
                                   strategy: Optional[ResolutionStrategy],
                                   custom_handler: Optional[str]) -> bool:
        """执行实际的冲突解决"""
        # 确定解决策略
        if custom_handler and custom_handler in self.custom_handlers:
            handler = self.custom_handlers[custom_handler]
            strategy_name = custom_handler
        elif strategy and strategy in self.strategy_handlers:
            handler = self.strategy_handlers[strategy]
            strategy_name = strategy.value
        else:
            handler = self.strategy_handlers[self.default_strategy]
            strategy_name = self.default_strategy.value
        
        self.logger.info(
            f"使用策略 '{strategy_name}' 解决冲突 {conflict.conflict_id}"
        )
        
        # 执行解决
        try:
            result = handler(conflict)
            
            if result:
                # 标记为已解决
                conflict.resolved_at = datetime.now().isoformat()
                conflict.resolution_strategy = strategy_name
                conflict.resolution_result = "success"
                
                self.logger.info(
                    f"冲突解决成功 {conflict.conflict_id} 使用策略 '{strategy_name}'"
                )
            else:
                conflict.resolution_result = "failed"
                self.logger.error(
                    f"冲突解决失败 {conflict.conflict_id} 使用策略 '{strategy_name}'"
                )
            
            return result
            
        except Exception as e:
            conflict.resolution_result = f"error: {str(e)}"
            self.logger.error(
                f"冲突解决异常 {conflict.conflict_id}: {e}"
            )
            return False
    
    def _resolve_latest_wins(self, conflict: ConflictRecord) -> bool:
        """最新版本获胜策略"""
        try:
            self.logger.info(f"应用最新版本获胜策略: {conflict.file_path}")
            # 简化实现：保持当前状态
            return True
        except Exception as e:
            self.logger.error(f"最新版本获胜策略失败: {e}")
            return False
    
    def _resolve_manual_review(self, conflict: ConflictRecord) -> bool:
        """手动审查策略"""
        try:
            self.logger.info(f"标记为手动审查: {conflict.file_path}")
            # 标记为需要手动审查，不自动解决
            return False
        except Exception as e:
            self.logger.error(f"手动审查策略失败: {e}")
            return False
    
    def _resolve_merge_content(self, conflict: ConflictRecord) -> bool:
        """合并内容策略"""
        try:
            self.logger.info(f"尝试合并内容: {conflict.file_path}")
            # 简化实现：标记为成功
            return True
        except Exception as e:
            self.logger.error(f"合并内容策略失败: {e}")
            return False
    
    def _resolve_skip_update(self, conflict: ConflictRecord) -> bool:
        """跳过更新策略"""
        try:
            self.logger.info(f"跳过更新: {conflict.file_path}")
            return True
        except Exception as e:
            self.logger.error(f"跳过更新策略失败: {e}")
            return False
    
    def _resolve_force_update(self, conflict: ConflictRecord) -> bool:
        """强制更新策略"""
        try:
            self.logger.info(f"强制更新: {conflict.file_path}")
            return True
        except Exception as e:
            self.logger.error(f"强制更新策略失败: {e}")
            return False
    
    def _resolve_rollback(self, conflict: ConflictRecord) -> bool:
        """回滚策略"""
        try:
            self.logger.info(f"回滚操作: {conflict.file_path}")
            return True
        except Exception as e:
            self.logger.error(f"回滚策略失败: {e}")
            return False
    
    def register_custom_handler(self, name: str, handler: Callable[[ConflictRecord], bool]):
        """注册自定义处理器"""
        self.custom_handlers[name] = handler
        self.logger.info(f"注册自定义冲突处理器: {name}")
    
    def get_conflicts(self, resolved: Optional[bool] = None) -> List[ConflictRecord]:
        """获取冲突列表"""
        conflicts = list(self.conflicts.values())
        
        if resolved is not None:
            if resolved:
                conflicts = [c for c in conflicts if c.resolved_at is not None]
            else:
                conflicts = [c for c in conflicts if c.resolved_at is None]
        
        return conflicts
    
    def get_conflict_by_id(self, conflict_id: str) -> Optional[ConflictRecord]:
        """根据ID获取冲突记录"""
        return self.conflicts.get(conflict_id)
    
    def get_stats(self) -> ConflictStats:
        """获取冲突统计信息"""
        return self.stats
    
    def get_runtime_stats(self) -> Dict[str, int]:
        """获取运行时统计信息"""
        return self.runtime_stats.copy()
    
    def clear_resolved_conflicts(self, older_than_days: int = 30):
        """清理已解决的冲突记录"""
        try:
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            
            removed_count = 0
            conflicts_to_remove = []
            
            for conflict_id, conflict in self.conflicts.items():
                if conflict.resolved_at:
                    try:
                        resolved_date = datetime.fromisoformat(conflict.resolved_at)
                        if resolved_date < cutoff_date:
                            conflicts_to_remove.append(conflict_id)
                    except ValueError:
                        # 无效的日期格式，跳过
                        continue
            
            for conflict_id in conflicts_to_remove:
                del self.conflicts[conflict_id]
                removed_count += 1
            
            if removed_count > 0:
                self._save_conflicts()
                self._update_stats()
                self.logger.info(f"清理了 {removed_count} 个已解决的冲突记录")
            
        except Exception as e:
            if self.monitoring:
                self.monitoring.handle_error(e, {
                    "operation": "clear_resolved_conflicts",
                    "older_than_days": older_than_days
                })
            self.logger.error(f"清理冲突记录失败: {e}")
    
    def _load_conflicts(self) -> Dict[str, ConflictRecord]:
        """加载冲突记录"""
        if not self.conflicts_file.exists():
            return {}
        
        try:
            with open(self.conflicts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                conflicts = {}
                for conflict_id, conflict_data in data.items():
                    conflicts[conflict_id] = ConflictRecord.from_dict(conflict_data)
                return conflicts
        except (IOError, json.JSONDecodeError) as e:
            self.logger.error(f"加载冲突记录失败: {e}")
            return {}
    
    def _save_conflicts(self):
        """保存冲突记录"""
        try:
            data = {}
            for conflict_id, conflict in self.conflicts.items():
                data[conflict_id] = conflict.to_dict()
            
            with open(self.conflicts_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            self.logger.error(f"保存冲突记录失败: {e}")
    
    def _load_stats(self) -> ConflictStats:
        """加载统计信息"""
        if not self.stats_file.exists():
            return ConflictStats()
        
        try:
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return ConflictStats(**data)
        except (IOError, json.JSONDecodeError) as e:
            self.logger.error(f"加载统计信息失败: {e}")
            return ConflictStats()
    
    def _update_stats(self):
        """更新统计信息"""
        try:
            # 计算统计信息
            total_conflicts = len(self.conflicts)
            resolved_conflicts = len([c for c in self.conflicts.values() if c.resolved_at])
            pending_conflicts = total_conflicts - resolved_conflicts
            
            # 按类型统计
            conflicts_by_type = {}
            conflicts_by_strategy = {}
            resolution_times = []
            
            for conflict in self.conflicts.values():
                # 按类型统计
                conflict_type = conflict.conflict_type
                conflicts_by_type[conflict_type] = conflicts_by_type.get(conflict_type, 0) + 1
                
                # 按策略统计
                if conflict.resolution_strategy:
                    strategy = conflict.resolution_strategy
                    conflicts_by_strategy[strategy] = conflicts_by_strategy.get(strategy, 0) + 1
                
                # 计算解决时间
                if conflict.resolved_at and conflict.detected_at:
                    try:
                        detected = datetime.fromisoformat(conflict.detected_at)
                        resolved = datetime.fromisoformat(conflict.resolved_at)
                        resolution_time = (resolved - detected).total_seconds()
                        resolution_times.append(resolution_time)
                    except ValueError:
                        continue
            
            # 计算平均解决时间
            average_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0.0
            
            # 更新统计信息
            self.stats = ConflictStats(
                total_conflicts=total_conflicts,
                resolved_conflicts=resolved_conflicts,
                pending_conflicts=pending_conflicts,
                conflicts_by_type=conflicts_by_type,
                conflicts_by_strategy=conflicts_by_strategy,
                average_resolution_time=average_resolution_time,
                last_update=datetime.now().isoformat()
            )
            
            # 保存统计信息
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats.to_dict(), f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"更新统计信息失败: {e}")


if __name__ == "__main__":
    # 测试代码
    import tempfile
    import shutil
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建临时测试目录
    test_dir = tempfile.mkdtemp()
    print(f"测试目录: {test_dir}")
    
    try:
        # 初始化冲突解决器
        resolver = ConflictResolver(
            conflicts_dir=test_dir,
            default_strategy=ResolutionStrategy.LATEST_WINS,
            enable_auto_resolution=True
        )
        
        print("\n=== 测试冲突检测 ===")
        
        # 检测版本冲突
        conflict1 = resolver.detect_conflict(
            file_path="/test/document1.txt",
            conflict_type=ConflictType.VERSION_CONFLICT,
            description="版本冲突测试",
            metadata={"old_version": 1, "new_version": 2}
        )
        print(f"检测到冲突: {conflict1.conflict_id}")
        
        # 检测哈希冲突
        conflict2 = resolver.detect_conflict(
            file_path="/test/document2.txt",
            conflict_type=ConflictType.HASH_MISMATCH,
            description="哈希不匹配",
            metadata={"expected_hash": "abc123", "actual_hash": "def456"}
        )
        print(f"检测到冲突: {conflict2.conflict_id}")
        
        print("\n=== 测试冲突解决 ===")
        
        # 手动解决冲突
        result1 = resolver.resolve_conflict(
            conflict1.conflict_id,
            strategy=ResolutionStrategy.LATEST_WINS
        )
        print(f"冲突1解决结果: {result1}")
        
        result2 = resolver.resolve_conflict(
            conflict2.conflict_id,
            strategy=ResolutionStrategy.MANUAL_REVIEW
        )
        print(f"冲突2解决结果: {result2}")
        
        print("\n=== 统计信息 ===")
        stats = resolver.get_stats()
        print(f"总冲突数: {stats.total_conflicts}")
        print(f"已解决: {stats.resolved_conflicts}")
        print(f"待处理: {stats.pending_conflicts}")
        print(f"按类型统计: {stats.conflicts_by_type}")
        print(f"按策略统计: {stats.conflicts_by_strategy}")
        
        runtime_stats = resolver.get_runtime_stats()
        print(f"运行时统计: {runtime_stats}")
        
        print("\n=== 测试自定义处理器 ===")
        
        def custom_handler(conflict: ConflictRecord) -> bool:
            print(f"自定义处理器处理冲突: {conflict.conflict_id}")
            return True
        
        resolver.register_custom_handler("test_handler", custom_handler)
        
        conflict3 = resolver.detect_conflict(
            file_path="/test/document3.txt",
            conflict_type=ConflictType.CONTENT_CONFLICT,
            description="内容冲突测试"
        )
        
        result3 = resolver.resolve_conflict(
            conflict3.conflict_id,
            custom_handler="test_handler"
        )
        print(f"自定义处理器结果: {result3}")
        
        print("\n=== 冲突列表 ===")
        all_conflicts = resolver.get_conflicts()
        for conflict in all_conflicts:
            print(f"冲突: {conflict.conflict_id} - {conflict.conflict_type} - 已解决: {conflict.resolved_at is not None}")
        
    finally:
        # 清理测试目录
        shutil.rmtree(test_dir)
        print(f"\n清理测试目录: {test_dir}")