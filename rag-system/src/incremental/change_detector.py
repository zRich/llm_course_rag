#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
变更检测器 - ChangeDetector

实现基于MD5哈希的文件变更检测功能
支持文件添加、修改、删除的检测
提供高效的批量检测能力

作者: RAG系统开发团队
日期: 2024-01-15
"""

import hashlib
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# 尝试导入监控模块
try:
    from .monitoring import get_monitoring_manager
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


@dataclass
class FileMetadata:
    """文件元数据"""
    file_path: str
    hash: str
    size: int
    mtime: float
    last_check: str
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FileMetadata':
        """从字典创建实例"""
        return cls(**data)


@dataclass
class ChangeRecord:
    """变更记录"""
    file_path: str
    change_type: str  # added, modified, deleted
    timestamp: str
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    old_size: Optional[int] = None
    new_size: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChangeRecord':
        """从字典创建实例"""
        return cls(**data)


class ChangeDetector:
    """变更检测器
    
    功能:
    1. 计算文件MD5哈希值
    2. 检测文件变更（添加、修改、删除）
    3. 管理文件元数据
    4. 记录变更历史
    """
    
    def __init__(self, 
                 metadata_file: str = "doc_metadata.json",
                 change_log_file: str = "changes.json",
                 chunk_size: int = 4096,
                 max_hash_size: Optional[int] = None,
                 config: Optional[Dict] = None):
        """
        初始化变更检测器
        
        Args:
            metadata_file: 元数据文件路径
            change_log_file: 变更日志文件路径
            chunk_size: 文件读取块大小
            max_hash_size: 最大哈希计算大小（字节）
            config: 配置字典
        """
        self.metadata_file = metadata_file
        self.change_log_file = change_log_file
        self.chunk_size = chunk_size
        self.max_hash_size = max_hash_size
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
        
        # 加载现有元数据
        self.metadata: Dict[str, FileMetadata] = self._load_metadata()
        self.change_history: List[ChangeRecord] = self._load_change_history()
        
        # 统计信息
        self.stats = {
            'files_scanned': 0,
            'files_added': 0,
            'files_modified': 0,
            'files_deleted': 0,
            'files_unchanged': 0,
            'total_scan_time': 0.0,
            'total_hash_time': 0.0
        }
    
    def calculate_file_hash(self, file_path: str) -> str:
        """
        计算文件MD5哈希值
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件的MD5哈希值
        """
        import time
        start_time = time.time()
        
        hash_obj = hashlib.md5()
        total_read = 0
        
        try:
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        break
                    
                    hash_obj.update(chunk)
                    total_read += len(chunk)
                    
                    # 如果设置了最大大小限制
                    if self.max_hash_size and total_read >= self.max_hash_size:
                        break
            
            hash_value = hash_obj.hexdigest()
            hash_time = time.time() - start_time
            self.stats['total_hash_time'] += hash_time
            
            # 记录监控指标
            if self.monitoring:
                self.monitoring.performance_monitor.metrics_collector.record_timer(
                    "file_hash_calculation", hash_time
                )
                self.monitoring.performance_monitor.metrics_collector.record_metric(
                    "file_size_hashed", total_read
                )
            
            return hash_value
            
        except (IOError, OSError) as e:
            error_msg = f"计算文件哈希失败 {file_path}: {e}"
            self.logger.error(error_msg)
            
            if self.monitoring:
                self.monitoring.handle_error(e, {
                    "operation": "calculate_file_hash",
                    "file_path": file_path
                })
            
            return ""
    
    def get_file_info(self, file_path: str) -> Tuple[int, float]:
        """
        获取文件基本信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            (文件大小, 修改时间)
        """
        try:
            stat = os.stat(file_path)
            return stat.st_size, stat.st_mtime
        except (IOError, OSError) as e:
            error_msg = f"获取文件信息失败 {file_path}: {e}"
            self.logger.error(error_msg)
            
            if self.monitoring:
                self.monitoring.handle_error(e, {
                    "operation": "get_file_info",
                    "file_path": file_path
                })
            
            return 0, 0.0
    
    def detect_changes(self, file_paths: List[str], force_scan: bool = False) -> Dict[str, List[str]]:
        """
        检测文件变更
        
        Args:
            file_paths: 要检测的文件路径列表
            force_scan: 是否强制重新扫描
            
        Returns:
            变更结果字典，包含added, modified, deleted, unchanged列表
        """
        import time
        start_time = time.time()
        
        try:
            self.logger.info(f"开始检测文件变更，共{len(file_paths)}个文件")
            
            # 使用监控计时器
            if self.monitoring:
                with self.monitoring.timer("change_detection"):
                    result = self._perform_change_detection(file_paths, force_scan)
            else:
                result = self._perform_change_detection(file_paths, force_scan)
            
            # 更新统计信息
            scan_time = time.time() - start_time
            self.stats['total_scan_time'] += scan_time
            
            # 记录监控日志
            if self.monitoring:
                self.monitoring.log_operation(
                    "change_detection",
                    {
                        "files_processed": len(file_paths),
                        "changes_found": sum(len(changes) for changes in result.values()),
                        "scan_duration": scan_time,
                        "force_scan": force_scan
                    }
                )
                
                # 记录指标
                self.monitoring.performance_monitor.metrics_collector.record_metric(
                    "files_processed", len(file_paths)
                )
                self.monitoring.performance_monitor.metrics_collector.record_metric(
                    "changes_detected", sum(len(changes) for changes in result.values())
                )
            
            self.logger.info(
                f"变更检测完成: 新增{len(result['added'])}, 修改{len(result['modified'])}, "
                f"删除{len(result['deleted'])}, 未变更{len(result['unchanged'])}, "
                f"耗时{scan_time:.2f}秒"
            )
            
            return result
            
        except Exception as e:
            if self.monitoring:
                self.monitoring.handle_error(e, {
                    "operation": "detect_changes",
                    "file_count": len(file_paths),
                    "force_scan": force_scan
                })
            self.logger.error(f"变更检测失败: {e}")
            raise
    
    def _perform_change_detection(self, file_paths: List[str], force_scan: bool) -> Dict[str, List[str]]:
        """执行实际的变更检测"""
        # 当前文件集合
        current_files = set(file_paths)
        # 之前的文件集合
        previous_files = set(self.metadata.keys())
        
        # 集合运算检测变更
        added_files = current_files - previous_files
        deleted_files = previous_files - current_files
        potential_modified = current_files & previous_files
        
        # 详细检测结果
        added = list(added_files)
        deleted = list(deleted_files)
        modified = []
        unchanged = []
        
        # 检测修改的文件
        for file_path in potential_modified:
            self.stats['files_scanned'] += 1
            
            # 获取当前文件信息
            current_size, current_mtime = self.get_file_info(file_path)
            
            # 获取之前的元数据
            old_metadata = self.metadata[file_path]
            
            # 快速检查：大小或修改时间变化
            if (not force_scan and 
                current_size == old_metadata.size and 
                current_mtime == old_metadata.mtime):
                unchanged.append(file_path)
                continue
            
            # 计算当前哈希值
            current_hash = self.calculate_file_hash(file_path)
            if not current_hash:
                continue
            
            # 比较哈希值
            if current_hash != old_metadata.hash:
                modified.append(file_path)
                self.stats['files_modified'] += 1
                
                # 记录变更
                change_record = ChangeRecord(
                    file_path=file_path,
                    change_type="modified",
                    timestamp=datetime.now().isoformat(),
                    old_hash=old_metadata.hash,
                    new_hash=current_hash,
                    old_size=old_metadata.size,
                    new_size=current_size
                )
                self.change_history.append(change_record)
                
                # 更新元数据
                self.metadata[file_path] = FileMetadata(
                    file_path=file_path,
                    hash=current_hash,
                    size=current_size,
                    mtime=current_mtime,
                    last_check=datetime.now().isoformat()
                )
            else:
                unchanged.append(file_path)
                self.stats['files_unchanged'] += 1
                
                # 更新检查时间
                self.metadata[file_path].last_check = datetime.now().isoformat()
        
        # 处理新增文件
        for file_path in added:
            self.stats['files_scanned'] += 1
            self.stats['files_added'] += 1
            
            # 获取文件信息
            current_size, current_mtime = self.get_file_info(file_path)
            current_hash = self.calculate_file_hash(file_path)
            
            if current_hash:
                # 记录变更
                change_record = ChangeRecord(
                    file_path=file_path,
                    change_type="added",
                    timestamp=datetime.now().isoformat(),
                    new_hash=current_hash,
                    new_size=current_size
                )
                self.change_history.append(change_record)
                
                # 添加元数据
                self.metadata[file_path] = FileMetadata(
                    file_path=file_path,
                    hash=current_hash,
                    size=current_size,
                    mtime=current_mtime,
                    last_check=datetime.now().isoformat()
                )
        
        # 处理删除文件
        for file_path in deleted:
            self.stats['files_deleted'] += 1
            
            # 记录变更
            old_metadata = self.metadata[file_path]
            change_record = ChangeRecord(
                file_path=file_path,
                change_type="deleted",
                timestamp=datetime.now().isoformat(),
                old_hash=old_metadata.hash,
                old_size=old_metadata.size
            )
            self.change_history.append(change_record)
            
            # 删除元数据
            del self.metadata[file_path]
        
        # 保存更新后的数据
        self._save_metadata()
        self._save_change_history()
        
        return {
            'added': added,
            'modified': modified,
            'deleted': deleted,
            'unchanged': unchanged
        }


    def get_file_metadata(self, file_path: str) -> Optional[FileMetadata]:
        """
        获取文件元数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件元数据，如果不存在返回None
        """
        return self.metadata.get(file_path)
    
    def get_change_history(self, file_path: Optional[str] = None, 
                          change_type: Optional[str] = None,
                          limit: Optional[int] = None) -> List[ChangeRecord]:
        """
        获取变更历史
        
        Args:
            file_path: 文件路径过滤
            change_type: 变更类型过滤
            limit: 返回数量限制
            
        Returns:
            变更记录列表
        """
        history = self.change_history
        
        # 应用过滤条件
        if file_path:
            history = [r for r in history if r.file_path == file_path]
        
        if change_type:
            history = [r for r in history if r.change_type == change_type]
        
        # 按时间倒序排列
        history.sort(key=lambda x: x.timestamp, reverse=True)
        
        # 应用数量限制
        if limit:
            history = history[:limit]
        
        return history
    
    def get_stats(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            **self.stats,
            'total_files': len(self.metadata),
            'total_changes': len(self.change_history),
            'avg_hash_time': (self.stats['total_hash_time'] / max(1, self.stats['files_scanned'])),
            'avg_scan_time': (self.stats['total_scan_time'] / max(1, 1))
        }
    
    def cleanup_old_changes(self, days: int = 30):
        """
        清理旧的变更记录
        
        Args:
            days: 保留天数
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()
        
        original_count = len(self.change_history)
        self.change_history = [
            record for record in self.change_history 
            if record.timestamp > cutoff_str
        ]
        
        cleaned_count = original_count - len(self.change_history)
        if cleaned_count > 0:
            self.logger.info(f"清理了{cleaned_count}条旧变更记录")
            self._save_change_history()
    
    def _load_metadata(self) -> Dict[str, FileMetadata]:
        """
        加载元数据文件
        
        Returns:
            元数据字典
        """
        if not os.path.exists(self.metadata_file):
            return {}
        
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    path: FileMetadata.from_dict(metadata)
                    for path, metadata in data.items()
                }
        except (IOError, json.JSONDecodeError) as e:
            self.logger.error(f"加载元数据文件失败: {e}")
            return {}
    
    def _save_metadata(self):
        """
        保存元数据文件
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.metadata_file) if os.path.dirname(self.metadata_file) else '.', exist_ok=True)
            
            data = {
                path: metadata.to_dict()
                for path, metadata in self.metadata.items()
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except IOError as e:
            self.logger.error(f"保存元数据文件失败: {e}")
    
    def _load_change_history(self) -> List[ChangeRecord]:
        """
        加载变更历史文件
        
        Returns:
            变更记录列表
        """
        if not os.path.exists(self.change_log_file):
            return []
        
        try:
            with open(self.change_log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [
                    ChangeRecord.from_dict(record)
                    for record in data
                ]
        except (IOError, json.JSONDecodeError) as e:
            self.logger.error(f"加载变更历史文件失败: {e}")
            return []
    
    def _save_change_history(self):
        """
        保存变更历史文件
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.change_log_file) if os.path.dirname(self.change_log_file) else '.', exist_ok=True)
            
            data = [record.to_dict() for record in self.change_history]
            
            with open(self.change_log_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except IOError as e:
            self.logger.error(f"保存变更历史文件失败: {e}")


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
        # 创建测试文件
        test_files = []
        for i in range(3):
            file_path = os.path.join(test_dir, f"test_{i}.txt")
            with open(file_path, 'w') as f:
                f.write(f"这是测试文件 {i}\n内容: {i * 100}")
            test_files.append(file_path)
        
        # 初始化变更检测器
        detector = ChangeDetector(
            metadata_file=os.path.join(test_dir, "metadata.json"),
            change_log_file=os.path.join(test_dir, "changes.json")
        )
        
        # 第一次扫描（全部为新增）
        print("\n=== 第一次扫描 ===")
        changes = detector.detect_changes(test_files)
        print(f"变更结果: {changes}")
        print(f"统计信息: {detector.get_stats()}")
        
        # 修改一个文件
        with open(test_files[1], 'a') as f:
            f.write("\n追加内容")
        
        # 第二次扫描（应该检测到修改）
        print("\n=== 第二次扫描 ===")
        changes = detector.detect_changes(test_files)
        print(f"变更结果: {changes}")
        print(f"统计信息: {detector.get_stats()}")
        
        # 删除一个文件
        os.remove(test_files[2])
        test_files.pop()
        
        # 第三次扫描（应该检测到删除）
        print("\n=== 第三次扫描 ===")
        changes = detector.detect_changes(test_files)
        print(f"变更结果: {changes}")
        print(f"统计信息: {detector.get_stats()}")
        
        # 显示变更历史
        print("\n=== 变更历史 ===")
        history = detector.get_change_history()
        for record in history:
            print(f"{record.timestamp}: {record.file_path} - {record.change_type}")
        
    finally:
        # 清理测试目录
        shutil.rmtree(test_dir)
        print(f"\n清理测试目录: {test_dir}")