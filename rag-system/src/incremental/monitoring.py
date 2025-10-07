"""增量更新系统监控和日志模块"""

import os
import sys
import time
import psutil
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import traceback
from contextlib import contextmanager

@dataclass
class MetricData:
    """指标数据"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }

@dataclass
class PerformanceMetrics:
    """性能指标"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    processing_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    active_tasks: int = 0
    queue_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "disk_usage": self.disk_usage,
            "processing_time": self.processing_time,
            "throughput": self.throughput,
            "error_rate": self.error_rate,
            "active_tasks": self.active_tasks,
            "queue_size": self.queue_size,
            "timestamp": datetime.now().isoformat()
        }

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
        
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """记录指标"""
        with self.lock:
            metric = MetricData(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics.append(metric)
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """增加计数器"""
        with self.lock:
            self.counters[name] += value
            self.record_metric(f"{name}_count", self.counters[name], tags)
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """设置仪表盘值"""
        with self.lock:
            self.gauges[name] = value
            self.record_metric(f"{name}_gauge", value, tags)
    
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """记录计时器"""
        with self.lock:
            self.timers[name].append(duration)
            # 保持最近100个计时记录
            if len(self.timers[name]) > 100:
                self.timers[name] = self.timers[name][-100:]
            
            # 计算平均值
            avg_duration = sum(self.timers[name]) / len(self.timers[name])
            self.record_metric(f"{name}_avg_duration", avg_duration, tags)
    
    def get_metrics(self, since: Optional[datetime] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取指标"""
        with self.lock:
            metrics = list(self.metrics)
            
            if since:
                metrics = [m for m in metrics if m.timestamp >= since]
            
            if limit:
                metrics = metrics[-limit:]
            
            return [m.to_dict() for m in metrics]
    
    def get_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        with self.lock:
            return {
                "total_metrics": len(self.metrics),
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "timer_summaries": {
                    name: {
                        "count": len(times),
                        "avg": sum(times) / len(times) if times else 0,
                        "min": min(times) if times else 0,
                        "max": max(times) if times else 0
                    }
                    for name, times in self.timers.items()
                }
            }

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.metrics_collector = MetricsCollector()
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self) -> None:
        """开始监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("性能监控已启动")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("性能监控已停止")
    
    def _monitor_loop(self) -> None:
        """监控循环"""
        while self.is_monitoring:
            try:
                metrics = self._collect_system_metrics()
                
                # 记录系统指标
                self.metrics_collector.set_gauge("cpu_usage", metrics.cpu_usage)
                self.metrics_collector.set_gauge("memory_usage", metrics.memory_usage)
                self.metrics_collector.set_gauge("disk_usage", metrics.disk_usage)
                
                # 检查阈值并发出警告
                self._check_thresholds(metrics)
                
            except Exception as e:
                self.logger.error(f"性能监控错误: {e}")
            
            time.sleep(self.check_interval)
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """收集系统指标"""
        # CPU使用率
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # 磁盘使用率
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        return PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage
        )
    
    def _check_thresholds(self, metrics: PerformanceMetrics) -> None:
        """检查阈值"""
        if metrics.cpu_usage > 80:
            self.logger.warning(f"CPU使用率过高: {metrics.cpu_usage}%")
        
        if metrics.memory_usage > 80:
            self.logger.warning(f"内存使用率过高: {metrics.memory_usage}%")
        
        if metrics.disk_usage > 90:
            self.logger.warning(f"磁盘使用率过高: {metrics.disk_usage}%")
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """获取当前指标"""
        return self._collect_system_metrics()
    
    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取指标历史"""
        since = datetime.now() - timedelta(hours=hours)
        return self.metrics_collector.get_metrics(since=since)

class ErrorHandler:
    """错误处理器"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """处理错误"""
        error_type = type(error).__name__
        error_message = str(error)
        
        with self.lock:
            self.error_counts[error_type] += 1
            
            error_info = {
                "type": error_type,
                "message": error_message,
                "traceback": traceback.format_exc(),
                "context": context or {},
                "timestamp": datetime.now().isoformat(),
                "count": self.error_counts[error_type]
            }
            
            self.error_history.append(error_info)
            
            # 保持历史记录在限制内
            if len(self.error_history) > self.max_history:
                self.error_history = self.error_history[-self.max_history:]
        
        # 记录日志
        self.logger.error(f"{error_type}: {error_message}", exc_info=True)
        
        # 如果错误频繁发生，发出警告
        if self.error_counts[error_type] > 10:
            self.logger.warning(f"错误 {error_type} 发生次数过多: {self.error_counts[error_type]}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        with self.lock:
            return {
                "total_errors": len(self.error_history),
                "error_counts": dict(self.error_counts),
                "recent_errors": self.error_history[-10:] if self.error_history else []
            }
    
    def get_error_rate(self, minutes: int = 60) -> float:
        """获取错误率"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self.lock:
            recent_errors = [
                error for error in self.error_history
                if datetime.fromisoformat(error["timestamp"]) >= cutoff_time
            ]
            
            return len(recent_errors) / max(minutes, 1)

class IncrementalUpdateLogger:
    """增量更新专用日志器"""
    
    def __init__(self, log_dir: str, log_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志级别
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # 创建不同类型的日志器
        self.main_logger = self._create_logger("incremental_update", "main.log")
        self.change_logger = self._create_logger("change_detection", "changes.log")
        self.version_logger = self._create_logger("version_management", "versions.log")
        self.index_logger = self._create_logger("incremental_indexing", "indexing.log")
        self.conflict_logger = self._create_logger("conflict_resolution", "conflicts.log")
        self.api_logger = self._create_logger("api", "api.log")
        
    def _create_logger(self, name: str, filename: str) -> logging.Logger:
        """创建日志器"""
        logger = logging.getLogger(f"incremental_update.{name}")
        logger.setLevel(self.log_level)
        
        # 避免重复添加处理器
        if not logger.handlers:
            # 文件处理器
            file_handler = logging.FileHandler(self.log_dir / filename, encoding='utf-8')
            file_handler.setLevel(self.log_level)
            
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)  # 控制台只显示警告及以上级别
            
            # 格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def log_change_detection(self, message: str, level: str = "INFO", **kwargs) -> None:
        """记录变更检测日志"""
        getattr(self.change_logger, level.lower())(message, extra=kwargs)
    
    def log_version_management(self, message: str, level: str = "INFO", **kwargs) -> None:
        """记录版本管理日志"""
        getattr(self.version_logger, level.lower())(message, extra=kwargs)
    
    def log_incremental_indexing(self, message: str, level: str = "INFO", **kwargs) -> None:
        """记录增量索引日志"""
        getattr(self.index_logger, level.lower())(message, extra=kwargs)
    
    def log_conflict_resolution(self, message: str, level: str = "INFO", **kwargs) -> None:
        """记录冲突解决日志"""
        getattr(self.conflict_logger, level.lower())(message, extra=kwargs)
    
    def log_api_request(self, message: str, level: str = "INFO", **kwargs) -> None:
        """记录API请求日志"""
        getattr(self.api_logger, level.lower())(message, extra=kwargs)
    
    def log_main(self, message: str, level: str = "INFO", **kwargs) -> None:
        """记录主日志"""
        getattr(self.main_logger, level.lower())(message, extra=kwargs)

class MonitoringManager:
    """监控管理器"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self.log_dir = Path(config_dict.get("metadata_directory", "./metadata")) / "logs"
        
        # 初始化组件
        self.logger = IncrementalUpdateLogger(
            str(self.log_dir),
            config_dict.get("log_level", "INFO")
        )
        self.performance_monitor = PerformanceMonitor(
            config_dict.get("health_check_interval", 60)
        )
        self.error_handler = ErrorHandler(
            str(self.log_dir / "errors.log")
        )
        
        # 启动监控
        if config_dict.get("monitoring_enabled", True):
            self.performance_monitor.start_monitoring()
    
    def __del__(self):
        """析构函数"""
        try:
            self.performance_monitor.stop_monitoring()
        except:
            pass
    
    @contextmanager
    def timer(self, operation_name: str):
        """计时上下文管理器"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.performance_monitor.metrics_collector.record_timer(
                operation_name, duration
            )
    
    def log_operation(self, operation: str, details: Dict[str, Any], level: str = "INFO") -> None:
        """记录操作日志"""
        message = f"操作: {operation}, 详情: {json.dumps(details, ensure_ascii=False)}"
        self.logger.log_main(message, level)
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """处理错误"""
        self.error_handler.handle_error(error, context)
    
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        current_metrics = self.performance_monitor.get_current_metrics()
        error_summary = self.error_handler.get_error_summary()
        metrics_summary = self.performance_monitor.metrics_collector.get_summary()
        
        return {
            "status": "healthy" if current_metrics.cpu_usage < 80 and current_metrics.memory_usage < 80 else "warning",
            "performance": current_metrics.to_dict(),
            "errors": error_summary,
            "metrics": metrics_summary,
            "timestamp": datetime.now().isoformat()
        }
    
    def export_logs(self, output_file: str, hours: int = 24) -> None:
        """导出日志"""
        since = datetime.now() - timedelta(hours=hours)
        
        export_data = {
            "export_time": datetime.now().isoformat(),
            "time_range_hours": hours,
            "performance_metrics": self.performance_monitor.get_metrics_history(hours),
            "error_summary": self.error_handler.get_error_summary(),
            "system_health": self.get_system_health()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

# 全局监控管理器实例
_monitoring_manager: Optional[MonitoringManager] = None

def get_monitoring_manager(config: Optional[Dict[str, Any]] = None) -> MonitoringManager:
    """获取全局监控管理器实例"""
    global _monitoring_manager
    if _monitoring_manager is None:
        if config is None:
            raise ValueError("首次调用需要提供配置")
        _monitoring_manager = MonitoringManager(config)
    return _monitoring_manager

def setup_monitoring(config: Dict[str, Any]) -> MonitoringManager:
    """设置监控"""
    global _monitoring_manager
    _monitoring_manager = MonitoringManager(config)
    return _monitoring_manager