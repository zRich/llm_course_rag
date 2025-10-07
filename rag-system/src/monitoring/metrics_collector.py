"""指标收集器模块

收集、存储和分析系统运行指标。
"""

import time
import threading
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"          # 计数器
    GAUGE = "gauge"              # 仪表盘
    HISTOGRAM = "histogram"      # 直方图
    TIMER = "timer"              # 计时器
    RATE = "rate"                # 速率


@dataclass
class MetricRecord:
    """指标记录"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricAggregator:
    """指标聚合器"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self._lock = threading.RLock()
    
    def add_value(self, value: float, timestamp: Optional[float] = None):
        """添加值"""
        with self._lock:
            if timestamp is None:
                timestamp = time.time()
            
            self.values.append(value)
            self.timestamps.append(timestamp)
    
    def get_statistics(self, window_seconds: Optional[float] = None) -> Dict[str, float]:
        """获取统计信息"""
        with self._lock:
            if not self.values:
                return {
                    'count': 0,
                    'sum': 0.0,
                    'avg': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'median': 0.0,
                    'p95': 0.0,
                    'p99': 0.0
                }
            
            # 如果指定了时间窗口，过滤数据
            values = list(self.values)
            timestamps = list(self.timestamps)
            
            if window_seconds is not None:
                current_time = time.time()
                cutoff_time = current_time - window_seconds
                
                filtered_data = [(v, t) for v, t in zip(values, timestamps) if t >= cutoff_time]
                if filtered_data:
                    values, timestamps = zip(*filtered_data)
                    values = list(values)
                else:
                    values = []
            
            if not values:
                return {
                    'count': 0,
                    'sum': 0.0,
                    'avg': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'median': 0.0,
                    'p95': 0.0,
                    'p99': 0.0
                }
            
            count = len(values)
            total = sum(values)
            avg = total / count
            min_val = min(values)
            max_val = max(values)
            
            # 计算百分位数
            sorted_values = sorted(values)
            median = statistics.median(sorted_values)
            
            p95_idx = int(0.95 * count)
            p99_idx = int(0.99 * count)
            
            p95 = sorted_values[min(p95_idx, count - 1)] if count > 0 else 0.0
            p99 = sorted_values[min(p99_idx, count - 1)] if count > 0 else 0.0
            
            return {
                'count': count,
                'sum': total,
                'avg': avg,
                'min': min_val,
                'max': max_val,
                'median': median,
                'p95': p95,
                'p99': p99
            }
    
    def get_rate(self, window_seconds: float = 60.0) -> float:
        """获取速率（每秒）"""
        with self._lock:
            if len(self.values) < 2:
                return 0.0
            
            current_time = time.time()
            cutoff_time = current_time - window_seconds
            
            # 计算时间窗口内的事件数
            count = sum(1 for t in self.timestamps if t >= cutoff_time)
            
            return count / window_seconds


class MetricsCollector:
    """指标收集器
    
    收集、存储和分析系统运行指标。
    """
    
    def __init__(self, retention_hours: float = 24.0):
        """
        初始化指标收集器
        
        Args:
            retention_hours: 数据保留时间（小时）
        """
        self.retention_hours = retention_hours
        self.retention_seconds = retention_hours * 3600
        
        # 存储不同类型的指标
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, MetricAggregator] = {}
        self.timers: Dict[str, MetricAggregator] = {}
        self.rates: Dict[str, MetricAggregator] = {}
        
        # 存储所有指标记录
        self.records: List[MetricRecord] = []
        
        # 指标元数据
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}
        
        self._lock = threading.RLock()
        
        # 启动清理线程
        self._cleanup_thread = threading.Thread(target=self._cleanup_old_data, daemon=True)
        self._cleanup_thread.start()
        
        logger.info(f"Metrics collector initialized with {retention_hours}h retention")
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """增加计数器
        
        Args:
            name: 指标名称
            value: 增加值
            tags: 标签
        """
        with self._lock:
            self.counters[name] += value
            
            record = MetricRecord(
                name=name,
                value=value,
                metric_type=MetricType.COUNTER,
                timestamp=time.time(),
                tags=tags or {}
            )
            self.records.append(record)
            
            logger.debug(f"Counter '{name}' incremented by {value}, total: {self.counters[name]}")
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """设置仪表盘值
        
        Args:
            name: 指标名称
            value: 设置值
            tags: 标签
        """
        with self._lock:
            self.gauges[name] = value
            
            record = MetricRecord(
                name=name,
                value=value,
                metric_type=MetricType.GAUGE,
                timestamp=time.time(),
                tags=tags or {}
            )
            self.records.append(record)
            
            logger.debug(f"Gauge '{name}' set to {value}")
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """记录直方图值
        
        Args:
            name: 指标名称
            value: 记录值
            tags: 标签
        """
        with self._lock:
            if name not in self.histograms:
                self.histograms[name] = MetricAggregator()
            
            timestamp = time.time()
            self.histograms[name].add_value(value, timestamp)
            
            record = MetricRecord(
                name=name,
                value=value,
                metric_type=MetricType.HISTOGRAM,
                timestamp=timestamp,
                tags=tags or {}
            )
            self.records.append(record)
            
            logger.debug(f"Histogram '{name}' recorded value: {value}")
    
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """记录计时器值
        
        Args:
            name: 指标名称
            duration: 持续时间（秒）
            tags: 标签
        """
        with self._lock:
            if name not in self.timers:
                self.timers[name] = MetricAggregator()
            
            timestamp = time.time()
            self.timers[name].add_value(duration, timestamp)
            
            record = MetricRecord(
                name=name,
                value=duration,
                metric_type=MetricType.TIMER,
                timestamp=timestamp,
                tags=tags or {}
            )
            self.records.append(record)
            
            logger.debug(f"Timer '{name}' recorded duration: {duration}s")
    
    def record_rate(self, name: str, count: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """记录速率事件
        
        Args:
            name: 指标名称
            count: 事件数量
            tags: 标签
        """
        with self._lock:
            if name not in self.rates:
                self.rates[name] = MetricAggregator()
            
            timestamp = time.time()
            self.rates[name].add_value(count, timestamp)
            
            record = MetricRecord(
                name=name,
                value=count,
                metric_type=MetricType.RATE,
                timestamp=timestamp,
                tags=tags or {}
            )
            self.records.append(record)
            
            logger.debug(f"Rate '{name}' recorded count: {count}")
    
    def get_counter(self, name: str) -> float:
        """获取计数器值"""
        with self._lock:
            return self.counters.get(name, 0.0)
    
    def get_gauge(self, name: str) -> Optional[float]:
        """获取仪表盘值"""
        with self._lock:
            return self.gauges.get(name)
    
    def get_histogram_stats(self, name: str, window_seconds: Optional[float] = None) -> Optional[Dict[str, float]]:
        """获取直方图统计"""
        with self._lock:
            if name not in self.histograms:
                return None
            return self.histograms[name].get_statistics(window_seconds)
    
    def get_timer_stats(self, name: str, window_seconds: Optional[float] = None) -> Optional[Dict[str, float]]:
        """获取计时器统计"""
        with self._lock:
            if name not in self.timers:
                return None
            return self.timers[name].get_statistics(window_seconds)
    
    def get_rate_value(self, name: str, window_seconds: float = 60.0) -> Optional[float]:
        """获取速率值"""
        with self._lock:
            if name not in self.rates:
                return None
            return self.rates[name].get_rate(window_seconds)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        with self._lock:
            metrics = {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histograms': {},
                'timers': {},
                'rates': {}
            }
            
            # 获取聚合指标的统计信息
            for name, aggregator in self.histograms.items():
                metrics['histograms'][name] = aggregator.get_statistics()
            
            for name, aggregator in self.timers.items():
                metrics['timers'][name] = aggregator.get_statistics()
            
            for name, aggregator in self.rates.items():
                metrics['rates'][name] = {
                    'rate_1m': aggregator.get_rate(60),
                    'rate_5m': aggregator.get_rate(300),
                    'rate_15m': aggregator.get_rate(900)
                }
            
            return metrics
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        with self._lock:
            return {
                'total_records': len(self.records),
                'counters_count': len(self.counters),
                'gauges_count': len(self.gauges),
                'histograms_count': len(self.histograms),
                'timers_count': len(self.timers),
                'rates_count': len(self.rates),
                'retention_hours': self.retention_hours,
                'oldest_record': min([r.timestamp for r in self.records]) if self.records else None,
                'newest_record': max([r.timestamp for r in self.records]) if self.records else None
            }
    
    def query_records(self, metric_name: Optional[str] = None, 
                     metric_type: Optional[MetricType] = None,
                     start_time: Optional[float] = None,
                     end_time: Optional[float] = None,
                     tags: Optional[Dict[str, str]] = None) -> List[MetricRecord]:
        """查询指标记录
        
        Args:
            metric_name: 指标名称过滤
            metric_type: 指标类型过滤
            start_time: 开始时间
            end_time: 结束时间
            tags: 标签过滤
            
        Returns:
            匹配的指标记录列表
        """
        with self._lock:
            filtered_records = self.records
            
            # 按名称过滤
            if metric_name:
                filtered_records = [r for r in filtered_records if r.name == metric_name]
            
            # 按类型过滤
            if metric_type:
                filtered_records = [r for r in filtered_records if r.metric_type == metric_type]
            
            # 按时间过滤
            if start_time:
                filtered_records = [r for r in filtered_records if r.timestamp >= start_time]
            
            if end_time:
                filtered_records = [r for r in filtered_records if r.timestamp <= end_time]
            
            # 按标签过滤
            if tags:
                def matches_tags(record_tags: Dict[str, str]) -> bool:
                    return all(record_tags.get(k) == v for k, v in tags.items())
                
                filtered_records = [r for r in filtered_records if matches_tags(r.tags)]
            
            return filtered_records
    
    def reset_counter(self, name: str):
        """重置计数器"""
        with self._lock:
            if name in self.counters:
                self.counters[name] = 0.0
                logger.info(f"Counter '{name}' reset to 0")
    
    def reset_all_counters(self):
        """重置所有计数器"""
        with self._lock:
            for name in self.counters:
                self.counters[name] = 0.0
            logger.info("All counters reset to 0")
    
    def clear_metrics(self):
        """清空所有指标"""
        with self._lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.timers.clear()
            self.rates.clear()
            self.records.clear()
            logger.info("All metrics cleared")
    
    def _cleanup_old_data(self):
        """清理过期数据"""
        while True:
            try:
                time.sleep(300)  # 每5分钟清理一次
                
                current_time = time.time()
                cutoff_time = current_time - self.retention_seconds
                
                with self._lock:
                    # 清理记录
                    old_count = len(self.records)
                    self.records = [r for r in self.records if r.timestamp >= cutoff_time]
                    new_count = len(self.records)
                    
                    if old_count > new_count:
                        logger.debug(f"Cleaned up {old_count - new_count} old metric records")
                
            except Exception as e:
                logger.error(f"Error during metrics cleanup: {str(e)}")


class TimerContext:
    """计时器上下文管理器"""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.record_timer(self.name, duration, self.tags)


def timer_decorator(collector: MetricsCollector, name: str, tags: Optional[Dict[str, str]] = None):
    """计时器装饰器
    
    Args:
        collector: 指标收集器
        name: 指标名称
        tags: 标签
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with TimerContext(collector, name, tags):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# 全局指标收集器实例
global_metrics_collector = MetricsCollector()