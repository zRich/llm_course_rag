#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lesson16 高级RAG优化实现模板
解决缓存策略、性能监控和自适应优化缺失问题

功能特性：
1. 智能缓存策略和管理
2. 实时性能监控和分析
3. 自适应查询优化
4. 动态参数调优
5. 系统资源管理和负载均衡
"""

import logging
import time
import json
import hashlib
import threading
import psutil
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path
import pickle
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """缓存策略枚举"""
    LRU = "lru"  # 最近最少使用
    LFU = "lfu"  # 最少使用频率
    TTL = "ttl"  # 生存时间
    ADAPTIVE = "adaptive"  # 自适应

class OptimizationLevel(Enum):
    """优化级别枚举"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"

class MetricType(Enum):
    """指标类型枚举"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    RESOURCE_USAGE = "resource_usage"
    CACHE_HIT_RATE = "cache_hit_rate"

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[timedelta] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return datetime.now() - self.created_at > self.ttl
    
    def update_access(self):
        """更新访问信息"""
        self.last_accessed = datetime.now()
        self.access_count += 1

@dataclass
class PerformanceMetric:
    """性能指标"""
    metric_type: MetricType
    value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationResult:
    """优化结果"""
    parameter: str
    old_value: Any
    new_value: Any
    improvement: float
    confidence: float
    timestamp: datetime

class SmartCache:
    """智能缓存系统"""
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                 default_ttl: Optional[timedelta] = None):
        self.max_size = max_size
        self.strategy = strategy
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = deque()  # For LRU
        self._frequency_counter = defaultdict(int)  # For LFU
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_size_bytes': 0
        }
        
        # 自适应策略参数
        self._adaptive_weights = {
            'recency': 0.4,
            'frequency': 0.3,
            'size': 0.2,
            'ttl': 0.1
        }
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            
            # 检查是否过期
            if entry.is_expired():
                self._remove_entry(key)
                self._stats['misses'] += 1
                return None
            
            # 更新访问信息
            entry.update_access()
            self._update_access_tracking(key)
            self._stats['hits'] += 1
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """存储缓存值"""
        with self._lock:
            # 计算值的大小
            size_bytes = self._estimate_size(value)
            
            # 如果键已存在，先移除旧条目
            if key in self._cache:
                self._remove_entry(key)
            
            # 检查是否需要腾出空间
            while len(self._cache) >= self.max_size:
                self._evict_entry()
            
            # 创建新条目
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes
            )
            
            self._cache[key] = entry
            self._update_access_tracking(key)
            self._stats['total_size_bytes'] += size_bytes
    
    def _evict_entry(self) -> None:
        """根据策略驱逐条目"""
        if not self._cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            key_to_evict = self._get_lru_key()
        elif self.strategy == CacheStrategy.LFU:
            key_to_evict = self._get_lfu_key()
        elif self.strategy == CacheStrategy.TTL:
            key_to_evict = self._get_ttl_key()
        else:  # ADAPTIVE
            key_to_evict = self._get_adaptive_key()
        
        if key_to_evict:
            self._remove_entry(key_to_evict)
            self._stats['evictions'] += 1
    
    def _get_lru_key(self) -> Optional[str]:
        """获取最近最少使用的键"""
        if not self._access_order:
            return list(self._cache.keys())[0] if self._cache else None
        return self._access_order[0]
    
    def _get_lfu_key(self) -> Optional[str]:
        """获取最少使用频率的键"""
        if not self._cache:
            return None
        
        min_frequency = float('inf')
        lfu_key = None
        
        for key, entry in self._cache.items():
            if entry.access_count < min_frequency:
                min_frequency = entry.access_count
                lfu_key = key
        
        return lfu_key
    
    def _get_ttl_key(self) -> Optional[str]:
        """获取最早过期的键"""
        if not self._cache:
            return None
        
        earliest_expiry = None
        ttl_key = None
        
        for key, entry in self._cache.items():
            if entry.ttl is None:
                continue
            
            expiry_time = entry.created_at + entry.ttl
            if earliest_expiry is None or expiry_time < earliest_expiry:
                earliest_expiry = expiry_time
                ttl_key = key
        
        return ttl_key or list(self._cache.keys())[0]
    
    def _get_adaptive_key(self) -> Optional[str]:
        """自适应选择要驱逐的键"""
        if not self._cache:
            return None
        
        scores = {}
        current_time = datetime.now()
        
        for key, entry in self._cache.items():
            # 计算各个维度的分数
            recency_score = (current_time - entry.last_accessed).total_seconds()
            frequency_score = 1.0 / (entry.access_count + 1)
            size_score = entry.size_bytes
            
            ttl_score = 0
            if entry.ttl:
                remaining_ttl = (entry.created_at + entry.ttl - current_time).total_seconds()
                ttl_score = max(0, -remaining_ttl)  # 负值表示已过期
            
            # 加权综合分数（分数越高越应该被驱逐）
            total_score = (
                self._adaptive_weights['recency'] * recency_score +
                self._adaptive_weights['frequency'] * frequency_score +
                self._adaptive_weights['size'] * size_score +
                self._adaptive_weights['ttl'] * ttl_score
            )
            
            scores[key] = total_score
        
        # 返回分数最高的键
        return max(scores.keys(), key=lambda k: scores[k])
    
    def _remove_entry(self, key: str) -> None:
        """移除缓存条目"""
        if key in self._cache:
            entry = self._cache[key]
            self._stats['total_size_bytes'] -= entry.size_bytes
            del self._cache[key]
        
        # 清理访问跟踪
        if key in self._access_order:
            self._access_order.remove(key)
        if key in self._frequency_counter:
            del self._frequency_counter[key]
    
    def _update_access_tracking(self, key: str) -> None:
        """更新访问跟踪"""
        # 更新LRU跟踪
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        # 更新LFU跟踪
        self._frequency_counter[key] += 1
    
    def _estimate_size(self, value: Any) -> int:
        """估算值的大小"""
        try:
            return len(pickle.dumps(value))
        except:
            # 如果无法序列化，使用简单估算
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in value.items())
            else:
                return 1024  # 默认1KB
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'total_size_bytes': self._stats['total_size_bytes'],
                'strategy': self.strategy.value
            }
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._frequency_counter.clear()
            self._stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'total_size_bytes': 0
            }

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, history_size: int = 10000):
        self.history_size = history_size
        self._metrics: Dict[MetricType, deque] = {
            metric_type: deque(maxlen=history_size) for metric_type in MetricType
        }
        self._lock = threading.RLock()
        self._alert_thresholds = {
            MetricType.LATENCY: 5.0,  # 5秒
            MetricType.THROUGHPUT: 10.0,  # 10 QPS
            MetricType.ACCURACY: 0.8,  # 80%
            MetricType.RESOURCE_USAGE: 0.9,  # 90%
            MetricType.CACHE_HIT_RATE: 0.5  # 50%
        }
        self._alert_callbacks: List[Callable] = []
    
    def record_metric(self, metric_type: MetricType, value: float, 
                     context: Dict[str, Any] = None) -> None:
        """记录性能指标"""
        with self._lock:
            metric = PerformanceMetric(
                metric_type=metric_type,
                value=value,
                timestamp=datetime.now(),
                context=context or {}
            )
            
            self._metrics[metric_type].append(metric)
            
            # 检查是否需要触发告警
            self._check_alerts(metric_type, value)
    
    def get_metrics(self, metric_type: MetricType, 
                   time_window: Optional[timedelta] = None) -> List[PerformanceMetric]:
        """获取指定类型的指标"""
        with self._lock:
            metrics = list(self._metrics[metric_type])
            
            if time_window:
                cutoff_time = datetime.now() - time_window
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
            return metrics
    
    def get_statistics(self, metric_type: MetricType, 
                      time_window: Optional[timedelta] = None) -> Dict[str, float]:
        """获取指标统计信息"""
        metrics = self.get_metrics(metric_type, time_window)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            'count': len(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'median': sorted(values)[len(values) // 2],
            'p95': sorted(values)[int(len(values) * 0.95)] if len(values) > 20 else max(values),
            'p99': sorted(values)[int(len(values) * 0.99)] if len(values) > 100 else max(values)
        }
    
    def _check_alerts(self, metric_type: MetricType, value: float) -> None:
        """检查告警条件"""
        threshold = self._alert_thresholds.get(metric_type)
        if threshold is None:
            return
        
        should_alert = False
        
        if metric_type in [MetricType.LATENCY, MetricType.RESOURCE_USAGE]:
            # 这些指标值越高越不好
            should_alert = value > threshold
        elif metric_type in [MetricType.THROUGHPUT, MetricType.ACCURACY, MetricType.CACHE_HIT_RATE]:
            # 这些指标值越低越不好
            should_alert = value < threshold
        
        if should_alert:
            self._trigger_alert(metric_type, value, threshold)
    
    def _trigger_alert(self, metric_type: MetricType, value: float, threshold: float) -> None:
        """触发告警"""
        alert_info = {
            'metric_type': metric_type.value,
            'value': value,
            'threshold': threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.warning(f"性能告警: {metric_type.value} = {value}, 阈值 = {threshold}")
        
        for callback in self._alert_callbacks:
            try:
                callback(alert_info)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")
    
    def add_alert_callback(self, callback: Callable) -> None:
        """添加告警回调"""
        self._alert_callbacks.append(callback)
    
    def set_threshold(self, metric_type: MetricType, threshold: float) -> None:
        """设置告警阈值"""
        self._alert_thresholds[metric_type] = threshold
    
    def get_system_metrics(self) -> Dict[str, float]:
        """获取系统资源指标"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_usage': cpu_percent / 100.0,
                'memory_usage': memory.percent / 100.0,
                'disk_usage': disk.percent / 100.0,
                'memory_available_gb': memory.available / (1024**3),
                'disk_free_gb': disk.free / (1024**3)
            }
        except Exception as e:
            logger.error(f"获取系统指标失败: {e}")
            return {}

class AdaptiveOptimizer:
    """自适应优化器"""
    
    def __init__(self, monitor: PerformanceMonitor, 
                 optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.monitor = monitor
        self.optimization_level = optimization_level
        self._parameters = {}
        self._optimization_history: List[OptimizationResult] = []
        self._lock = threading.RLock()
        
        # 优化策略配置
        self._strategies = {
            OptimizationLevel.CONSERVATIVE: {
                'min_samples': 100,
                'confidence_threshold': 0.8,
                'max_change_ratio': 0.1
            },
            OptimizationLevel.BALANCED: {
                'min_samples': 50,
                'confidence_threshold': 0.7,
                'max_change_ratio': 0.2
            },
            OptimizationLevel.AGGRESSIVE: {
                'min_samples': 20,
                'confidence_threshold': 0.6,
                'max_change_ratio': 0.5
            }
        }
    
    def register_parameter(self, name: str, current_value: Any, 
                          value_range: Tuple[Any, Any] = None,
                          optimization_target: MetricType = MetricType.LATENCY) -> None:
        """注册可优化参数"""
        with self._lock:
            self._parameters[name] = {
                'current_value': current_value,
                'value_range': value_range,
                'optimization_target': optimization_target,
                'last_optimized': datetime.now(),
                'optimization_count': 0
            }
    
    def optimize_parameters(self) -> List[OptimizationResult]:
        """优化参数"""
        with self._lock:
            results = []
            strategy = self._strategies[self.optimization_level]
            
            for param_name, param_info in self._parameters.items():
                # 检查是否有足够的数据进行优化
                target_metric = param_info['optimization_target']
                recent_metrics = self.monitor.get_metrics(
                    target_metric, 
                    timedelta(minutes=30)
                )
                
                if len(recent_metrics) < strategy['min_samples']:
                    continue
                
                # 分析当前性能
                current_performance = self._analyze_performance(recent_metrics)
                
                # 生成优化建议
                optimization = self._generate_optimization(
                    param_name, param_info, current_performance, strategy
                )
                
                if optimization:
                    results.append(optimization)
                    # 更新参数信息
                    param_info['current_value'] = optimization.new_value
                    param_info['last_optimized'] = datetime.now()
                    param_info['optimization_count'] += 1
            
            self._optimization_history.extend(results)
            return results
    
    def _analyze_performance(self, metrics: List[PerformanceMetric]) -> Dict[str, float]:
        """分析性能数据"""
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        # 计算趋势
        if len(values) >= 10:
            recent_avg = sum(values[-10:]) / 10
            older_avg = sum(values[:-10]) / (len(values) - 10)
            trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        else:
            trend = 0
        
        # 计算变异系数
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        std_dev = variance ** 0.5
        cv = std_dev / mean_val if mean_val > 0 else 0
        
        return {
            'mean': mean_val,
            'std_dev': std_dev,
            'coefficient_of_variation': cv,
            'trend': trend,
            'sample_count': len(values)
        }
    
    def _generate_optimization(self, param_name: str, param_info: Dict,
                              performance: Dict[str, float], 
                              strategy: Dict) -> Optional[OptimizationResult]:
        """生成优化建议"""
        current_value = param_info['current_value']
        value_range = param_info.get('value_range')
        target_metric = param_info['optimization_target']
        
        # 基于性能分析决定优化方向
        if target_metric == MetricType.LATENCY:
            # 延迟需要降低
            improvement_needed = performance['mean'] > 1.0 or performance['trend'] > 0.1
        elif target_metric == MetricType.THROUGHPUT:
            # 吞吐量需要提高
            improvement_needed = performance['mean'] < 100 or performance['trend'] < -0.1
        else:
            improvement_needed = performance['coefficient_of_variation'] > 0.3
        
        if not improvement_needed:
            return None
        
        # 生成新值
        new_value = self._calculate_new_value(
            current_value, value_range, strategy['max_change_ratio']
        )
        
        if new_value == current_value:
            return None
        
        # 估算改进程度和置信度
        improvement = self._estimate_improvement(performance, target_metric)
        confidence = min(performance['sample_count'] / strategy['min_samples'], 1.0)
        
        if confidence < strategy['confidence_threshold']:
            return None
        
        return OptimizationResult(
            parameter=param_name,
            old_value=current_value,
            new_value=new_value,
            improvement=improvement,
            confidence=confidence,
            timestamp=datetime.now()
        )
    
    def _calculate_new_value(self, current_value: Any, value_range: Optional[Tuple], 
                           max_change_ratio: float) -> Any:
        """计算新的参数值"""
        if isinstance(current_value, (int, float)):
            # 数值类型参数
            change = current_value * max_change_ratio
            
            # 随机选择增加或减少（实际应该基于性能分析）
            import random
            direction = random.choice([-1, 1])
            new_value = current_value + direction * change
            
            # 应用范围限制
            if value_range:
                new_value = max(value_range[0], min(value_range[1], new_value))
            
            return type(current_value)(new_value)
        
        elif isinstance(current_value, str):
            # 字符串类型参数（如策略名称）
            # 这里需要根据具体参数类型实现
            return current_value
        
        else:
            return current_value
    
    def _estimate_improvement(self, performance: Dict[str, float], 
                             target_metric: MetricType) -> float:
        """估算改进程度"""
        # 简化的改进估算
        if target_metric == MetricType.LATENCY:
            # 基于当前延迟和变异性估算可能的改进
            base_improvement = min(0.3, performance['coefficient_of_variation'])
            trend_penalty = max(0, performance['trend']) * 0.1
            return max(0.05, base_improvement - trend_penalty)
        
        elif target_metric == MetricType.THROUGHPUT:
            # 基于当前吞吐量趋势估算改进
            return max(0.05, min(0.5, -performance['trend'] * 2))
        
        else:
            return 0.1  # 默认10%改进
    
    def get_optimization_history(self, limit: int = 100) -> List[OptimizationResult]:
        """获取优化历史"""
        with self._lock:
            return self._optimization_history[-limit:]
    
    def get_parameter_status(self) -> Dict[str, Dict]:
        """获取参数状态"""
        with self._lock:
            return dict(self._parameters)

class RAGOptimizer:
    """RAG系统优化器"""
    
    def __init__(self, cache_size: int = 10000, 
                 optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.cache = SmartCache(max_size=cache_size, strategy=CacheStrategy.ADAPTIVE)
        self.monitor = PerformanceMonitor()
        self.optimizer = AdaptiveOptimizer(self.monitor, optimization_level)
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._optimization_interval = 300  # 5分钟
        self._last_optimization = datetime.now()
        
        # 注册可优化参数
        self._register_default_parameters()
        
        # 启动后台优化任务
        self._start_background_optimization()
    
    def _register_default_parameters(self):
        """注册默认可优化参数"""
        self.optimizer.register_parameter(
            'cache_size', 
            self.cache.max_size, 
            (1000, 50000),
            MetricType.CACHE_HIT_RATE
        )
        
        self.optimizer.register_parameter(
            'thread_pool_size',
            self._executor._max_workers,
            (2, 16),
            MetricType.THROUGHPUT
        )
    
    def _start_background_optimization(self):
        """启动后台优化任务"""
        def optimization_loop():
            while True:
                try:
                    time.sleep(60)  # 每分钟检查一次
                    
                    if (datetime.now() - self._last_optimization).total_seconds() >= self._optimization_interval:
                        self._run_optimization()
                        self._last_optimization = datetime.now()
                        
                except Exception as e:
                    logger.error(f"后台优化任务失败: {e}")
        
        optimization_thread = threading.Thread(target=optimization_loop, daemon=True)
        optimization_thread.start()
    
    def _run_optimization(self):
        """运行优化"""
        logger.info("开始自动优化...")
        
        # 记录系统指标
        system_metrics = self.monitor.get_system_metrics()
        for metric_name, value in system_metrics.items():
            if 'usage' in metric_name:
                self.monitor.record_metric(MetricType.RESOURCE_USAGE, value)
        
        # 记录缓存指标
        cache_stats = self.cache.get_stats()
        self.monitor.record_metric(MetricType.CACHE_HIT_RATE, cache_stats['hit_rate'])
        
        # 执行参数优化
        optimizations = self.optimizer.optimize_parameters()
        
        if optimizations:
            logger.info(f"完成 {len(optimizations)} 项优化")
            for opt in optimizations:
                logger.info(f"  {opt.parameter}: {opt.old_value} -> {opt.new_value} "
                          f"(预期改进: {opt.improvement:.1%}, 置信度: {opt.confidence:.1%})")
                
                # 应用优化
                self._apply_optimization(opt)
        else:
            logger.info("未发现需要优化的参数")
    
    def _apply_optimization(self, optimization: OptimizationResult):
        """应用优化结果"""
        param_name = optimization.parameter
        new_value = optimization.new_value
        
        try:
            if param_name == 'cache_size':
                # 调整缓存大小需要重新创建缓存
                old_cache = self.cache
                self.cache = SmartCache(
                    max_size=int(new_value),
                    strategy=old_cache.strategy,
                    default_ttl=old_cache.default_ttl
                )
                logger.info(f"缓存大小已调整为: {new_value}")
                
            elif param_name == 'thread_pool_size':
                # 调整线程池大小
                old_executor = self._executor
                self._executor = ThreadPoolExecutor(max_workers=int(new_value))
                old_executor.shutdown(wait=False)
                logger.info(f"线程池大小已调整为: {new_value}")
                
        except Exception as e:
            logger.error(f"应用优化失败 {param_name}: {e}")
    
    def cached_operation(self, operation_key: str, operation_func: Callable, 
                        *args, ttl: Optional[timedelta] = None, **kwargs) -> Any:
        """缓存操作结果"""
        # 生成缓存键
        cache_key = self._generate_cache_key(operation_key, args, kwargs)
        
        # 尝试从缓存获取
        start_time = time.time()
        cached_result = self.cache.get(cache_key)
        
        if cached_result is not None:
            # 缓存命中
            latency = time.time() - start_time
            self.monitor.record_metric(MetricType.LATENCY, latency, {'cache_hit': True})
            return cached_result
        
        # 缓存未命中，执行操作
        try:
            result = operation_func(*args, **kwargs)
            operation_latency = time.time() - start_time
            
            # 存储到缓存
            self.cache.put(cache_key, result, ttl)
            
            # 记录指标
            self.monitor.record_metric(MetricType.LATENCY, operation_latency, {'cache_hit': False})
            
            return result
            
        except Exception as e:
            error_latency = time.time() - start_time
            self.monitor.record_metric(MetricType.LATENCY, error_latency, {'error': True})
            raise
    
    def _generate_cache_key(self, operation_key: str, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        # 创建参数的哈希
        param_str = f"{args}_{sorted(kwargs.items())}"
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:16]
        return f"{operation_key}_{param_hash}"
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'cache_stats': self.cache.get_stats(),
            'system_metrics': self.monitor.get_system_metrics(),
            'optimization_history': [{
                'parameter': opt.parameter,
                'improvement': opt.improvement,
                'confidence': opt.confidence,
                'timestamp': opt.timestamp.isoformat()
            } for opt in self.optimizer.get_optimization_history(10)]
        }
        
        # 添加各类指标统计
        for metric_type in MetricType:
            stats = self.monitor.get_statistics(metric_type, timedelta(hours=1))
            if stats:
                report[f'{metric_type.value}_stats'] = stats
        
        return report
    
    def set_optimization_interval(self, seconds: int):
        """设置优化间隔"""
        self._optimization_interval = seconds
    
    def force_optimization(self) -> List[OptimizationResult]:
        """强制执行优化"""
        self._run_optimization()
        return self.optimizer.get_optimization_history(10)

def main():
    """示例用法"""
    print("高级RAG优化系统测试\n" + "="*50)
    
    # 创建优化器
    rag_optimizer = RAGOptimizer(
        cache_size=5000,
        optimization_level=OptimizationLevel.BALANCED
    )
    
    # 模拟一些操作
    def expensive_operation(query: str, param: int = 1) -> str:
        """模拟耗时操作"""
        time.sleep(0.1)  # 模拟处理时间
        return f"Result for '{query}' with param {param}"
    
    print("执行缓存操作测试...")
    
    # 测试缓存操作
    queries = ["机器学习", "深度学习", "自然语言处理", "计算机视觉"]
    
    for i in range(20):
        query = queries[i % len(queries)]
        
        start_time = time.time()
        result = rag_optimizer.cached_operation(
            "search_operation",
            expensive_operation,
            query,
            param=i % 3
        )
        end_time = time.time()
        
        print(f"查询 '{query}': {end_time - start_time:.3f}s")
        
        # 记录吞吐量指标
        rag_optimizer.monitor.record_metric(
            MetricType.THROUGHPUT, 
            1.0 / (end_time - start_time)
        )
    
    # 等待一段时间让监控收集数据
    time.sleep(2)
    
    # 强制执行优化
    print("\n执行性能优化...")
    optimizations = rag_optimizer.force_optimization()
    
    if optimizations:
        print(f"完成 {len(optimizations)} 项优化:")
        for opt in optimizations:
            print(f"  {opt.parameter}: {opt.old_value} -> {opt.new_value}")
    else:
        print("未发现需要优化的参数")
    
    # 生成性能报告
    print("\n性能报告:")
    report = rag_optimizer.get_performance_report()
    
    print(f"缓存命中率: {report['cache_stats']['hit_rate']:.1%}")
    print(f"缓存大小: {report['cache_stats']['size']}/{report['cache_stats']['max_size']}")
    
    if 'latency_stats' in report:
        latency_stats = report['latency_stats']
        print(f"平均延迟: {latency_stats['mean']:.3f}s")
        print(f"P95延迟: {latency_stats['p95']:.3f}s")
    
    if 'throughput_stats' in report:
        throughput_stats = report['throughput_stats']
        print(f"平均吞吐量: {throughput_stats['mean']:.1f} QPS")
    
    system_metrics = report['system_metrics']
    if system_metrics:
        print(f"CPU使用率: {system_metrics.get('cpu_usage', 0):.1%}")
        print(f"内存使用率: {system_metrics.get('memory_usage', 0):.1%}")

if __name__ == "__main__":
    main()