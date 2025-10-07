#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lesson18 生产环境监控实现模板
解决日志管理、性能分析和故障诊断缺失问题

功能特性：
1. 结构化日志管理和分析
2. 实时性能监控和分析
3. 自动故障检测和诊断
4. 分布式链路追踪
5. 业务指标监控和报警
"""

import logging
import time
import json
import os
import uuid
import threading
import traceback
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import asyncio
import aiohttp
import psutil
import numpy as np
from collections import defaultdict, deque
import sqlite3
import redis
from elasticsearch import Elasticsearch
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
import structlog
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.flask import FlaskInstrumentor
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify
import requests

# 配置结构化日志
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """指标类型枚举"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    """告警严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class LogEntry:
    """日志条目"""
    timestamp: datetime
    level: LogLevel
    message: str
    service: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    stack_trace: Optional[str] = None

@dataclass
class MetricPoint:
    """指标数据点"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class PerformanceMetrics:
    """性能指标"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    response_time: float
    throughput: float
    error_rate: float
    timestamp: datetime

@dataclass
class TraceSpan:
    """链路追踪跨度"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"

@dataclass
class Alert:
    """告警信息"""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    timestamp: datetime
    source: str
    metric_name: str
    current_value: float
    threshold: float
    status: str = "active"
    resolved_at: Optional[datetime] = None

class StructuredLogger:
    """结构化日志管理器"""
    
    def __init__(self, service_name: str, log_level: LogLevel = LogLevel.INFO):
        self.service_name = service_name
        self.log_level = log_level
        self.logger = structlog.get_logger(service_name)
        self.log_storage: List[LogEntry] = []
        self.max_logs = 10000
        self._lock = threading.RLock()
        
        # 配置文件日志
        self._setup_file_logging()
        
        # 配置Elasticsearch（如果可用）
        self.es_client = self._setup_elasticsearch()
    
    def _setup_file_logging(self):
        """设置文件日志"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(
            log_dir / f"{self.service_name}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, self.log_level.value.upper()))
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # 添加到根日志器
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.setLevel(getattr(logging, self.log_level.value.upper()))
    
    def _setup_elasticsearch(self) -> Optional[Elasticsearch]:
        """设置Elasticsearch客户端"""
        try:
            es_host = os.getenv('ELASTICSEARCH_HOST', 'localhost:9200')
            es_client = Elasticsearch([es_host])
            
            # 测试连接
            if es_client.ping():
                logger.info("Elasticsearch连接成功", host=es_host)
                return es_client
            else:
                logger.warning("Elasticsearch连接失败", host=es_host)
                return None
        except Exception as e:
            logger.warning("Elasticsearch初始化失败", error=str(e))
            return None
    
    def log(self, level: LogLevel, message: str, **kwargs):
        """记录日志"""
        # 获取调用上下文
        trace_id = kwargs.pop('trace_id', None)
        span_id = kwargs.pop('span_id', None)
        user_id = kwargs.pop('user_id', None)
        request_id = kwargs.pop('request_id', None)
        
        # 创建日志条目
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            service=self.service_name,
            trace_id=trace_id,
            span_id=span_id,
            user_id=user_id,
            request_id=request_id,
            metadata=kwargs
        )
        
        # 存储到内存
        with self._lock:
            self.log_storage.append(log_entry)
            if len(self.log_storage) > self.max_logs:
                self.log_storage.pop(0)
        
        # 记录到结构化日志
        log_data = {
            'service': self.service_name,
            'trace_id': trace_id,
            'span_id': span_id,
            'user_id': user_id,
            'request_id': request_id,
            **kwargs
        }
        
        getattr(self.logger, level.value)(message, **log_data)
        
        # 发送到Elasticsearch
        if self.es_client:
            self._send_to_elasticsearch(log_entry)
    
    def _send_to_elasticsearch(self, log_entry: LogEntry):
        """发送日志到Elasticsearch"""
        try:
            doc = {
                '@timestamp': log_entry.timestamp.isoformat(),
                'level': log_entry.level.value,
                'message': log_entry.message,
                'service': log_entry.service,
                'trace_id': log_entry.trace_id,
                'span_id': log_entry.span_id,
                'user_id': log_entry.user_id,
                'request_id': log_entry.request_id,
                'metadata': log_entry.metadata
            }
            
            index_name = f"rag-logs-{datetime.now().strftime('%Y.%m.%d')}"
            self.es_client.index(index=index_name, body=doc)
            
        except Exception as e:
            # 避免日志记录失败影响主业务
            print(f"发送日志到Elasticsearch失败: {e}")
    
    def info(self, message: str, **kwargs):
        """记录信息日志"""
        self.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """记录错误日志"""
        self.log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """记录严重错误日志"""
        self.log(LogLevel.CRITICAL, message, **kwargs)
    
    def search_logs(self, query: str = None, level: LogLevel = None, 
                   start_time: datetime = None, end_time: datetime = None,
                   limit: int = 100) -> List[LogEntry]:
        """搜索日志"""
        with self._lock:
            results = list(self.log_storage)
        
        # 应用过滤条件
        if level:
            results = [log for log in results if log.level == level]
        
        if start_time:
            results = [log for log in results if log.timestamp >= start_time]
        
        if end_time:
            results = [log for log in results if log.timestamp <= end_time]
        
        if query:
            query_lower = query.lower()
            results = [
                log for log in results 
                if query_lower in log.message.lower() or 
                   any(query_lower in str(v).lower() for v in log.metadata.values())
            ]
        
        # 按时间倒序排列
        results.sort(key=lambda x: x.timestamp, reverse=True)
        
        return results[:limit]

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.registry = CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.RLock()
        
        # 初始化基础指标
        self._init_basic_metrics()
        
        # 启动系统指标收集
        self._start_system_metrics_collection()
    
    def _init_basic_metrics(self):
        """初始化基础指标"""
        # 请求计数器
        self.request_count = Counter(
            'requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        # 请求延迟
        self.request_duration = Histogram(
            'request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # 活跃连接数
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        # 系统资源指标
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'disk_usage_percent',
            'Disk usage percentage',
            registry=self.registry
        )
        
        # 业务指标
        self.search_requests = Counter(
            'search_requests_total',
            'Total search requests',
            ['query_type', 'status'],
            registry=self.registry
        )
        
        self.search_latency = Histogram(
            'search_latency_seconds',
            'Search latency in seconds',
            ['query_type'],
            registry=self.registry
        )
        
        self.vector_db_operations = Counter(
            'vector_db_operations_total',
            'Vector database operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        self.cache_hit_rate = Gauge(
            'cache_hit_rate',
            'Cache hit rate',
            registry=self.registry
        )
    
    def _start_system_metrics_collection(self):
        """启动系统指标收集"""
        def collect_system_metrics():
            while True:
                try:
                    # CPU使用率
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.cpu_usage.set(cpu_percent)
                    
                    # 内存使用率
                    memory = psutil.virtual_memory()
                    self.memory_usage.set(memory.percent)
                    
                    # 磁盘使用率
                    disk = psutil.disk_usage('/')
                    self.disk_usage.set(disk.percent)
                    
                    # 记录历史数据
                    timestamp = datetime.now()
                    with self._lock:
                        self.metric_history['cpu_usage'].append((timestamp, cpu_percent))
                        self.metric_history['memory_usage'].append((timestamp, memory.percent))
                        self.metric_history['disk_usage'].append((timestamp, disk.percent))
                    
                    time.sleep(30)  # 每30秒收集一次
                    
                except Exception as e:
                    logger.error("系统指标收集失败", error=str(e))
                    time.sleep(5)
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """记录请求指标"""
        self.request_count.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_search(self, query_type: str, status: str, latency: float):
        """记录搜索指标"""
        self.search_requests.labels(query_type=query_type, status=status).inc()
        self.search_latency.labels(query_type=query_type).observe(latency)
    
    def record_vector_db_operation(self, operation: str, status: str):
        """记录向量数据库操作"""
        self.vector_db_operations.labels(operation=operation, status=status).inc()
    
    def update_cache_hit_rate(self, hit_rate: float):
        """更新缓存命中率"""
        self.cache_hit_rate.set(hit_rate)
    
    def get_metrics_data(self) -> str:
        """获取Prometheus格式的指标数据"""
        return generate_latest(self.registry)
    
    def get_metric_history(self, metric_name: str, duration: timedelta = timedelta(hours=1)) -> List[tuple]:
        """获取指标历史数据"""
        cutoff_time = datetime.now() - duration
        
        with self._lock:
            history = self.metric_history.get(metric_name, deque())
            return [(ts, value) for ts, value in history if ts >= cutoff_time]
    
    def get_performance_summary(self) -> PerformanceMetrics:
        """获取性能摘要"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # 计算网络IO
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # 获取最近的响应时间和吞吐量（简化计算）
            response_time = self._calculate_avg_response_time()
            throughput = self._calculate_throughput()
            error_rate = self._calculate_error_rate()
            
            return PerformanceMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_io=network_io,
                response_time=response_time,
                throughput=throughput,
                error_rate=error_rate,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error("获取性能摘要失败", error=str(e))
            return PerformanceMetrics(
                cpu_usage=0, memory_usage=0, disk_usage=0,
                network_io={}, response_time=0, throughput=0,
                error_rate=0, timestamp=datetime.now()
            )
    
    def _calculate_avg_response_time(self) -> float:
        """计算平均响应时间"""
        # 简化实现：从Prometheus指标中获取
        try:
            # 这里应该从实际的指标数据中计算
            return 0.1  # 示例值
        except:
            return 0.0
    
    def _calculate_throughput(self) -> float:
        """计算吞吐量（请求/秒）"""
        try:
            # 这里应该从实际的指标数据中计算
            return 100.0  # 示例值
        except:
            return 0.0
    
    def _calculate_error_rate(self) -> float:
        """计算错误率"""
        try:
            # 这里应该从实际的指标数据中计算
            return 0.01  # 示例值：1%错误率
        except:
            return 0.0

class DistributedTracer:
    """分布式链路追踪"""
    
    def __init__(self, service_name: str, jaeger_endpoint: str = None):
        self.service_name = service_name
        self.spans: Dict[str, TraceSpan] = {}
        self.active_spans: Dict[str, str] = {}  # thread_id -> span_id
        self._lock = threading.RLock()
        
        # 设置OpenTelemetry
        self._setup_opentelemetry(jaeger_endpoint)
    
    def _setup_opentelemetry(self, jaeger_endpoint: str = None):
        """设置OpenTelemetry追踪"""
        try:
            # 设置追踪提供者
            trace.set_tracer_provider(TracerProvider())
            tracer = trace.get_tracer(__name__)
            
            # 配置Jaeger导出器
            if jaeger_endpoint:
                jaeger_exporter = JaegerExporter(
                    agent_host_name="localhost",
                    agent_port=6831,
                )
                
                span_processor = BatchSpanProcessor(jaeger_exporter)
                trace.get_tracer_provider().add_span_processor(span_processor)
            
            # 自动仪表化
            RequestsInstrumentor().instrument()
            
            self.tracer = tracer
            logger.info("OpenTelemetry追踪已初始化")
            
        except Exception as e:
            logger.warning("OpenTelemetry初始化失败", error=str(e))
            self.tracer = None
    
    @contextmanager
    def start_span(self, operation_name: str, parent_span_id: str = None, **tags):
        """启动新的跨度"""
        span_id = str(uuid.uuid4())
        trace_id = parent_span_id or str(uuid.uuid4())
        
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.now(),
            end_time=None,
            duration=None,
            tags=tags
        )
        
        with self._lock:
            self.spans[span_id] = span
            thread_id = threading.get_ident()
            self.active_spans[thread_id] = span_id
        
        try:
            yield span
        except Exception as e:
            span.status = "error"
            span.tags['error'] = True
            span.tags['error.message'] = str(e)
            span.logs.append({
                'timestamp': datetime.now().isoformat(),
                'level': 'error',
                'message': str(e),
                'stack_trace': traceback.format_exc()
            })
            raise
        finally:
            # 结束跨度
            span.end_time = datetime.now()
            span.duration = (span.end_time - span.start_time).total_seconds()
            
            with self._lock:
                thread_id = threading.get_ident()
                if thread_id in self.active_spans:
                    del self.active_spans[thread_id]
    
    def get_current_span(self) -> Optional[TraceSpan]:
        """获取当前活跃的跨度"""
        thread_id = threading.get_ident()
        with self._lock:
            span_id = self.active_spans.get(thread_id)
            if span_id:
                return self.spans.get(span_id)
        return None
    
    def add_log(self, message: str, level: str = "info", **fields):
        """向当前跨度添加日志"""
        span = self.get_current_span()
        if span:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'message': message,
                **fields
            }
            span.logs.append(log_entry)
    
    def set_tag(self, key: str, value: Any):
        """设置当前跨度的标签"""
        span = self.get_current_span()
        if span:
            span.tags[key] = value
    
    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """获取完整的追踪链路"""
        with self._lock:
            return [span for span in self.spans.values() if span.trace_id == trace_id]
    
    def search_spans(self, operation_name: str = None, tags: Dict[str, Any] = None,
                    start_time: datetime = None, end_time: datetime = None) -> List[TraceSpan]:
        """搜索跨度"""
        with self._lock:
            results = list(self.spans.values())
        
        if operation_name:
            results = [span for span in results if operation_name in span.operation_name]
        
        if tags:
            results = [
                span for span in results
                if all(span.tags.get(k) == v for k, v in tags.items())
            ]
        
        if start_time:
            results = [span for span in results if span.start_time >= start_time]
        
        if end_time:
            results = [span for span in results if span.start_time <= end_time]
        
        return sorted(results, key=lambda x: x.start_time, reverse=True)

class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metric_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.anomalies: List[Dict] = []
        self._lock = threading.RLock()
    
    def add_metric_point(self, metric_name: str, value: float, timestamp: datetime = None):
        """添加指标数据点"""
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            self.metric_windows[metric_name].append((timestamp, value))
            
            # 检测异常
            self._detect_anomaly(metric_name, value, timestamp)
    
    def set_threshold(self, metric_name: str, min_value: float = None, max_value: float = None,
                    std_multiplier: float = None):
        """设置异常检测阈值"""
        self.thresholds[metric_name] = {
            'min_value': min_value,
            'max_value': max_value,
            'std_multiplier': std_multiplier
        }
    
    def _detect_anomaly(self, metric_name: str, value: float, timestamp: datetime):
        """检测异常"""
        threshold_config = self.thresholds.get(metric_name, {})
        
        # 基于固定阈值的检测
        min_value = threshold_config.get('min_value')
        max_value = threshold_config.get('max_value')
        
        if min_value is not None and value < min_value:
            self._record_anomaly(metric_name, value, timestamp, f"值低于最小阈值 {min_value}")
            return
        
        if max_value is not None and value > max_value:
            self._record_anomaly(metric_name, value, timestamp, f"值超过最大阈值 {max_value}")
            return
        
        # 基于统计的异常检测
        std_multiplier = threshold_config.get('std_multiplier')
        if std_multiplier and len(self.metric_windows[metric_name]) >= 10:
            values = [v for _, v in self.metric_windows[metric_name]]
            mean = np.mean(values)
            std = np.std(values)
            
            if abs(value - mean) > std_multiplier * std:
                self._record_anomaly(
                    metric_name, value, timestamp,
                    f"值偏离均值超过 {std_multiplier} 个标准差"
                )
    
    def _record_anomaly(self, metric_name: str, value: float, timestamp: datetime, reason: str):
        """记录异常"""
        anomaly = {
            'id': str(uuid.uuid4()),
            'metric_name': metric_name,
            'value': value,
            'timestamp': timestamp,
            'reason': reason,
            'severity': self._calculate_severity(metric_name, value)
        }
        
        self.anomalies.append(anomaly)
        logger.warning("检测到异常", **anomaly)
    
    def _calculate_severity(self, metric_name: str, value: float) -> str:
        """计算异常严重程度"""
        # 简化的严重程度计算
        if 'cpu' in metric_name.lower() or 'memory' in metric_name.lower():
            if value > 90:
                return "critical"
            elif value > 80:
                return "high"
            elif value > 70:
                return "medium"
            else:
                return "low"
        
        return "medium"
    
    def get_recent_anomalies(self, duration: timedelta = timedelta(hours=1)) -> List[Dict]:
        """获取最近的异常"""
        cutoff_time = datetime.now() - duration
        return [
            anomaly for anomaly in self.anomalies
            if anomaly['timestamp'] >= cutoff_time
        ]

class ProductionMonitor:
    """生产环境监控主类"""
    
    def __init__(self, service_name: str, config: Dict[str, Any] = None):
        self.service_name = service_name
        self.config = config or {}
        
        # 初始化组件
        self.logger = StructuredLogger(service_name)
        self.metrics = MetricsCollector(service_name)
        self.tracer = DistributedTracer(service_name)
        self.anomaly_detector = AnomalyDetector()
        
        # 告警管理
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable] = []
        
        # 健康检查
        self.health_status = HealthStatus.UNKNOWN
        self.health_checks: List[Callable] = []
        
        # 启动监控任务
        self._start_monitoring_tasks()
    
    def _start_monitoring_tasks(self):
        """启动监控任务"""
        # 启动异常检测任务
        def anomaly_detection_loop():
            while True:
                try:
                    # 获取最新的性能指标
                    perf_metrics = self.metrics.get_performance_summary()
                    
                    # 添加到异常检测器
                    self.anomaly_detector.add_metric_point('cpu_usage', perf_metrics.cpu_usage)
                    self.anomaly_detector.add_metric_point('memory_usage', perf_metrics.memory_usage)
                    self.anomaly_detector.add_metric_point('response_time', perf_metrics.response_time)
                    self.anomaly_detector.add_metric_point('error_rate', perf_metrics.error_rate)
                    
                    # 检查异常并生成告警
                    recent_anomalies = self.anomaly_detector.get_recent_anomalies(timedelta(minutes=1))
                    for anomaly in recent_anomalies:
                        self._create_alert_from_anomaly(anomaly)
                    
                    time.sleep(60)  # 每分钟检查一次
                    
                except Exception as e:
                    self.logger.error("异常检测循环失败", error=str(e))
                    time.sleep(30)
        
        # 启动健康检查任务
        def health_check_loop():
            while True:
                try:
                    self._perform_health_checks()
                    time.sleep(30)  # 每30秒检查一次
                except Exception as e:
                    self.logger.error("健康检查循环失败", error=str(e))
                    time.sleep(10)
        
        # 启动后台线程
        threading.Thread(target=anomaly_detection_loop, daemon=True).start()
        threading.Thread(target=health_check_loop, daemon=True).start()
    
    def _create_alert_from_anomaly(self, anomaly: Dict):
        """从异常创建告警"""
        alert = Alert(
            id=str(uuid.uuid4()),
            name=f"异常检测: {anomaly['metric_name']}",
            description=anomaly['reason'],
            severity=AlertSeverity(anomaly['severity']),
            timestamp=anomaly['timestamp'],
            source=self.service_name,
            metric_name=anomaly['metric_name'],
            current_value=anomaly['value'],
            threshold=0  # 异常检测没有固定阈值
        )
        
        self.alerts.append(alert)
        self.logger.warning("生成告警", alert_id=alert.id, metric=alert.metric_name)
        
        # 调用告警回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error("告警回调失败", error=str(e))
    
    def add_health_check(self, check_func: Callable[[], bool], name: str = None):
        """添加健康检查函数"""
        self.health_checks.append((check_func, name or check_func.__name__))
    
    def _perform_health_checks(self):
        """执行健康检查"""
        if not self.health_checks:
            self.health_status = HealthStatus.UNKNOWN
            return
        
        failed_checks = []
        
        for check_func, check_name in self.health_checks:
            try:
                if not check_func():
                    failed_checks.append(check_name)
            except Exception as e:
                failed_checks.append(f"{check_name} (异常: {e})")
        
        if not failed_checks:
            self.health_status = HealthStatus.HEALTHY
        elif len(failed_checks) < len(self.health_checks) / 2:
            self.health_status = HealthStatus.DEGRADED
        else:
            self.health_status = HealthStatus.UNHEALTHY
        
        if failed_checks:
            self.logger.warning("健康检查失败", failed_checks=failed_checks)
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    @contextmanager
    def trace_operation(self, operation_name: str, **tags):
        """追踪操作"""
        with self.tracer.start_span(operation_name, **tags) as span:
            start_time = time.time()
            try:
                yield span
            except Exception as e:
                self.logger.error(f"{operation_name} 失败", error=str(e), trace_id=span.trace_id)
                raise
            finally:
                duration = time.time() - start_time
                self.logger.info(f"{operation_name} 完成", duration=duration, trace_id=span.trace_id)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取监控面板数据"""
        perf_metrics = self.metrics.get_performance_summary()
        recent_alerts = [alert for alert in self.alerts if alert.status == "active"]
        recent_anomalies = self.anomaly_detector.get_recent_anomalies()
        
        return {
            'service_name': self.service_name,
            'health_status': self.health_status.value,
            'performance_metrics': asdict(perf_metrics),
            'active_alerts_count': len(recent_alerts),
            'recent_anomalies_count': len(recent_anomalies),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'disk_total': psutil.disk_usage('/').total
            },
            'uptime': time.time() - self._start_time if hasattr(self, '_start_time') else 0
        }
    
    def generate_report(self, duration: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
        """生成监控报告"""
        end_time = datetime.now()
        start_time = end_time - duration
        
        # 获取历史数据
        cpu_history = self.metrics.get_metric_history('cpu_usage', duration)
        memory_history = self.metrics.get_metric_history('memory_usage', duration)
        
        # 统计告警
        period_alerts = [
            alert for alert in self.alerts
            if start_time <= alert.timestamp <= end_time
        ]
        
        # 统计异常
        period_anomalies = [
            anomaly for anomaly in self.anomaly_detector.anomalies
            if start_time <= anomaly['timestamp'] <= end_time
        ]
        
        return {
            'report_period': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_hours': duration.total_seconds() / 3600
            },
            'performance_summary': {
                'avg_cpu_usage': np.mean([v for _, v in cpu_history]) if cpu_history else 0,
                'max_cpu_usage': max([v for _, v in cpu_history]) if cpu_history else 0,
                'avg_memory_usage': np.mean([v for _, v in memory_history]) if memory_history else 0,
                'max_memory_usage': max([v for _, v in memory_history]) if memory_history else 0
            },
            'alerts_summary': {
                'total_alerts': len(period_alerts),
                'critical_alerts': len([a for a in period_alerts if a.severity == AlertSeverity.CRITICAL]),
                'high_alerts': len([a for a in period_alerts if a.severity == AlertSeverity.HIGH]),
                'medium_alerts': len([a for a in period_alerts if a.severity == AlertSeverity.MEDIUM]),
                'low_alerts': len([a for a in period_alerts if a.severity == AlertSeverity.LOW])
            },
            'anomalies_summary': {
                'total_anomalies': len(period_anomalies),
                'by_metric': self._group_anomalies_by_metric(period_anomalies)
            },
            'health_status': self.health_status.value
        }
    
    def _group_anomalies_by_metric(self, anomalies: List[Dict]) -> Dict[str, int]:
        """按指标分组异常"""
        groups = defaultdict(int)
        for anomaly in anomalies:
            groups[anomaly['metric_name']] += 1
        return dict(groups)
    
    def start(self):
        """启动监控"""
        self._start_time = time.time()
        self.logger.info("生产环境监控已启动", service=self.service_name)
        
        # 设置异常检测阈值
        self.anomaly_detector.set_threshold('cpu_usage', max_value=90, std_multiplier=2)
        self.anomaly_detector.set_threshold('memory_usage', max_value=85, std_multiplier=2)
        self.anomaly_detector.set_threshold('response_time', max_value=5.0, std_multiplier=3)
        self.anomaly_detector.set_threshold('error_rate', max_value=0.05, std_multiplier=2)
    
    def stop(self):
        """停止监控"""
        self.logger.info("生产环境监控已停止", service=self.service_name)

def create_flask_app_with_monitoring(monitor: ProductionMonitor) -> Flask:
    """创建带监控的Flask应用"""
    app = Flask(__name__)
    
    @app.before_request
    def before_request():
        g.start_time = time.time()
        g.request_id = str(uuid.uuid4())
    
    @app.after_request
    def after_request(response):
        duration = time.time() - g.start_time
        
        # 记录请求指标
        monitor.metrics.record_request(
            method=request.method,
            endpoint=request.endpoint or 'unknown',
            status=response.status_code,
            duration=duration
        )
        
        # 记录请求日志
        monitor.logger.info(
            "HTTP请求",
            method=request.method,
            path=request.path,
            status=response.status_code,
            duration=duration,
            request_id=g.request_id
        )
        
        return response
    
    @app.route('/health')
    def health_check():
        """健康检查端点"""
        return jsonify({
            'status': monitor.health_status.value,
            'timestamp': datetime.now().isoformat(),
            'service': monitor.service_name
        })
    
    @app.route('/metrics')
    def metrics():
        """Prometheus指标端点"""
        return monitor.metrics.get_metrics_data(), 200, {
            'Content-Type': 'text/plain; charset=utf-8'
        }
    
    @app.route('/dashboard')
    def dashboard():
        """监控面板数据"""
        return jsonify(monitor.get_dashboard_data())
    
    @app.route('/report')
    def report():
        """监控报告"""
        hours = request.args.get('hours', 24, type=int)
        duration = timedelta(hours=hours)
        return jsonify(monitor.generate_report(duration))
    
    return app

def main():
    """示例用法"""
    print("生产环境监控系统测试\n" + "="*50)
    
    # 创建监控器
    monitor = ProductionMonitor("rag-system")
    
    # 添加健康检查
    def check_database():
        # 模拟数据库检查
        return True
    
    def check_vector_db():
        # 模拟向量数据库检查
        return True
    
    monitor.add_health_check(check_database, "database")
    monitor.add_health_check(check_vector_db, "vector_db")
    
    # 添加告警回调
    def alert_callback(alert: Alert):
        print(f"🚨 告警: {alert.name} - {alert.description}")
    
    monitor.add_alert_callback(alert_callback)
    
    # 启动监控
    monitor.start()
    
    # 模拟一些操作
    print("\n模拟监控操作...")
    
    # 模拟搜索操作
    with monitor.trace_operation("search_documents", query_type="semantic") as span:
        time.sleep(0.1)  # 模拟处理时间
        monitor.metrics.record_search("semantic", "success", 0.1)
        monitor.tracer.add_log("搜索完成", results_count=5)
    
    # 模拟向量数据库操作
    monitor.metrics.record_vector_db_operation("insert", "success")
    monitor.metrics.record_vector_db_operation("search", "success")
    
    # 更新缓存命中率
    monitor.metrics.update_cache_hit_rate(0.85)
    
    # 生成一些异常数据来测试异常检测
    monitor.anomaly_detector.add_metric_point("cpu_usage", 95.0)  # 高CPU使用率
    monitor.anomaly_detector.add_metric_point("response_time", 10.0)  # 高响应时间
    
    # 等待一段时间让监控系统处理
    time.sleep(2)
    
    # 获取监控面板数据
    dashboard_data = monitor.get_dashboard_data()
    print(f"\n监控面板数据:")
    print(f"- 服务状态: {dashboard_data['health_status']}")
    print(f"- CPU使用率: {dashboard_data['performance_metrics']['cpu_usage']:.1f}%")
    print(f"- 内存使用率: {dashboard_data['performance_metrics']['memory_usage']:.1f}%")
    print(f"- 活跃告警数: {dashboard_data['active_alerts_count']}")
    print(f"- 最近异常数: {dashboard_data['recent_anomalies_count']}")
    
    # 生成监控报告
    report = monitor.generate_report(timedelta(hours=1))
    print(f"\n监控报告 (最近1小时):")
    print(f"- 总告警数: {report['alerts_summary']['total_alerts']}")
    print(f"- 总异常数: {report['anomalies_summary']['total_anomalies']}")
    print(f"- 平均CPU使用率: {report['performance_summary']['avg_cpu_usage']:.1f}%")
    
    # 创建Flask应用进行演示
    app = create_flask_app_with_monitoring(monitor)
    
    print("\n监控API端点:")
    print("- 健康检查: GET /health")
    print("- 指标数据: GET /metrics")
    print("- 监控面板: GET /dashboard")
    print("- 监控报告: GET /report?hours=24")
    
    print("\n要启动Web服务器，请运行:")
    print("python lesson18_production_monitoring.py")
    
    # 停止监控
    monitor.stop()

if __name__ == "__main__":
    # 如果直接运行，启动演示
    if len(os.sys.argv) > 1 and os.sys.argv[1] == 'demo':
        main()
    else:
        # 启动Web服务器
        monitor = ProductionMonitor("rag-system")
        monitor.start()
        
        app = create_flask_app_with_monitoring(monitor)
        print("启动生产环境监控服务...")
        print("访问 http://localhost:5000/dashboard 查看监控面板")
        
        try:
            app.run(host='0.0.0.0', port=5000, debug=False)
        except KeyboardInterrupt:
            print("\n正在停止监控服务...")
            monitor.stop()