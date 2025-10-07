#!/usr/bin/env python3
"""
实验三：监控演示（简化版）

本实验演示系统监控、指标收集和告警功能。
"""

import time
import random
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading

# 指标类型
class MetricType(Enum):
    COUNTER = "counter"        # 计数器
    GAUGE = "gauge"           # 仪表盘
    HISTOGRAM = "histogram"   # 直方图
    TIMER = "timer"           # 计时器

# 告警级别
class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Metric:
    """指标数据"""
    name: str
    type: MetricType
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class Alert:
    """告警信息"""
    name: str
    level: AlertLevel
    message: str
    timestamp: float
    metric_name: str
    threshold: float
    current_value: float

class SimpleMetricsCollector:
    """简化的指标收集器"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """增加计数器"""
        with self._lock:
            self.counters[name] += value
            metric = Metric(name, MetricType.COUNTER, self.counters[name], time.time(), labels or {})
            self.metrics[name].append(metric)
            print(f"📊 计数器 {name}: {self.counters[name]}")
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """设置仪表盘值"""
        with self._lock:
            self.gauges[name] = value
            metric = Metric(name, MetricType.GAUGE, value, time.time(), labels or {})
            self.metrics[name].append(metric)
            print(f"📈 仪表盘 {name}: {value}")
    
    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """记录计时器"""
        with self._lock:
            self.timers[name].append(duration)
            # 保持最近100个记录
            if len(self.timers[name]) > 100:
                self.timers[name] = self.timers[name][-100:]
            
            metric = Metric(name, MetricType.TIMER, duration, time.time(), labels or {})
            self.metrics[name].append(metric)
            print(f"⏱️ 计时器 {name}: {duration:.3f}s")
    
    def get_counter(self, name: str) -> float:
        """获取计数器值"""
        return self.counters.get(name, 0.0)
    
    def get_gauge(self, name: str) -> float:
        """获取仪表盘值"""
        return self.gauges.get(name, 0.0)
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """获取计时器统计"""
        timers = self.timers.get(name, [])
        if not timers:
            return {}
        
        return {
            'count': len(timers),
            'min': min(timers),
            'max': max(timers),
            'avg': sum(timers) / len(timers),
            'p95': sorted(timers)[int(len(timers) * 0.95)] if len(timers) > 1 else timers[0]
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        with self._lock:
            return {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'timers': {name: self.get_timer_stats(name) for name in self.timers}
            }

class SimpleAlertManager:
    """简化的告警管理器"""
    
    def __init__(self):
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.alerts: List[Alert] = []
        self.alert_history: deque = deque(maxlen=100)
        self._lock = threading.RLock()
    
    def add_rule(self, metric_name: str, threshold: float, level: AlertLevel, 
                 condition: str = "greater", message: str = None):
        """添加告警规则"""
        with self._lock:
            self.rules[metric_name] = {
                'threshold': threshold,
                'level': level,
                'condition': condition,
                'message': message or f"{metric_name} {condition} than {threshold}"
            }
            print(f"📋 添加告警规则: {metric_name} {condition} {threshold} -> {level.value}")
    
    def check_alerts(self, metrics_collector: SimpleMetricsCollector):
        """检查告警"""
        with self._lock:
            current_time = time.time()
            
            for metric_name, rule in self.rules.items():
                current_value = None
                
                # 获取当前值
                if metric_name in metrics_collector.counters:
                    current_value = metrics_collector.counters[metric_name]
                elif metric_name in metrics_collector.gauges:
                    current_value = metrics_collector.gauges[metric_name]
                elif metric_name in metrics_collector.timers:
                    stats = metrics_collector.get_timer_stats(metric_name)
                    current_value = stats.get('avg', 0)
                
                if current_value is None:
                    continue
                
                # 检查条件
                should_alert = False
                if rule['condition'] == 'greater' and current_value > rule['threshold']:
                    should_alert = True
                elif rule['condition'] == 'less' and current_value < rule['threshold']:
                    should_alert = True
                elif rule['condition'] == 'equal' and abs(current_value - rule['threshold']) < 0.001:
                    should_alert = True
                
                if should_alert:
                    alert = Alert(
                        name=f"{metric_name}_alert",
                        level=rule['level'],
                        message=rule['message'],
                        timestamp=current_time,
                        metric_name=metric_name,
                        threshold=rule['threshold'],
                        current_value=current_value
                    )
                    
                    self.alerts.append(alert)
                    self.alert_history.append(alert)
                    
                    level_emoji = {
                        AlertLevel.INFO: "ℹ️",
                        AlertLevel.WARNING: "⚠️",
                        AlertLevel.ERROR: "❌",
                        AlertLevel.CRITICAL: "🚨"
                    }
                    
                    print(f"{level_emoji[rule['level']]} 告警: {alert.message} (当前值: {current_value:.3f})")
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return self.alerts.copy()
    
    def clear_alerts(self):
        """清除告警"""
        with self._lock:
            self.alerts.clear()

class SimpleHealthChecker:
    """简化的健康检查器"""
    
    def __init__(self):
        self.services: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def register_service(self, name: str, check_func, interval: float = 30.0):
        """注册服务健康检查"""
        with self._lock:
            self.services[name] = {
                'check_func': check_func,
                'interval': interval,
                'last_check': 0,
                'status': 'unknown',
                'message': ''
            }
            print(f"🏥 注册健康检查: {name}")
    
    def check_all_services(self) -> Dict[str, Dict[str, Any]]:
        """检查所有服务健康状态"""
        current_time = time.time()
        results = {}
        
        with self._lock:
            for name, service in self.services.items():
                if current_time - service['last_check'] >= service['interval']:
                    try:
                        is_healthy, message = service['check_func']()
                        service['status'] = 'healthy' if is_healthy else 'unhealthy'
                        service['message'] = message
                        service['last_check'] = current_time
                        
                        status_emoji = "✅" if is_healthy else "❌"
                        print(f"{status_emoji} 健康检查 {name}: {service['status']} - {message}")
                        
                    except Exception as e:
                        service['status'] = 'error'
                        service['message'] = str(e)
                        service['last_check'] = current_time
                        print(f"💥 健康检查 {name} 异常: {str(e)}")
                
                results[name] = {
                    'status': service['status'],
                    'message': service['message'],
                    'last_check': service['last_check']
                }
        
        return results

# 模拟服务
class MockRAGService:
    """模拟RAG服务"""
    
    def __init__(self, metrics_collector: SimpleMetricsCollector):
        self.metrics = metrics_collector
        self.request_count = 0
        self.error_count = 0
    
    def process_query(self, query: str):
        """处理查询"""
        start_time = time.time()
        self.request_count += 1
        
        try:
            # 模拟处理时间
            processing_time = random.uniform(0.1, 2.0)
            time.sleep(processing_time)
            
            # 模拟偶尔的错误
            if random.random() < 0.2:
                self.error_count += 1
                self.metrics.increment_counter('rag_errors')
                raise Exception("Processing error")
            
            # 记录指标
            duration = time.time() - start_time
            self.metrics.increment_counter('rag_requests')
            self.metrics.record_timer('rag_response_time', duration)
            self.metrics.set_gauge('rag_active_connections', random.randint(1, 10))
            
            return f"Response for: {query}"
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_timer('rag_response_time', duration)
            raise e
    
    def health_check(self):
        """健康检查"""
        if self.error_count > 5:
            return False, f"Too many errors: {self.error_count}"
        return True, "Service is healthy"

def run_monitoring_demo():
    """运行监控演示"""
    print("=" * 60)
    print("📊 监控演示实验")
    print("=" * 60)
    
    # 创建组件
    metrics_collector = SimpleMetricsCollector()
    alert_manager = SimpleAlertManager()
    health_checker = SimpleHealthChecker()
    rag_service = MockRAGService(metrics_collector)
    
    # 设置告警规则
    alert_manager.add_rule('rag_errors', 3, AlertLevel.WARNING, 'greater', '错误次数过多')
    alert_manager.add_rule('rag_response_time', 1.5, AlertLevel.ERROR, 'greater', '响应时间过长')
    alert_manager.add_rule('rag_active_connections', 8, AlertLevel.INFO, 'greater', '连接数较高')
    
    # 注册健康检查
    health_checker.register_service('rag_service', rag_service.health_check, 5.0)
    
    print("\n🚀 开始模拟请求...\n")
    
    # 模拟请求
    queries = [
        "什么是机器学习？",
        "深度学习原理",
        "自然语言处理",
        "计算机视觉",
        "推荐系统算法",
        "强化学习应用",
        "神经网络结构",
        "数据挖掘技术"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- 请求 {i}: {query} ---")
        
        try:
            response = rag_service.process_query(query)
            print(f"✅ 处理成功: {response[:30]}...")
        except Exception as e:
            print(f"❌ 处理失败: {str(e)}")
        
        # 检查告警
        alert_manager.check_alerts(metrics_collector)
        
        # 定期健康检查
        if i % 3 == 0:
            print("\n🏥 执行健康检查...")
            health_status = health_checker.check_all_services()
        
        time.sleep(0.5)
    
    # 显示最终统计
    print("\n" + "=" * 60)
    print("📊 监控统计报告")
    print("=" * 60)
    
    # 指标统计
    all_metrics = metrics_collector.get_all_metrics()
    
    print("\n📈 指标统计:")
    print(f"  请求总数: {all_metrics['counters'].get('rag_requests', 0)}")
    print(f"  错误总数: {all_metrics['counters'].get('rag_errors', 0)}")
    print(f"  活跃连接: {all_metrics['gauges'].get('rag_active_connections', 0)}")
    
    response_time_stats = all_metrics['timers'].get('rag_response_time', {})
    if response_time_stats:
        print(f"\n⏱️ 响应时间统计:")
        print(f"  请求次数: {response_time_stats.get('count', 0)}")
        print(f"  平均时间: {response_time_stats.get('avg', 0):.3f}s")
        print(f"  最小时间: {response_time_stats.get('min', 0):.3f}s")
        print(f"  最大时间: {response_time_stats.get('max', 0):.3f}s")
        print(f"  95分位数: {response_time_stats.get('p95', 0):.3f}s")
    
    # 告警统计
    active_alerts = alert_manager.get_active_alerts()
    print(f"\n🚨 活跃告警数: {len(active_alerts)}")
    
    if active_alerts:
        print("\n告警详情:")
        for alert in active_alerts[-5:]:  # 显示最近5个告警
            print(f"  {alert.level.value}: {alert.message}")
    
    # 健康状态
    print("\n🏥 服务健康状态:")
    health_status = health_checker.check_all_services()
    for service_name, status in health_status.items():
        status_emoji = "✅" if status['status'] == 'healthy' else "❌"
        print(f"  {status_emoji} {service_name}: {status['status']} - {status['message']}")
    
    print("\n✅ 监控演示完成！")

if __name__ == "__main__":
    run_monitoring_demo()