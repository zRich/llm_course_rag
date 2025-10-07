#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lesson17 企业级RAG部署实现模板
解决容器化部署、负载均衡和监控告警缺失问题

功能特性：
1. 容器化部署和编排
2. 负载均衡和服务发现
3. 健康检查和故障恢复
4. 监控告警和日志管理
5. 安全认证和权限控制
"""

import logging
import time
import json
import os
import signal
import threading
import subprocess
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import yaml
import requests
from flask import Flask, request, jsonify, g
from werkzeug.middleware.proxy_fix import ProxyFix
import redis
import psutil
import docker
from kubernetes import client, config
import consul
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import jwt
from functools import wraps

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentMode(Enum):
    """部署模式枚举"""
    STANDALONE = "standalone"
    DOCKER_COMPOSE = "docker_compose"
    KUBERNETES = "kubernetes"
    DOCKER_SWARM = "docker_swarm"

class ServiceStatus(Enum):
    """服务状态枚举"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"

class AlertLevel(Enum):
    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ServiceConfig:
    """服务配置"""
    name: str
    image: str
    port: int
    replicas: int = 1
    cpu_limit: str = "1000m"
    memory_limit: str = "1Gi"
    env_vars: Dict[str, str] = field(default_factory=dict)
    health_check_path: str = "/health"
    dependencies: List[str] = field(default_factory=list)

@dataclass
class HealthCheckResult:
    """健康检查结果"""
    service_name: str
    status: ServiceStatus
    response_time: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlertRule:
    """告警规则"""
    name: str
    condition: str
    threshold: float
    level: AlertLevel
    description: str
    enabled: bool = True

class PrometheusMetrics:
    """Prometheus指标收集器"""
    
    def __init__(self):
        # 请求计数器
        self.request_count = Counter(
            'rag_requests_total',
            'Total number of RAG requests',
            ['method', 'endpoint', 'status']
        )
        
        # 请求延迟直方图
        self.request_duration = Histogram(
            'rag_request_duration_seconds',
            'RAG request duration in seconds',
            ['method', 'endpoint']
        )
        
        # 活跃连接数
        self.active_connections = Gauge(
            'rag_active_connections',
            'Number of active connections'
        )
        
        # 缓存命中率
        self.cache_hit_rate = Gauge(
            'rag_cache_hit_rate',
            'Cache hit rate'
        )
        
        # 系统资源使用率
        self.cpu_usage = Gauge(
            'rag_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.memory_usage = Gauge(
            'rag_memory_usage_percent',
            'Memory usage percentage'
        )
        
        # 向量数据库指标
        self.vector_db_operations = Counter(
            'rag_vector_db_operations_total',
            'Total vector database operations',
            ['operation', 'status']
        )
        
        self.vector_db_latency = Histogram(
            'rag_vector_db_latency_seconds',
            'Vector database operation latency',
            ['operation']
        )
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """记录请求指标"""
        self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def update_system_metrics(self):
        """更新系统指标"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            self.cpu_usage.set(cpu_percent)
            self.memory_usage.set(memory.percent)
        except Exception as e:
            logger.error(f"更新系统指标失败: {e}")

class HealthChecker:
    """健康检查器"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.services: Dict[str, ServiceConfig] = {}
        self.health_status: Dict[str, HealthCheckResult] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []
    
    def register_service(self, service: ServiceConfig):
        """注册服务"""
        self.services[service.name] = service
        logger.info(f"已注册服务: {service.name}")
    
    def add_status_callback(self, callback: Callable[[str, HealthCheckResult], None]):
        """添加状态变化回调"""
        self._callbacks.append(callback)
    
    def start(self):
        """启动健康检查"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._check_loop, daemon=True)
        self._thread.start()
        logger.info("健康检查器已启动")
    
    def stop(self):
        """停止健康检查"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("健康检查器已停止")
    
    def _check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                for service_name, service_config in self.services.items():
                    result = self._check_service_health(service_config)
                    
                    # 检查状态是否发生变化
                    old_result = self.health_status.get(service_name)
                    if old_result is None or old_result.status != result.status:
                        logger.info(f"服务 {service_name} 状态变化: {result.status.value}")
                        
                        # 调用回调函数
                        for callback in self._callbacks:
                            try:
                                callback(service_name, result)
                            except Exception as e:
                                logger.error(f"健康检查回调失败: {e}")
                    
                    self.health_status[service_name] = result
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"健康检查循环异常: {e}")
                time.sleep(5)
    
    def _check_service_health(self, service: ServiceConfig) -> HealthCheckResult:
        """检查单个服务健康状态"""
        start_time = time.time()
        
        try:
            # 构建健康检查URL
            url = f"http://localhost:{service.port}{service.health_check_path}"
            
            # 发送健康检查请求
            response = requests.get(url, timeout=5)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                status = ServiceStatus.HEALTHY
                details = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
            else:
                status = ServiceStatus.UNHEALTHY
                details = {'status_code': response.status_code, 'response': response.text[:200]}
            
        except requests.exceptions.Timeout:
            response_time = time.time() - start_time
            status = ServiceStatus.UNHEALTHY
            details = {'error': 'timeout'}
            
        except requests.exceptions.ConnectionError:
            response_time = time.time() - start_time
            status = ServiceStatus.UNHEALTHY
            details = {'error': 'connection_error'}
            
        except Exception as e:
            response_time = time.time() - start_time
            status = ServiceStatus.UNKNOWN
            details = {'error': str(e)}
        
        return HealthCheckResult(
            service_name=service.name,
            status=status,
            response_time=response_time,
            timestamp=datetime.now(),
            details=details
        )
    
    def get_service_status(self, service_name: str) -> Optional[HealthCheckResult]:
        """获取服务状态"""
        return self.health_status.get(service_name)
    
    def get_all_status(self) -> Dict[str, HealthCheckResult]:
        """获取所有服务状态"""
        return dict(self.health_status)

class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self, algorithm: str = "round_robin"):
        self.algorithm = algorithm
        self.services: Dict[str, List[str]] = {}  # service_name -> [endpoints]
        self.current_index: Dict[str, int] = {}  # For round-robin
        self.health_checker: Optional[HealthChecker] = None
        self._lock = threading.RLock()
    
    def set_health_checker(self, health_checker: HealthChecker):
        """设置健康检查器"""
        self.health_checker = health_checker
    
    def register_service_endpoints(self, service_name: str, endpoints: List[str]):
        """注册服务端点"""
        with self._lock:
            self.services[service_name] = endpoints
            self.current_index[service_name] = 0
            logger.info(f"已注册服务端点 {service_name}: {endpoints}")
    
    def get_endpoint(self, service_name: str) -> Optional[str]:
        """获取服务端点"""
        with self._lock:
            endpoints = self.services.get(service_name, [])
            if not endpoints:
                return None
            
            # 过滤健康的端点
            healthy_endpoints = self._get_healthy_endpoints(service_name, endpoints)
            if not healthy_endpoints:
                logger.warning(f"服务 {service_name} 没有健康的端点")
                return None
            
            # 根据算法选择端点
            if self.algorithm == "round_robin":
                return self._round_robin_select(service_name, healthy_endpoints)
            elif self.algorithm == "random":
                import random
                return random.choice(healthy_endpoints)
            else:
                return healthy_endpoints[0]
    
    def _get_healthy_endpoints(self, service_name: str, endpoints: List[str]) -> List[str]:
        """获取健康的端点"""
        if not self.health_checker:
            return endpoints
        
        healthy_endpoints = []
        for endpoint in endpoints:
            # 简化：假设端点格式为 host:port
            try:
                host, port = endpoint.split(':')
                # 这里应该检查具体端点的健康状态
                # 简化实现：检查服务整体状态
                status = self.health_checker.get_service_status(service_name)
                if status and status.status == ServiceStatus.HEALTHY:
                    healthy_endpoints.append(endpoint)
            except:
                continue
        
        return healthy_endpoints or endpoints  # 如果没有健康端点，返回所有端点
    
    def _round_robin_select(self, service_name: str, endpoints: List[str]) -> str:
        """轮询选择端点"""
        current_idx = self.current_index.get(service_name, 0)
        endpoint = endpoints[current_idx % len(endpoints)]
        self.current_index[service_name] = (current_idx + 1) % len(endpoints)
        return endpoint

class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.alert_history: List[Dict] = []
        self.notification_channels: List[Callable] = []
        self._metrics_cache: Dict[str, float] = {}
    
    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.rules[rule.name] = rule
        logger.info(f"已添加告警规则: {rule.name}")
    
    def add_notification_channel(self, channel: Callable[[Dict], None]):
        """添加通知渠道"""
        self.notification_channels.append(channel)
    
    def update_metric(self, metric_name: str, value: float):
        """更新指标值"""
        self._metrics_cache[metric_name] = value
        self._check_alerts(metric_name, value)
    
    def _check_alerts(self, metric_name: str, value: float):
        """检查告警条件"""
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            # 简化的条件检查
            if metric_name in rule.condition:
                should_alert = self._evaluate_condition(rule, metric_name, value)
                
                if should_alert:
                    self._trigger_alert(rule, metric_name, value)
    
    def _evaluate_condition(self, rule: AlertRule, metric_name: str, value: float) -> bool:
        """评估告警条件"""
        # 简化的条件评估
        if ">" in rule.condition:
            return value > rule.threshold
        elif "<" in rule.condition:
            return value < rule.threshold
        elif "==" in rule.condition:
            return abs(value - rule.threshold) < 0.001
        else:
            return False
    
    def _trigger_alert(self, rule: AlertRule, metric_name: str, value: float):
        """触发告警"""
        alert = {
            'rule_name': rule.name,
            'level': rule.level.value,
            'description': rule.description,
            'metric_name': metric_name,
            'current_value': value,
            'threshold': rule.threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        self.alert_history.append(alert)
        logger.warning(f"告警触发: {rule.name} - {rule.description}")
        
        # 发送通知
        for channel in self.notification_channels:
            try:
                channel(alert)
            except Exception as e:
                logger.error(f"告警通知发送失败: {e}")
    
    def get_active_alerts(self, time_window: timedelta = timedelta(hours=1)) -> List[Dict]:
        """获取活跃告警"""
        cutoff_time = datetime.now() - time_window
        return [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) >= cutoff_time
        ]

class SecurityManager:
    """安全管理器"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.token_expiry = timedelta(hours=24)
        self.rate_limits: Dict[str, Dict] = {}  # client_id -> {count, reset_time}
        self.rate_limit_window = timedelta(minutes=1)
        self.rate_limit_max_requests = 100
    
    def generate_token(self, user_id: str, permissions: List[str] = None) -> str:
        """生成JWT令牌"""
        payload = {
            'user_id': user_id,
            'permissions': permissions or [],
            'exp': datetime.utcnow() + self.token_expiry,
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """验证JWT令牌"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("令牌已过期")
            return None
        except jwt.InvalidTokenError:
            logger.warning("无效令牌")
            return None
    
    def check_rate_limit(self, client_id: str) -> bool:
        """检查速率限制"""
        now = datetime.now()
        
        if client_id not in self.rate_limits:
            self.rate_limits[client_id] = {
                'count': 1,
                'reset_time': now + self.rate_limit_window
            }
            return True
        
        limit_info = self.rate_limits[client_id]
        
        # 检查是否需要重置计数器
        if now >= limit_info['reset_time']:
            limit_info['count'] = 1
            limit_info['reset_time'] = now + self.rate_limit_window
            return True
        
        # 检查是否超过限制
        if limit_info['count'] >= self.rate_limit_max_requests:
            return False
        
        limit_info['count'] += 1
        return True
    
    def require_auth(self, required_permissions: List[str] = None):
        """认证装饰器"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # 获取令牌
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    return jsonify({'error': 'Missing or invalid authorization header'}), 401
                
                token = auth_header[7:]  # Remove 'Bearer ' prefix
                
                # 验证令牌
                payload = self.verify_token(token)
                if not payload:
                    return jsonify({'error': 'Invalid or expired token'}), 401
                
                # 检查权限
                if required_permissions:
                    user_permissions = payload.get('permissions', [])
                    if not all(perm in user_permissions for perm in required_permissions):
                        return jsonify({'error': 'Insufficient permissions'}), 403
                
                # 检查速率限制
                client_id = payload['user_id']
                if not self.check_rate_limit(client_id):
                    return jsonify({'error': 'Rate limit exceeded'}), 429
                
                # 将用户信息添加到请求上下文
                g.current_user = payload
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator

class EnterpriseRAGDeployment:
    """企业级RAG部署管理器"""
    
    def __init__(self, config_path: str = "deployment_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # 初始化组件
        self.metrics = PrometheusMetrics()
        self.health_checker = HealthChecker()
        self.load_balancer = LoadBalancer()
        self.alert_manager = AlertManager()
        self.security_manager = SecurityManager(self.config.get('secret_key', 'default-secret'))
        
        # Flask应用
        self.app = Flask(__name__)
        self.app.wsgi_app = ProxyFix(self.app.wsgi_app)
        
        # 设置组件关联
        self.load_balancer.set_health_checker(self.health_checker)
        self.health_checker.add_status_callback(self._on_service_status_change)
        
        # 注册路由
        self._register_routes()
        
        # 配置告警规则
        self._setup_alert_rules()
        
        # 启动后台任务
        self._start_background_tasks()
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {self.config_path} 不存在，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'deployment_mode': 'standalone',
            'services': {
                'rag_api': {
                    'port': 8000,
                    'replicas': 2,
                    'health_check_path': '/health'
                },
                'vector_db': {
                    'port': 6333,
                    'replicas': 1,
                    'health_check_path': '/health'
                }
            },
            'monitoring': {
                'metrics_port': 9090,
                'alert_webhook': None
            },
            'security': {
                'enable_auth': True,
                'rate_limit': 100
            }
        }
    
    def _register_routes(self):
        """注册API路由"""
        
        @self.app.route('/health')
        def health_check():
            """健康检查端点"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            })
        
        @self.app.route('/metrics')
        def metrics():
            """Prometheus指标端点"""
            # 更新系统指标
            self.metrics.update_system_metrics()
            
            # 返回Prometheus格式的指标
            return generate_latest(), 200, {'Content-Type': 'text/plain; charset=utf-8'}
        
        @self.app.route('/api/v1/search', methods=['POST'])
        @self.security_manager.require_auth(['search'])
        def search():
            """搜索API"""
            start_time = time.time()
            
            try:
                data = request.get_json()
                query = data.get('query', '')
                
                if not query:
                    return jsonify({'error': 'Query is required'}), 400
                
                # 模拟搜索处理
                result = self._process_search(query)
                
                # 记录指标
                duration = time.time() - start_time
                self.metrics.record_request('POST', '/api/v1/search', 200, duration)
                
                return jsonify(result)
                
            except Exception as e:
                duration = time.time() - start_time
                self.metrics.record_request('POST', '/api/v1/search', 500, duration)
                logger.error(f"搜索处理失败: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/v1/status')
        @self.security_manager.require_auth(['admin'])
        def get_status():
            """获取系统状态"""
            return jsonify({
                'services': {
                    name: {
                        'status': result.status.value,
                        'response_time': result.response_time,
                        'last_check': result.timestamp.isoformat()
                    }
                    for name, result in self.health_checker.get_all_status().items()
                },
                'alerts': self.alert_manager.get_active_alerts(),
                'system_metrics': {
                    'cpu_usage': self.metrics.cpu_usage._value._value,
                    'memory_usage': self.metrics.memory_usage._value._value
                }
            })
        
        @self.app.route('/api/v1/auth/token', methods=['POST'])
        def get_token():
            """获取访问令牌"""
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')
            
            # 简化的认证逻辑
            if self._authenticate_user(username, password):
                permissions = self._get_user_permissions(username)
                token = self.security_manager.generate_token(username, permissions)
                return jsonify({'token': token})
            else:
                return jsonify({'error': 'Invalid credentials'}), 401
    
    def _process_search(self, query: str) -> Dict:
        """处理搜索请求"""
        # 模拟搜索逻辑
        time.sleep(0.1)  # 模拟处理时间
        
        return {
            'query': query,
            'results': [
                {
                    'id': '1',
                    'title': f'搜索结果 for {query}',
                    'content': f'这是关于 {query} 的相关内容...',
                    'score': 0.95
                }
            ],
            'total': 1,
            'processing_time': 0.1
        }
    
    def _authenticate_user(self, username: str, password: str) -> bool:
        """用户认证"""
        # 简化的认证逻辑
        valid_users = {
            'admin': 'admin123',
            'user': 'user123'
        }
        return valid_users.get(username) == password
    
    def _get_user_permissions(self, username: str) -> List[str]:
        """获取用户权限"""
        permissions_map = {
            'admin': ['search', 'admin', 'write'],
            'user': ['search']
        }
        return permissions_map.get(username, [])
    
    def _setup_alert_rules(self):
        """设置告警规则"""
        # CPU使用率告警
        self.alert_manager.add_rule(AlertRule(
            name="high_cpu_usage",
            condition="cpu_usage > threshold",
            threshold=80.0,
            level=AlertLevel.WARNING,
            description="CPU使用率过高"
        ))
        
        # 内存使用率告警
        self.alert_manager.add_rule(AlertRule(
            name="high_memory_usage",
            condition="memory_usage > threshold",
            threshold=85.0,
            level=AlertLevel.WARNING,
            description="内存使用率过高"
        ))
        
        # 响应时间告警
        self.alert_manager.add_rule(AlertRule(
            name="high_response_time",
            condition="response_time > threshold",
            threshold=5.0,
            level=AlertLevel.ERROR,
            description="响应时间过长"
        ))
        
        # 添加通知渠道
        self.alert_manager.add_notification_channel(self._send_alert_notification)
    
    def _send_alert_notification(self, alert: Dict):
        """发送告警通知"""
        webhook_url = self.config.get('monitoring', {}).get('alert_webhook')
        
        if webhook_url:
            try:
                requests.post(webhook_url, json=alert, timeout=5)
                logger.info(f"告警通知已发送: {alert['rule_name']}")
            except Exception as e:
                logger.error(f"发送告警通知失败: {e}")
        else:
            # 记录到日志
            logger.warning(f"告警: {alert['description']} - 当前值: {alert['current_value']}")
    
    def _on_service_status_change(self, service_name: str, result: HealthCheckResult):
        """服务状态变化回调"""
        if result.status == ServiceStatus.UNHEALTHY:
            self.alert_manager.update_metric(f"{service_name}_health", 0)
        else:
            self.alert_manager.update_metric(f"{service_name}_health", 1)
    
    def _start_background_tasks(self):
        """启动后台任务"""
        # 启动健康检查
        self.health_checker.start()
        
        # 启动指标收集任务
        def metrics_collection_loop():
            while True:
                try:
                    # 更新系统指标
                    self.metrics.update_system_metrics()
                    
                    # 更新告警管理器的指标
                    self.alert_manager.update_metric('cpu_usage', self.metrics.cpu_usage._value._value)
                    self.alert_manager.update_metric('memory_usage', self.metrics.memory_usage._value._value)
                    
                    time.sleep(30)  # 每30秒收集一次
                except Exception as e:
                    logger.error(f"指标收集失败: {e}")
                    time.sleep(5)
        
        metrics_thread = threading.Thread(target=metrics_collection_loop, daemon=True)
        metrics_thread.start()
    
    def register_service(self, service_config: ServiceConfig):
        """注册服务"""
        self.health_checker.register_service(service_config)
        
        # 注册负载均衡端点
        endpoints = [f"localhost:{service_config.port}"]
        if service_config.replicas > 1:
            # 为多副本生成端点
            endpoints = [f"localhost:{service_config.port + i}" for i in range(service_config.replicas)]
        
        self.load_balancer.register_service_endpoints(service_config.name, endpoints)
    
    def generate_docker_compose(self) -> str:
        """生成Docker Compose配置"""
        services = {}
        
        for service_name, service_info in self.config.get('services', {}).items():
            services[service_name] = {
                'image': f'{service_name}:latest',
                'ports': [f"{service_info['port']}:{service_info['port']}"],
                'environment': {
                    'SERVICE_NAME': service_name,
                    'SERVICE_PORT': service_info['port']
                },
                'healthcheck': {
                    'test': f"curl -f http://localhost:{service_info['port']}{service_info.get('health_check_path', '/health')} || exit 1",
                    'interval': '30s',
                    'timeout': '10s',
                    'retries': 3
                },
                'restart': 'unless-stopped'
            }
            
            # 添加副本配置
            if service_info.get('replicas', 1) > 1:
                services[service_name]['deploy'] = {
                    'replicas': service_info['replicas']
                }
        
        # 添加监控服务
        services['prometheus'] = {
            'image': 'prom/prometheus:latest',
            'ports': ['9090:9090'],
            'volumes': ['./prometheus.yml:/etc/prometheus/prometheus.yml'],
            'restart': 'unless-stopped'
        }
        
        services['grafana'] = {
            'image': 'grafana/grafana:latest',
            'ports': ['3000:3000'],
            'environment': {
                'GF_SECURITY_ADMIN_PASSWORD': 'admin'
            },
            'restart': 'unless-stopped'
        }
        
        compose_config = {
            'version': '3.8',
            'services': services
        }
        
        return yaml.dump(compose_config, default_flow_style=False)
    
    def generate_kubernetes_manifests(self) -> Dict[str, str]:
        """生成Kubernetes部署清单"""
        manifests = {}
        
        for service_name, service_info in self.config.get('services', {}).items():
            # Deployment
            deployment = {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'metadata': {
                    'name': service_name,
                    'labels': {'app': service_name}
                },
                'spec': {
                    'replicas': service_info.get('replicas', 1),
                    'selector': {
                        'matchLabels': {'app': service_name}
                    },
                    'template': {
                        'metadata': {
                            'labels': {'app': service_name}
                        },
                        'spec': {
                            'containers': [{
                                'name': service_name,
                                'image': f'{service_name}:latest',
                                'ports': [{
                                    'containerPort': service_info['port']
                                }],
                                'resources': {
                                    'limits': {
                                        'cpu': '1000m',
                                        'memory': '1Gi'
                                    },
                                    'requests': {
                                        'cpu': '500m',
                                        'memory': '512Mi'
                                    }
                                },
                                'livenessProbe': {
                                    'httpGet': {
                                        'path': service_info.get('health_check_path', '/health'),
                                        'port': service_info['port']
                                    },
                                    'initialDelaySeconds': 30,
                                    'periodSeconds': 10
                                },
                                'readinessProbe': {
                                    'httpGet': {
                                        'path': service_info.get('health_check_path', '/health'),
                                        'port': service_info['port']
                                    },
                                    'initialDelaySeconds': 5,
                                    'periodSeconds': 5
                                }
                            }]
                        }
                    }
                }
            }
            
            # Service
            service = {
                'apiVersion': 'v1',
                'kind': 'Service',
                'metadata': {
                    'name': f'{service_name}-service',
                    'labels': {'app': service_name}
                },
                'spec': {
                    'selector': {'app': service_name},
                    'ports': [{
                        'port': service_info['port'],
                        'targetPort': service_info['port']
                    }],
                    'type': 'ClusterIP'
                }
            }
            
            manifests[f'{service_name}-deployment.yaml'] = yaml.dump(deployment)
            manifests[f'{service_name}-service.yaml'] = yaml.dump(service)
        
        return manifests
    
    def run(self, host: str = '0.0.0.0', port: int = 8000, debug: bool = False):
        """运行应用"""
        logger.info(f"启动企业级RAG部署服务，监听 {host}:{port}")
        
        # 注册默认服务
        for service_name, service_info in self.config.get('services', {}).items():
            service_config = ServiceConfig(
                name=service_name,
                image=f'{service_name}:latest',
                port=service_info['port'],
                replicas=service_info.get('replicas', 1),
                health_check_path=service_info.get('health_check_path', '/health')
            )
            self.register_service(service_config)
        
        # 设置信号处理
        def signal_handler(signum, frame):
            logger.info("收到停止信号，正在关闭服务...")
            self.shutdown()
            exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 启动Flask应用
        self.app.run(host=host, port=port, debug=debug, threaded=True)
    
    def shutdown(self):
        """关闭服务"""
        logger.info("正在关闭企业级RAG部署服务...")
        self.health_checker.stop()
        logger.info("服务已关闭")

def main():
    """示例用法"""
    print("企业级RAG部署系统测试\n" + "="*50)
    
    # 创建部署配置文件
    config = {
        'deployment_mode': 'docker_compose',
        'secret_key': 'your-secret-key-here',
        'services': {
            'rag_api': {
                'port': 8000,
                'replicas': 2,
                'health_check_path': '/health'
            },
            'vector_db': {
                'port': 6333,
                'replicas': 1,
                'health_check_path': '/health'
            }
        },
        'monitoring': {
            'metrics_port': 9090,
            'alert_webhook': 'https://hooks.slack.com/your-webhook-url'
        },
        'security': {
            'enable_auth': True,
            'rate_limit': 100
        }
    }
    
    # 保存配置文件
    with open('deployment_config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print("已生成部署配置文件: deployment_config.yaml")
    
    # 创建部署管理器
    deployment = EnterpriseRAGDeployment('deployment_config.yaml')
    
    # 生成Docker Compose配置
    docker_compose = deployment.generate_docker_compose()
    with open('docker-compose.yml', 'w', encoding='utf-8') as f:
        f.write(docker_compose)
    print("已生成Docker Compose配置: docker-compose.yml")
    
    # 生成Kubernetes清单
    k8s_manifests = deployment.generate_kubernetes_manifests()
    os.makedirs('k8s', exist_ok=True)
    for filename, content in k8s_manifests.items():
        with open(f'k8s/{filename}', 'w', encoding='utf-8') as f:
            f.write(content)
    print(f"已生成Kubernetes清单文件 ({len(k8s_manifests)} 个文件)")
    
    # 生成Prometheus配置
    prometheus_config = {
        'global': {
            'scrape_interval': '15s'
        },
        'scrape_configs': [
            {
                'job_name': 'rag-api',
                'static_configs': [
                    {'targets': ['localhost:8000']}
                ],
                'metrics_path': '/metrics'
            }
        ]
    }
    
    with open('prometheus.yml', 'w', encoding='utf-8') as f:
        yaml.dump(prometheus_config, f, default_flow_style=False)
    print("已生成Prometheus配置: prometheus.yml")
    
    print("\n部署文件生成完成！")
    print("\n使用说明:")
    print("1. Docker Compose部署: docker-compose up -d")
    print("2. Kubernetes部署: kubectl apply -f k8s/")
    print("3. 启动开发服务器: python lesson17_enterprise_rag_deployment.py")
    print("\n监控访问:")
    print("- API文档: http://localhost:8000/")
    print("- 指标监控: http://localhost:8000/metrics")
    print("- Prometheus: http://localhost:9090")
    print("- Grafana: http://localhost:3000 (admin/admin)")

if __name__ == "__main__":
    # 如果直接运行，启动开发服务器
    if len(os.sys.argv) > 1 and os.sys.argv[1] == 'demo':
        main()
    else:
        deployment = EnterpriseRAGDeployment()
        deployment.run(debug=True)