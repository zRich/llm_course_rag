# 实验5：系统可靠性实验（Lesson 19-20）

## 实验概述

本实验是RAG实战课程的最后一个综合实验，专注于系统可靠性和高可用性设计。基于前四个实验构建的完整RAG系统，学生将学习如何通过监控告警、故障注入、自动恢复机制和灾难恢复演练来确保系统在生产环境中的稳定运行。

## 实验目标

- 建立完整的系统监控和告警体系
- 实现故障注入和混沌工程实践
- 构建自动恢复和自愈机制
- 设计和执行灾难恢复方案
- 建立系统可靠性评估体系

## 涉及课程

- **Lesson 19**：监控告警与可观测性
- **Lesson 20**：故障注入与恢复演练

## 技术栈

### 核心技术组件
- **监控系统**：Prometheus、Grafana、AlertManager
- **日志系统**：ELK Stack (Elasticsearch, Logstash, Kibana)
- **链路追踪**：Jaeger、Zipkin
- **故障注入**：Chaos Monkey、Litmus、Gremlin
- **自动化运维**：Ansible、Terraform
- **备份恢复**：Velero、Restic
- **负载测试**：K6、JMeter、Locust

### 继承技术栈
- 实验1-4的所有技术组件

## 前置条件

- 完成实验1-4
- 理解可观测性三大支柱（指标、日志、链路）
- 熟悉混沌工程理念
- 了解SRE（站点可靠性工程）实践
- 具备故障排查经验

## 实验步骤

### 第一阶段：监控告警与可观测性（Lesson 19）

1. **多维度监控体系**
   ```python
   # monitoring/metrics.py
   from prometheus_client import Counter, Histogram, Gauge, Summary
   import time
   import functools
   
   # 业务指标
   REQUEST_COUNT = Counter('rag_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
   REQUEST_DURATION = Histogram('rag_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
   ACTIVE_CONNECTIONS = Gauge('rag_active_connections', 'Active connections')
   QUERY_QUALITY = Summary('rag_query_quality_score', 'Query quality score')
   
   # 系统指标
   MEMORY_USAGE = Gauge('rag_memory_usage_bytes', 'Memory usage')
   CPU_USAGE = Gauge('rag_cpu_usage_percent', 'CPU usage percentage')
   DISK_USAGE = Gauge('rag_disk_usage_bytes', 'Disk usage', ['mount_point'])
   
   # 业务逻辑指标
   DOCUMENT_COUNT = Gauge('rag_documents_total', 'Total documents in system')
   VECTOR_COUNT = Gauge('rag_vectors_total', 'Total vectors in database')
   CACHE_HIT_RATE = Gauge('rag_cache_hit_rate', 'Cache hit rate')
   
   def monitor_performance(func):
       """性能监控装饰器"""
       @functools.wraps(func)
       async def wrapper(*args, **kwargs):
           start_time = time.time()
           method = func.__name__
           
           try:
               result = await func(*args, **kwargs)
               REQUEST_COUNT.labels(method=method, endpoint='api', status='success').inc()
               return result
           except Exception as e:
               REQUEST_COUNT.labels(method=method, endpoint='api', status='error').inc()
               raise
           finally:
               duration = time.time() - start_time
               REQUEST_DURATION.labels(method=method, endpoint='api').observe(duration)
       
       return wrapper
   ```

2. **智能告警规则**
   ```yaml
   # alerting/rules.yml
   groups:
   - name: rag_system_alerts
     rules:
     # 高优先级告警
     - alert: RAGSystemDown
       expr: up{job="rag-backend"} == 0
       for: 1m
       labels:
         severity: critical
         team: platform
       annotations:
         summary: "RAG system is down"
         description: "RAG backend service has been down for more than 1 minute"
         runbook_url: "https://wiki.company.com/runbooks/rag-system-down"
   
     - alert: HighErrorRate
       expr: rate(rag_requests_total{status="error"}[5m]) / rate(rag_requests_total[5m]) > 0.1
       for: 2m
       labels:
         severity: critical
         team: platform
       annotations:
         summary: "High error rate detected"
         description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"
   
     # 中优先级告警
     - alert: HighResponseTime
       expr: histogram_quantile(0.95, rate(rag_request_duration_seconds_bucket[5m])) > 2
       for: 5m
       labels:
         severity: warning
         team: platform
       annotations:
         summary: "High response time"
         description: "95th percentile response time is {{ $value }}s"
   
     - alert: HighMemoryUsage
       expr: rag_memory_usage_bytes / (1024*1024*1024) > 8
       for: 10m
       labels:
         severity: warning
         team: platform
       annotations:
         summary: "High memory usage"
         description: "Memory usage is {{ $value }}GB"
   
     # 低优先级告警
     - alert: LowCacheHitRate
       expr: rag_cache_hit_rate < 0.7
       for: 15m
       labels:
         severity: info
         team: platform
       annotations:
         summary: "Low cache hit rate"
         description: "Cache hit rate is {{ $value | humanizePercentage }}"
   ```

3. **分布式链路追踪**
   ```python
   # tracing/tracer.py
   from opentelemetry import trace
   from opentelemetry.exporter.jaeger.thrift import JaegerExporter
   from opentelemetry.sdk.trace import TracerProvider
   from opentelemetry.sdk.trace.export import BatchSpanProcessor
   import functools
   
   class DistributedTracer:
       def __init__(self, service_name: str, jaeger_endpoint: str):
           self.service_name = service_name
           
           # 配置追踪器
           trace.set_tracer_provider(TracerProvider())
           tracer = trace.get_tracer(__name__)
           
           # 配置Jaeger导出器
           jaeger_exporter = JaegerExporter(
               agent_host_name="jaeger",
               agent_port=6831,
           )
           
           span_processor = BatchSpanProcessor(jaeger_exporter)
           trace.get_tracer_provider().add_span_processor(span_processor)
           
           self.tracer = tracer
       
       def trace_function(self, operation_name: str = None):
           """函数追踪装饰器"""
           def decorator(func):
               @functools.wraps(func)
               async def wrapper(*args, **kwargs):
                   span_name = operation_name or f"{func.__module__}.{func.__name__}"
                   
                   with self.tracer.start_as_current_span(span_name) as span:
                       # 添加标签
                       span.set_attribute("service.name", self.service_name)
                       span.set_attribute("function.name", func.__name__)
                       
                       try:
                           result = await func(*args, **kwargs)
                           span.set_attribute("result.success", True)
                           return result
                       except Exception as e:
                           span.set_attribute("result.success", False)
                           span.set_attribute("error.message", str(e))
                           span.record_exception(e)
                           raise
               
               return wrapper
           return decorator
   
   # 使用示例
   tracer = DistributedTracer("rag-backend", "http://jaeger:14268")
   
   @tracer.trace_function("query_processing")
   async def process_query(query: str) -> str:
       # 查询处理逻辑
       with tracer.tracer.start_as_current_span("vector_search") as span:
           vectors = await search_vectors(query)
           span.set_attribute("vectors.count", len(vectors))
       
       with tracer.tracer.start_as_current_span("llm_generation") as span:
           answer = await generate_answer(query, vectors)
           span.set_attribute("answer.length", len(answer))
       
       return answer
   ```

4. **日志聚合和分析**
   ```python
   # logging/structured_logger.py
   import json
   import logging
   from datetime import datetime
   from typing import Dict, Any
   
   class StructuredLogger:
       def __init__(self, service_name: str, version: str):
           self.service_name = service_name
           self.version = version
           self.logger = logging.getLogger(service_name)
           
           # 配置JSON格式化器
           handler = logging.StreamHandler()
           handler.setFormatter(self.JSONFormatter())
           self.logger.addHandler(handler)
           self.logger.setLevel(logging.INFO)
       
       class JSONFormatter(logging.Formatter):
           def format(self, record):
               log_entry = {
                   "timestamp": datetime.utcnow().isoformat(),
                   "level": record.levelname,
                   "message": record.getMessage(),
                   "module": record.module,
                   "function": record.funcName,
                   "line": record.lineno,
               }
               
               # 添加额外字段
               if hasattr(record, 'extra_fields'):
                   log_entry.update(record.extra_fields)
               
               return json.dumps(log_entry)
       
       def log_query(self, query: str, user_id: str, response_time: float, 
                    result_count: int, success: bool):
           """记录查询日志"""
           extra_fields = {
               "event_type": "query",
               "user_id": user_id,
               "query_length": len(query),
               "response_time": response_time,
               "result_count": result_count,
               "success": success,
               "service": self.service_name,
               "version": self.version
           }
           
           self.logger.info(
               f"Query processed: success={success}, time={response_time}s",
               extra={'extra_fields': extra_fields}
           )
       
       def log_error(self, error: Exception, context: Dict[str, Any]):
           """记录错误日志"""
           extra_fields = {
               "event_type": "error",
               "error_type": type(error).__name__,
               "error_message": str(error),
               "context": context,
               "service": self.service_name,
               "version": self.version
           }
           
           self.logger.error(
               f"Error occurred: {error}",
               extra={'extra_fields': extra_fields}
           )
   ```

### 第二阶段：故障注入与恢复演练（Lesson 20）

1. **混沌工程框架**
   ```python
   # chaos/chaos_monkey.py
   import random
   import asyncio
   import psutil
   from typing import List, Dict, Any
   from enum import Enum
   
   class FailureType(Enum):
       CPU_STRESS = "cpu_stress"
       MEMORY_LEAK = "memory_leak"
       NETWORK_DELAY = "network_delay"
       DISK_FULL = "disk_full"
       SERVICE_CRASH = "service_crash"
       DATABASE_SLOW = "database_slow"
   
   class ChaosMonkey:
       def __init__(self, config: Dict[str, Any]):
           self.config = config
           self.active_experiments = {}
           self.metrics_collector = MetricsCollector()
       
       async def run_experiment(self, experiment_id: str, failure_type: FailureType, 
                              duration: int, intensity: float):
           """运行混沌实验"""
           print(f"Starting chaos experiment: {experiment_id}")
           
           try:
               # 记录实验开始
               await self.metrics_collector.record_experiment_start(
                   experiment_id, failure_type, intensity
               )
               
               # 执行故障注入
               if failure_type == FailureType.CPU_STRESS:
                   await self.inject_cpu_stress(intensity, duration)
               elif failure_type == FailureType.MEMORY_LEAK:
                   await self.inject_memory_leak(intensity, duration)
               elif failure_type == FailureType.NETWORK_DELAY:
                   await self.inject_network_delay(intensity, duration)
               elif failure_type == FailureType.DATABASE_SLOW:
                   await self.inject_database_slowdown(intensity, duration)
               
               # 监控系统响应
               await self.monitor_system_response(experiment_id, duration)
               
           except Exception as e:
               print(f"Experiment {experiment_id} failed: {e}")
           finally:
               # 清理和恢复
               await self.cleanup_experiment(experiment_id)
               await self.metrics_collector.record_experiment_end(experiment_id)
       
       async def inject_cpu_stress(self, intensity: float, duration: int):
           """注入CPU压力"""
           def cpu_stress():
               end_time = time.time() + duration
               while time.time() < end_time:
                   # 消耗CPU资源
                   for _ in range(int(1000000 * intensity)):
                       pass
                   time.sleep(0.01)
           
           # 在后台线程中运行CPU压力
           import threading
           thread = threading.Thread(target=cpu_stress)
           thread.start()
           
           await asyncio.sleep(duration)
           
       async def inject_memory_leak(self, intensity: float, duration: int):
           """注入内存泄漏"""
           memory_hog = []
           leak_size = int(1024 * 1024 * intensity)  # MB
           
           try:
               for _ in range(duration):
                   # 分配内存但不释放
                   memory_hog.append(bytearray(leak_size))
                   await asyncio.sleep(1)
           finally:
               # 清理内存
               del memory_hog
       
       async def inject_network_delay(self, delay_ms: float, duration: int):
           """注入网络延迟"""
           # 使用tc命令添加网络延迟
           import subprocess
           
           try:
               # 添加延迟
               subprocess.run([
                   "tc", "qdisc", "add", "dev", "eth0", "root", "netem", 
                   "delay", f"{delay_ms}ms"
               ], check=True)
               
               await asyncio.sleep(duration)
               
           finally:
               # 移除延迟
               subprocess.run([
                   "tc", "qdisc", "del", "dev", "eth0", "root"
               ], check=False)
   ```

2. **自动恢复机制**
   ```python
   # recovery/auto_recovery.py
   import asyncio
   from typing import Dict, List, Callable
   from dataclasses import dataclass
   from enum import Enum
   
   class HealthStatus(Enum):
       HEALTHY = "healthy"
       DEGRADED = "degraded"
       UNHEALTHY = "unhealthy"
       CRITICAL = "critical"
   
   @dataclass
   class HealthCheck:
       name: str
       check_function: Callable
       threshold: float
       recovery_action: Callable
       max_retries: int = 3
   
   class AutoRecoverySystem:
       def __init__(self):
           self.health_checks: List[HealthCheck] = []
           self.recovery_actions: Dict[str, Callable] = {}
           self.circuit_breakers: Dict[str, CircuitBreaker] = {}
       
       def register_health_check(self, health_check: HealthCheck):
           """注册健康检查"""
           self.health_checks.append(health_check)
       
       async def monitor_system_health(self):
           """持续监控系统健康状态"""
           while True:
               try:
                   for check in self.health_checks:
                       status = await self.perform_health_check(check)
                       
                       if status != HealthStatus.HEALTHY:
                           await self.trigger_recovery(check, status)
                   
                   await asyncio.sleep(30)  # 每30秒检查一次
                   
               except Exception as e:
                   print(f"Health monitoring error: {e}")
                   await asyncio.sleep(60)
       
       async def perform_health_check(self, check: HealthCheck) -> HealthStatus:
           """执行健康检查"""
           try:
               result = await check.check_function()
               
               if result >= check.threshold:
                   return HealthStatus.HEALTHY
               elif result >= check.threshold * 0.7:
                   return HealthStatus.DEGRADED
               elif result >= check.threshold * 0.3:
                   return HealthStatus.UNHEALTHY
               else:
                   return HealthStatus.CRITICAL
                   
           except Exception as e:
               print(f"Health check {check.name} failed: {e}")
               return HealthStatus.CRITICAL
       
       async def trigger_recovery(self, check: HealthCheck, status: HealthStatus):
           """触发自动恢复"""
           print(f"Triggering recovery for {check.name}, status: {status}")
           
           for attempt in range(check.max_retries):
               try:
                   await check.recovery_action()
                   
                   # 验证恢复效果
                   await asyncio.sleep(10)
                   new_status = await self.perform_health_check(check)
                   
                   if new_status == HealthStatus.HEALTHY:
                       print(f"Recovery successful for {check.name}")
                       break
                       
               except Exception as e:
                   print(f"Recovery attempt {attempt + 1} failed: {e}")
           
           else:
               print(f"All recovery attempts failed for {check.name}")
               await self.escalate_incident(check, status)
   
   # 健康检查实现
   async def check_database_health() -> float:
       """检查数据库健康状态"""
       try:
           start_time = time.time()
           # 执行简单查询
           await database.execute("SELECT 1")
           response_time = time.time() - start_time
           
           # 返回健康分数（响应时间越短分数越高）
           return max(0, 1 - response_time)
       except:
           return 0
   
   async def restart_database_connection():
       """重启数据库连接"""
       await database.disconnect()
       await asyncio.sleep(5)
       await database.connect()
   
   # 注册健康检查
   recovery_system = AutoRecoverySystem()
   recovery_system.register_health_check(
       HealthCheck(
           name="database",
           check_function=check_database_health,
           threshold=0.8,
           recovery_action=restart_database_connection,
           max_retries=3
       )
   )
   ```

3. **熔断器模式**
   ```python
   # recovery/circuit_breaker.py
   import time
   import asyncio
   from enum import Enum
   from typing import Callable, Any
   
   class CircuitState(Enum):
       CLOSED = "closed"      # 正常状态
       OPEN = "open"          # 熔断状态
       HALF_OPEN = "half_open"  # 半开状态
   
   class CircuitBreaker:
       def __init__(self, failure_threshold: int = 5, timeout: int = 60, 
                    expected_exception: Exception = Exception):
           self.failure_threshold = failure_threshold
           self.timeout = timeout
           self.expected_exception = expected_exception
           
           self.failure_count = 0
           self.last_failure_time = None
           self.state = CircuitState.CLOSED
       
       async def call(self, func: Callable, *args, **kwargs) -> Any:
           """通过熔断器调用函数"""
           if self.state == CircuitState.OPEN:
               if self._should_attempt_reset():
                   self.state = CircuitState.HALF_OPEN
               else:
                   raise CircuitBreakerOpenError("Circuit breaker is open")
           
           try:
               result = await func(*args, **kwargs)
               self._on_success()
               return result
               
           except self.expected_exception as e:
               self._on_failure()
               raise e
       
       def _should_attempt_reset(self) -> bool:
           """判断是否应该尝试重置熔断器"""
           return (time.time() - self.last_failure_time) >= self.timeout
       
       def _on_success(self):
           """成功时的处理"""
           self.failure_count = 0
           self.state = CircuitState.CLOSED
       
       def _on_failure(self):
           """失败时的处理"""
           self.failure_count += 1
           self.last_failure_time = time.time()
           
           if self.failure_count >= self.failure_threshold:
               self.state = CircuitState.OPEN
   
   class CircuitBreakerOpenError(Exception):
       pass
   
   # 使用示例
   db_circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30)
   
   async def query_with_circuit_breaker(query: str):
       """带熔断器的数据库查询"""
       return await db_circuit_breaker.call(database.execute, query)
   ```

4. **灾难恢复演练**
   ```python
   # disaster_recovery/dr_orchestrator.py
   import asyncio
   from typing import List, Dict, Any
   from dataclasses import dataclass
   from enum import Enum
   
   class DisasterType(Enum):
       DATABASE_FAILURE = "database_failure"
       NETWORK_PARTITION = "network_partition"
       DATA_CENTER_OUTAGE = "data_center_outage"
       SECURITY_BREACH = "security_breach"
   
   @dataclass
   class RecoveryStep:
       name: str
       action: Callable
       timeout: int
       rollback_action: Callable = None
       critical: bool = True
   
   class DisasterRecoveryOrchestrator:
       def __init__(self):
           self.recovery_plans: Dict[DisasterType, List[RecoveryStep]] = {}
           self.backup_systems = BackupManager()
           self.notification_system = NotificationSystem()
       
       def register_recovery_plan(self, disaster_type: DisasterType, 
                                steps: List[RecoveryStep]):
           """注册灾难恢复计划"""
           self.recovery_plans[disaster_type] = steps
       
       async def execute_disaster_recovery(self, disaster_type: DisasterType, 
                                         context: Dict[str, Any]):
           """执行灾难恢复"""
           print(f"Executing disaster recovery for {disaster_type}")
           
           # 发送紧急通知
           await self.notification_system.send_emergency_alert(
               f"Disaster recovery initiated: {disaster_type}"
           )
           
           recovery_steps = self.recovery_plans.get(disaster_type, [])
           completed_steps = []
           
           try:
               for step in recovery_steps:
                   print(f"Executing recovery step: {step.name}")
                   
                   try:
                       await asyncio.wait_for(
                           step.action(context), 
                           timeout=step.timeout
                       )
                       completed_steps.append(step)
                       print(f"Step {step.name} completed successfully")
                       
                   except asyncio.TimeoutError:
                       print(f"Step {step.name} timed out")
                       if step.critical:
                           raise DisasterRecoveryError(f"Critical step {step.name} failed")
                   
                   except Exception as e:
                       print(f"Step {step.name} failed: {e}")
                       if step.critical:
                           raise DisasterRecoveryError(f"Critical step {step.name} failed: {e}")
               
               print("Disaster recovery completed successfully")
               await self.notification_system.send_recovery_success_alert()
               
           except Exception as e:
               print(f"Disaster recovery failed: {e}")
               
               # 执行回滚
               await self.rollback_recovery_steps(completed_steps)
               
               await self.notification_system.send_recovery_failure_alert(str(e))
               raise
       
       async def rollback_recovery_steps(self, completed_steps: List[RecoveryStep]):
           """回滚已完成的恢复步骤"""
           for step in reversed(completed_steps):
               if step.rollback_action:
                   try:
                       await step.rollback_action()
                       print(f"Rolled back step: {step.name}")
                   except Exception as e:
                       print(f"Rollback failed for step {step.name}: {e}")
   
   # 恢复步骤实现
   async def switch_to_backup_database(context: Dict[str, Any]):
       """切换到备份数据库"""
       backup_db_url = context.get("backup_db_url")
       await database.switch_connection(backup_db_url)
   
   async def restore_from_backup(context: Dict[str, Any]):
       """从备份恢复数据"""
       backup_timestamp = context.get("backup_timestamp")
       await backup_manager.restore_backup(backup_timestamp)
   
   async def update_dns_records(context: Dict[str, Any]):
       """更新DNS记录指向备用服务器"""
       backup_ip = context.get("backup_ip")
       await dns_manager.update_record("rag.example.com", backup_ip)
   
   # 注册恢复计划
   dr_orchestrator = DisasterRecoveryOrchestrator()
   dr_orchestrator.register_recovery_plan(
       DisasterType.DATABASE_FAILURE,
       [
           RecoveryStep("switch_to_backup_db", switch_to_backup_database, 30),
           RecoveryStep("restore_from_backup", restore_from_backup, 300),
           RecoveryStep("verify_data_integrity", verify_data_integrity, 60),
       ]
   )
   ```

## 实验任务

### 任务1：监控告警体系

**目标**：建立完整的系统监控和告警体系

**具体要求**：
1. 配置Prometheus + Grafana监控栈
2. 实现多维度指标收集（业务、系统、应用）
3. 设计智能告警规则和通知机制
4. 建立分布式链路追踪系统

**评估指标**：
- 监控覆盖率（>95%）
- 告警准确率（>90%）
- 平均故障发现时间（<5分钟）
- 误报率（<5%）

### 任务2：故障注入实验

**目标**：通过混沌工程验证系统韧性

**具体要求**：
1. 实现至少5种不同类型的故障注入
2. 设计渐进式故障注入策略
3. 监控和记录系统在故障下的表现
4. 分析系统薄弱环节并提出改进建议

**评估指标**：
- 故障注入覆盖率
- 系统恢复时间
- 数据一致性保证
- 用户体验影响程度

### 任务3：自动恢复机制

**目标**：实现智能的自动恢复和自愈系统

**具体要求**：
1. 实现健康检查和自动恢复机制
2. 配置熔断器和限流保护
3. 建立多级恢复策略
4. 实现故障隔离和降级机制

**评估指标**：
- 自动恢复成功率（>80%）
- 恢复时间（<2分钟）
- 服务可用性（>99.9%）
- 故障影响范围控制

### 任务4：灾难恢复演练

**目标**：验证灾难恢复计划的有效性

**具体要求**：
1. 设计多种灾难场景的恢复计划
2. 实现自动化的灾难恢复流程
3. 进行定期的恢复演练
4. 建立恢复时间目标（RTO）和恢复点目标（RPO）

**评估指标**：
- 恢复计划完整性
- 恢复时间目标达成率
- 数据丢失量控制
- 演练成功率

## 可靠性指标体系

### SLI（服务水平指标）
```python
# reliability/sli_metrics.py
from dataclasses import dataclass
from typing import Dict, List
import time

@dataclass
class SLI:
    name: str
    description: str
    measurement_window: int  # 秒
    target_value: float
    current_value: float = 0.0

class SLICollector:
    def __init__(self):
        self.slis: Dict[str, SLI] = {
            "availability": SLI(
                name="availability",
                description="Service availability percentage",
                measurement_window=3600,  # 1小时
                target_value=99.9
            ),
            "latency_p95": SLI(
                name="latency_p95",
                description="95th percentile response time",
                measurement_window=300,   # 5分钟
                target_value=1.0  # 1秒
            ),
            "error_rate": SLI(
                name="error_rate",
                description="Error rate percentage",
                measurement_window=300,   # 5分钟
                target_value=1.0  # 1%
            ),
            "throughput": SLI(
                name="throughput",
                description="Requests per second",
                measurement_window=60,    # 1分钟
                target_value=100.0
            )
        }
    
    async def collect_sli_metrics(self) -> Dict[str, float]:
        """收集SLI指标"""
        metrics = {}
        
        # 可用性
        uptime_ratio = await self.calculate_uptime_ratio()
        self.slis["availability"].current_value = uptime_ratio * 100
        metrics["availability"] = uptime_ratio * 100
        
        # 延迟
        p95_latency = await self.calculate_p95_latency()
        self.slis["latency_p95"].current_value = p95_latency
        metrics["latency_p95"] = p95_latency
        
        # 错误率
        error_rate = await self.calculate_error_rate()
        self.slis["error_rate"].current_value = error_rate * 100
        metrics["error_rate"] = error_rate * 100
        
        # 吞吐量
        throughput = await self.calculate_throughput()
        self.slis["throughput"].current_value = throughput
        metrics["throughput"] = throughput
        
        return metrics
```

### SLO（服务水平目标）
```yaml
# reliability/slo_config.yml
slos:
  - name: "availability_slo"
    description: "Service should be available 99.9% of the time"
    sli: "availability"
    target: 99.9
    measurement_window: "30d"
    alert_threshold: 99.5
    
  - name: "latency_slo"
    description: "95% of requests should complete within 1 second"
    sli: "latency_p95"
    target: 1.0
    measurement_window: "1h"
    alert_threshold: 1.5
    
  - name: "error_budget_slo"
    description: "Error rate should be less than 1%"
    sli: "error_rate"
    target: 1.0
    measurement_window: "1h"
    alert_threshold: 2.0
```

## 故障演练场景

### 场景1：数据库故障
```python
async def database_failure_drill():
    """数据库故障演练"""
    print("Starting database failure drill...")
    
    # 1. 模拟主数据库故障
    await chaos_monkey.inject_database_failure()
    
    # 2. 验证自动切换到备库
    await verify_database_failover()
    
    # 3. 测试读写功能
    await test_database_operations()
    
    # 4. 恢复主数据库
    await restore_primary_database()
    
    # 5. 验证数据一致性
    await verify_data_consistency()
```

### 场景2：网络分区
```python
async def network_partition_drill():
    """网络分区演练"""
    print("Starting network partition drill...")
    
    # 1. 模拟网络分区
    await chaos_monkey.inject_network_partition()
    
    # 2. 验证服务降级
    await verify_service_degradation()
    
    # 3. 测试部分功能可用性
    await test_partial_functionality()
    
    # 4. 恢复网络连接
    await restore_network_connectivity()
    
    # 5. 验证服务恢复
    await verify_service_recovery()
```

## 评估标准

### 监控告警（25分）

- [ ] 监控覆盖率（10分）
- [ ] 告警准确性（8分）
- [ ] 可视化质量（7分）

### 故障注入（25分）

- [ ] 故障类型覆盖（10分）
- [ ] 实验设计科学性（8分）
- [ ] 结果分析深度（7分）

### 自动恢复（25分）

- [ ] 恢复机制完整性（10分）
- [ ] 恢复成功率（8分）
- [ ] 恢复时间（7分）

### 灾难恢复（25分）

- [ ] 恢复计划完整性（10分）
- [ ] 演练执行效果（8分）
- [ ] RTO/RPO达成（7分）

## 常见问题

### 监控问题

**Q: 监控数据不准确？**
A: 检查指标定义、采集频率、聚合方式。

**Q: 告警风暴？**
A: 优化告警规则、增加抑制机制、调整阈值。

**Q: 链路追踪丢失？**
A: 检查采样率、网络连接、存储容量。

### 恢复问题

**Q: 自动恢复失败？**
A: 检查健康检查逻辑、恢复动作、权限配置。

**Q: 数据不一致？**
A: 验证备份策略、同步机制、一致性检查。

## 参考资源

- [Site Reliability Engineering](https://sre.google/books/)
- [Chaos Engineering](https://principlesofchaos.org/)
- [Prometheus监控指南](https://prometheus.io/docs/)
- [Grafana可视化](https://grafana.com/docs/)
- [Jaeger链路追踪](https://www.jaegertracing.io/docs/)

## 实验时间安排

- **理论学习**：4-5小时（SRE和混沌工程理论）
- **监控系统搭建**：8-10小时（Prometheus + Grafana）
- **告警配置**：4-6小时（告警规则和通知）
- **故障注入开发**：8-10小时（混沌工程实现）
- **自动恢复开发**：6-8小时（恢复机制实现）
- **灾难恢复计划**：6-8小时（DR计划和演练）
- **测试和验证**：8-10小时（各种场景测试）
- **报告撰写**：4-5小时

**总计**：48-62小时

## 提交要求

1. **完整的监控系统**：包含监控、告警、可视化
2. **故障注入框架**：可重复执行的混沌实验
3. **自动恢复系统**：智能的故障检测和恢复
4. **灾难恢复手册**：详细的恢复计划和操作指南
5. **可靠性评估报告**：系统可靠性分析和改进建议

## 课程总结

完成本实验后，学生将掌握：

1. **完整的RAG系统开发能力**：从基础搭建到生产部署
2. **企业级系统设计思维**：性能、可靠性、可扩展性
3. **DevOps和SRE实践**：监控、部署、运维自动化
4. **故障处理和恢复能力**：混沌工程和灾难恢复
5. **系统优化和调优技能**：性能分析和持续改进

这套完整的实验体系为学生提供了从理论到实践的全面训练，确保能够在实际工作中构建和维护高质量的RAG系统。