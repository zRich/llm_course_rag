"""系统监控器模块

提供系统健康检查、状态监控和综合监控功能。
"""

import time
import threading
import psutil
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    WARNING = "warning"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """健康检查配置"""
    name: str
    check_function: Callable[[], bool]
    description: str = ""
    timeout_seconds: float = 5.0
    interval_seconds: float = 30.0
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    duration_ms: float
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    load_average: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_available_mb': self.memory_available_mb,
            'disk_usage_percent': self.disk_usage_percent,
            'disk_free_gb': self.disk_free_gb,
            'network_bytes_sent': self.network_bytes_sent,
            'network_bytes_recv': self.network_bytes_recv,
            'process_count': self.process_count,
            'load_average': self.load_average
        }


class HealthChecker(ABC):
    """健康检查器抽象基类"""
    
    @abstractmethod
    def check(self) -> bool:
        """执行健康检查
        
        Returns:
            检查是否通过
        """
        pass


class DatabaseHealthChecker(HealthChecker):
    """数据库健康检查器"""
    
    def __init__(self, connection_test_func: Callable[[], bool]):
        self.connection_test_func = connection_test_func
    
    def check(self) -> bool:
        """检查数据库连接"""
        try:
            return self.connection_test_func()
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False


class ServiceHealthChecker(HealthChecker):
    """服务健康检查器"""
    
    def __init__(self, service_url: str, timeout: float = 5.0):
        self.service_url = service_url
        self.timeout = timeout
    
    def check(self) -> bool:
        """检查服务可用性"""
        try:
            import requests
            response = requests.get(self.service_url, timeout=self.timeout)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Service health check failed for {self.service_url}: {str(e)}")
            return False


class MemoryHealthChecker(HealthChecker):
    """内存健康检查器"""
    
    def __init__(self, max_usage_percent: float = 90.0):
        self.max_usage_percent = max_usage_percent
    
    def check(self) -> bool:
        """检查内存使用率"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent < self.max_usage_percent
        except Exception as e:
            logger.error(f"Memory health check failed: {str(e)}")
            return False


class DiskHealthChecker(HealthChecker):
    """磁盘健康检查器"""
    
    def __init__(self, path: str = "/", max_usage_percent: float = 90.0):
        self.path = path
        self.max_usage_percent = max_usage_percent
    
    def check(self) -> bool:
        """检查磁盘使用率"""
        try:
            disk = psutil.disk_usage(self.path)
            usage_percent = (disk.used / disk.total) * 100
            return usage_percent < self.max_usage_percent
        except Exception as e:
            logger.error(f"Disk health check failed: {str(e)}")
            return False


class SystemMonitorStatistics:
    """系统监控统计"""
    
    def __init__(self):
        self.total_checks = 0
        self.successful_checks = 0
        self.failed_checks = 0
        self.checks_by_name = {}
        self.system_metrics_history = []
        self.health_history = []
        self._lock = threading.RLock()
    
    def record_check(self, result: HealthCheckResult):
        """记录健康检查结果"""
        with self._lock:
            self.total_checks += 1
            
            if result.status == HealthStatus.HEALTHY:
                self.successful_checks += 1
            else:
                self.failed_checks += 1
            
            if result.name not in self.checks_by_name:
                self.checks_by_name[result.name] = {
                    'total': 0,
                    'successful': 0,
                    'failed': 0,
                    'last_result': None
                }
            
            check_stats = self.checks_by_name[result.name]
            check_stats['total'] += 1
            check_stats['last_result'] = result
            
            if result.status == HealthStatus.HEALTHY:
                check_stats['successful'] += 1
            else:
                check_stats['failed'] += 1
            
            # 记录健康检查历史
            self.health_history.append({
                'timestamp': result.timestamp,
                'name': result.name,
                'status': result.status.value,
                'duration_ms': result.duration_ms
            })
            
            # 保持历史记录在合理范围内
            if len(self.health_history) > 1000:
                self.health_history = self.health_history[-500:]
    
    def record_system_metrics(self, metrics: SystemMetrics):
        """记录系统指标"""
        with self._lock:
            self.system_metrics_history.append(metrics.to_dict())
            
            # 保持历史记录在合理范围内
            if len(self.system_metrics_history) > 1000:
                self.system_metrics_history = self.system_metrics_history[-500:]
    
    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        with self._lock:
            success_rate = (self.successful_checks / self.total_checks * 100) if self.total_checks > 0 else 0
            
            return {
                'total_checks': self.total_checks,
                'successful_checks': self.successful_checks,
                'failed_checks': self.failed_checks,
                'success_rate': round(success_rate, 2),
                'checks_by_name': dict(self.checks_by_name),
                'recent_health_checks': self.health_history[-10:] if self.health_history else [],
                'recent_system_metrics': self.system_metrics_history[-5:] if self.system_metrics_history else []
            }


class SystemMonitor:
    """系统监控器
    
    提供系统健康检查、指标收集和监控功能。
    """
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.health_checks: Dict[str, HealthCheck] = {}
        self.statistics = SystemMonitorStatistics()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # 添加默认的系统健康检查
        self._add_default_checks()
        
        logger.info(f"System monitor initialized with check interval: {check_interval}s")
    
    def _add_default_checks(self):
        """添加默认的系统健康检查"""
        # 内存检查
        memory_checker = MemoryHealthChecker(max_usage_percent=90.0)
        self.add_health_check(HealthCheck(
            name="memory_usage",
            check_function=memory_checker.check,
            description="Check system memory usage",
            interval_seconds=30.0
        ))
        
        # 磁盘检查
        disk_checker = DiskHealthChecker(max_usage_percent=90.0)
        self.add_health_check(HealthCheck(
            name="disk_usage",
            check_function=disk_checker.check,
            description="Check system disk usage",
            interval_seconds=60.0
        ))
    
    def add_health_check(self, health_check: HealthCheck):
        """添加健康检查
        
        Args:
            health_check: 健康检查配置
        """
        with self._lock:
            self.health_checks[health_check.name] = health_check
            logger.info(f"Added health check: {health_check.name}")
    
    def remove_health_check(self, name: str) -> bool:
        """移除健康检查
        
        Args:
            name: 健康检查名称
            
        Returns:
            是否移除成功
        """
        with self._lock:
            if name in self.health_checks:
                del self.health_checks[name]
                logger.info(f"Removed health check: {name}")
                return True
            return False
    
    def run_health_check(self, name: str) -> Optional[HealthCheckResult]:
        """运行单个健康检查
        
        Args:
            name: 健康检查名称
            
        Returns:
            健康检查结果
        """
        with self._lock:
            if name not in self.health_checks:
                logger.warning(f"Health check not found: {name}")
                return None
            
            health_check = self.health_checks[name]
            
            if not health_check.enabled:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message="Health check is disabled",
                    timestamp=time.time(),
                    duration_ms=0
                )
            
            start_time = time.time()
            
            try:
                # 使用超时执行检查
                result = self._run_with_timeout(
                    health_check.check_function,
                    health_check.timeout_seconds
                )
                
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = "Health check passed" if result else "Health check failed"
                
                check_result = HealthCheckResult(
                    name=name,
                    status=status,
                    message=message,
                    timestamp=start_time,
                    duration_ms=duration_ms
                )
                
            except TimeoutError:
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                
                check_result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check timed out after {health_check.timeout_seconds}s",
                    timestamp=start_time,
                    duration_ms=duration_ms,
                    error="Timeout"
                )
                
            except Exception as e:
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                
                check_result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed with error: {str(e)}",
                    timestamp=start_time,
                    duration_ms=duration_ms,
                    error=str(e)
                )
            
            # 记录统计
            self.statistics.record_check(check_result)
            
            return check_result
    
    def _run_with_timeout(self, func: Callable, timeout: float) -> Any:
        """在超时限制内运行函数
        
        Args:
            func: 要执行的函数
            timeout: 超时时间（秒）
            
        Returns:
            函数执行结果
            
        Raises:
            TimeoutError: 如果执行超时
        """
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function execution timed out after {timeout}s")
        
        # 设置超时信号
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = func()
            signal.alarm(0)  # 取消超时
            return result
        finally:
            signal.signal(signal.SIGALRM, old_handler)
    
    def run_all_health_checks(self) -> List[HealthCheckResult]:
        """运行所有健康检查
        
        Returns:
            所有健康检查结果
        """
        results = []
        
        with self._lock:
            for name in self.health_checks.keys():
                result = self.run_health_check(name)
                if result:
                    results.append(result)
        
        return results
    
    def get_system_metrics(self) -> SystemMetrics:
        """获取系统指标
        
        Returns:
            系统指标
        """
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存信息
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / (1024 * 1024)
            
            # 磁盘信息
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024 * 1024 * 1024)
            
            # 网络信息
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # 进程数量
            process_count = len(psutil.pids())
            
            # 负载平均值（仅在Unix系统上可用）
            load_average = None
            try:
                load_average = list(psutil.getloadavg())
            except AttributeError:
                pass  # Windows系统不支持
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                process_count=process_count,
                load_average=load_average
            )
            
            # 记录统计
            self.statistics.record_system_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {str(e)}")
            raise
    
    def get_overall_health(self) -> HealthStatus:
        """获取系统整体健康状态
        
        Returns:
            整体健康状态
        """
        results = self.run_all_health_checks()
        
        if not results:
            return HealthStatus.UNKNOWN
        
        # 如果有任何检查失败，则认为系统不健康
        unhealthy_count = sum(1 for result in results if result.status == HealthStatus.UNHEALTHY)
        warning_count = sum(1 for result in results if result.status == HealthStatus.WARNING)
        
        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY
        elif warning_count > 0:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def start_monitoring(self):
        """启动监控"""
        with self._lock:
            if self._running:
                logger.warning("System monitor is already running")
                return
            
            self._running = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            
            logger.info("System monitor started")
    
    def stop_monitoring(self):
        """停止监控"""
        with self._lock:
            if not self._running:
                logger.warning("System monitor is not running")
                return
            
            self._running = False
            
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)
            
            logger.info("System monitor stopped")
    
    def _monitor_loop(self):
        """监控循环"""
        logger.info("System monitor loop started")
        
        while self._running:
            try:
                # 运行健康检查
                self.run_all_health_checks()
                
                # 收集系统指标
                self.get_system_metrics()
                
                # 等待下一次检查
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {str(e)}")
                time.sleep(self.check_interval)
        
        logger.info("System monitor loop stopped")
    
    def enable_health_check(self, name: str) -> bool:
        """启用健康检查
        
        Args:
            name: 健康检查名称
            
        Returns:
            是否操作成功
        """
        with self._lock:
            if name in self.health_checks:
                self.health_checks[name].enabled = True
                logger.info(f"Enabled health check: {name}")
                return True
            return False
    
    def disable_health_check(self, name: str) -> bool:
        """禁用健康检查
        
        Args:
            name: 健康检查名称
            
        Returns:
            是否操作成功
        """
        with self._lock:
            if name in self.health_checks:
                self.health_checks[name].enabled = False
                logger.info(f"Disabled health check: {name}")
                return True
            return False
    
    def get_health_checks(self) -> List[HealthCheck]:
        """获取所有健康检查"""
        with self._lock:
            return list(self.health_checks.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats = self.statistics.get_summary()
            stats.update({
                'total_health_checks': len(self.health_checks),
                'enabled_health_checks': sum(1 for check in self.health_checks.values() if check.enabled),
                'disabled_health_checks': sum(1 for check in self.health_checks.values() if not check.enabled),
                'monitoring_running': self._running,
                'check_interval': self.check_interval
            })
            return stats
    
    def clear_statistics(self):
        """清空统计信息"""
        with self._lock:
            self.statistics = SystemMonitorStatistics()
            logger.info("System monitor statistics cleared")


# 全局系统监控器实例
global_system_monitor = SystemMonitor()