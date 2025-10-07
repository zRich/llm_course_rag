"""告警管理器模块

管理告警规则、触发告警和发送通知。
"""

import time
import threading
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCondition(Enum):
    """告警条件"""
    GREATER_THAN = "gt"          # 大于
    LESS_THAN = "lt"             # 小于
    EQUAL = "eq"                 # 等于
    NOT_EQUAL = "ne"             # 不等于
    GREATER_EQUAL = "ge"         # 大于等于
    LESS_EQUAL = "le"            # 小于等于
    CONTAINS = "contains"        # 包含
    NOT_CONTAINS = "not_contains" # 不包含


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    metric_name: str
    condition: AlertCondition
    threshold: Union[float, str]
    level: AlertLevel = AlertLevel.WARNING
    description: str = ""
    enabled: bool = True
    cooldown_seconds: float = 300.0  # 冷却时间
    evaluation_window: float = 60.0  # 评估窗口
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        if self.evaluation_window <= 0:
            raise ValueError("evaluation_window must be positive")


@dataclass
class AlertEvent:
    """告警事件"""
    rule_name: str
    metric_name: str
    level: AlertLevel
    message: str
    current_value: Any
    threshold: Union[float, str]
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None


class AlertNotifier(ABC):
    """告警通知器抽象基类"""
    
    @abstractmethod
    def send_alert(self, event: AlertEvent) -> bool:
        """发送告警
        
        Args:
            event: 告警事件
            
        Returns:
            是否发送成功
        """
        pass


class LogNotifier(AlertNotifier):
    """日志通知器"""
    
    def send_alert(self, event: AlertEvent) -> bool:
        """通过日志发送告警"""
        try:
            log_level = {
                AlertLevel.INFO: logging.INFO,
                AlertLevel.WARNING: logging.WARNING,
                AlertLevel.ERROR: logging.ERROR,
                AlertLevel.CRITICAL: logging.CRITICAL
            }.get(event.level, logging.WARNING)
            
            message = f"ALERT [{event.level.value.upper()}] {event.rule_name}: {event.message} (current: {event.current_value}, threshold: {event.threshold})"
            logger.log(log_level, message)
            return True
        except Exception as e:
            logger.error(f"Failed to send log alert: {str(e)}")
            return False


class ConsoleNotifier(AlertNotifier):
    """控制台通知器"""
    
    def send_alert(self, event: AlertEvent) -> bool:
        """通过控制台发送告警"""
        try:
            timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))
            print(f"\n{'='*60}")
            print(f"ALERT: {event.rule_name}")
            print(f"Level: {event.level.value.upper()}")
            print(f"Time: {timestamp_str}")
            print(f"Metric: {event.metric_name}")
            print(f"Message: {event.message}")
            print(f"Current Value: {event.current_value}")
            print(f"Threshold: {event.threshold}")
            if event.tags:
                print(f"Tags: {event.tags}")
            print(f"{'='*60}\n")
            return True
        except Exception as e:
            logger.error(f"Failed to send console alert: {str(e)}")
            return False


class CallbackNotifier(AlertNotifier):
    """回调函数通知器"""
    
    def __init__(self, callback: Callable[[AlertEvent], bool]):
        self.callback = callback
    
    def send_alert(self, event: AlertEvent) -> bool:
        """通过回调函数发送告警"""
        try:
            return self.callback(event)
        except Exception as e:
            logger.error(f"Failed to send callback alert: {str(e)}")
            return False


class AlertStatistics:
    """告警统计"""
    
    def __init__(self):
        self.total_alerts = 0
        self.alerts_by_level = {level: 0 for level in AlertLevel}
        self.alerts_by_rule = {}
        self.resolved_alerts = 0
        self.failed_notifications = 0
        self.alert_history = []
        self._lock = threading.RLock()
    
    def record_alert(self, event: AlertEvent, notification_success: bool = True):
        """记录告警"""
        with self._lock:
            self.total_alerts += 1
            self.alerts_by_level[event.level] += 1
            
            if event.rule_name not in self.alerts_by_rule:
                self.alerts_by_rule[event.rule_name] = 0
            self.alerts_by_rule[event.rule_name] += 1
            
            if not notification_success:
                self.failed_notifications += 1
            
            # 记录告警历史
            self.alert_history.append({
                'timestamp': event.timestamp,
                'rule_name': event.rule_name,
                'level': event.level.value,
                'message': event.message,
                'notification_success': notification_success
            })
            
            # 保持历史记录在合理范围内
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-500:]
    
    def record_resolution(self, event: AlertEvent):
        """记录告警解决"""
        with self._lock:
            self.resolved_alerts += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        with self._lock:
            return {
                'total_alerts': self.total_alerts,
                'resolved_alerts': self.resolved_alerts,
                'active_alerts': self.total_alerts - self.resolved_alerts,
                'failed_notifications': self.failed_notifications,
                'alerts_by_level': {level.value: count for level, count in self.alerts_by_level.items()},
                'alerts_by_rule': dict(self.alerts_by_rule),
                'recent_alerts': self.alert_history[-10:] if self.alert_history else []
            }


class AlertManager:
    """告警管理器
    
    管理告警规则、评估指标和发送通知。
    """
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.notifiers: List[AlertNotifier] = []
        self.active_alerts: Dict[str, AlertEvent] = {}  # 活跃告警
        self.alert_cooldowns: Dict[str, float] = {}  # 告警冷却时间
        self.statistics = AlertStatistics()
        self._lock = threading.RLock()
        
        # 默认添加日志通知器
        self.add_notifier(LogNotifier())
        
        logger.info("Alert manager initialized")
    
    def add_rule(self, rule: AlertRule):
        """添加告警规则
        
        Args:
            rule: 告警规则
        """
        with self._lock:
            self.rules[rule.name] = rule
            logger.info(f"Added alert rule: {rule.name} ({rule.metric_name} {rule.condition.value} {rule.threshold})")
    
    def remove_rule(self, rule_name: str) -> bool:
        """移除告警规则
        
        Args:
            rule_name: 规则名称
            
        Returns:
            是否移除成功
        """
        with self._lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
                # 清理相关的活跃告警和冷却时间
                if rule_name in self.active_alerts:
                    del self.active_alerts[rule_name]
                if rule_name in self.alert_cooldowns:
                    del self.alert_cooldowns[rule_name]
                logger.info(f"Removed alert rule: {rule_name}")
                return True
            return False
    
    def add_notifier(self, notifier: AlertNotifier):
        """添加通知器
        
        Args:
            notifier: 通知器实例
        """
        with self._lock:
            self.notifiers.append(notifier)
            logger.info(f"Added notifier: {type(notifier).__name__}")
    
    def evaluate_metric(self, metric_name: str, value: Any, tags: Optional[Dict[str, str]] = None):
        """评估指标是否触发告警
        
        Args:
            metric_name: 指标名称
            value: 指标值
            tags: 指标标签
        """
        current_time = time.time()
        
        with self._lock:
            # 查找匹配的规则
            matching_rules = [rule for rule in self.rules.values() 
                            if rule.metric_name == metric_name and rule.enabled]
            
            for rule in matching_rules:
                # 检查冷却时间
                if rule.name in self.alert_cooldowns:
                    if current_time - self.alert_cooldowns[rule.name] < rule.cooldown_seconds:
                        continue
                
                # 评估条件
                if self._evaluate_condition(rule, value):
                    self._trigger_alert(rule, value, current_time, tags or {})
                else:
                    # 检查是否需要解决告警
                    if rule.name in self.active_alerts:
                        self._resolve_alert(rule.name, current_time)
    
    def _evaluate_condition(self, rule: AlertRule, value: Any) -> bool:
        """评估告警条件
        
        Args:
            rule: 告警规则
            value: 当前值
            
        Returns:
            是否满足告警条件
        """
        try:
            if rule.condition == AlertCondition.GREATER_THAN:
                return float(value) > float(rule.threshold)
            elif rule.condition == AlertCondition.LESS_THAN:
                return float(value) < float(rule.threshold)
            elif rule.condition == AlertCondition.EQUAL:
                return value == rule.threshold
            elif rule.condition == AlertCondition.NOT_EQUAL:
                return value != rule.threshold
            elif rule.condition == AlertCondition.GREATER_EQUAL:
                return float(value) >= float(rule.threshold)
            elif rule.condition == AlertCondition.LESS_EQUAL:
                return float(value) <= float(rule.threshold)
            elif rule.condition == AlertCondition.CONTAINS:
                return str(rule.threshold) in str(value)
            elif rule.condition == AlertCondition.NOT_CONTAINS:
                return str(rule.threshold) not in str(value)
            else:
                logger.warning(f"Unknown alert condition: {rule.condition}")
                return False
        except (ValueError, TypeError) as e:
            logger.error(f"Error evaluating condition for rule {rule.name}: {str(e)}")
            return False
    
    def _trigger_alert(self, rule: AlertRule, value: Any, timestamp: float, tags: Dict[str, str]):
        """触发告警
        
        Args:
            rule: 告警规则
            value: 当前值
            timestamp: 时间戳
            tags: 标签
        """
        # 创建告警事件
        message = rule.description or f"Metric {rule.metric_name} {rule.condition.value} {rule.threshold}"
        
        event = AlertEvent(
            rule_name=rule.name,
            metric_name=rule.metric_name,
            level=rule.level,
            message=message,
            current_value=value,
            threshold=rule.threshold,
            timestamp=timestamp,
            tags={**rule.tags, **tags}
        )
        
        # 记录活跃告警
        self.active_alerts[rule.name] = event
        self.alert_cooldowns[rule.name] = timestamp
        
        # 发送通知
        notification_success = self._send_notifications(event)
        
        # 记录统计
        self.statistics.record_alert(event, notification_success)
        
        logger.warning(f"Alert triggered: {rule.name} - {message} (value: {value})")
    
    def _resolve_alert(self, rule_name: str, timestamp: float):
        """解决告警
        
        Args:
            rule_name: 规则名称
            timestamp: 时间戳
        """
        if rule_name in self.active_alerts:
            event = self.active_alerts[rule_name]
            event.resolved = True
            event.resolved_at = timestamp
            
            # 发送解决通知
            resolution_event = AlertEvent(
                rule_name=rule_name,
                metric_name=event.metric_name,
                level=AlertLevel.INFO,
                message=f"Alert resolved: {event.message}",
                current_value="resolved",
                threshold=event.threshold,
                timestamp=timestamp,
                tags=event.tags,
                resolved=True,
                resolved_at=timestamp
            )
            
            self._send_notifications(resolution_event)
            
            # 移除活跃告警
            del self.active_alerts[rule_name]
            
            # 记录统计
            self.statistics.record_resolution(event)
            
            logger.info(f"Alert resolved: {rule_name}")
    
    def _send_notifications(self, event: AlertEvent) -> bool:
        """发送通知
        
        Args:
            event: 告警事件
            
        Returns:
            是否所有通知都发送成功
        """
        if not self.notifiers:
            logger.warning("No notifiers configured")
            return False
        
        success_count = 0
        for notifier in self.notifiers:
            try:
                if notifier.send_alert(event):
                    success_count += 1
            except Exception as e:
                logger.error(f"Notifier {type(notifier).__name__} failed: {str(e)}")
        
        return success_count == len(self.notifiers)
    
    def get_active_alerts(self) -> List[AlertEvent]:
        """获取活跃告警"""
        with self._lock:
            return list(self.active_alerts.values())
    
    def get_rules(self) -> List[AlertRule]:
        """获取所有规则"""
        with self._lock:
            return list(self.rules.values())
    
    def enable_rule(self, rule_name: str) -> bool:
        """启用规则
        
        Args:
            rule_name: 规则名称
            
        Returns:
            是否操作成功
        """
        with self._lock:
            if rule_name in self.rules:
                self.rules[rule_name].enabled = True
                logger.info(f"Enabled alert rule: {rule_name}")
                return True
            return False
    
    def disable_rule(self, rule_name: str) -> bool:
        """禁用规则
        
        Args:
            rule_name: 规则名称
            
        Returns:
            是否操作成功
        """
        with self._lock:
            if rule_name in self.rules:
                self.rules[rule_name].enabled = False
                # 解决相关的活跃告警
                if rule_name in self.active_alerts:
                    self._resolve_alert(rule_name, time.time())
                logger.info(f"Disabled alert rule: {rule_name}")
                return True
            return False
    
    def force_resolve_alert(self, rule_name: str) -> bool:
        """强制解决告警
        
        Args:
            rule_name: 规则名称
            
        Returns:
            是否操作成功
        """
        with self._lock:
            if rule_name in self.active_alerts:
                self._resolve_alert(rule_name, time.time())
                logger.info(f"Forcibly resolved alert: {rule_name}")
                return True
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats = self.statistics.get_summary()
            stats.update({
                'total_rules': len(self.rules),
                'enabled_rules': sum(1 for rule in self.rules.values() if rule.enabled),
                'disabled_rules': sum(1 for rule in self.rules.values() if not rule.enabled),
                'total_notifiers': len(self.notifiers),
                'active_alerts_count': len(self.active_alerts)
            })
            return stats
    
    def clear_statistics(self):
        """清空统计信息"""
        with self._lock:
            self.statistics = AlertStatistics()
            logger.info("Alert statistics cleared")


# 预定义的常用告警规则创建函数
def create_error_rate_rule(name: str, threshold: float = 0.05, level: AlertLevel = AlertLevel.WARNING) -> AlertRule:
    """创建错误率告警规则"""
    return AlertRule(
        name=name,
        metric_name="error_rate",
        condition=AlertCondition.GREATER_THAN,
        threshold=threshold,
        level=level,
        description=f"Error rate exceeds {threshold*100}%"
    )


def create_response_time_rule(name: str, threshold: float = 1.0, level: AlertLevel = AlertLevel.WARNING) -> AlertRule:
    """创建响应时间告警规则"""
    return AlertRule(
        name=name,
        metric_name="response_time",
        condition=AlertCondition.GREATER_THAN,
        threshold=threshold,
        level=level,
        description=f"Response time exceeds {threshold}s"
    )


def create_throughput_rule(name: str, threshold: float = 10.0, level: AlertLevel = AlertLevel.WARNING) -> AlertRule:
    """创建吞吐量告警规则"""
    return AlertRule(
        name=name,
        metric_name="throughput",
        condition=AlertCondition.LESS_THAN,
        threshold=threshold,
        level=level,
        description=f"Throughput below {threshold} requests/sec"
    )


# 全局告警管理器实例
global_alert_manager = AlertManager()