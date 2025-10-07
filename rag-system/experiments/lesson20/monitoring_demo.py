#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控告警演示
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.monitoring import (
    global_system_monitor, global_alert_manager, global_metrics_collector,
    create_error_rate_rule, create_response_time_rule,
    AlertLevel, AlertCondition, AlertRule
)
import time
import random

def demo_metrics_collection():
    """演示指标收集"""
    print("=== 指标收集演示 ===")
    
    collector = global_metrics_collector
    
    # 模拟业务指标
    print("📊 开始收集指标...")
    
    for i in range(10):
        # 模拟请求处理
        response_time = random.uniform(0.1, 2.0)
        success = random.random() > 0.1  # 90%成功率
        
        # 记录指标
        collector.record_counter("requests_total", 1, {"status": "success" if success else "error"})
        collector.record_timer("response_time", response_time, {"endpoint": "/api/search"})
        collector.record_gauge("active_connections", random.randint(10, 50))
        
        if success:
            print(f"✅ 请求 {i+1}: 成功, 响应时间: {response_time:.2f}s")
        else:
            print(f"❌ 请求 {i+1}: 失败, 响应时间: {response_time:.2f}s")
        
        time.sleep(0.2)
    
    # 显示指标摘要
    print("\n📈 指标摘要:")
    metrics = collector.get_metrics_summary()
    for name, data in metrics.items():
        if 'count' in data:
            print(f"   {name}: {data['count']} 次")
        elif 'value' in data:
            print(f"   {name}: {data['value']:.2f}")

def demo_alert_system():
    """演示告警系统"""
    print("\n=== 告警系统演示 ===")
    
    alert_manager = global_alert_manager
    
    # 添加告警规则
    error_rule = create_error_rate_rule(
        "high_error_rate",
        threshold=0.15,  # 15%错误率
        level=AlertLevel.WARNING
    )
    
    response_rule = create_response_time_rule(
        "slow_response",
        threshold=1.5,  # 1.5秒响应时间
        level=AlertLevel.ERROR
    )
    
    # 自定义规则：活跃连接数过高
    connection_rule = AlertRule(
        name="high_connections",
        metric_name="active_connections",
        condition=AlertCondition.GREATER_THAN,
        threshold=40,
        level=AlertLevel.WARNING,
        description="活跃连接数过高"
    )
    
    alert_manager.add_rule(error_rule)
    alert_manager.add_rule(response_rule)
    alert_manager.add_rule(connection_rule)
    
    print("🚨 开始监控告警...")
    
    # 模拟指标变化触发告警
    for i in range(15):
        # 逐渐增加错误率
        error_rate = 0.05 + (i * 0.02)
        
        # 随机响应时间，偶尔很慢
        response_time = random.uniform(0.5, 3.0) if random.random() > 0.7 else random.uniform(0.1, 1.0)
        
        # 活跃连接数
        connections = random.randint(20, 60)
        
        # 评估指标
        alert_manager.evaluate_metric("error_rate", error_rate)
        alert_manager.evaluate_metric("response_time", response_time)
        alert_manager.evaluate_metric("active_connections", connections)
        
        print(f"📊 周期 {i+1}: 错误率={error_rate:.2f}, 响应时间={response_time:.2f}s, 连接数={connections}")
        
        # 显示活跃告警
        active_alerts = alert_manager.get_active_alerts()
        if active_alerts:
            for alert in active_alerts:
                print(f"   🚨 告警: {alert.rule_name} - {alert.message}")
        
        time.sleep(0.5)
    
    # 显示告警统计
    print("\n📊 告警统计:")
    stats = alert_manager.get_statistics()
    print(f"   总告警数: {stats['total_alerts']}")
    print(f"   活跃告警: {stats['active_alerts']}")
    print(f"   已解决: {stats['resolved_alerts']}")

def demo_system_health():
    """演示系统健康检查"""
    print("\n=== 系统健康检查演示 ===")
    
    monitor = global_system_monitor
    
    # 运行健康检查
    print("🏥 执行健康检查...")
    results = monitor.run_all_health_checks()
    
    for result in results:
        status_icon = "✅" if result.status.value == "healthy" else "❌"
        print(f"{status_icon} {result.name}: {result.message} ({result.duration_ms:.1f}ms)")
    
    # 获取系统指标
    print("\n💻 系统指标:")
    try:
        metrics = monitor.get_system_metrics()
        print(f"   CPU使用率: {metrics.cpu_percent:.1f}%")
        print(f"   内存使用率: {metrics.memory_percent:.1f}%")
        print(f"   可用内存: {metrics.memory_available_mb:.0f}MB")
        print(f"   磁盘使用率: {metrics.disk_usage_percent:.1f}%")
        print(f"   进程数: {metrics.process_count}")
        if metrics.load_average:
            print(f"   负载平均值: {metrics.load_average}")
    except Exception as e:
        print(f"   ❌ 获取系统指标失败: {str(e)}")
    
    # 获取整体健康状态
    overall_health = monitor.get_overall_health()
    health_icon = {
        "healthy": "💚",
        "warning": "💛", 
        "unhealthy": "❤️",
        "unknown": "🤍"
    }.get(overall_health.value, "🤍")
    
    print(f"\n{health_icon} 系统整体健康状态: {overall_health.value.upper()}")

if __name__ == "__main__":
    demo_metrics_collection()
    demo_alert_system()
    demo_system_health()
    
    print("\n🎯 实验三完成！请观察监控告警系统的运行情况。")