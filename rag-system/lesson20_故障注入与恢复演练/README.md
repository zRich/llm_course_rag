# Lesson 20: 故障注入与恢复演练

## 课程概览

本课程是《LLM实战》课程第20节，专注于在RAG系统中实现故障注入和恢复机制，提升系统的可靠性和容错能力。

### 🎯 学习目标

- 理解故障注入技术的原理和应用场景
- 掌握系统恢复机制的设计和实现
- 学会构建监控告警系统
- 在RAG系统中应用容错设计模式

### ⏰ 课程时长

45分钟（理论讲解 + 实践演练）

### 📚 课程内容

1. **故障注入基础** (10分钟)
   - 故障类型和注入策略
   - 装饰器模式应用
   - 多种故障场景演示

2. **恢复机制实践** (15分钟)
   - 重试机制和退避策略
   - 熔断器模式
   - 降级服务设计

3. **监控告警系统** (10分钟)
   - 指标收集和分析
   - 告警规则配置
   - 系统健康检查

4. **RAG系统集成** (10分钟)
   - 容错机制集成
   - 性能监控
   - 最佳实践总结

## 🚀 快速开始

### 环境准备

```bash
# 1. 切换到lesson20分支
git checkout lesson20

# 2. 安装依赖
pip install psutil requests

# 3. 验证环境
python -c "from src.fault_injection import FaultInjector; print('✅ 故障注入模块正常')"
python -c "from src.recovery import CircuitBreaker; print('✅ 恢复机制模块正常')"
python -c "from src.monitoring import global_system_monitor; print('✅ 监控模块正常')"
```

### 实验运行

```bash
# 进入实验目录
cd experiments/lesson20

# 运行实验一：故障注入基础
python fault_injection_demo.py

# 运行实验二：恢复机制实践
python recovery_demo.py

# 运行实验三：监控告警系统
python monitoring_demo.py

# 运行实验四：RAG系统集成
python rag_integration_demo.py
```

## 📁 项目结构

```
rag-system/
├── src/
│   ├── fault_injection/          # 故障注入框架
│   │   ├── __init__.py
│   │   ├── fault_injector.py      # 核心故障注入器
│   │   └── fault_types.py         # 故障类型定义
│   ├── recovery/                  # 恢复机制框架
│   │   ├── __init__.py
│   │   ├── retry_mechanism.py     # 重试机制
│   │   ├── circuit_breaker.py     # 熔断器
│   │   └── fallback_service.py    # 降级服务
│   └── monitoring/                # 监控告警系统
│       ├── __init__.py
│       ├── metrics_collector.py   # 指标收集器
│       ├── alert_manager.py       # 告警管理器
│       └── system_monitor.py      # 系统监控器
├── experiments/lesson20/          # 实验代码
│   ├── fault_injection_demo.py
│   ├── recovery_demo.py
│   ├── monitoring_demo.py
│   └── rag_integration_demo.py
└── lesson20_故障注入与恢复演练/
    ├── 教师讲义.md
    ├── 学生实验指导.md
    └── README.md
```

## 🔧 核心组件

### 故障注入框架

- **FaultInjector**: 核心故障注入器，支持多种故障类型
- **FaultType**: 故障类型枚举（网络超时、服务不可用、数据损坏等）
- **装饰器支持**: 简化方法级故障注入

### 恢复机制框架

- **RetryMechanism**: 智能重试机制，支持多种退避策略
- **CircuitBreaker**: 熔断器模式，防止级联故障
- **FallbackService**: 降级服务，提供备用方案

### 监控告警系统

- **MetricsCollector**: 指标收集器，支持多种指标类型
- **AlertManager**: 告警管理器，支持规则配置和多渠道通知
- **SystemMonitor**: 系统监控器，提供健康检查和资源监控

## 📊 实验效果预览

### 故障注入演示
```
=== 基础故障注入演示 ===
✅ 调用 1: 成功 - {'status': 'success', 'data': 'response_42'}
❌ 调用 2: 失败 - 网络超时: 请求超时
✅ 调用 3: 成功 - {'status': 'success', 'data': 'response_17'}
📊 统计结果: 成功 7 次, 失败 3 次
📈 故障率: 30.0%
```

### 熔断器演示
```
=== 熔断器模式演示 ===
❌ 调用 1: 服务错误 (请求 1) (状态: CLOSED)
❌ 调用 2: 服务错误 (请求 2) (状态: CLOSED)
❌ 调用 3: 服务错误 (请求 3) (状态: OPEN)
❌ 调用 4: 熔断器开启，拒绝调用 (状态: OPEN)
```

### 监控告警演示
```
=== 告警系统演示 ===
📊 周期 1: 错误率=0.07, 响应时间=0.85s, 连接数=35
📊 周期 2: 错误率=0.09, 响应时间=1.20s, 连接数=42
   🚨 告警: high_connections - 活跃连接数过高: 42 > 40
📊 周期 3: 错误率=0.11, 响应时间=2.10s, 连接数=38
   🚨 告警: slow_response - 响应时间过慢: 2.10s > 1.5s
```

## 🎓 学习成果

完成本课程后，学生将能够：

1. **设计容错系统**：理解并应用故障注入和恢复机制
2. **实现监控体系**：构建完整的监控告警系统
3. **优化系统可靠性**：在实际项目中应用容错设计模式
4. **处理生产问题**：具备分析和解决系统故障的能力

## 🔗 相关资源

### 延伸阅读
- [Chaos Engineering原理](https://principlesofchaos.org/)
- [微服务容错模式](https://microservices.io/patterns/reliability/)
- [Site Reliability Engineering](https://sre.google/)

### 工具推荐
- [Chaos Monkey](https://netflix.github.io/chaosmonkey/) - Netflix开源的故障注入工具
- [Hystrix](https://github.com/Netflix/Hystrix) - 熔断器库
- [Prometheus](https://prometheus.io/) - 监控系统

## ⚠️ 注意事项

1. **安全使用**：故障注入仅在测试环境使用，严禁在生产环境直接应用
2. **参数调优**：恢复机制参数需根据实际业务场景调整
3. **监控平衡**：避免过度监控导致系统负担
4. **渐进部署**：容错机制应该渐进式部署和验证

## 🤝 贡献指南

如果您发现问题或有改进建议，请：

1. 提交Issue描述问题
2. Fork项目并创建功能分支
3. 提交Pull Request

## 📄 许可证

本课程内容遵循MIT许可证。

---

**课程作者**: LLM实战课程组  
**更新时间**: 2024年1月  
**版本**: v1.0

🎯 **开始学习**: 请先阅读[教师讲义.md](./教师讲义.md)，然后按照[学生实验指导.md](./学生实验指导.md)进行实践！