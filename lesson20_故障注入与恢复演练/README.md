# Lesson 20 - 故障注入与恢复演练

## 📖 课程概览

本节课是RAG系统工程化系列的第20课，专注于提升系统的可靠性和容错能力。通过故障注入技术和自动恢复机制，学生将学会构建更加健壮的RAG系统。

### 🎯 学习目标

- 理解故障注入在系统测试中的重要作用
- 掌握故障注入框架的设计和实现
- 学会构建自动恢复机制（重试、熔断器）
- 建立完整的监控告警体系
- 进行系统弹性测试和故障演练

### 📚 课程内容

1. **故障注入框架**
   - 故障类型定义和分类
   - 可配置的故障注入器
   - 故障统计和分析

2. **自动恢复机制**
   - 指数退避重试策略
   - 熔断器模式实现
   - 降级服务设计

3. **监控告警系统**
   - 关键指标收集
   - 实时告警机制
   - 监控面板展示

4. **弹性测试演练**
   - 综合故障场景测试
   - 系统恢复能力验证
   - 性能影响分析

## 🚀 快速开始

### 环境准备

1. **确保已完成前19节课的内容**
2. **安装必要依赖**：
   ```bash
   pip install asyncio aiohttp tenacity prometheus-client
   ```

### 目录结构

```
lesson20_故障注入与恢复演练/
├── README.md                 # 本文件
├── 教师讲义.md               # 详细的教学内容
├── 学生实验指导.md           # 实验步骤和代码示例
└── code_examples/            # 代码示例目录
    ├── fault_injection/      # 故障注入模块
    ├── recovery/            # 恢复机制模块
    ├── monitoring/          # 监控模块
    └── tests/              # 测试脚本

## 导航与配套材料（与最佳实践对齐）
- [教师讲稿](./teacher-script.md)
- [黑板/投屏操作步骤](./blackboard-steps.md)
- [提交前检查清单](./checklist.md)
- 示例与配置：`examples/` → [fault_plan.json](./examples/fault_plan.json), [sample_logs.txt](./examples/sample_logs.txt)
- 学生提交模板：`templates/` → [submission_template.md](./templates/submission_template.md)
```

### 核心组件

#### 1. 故障注入器 (FaultInjector)

```python
from fault_injection.fault_injector import FaultInjector, FaultType

# 创建故障注入器
injector = FaultInjector(failure_rate=0.1, enabled=True)

# 注入故障
injector.inject_fault(FaultType.SERVICE_UNAVAILABLE, "search_operation")
```

#### 2. 重试机制 (RetryMechanism)

```python
from recovery.retry_mechanism import retry_with_backoff, RetryConfig

# 配置重试策略
config = RetryConfig(max_attempts=3, base_delay=1.0)

@retry_with_backoff(config)
def unreliable_operation():
    # 可能失败的操作
    pass
```

#### 3. 熔断器 (CircuitBreaker)

```python
from recovery.retry_mechanism import CircuitBreaker

# 创建熔断器
cb = CircuitBreaker(failure_threshold=5, timeout=60.0)

# 通过熔断器调用函数
result = cb.call(risky_function, arg1, arg2)
```

#### 4. 监控系统 (SystemMonitor)

```python
from monitoring.metrics_collector import SystemMonitor

# 创建监控器
monitor = SystemMonitor()

# 记录请求指标
monitor.record_request(success=True, duration=1.5, operation="query")

# 获取监控面板数据
dashboard = monitor.get_dashboard_data()
```

## 🧪 实验流程

### 步骤1：故障注入测试 (10分钟)

1. 实现故障注入器类
2. 测试不同类型的故障注入
3. 观察故障对系统的影响

### 步骤2：恢复机制实现 (15分钟)

1. 实现重试装饰器
2. 构建熔断器组件
3. 集成到RAG系统中

### 步骤3：监控告警系统 (10分钟)

1. 实现指标收集器
2. 建立告警机制
3. 创建监控面板

### 步骤4：综合测试演练 (10分钟)

1. 运行弹性测试脚本
2. 分析测试结果
3. 优化系统配置

## 📊 关键指标

### 系统可靠性指标

- **成功率 (Success Rate)**：成功请求占总请求的比例
- **错误率 (Error Rate)**：失败请求占总请求的比例
- **平均响应时间 (Average Response Time)**：请求处理的平均耗时
- **恢复时间 (Recovery Time)**：从故障到恢复的时间

### 故障注入指标

- **故障注入率 (Fault Injection Rate)**：故障注入的频率
- **故障类型分布 (Fault Type Distribution)**：不同故障类型的占比
- **故障影响范围 (Fault Impact Scope)**：故障影响的操作数量

### 恢复机制指标

- **重试成功率 (Retry Success Rate)**：重试操作的成功比例
- **熔断器触发次数 (Circuit Breaker Trips)**：熔断器开启的次数
- **降级服务使用率 (Fallback Usage Rate)**：降级服务的使用频率

## 🔧 配置参数

### 故障注入配置

```python
# 故障注入器配置
FAULT_INJECTION_CONFIG = {
    "failure_rate": 0.1,        # 故障注入率 (10%)
    "enabled": True,            # 是否启用故障注入
    "fault_types": [            # 启用的故障类型
        "network_timeout",
        "service_unavailable",
        "data_corruption"
    ]
}
```

### 重试机制配置

```python
# 重试配置
RETRY_CONFIG = {
    "max_attempts": 3,          # 最大重试次数
    "base_delay": 1.0,          # 基础延迟时间 (秒)
    "max_delay": 60.0,          # 最大延迟时间 (秒)
    "exponential_base": 2       # 指数退避基数
}
```

### 熔断器配置

```python
# 熔断器配置
CIRCUIT_BREAKER_CONFIG = {
    "failure_threshold": 5,     # 失败阈值
    "timeout": 60.0,           # 超时时间 (秒)
    "half_open_max_calls": 3   # 半开状态最大调用次数
}
```

### 监控告警配置

```python
# 告警阈值配置
ALERT_THRESHOLDS = {
    "error_rate": 0.1,          # 错误率阈值 (10%)
    "response_time": 5.0,       # 响应时间阈值 (5秒)
    "circuit_breaker_open": 1   # 熔断器开启告警
}
```

## 🎓 学习成果

完成本节课后，学生将能够：

1. **设计故障注入策略**：根据系统特点设计合适的故障注入方案
2. **实现自动恢复机制**：构建重试、熔断器等容错组件
3. **建立监控体系**：设计关键指标和告警规则
4. **进行弹性测试**：验证系统在故障情况下的表现
5. **优化系统可靠性**：基于测试结果改进系统设计

## 🔍 常见问题

### Q1: 故障注入会影响生产环境吗？

**A**: 本课程的故障注入仅用于测试环境。在生产环境中进行故障注入需要更加谨慎的设计和控制机制。

### Q2: 如何确定合适的重试次数？

**A**: 重试次数应该根据故障类型和业务需求来确定。一般建议：
- 网络故障：3-5次重试
- 服务暂时不可用：2-3次重试
- 数据错误：不建议重试

### Q3: 熔断器什么时候会开启？

**A**: 当连续失败次数达到设定阈值时，熔断器会开启。开启后会阻止后续请求，直到超时时间后尝试恢复。

### Q4: 如何选择监控指标？

**A**: 建议选择与业务目标直接相关的指标：
- 用户体验相关：响应时间、成功率
- 系统健康相关：错误率、资源使用率
- 业务指标相关：查询量、转化率

## 📚 参考资料

1. **《混沌工程》** - Casey Rosenthal & Nora Jones
2. **《Site Reliability Engineering》** - Google SRE Team
3. **《Building Microservices》** - Sam Newman
4. **Netflix Chaos Engineering** - 故障注入最佳实践
5. **Hystrix Documentation** - 熔断器模式参考

## 🤝 贡献

如果你发现任何问题或有改进建议，欢迎：

1. 提交Issue报告问题
2. 提交Pull Request改进代码
3. 分享你的实验心得和经验

---

**课程信息**
- 课程编号：Lesson 20
- 课程时长：45分钟
- 难度等级：中级
- 前置课程：Lesson 01-19
- 后续课程：Lesson 21 (生成控制与防幻觉)

**最后更新**：2024年1月
**版本**：v1.0