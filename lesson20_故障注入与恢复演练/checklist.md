# Lesson 20 提交前检查清单：故障注入与恢复演练

## 文档与结构
- README 导航完善；包含 teacher-script、blackboard-steps、checklist、examples、templates。

## 覆盖与实现
- 覆盖至少三类故障与一组组合故障。
- 恢复机制生效：退避重试、熔断、降级、限流。
- 错误结构统一；阈值与告警配置清晰。

## 指标与实验
- 错误率与 MTTR 达标；重试成功率提升明显。
- 熔断触发次数与降级使用率记录。
- 日志与截图可复现。

## 提交物与报告
- `examples/fault_plan.json` 与运行日志片段。
- 指标面板截图与阈值配置说明。
- `templates/submission_template.md` 完成度。