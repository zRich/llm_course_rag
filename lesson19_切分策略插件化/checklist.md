# Lesson 19 提交前检查清单：切分策略插件化

## 文档与结构
- README 导航完整；包含 teacher-script、blackboard-steps、checklist、examples、templates。

## 接口与实现
- 策略接口统一：方法与输出字段一致。
- 注册/发现稳定：未注册策略返回标准错误，支持降级。
- 参数校验：越界有提示与回退策略。

## 指标与实验
- 至少两种策略对比：平均片长、重叠率、边界命中率、耗时。
- 日志片段与截图归档；结果可复现。

## 提交物与报告
- `examples/raw_text.txt`、`examples/plugin_config.json`。
- 指标表与对比结论；模板填写完整。