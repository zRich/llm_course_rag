# Lesson 18 提交前检查清单：文本清洗与去噪

## 文档与结构
- README 导航完善；包含 teacher-script、blackboard-steps、checklist、examples、templates。

## 实现与质量
- 规范化：换行/空白/Unicode 规则正确且可配置。
- 去噪：常见噪声规则命中率高，误删率低。
- 语言检测：白名单与降级策略清晰。
- 去重：重复段落与近重复样本移除有效。

## 指标与实验
- 噪声移除率 ≥ 80%（示例集），内容保留率 ≥ 95%。
- 编码异常处理覆盖率与降级成功率。
- 过度清洗实验与回滚证明。

## 提交物与报告
- `examples/noisy_samples.txt`、`examples/cleaning_config.json`。
- 前后对比文本、日志片段、指标汇总与截图。
- `templates/submission_template.md` 完整填写。