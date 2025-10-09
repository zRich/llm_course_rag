# Lesson 15 提交前检查清单（教师/学生）

## 文档与结构
- [ ] 课时文档齐备：`README.md`、`teacher-script.md`、`blackboard-steps.md`、`checklist.md`、`examples/`、`templates/`。
- [ ] README 链接到上述文件与目录，形成自足导航。

## 实现与一致性
- [ ] `process_documents_batch` 已实现检查点与恢复：支持 `batch_id`、分片 `shard`。
- [ ] 幂等键设计统一：`source + normalized_id + content_hash + layer`；避免自增 ID。
- [ ] 重试与退避：最大重试次数、指数退避与抖动，失败降级生效。
- [ ] 错误结构统一：`{ error: { code/message/details/hints } }`。
- [ ] 字段命名与指标口径在文档与代码保持一致。

## 指标与实验
- [ ] 首次运行与恢复运行的日志与统计截图齐备。
- [ ] 指标达标：完成率 ≥ 99%，重复率 ≤ 1%，恢复时间 ≤ 1 分钟。
- [ ] 吞吐与延迟：提供 p50/p95；失败原因分布与 `last_error` 可追溯。

## 提交物与报告
- [ ] 填写 `templates/submission_template.md`：实现概述、配置参数、实验数据与截图。
- [ ] 附上关键日志片段（进度、重试、检查点写入与恢复时间）。
- [ ] 明确边界声明：未覆盖分布式队列/强一致事务的深度实现。