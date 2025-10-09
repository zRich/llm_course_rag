# Lesson 16 提交前检查清单（教师/学生）

## 文档与结构
- [ ] 课时文档齐备：`README.md`、`teacher-script.md`、`blackboard-steps.md`、`checklist.md`、`examples/`、`templates/`。
- [ ] README 链接到上述文件与目录，形成自足导航。

## 实现与一致性
- [ ] 增量更新入口与差分检测（upsert/update/delete/tombstone）已实现。
- [ ] 版本化幂等键统一：`source + normalized_id + content_hash + layer + version`。
- [ ] 失效与重建顺序正确：先失效缓存，再更新索引，最后重建缓存。
- [ ] 错误结构统一：`{ error: { code/message/details/hints } }`。
- [ ] 字段命名与指标口径在文档与代码保持一致。

## 指标与实验
- [ ] 两次运行（含故障注入）日志与统计截图齐备。
- [ ] 指标达标：更新正确率 ≥ 99.5%，重复率 ≤ 0.5%，失效命中率 ≥ 99%。
- [ ] 陈旧窗口 ≤ 5 分钟；吞吐与延迟提供 p50/p95；失败原因分布与 `last_error` 可追溯。

## 提交物与报告
- [ ] 填写 `templates/submission_template.md`：实现概述、配置参数、实验数据与截图。
- [ ] 附上关键日志片段（差分统计、失效事件、局部重建与陈旧窗口计算）。
- [ ] 明确边界声明：未覆盖跨进程强一致/事务与索引压缩细节（交由 Lesson 17）。