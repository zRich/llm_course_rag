# 提交前检查清单（Lesson 11）

## 课次合规
- [ ] 课次三要素齐备：教学目标（SMART）/重点/难点。
- [ ] 教学文档与最小集合已创建：`README.md`、`teacher-script.md`、`blackboard-steps.md`、`checklist.md`、`examples/`、`templates/`。
- [ ] README 链接到最小集合文件与目录。

## 示例与一致性
- [ ] 提供至少 1 条输入示例与 1 条错误响应示例。
- [ ] 字段命名一致：`chunk_id/document_id/chunk_index/content/score/start_position/end_position/metadata`。
- [ ] 错误结构统一：`{ error: { code/message/details/hints } }`。

## 教师可执教
- [ ] 时间轴明确：导入/讲解/实践/总结。
- [ ] 讲稿包含话术与演示指令；黑板步骤可直接执行。
- [ ] 课堂提问与练习（含参考答案）已提供。

## 学生任务与验收
- [ ] 学生需完成 ≥3 组配置（`CHUNK_SIZE/CHUNK_OVERLAP`）并提交报告。
- [ ] 指标表齐全：分块数、平均 token/char、`total_found`、来源命中率。
- [ ] 提供错误路径与兜底策略示例。
- [ ] 不可虚构声明：日志与截图真实可复核。

## 连续性与边界
- [ ] 承接 Lesson 10（有/无重排的对比）并引出 Lesson 14（缓存策略）。
- [ ] 边界声明：不覆盖嵌入模型更换与高级语言学分块。