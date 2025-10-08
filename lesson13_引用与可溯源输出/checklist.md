# 提交前检查清单（Lesson 13）— 引用与可溯源输出

## 课次合规
- [ ] 三要素齐备：教学目标（SMART）/重点/难点。
- [ ] 教学文档最小集合已创建：`README.md`、`teacher-script.md`、`blackboard-steps.md`、`checklist.md`、`examples/`、`templates/`。
- [ ] README 已链接最小集合文件与目录。

## 示例与一致性
- [ ] 至少 1 条输入示例与 1 条错误响应示例（带引用场景）。
- [ ] 字段命名一致：
  - 检索：`chunk_id/document_id/chunk_index/content/score/start_position/end_position/metadata`；
  - 问答来源：`sources[].document_id/.document_filename/.chunk_id/.score`；
  - 错误结构：`{ error: { code/message/details/hints } }`。
- [ ] 引用样式一致：行内 [1]/[2] 与 legend 映射统一。

## 教师可执教
- [ ] 时间轴明确：导入/讲解/实践/总结。
- [ ] 讲稿含话术与演示指令；黑板步骤可直接执行。
- [ ] 提问与练习（含参考答案）已提供或链接。

## 学生任务与验收
- [ ] 学生完成 ≥3 条问题的引用输出与一致性验证。
- [ ] 指标表齐全：覆盖率、一致性、重复引用去重与排序合理性。
- [ ] 提供错误路径与兜底策略（缺失引用、引用不可定位、命名不一致）。
- [ ] 不可虚构声明：日志与截图真实可复核。

## 连续性与边界
- [ ] 承接 Lesson 12（来源统一与元数据归一化），引出 Lesson 14（缓存策略与键设计）。
- [ ] 边界声明：不覆盖复杂跨模态引用与自动纠错；不更换模型。