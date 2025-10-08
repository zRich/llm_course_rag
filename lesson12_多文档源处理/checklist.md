# 提交前检查清单（Lesson 12）— 多文档源处理

## 课次合规
- [ ] 三要素齐备：教学目标（SMART）/重点/难点。
- [ ] 教学文档最小集合已创建：`README.md`、`teacher-script.md`、`blackboard-steps.md`、`checklist.md`、`examples/`、`templates/`。
- [ ] README 已链接最小集合文件与目录。

## 示例与一致性
- [ ] 至少 1 条输入示例与 1 条错误响应示例（多源场景）。
- [ ] 字段命名一致：
  - 文档：`document_id/document_filename/source_type/metadata/status`；
  - 检索：`chunk_id/document_id/chunk_index/content/score/start_position/end_position/metadata`；
  - 问答来源：`sources[].document_id/.document_filename/.chunk_id/.score`。
- [ ] 错误结构统一：`{ error: { code/message/details/hints } }`。

## 教师可执教
- [ ] 时间轴明确：导入/讲解/实践/总结。
- [ ] 讲稿含话术与演示指令；黑板步骤可直接执行。
- [ ] 提问与练习（含参考答案）已提供或链接。

## 学生任务与验收
- [ ] 学生完成 ≥3 类来源接入并端到端验证（PDF/TXT/URL）。
- [ ] 指标表齐全：解析成功率、去重率、跨源召回覆盖、来源一致性。
- [ ] 提供错误路径与兜底策略（抓取失败、解析异常、重复文档）。
- [ ] 不可虚构声明：日志与截图真实可复核。

## 连续性与边界
- [ ] 承接 Lesson 11（分块策略影响覆盖与来源质量），引出 Lesson 14（缓存策略）。
- [ ] 边界声明：不覆盖高级正文抽取与跨模态解析，不更换嵌入模型。