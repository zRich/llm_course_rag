# 提交前检查清单（教师/学生）— Lesson 08 混合检索融合策略

## 功能
- [ ] 支持 `rrf` 与 `weighted` 两种融合方法可切换。
- [ ] 正确进行分数归一化（min-max 或其他说明）。
- [ ] 完成 Top-K 合并、排序与去重（以 `id` 或组合键）。

## 性能
- [ ] Top-10 响应时间 ≤ 200ms（本地示例脚本，单次运行）。
- [ ] 融合集合大小可配置（如 `top_k` 参数）。

## 一致性
- [ ] 输出字段命名统一：`id/title/score/source/rank/fused_score/method`。
- [ ] 空路兜底与 `warnings` 字段存在且可读。

## 测试与提交
- [ ] 至少 2 个单元测试：空输入、重复 `id` 去重。
- [ ] 提交物包含：脚本运行参数、输出片段、对比分析。
- [ ] 按 `templates/lab_submission_template.md` 填写并放置在指定目录。