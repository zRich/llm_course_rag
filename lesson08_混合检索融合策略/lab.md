# 课后实验任务 — Lesson 08 混合检索融合策略

## 任务概述
实现一套可复用的检索融合模块，支持 RRF 与线性加权两种方法，完成去重、排序与参数化，并撰写至少 2 项单元测试。

## 任务要求
1. 抽象融合函数 `fuse_results(keyword_results, vector_results, method, **params)`：
   - `method` 支持 `rrf` 与 `weighted`；
   - 参数包括 `k/w_k/w_v/top_k` 等；
   - 输出字段：`id/title/source/fused_score/method`。
2. 归一化与去重：
   - 对分数做 min-max 归一化（或说明其他口径）；
   - 以 `id` 或 `source+offset` 为去重键。
3. 单元测试（至少 2 项）：
   - 空输入与单路为空；
   - 重复 `id` 的去重与排序。
4. 提交物：
   - 按 `templates/lab_submission_template.md` 填写；
   - 包含运行指令、参数说明与输出片段；
   - 对比 `rrf` 与 `weighted` 的差异与适用场景。

## 评分建议（供教师参考）
- 功能正确性（40%）：方法切换、参数化、去重与排序。
- 规范一致性（20%）：字段命名、归一化口径与提交模板完整性。
- 测试与复现（20%）：单元测试覆盖与可重复运行。
- 分析与结论（20%）：对比分析与改进建议的清晰度。

## 参考与提示
- 可直接复用 `examples/fusion_demo.py` 的数据结构与逻辑，抽象为库函数。
- 若对真实数据操作受限，可先用模拟结果完成融合与测试，再替换为真实来源。