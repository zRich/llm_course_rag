# Lesson 08：混合检索融合策略（Keyword + Vector Fusion）

## 课次信息
- 时长：45/90 分钟
- 前置：Lesson 07（已实现关键词检索与中文分词；Postgres全文检索配置与索引完成）
- 后置：Lesson 09（评估与调优；将复用本课的融合策略与指标）

## 教学目标（SMART）
- S：能在检索场景中实现关键词检索与向量检索的融合（RRF/线性加权），并对 Top-K 结果做去重与排序。
- M：Top-10 命中率提升≥10%；响应时间≤200ms；融合函数覆盖单元测试≥2项。
- A：在提供的示例与模板基础上，独立完成融合实现并跑通演示脚本。
- R：与模块目标一致，服务课程主线的检索效果提升与工程可用性。
- T：45/90 分钟内完成并通过验收清单。

## 教学重点
- 检索融合的动机与收益：关键词（召回与精确度）+ 向量（语义泛化）。
- 常用融合方法：RRF（Reciprocal Rank Fusion）、线性加权（score normalization + weighted sum）。
- 结果规范：去重、排序、一致性字段（`id/title/score/source`）。

## 教学难点
- 分数归一化与不同来源的分值尺度对齐。
- 冲突条目（同文档不同片段）与去重策略（`id` 或 `source+offset`）。
- 性能与效果权衡（检索并发、缓存、Top-K 合并成本）。

## 知识点详解
1) RRF 融合：`rrf_score = Σ (1 / (k + rank_i))`，k 通常取 60-120 之间，避免尾部过度放大。
2) 线性加权：对关键词分数与向量分数做归一化（min-max 或 z-score），再按权重合成。
3) 归一化策略：推荐 min-max 到 [0,1]，并记录原始分数供调试；异常值处理与截断。
4) 去重策略：优先以唯一 `id` 去重；无唯一 id 时采用 `source+start_offset` 组合键。
5) 一致性口径：融合输出的字段、排序稳定性、可重复实验的输入与固定种子。

## 完整示例与演示
- 输入：两路检索结果（`keyword_results.json`、`vector_results.json`），各包含 `id/title/score/source` 与 `rank`。
- 期望输出：融合后的 Top-K 列表，含 `fused_score`、`method`（`rrf`/`weighted`）。
- 异常与兜底：任一路为空时仅返回另一条路结果；分数缺失时使用 rank 兜底；记录 `warnings`。

## 授课老师指导
- 时间轴与脚本：导入10 + 讲解15 + 实践15 + 总结5 分钟（45’版本）。
- 提问与互动：为何需要归一化？RRF 相比线性加权的优势是什么？如何做去重？
- 练习安排：修改权重与 k 值观察指标变化；实现自定义去重键。

### 教师讲稿（照读链接）
- 见 `teacher-script.md`（含话术与现场演示指令）。

### 黑板/投屏操作步骤
- 见 `blackboard-steps.md`（逐步展示示例输入/输出与一致性检查）。

## 学生任务
- 课堂任务：在示例脚本中切换 `rrf` 与 `weighted` 模式并提交输出。
- 课后任务：将融合函数抽象到库模块，并增加 2 个极端用例的单元测试。

## 对齐与连续性
- 见 `alignment.md`（承接 Lesson 07、引出 Lesson 09 指标评估）。

## 提交与验收
- 见 `acceptance.md`（阈值、不可虚构声明、流程与一致性检查表）。

## 术语与概念定义
- 见 `terminology.md`（BM25、RRF、nDCG、MRR、min-max、去重键等）。

## 提交前检查清单
- 见 `checklist.md`（功能/性能/一致性/测试覆盖）。

## 边界声明
- 见 `boundaries.md`（不含重排序模型与学习排序；外部依赖说明与替代方案）。

## 导航
- 示例：[`examples/fusion_demo.py`](examples/fusion_demo.py)
- 模板：[`templates/lab_submission_template.md`](templates/lab_submission_template.md)
- 教师讲稿：[`teacher-script.md`](teacher-script.md)
- 黑板/投屏步骤：[`blackboard-steps.md`](blackboard-steps.md)
- 检查清单：[`checklist.md`](checklist.md)
- 术语与概念：[`terminology.md`](terminology.md)
- 提交与验收：[`acceptance.md`](acceptance.md)
- 对齐与连续性：[`alignment.md`](alignment.md)
- 边界声明：[`boundaries.md`](boundaries.md)
- 环境与依赖：[`SETUP.md`](SETUP.md)
- 课后实验：[`lab.md`](lab.md)