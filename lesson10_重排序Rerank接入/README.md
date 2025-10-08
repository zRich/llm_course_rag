# Lesson 10：重排序（Rerank）接入

## 课次信息
- 时长：90 分钟
- 前置：Lesson 09（已完成元数据过滤，得到候选集）
- 后置：Lesson 11（在重排序结果上进行答案抽取与上下文拼接）

## 教学目标（SMART）
- S：能将 Rerank 接口集成到检索管道，对候选集进行重排序，并以端到端示例展示融合策略与稳定排序。
- M：在示例数据集上，MRR@10 或 nDCG@10 相比基线提升 ≥ 10%；带 Rerank 的查询延迟（100 条候选）≤ 500ms（使用启发式或本地小模型）。
- A：在提供的示例与模板下，独立完成运行与提交。
- R：对齐 RAG 主线（检索→融合→过滤→重排序→答案抽取）。
- T：90 分钟内完成演示、练习与验收。

## 教学重点
- Rerank 类型与接口：Cross-Encoder、Bi-Encoder 作为 reranker 的差异与选型。
- 分数融合与归一化：线性加权（`alpha`）与归一化（min-max）；稳定排序与并列打断（tie-breaking）。
- Top-K 重排序范围与性能取舍：只对前 K 做重排 vs 全量。
- 统一错误结构与命名一致性：`{ error: { code/message/details/hints } }`。

## 教学难点
- 依赖与性能：是否启用 `sentence-transformers`/`FlagEmbedding`；CPU/GPU 与网络下载的权衡。
- 多语言与文本截断：长文本分片、语言不一致导致分数尺度不一。
- 与基线分数融合的尺度对齐：不同检索器分数区间与归一化策略。

## 知识点详解
1) 接口与抽象
   - 统一 `Reranker.score(query, docs) -> List[float]`；默认启发式回退（无外部依赖）。
   - 可选启用 Cross-Encoder（命令参数 `--use_ce`），否则使用词重合启发式。
2) 分数融合与稳定排序
   - 归一化（min-max）；最终分数 `final = alpha * base + (1 - alpha) * rerank_norm`。
   - 稳定排序：分数并列按 `base_score` 与 `id` 打断；保证可重复性与可验收性。
3) Top-K 范围
   - 仅对前 K 候选进行重排序（降低延迟）；其余保留基线顺序。
4) 指标与评估
   - 示例实现 MRR 与 nDCG 的计算；对比基线与重排序效果。
5) 错误路径与兜底
   - 依赖不可用或模型加载失败时回退启发式；非法参数（`alpha`/`top_k`）返回统一错误结构。

## 完整示例与演示
- 输入：`examples/rerank_demo.py` 中的查询与候选集（含基线分数）。
- 期望输出：重排序后的 top-K 列表、分数融合结果、评估指标对比。
- 异常与兜底：
  - `alpha` 不在 [0,1]、`top_k` 非法 → 400；
  - 依赖加载异常 → 回退启发式并提示 `hints`。

## 授课老师指导
- 时间轴与脚本：
  - 导入 10min：目标与产物、reranker 类型与接口抽象。
  - 讲解 25min：分数融合、归一化、稳定排序与 Top-K 范围。
  - 实践 40min：运行示例脚本，切换 `alpha/top_k/--use_ce`，观察指标与延迟；注入错误参数并查看错误结构。
  - 总结 15min：提交物核查与边界声明，指向下一课的答案抽取。
- 细节与话术：见 `teacher-script.md`；黑板/投屏步骤见 `blackboard-steps.md`。

### 课堂提问与练习（含参考答案）
- 请见 `questions.md`，覆盖 Cross/BI 差异、融合权重与稳定排序语义。

### 常见误区与应对话术
- 直接用未归一化的分数融合 → 指出需要 min-max 或 z-score。
- 将 Bi-Encoder 相似度当作 Cross-Encoder 语义分数 → 澄清两者粒度与尺度差异。
- 忽略稳定排序导致提交不可复现 → 强制并列打断规则与随机性禁用。

### 黑板/投屏操作步骤（逐步）
1. 投屏接口与融合公式；2. 展示 `alpha/top_k` 的影响；3. 演示错误结构；4. 打开检查清单与提交模板现场填写要点。

## 学生任务
- 课堂任务：修改 `alpha/top_k` 并对比 MRR/nDCG；注入 2 条错误参数，截图错误结构。
- 课后任务：完成 `lab.md` 要求，提交模板与演示截图。

## 对齐与连续性
- 与 Module-RAG 主线对齐：候选生成与过滤 → 重排序 → 答案抽取。
- 承接 Lesson 09 的过滤结果，输出稳定重排序的候选供 Lesson 11 使用。

## 提交与验收
- 评估标准与阈值：见 `acceptance.md`；指标提升与稳定排序为必查项。

## 术语与概念定义
- 见 `terminology.md`，统一核心术语与评估指标。

## 提交前检查清单（教师/学生）
- 见 `checklist.md`，覆盖功能、性能、指标、一致性与错误路径。

## 边界声明
- 见 `boundaries.md`，明确不覆盖项与依赖回退策略。

## 环境与依赖
- 见 `SETUP.md`，包含可选依赖安装与运行指令。

## 课后实验
- 见 `lab.md`，形成教学闭环与评分建议。

## 导航
- 示例：[`examples/rerank_demo.py`](examples/rerank_demo.py)
- 模板：[`templates/lab_submission_template.md`](templates/lab_submission_template.md)
- 教师讲稿：[`teacher-script.md`](teacher-script.md)
- 黑板/投屏：[`blackboard-steps.md`](blackboard-steps.md)
- 检查清单：[`checklist.md`](checklist.md)
- 提问练习：[`questions.md`](questions.md)
- 术语与概念：[`terminology.md`](terminology.md)
- 提交与验收：[`acceptance.md`](acceptance.md)
- 对齐与连续性：[`alignment.md`](alignment.md)
- 边界声明：[`boundaries.md`](boundaries.md)
- 环境与依赖：[`SETUP.md`](SETUP.md)
- 课后实验：[`lab.md`](lab.md)