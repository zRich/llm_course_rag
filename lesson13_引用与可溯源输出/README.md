# Lesson 13：引用与可溯源输出（Citations & Traceable Answers）

## 课次信息
- 时长：90 分钟
- 前置：Lesson 12（多文档源统一解析与元数据归一化）
- 后置：Lesson 14（缓存策略与命中率优化，将复用本课的来源与引用结构）

## 教学目标（SMART）
- S：能在中文场景下输出“带引用且可溯源”的问答结果，保证引用指向真实来源片段并可复核。
- M：
  - 引用覆盖率 ≥ 80%（答案关键句均有来源引用）；
  - 引用一致性 ≥ 95%（`sources` 字段与原文片段一致且可定位）；
  - 错误结构统一（`{ error: { code/message/details/hints } }`），字段命名一致。
- A：在现有项目内独立完成 ≥3 条问题的端到端问答与引用验证。
- R：服务 RAG 主线，提升答案的可信度与复核性，支撑评审与合规。
- T：在 90 分钟内完成并提交模板化报告，日志与截图可复核。

## 教学重点
- 输出契约与引用格式：`answer` + `sources[]` 的命名与结构统一。
- 引用定位信息：`chunk_id/start_position/end_position/document_id/document_filename/metadata/score`。
- 引用样式：行内标注（如 [1]）、图例映射（legend）、可点击链接（如 URL 或文档锚点）。
- 一致性检查：答案语句与来源片段的严格对齐与可定位性。

## 教学难点
- 片段对齐与边界：按 `start_position/end_position` 或 `chunk_index` 精准定位。
- 多引用融合：同一句引用多来源时的排序与展示策略。
- 异常与兜底：缺失引用、引用指向不存在的片段、命名不一致。
- 与重排/融合的交互：排序变化导致引用选择差异。

## 知识点详解
1) 输出契约与字段一致性：
   - 问答响应必须包含：
     - `answer`: 最终答案文本；
     - `sources[]`: 引用数组，字段统一为：
       - `document_id/document_filename/chunk_id/chunk_index/content/score/start_position/end_position/metadata`。
   - 错误结构统一：`{ error: { code/message/details/hints } }`。
2) 引用样式与展示：
   - 行内标注：在答案中以 [1]/[2] 形式标注，`legend` 映射到 `sources[]`；
   - 直接引用：在 `answer` 后附引用摘要段落（避免冗长，保留核心句）；
   - 链接引用：若来源为 URL/可定位文档锚点，提供可点击链接。
3) 对齐与验证：
   - 校验 `content` 包含被引用的答案关键语句或其等价表达；
   - 根据 `start_position/end_position` 或 `chunk_index` 验证可定位性；
   - 记录与比对：引用覆盖率、引用一致性、重复引用去重。
4) 融合与重排的影响：
   - 开启/关闭重排比较引用来源排序与答案质量；
   - RRF/线性融合时引用选择策略（优先分数高且语义贴合）。

## 完整示例与演示
- 输入：见 `examples/input.json`（开启 `include_citations` 与 `citation_style`）。
- 期望输出：
  - `answer`（含行内 [1]/[2] 或 legend 样式）；
  - `sources[]`：`document_id/document_filename/chunk_id/chunk_index/content/score/start_position/end_position/metadata`；
  - 示例错误：见 `examples/error_response.json`（引用 chunk 不存在或命名不一致）。

## 授课老师指导
- 讲稿：见 `teacher-script.md`（时间轴、话术、演示指令）。
- 黑板/投屏步骤：见 `blackboard-steps.md`（逐步 API 演示与一致性检查）。
- 提交前检查：见 `checklist.md`。

## 学生任务
- 课堂：为 ≥3 个问题生成“带引用且可溯源”的答案，对比开启/关闭重排的差异；提交模板。
- 课后：扩展一种引用样式（如行内 + 链接），补充异常与兜底策略。

## 对齐与连续性
- 对齐 RAG 主线契约：字段命名与错误结构统一；引用与来源可定位。
- 承接 Lesson 12 的来源统一与元数据归一；引出 Lesson 14 的缓存优化（来源与引用缓存键）。

## 提交与验收
- 评估标准：引用覆盖率、引用一致性、命名与错误结构统一、日志与截图可复核。
- 验收流程：复制 `templates/submission.md` 填写；附响应片段与截图；不可虚构声明。

## 术语与概念定义
- Citation：引用样式与展示策略（行内/legend/链接）。
- Traceability：来源可定位（位置/chunk/链接）与一致性验证。
- Coverage & Consistency：覆盖率与一致性指标的定义与统计方法。

## 提交前检查清单
- 见 `checklist.md`。

## 边界声明
- 不覆盖复杂跨模态引用与自动纠错；
- 不更换检索与嵌入模型；参数调优在已完成的课次中处理。