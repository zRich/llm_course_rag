# Lesson 09：元数据过滤

## 课次信息
- 时长：90 分钟
- 前置：Lesson 08（已实现混合检索融合，产生候选集）
- 后置：Lesson 10（将在过滤后的候选集上继续优化排序与答案抽取）

## 教学目标（SMART）
- S：能为检索候选集正确应用元数据过滤（来源、类型、时间区间、标签、语言、发布状态等），并在课堂示例中实现端到端过滤管道。
- M：过滤精确率（Precision of filters）≥ 95%；误筛率（错误通过率）≤ 5%；在 1k 文档下带过滤查询延迟 ≤ 300ms（本地示例环境）。
- A：在提供的数据与工具下，独立完成可运行示例与提交模板。
- R：与 RAG 主线一致，服务融合后的候选集收敛与质量提升。
- T：90 分钟内完成并验收（含演示与提交物核查）。

## 教学重点
- 元数据模式与字段规范（schema）：命名一致性、类型与取值域。
- 常用过滤算子：`eq/in/contains/range/exists` 与布尔组合 `and/or`。
- 预过滤 vs 后过滤：索引阶段归一化与查询阶段裁剪的取舍。
- 一致性与错误结构：统一 `{ error: { code/message/details/hints } }`。

## 教学难点
- 日期/时间区间解析与统一格式（ISO 8601）。
- 多值字段（如 `tags`）的包含语义；字符串 vs 列表的处理一致性。
- 缺失字段与类型不匹配的稳健性；AND/OR 优先级与短路规则。

## 知识点详解
1) 字段与命名一致性
   - 必备字段建议：`id/title/source/doc_type/tags/language/published/created_at/updated_at/views`
   - 统一命名风格（蛇形或小驼峰），示例统一使用下划线风格。
2) 过滤 DSL 结构（课堂使用）
   - 布尔结构：
     ```json
     {
       "and": [
         {"eq": {"field": "language", "value": "zh"}},
         {"in": {"field": "doc_type", "values": ["guide", "faq"]}},
         {"contains": {"field": "tags", "value": "元数据"}},
         {"range": {"field": "created_at", "gte": "2024-01-01", "lte": "2024-12-31"}},
         {"eq": {"field": "published", "value": true}},
         {"range": {"field": "views", "gte": 100}}
       ]
     }
     ```
   - 错误结构统一：`{ "error": { "code": 400, "message": "...", "details": { ... }, "hints": ["..."] } }`
3) 预过滤与后过滤
   - 预过滤：在索引或候选生成前应用（减少无效匹配，提升性能）。
   - 后过滤：在融合结果后按元数据裁剪（提升相关性与一致性）。
4) 稳健性策略
   - 缺失字段：`exists` 明确约束；对缺失时的默认行为要可预期（如视为不命中）。
   - 类型错误：统一返回 400/422，包含 `field/type/expected/actual`。

## 完整示例与演示
- 输入：上方 DSL；数据集参见 `examples/metadata_filter_demo.py`。
- 期望输出：满足 AND 组合下所有条件的文档列表（含命中解释）。
- 异常与兜底：
  - 未知算子 → 400；日期解析失败 → 422；`values` 非列表 → 400。
  - 返回统一错误结构并给出修正提示（hints）。

## 授课老师指导
- 时间轴与脚本：
  - 导入 10min：目标与产物、字段规范与算子总览。
  - 讲解 25min：DSL、布尔组合、预/后过滤策略与性能取舍。
  - 实践 40min：现场运行 `examples/metadata_filter_demo.py`，增改条件、观察输出与错误。
  - 总结 15min：提交物结构核查、对齐与边界声明。
- 话术与演示指令：详见 `teacher-script.md`；黑板/投屏步骤见 `blackboard-steps.md`。

### 课堂提问与练习（含参考答案）
- 请见 `questions.md`，覆盖概念理解、布尔组合、类型稳健性与性能考量。

### 常见误区与应对话术
- 类型混淆（字符串日期 vs `datetime`）→ 统一解析与明确错误返回。
- `contains` 对字符串与列表不统一 → 显式同时支持并测试。
- `in` 的 `values` 空列表语义不清 → 约定为空列表视为“不命中”。

### 黑板/投屏操作步骤（逐步）
1. 投屏课程目标与字段规范；2. 展示 DSL 与错误结构；3. 运行示例脚本并逐步增减条件；4. 打开检查清单逐项对照；5. 现场填写提交模板要点。

## 学生任务
- 课堂任务：在示例脚本中增补 2 条新过滤条件（`exists` 与另一种 `range`），并验证命中与错误路径。
- 课后任务：完成 `lab.md` 要求，提交 `templates/lab_submission_template.md` 填写后的内容与演示截图。

## 对齐与连续性
- 与 Module-RAG 主线契约保持一致：候选生成 → 融合 → 元数据过滤 → 排序优化。
- 承接 Lesson 08 的融合结果，输出过滤后的候选集供 Lesson 10 使用。
- 仓库结构与命名统一，示例与模板位置符合最佳实践。

## 提交与验收
- 评估标准与阈值：详见 `acceptance.md`；错误结构与命名一致性为必查项。

## 术语与概念定义
- 请见 `terminology.md`，统一核心概念与评估指标。

## 提交前检查清单（教师/学生）
- 请见 `checklist.md`，覆盖功能、性能、一致性与错误路径。

## 边界声明
- 请见 `boundaries.md`，明确不覆盖范围与替代方案。

## 环境与依赖
- 请见 `SETUP.md`，包含运行指令与常见问题。

## 课后实验
- 请见 `lab.md`，形成教学闭环与评分建议。

## 导航
- 示例：`examples/metadata_filter_demo.py`
- 模板：`templates/lab_submission_template.md`
- 教师讲稿：`teacher-script.md`
- 黑板/投屏：`blackboard-steps.md`
- 检查清单：`checklist.md`
- 提问练习：`questions.md`
- 术语与概念：`terminology.md`
- 提交与验收：`acceptance.md`
- 对齐与连续性：`alignment.md`
- 边界声明：`boundaries.md`
- 环境与依赖：`SETUP.md`
- 课后实验：`lab.md`