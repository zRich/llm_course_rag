# Lesson 09 提交前检查清单：元数据过滤

## 功能
- 支持 `eq/in/contains/range/exists` 五类算子与 `and/or` 组合。
- 多值字段（如 `tags`）与字符串目标的 `contains` 语义清晰且一致。
- 缺失字段处理明确（`exists`），默认不命中且不抛未捕获异常。

## 性能
- 在 1k 文档下，带过滤查询的延迟 ≤ 300ms（示例环境）。
- 高选择性字段优先过滤（如 `language/doc_type/published`）。

## 一致性
- 字段命名与模式与课堂示例一致；错误结构统一。
- DSL 字段与代码实现一致；提交模板字段填写完整。

## 错误路径与兜底
- 至少包含 2 条错误演示：未知算子、类型/日期解析错误。
- 错误返回包含 `code/message/details/hints`，且与 README 描述一致。

## 测试与提交
- 演示脚本可直接运行并复现课堂示例。
- 已填写 `templates/lab_submission_template.md` 并附运行截图。