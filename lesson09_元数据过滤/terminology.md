# Lesson 09 术语与概念定义：元数据过滤

- 元数据（metadata）：描述文档的结构化信息（来源、类型、时间、标签等）。
- 模式（schema）：字段命名与类型的统一约定。
- 选择性（selectivity）：过滤条件对结果集缩小的程度，越高越好。
- 过滤算子：`eq/in/contains/range/exists`。
- 布尔组合：`and/or` 逻辑结构与短路规则。
- 预过滤（pre-filtering）：在候选生成前的过滤；后过滤（post-filtering）：在融合结果后的裁剪。
- 统一错误结构：`{ error: { code/message/details/hints } }`。
- 一致性：命名、类型与错误结构在文档与代码间保持统一。