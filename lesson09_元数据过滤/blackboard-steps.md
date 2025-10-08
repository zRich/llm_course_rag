# Lesson 09 黑板/投屏操作步骤：元数据过滤

1. 展示课程目标与三要素（目标/重点/难点），点出字段规范与算子总览。
2. 投屏过滤 DSL（AND 组合 + `eq/in/contains/range/exists`），同时展示统一错误结构。
3. 现场运行 `examples/metadata_filter_demo.py`：
   - 第一次运行：基础 DSL，观察命中。
   - 第二次运行：修改 `contains` 目标类型（字符串/列表），对比结果。
   - 第三次运行：注入未知算子或类型错误，展示错误结构。
4. 打开 `checklist.md`，逐项勾选功能/性能/一致性与错误路径。
5. 现场演示填写 `templates/lab_submission_template.md`，强调命名与口径一致性。
6. 课堂小结与下一课衔接：过滤后的候选集将进入排序优化。