# 黑板/投屏操作步骤：Lesson 08 混合检索融合策略

1. 投屏课程目标与产出
   - 展示 `README.md` 的 SMART 目标与验收口径。
   - 说明融合的两种方法：RRF 与线性加权。

2. 展示输入示例与字段规范
   - 打开 `examples/fusion_demo.py`，说明输入两路结果的字段：`id/title/score/source/rank`。
   - 指出融合输出增加 `fused_score/method/warnings`。

3. 现场运行演示脚本
   - 运行 `python examples/fusion_demo.py --method rrf --top_k 10`。
   - 运行 `python examples/fusion_demo.py --method weighted --top_k 10 --w_k 0.6 --w_v 0.4`。

4. 一致性检查与错误码
   - 检查输出是否按 `fused_score` 递减排序，且无重复 `id`。
   - 当任一路为空时，展示 `warnings` 字段；说明兜底逻辑。

5. 现场填写提交模板
   - 打开 `templates/lab_submission_template.md`。
   - 现场演示如何粘贴输出片段与参数说明，补充对比分析。

6. 课堂小结与下一课衔接
   - 解释如何在 Lesson 09 中使用这些输出做评估（nDCG/MRR）。