# Lesson 10 术语与概念定义：重排序（Rerank）接入

- Rerank：对候选进行二次打分与重排的过程。
- Cross-Encoder：将 query 与 doc 共同编码后打分的模型。
- Bi-Encoder：分别编码后用向量相似度打分的模型。
- 融合权重（alpha）：控制基线分数与 Rerank 分数的线性权重。
- 稳定排序（stable sort）：分数并列时采用确定性打断规则的排序。
- MRR/nDCG：评估检索/排序质量的常用指标。
- Top-K：仅对前 K 条候选进行重排的范围。