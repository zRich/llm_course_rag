# 术语与概念定义 — Lesson 08 混合检索融合策略

- BM25：一种经典的文本匹配评分函数，常用于关键词检索。
- TF-IDF：词频-逆文档频率，衡量词在文档与语料中的重要性。
- 向量检索（Vector Search）：将文本编码为向量，通过相似度（如余弦）进行检索。
- RRF（Reciprocal Rank Fusion）：以名次为基础的融合方法，分数为若干路 `1/(k+rank)` 之和。
- 线性加权（Weighted Sum）：将不同来源的分数归一化后按权重线性组合。
- 归一化（Normalization）：将分数映射到统一区间，如 min-max 到 [0,1]。
- 去重键（Dedup Key）：用于识别并合并重复结果的键，一般为 `id` 或 `source+offset`。
- Top-K：返回前 K 条结果，常用于限制输出与性能控制。
- 命中率（Hit Rate）：Top-K 中包含相关项的比例。
- MRR（Mean Reciprocal Rank）：平均倒数排名，衡量首个正确结果的排名质量。
- nDCG（Normalized Discounted Cumulative Gain）：归一化折损累计增益，衡量排序整体质量。