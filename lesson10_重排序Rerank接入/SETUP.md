# Lesson 10 环境与依赖：重排序（Rerank）接入

## 环境准备
- 语言：`Python 3.10+`
- 依赖：标准库即可；可选 `sentence-transformers` 用于 Cross-Encoder。

## 安装（可选）
```bash
pip install -U sentence-transformers
```

## 运行
```bash
python lesson10_重排序Rerank接入/examples/rerank_demo.py --alpha 0.5 --top_k 100
python lesson10_重排序Rerank接入/examples/rerank_demo.py --alpha 0.5 --top_k 100 --use_ce
```

## 常见问题
- 模型下载失败或无 GPU：自动回退启发式；不阻塞课堂演示。
- 指标无提升：检查 `alpha/top_k` 与归一化；确认相关样本是否存在。
- 顺序不稳定：确保并列打断规则（`final_score/base_score/id`）启用。