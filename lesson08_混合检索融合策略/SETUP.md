# 环境与依赖准备 — Lesson 08 混合检索融合策略

## 运行环境
- Python ≥ 3.9（建议 3.10+）。
- 可选：PostgreSQL（用于关键词检索真实数据）、向量库/嵌入服务（用于语义检索）。

## 依赖安装（示例）
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# 若仅运行融合示例脚本，无额外第三方依赖
# 如需和 Lesson 07 联动的中文分词/数据库访问，请根据课程仓库要求安装：
pip install jieba psycopg2-binary
```

## 目录说明
- `examples/fusion_demo.py`：融合演示脚本，可直接运行并切换方法。
- `templates/lab_submission_template.md`：提交模板，按要求填写并附输出片段。

## 运行指令
```bash
python examples/fusion_demo.py --method rrf --top_k 10
python examples/fusion_demo.py --method weighted --top_k 10 --w_k 0.6 --w_v 0.4
```

## 常见问题
- 报错 "No module named ..."：请确认虚拟环境已激活并安装依赖。
- 输出为空：检查示例数据是否加载成功；或切换方法后再试。