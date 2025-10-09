# Lesson 16 黑板/投屏操作步骤（逐步）

## 准备
- 打开 Lesson 16 目录，展示 `README.md` 的“示例与演示”章节与 `checklist.md`。
- 预览 `examples/input_delta.json` 增量输入结构。

## 步骤
1. 说明差分类型与关键字段（upserts/updates/deletes、ETag/Last-Modified、content_hash、version）。
2. 运行增量更新（演示命令/伪代码）：
   - 运行应用：`uv run rag-server`
   - 触发增量更新（示例）：在课堂脚本中调用 `process_incremental_updates(input_delta)`。
3. 展示日志关键信息：
   - 差分统计：`added=A, updated=U, deleted=D` 与来源字段说明。
   - 失效事件：`invalidate: qa/retrieval/rerank/vector`，展示命名空间与键样例。
   - 局部重建：分块范围与向量重建数量；重试与退避参数。
4. 故障注入：
   - 模拟向量服务超时或写冲突（演示环境开关）。
   - 观察重试退避与租约回退；记录 `last_error` 与原因分布。
5. 再次运行（验证）：
   - 展示重复率 ≤ 0.5%、失效命中率 ≥ 99%、陈旧窗口 ≤ 5 分钟。
   - 记录吞吐与延迟：`latency(p50/p95)` 与 `throughput`。
6. 打开 `checklist.md`，与全班一起根据验收标准逐条勾选。

## 检查点
- 键与命名：`idempotency_key` 与失效命名空间统一且可追踪。
- 指标：`update_accuracy`、`dup_rate`、`invalidate_hit_rate`、`stale_window` 输出完整。
- 错误结构：符合 `{ error: { code/message/details/hints } }`。

## 演示后收尾
- 保存日志片段与截图，作为提交物的一部分。
- 提醒学生按模板整理实验报告并提交。