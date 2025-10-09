# Lesson 15 黑板/投屏操作步骤（逐步）

## 准备
- 打开 Lesson 15 目录，展示 `README.md` 的“示例与演示”章节与 `checklist.md`。
- 预览 `examples/input_example.json` 批量输入结构。

## 步骤
1. 说明批量任务结构与关键字段（doc_ids、metadata、batch_id）。
2. 运行批处理（演示命令/伪代码）：
   - 运行应用：`uv run rag-server`
   - 触发批处理（示例）：在课堂演示脚本中调用 `process_documents_batch(input_example)`。
3. 展示日志关键信息：
   - 进度输出：`processed=N, failed=F, duplicates=D, throughput=X doc/min`。
   - 重试与退避：`attempts`, `backoff_ms`, `last_error`。
   - 检查点写入：`checkpoint:{batch_id}:{shard}`。
4. 故障注入：
   - 提高解析失败概率或暂时断开外部模型（演示环境中的开关）。
   - 观察失败原因分布与降级路径启动（简化清洗+固定分块）。
5. 再次运行（恢复）：
   - 展示“只处理未完成项”，重复率 ≤ 1%。
   - 记录恢复耗时：`resume_time=XXX ms`。
6. 打开 `checklist.md`，与全班一起根据验收标准逐条勾选。

## 检查点
- 键与命名：`batch_id`、`shard`、`idempotency_key` 统一且可追踪。
- 指标：完成率、重复率、失败率、吞吐、延迟与恢复时间输出完整。
- 错误结构：符合 `{ error: { code/message/details/hints } }`。

## 演示后收尾
- 保存日志片段与截图，作为提交物的一部分。
- 提醒学生按模板整理实验报告并提交。