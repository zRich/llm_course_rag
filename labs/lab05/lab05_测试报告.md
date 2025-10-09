实验 5（Lab05）端到端测试报告

更新时间：2025-10-09 02:22（本地环境）

环境与版本
- 服务启动：`uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload`
- 预览地址：`http://localhost:8000/`
- 测试脚本：`labs/full/lab05/scripts/test_lab05_e2e.sh`

测试范围
- 健康检查、系统统计
- 文档上传、分块与向量化
- 向量检索与过滤 DSL（`eq`、`exists`）
- 全量重建（reindex）端点
- 文档删除与向量级联清理
- 文档列表重定向行为

执行过程与结果摘要
- 健康检查：`/api/v1/system/health` 返回所有组件 `healthy`（数据库、向量存储、嵌入服务、火山引擎 API）。
- 文档上传：上传 `lab05_manual_test.txt` 成功，返回 `id = f87bc5a8-41f1-4694-8c5e-8f040479953b`；初始 `is_vectorized = false`，`chunk_count = 2`。
- 增量向量化：调用增量向量化后，两个分块均更新为 `is_vectorized = true`，处理结果显示 `processed_docs = 1`、`processed_chunks = 2`。
- 向量检索与过滤：
  - 检索 `query = "Lab05"` 且过滤 `document_filename == "lab05_manual_test.txt"` 时，返回 1 个相关结果（匹配该文档分块）。
  - 删除文档后再次检索，同等过滤返回 0 个结果，验证向量级联清理生效。
- 全量重建（reindex）：触发 `/api/v1/vectors/reindex` 返回 500 错误：`"VectorService" object has no attribute "reindex_all_documents"`，需要修复服务方法或路由实现。
- 系统统计：`/api/v1/system/stats` 正常返回（含文档与分块计数、向量化统计等），当前状态健康。
- 文档列表重定向：
  - 访问 `/api/v1/documents`（无尾斜杠）返回 307 并重定向。
  - 访问 `/api/v1/documents/`（有尾斜杠）返回 200 并列出文档信息。

关键输出摘录
- 检索（删除前）：
  `{ "success": true, "message": "搜索完成，找到 1 个相关结果", "query": "Lab05", "total_found": 1, "results": [{ "document_filename": "lab05_manual_test.txt", "chunk_index": 0, "score": 0.48 } ] }`
- 检索（删除后）：
  `{ "success": true, "message": "搜索完成，找到 0 个相关结果", "query": "Lab05", "total_found": 0, "results": [] }`
- 重建（reindex）：
  `{ "success": false, "error_code": "HTTP_500", "message": "重新索引失败: 'VectorService' object has no attribute 'reindex_all_documents'" }`

结论
- 通过的功能：健康检查、文档上传、分块处理、增量向量化、向量检索过滤 DSL、文档删除级联清理、系统统计、文档列表重定向行为。
- 存在问题：全量重建端点返回 500，疑似缺失 `VectorService.reindex_all_documents` 方法或未正确调用实现。

修复建议
- 在 `src/services/vector_service.py`（或对应实现文件）中补充 `reindex_all_documents()` 方法，并在路由 `/api/v1/vectors/reindex` 中调用。
- 方法应：
  - 支持 `force: bool` 参数以在存在向量时强制重建；
  - 迭代所有已处理文档与分块，重新生成并写入向量存储；
  - 返回重建统计（重建文档数、分块数、耗时、错误计数）。
- 为该端点添加集成测试，断言 200 响应与统计字段。

附：操作路径与产物
- 手册：`labs/lab05/lab05_操作手册.md`
- 报告：`labs/lab05/lab05_测试报告.md`
- 脚本：`labs/full/lab05/scripts/test_lab05_e2e.sh`