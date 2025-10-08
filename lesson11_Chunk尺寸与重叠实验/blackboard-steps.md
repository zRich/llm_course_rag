# 黑板/投屏操作步骤（Lesson 11）

## 逐步展示
1. 展示 `.env` 中的分块配置：`CHUNK_SIZE` 与 `CHUNK_OVERLAP`。
2. 切换到第一组配置（示例：`1000/200`），保存并重向量化。
3. 打开 API 文档（`/docs`），调用：
   - `GET /api/v1/system/health`（健康检查）
   - `GET /api/v1/documents/`（文档列表与分块状态）
   - `POST /api/v1/vectors/vectorize`（向量化）
   - `POST /api/v1/retrieval/search`（混合检索与重排）
   - `POST /api/v1/qa/ask`（问答端到端）
4. 实时展示响应片段：
   - 检索结果：`chunk_id/document_id/chunk_index/content/score/start_position/end_position/metadata`
   - 问答来源：`sources[].document_id/.document_filename/.chunk_id/.score`
5. 将第二、第三组配置重复 2–4 步，现场对比核心字段与指标。
6. 打开提交模板 `templates/submission.md`，示范如何填写：配置、日志、指标与结论。

## 一致性检查点
- 字段命名一致：检索结果与问答来源字段与契约一致。
- 错误结构统一：`{ error: { code/message/details/hints } }`。
- 改配置后必须重向量化与索引刷新。

## 现场演示提示
- 中文语料建议尺寸与重叠区间：`600..1500` / `100..300`。
- 对比有/无重排时的排序差异与答案质量。
- 强调不可虚构：日志与截图需可复核。