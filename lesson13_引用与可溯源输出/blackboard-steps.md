# 黑板/投屏操作步骤（Lesson 13）— 引用与可溯源输出

## 逐步展示
1. 展示 `.env` 与上一课的来源与分块配置。
2. 检索候选：
   - `POST /api/v1/retrieval/search`（确认 `content/score/chunk_id/chunk_index/start_position/end_position/metadata`）。
3. 问答输出（带引用）：
   - `POST /api/v1/qa/ask`（设置 `include_citations=true` 与 `citation_style=legend` 或 `inline`）。
   - 展示响应：
     - `answer`：含行内 [1]/[2] 或 legend 样式；
     - `sources[]`：`document_id/document_filename/chunk_id/chunk_index/content/score/start_position/end_position/metadata`。
4. 一致性检查：
   - 逐句对照答案关键句与 `sources[].content`；
   - 根据 `start_position/end_position` 或 `chunk_index` 验证可定位；
   - 检查去重与排序（与重排/融合的交互）。
5. 错误演示：
   - 触发缺失引用或引用指向不存在的 `chunk_id`，展示统一错误结构：`{ error: { code/message/details/hints } }`。
6. 现场填写 `templates/submission.md`：配置、日志、指标、响应片段与截图。

## 一致性检查点
- 字段命名统一：
  - 检索：`chunk_id/document_id/chunk_index/content/score/start_position/end_position/metadata`；
  - 问答来源：`sources[].document_id/.document_filename/.chunk_id/.score`；
  - 错误结构：`{ error: { code/message/details/hints } }`。
- 改排序/融合策略后保持引用字段与样式一致；日志需含时间戳与接口路径。

## 现场演示提示
- 行内标注与 legend 可同时演示，强调映射关系；
- 中文场景建议保留摘要引用以避免冗长，但保证可定位；
- 强调不可虚构：所有引用需可在来源片段中复核。