# 黑板/投屏操作步骤（Lesson 12）— 多文档源处理

## 逐步展示
1. 展示 `.env` 与项目分块配置（承接 Lesson 11）。
2. 上传三类来源：
   - `POST /api/v1/documents/upload`（PDF：`sample.pdf`）
   - `POST /api/v1/documents/upload`（TXT：`sample.txt`）
   - `POST /api/v1/documents/upload`（URL：`https://example.com/article`，或课程站点示例）
3. 查看文档列表与解析状态：
   - `GET /api/v1/documents/`（字段：`document_id/document_filename/source_type/metadata/status`）
4. 去重与版本化演示：
   - 重复上传同一文档，展示指纹/哈希比对；
   - 更新版本后，刷新索引并记录日志。
5. 向量化与检索：
   - `POST /api/v1/vectors/vectorize`（完成三源向量化）
   - `POST /api/v1/retrieval/search`（按 `source_type in ["pdf","url"]` 过滤；展示融合与重排差异）
6. 问答端到端：
   - `POST /api/v1/qa/ask`，展示 `sources[].document_id/.document_filename/.chunk_id/.score` 与答案一致性。
7. 现场填写提交模板：
   - 打开 `templates/submission.md`，示范配置、日志、指标与结论的填写。

## 一致性检查点
- 字段命名一致：
  - 检索：`chunk_id/document_id/chunk_index/content/score/start_position/end_position/metadata`。
  - 文档：`document_id/document_filename/source_type/metadata/status`。
  - 问答来源：`sources[].document_id/.document_filename/.chunk_id/.score`。
- 错误结构统一：`{ error: { code/message/details/hints } }`。
- 改数据后必须重向量化与索引刷新；日志需含时间戳与接口路径。

## 现场演示提示
- URL 抓取不稳定时，使用静态快照或课程提供的示例 URL。
- 比较开启/关闭重排时的排序差异与答案质量。
- 强调不可虚构：日志与截图需可复核，命名与错误结构统一。