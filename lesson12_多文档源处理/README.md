# Lesson 12：多文档源处理（Multi-Source Ingestion & Retrieval）

## 课次信息
- 时长：90 分钟
- 前置：Lesson 11（Chunk 尺寸与重叠区间已验证）
- 后置：Lesson 14（缓存策略与命中率优化，将复用本课产出）

## 教学目标（SMART）
- S：能在中文语料场景中完成“多文档源”接入（本地 PDF/TXT、Web URL），统一解析与元数据归一化，并在 90 分钟内完成端到端检索与问答验证。
- M：
  - 解析成功率 ≥ 95%；重复文档去重率 ≥ 90%；
  - 跨源检索召回覆盖率 ≥ 80%（以人工对齐问题域为准）；
  - 错误响应结构统一（`{ error: { code/message/details/hints } }`），字段命名一致。
- A：在现有项目与工具内独立完成 3 个来源的接入与验证。
- R：服务 RAG 主线，保障来源异构数据能统一入库与检索。
- T：在 90 分钟内完成并提交模板化报告，日志与截图可复核。

## 教学重点
- 多源接入适配器与解析器（PDF/TXT/URL）。
- 元数据归一化与字段命名一致性（`document_id/document_filename/source_type/...`）。
- 去重与版本化：`document_id` 唯一、更新策略与索引刷新。
- 跨源检索与过滤：按 `source_type`/`metadata` 过滤，结果融合与排序一致性。

## 教学难点
- 异构解析失败与编码问题（中文断句、PDF 结构、URL 抓取）。
- 元数据映射不一致导致过滤失败或来源追踪困难。
- 重复文档与 `document_id` 冲突、索引未刷新造成旧数据残留。
- 跨源分数归一化与融合策略的可比性。

## 知识点详解
1) 来源适配与解析：
   - PDF 与 TXT 解析策略与常见坑（分页符、表格、乱码）。
   - URL 抓取与正文抽取的边界（反爬/速率限制/动态渲染）。
2) 元数据归一化与命名一致性：
   - 关键字段：`document_id/document_filename/source_type/content/metadata`。
   - 统一错误结构：`{ error: { code/message/details/hints } }`。
3) 去重与版本化：
   - 指纹与哈希；重复判定与更新策略；索引刷新时机。
4) 端到端检索与过滤：
   - 向量化、关键词检索、过滤 DSL、融合与重排；
   - 过滤示例：按 `source_type in ["pdf","url"]` 与时间范围。

## 完整示例与演示
- 输入：见 `examples/input.json`（三源接入与混合检索 + 问答）。
- 期望输出：
  - 检索：`chunk_id/document_id/chunk_index/content/score/start_position/end_position/metadata`；
  - 问答来源：`sources[].document_id/.document_filename/.chunk_id/.score`。
- 异常与兜底：见 `examples/error_response.json`（非法来源或重复文档）。

## 授课老师指导
- 讲稿：见 `teacher-script.md`（时间轴、话术、演示指令）。
- 黑板/投屏步骤：见 `blackboard-steps.md`（逐步 API 演示与一致性检查）。
- 提交前检查：见 `checklist.md`。

## 学生任务
- 课堂：完成 3 源接入（PDF/TXT/URL），统一元数据并端到端验证；提交模板。
- 课后：扩展一类来源（如 `docx` 或站点抓取），撰写异常与兜底策略。

## 对齐与连续性
- 与 Module 与 Syllabus 的主线契约一致：字段命名与错误结构统一。
- 承接 Lesson 11 的分块策略，对检索覆盖与来源质量进行交叉验证。
- 引出 Lesson 14 的缓存与命中率优化（来源维度的缓存键与复用）。

## 提交与验收
- 评估标准：解析成功率、去重率、跨源召回覆盖、来源一致性、错误结构统一。
- 验收流程：复制 `templates/submission.md` 填写；附日志与截图；字段命名与错误结构一致；不可虚构声明。

## 术语与概念定义
- Source Adapter/Parser：来源适配与解析组件。
- Normalization：元数据归一化与字段一致性。
- Deduplication：指纹/哈希去重与版本化策略。
- Fusion/Rerank：融合策略与重排服务，保证排序与答案质量。

## 提交前检查清单
- 见 `checklist.md`。

## 边界声明
- 不覆盖高级正文抽取（JS 动态与复杂反爬）与跨模态解析；
- 不更换嵌入与检索模型；参数调优在 Lesson 11 与后续课处理。