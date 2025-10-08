# Lesson 09 环境与依赖：元数据过滤

## 环境准备
- 语言：`Python 3.10+`
- 依赖：标准库（`json`, `datetime`）即可；无需第三方包。

## 运行指令
```bash
python lesson09_元数据过滤/examples/metadata_filter_demo.py
```

## 常见问题
- 日期解析失败：统一使用 ISO 8601；示例已支持 `YYYY-MM-DD` 与 `YYYY-MM-DDTHH:MM:SS`。
- `in.values` 为空：按约定视为不命中，避免误选。
- `contains` 目标类型不匹配：分别支持字符串与列表；其他类型默认不命中。