# 第18节课：文本清洗模块

## 课程概述

本节课实现了一个完整的文本清洗系统，包含噪声检测、质量评估、自动化清洗流程和Web API接口。适合45分钟的课堂教学和实验。

## 模块结构

```
lesson18_text_cleaning/
├── text_cleaner.py      # 核心文本清洗类
├── noise_detector.py    # 噪声检测模块
├── quality_evaluator.py # 文本质量评估
├── cleaning_pipeline.py # 自动化清洗流程
├── api.py              # FastAPI接口
├── config.py           # 配置管理
├── test_text_cleaning.py # 功能测试
├── demo.py             # 演示脚本
└── README.md           # 说明文档
```

## 核心功能

### 1. 文本清洗 (TextCleaner)
- 去除多余空格和换行符
- 清理特殊字符和控制字符
- Unicode标准化
- URL和邮箱地址清理
- 自定义清洗规则

### 2. 噪声检测 (NoiseDetector)
- 编码问题检测
- 重复内容识别
- 文本长度验证
- 字符分布分析

### 3. 质量评估 (QualityEvaluator)
- 可读性评分
- 完整性检查
- 一致性分析
- 信息密度计算
- 结构质量评估

### 4. 自动化流程 (CleaningPipeline)
- 批量文本处理
- 多线程并发处理
- 质量过滤
- 统计报告生成

### 5. Web API接口
- RESTful API设计
- 单文本清洗接口
- 批量处理接口
- 质量评估接口

## 快速开始

### 1. 基础使用

```python
from text_cleaner import TextCleaner
from quality_evaluator import QualityEvaluator

# 创建清洗器
cleaner = TextCleaner()

# 清洗文本
text = "  这是一个测试文本...   \n\n  "
cleaned_text = cleaner.clean(text)
print(f"清洗后: {cleaned_text}")

# 质量评估
evaluator = QualityEvaluator()
result = evaluator.evaluate(cleaned_text)
print(f"质量评分: {result['overall_score']:.2f}")
```

### 2. 使用清洗流程

```python
from cleaning_pipeline import CleaningPipeline

# 创建流程
pipeline = CleaningPipeline()

# 批量处理
texts = ["文本1", "文本2", "文本3"]
results = pipeline.process_batch(texts)

# 查看统计
stats = pipeline.get_statistics()
print(f"处理成功: {stats['success_count']}")
print(f"平均质量: {stats['avg_quality']:.2f}")
```

### 3. 启动API服务

```bash
# 安装依赖
pip install fastapi uvicorn

# 启动服务
python api.py
```

访问 http://localhost:8000/docs 查看API文档

### 4. 运行演示

```bash
# 运行演示脚本
python demo.py

# 运行测试
python test_text_cleaning.py
```

## API接口说明

### 单文本清洗
```http
POST /clean
Content-Type: application/json

{
  "text": "要清洗的文本",
  "options": {
    "remove_extra_spaces": true,
    "remove_special_chars": true
  }
}
```

### 批量处理
```http
POST /clean/batch
Content-Type: application/json

{
  "texts": ["文本1", "文本2"],
  "options": {...}
}
```

### 质量评估
```http
POST /evaluate
Content-Type: application/json

{
  "text": "要评估的文本"
}
```

## 配置说明

系统支持多种配置方式：

1. **环境变量配置**
```bash
export API_PORT=8080
export MIN_QUALITY_SCORE=0.7
export BATCH_SIZE=50
```

2. **配置文件** (config.json)
```json
{
  "cleaning": {
    "min_text_length": 10,
    "max_repetition_ratio": 0.3
  },
  "api": {
    "port": 8000,
    "debug": false
  }
}
```

3. **代码配置**
```python
from config import get_config

config = get_config()
config.update_config('cleaning', min_text_length=20)
```

## 课堂实验建议

### 实验1：基础文本清洗 (10分钟)
1. 使用TextCleaner清洗不同类型的文本
2. 观察清洗前后的差异
3. 调整清洗参数，观察效果变化

### 实验2：噪声检测 (10分钟)
1. 使用NoiseDetector检测文本问题
2. 分析不同噪声类型的检测结果
3. 理解噪声评分的计算方法

### 实验3：质量评估 (10分钟)
1. 使用QualityEvaluator评估文本质量
2. 分析各个维度的评分
3. 根据建议改进文本质量

### 实验4：完整流程 (10分钟)
1. 使用CleaningPipeline处理批量文本
2. 观察处理统计和质量分布
3. 调整质量阈值，观察过滤效果

### 实验5：API接口 (5分钟)
1. 启动API服务
2. 使用curl或Postman测试接口
3. 查看API文档和响应格式

## 扩展思考

1. **性能优化**：如何提高大批量文本的处理速度？
2. **规则定制**：如何为特定领域定制清洗规则？
3. **质量标准**：如何根据应用场景调整质量评估标准？
4. **实时处理**：如何实现流式文本清洗？

## 常见问题

**Q: 如何处理特殊编码的文本？**
A: NoiseDetector会自动检测编码问题，TextCleaner支持Unicode标准化。

**Q: 质量评分偏低怎么办？**
A: 查看QualityEvaluator的详细建议，针对性地改进文本内容。

**Q: 如何自定义清洗规则？**
A: 修改config.py中的清洗配置，或继承TextCleaner类添加自定义方法。

**Q: API服务如何部署？**
A: 使用uvicorn或gunicorn部署，支持Docker容器化部署。

## 总结

本节课通过实现完整的文本清洗系统，学习了：
- 文本预处理的核心技术
- 噪声检测和质量评估方法
- 自动化处理流程设计
- Web API接口开发
- 配置管理和系统架构

这些技能在RAG系统的文档预处理阶段非常重要，直接影响检索和生成的质量。