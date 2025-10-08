# Lesson 06：最小检索与生成 - 黑板/投屏操作步骤

## 投屏内容准备清单

### 必备投屏材料
- [ ] RAG系统架构图
- [ ] 核心代码示例文件
- [ ] 错误处理示例
- [ ] 参数配置对比表
- [ ] 性能测试结果模板

## 逐步操作指南

### 第一部分：系统架构展示（10分钟）

#### 步骤1：展示RAG系统架构图
**投屏内容：**
```
RAG系统架构
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  用户问题   │───▶│  向量化     │───▶│  向量检索   │───▶│  上下文构建 │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐           │
│  格式化输出 │◀───│  LLM生成    │◀───│  Prompt构建 │◀──────────┘
└─────────────┘    └─────────────┘    └─────────────┘
```

**讲解要点：**
- 指向每个组件，解释其作用
- 强调数据流向的重要性
- 说明每个步骤的输入输出

#### 步骤2：展示核心组件代码结构
**投屏代码：**
```python
class RAGSystem:
    def __init__(self, llm_client, embedder, vector_store, config):
        self.llm_client = llm_client      # LLM客户端
        self.embedder = embedder          # 文本向量化器
        self.vector_store = vector_store  # 向量数据库
        self.config = config              # 配置参数
        
    def query(self, question: str) -> RAGResponse:
        # 1. 向量化问题
        # 2. 检索相关文档
        # 3. 构建上下文
        # 4. 生成答案
        pass
```

**操作要点：**
- 逐行解释每个组件的作用
- 强调依赖注入的设计模式
- 预告后续的详细实现

### 第二部分：核心功能演示（35分钟）

#### 步骤3：同步查询演示
**投屏操作：**

1. **打开终端，切换到项目目录**
```bash
cd /Users/richzhao/dev/llm_courses/courses/11_rag/lesson06_最小检索与生成
```

2. **启动Python交互环境**
```bash
python -i examples/mvp_rag_demo.py
```

3. **演示系统初始化**
```python
# 投屏显示初始化代码
from rag_system import create_rag_system

# 创建RAG系统实例
rag_system = create_rag_system()
print("RAG系统初始化完成")
```

4. **演示基础查询**
```python
# 投屏显示查询代码
question = "什么是向量数据库？"
print(f"用户问题：{question}")

# 执行查询
response = rag_system.query(question)

# 显示结果
print(f"答案：{response.answer}")
print(f"相关文档数：{len(response.retrieved_docs)}")
print(f"置信度：{response.confidence}")
```

**预期输出展示：**
```
用户问题：什么是向量数据库？
答案：向量数据库是一种专门用于存储和检索高维向量数据的数据库系统。它通过向量相似度计算来快速找到相关内容，广泛应用于推荐系统、搜索引擎和AI应用中。
相关文档数：3
置信度：0.85
```

#### 步骤4：异步查询演示
**投屏操作：**

1. **展示异步代码**
```python
import asyncio

async def async_demo():
    question = "RAG系统的优势是什么？"
    print(f"异步查询：{question}")
    
    response = await rag_system.aquery(question)
    print(f"异步结果：{response.answer[:100]}...")
    
# 运行异步演示
asyncio.run(async_demo())
```

2. **对比同步和异步性能**
```python
import time

# 同步查询计时
start_time = time.time()
sync_response = rag_system.query("什么是embedding？")
sync_time = time.time() - start_time

# 异步查询计时
start_time = time.time()
async_response = asyncio.run(rag_system.aquery("什么是embedding？"))
async_time = time.time() - start_time

print(f"同步查询耗时：{sync_time:.2f}秒")
print(f"异步查询耗时：{async_time:.2f}秒")
```

#### 步骤5：流式输出演示
**投屏操作：**

1. **展示流式查询代码**
```python
async def stream_demo():
    question = "详细解释RAG的工作流程"
    print(f"流式查询：{question}")
    print("流式输出：", end="")
    
    async for chunk in rag_system.stream_query(question):
        print(chunk.content, end="", flush=True)
    print()  # 换行

# 运行流式演示
asyncio.run(stream_demo())
```

2. **观察实时输出效果**
- 强调逐字显示的用户体验
- 解释流式输出的技术原理
- 说明适用场景

### 第三部分：错误处理演示（10分钟）

#### 步骤6：错误场景模拟
**投屏操作：**

1. **展示错误处理代码框架**
```python
from rag_system import RetrievalError, GenerationError, ContextTooLongError

def handle_rag_errors(question):
    try:
        response = rag_system.query(question)
        return response
    except RetrievalError as e:
        print(f"检索错误：{e}")
        return "抱歉，无法检索到相关信息"
    except GenerationError as e:
        print(f"生成错误：{e}")
        return "抱歉，生成答案时出现问题"
    except ContextTooLongError as e:
        print(f"上下文过长：{e}")
        return "问题过于复杂，请简化后重试"
```

2. **模拟各种错误场景**
```python
# 场景1：空问题
print("=== 错误场景1：空问题 ===")
result = handle_rag_errors("")

# 场景2：超长问题
print("=== 错误场景2：超长问题 ===")
long_question = "什么是" * 1000  # 故意创建超长问题
result = handle_rag_errors(long_question)

# 场景3：无关问题
print("=== 错误场景3：无关问题 ===")
result = handle_rag_errors("今天天气怎么样？")
```

#### 步骤7：调试信息展示
**投屏操作：**

1. **启用调试模式**
```python
# 启用详细日志
rag_system.enable_debug_logging()

# 执行查询并查看调试信息
response = rag_system.query("什么是向量检索？")

# 显示调试信息
print("=== 调试信息 ===")
print(f"检索到的文档：{len(response.debug_info.retrieved_docs)}")
print(f"向量化耗时：{response.debug_info.embedding_time:.3f}秒")
print(f"检索耗时：{response.debug_info.retrieval_time:.3f}秒")
print(f"生成耗时：{response.debug_info.generation_time:.3f}秒")
```

### 第四部分：参数调优演示（8分钟）

#### 步骤8：参数对比实验
**投屏操作：**

1. **展示参数配置对比表**
```python
# 创建不同配置的RAG系统
configs = [
    {"top_k": 3, "score_threshold": 0.8, "temperature": 0.1},
    {"top_k": 5, "score_threshold": 0.7, "temperature": 0.3},
    {"top_k": 10, "score_threshold": 0.5, "temperature": 0.7}
]

question = "什么是RAG系统？"

print("参数配置对比实验")
print("=" * 50)

for i, config in enumerate(configs, 1):
    print(f"\n配置{i}：{config}")
    
    # 创建新的RAG系统实例
    rag_system_test = create_rag_system(config)
    response = rag_system_test.query(question)
    
    print(f"答案长度：{len(response.answer)}")
    print(f"相关文档数：{len(response.retrieved_docs)}")
    print(f"置信度：{response.confidence:.2f}")
    print(f"答案预览：{response.answer[:100]}...")
```

2. **分析参数影响**
```python
# 投屏显示分析结果
print("\n参数影响分析：")
print("- top_k越大，检索到的文档越多，但可能引入噪音")
print("- score_threshold越高，文档质量越好，但可能检索不到内容")
print("- temperature越高，生成内容越有创意，但可能偏离主题")
```

### 第五部分：课堂练习指导（8分钟）

#### 步骤9：现场练习演示
**投屏操作：**

1. **展示练习任务模板**
```python
# 练习1：参数调优实验
def parameter_tuning_exercise():
    """
    任务：测试不同参数组合，找出最佳配置
    """
    test_questions = [
        "什么是向量数据库？",
        "RAG系统有什么优势？",
        "如何优化检索效果？"
    ]
    
    # TODO: 学生填写不同的参数配置
    configs_to_test = [
        # 配置1：保守型
        {"top_k": 3, "score_threshold": 0.8},
        # 配置2：平衡型
        {"top_k": 5, "score_threshold": 0.7},
        # 配置3：激进型
        {"top_k": 10, "score_threshold": 0.5}
    ]
    
    # 测试代码框架
    for config in configs_to_test:
        print(f"测试配置：{config}")
        # TODO: 学生实现测试逻辑
```

2. **展示提交模板**
```python
# 练习结果提交模板
class ExerciseResult:
    def __init__(self):
        self.best_config = {}           # 最佳参数配置
        self.performance_metrics = {}   # 性能指标
        self.observations = []          # 观察结果
        
    def to_dict(self):
        return {
            "best_config": self.best_config,
            "performance_metrics": self.performance_metrics,
            "observations": self.observations
        }

# 示例填写
result = ExerciseResult()
result.best_config = {"top_k": 5, "score_threshold": 0.7, "temperature": 0.3}
result.performance_metrics = {"avg_response_time": 1.2, "avg_confidence": 0.82}
result.observations = ["top_k=5时效果最佳", "temperature过高会偏离主题"]
```

#### 步骤10：检查清单展示
**投屏内容：**

```
课堂练习检查清单
□ RAG系统成功初始化
□ 同步查询正常工作
□ 异步查询正常工作
□ 流式查询正常工作
□ 错误处理机制有效
□ 参数调优实验完成
□ 性能测试数据记录
□ 提交模板正确填写
□ 代码符合规范要求
□ 文档注释完整清晰
```

## 投屏技术要点

### 屏幕布局建议
- **左侧**：代码编辑器（60%屏幕宽度）
- **右侧**：终端输出（40%屏幕宽度）
- **字体大小**：至少16pt，确保后排学生能看清

### 演示节奏控制
- **代码展示**：逐行解释，不要跳跃
- **运行结果**：等待输出完成再继续
- **错误演示**：故意制造错误，展示处理过程
- **互动时机**：每个大步骤后暂停，询问学生理解情况

### 常见技术问题处理
1. **代码运行失败**：准备备用代码片段
2. **网络连接问题**：准备离线演示数据
3. **性能问题**：准备预录制的演示视频
4. **环境问题**：准备Docker容器化环境

## 课后整理要点

### 投屏内容保存
- [ ] 保存所有演示代码到 `examples/` 目录
- [ ] 保存错误处理示例到 `examples/error_handling.py`
- [ ] 保存参数配置对比到 `examples/config_comparison.py`

### 学生反馈收集
- [ ] 记录学生提出的问题
- [ ] 收集参数调优实验结果
- [ ] 整理常见错误和解决方案

### 下次课准备
- [ ] 根据学生表现调整下次课难度
- [ ] 准备针对性的复习材料
- [ ] 更新常见问题FAQ