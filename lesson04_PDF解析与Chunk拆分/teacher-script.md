# Lesson 04 教师讲稿与授课脚本

## 时间轴与脚本安排
- **导入阶段**：10分钟
- **理论讲解**：30分钟  
- **演示实践**：35分钟
- **Exercise练习**：10分钟
- **总结答疑**：5分钟

---

## 开场导入（10分钟）

### 教师开场白（可直接照读）

"同学们好！今天我们进入第四课：PDF解析与Chunk拆分。这是我们RAG系统构建的关键一步。

**今天的目标与产物**：
- 核心成果：掌握PyMuPDF进行PDF文档解析
- 提交物：完整的文档处理模块和Chunk拆分算法
- 质量标准：文档解析成功率≥95%，Chunk质量评分≥85分

**为什么这节课很重要？**
前面三课我们搭建了环境、启动了服务、建立了数据模型。现在我们要处理真正的数据了！PDF文档是企业中最常见的知识载体，如何高质量地解析和拆分，直接决定了后续检索和生成的效果。

让我先给大家看一个对比：同样的PDF文档，不同的处理方式会产生什么样的差异..."

### 核心概念预告
- **PyMuPDF**：高性能PDF处理库，基于C++实现
- **Chunk拆分**：将长文档分割为适合向量化的语义单元
- **语义边界**：保持文本语义完整性的分割点
- **重叠策略**：相邻Chunk间的内容重叠，提高检索召回率

---

## 理论讲解阶段（30分钟）

### 1. PDF文档结构与解析原理（10分钟）

#### 教师讲稿
"首先我们来理解PDF文档的内部结构。PDF不是简单的文本文件，它是一个复杂的文档格式..."

#### 核心知识点
```python
# PDF文档的层次结构
pdf_structure = {
    "文档级别": {
        "元数据": "标题、作者、创建时间等",
        "页面集合": "多个页面对象的容器"
    },
    "页面级别": {
        "页面属性": "尺寸、旋转角度、媒体框",
        "内容流": "文本、图像、图形的绘制指令"
    },
    "内容级别": {
        "文本对象": "字符、字体、位置信息",
        "图像对象": "图片数据和位置",
        "图形对象": "线条、形状、颜色"
    }
}
```

#### 演示要点
- 打开一个PDF文件，展示其内部结构
- 对比不同PDF的复杂度差异
- 解释为什么需要专业的解析库

### 2. PyMuPDF核心API详解（10分钟）

#### 教师讲稿
"PyMuPDF是目前最强大的PDF处理库之一。它的Python接口叫做fitz，这个名字来源于..."

#### 核心API演示
```python
import fitz  # PyMuPDF的Python接口

# 基础操作流程
def demonstrate_pymupdf_basics():
    # 1. 打开文档
    doc = fitz.open("sample.pdf")
    print(f"文档页数: {doc.page_count}")
    print(f"文档元数据: {doc.metadata}")
    
    # 2. 遍历页面
    for page_num in range(doc.page_count):
        page = doc[page_num]
        
        # 3. 提取文本
        text = page.get_text()
        
        # 4. 获取文本块（保留位置信息）
        blocks = page.get_text("dict")
        
        # 5. 提取图片
        images = page.get_images()
    
    # 6. 关闭文档
    doc.close()
```

#### 关键参数说明
- `get_text()`：简单文本提取
- `get_text("dict")`：结构化文本提取，包含位置信息
- `get_text("blocks")`：按文本块提取
- `get_images()`：提取页面中的图片

### 3. Chunk拆分策略设计（10分钟）

#### 教师讲稿
"文档解析出来后，我们需要将长文档拆分成适合向量化的小块。这个过程叫做Chunking。好的Chunk拆分策略需要平衡三个因素..."

#### 拆分策略对比
```python
# 不同拆分策略的特点
chunking_strategies = {
    "固定长度拆分": {
        "优点": ["实现简单", "处理速度快", "内存占用可控"],
        "缺点": ["可能破坏语义", "上下文不完整"],
        "适用场景": "简单文档、快速原型"
    },
    "语义边界拆分": {
        "优点": ["保持语义完整", "上下文连贯"],
        "缺点": ["实现复杂", "长度不均匀"],
        "适用场景": "高质量要求、复杂文档"
    },
    "滑动窗口拆分": {
        "优点": ["信息覆盖全面", "检索召回率高"],
        "缺点": ["存储空间大", "可能有重复"],
        "适用场景": "检索精度要求高"
    }
}
```

#### 重叠策略设计
- **重叠比例**：通常设置为10-20%
- **重叠内容**：句子级别的重叠，避免词汇截断
- **边界处理**：在标点符号处进行重叠切分

---

## 演示实践阶段（35分钟）

### 1. PyMuPDF基础操作演示（15分钟）

#### 现场演示脚本
"现在我来演示如何使用PyMuPDF解析PDF文档。请大家跟着我的操作..."

```python
# 演示代码 - 基础PDF解析
import fitz
import json
from pathlib import Path

def demo_pdf_parsing():
    # 打开示例PDF
    pdf_path = "examples/sample_document.pdf"
    doc = fitz.open(pdf_path)
    
    print("=== 文档基本信息 ===")
    print(f"页数: {doc.page_count}")
    print(f"标题: {doc.metadata.get('title', '未知')}")
    print(f"作者: {doc.metadata.get('author', '未知')}")
    
    # 提取第一页内容
    page = doc[0]
    print("\n=== 第一页文本内容 ===")
    text = page.get_text()
    print(text[:500] + "..." if len(text) > 500 else text)
    
    # 获取结构化信息
    print("\n=== 文本块信息 ===")
    blocks = page.get_text("dict")
    for block in blocks["blocks"][:3]:  # 只显示前3个块
        if "lines" in block:
            print(f"块类型: 文本块, 行数: {len(block['lines'])}")
    
    doc.close()

# 现场执行演示
demo_pdf_parsing()
```

#### 演示重点
- 展示不同PDF文档的解析结果差异
- 对比简单文本提取和结构化提取的区别
- 处理常见的解析异常情况

### 2. Chunk拆分算法实现（20分钟）

#### 现场编码演示
"接下来我们实现一个智能的Chunk拆分算法。这个算法需要考虑语义边界和长度控制..."

```python
# 演示代码 - 智能Chunk拆分
import re
from typing import List, Dict

class SmartChunker:
    def __init__(self, max_chunk_size: int = 1000, overlap_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
        # 语义边界标识符（按优先级排序）
        self.boundary_patterns = [
            r'\n\n+',          # 段落边界（最高优先级）
            r'[。！？]\s*\n',   # 句子边界
            r'[。！？]\s+',     # 句子内边界
            r'[，；]\s+',       # 子句边界
            r'\s+',            # 词汇边界（最低优先级）
        ]
    
    def split_text(self, text: str) -> List[Dict]:
        """智能文本拆分"""
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # 确定当前chunk的结束位置
            end_pos = self._find_optimal_split_point(
                text, current_pos, current_pos + self.max_chunk_size
            )
            
            # 提取chunk内容
            chunk_text = text[current_pos:end_pos].strip()
            
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'start_pos': current_pos,
                    'end_pos': end_pos,
                    'length': len(chunk_text)
                })
            
            # 计算下一个chunk的起始位置（考虑重叠）
            current_pos = max(end_pos - self.overlap_size, end_pos)
        
        return chunks
    
    def _find_optimal_split_point(self, text: str, start: int, max_end: int) -> int:
        """寻找最优的分割点"""
        if max_end >= len(text):
            return len(text)
        
        # 在最大长度范围内寻找最佳边界
        search_text = text[start:max_end]
        
        for pattern in self.boundary_patterns:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                # 选择最接近最大长度的边界点
                best_match = matches[-1]
                return start + best_match.end()
        
        # 如果没有找到合适的边界，在最大长度处强制分割
        return max_end

# 现场演示
def demo_chunking():
    chunker = SmartChunker(max_chunk_size=500, overlap_size=50)
    
    # 使用前面解析的PDF文本
    sample_text = """
    人工智能（Artificial Intelligence，AI）是计算机科学的一个分支。
    它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    自诞生以来，理论和技术日益成熟，应用领域也不断扩大。
    
    可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
    """
    
    chunks = chunker.split_text(sample_text)
    
    print("=== Chunk拆分结果 ===")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(f"  长度: {chunk['length']}")
        print(f"  内容: {chunk['text'][:100]}...")
        print()

# 执行演示
demo_chunking()
```

#### 演示要点
- 解释每个拆分策略的选择原因
- 展示不同参数设置对结果的影响
- 处理边界情况和异常处理

---

## Exercise练习阶段（10分钟）

### 课堂Exercise任务
"现在请大家完成一个小Exercise：使用我们刚才学习的方法，解析提供的PDF文档并进行Chunk拆分。"

#### Exercise要求
1. 使用PyMuPDF解析`examples/exercise_document.pdf`
2. 实现基础的Chunk拆分功能
3. 输出拆分结果的统计信息

#### Exercise代码模板
```python
# Exercise: PDF解析与Chunk拆分
def exercise_pdf_chunking():
    # TODO: 1. 打开PDF文档
    # TODO: 2. 提取所有页面的文本
    # TODO: 3. 使用SmartChunker进行拆分
    # TODO: 4. 输出统计信息
    pass

# 请在这里完成你的实现
```

#### 预期输出示例
```
=== PDF解析结果 ===
文档页数: 5
总文本长度: 12,450字符

=== Chunk拆分结果 ===
总Chunk数: 18
平均长度: 692字符
最大长度: 998字符
最小长度: 234字符
```

---

## 总结答疑阶段（5分钟）

### 课程总结（可直接照读）
"今天我们完成了RAG系统中非常重要的一个环节：文档解析与Chunk拆分。

**核心收获**：
1. 掌握了PyMuPDF的核心API和使用方法
2. 理解了不同Chunk拆分策略的优缺点
3. 实现了智能的语义边界识别算法
4. 学会了处理PDF解析中的常见问题

**下节课预告**：
下节课我们将学习Embedding与向量入库，把今天产出的Chunk数据转换为向量表示，存储到向量数据库中。

**课后任务提醒**：
请大家完成课后作业，优化Chunk拆分算法，处理更复杂的文档结构。记住，文档处理的质量直接影响整个RAG系统的效果！"

### 常见问题预设
1. **Q**: PyMuPDF处理加密PDF怎么办？
   **A**: 需要提供密码参数，或者使用其他工具先解密。

2. **Q**: 中文文档的分词边界如何处理？
   **A**: 可以集成jieba分词库，在词汇边界处进行拆分。

3. **Q**: 如何处理表格和图片混合的复杂文档？
   **A**: 这是高级话题，需要结合OCR和表格识别技术，超出本课范围。

### 提问与互动环节
- 鼓励学生提问遇到的具体问题
- 针对Exercise中的难点进行答疑
- 预告下节课的内容和准备事项