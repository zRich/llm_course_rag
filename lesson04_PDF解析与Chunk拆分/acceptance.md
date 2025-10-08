# 第4课：PDF解析与Chunk拆分 - 提交与验收标准

## 验收目标

### 核心目标
学生能够独立实现一个健壮的PDF解析与文档分块系统，具备处理各种PDF格式的能力，并能根据不同场景选择合适的分块策略。

### 验收原则
1. **功能完整性**：所有核心功能正常工作
2. **代码质量**：遵循编程规范，具备良好的可读性和可维护性
3. **错误处理**：具备完善的异常处理机制
4. **性能要求**：满足基本的性能指标
5. **文档完整**：提供清晰的使用说明和API文档

## 核心功能验收标准

### 1. PDF文本提取功能

#### 基础要求 (必须满足)
- **功能描述**：能够从PDF文件中提取文本内容
- **输入格式**：支持标准PDF文件 (.pdf)
- **输出格式**：纯文本字符串
- **测试用例**：
  ```python
  def test_pdf_extraction():
      processor = PDFProcessor()
      text = processor.extract_text("sample.pdf")
      assert isinstance(text, str)
      assert len(text) > 0
      assert "预期关键词" in text
  ```

#### 高级要求 (加分项)
- **多页处理**：正确处理多页PDF文档
- **格式保持**：保留基本的段落结构
- **特殊字符**：正确处理中文、特殊符号等
- **测试标准**：
  ```python
  def test_advanced_extraction():
      text = processor.extract_text("complex.pdf")
      # 检查段落结构
      paragraphs = text.split('\n\n')
      assert len(paragraphs) > 1
      # 检查中文支持
      assert any('中文' in p for p in paragraphs)
  ```

### 2. 文档分块功能

#### 固定长度分块 (必须实现)
- **功能要求**：按指定长度分割文档
- **参数支持**：chunk_size, overlap
- **验收标准**：
  ```python
  def test_fixed_length_chunking():
      chunker = DocumentChunker(chunk_size=1000, overlap=100)
      chunks = chunker.split_text(long_text)
      
      # 检查块数量合理
      expected_chunks = len(long_text) // 900  # 考虑重叠
      assert abs(len(chunks) - expected_chunks) <= 2
      
      # 检查块大小
      for chunk in chunks[:-1]:  # 最后一块可能较短
          assert 900 <= len(chunk) <= 1100  # 允许10%误差
      
      # 检查重叠
      if len(chunks) > 1:
          overlap_text = find_overlap(chunks[0], chunks[1])
          assert 80 <= len(overlap_text) <= 120
  ```

#### 语义分块 (推荐实现)
- **功能要求**：按语义边界分割文档
- **边界识别**：句子、段落、章节
- **验收标准**：
  ```python
  def test_semantic_chunking():
      chunker = SemanticChunker(max_chunk_size=1000)
      chunks = chunker.split_text(structured_text)
      
      # 检查语义完整性
      for chunk in chunks:
          # 不应该在句子中间断开
          assert not chunk.endswith(' ')
          # 应该以句号或段落结束
          assert chunk.endswith(('.', '。', '\n'))
  ```

### 3. 错误处理与异常管理

#### 必须处理的异常类型
1. **文件不存在**：FileNotFoundError
2. **PDF格式错误**：PDFReadError
3. **编码问题**：UnicodeDecodeError
4. **内存不足**：MemoryError

#### 验收测试
```python
def test_error_handling():
    processor = PDFProcessor()
    
    # 测试文件不存在
    with pytest.raises(FileNotFoundError):
        processor.extract_text("nonexistent.pdf")
    
    # 测试损坏的PDF
    result = processor.extract_text("corrupted.pdf")
    assert result == "" or "ERROR" in result
    
    # 测试空文件
    result = processor.extract_text("empty.pdf")
    assert isinstance(result, str)
```

## 性能验收标准

### 处理速度要求
- **小文件** (< 1MB)：处理时间 < 2秒
- **中等文件** (1-10MB)：处理时间 < 10秒
- **大文件** (10-50MB)：处理时间 < 60秒

### 内存使用要求
- **内存峰值**：不超过文件大小的5倍
- **内存泄漏**：处理完成后内存应释放

### 性能测试用例
```python
import time
import psutil
import os

def test_performance():
    processor = PDFProcessor()
    process = psutil.Process(os.getpid())
    
    # 记录初始内存
    initial_memory = process.memory_info().rss
    
    # 性能测试
    start_time = time.time()
    text = processor.extract_text("medium_file.pdf")  # 5MB文件
    end_time = time.time()
    
    # 检查处理时间
    processing_time = end_time - start_time
    assert processing_time < 10, f"处理时间过长: {processing_time}秒"
    
    # 检查内存使用
    peak_memory = process.memory_info().rss
    memory_increase = peak_memory - initial_memory
    file_size = os.path.getsize("medium_file.pdf")
    assert memory_increase < file_size * 5, "内存使用过多"
```

## 代码质量验收标准

### 代码结构要求
1. **模块化设计**：功能分离，职责单一
2. **接口设计**：清晰的API接口
3. **配置管理**：支持参数配置
4. **日志记录**：关键操作有日志

### 代码规范检查
```python
# 示例：良好的代码结构
class PDFProcessor:
    """PDF文档处理器
    
    提供PDF文本提取和预处理功能
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化处理器
        
        Args:
            config: 配置参数字典
        """
        self.config = config or {}
        self.logger = self._setup_logger()
    
    def extract_text(self, pdf_path: str) -> str:
        """提取PDF文本
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            提取的文本内容
            
        Raises:
            FileNotFoundError: 文件不存在
            PDFReadError: PDF读取失败
        """
        try:
            self.logger.info(f"开始处理PDF: {pdf_path}")
            # 实现逻辑
            return extracted_text
        except Exception as e:
            self.logger.error(f"PDF处理失败: {e}")
            raise
```

### 文档要求
1. **类和函数文档**：完整的docstring
2. **类型注解**：参数和返回值类型
3. **使用示例**：README中的使用说明
4. **API文档**：自动生成的API文档

## 测试验收标准

### 单元测试要求
- **测试覆盖率**：≥ 80%
- **测试用例数量**：每个核心功能至少3个测试用例
- **边界测试**：包含边界条件和异常情况

### 集成测试要求
- **端到端测试**：完整的处理流程测试
- **多格式测试**：不同类型的PDF文件测试
- **性能测试**：大文件处理测试

### 测试示例
```python
import pytest
from pdf_processor import PDFProcessor, DocumentChunker

class TestPDFProcessor:
    def setup_method(self):
        self.processor = PDFProcessor()
    
    def test_extract_simple_pdf(self):
        """测试简单PDF提取"""
        text = self.processor.extract_text("simple.pdf")
        assert len(text) > 100
        assert "测试内容" in text
    
    def test_extract_empty_pdf(self):
        """测试空PDF处理"""
        text = self.processor.extract_text("empty.pdf")
        assert text == ""
    
    def test_extract_nonexistent_file(self):
        """测试不存在文件"""
        with pytest.raises(FileNotFoundError):
            self.processor.extract_text("nonexistent.pdf")

class TestDocumentChunker:
    def setup_method(self):
        self.chunker = DocumentChunker(chunk_size=1000, overlap=100)
    
    def test_fixed_length_chunking(self):
        """测试固定长度分块"""
        text = "这是一个很长的文本..." * 100
        chunks = self.chunker.split_text(text)
        
        assert len(chunks) > 1
        for chunk in chunks[:-1]:
            assert 900 <= len(chunk) <= 1100
    
    def test_overlap_functionality(self):
        """测试重叠功能"""
        text = "句子一。句子二。句子三。" * 50
        chunks = self.chunker.split_text(text)
        
        if len(chunks) > 1:
            # 检查相邻块之间的重叠
            overlap = self._find_overlap(chunks[0], chunks[1])
            assert len(overlap) >= 50  # 至少50字符重叠
```

## 提交物检查清单

### 必须提交的文件
- [ ] `pdf_processor.py` - 主要实现文件
- [ ] `chunker.py` - 分块功能实现
- [ ] `config.py` - 配置管理
- [ ] `requirements.txt` - 依赖列表
- [ ] `README.md` - 使用说明
- [ ] `tests/` - 测试文件目录
- [ ] `examples/` - 示例代码

### 代码质量检查
- [ ] 所有函数都有类型注解
- [ ] 所有类和函数都有docstring
- [ ] 代码遵循PEP8规范
- [ ] 没有硬编码的路径或参数
- [ ] 异常处理完善
- [ ] 日志记录合理

### 功能完整性检查
- [ ] PDF文本提取功能正常
- [ ] 固定长度分块功能正常
- [ ] 重叠功能正常工作
- [ ] 错误处理机制完善
- [ ] 配置参数可调整
- [ ] 性能满足要求

### 测试完整性检查
- [ ] 单元测试覆盖率 ≥ 80%
- [ ] 所有测试用例通过
- [ ] 包含边界条件测试
- [ ] 包含异常情况测试
- [ ] 性能测试通过

## 评分标准

### 基础功能 (60分)
- PDF文本提取 (20分)
  - 基本提取功能 (15分)
  - 多页处理 (3分)
  - 特殊字符处理 (2分)
- 文档分块 (25分)
  - 固定长度分块 (15分)
  - 重叠功能 (5分)
  - 参数配置 (5分)
- 错误处理 (15分)
  - 异常捕获 (8分)
  - 错误信息 (4分)
  - 降级策略 (3分)

### 代码质量 (25分)
- 代码结构 (10分)
  - 模块化设计 (5分)
  - 接口设计 (3分)
  - 配置管理 (2分)
- 代码规范 (8分)
  - PEP8规范 (3分)
  - 类型注解 (3分)
  - 文档字符串 (2分)
- 日志和调试 (7分)
  - 日志记录 (4分)
  - 调试信息 (3分)

### 测试和文档 (15分)
- 单元测试 (8分)
  - 测试覆盖率 (4分)
  - 测试质量 (4分)
- 文档完整性 (7分)
  - README文档 (3分)
  - API文档 (2分)
  - 使用示例 (2分)

## 常见问题和解决方案

### Q1: PDF解析出现乱码
**问题描述**：提取的文本包含乱码或特殊字符
**检查点**：
- 编码检测和处理
- 字体映射问题
- OCR备用方案

**解决方案**：
```python
def extract_text_robust(pdf_path):
    try:
        # 主要方法
        text = extract_with_pypdf2(pdf_path)
        if is_garbled(text):
            raise EncodingError("文本乱码")
        return text
    except (EncodingError, UnicodeDecodeError):
        # 备用方法
        return extract_with_ocr(pdf_path)
```

### Q2: 大文件处理内存溢出
**问题描述**：处理大型PDF文件时内存不足
**检查点**：
- 内存使用监控
- 流式处理实现
- 资源释放机制

**解决方案**：
```python
def process_large_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text_chunks = []
        
        for page_num in range(len(reader.pages)):
            page_text = reader.pages[page_num].extract_text()
            text_chunks.append(page_text)
            
            # 定期清理内存
            if page_num % 10 == 0:
                gc.collect()
        
        return '\n'.join(text_chunks)
```

### Q3: 分块边界不合理
**问题描述**：分块在句子中间断开，影响语义
**检查点**：
- 边界检测逻辑
- 分隔符优先级
- 最小块大小设置

**解决方案**：
```python
def smart_split(text, chunk_size, separators):
    chunks = []
    current_chunk = ""
    
    for separator in separators:
        if len(current_chunk) >= chunk_size:
            # 在合适的分隔符处断开
            split_point = current_chunk.rfind(separator, 0, chunk_size)
            if split_point > chunk_size * 0.5:  # 至少保留一半内容
                chunks.append(current_chunk[:split_point])
                current_chunk = current_chunk[split_point:]
    
    return chunks
```

## 提交前最终确认

### 功能验证
- [ ] 使用提供的测试PDF文件验证所有功能
- [ ] 确认错误处理机制正常工作
- [ ] 验证性能指标满足要求

### 代码检查
- [ ] 运行代码质量检查工具 (flake8, mypy)
- [ ] 确认所有测试用例通过
- [ ] 检查文档完整性

### 提交准备
- [ ] 清理临时文件和调试代码
- [ ] 更新README和版本信息
- [ ] 准备演示用的示例文件

### 最终测试
```bash
# 运行完整测试套件
python -m pytest tests/ -v --cov=src --cov-report=html

# 代码质量检查
flake8 src/
mypy src/

# 性能测试
python tests/performance_test.py
```

通过以上验收标准的检查，确保提交的PDF解析与Chunk拆分系统具备企业级应用的质量和可靠性。