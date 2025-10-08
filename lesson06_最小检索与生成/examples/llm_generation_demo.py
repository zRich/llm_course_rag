#!/usr/bin/env python3
"""
Lesson 06 - LLM生成演示
演示大语言模型调用和Prompt工程的基本实现
"""

import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class GenerationConfig:
    """生成配置"""
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@dataclass
class GenerationResult:
    """生成结果"""
    text: str
    tokens_used: int
    generation_time: float
    model_name: str


class LLMProvider(ABC):
    """LLM提供者抽象基类"""
    
    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """生成文本"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """获取模型名称"""
        pass


class MockLLMProvider(LLMProvider):
    """模拟LLM提供者（用于演示）"""
    
    def __init__(self, model_name: str = "mock-gpt-3.5"):
        self.model_name = model_name
    
    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """模拟文本生成"""
        start_time = time.time()
        
        # 模拟生成延迟
        time.sleep(0.1)
        
        # 基于prompt生成模拟回答
        if "什么是" in prompt or "介绍" in prompt:
            response = self._generate_definition_response(prompt)
        elif "如何" in prompt or "怎么" in prompt:
            response = self._generate_howto_response(prompt)
        elif "比较" in prompt or "区别" in prompt:
            response = self._generate_comparison_response(prompt)
        else:
            response = self._generate_general_response(prompt)
        
        generation_time = time.time() - start_time
        
        return GenerationResult(
            text=response,
            tokens_used=len(response.split()),
            generation_time=generation_time,
            model_name=self.model_name
        )
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def _generate_definition_response(self, prompt: str) -> str:
        return """基于提供的参考文档，我可以为您解释相关概念。

这个概念涉及多个方面：
1. 基本定义和核心特征
2. 主要应用场景和用途
3. 与其他相关概念的关系
4. 实际应用中的注意事项

根据文档内容，这是一个重要的技术概念，在实际应用中具有广泛的价值。"""
    
    def _generate_howto_response(self, prompt: str) -> str:
        return """根据参考文档，我为您提供以下步骤指导：

**实施步骤：**
1. 准备阶段：确保环境和依赖项就绪
2. 配置阶段：设置必要的参数和选项
3. 执行阶段：按照最佳实践进行操作
4. 验证阶段：检查结果并进行必要调整

**注意事项：**
- 遵循文档中提到的最佳实践
- 注意潜在的风险点和解决方案
- 建议在测试环境中先行验证"""
    
    def _generate_comparison_response(self, prompt: str) -> str:
        return """基于参考文档，我为您分析两者的主要区别：

**相似点：**
- 都属于同一技术领域
- 具有相似的应用场景
- 遵循类似的基本原理

**主要区别：**
1. **技术实现**：在底层实现机制上有所不同
2. **性能特征**：在效率和资源消耗方面各有优势
3. **适用场景**：针对不同的使用场景有不同的优化
4. **学习成本**：在掌握难度和学习曲线上存在差异

建议根据具体需求选择最适合的方案。"""
    
    def _generate_general_response(self, prompt: str) -> str:
        return """根据提供的参考文档，我理解您的问题并提供以下回答：

这是一个很好的问题，涉及到多个重要方面。基于文档内容，我可以从以下角度来分析：

1. **核心要点**：文档中强调的关键信息
2. **实践建议**：基于最佳实践的具体建议
3. **注意事项**：需要特别关注的重要细节

如果您需要更具体的信息，建议参考文档中的详细说明部分。"""


class OpenAIProvider(LLMProvider):
    """OpenAI API提供者"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("需要设置OPENAI_API_KEY环境变量")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("请安装openai库: pip install openai")
    
    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """使用OpenAI API生成文本"""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty
            )
            
            generation_time = time.time() - start_time
            
            return GenerationResult(
                text=response.choices[0].message.content,
                tokens_used=response.usage.total_tokens,
                generation_time=generation_time,
                model_name=self.model
            )
        
        except Exception as e:
            raise RuntimeError(f"OpenAI API调用失败: {e}")
    
    def get_model_name(self) -> str:
        return self.model


class PromptTemplate:
    """Prompt模板类"""
    
    def __init__(self, template: str):
        self.template = template
    
    def format(self, **kwargs) -> str:
        """格式化模板"""
        return self.template.format(**kwargs)


class RAGPromptTemplates:
    """RAG专用Prompt模板集合"""
    
    # 基础问答模板
    BASIC_QA = PromptTemplate("""你是一个专业的AI助手，请根据以下参考文档回答用户问题。

参考文档：
{context}

用户问题：{question}

请注意：
1. 只基于提供的参考文档回答问题
2. 如果文档中没有相关信息，请明确说明
3. 回答要准确、简洁、有条理
4. 可以引用具体的文档片段支持你的回答

回答：""")
    
    # 带引用的问答模板
    QA_WITH_CITATION = PromptTemplate("""你是一个专业的AI助手，请根据参考文档回答问题，并提供引用。

参考文档：
{context}

用户问题：{question}

要求：
1. 基于参考文档提供准确回答
2. 在回答中标注引用来源 [文档X]
3. 如果信息不足，请明确说明
4. 保持回答的逻辑性和条理性

回答：""")
    
    # 多步推理模板
    CHAIN_OF_THOUGHT = PromptTemplate("""你是一个专业的AI助手，请根据参考文档回答问题，并展示推理过程。

参考文档：
{context}

用户问题：{question}

请按以下步骤回答：
1. **理解问题**：分析用户问题的核心要点
2. **查找信息**：从参考文档中找到相关信息
3. **推理分析**：基于文档信息进行逻辑推理
4. **得出结论**：提供最终答案

回答：""")
    
    # 对比分析模板
    COMPARISON = PromptTemplate("""你是一个专业的AI助手，请根据参考文档进行对比分析。

参考文档：
{context}

对比问题：{question}

请从以下角度进行分析：
1. **相似点**：找出共同特征和属性
2. **差异点**：分析主要区别和不同
3. **适用场景**：说明各自的最佳应用场景
4. **选择建议**：基于不同需求提供选择建议

分析：""")


class LLMGenerator:
    """LLM生成器"""
    
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.default_config = GenerationConfig()
    
    def generate(self, prompt: str, config: GenerationConfig = None) -> GenerationResult:
        """生成文本"""
        config = config or self.default_config
        return self.provider.generate(prompt, config)
    
    def generate_with_template(self, template: PromptTemplate, 
                             config: GenerationConfig = None, **kwargs) -> GenerationResult:
        """使用模板生成文本"""
        prompt = template.format(**kwargs)
        return self.generate(prompt, config)


def demo_basic_generation():
    """演示基础文本生成"""
    print("=== 基础文本生成演示 ===")
    
    # 创建模拟LLM提供者
    provider = MockLLMProvider()
    generator = LLMGenerator(provider)
    
    # 测试不同类型的问题
    test_prompts = [
        "什么是RAG系统？请详细介绍。",
        "如何实现向量检索？",
        "比较余弦相似度和欧几里得距离的区别。"
    ]
    
    for prompt in test_prompts:
        print(f"\n问题: {prompt}")
        print("-" * 50)
        
        result = generator.generate(prompt)
        
        print(f"回答: {result.text}")
        print(f"模型: {result.model_name}")
        print(f"Token数: {result.tokens_used}")
        print(f"生成时间: {result.generation_time:.2f}秒")


def demo_prompt_templates():
    """演示Prompt模板使用"""
    print("\n=== Prompt模板演示 ===")
    
    provider = MockLLMProvider()
    generator = LLMGenerator(provider)
    
    # 模拟检索到的文档
    context = """
    文档1: RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI架构，通过外部知识库增强生成质量。
    
    文档2: 向量检索使用向量相似度来找到最相关的文档片段，常用的相似度计算方法包括余弦相似度、点积相似度等。
    
    文档3: Qdrant是一个高性能的向量数据库，专门用于存储和检索高维向量数据，支持多种相似度计算方法。
    """
    
    question = "RAG系统是如何工作的？"
    
    # 测试不同模板
    templates = [
        ("基础问答", RAGPromptTemplates.BASIC_QA),
        ("带引用问答", RAGPromptTemplates.QA_WITH_CITATION),
        ("多步推理", RAGPromptTemplates.CHAIN_OF_THOUGHT)
    ]
    
    for template_name, template in templates:
        print(f"\n--- {template_name} ---")
        
        result = generator.generate_with_template(
            template,
            context=context,
            question=question
        )
        
        print(f"回答: {result.text[:200]}...")
        print(f"Token数: {result.tokens_used}")


def demo_generation_configs():
    """演示不同生成配置的效果"""
    print("\n=== 生成配置演示 ===")
    
    provider = MockLLMProvider()
    generator = LLMGenerator(provider)
    
    prompt = "请解释机器学习的基本概念。"
    
    # 测试不同配置
    configs = [
        ("保守配置", GenerationConfig(temperature=0.1, max_tokens=200)),
        ("平衡配置", GenerationConfig(temperature=0.7, max_tokens=500)),
        ("创意配置", GenerationConfig(temperature=1.2, max_tokens=800))
    ]
    
    for config_name, config in configs:
        print(f"\n--- {config_name} ---")
        print(f"Temperature: {config.temperature}, Max Tokens: {config.max_tokens}")
        
        result = generator.generate(prompt, config)
        
        print(f"回答: {result.text[:150]}...")
        print(f"实际Token数: {result.tokens_used}")


def demo_openai_integration():
    """演示OpenAI API集成（需要API密钥）"""
    print("\n=== OpenAI API集成演示 ===")
    
    try:
        # 检查是否有API密钥
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("未设置OPENAI_API_KEY环境变量，跳过OpenAI演示")
            return
        
        # 创建OpenAI提供者
        provider = OpenAIProvider(model="gpt-3.5-turbo")
        generator = LLMGenerator(provider)
        
        # 测试实际API调用
        context = "RAG系统结合了检索和生成技术，能够基于外部知识库生成更准确的回答。"
        question = "RAG系统有什么优势？"
        
        result = generator.generate_with_template(
            RAGPromptTemplates.BASIC_QA,
            context=context,
            question=question
        )
        
        print(f"问题: {question}")
        print(f"回答: {result.text}")
        print(f"模型: {result.model_name}")
        print(f"Token使用: {result.tokens_used}")
        print(f"生成时间: {result.generation_time:.2f}秒")
    
    except Exception as e:
        print(f"OpenAI API演示失败: {e}")


if __name__ == "__main__":
    print("Lesson 06 - LLM生成演示")
    print("=" * 50)
    
    # 运行演示
    demo_basic_generation()
    demo_prompt_templates()
    demo_generation_configs()
    demo_openai_integration()
    
    print("\n演示完成！")
    print("\n关键要点:")
    print("1. Prompt模板能够标准化和优化LLM交互")
    print("2. 生成配置参数直接影响输出质量和风格")
    print("3. 不同的模板适用于不同类型的任务")
    print("4. 实际部署时需要考虑API成本和延迟")