# Lesson 01: 课程启动 & 项目脚手架

## 课程目标

- 理解RAG系统的整体架构
- 掌握uv包管理工具的使用
- 搭建FastAPI项目脚手架
- 实现健康检查接口

## 理论知识

### RAG系统架构概述

RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI架构模式：

1. **检索阶段**: 从知识库中检索相关信息
2. **增强阶段**: 将检索到的信息作为上下文
3. **生成阶段**: 基于上下文生成最终答案

### 技术栈选择

- **uv**: 现代Python包管理器，比pip快10-100倍
- **FastAPI**: 现代、快速的Web框架，自动生成OpenAPI文档
- **SQLModel**: 类型安全的ORM，结合SQLAlchemy和Pydantic
- **Qdrant**: 高性能向量数据库
- **PostgreSQL**: 可靠的关系型数据库

## 实践操作

### 1. 环境准备

```bash
# 安装uv (如果还没有安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 验证安装
uv --version
```

### 2. 项目初始化

```bash
# 创建项目
uv init rag-course
cd rag-course

# 设置Python版本
uv python pin 3.11
```

### 3. 添加基础依赖

```bash
# Web框架
uv add fastapi uvicorn[standard]

# 配置管理
uv add pydantic-settings python-dotenv

# 日志
uv add loguru

# 开发工具
uv add --dev ruff pytest httpx
```

### 4. 项目结构

创建以下目录结构：

```
rag-course/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── logging.py
│   └── api/
│       ├── __init__.py
│       └── health.py
├── tests/
│   ├── __init__.py
│   └── test_health.py
├── .env.example
├── .gitignore
└── README.md
```

## 验收标准

1. ✅ 项目结构完整
2. ✅ 依赖安装成功
3. ✅ FastAPI应用可以启动
4. ✅ 健康检查接口返回正确响应
5. ✅ 测试用例通过

## 作业

1. 完成项目脚手架搭建
2. 实现健康检查接口
3. 编写基础测试用例
4. 配置开发环境

## 下节预告

下节课将学习如何使用Docker Compose启动RAG系统的依赖服务（PostgreSQL、Qdrant、MinIO）。