# Lesson 02 黑板演示步骤 - 容器化与依赖服务启动

## 演示环境准备

### 硬件要求
- 投影仪/大屏幕
- 演示电脑（8GB+ RAM）
- 网络连接（用于拉取镜像）

### 软件准备
- Docker Desktop（已启动）
- VS Code（字体大小18+）
- 终端（字体大小18+）
- 浏览器

### 演示目录结构
```
~/demo/lesson02/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── src/
│   ├── main.py
│   └── __init__.py
└── README.md
```

---

## 第一部分：容器化概念讲解 (15分钟)

### 步骤1：绘制传统部署架构图
**在黑板左侧画图**
```
传统部署方式：
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   应用 A    │  │   应用 B    │  │   应用 C    │
├─────────────┤  ├─────────────┤  ├─────────────┤
│  Python 3.8 │  │  Python 3.9 │  │  Python 3.10│
│  依赖包 v1  │  │  依赖包 v2  │  │  依赖包 v3  │
├─────────────┤  ├─────────────┤  ├─────────────┤
│  Ubuntu 18  │  │  Ubuntu 20  │  │  Ubuntu 22  │
├─────────────┤  ├─────────────┤  ├─────────────┤
│  服务器 A   │  │  服务器 B   │  │  服务器 C   │
└─────────────┘  └─────────────┘  └─────────────┘

问题：
❌ 环境不一致
❌ 资源浪费
❌ 部署复杂
❌ 扩展困难
```

### 步骤2：绘制容器化部署架构图
**在黑板右侧画图**
```
容器化部署方式：
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  容器 A     │  │  容器 B     │  │  容器 C     │
│ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │
│ │ 应用+环境│ │  │ │ 应用+环境│ │  │ │ 应用+环境│ │
│ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │
└─────────────┘  └─────────────┘  └─────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                ┌─────────────────┐
                │  Docker Engine  │
                ├─────────────────┤
                │   Host OS       │
                ├─────────────────┤
                │   物理服务器     │
                └─────────────────┘

优势：
✅ 环境一致
✅ 资源高效
✅ 部署简单
✅ 易于扩展
```

### 步骤3：核心概念解释
**在黑板中央写出关键概念**
```
Docker 三大核心概念：

1. 镜像 (Image)
   📦 只读模板
   📦 包含运行环境
   📦 可以分享复用

2. 容器 (Container)  
   🏃 镜像的运行实例
   🏃 独立的进程空间
   🏃 可以启停删除

3. 仓库 (Registry)
   🏪 存储镜像的地方
   🏪 Docker Hub (公共)
   🏪 私有仓库
```

---

## 第二部分：Dockerfile编写演示 (20分钟)

### 步骤4：创建演示项目
**在终端中操作，投屏显示**
```bash
# 创建演示目录
mkdir -p ~/demo/lesson02
cd ~/demo/lesson02

# 创建项目结构
mkdir src
touch Dockerfile requirements.txt
touch src/main.py src/__init__.py
```

### 步骤5：编写requirements.txt
**在VS Code中创建，边写边解释**
```txt
# Web框架
fastapi==0.104.1
uvicorn[standard]==0.24.0

# 数据库相关
psycopg2-binary==2.9.9
sqlalchemy==2.0.23

# 缓存
redis==5.0.1

# 工具库
pydantic==2.5.0
python-multipart==0.0.6
```

### 步骤6：编写简单的FastAPI应用
**在VS Code中创建src/main.py**
```python
from fastapi import FastAPI
import os

app = FastAPI(title="RAG Application", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Hello RAG World!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database_url": os.getenv("DATABASE_URL", "not configured"),
        "redis_url": os.getenv("REDIS_URL", "not configured")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 步骤7：逐步编写Dockerfile
**在黑板上写出Dockerfile结构，然后在VS Code中实现**

**黑板上的结构图**：
```
Dockerfile 构建步骤：
1. 选择基础镜像 ← FROM
2. 设置工作目录 ← WORKDIR  
3. 设置环境变量 ← ENV
4. 安装系统依赖 ← RUN
5. 复制依赖文件 ← COPY
6. 安装Python包 ← RUN pip
7. 复制应用代码 ← COPY
8. 创建用户     ← RUN useradd
9. 暴露端口     ← EXPOSE
10. 启动命令    ← CMD
```

**在VS Code中逐步编写**：
```dockerfile
# 步骤1：选择基础镜像
FROM python:3.11-slim

# 步骤2：设置工作目录
WORKDIR /app

# 步骤3：设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 步骤4：安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 步骤5：复制并安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 步骤6：复制应用代码
COPY src/ ./src/

# 步骤7：创建非root用户
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# 步骤8：暴露端口
EXPOSE 8000

# 步骤9：健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 步骤10：启动命令
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 步骤8：构建和测试镜像
**在终端中演示**
```bash
# 构建镜像
docker build -t rag-app:v1.0 .

# 查看构建的镜像
docker images | grep rag-app

# 运行容器
docker run -d -p 8000:8000 --name rag-demo rag-app:v1.0

# 测试应用
curl http://localhost:8000/
curl http://localhost:8000/health

# 查看容器状态
docker ps

# 查看容器日志
docker logs rag-demo
```

---

## 第三部分：Docker Compose多服务编排 (25分钟)

### 步骤9：绘制服务架构图
**在黑板上绘制RAG系统架构**
```
RAG 系统服务架构：

┌─────────────────────────────────────────────────────┐
│                    用户请求                          │
└─────────────────┬───────────────────────────────────┘
                  │
          ┌───────▼────────┐
          │   FastAPI      │ :8000
          │   应用服务      │
          └───────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼───┐    ┌───▼───┐    ┌───▼───┐
│PostgreSQL│  │ Redis │    │Qdrant │
│ :5432  │  │ :6379 │    │ :6333 │
│数据库   │  │ 缓存  │    │向量库  │
└────────┘  └───────┘    └───────┘
                  │
              ┌───▼───┐
              │ MinIO │ :9000
              │对象存储│
              └───────┘
```

### 步骤10：编写docker-compose.yml
**在黑板上写出配置结构，然后在VS Code中实现**

**黑板上的结构**：
```
docker-compose.yml 结构：
├── version: '3.8'
├── services:
│   ├── app (FastAPI应用)
│   ├── postgres (数据库)
│   ├── redis (缓存)
│   ├── qdrant (向量数据库)
│   └── minio (对象存储)
├── volumes: (数据持久化)
└── networks: (服务通信)
```

**在VS Code中逐步编写**：
```yaml
version: '3.8'

services:
  # FastAPI应用服务
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://rag:ragpass@postgres:5432/rag_db
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    networks:
      - rag-network
    restart: unless-stopped

  # PostgreSQL数据库
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: rag_db
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: ragpass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - rag-network
    restart: unless-stopped

  # Redis缓存
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - rag-network
    restart: unless-stopped

  # Qdrant向量数据库
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    networks:
      - rag-network
    restart: unless-stopped

  # MinIO对象存储
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"
    networks:
      - rag-network
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
  minio_data:

networks:
  rag-network:
    driver: bridge
```

### 步骤11：启动多服务栈
**在终端中演示**
```bash
# 清理之前的单容器
docker stop rag-demo
docker rm rag-demo

# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看服务日志
docker-compose logs -f app
```

### 步骤12：验证服务连通性
**在终端中逐一测试**
```bash
# 测试FastAPI应用
curl http://localhost:8000/health

# 测试PostgreSQL连接
docker-compose exec postgres psql -U rag -d rag_db -c "SELECT version();"

# 测试Redis连接
docker-compose exec redis redis-cli ping

# 测试Qdrant API
curl http://localhost:6333/collections

# 测试MinIO控制台
echo "打开浏览器访问: http://localhost:9001"
echo "用户名: minioadmin"
echo "密码: minioadmin"
```

---

## 第四部分：故障排查演示 (10分钟)

### 步骤13：模拟端口冲突问题
**在黑板上写出排查步骤**
```
故障排查流程：
1. 查看错误信息 ← docker-compose logs
2. 检查端口占用 ← netstat -tulpn
3. 检查服务状态 ← docker-compose ps
4. 进入容器调试 ← docker-compose exec
5. 查看网络连接 ← docker network ls
```

**在终端中演示**
```bash
# 模拟端口冲突
# 启动一个占用8000端口的服务
python -m http.server 8000 &

# 尝试启动docker-compose
docker-compose up app

# 查看错误信息
docker-compose logs app

# 解决方案：修改端口映射
# 在docker-compose.yml中修改端口为8001:8000
```

### 步骤14：网络连通性测试
**在终端中演示**
```bash
# 进入应用容器
docker-compose exec app bash

# 测试网络连通性
ping postgres
ping redis
nslookup postgres

# 查看环境变量
env | grep -E "DATABASE_URL|REDIS_URL"

# 退出容器
exit
```

### 步骤15：数据持久化验证
**在终端中演示**
```bash
# 在数据库中创建测试数据
docker-compose exec postgres psql -U rag -d rag_db -c "
CREATE TABLE test_table (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO test_table (name) VALUES ('test data');
"

# 停止所有服务
docker-compose down

# 重新启动服务
docker-compose up -d

# 验证数据是否还存在
docker-compose exec postgres psql -U rag -d rag_db -c "SELECT * FROM test_table;"
```

---

## 第五部分：Exercise指导 (15分钟)

### 步骤16：Exercise任务分配
**在黑板上写出Exercise任务**

Exercise任务：

基础任务 (必做)：
□ 1. 创建自己的Dockerfile
□ 2. 构建应用镜像
□ 3. 运行单个容器
□ 4. 测试应用访问

进阶任务 (选做)：
□ 5. 编写docker-compose.yml
□ 6. 启动多服务栈
□ 7. 验证服务间连通性
□ 8. 添加健康检查

挑战任务 (高级)：
□ 9. 添加监控服务
□ 10. 配置负载均衡
```

### 步骤17：巡回指导要点
**准备常见问题的解决方案**
```bash
# 常见问题1：构建失败
docker build --no-cache -t rag-app .

# 常见问题2：容器启动失败
docker logs <container_name>

# 常见问题3：端口访问不了
docker port <container_name>
netstat -tulpn | grep 8000

# 常见问题4：服务连接失败
docker-compose exec app ping postgres
```

---

## 第六部分：总结与答疑 (10分钟)

### 步骤18：知识点总结
**在黑板上总结关键知识点**
```
今日学习成果：

技术技能：
✅ Docker镜像构建
✅ 容器运行管理
✅ 多服务编排
✅ 网络配置
✅ 数据持久化

实践能力：
✅ 故障排查
✅ 日志分析
✅ 性能监控
✅ 安全配置

企业技能：
✅ 环境标准化
✅ 部署自动化
✅ 服务治理
✅ 运维监控
```

### 步骤19：课后作业布置
**在黑板上写出作业要求**
```
课后作业：

必做作业：
1. 完善docker-compose.yml配置
   - 添加健康检查
   - 配置资源限制
   - 添加环境变量文件

2. 编写部署文档
   - 环境要求说明
   - 部署步骤详解
   - 故障排查指南

选做作业：
1. 添加监控服务 (Prometheus + Grafana)
2. 研究多阶段构建优化镜像大小
3. 配置CI/CD自动化部署
```

### 步骤20：下节课预告
**在黑板上写出预习内容**
```
下节课预告：数据模型设计

预习内容：
□ 复习SQL基础语法
□ 了解SQLAlchemy ORM
□ 思考RAG系统数据结构
□ 准备PostgreSQL环境

知识连接：
今天搭建的PostgreSQL → 下节课的数据模型
今天的容器化环境 → 下节课的开发环境
```

---

## 演示注意事项

### 时间控制
- 每个步骤严格按时间执行
- 预留5分钟缓冲时间
- 复杂操作提前准备备用方案

### 互动技巧
- 每10分钟提问一次
- 鼓励学生提出问题
- 及时回应学生反馈

### 技术准备
- 提前拉取所有Docker镜像
- 准备网络问题的备用方案
- 测试所有演示命令

### 应急预案
- 网络断开：使用本地镜像
- 系统卡顿：切换到备用机器
- 投屏问题：准备纸质材料