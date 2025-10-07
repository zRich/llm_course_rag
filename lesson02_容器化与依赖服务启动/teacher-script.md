# Lesson 02 教师讲稿 - 容器化与依赖服务启动

## 课前准备检查清单 (5分钟)

### 环境检查
```bash
# 检查Docker Desktop是否运行
docker --version
docker-compose --version

# 检查系统资源
docker system info | grep -E "CPUs|Total Memory"

# 准备演示项目
cd ~/demo && mkdir exercise && cd exercise
```

### 投屏准备
- [ ] Docker Desktop界面
- [ ] 终端窗口（字体大小18+）
- [ ] VS Code编辑器
- [ ] 浏览器（localhost测试页面）

---

## 开场导入 (10分钟)

### 开场白（可直接照读）
"同学们好！今天我们进入第二课：容器化与依赖服务启动。

通过上节课，我们已经有了一个基本的FastAPI应用。但是在实际的企业开发中，我们的应用不是孤立存在的，它需要数据库、缓存、文件存储等各种服务的支持。

今天我们要解决的核心问题是：**如何在不同的开发环境中，快速、一致地启动这些复杂的服务栈？**"

### 问题引入
**提问**: "大家在之前的项目中，有没有遇到过'在我的机器上能跑'这样的问题？"

**等待回应，然后继续**:
"这就是我们今天要解决的问题。容器化技术让我们能够实现'一次构建，到处运行'。"

### 今天的学习目标
"今天结束后，你们将能够：
1. 理解容器化的核心概念和优势
2. 编写Dockerfile构建应用镜像
3. 使用Docker Compose编排多服务应用
4. 成功启动完整的RAG系统服务栈"

---

## 理论讲解：容器化技术概述 (15分钟)

### 什么是容器化？
"让我用一个生活中的例子来解释容器化。

想象一下，你要搬家。传统的方式是什么？把所有东西散装在车里，到了新家再重新整理。但是如果用集装箱呢？所有东西都在箱子里，搬到哪里都是完整的。

容器化就是软件世界的'集装箱'。"

**在黑板上画图**:
```
传统部署:
[应用A] [应用B] [应用C]
    ↓       ↓       ↓
[操作系统] [操作系统] [操作系统]
    ↓       ↓       ↓
[服务器A] [服务器B] [服务器C]

容器化部署:
[容器A] [容器B] [容器C]
         ↓
    [Docker引擎]
         ↓
     [操作系统]
         ↓
      [服务器]
```

### 容器化的核心概念
"容器化有三个核心概念，我们用快递来类比：

1. **镜像(Image)** = 快递模板
   - 只读的模板，包含应用运行所需的一切
   - 就像快递公司的标准包装盒

2. **容器(Container)** = 实际的快递包裹
   - 镜像的运行实例
   - 每个包裹都是独立的，互不影响

3. **仓库(Registry)** = 快递中转站
   - 存储和分发镜像的地方
   - Docker Hub就是最大的公共仓库"

### 容器化的优势
**提问**: "大家觉得容器化能解决什么问题？"

**引导学生回答，然后总结**:
"容器化的五大优势：
1. **环境一致性** - 开发、测试、生产完全一致
2. **资源效率** - 比虚拟机更轻量，启动更快
3. **可移植性** - 一次构建，到处运行
4. **可扩展性** - 易于水平扩展
5. **隔离性** - 进程、网络、文件系统隔离"

---

## 实践演示一：创建Dockerfile (20分钟)

### 演示准备
"现在我们来实际操作。我要为我们的FastAPI应用创建一个Docker镜像。"

**打开VS Code，创建新文件**:
```bash
# 在演示目录中
touch Dockerfile
```

### 逐步编写Dockerfile
"Dockerfile就像是一个菜谱，告诉Docker如何一步步构建我们的镜像。"

**边写边解释每一行**:
```dockerfile
# 第一步：选择基础镜像
FROM python:3.11-slim
# 解释：我们选择官方的Python镜像，slim版本更小

# 第二步：设置工作目录
WORKDIR /app
# 解释：就像进入厨房准备做菜

# 第三步：设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
# 解释：确保Python输出不被缓冲，便于调试

# 第四步：安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*
# 解释：安装编译工具，然后清理缓存

# 第五步：复制并安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
# 解释：先复制依赖文件，利用Docker层缓存

# 第六步：复制应用代码
COPY src/ ./src/
# 解释：代码变化频繁，放在后面

# 第七步：创建非root用户（安全最佳实践）
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app
# 解释：不要用root用户运行应用

# 第八步：暴露端口
EXPOSE 8000
# 解释：声明应用监听的端口

# 第九步：健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
# 解释：让Docker知道如何检查应用是否健康

# 第十步：启动命令
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
# 解释：容器启动时执行的命令
```

### 构建镜像演示
```bash
# 构建镜像
docker build -t rag-app:latest .

# 解释构建过程中的每一步
# 展示层缓存的效果
```

### 运行容器测试
```bash
# 运行容器
docker run -d -p 8000:8000 --name rag-app-test rag-app:latest

# 测试应用
curl http://localhost:8000/health

# 查看容器状态
docker ps

# 查看日志
docker logs rag-app-test
```

**课堂互动**: "大家看到了什么？应用是不是成功运行了？"

---

## 实践演示二：Docker Compose服务编排 (25分钟)

### 引入多服务场景
"现在我们的应用能跑了，但是它还需要数据库、缓存等服务。如果一个个手动启动，会很麻烦。这就是Docker Compose的用武之地。"

**类比解释**:
"如果说Docker是单个集装箱，那么Docker Compose就是整个货轮的调度系统。"

### 创建docker-compose.yml
**创建新文件并逐步编写**:
```yaml
# 版本声明
version: '3.8'

# 服务定义
services:
  # FastAPI应用服务
  app:
    build: .  # 使用当前目录的Dockerfile构建
    ports:
      - "8000:8000"  # 端口映射
    environment:
      - DATABASE_URL=postgresql://rag:ragpass@postgres:5432/rag_db
      - REDIS_URL=redis://redis:6379
    depends_on:  # 依赖关系
      - postgres
      - redis
    networks:
      - rag-network
    restart: unless-stopped  # 重启策略

  # PostgreSQL数据库
  postgres:
    image: postgres:15  # 使用官方镜像
    environment:
      POSTGRES_DB: rag_db
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: ragpass
    volumes:
      - postgres_data:/var/lib/postgresql/data  # 数据持久化
    ports:
      - "5432:5432"
    networks:
      - rag-network
    restart: unless-stopped

  # Redis缓存
  redis:
    image: redis:7-alpine  # 使用轻量版本
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

# 数据卷定义
volumes:
  postgres_data:
  redis_data:
  qdrant_data:
  minio_data:

# 网络定义
networks:
  rag-network:
    driver: bridge
```

### 逐步解释配置
"让我解释一下这个配置文件的关键部分：

1. **services**: 定义了5个服务，每个服务都是一个容器
2. **networks**: 创建了一个自定义网络，让服务间可以通信
3. **volumes**: 定义了数据卷，确保数据持久化
4. **depends_on**: 定义了服务启动顺序"

### 启动服务栈演示
```bash
# 清理之前的测试容器
docker stop rag-app-test
docker rm rag-app-test

# 启动所有服务
docker-compose up -d

# 解释启动过程
# 展示服务启动顺序
```

### 验证服务状态
```bash
# 查看所有服务状态
docker-compose ps

# 查看服务日志
docker-compose logs app
docker-compose logs postgres

# 测试服务连通性
curl http://localhost:8000/health
curl http://localhost:9000  # MinIO控制台
```

**课堂互动**: "大家看到了什么？所有服务都启动成功了吗？"

---

## 故障排查演示 (10分钟)

### 模拟常见问题
"在实际使用中，我们经常会遇到一些问题。让我演示几个常见的故障排查方法。"

### 问题1：端口冲突
```bash
# 模拟端口被占用
# 修改docker-compose.yml中的端口映射
# 演示如何解决端口冲突
```

### 问题2：服务无法连接
```bash
# 进入应用容器
docker-compose exec app bash

# 测试网络连通性
ping postgres
ping redis

# 查看环境变量
env | grep DATABASE_URL
```

### 问题3：数据持久化验证
```bash
# 停止服务
docker-compose down

# 重新启动
docker-compose up -d

# 验证数据是否还在
docker-compose exec postgres psql -U rag -d rag_db -c "\l"
```

---

## Exercise指导 (15分钟)

### Exercise任务分配
"现在轮到大家动手了。我给大家15分钟时间，完成以下任务：

1. **基础任务**（所有人必做）：
   - 创建自己的Dockerfile
   - 构建并运行应用容器

2. **进阶任务**（完成基础任务后）：
   - 编写docker-compose.yml
   - 启动多服务栈"

### 巡回指导要点
- 检查Dockerfile语法
- 帮助解决构建错误
- 指导端口配置问题
- 协助网络连通性测试

### 常见问题应对话术
**学生问**: "我的容器启动失败了"
**回答**: "让我们一起看看日志。运行`docker logs <容器名>`，看看具体的错误信息是什么。"

**学生问**: "为什么我的应用连不上数据库？"
**回答**: "检查三个地方：1. 数据库容器是否启动成功；2. 网络配置是否正确；3. 连接字符串是否正确。"

---

## 总结与答疑 (10分钟)

### 知识点回顾
"让我们回顾一下今天学到的核心内容：

1. **容器化概念**: 镜像、容器、仓库
2. **Dockerfile编写**: 分层构建、最佳实践
3. **Docker Compose**: 多服务编排、网络配置
4. **企业级架构**: 数据库、缓存、存储的集成"

### 关键技能强化
"今天最重要的技能是什么？
- **故障排查能力**: 会看日志、会测试连通性
- **配置管理能力**: 理解环境变量、网络、存储
- **系统思维**: 理解服务间的依赖关系"

### 课后作业布置
"课后作业有两个层次：

**必做作业**:
1. 完善你的docker-compose.yml，添加健康检查
2. 编写一个简单的部署文档

**选做作业**:
1. 添加Prometheus和Grafana监控
2. 研究Docker的多阶段构建"

### 下节课预告
"下节课我们将学习数据模型设计。今天搭建的PostgreSQL数据库，将成为我们存储RAG系统数据的基础。

**预习建议**:
1. 复习SQL基础语法
2. 了解ORM的概念
3. 思考RAG系统需要哪些数据表"

### 答疑时间
"现在是答疑时间，大家有什么问题吗？"

**常见问题预案**:
- Docker Desktop安装问题
- 内存不足问题
- 网络连接问题
- 权限问题

---

## 课后反思记录

### 教学效果评估
- [ ] 学生参与度如何？
- [ ] 哪些概念需要重点强化？
- [ ] 实践环节是否顺利？
- [ ] 时间分配是否合理？

### 改进建议
- [ ] 需要增加的演示内容
- [ ] 需要简化的复杂概念
- [ ] 需要补充的故障排查场景
- [ ] 需要调整的Exercise难度

### 学生反馈收集
- [ ] 课堂理解度调查
- [ ] 实践难点收集
- [ ] 改进建议征集