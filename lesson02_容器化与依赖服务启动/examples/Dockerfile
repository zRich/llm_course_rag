# RAG应用容器化 - 生产级Dockerfile示例
# 基于Python 3.11官方镜像的多阶段构建

# ================================
# 第一阶段：构建阶段
# ================================
FROM python:3.11-slim as builder

# 设置构建时的环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建虚拟环境
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制并安装Python依赖
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ================================
# 第二阶段：运行阶段
# ================================
FROM python:3.11-slim as runtime

# 设置运行时环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 从构建阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv

# 创建应用用户（安全最佳实践）
RUN groupadd -r app && useradd -r -g app app

# 设置工作目录
WORKDIR /app

# 复制应用代码
COPY --chown=app:app src/ ./src/
COPY --chown=app:app alembic/ ./alembic/
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app .env.example ./.env

# 创建必要的目录
RUN mkdir -p /app/logs /app/uploads && \
    chown -R app:app /app

# 切换到非root用户
USER app

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ================================
# 构建和运行说明
# ================================
# 构建镜像：
# docker build -t rag-app:latest .
#
# 运行容器：
# docker run -d \
#   --name rag-app \
#   -p 8000:8000 \
#   -e DATABASE_URL=postgresql://user:pass@host:5432/db \
#   -e REDIS_URL=redis://host:6379 \
#   -v /host/logs:/app/logs \
#   rag-app:latest
#
# 开发模式运行：
# docker run -it \
#   --name rag-app-dev \
#   -p 8000:8000 \
#   -v $(pwd)/src:/app/src \
#   rag-app:latest \
#   uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload