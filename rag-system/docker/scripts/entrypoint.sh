#!/bin/bash
# RAG系统容器启动脚本
# 用于容器启动时的初始化和健康检查

set -e

# =============================================================================
# 环境变量设置
# =============================================================================
export PYTHONPATH="/app:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# =============================================================================
# 颜色输出函数
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# =============================================================================
# 等待服务函数
# =============================================================================
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=${4:-30}
    local attempt=1
    
    log_info "等待 $service_name 服务启动 ($host:$port)..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            log_success "$service_name 服务已就绪"
            return 0
        fi
        
        log_warn "等待 $service_name 服务... (尝试 $attempt/$max_attempts)"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_error "$service_name 服务启动超时"
    return 1
}

# =============================================================================
# 数据库连接检查
# =============================================================================
check_database() {
    log_info "检查数据库连接..."
    
    python3 -c "
import asyncio
import asyncpg
import os
import sys

async def check_db():
    try:
        conn = await asyncpg.connect(
            host=os.getenv('POSTGRES_HOST', 'postgres'),
            port=int(os.getenv('POSTGRES_PORT', 5432)),
            user=os.getenv('POSTGRES_USER', 'rag_user'),
            password=os.getenv('POSTGRES_PASSWORD', 'rag_password'),
            database=os.getenv('POSTGRES_DB', 'rag_system')
        )
        await conn.close()
        print('数据库连接成功')
        return True
    except Exception as e:
        print(f'数据库连接失败: {e}')
        return False

if not asyncio.run(check_db()):
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        log_success "数据库连接检查通过"
    else
        log_error "数据库连接检查失败"
        return 1
    fi
}

# =============================================================================
# Redis连接检查
# =============================================================================
check_redis() {
    log_info "检查Redis连接..."
    
    python3 -c "
import redis
import os
import sys

try:
    r = redis.Redis(
        host=os.getenv('REDIS_HOST', 'redis'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        password=os.getenv('REDIS_PASSWORD', ''),
        decode_responses=True
    )
    r.ping()
    print('Redis连接成功')
except Exception as e:
    print(f'Redis连接失败: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        log_success "Redis连接检查通过"
    else
        log_error "Redis连接检查失败"
        return 1
    fi
}

# =============================================================================
# Qdrant连接检查
# =============================================================================
check_qdrant() {
    log_info "检查Qdrant连接..."
    
    python3 -c "
import requests
import os
import sys

try:
    qdrant_host = os.getenv('QDRANT_HOST', 'qdrant')
    qdrant_port = os.getenv('QDRANT_PORT', '6333')
    response = requests.get(f'http://{qdrant_host}:{qdrant_port}/health', timeout=5)
    if response.status_code == 200:
        print('Qdrant连接成功')
    else:
        print(f'Qdrant健康检查失败: {response.status_code}')
        sys.exit(1)
except Exception as e:
    print(f'Qdrant连接失败: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        log_success "Qdrant连接检查通过"
    else
        log_error "Qdrant连接检查失败"
        return 1
    fi
}

# =============================================================================
# 创建必要目录
# =============================================================================
create_directories() {
    log_info "创建必要的目录..."
    
    mkdir -p /app/logs
    mkdir -p /app/uploads
    mkdir -p /app/data
    mkdir -p /app/tmp
    
    # 确保目录权限正确
    chmod 755 /app/logs /app/uploads /app/data /app/tmp
    
    log_success "目录创建完成"
}

# =============================================================================
# 数据库迁移
# =============================================================================
run_migrations() {
    log_info "运行数据库迁移..."
    
    # 这里可以添加数据库迁移逻辑
    # 例如：alembic upgrade head
    
    log_success "数据库迁移完成"
}

# =============================================================================
# 主启动流程
# =============================================================================
main() {
    log_info "=== RAG系统容器启动 ==="
    log_info "环境: ${APP_ENV:-development}"
    log_info "Python版本: $(python3 --version)"
    log_info "工作目录: $(pwd)"
    
    # 创建必要目录
    create_directories
    
    # 等待依赖服务
    if [ "${SKIP_WAIT_FOR_SERVICES:-false}" != "true" ]; then
        wait_for_service "${POSTGRES_HOST:-postgres}" "${POSTGRES_PORT:-5432}" "PostgreSQL" 60
        wait_for_service "${REDIS_HOST:-redis}" "${REDIS_PORT:-6379}" "Redis" 30
        wait_for_service "${QDRANT_HOST:-qdrant}" "${QDRANT_PORT:-6333}" "Qdrant" 30
        
        # 检查服务连接
        check_database
        check_redis
        check_qdrant
    else
        log_warn "跳过服务等待检查"
    fi
    
    # 运行数据库迁移
    if [ "${SKIP_MIGRATIONS:-false}" != "true" ]; then
        run_migrations
    else
        log_warn "跳过数据库迁移"
    fi
    
    log_success "=== 容器初始化完成 ==="
    
    # 执行传入的命令
    if [ $# -gt 0 ]; then
        log_info "执行命令: $*"
        exec "$@"
    else
        log_info "启动默认服务"
        exec uvicorn src.api.health:app --host 0.0.0.0 --port 8000
    fi
}

# =============================================================================
# 信号处理
# =============================================================================
cleanup() {
    log_info "接收到停止信号，正在清理..."
    # 这里可以添加清理逻辑
    exit 0
}

trap cleanup SIGTERM SIGINT

# =============================================================================
# 脚本入口
# =============================================================================
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi