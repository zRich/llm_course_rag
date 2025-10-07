#!/bin/bash

# RAG系统数据备份脚本
# 用于备份PostgreSQL数据库、Redis数据、Qdrant集合和MinIO对象存储

set -e  # 遇到错误立即退出

# 配置变量
BACKUP_DIR="./backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="rag_backup_${TIMESTAMP}"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"

# 数据库配置
DB_CONTAINER="rag-system-postgres-1"
DB_NAME="rag_db"
DB_USER="rag_user"
DB_PASSWORD="rag_password"

# Redis配置
REDIS_CONTAINER="rag-system-redis-1"

# Qdrant配置
QDRANT_CONTAINER="rag-system-qdrant-1"

# MinIO配置
MINIO_CONTAINER="rag-system-minio-1"
MINIO_BUCKET="rag-documents"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker容器是否运行
check_container() {
    local container_name=$1
    if ! docker ps --format "table {{.Names}}" | grep -q "^${container_name}$"; then
        log_error "容器 ${container_name} 未运行"
        return 1
    fi
    return 0
}

# 创建备份目录
create_backup_dir() {
    log_info "创建备份目录: ${BACKUP_PATH}"
    mkdir -p "${BACKUP_PATH}"
}

# 备份PostgreSQL数据库
backup_postgres() {
    log_info "开始备份PostgreSQL数据库..."
    
    if ! check_container "${DB_CONTAINER}"; then
        return 1
    fi
    
    local backup_file="${BACKUP_PATH}/postgres_${TIMESTAMP}.sql"
    
    docker exec "${DB_CONTAINER}" pg_dump \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        --no-password \
        --verbose \
        --clean \
        --if-exists \
        --create > "${backup_file}"
    
    if [ $? -eq 0 ]; then
        log_info "PostgreSQL备份完成: ${backup_file}"
        # 压缩备份文件
        gzip "${backup_file}"
        log_info "备份文件已压缩: ${backup_file}.gz"
    else
        log_error "PostgreSQL备份失败"
        return 1
    fi
}

# 备份Redis数据
backup_redis() {
    log_info "开始备份Redis数据..."
    
    if ! check_container "${REDIS_CONTAINER}"; then
        return 1
    fi
    
    local backup_file="${BACKUP_PATH}/redis_${TIMESTAMP}.rdb"
    
    # 触发Redis保存
    docker exec "${REDIS_CONTAINER}" redis-cli BGSAVE
    
    # 等待保存完成
    sleep 5
    
    # 复制RDB文件
    docker cp "${REDIS_CONTAINER}:/data/dump.rdb" "${backup_file}"
    
    if [ $? -eq 0 ]; then
        log_info "Redis备份完成: ${backup_file}"
        # 压缩备份文件
        gzip "${backup_file}"
        log_info "备份文件已压缩: ${backup_file}.gz"
    else
        log_error "Redis备份失败"
        return 1
    fi
}

# 备份Qdrant数据
backup_qdrant() {
    log_info "开始备份Qdrant数据..."
    
    if ! check_container "${QDRANT_CONTAINER}"; then
        return 1
    fi
    
    local backup_dir="${BACKUP_PATH}/qdrant_${TIMESTAMP}"
    mkdir -p "${backup_dir}"
    
    # 复制Qdrant存储目录
    docker cp "${QDRANT_CONTAINER}:/qdrant/storage" "${backup_dir}/"
    
    if [ $? -eq 0 ]; then
        log_info "Qdrant备份完成: ${backup_dir}"
        # 压缩备份目录
        tar -czf "${backup_dir}.tar.gz" -C "${BACKUP_PATH}" "qdrant_${TIMESTAMP}"
        rm -rf "${backup_dir}"
        log_info "备份文件已压缩: ${backup_dir}.tar.gz"
    else
        log_error "Qdrant备份失败"
        return 1
    fi
}

# 备份MinIO数据
backup_minio() {
    log_info "开始备份MinIO数据..."
    
    if ! check_container "${MINIO_CONTAINER}"; then
        return 1
    fi
    
    local backup_dir="${BACKUP_PATH}/minio_${TIMESTAMP}"
    mkdir -p "${backup_dir}"
    
    # 复制MinIO数据目录
    docker cp "${MINIO_CONTAINER}:/data" "${backup_dir}/"
    
    if [ $? -eq 0 ]; then
        log_info "MinIO备份完成: ${backup_dir}"
        # 压缩备份目录
        tar -czf "${backup_dir}.tar.gz" -C "${BACKUP_PATH}" "minio_${TIMESTAMP}"
        rm -rf "${backup_dir}"
        log_info "备份文件已压缩: ${backup_dir}.tar.gz"
    else
        log_error "MinIO备份失败"
        return 1
    fi
}

# 创建备份清单
create_manifest() {
    log_info "创建备份清单..."
    
    local manifest_file="${BACKUP_PATH}/backup_manifest.txt"
    
    cat > "${manifest_file}" << EOF
RAG系统数据备份清单
==================

备份时间: $(date)
备份名称: ${BACKUP_NAME}
备份路径: ${BACKUP_PATH}

备份内容:
EOF
    
    # 列出备份文件
    ls -la "${BACKUP_PATH}" >> "${manifest_file}"
    
    log_info "备份清单创建完成: ${manifest_file}"
}

# 清理旧备份
cleanup_old_backups() {
    log_info "清理7天前的旧备份..."
    
    find "${BACKUP_DIR}" -name "rag_backup_*" -type d -mtime +7 -exec rm -rf {} \; 2>/dev/null || true
    
    log_info "旧备份清理完成"
}

# 主函数
main() {
    log_info "开始RAG系统数据备份..."
    log_info "备份时间: $(date)"
    
    # 创建备份目录
    create_backup_dir
    
    # 执行各项备份
    backup_postgres
    backup_redis
    backup_qdrant
    backup_minio
    
    # 创建备份清单
    create_manifest
    
    # 清理旧备份
    cleanup_old_backups
    
    log_info "备份完成！备份位置: ${BACKUP_PATH}"
    
    # 显示备份大小
    local backup_size=$(du -sh "${BACKUP_PATH}" | cut -f1)
    log_info "备份总大小: ${backup_size}"
}

# 帮助信息
show_help() {
    echo "RAG系统数据备份脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示帮助信息"
    echo "  --postgres     仅备份PostgreSQL"
    echo "  --redis        仅备份Redis"
    echo "  --qdrant       仅备份Qdrant"
    echo "  --minio        仅备份MinIO"
    echo ""
    echo "示例:"
    echo "  $0                # 备份所有服务"
    echo "  $0 --postgres     # 仅备份PostgreSQL"
}

# 解析命令行参数
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    --postgres)
        create_backup_dir
        backup_postgres
        create_manifest
        ;;
    --redis)
        create_backup_dir
        backup_redis
        create_manifest
        ;;
    --qdrant)
        create_backup_dir
        backup_qdrant
        create_manifest
        ;;
    --minio)
        create_backup_dir
        backup_minio
        create_manifest
        ;;
    "")
        main
        ;;
    *)
        log_error "未知选项: $1"
        show_help
        exit 1
        ;;
esac