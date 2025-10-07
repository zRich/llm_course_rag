#!/bin/bash
# RAG系统健康检查脚本
# 用于Docker容器的健康状态检查

set -e

# =============================================================================
# 配置参数
# =============================================================================
HEALTH_CHECK_URL="http://localhost:8000/health"
TIMEOUT=10
MAX_RETRIES=3
RETRY_INTERVAL=2

# =============================================================================
# 颜色输出
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1" >&2
}

log_error() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${RED}[ERROR]${NC} $1" >&2
}

log_success() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${GREEN}[SUCCESS]${NC} $1" >&2
}

# =============================================================================
# 基础健康检查
# =============================================================================
basic_health_check() {
    local url=$1
    local timeout=$2
    
    # 使用curl进行HTTP健康检查
    if command -v curl >/dev/null 2>&1; then
        response=$(curl -s -w "%{http_code}" -o /dev/null --max-time "$timeout" "$url" 2>/dev/null || echo "000")
        if [ "$response" = "200" ]; then
            return 0
        else
            log_error "HTTP健康检查失败，状态码: $response"
            return 1
        fi
    # 备用：使用wget
    elif command -v wget >/dev/null 2>&1; then
        if wget -q --timeout="$timeout" --tries=1 -O /dev/null "$url" 2>/dev/null; then
            return 0
        else
            log_error "HTTP健康检查失败 (wget)"
            return 1
        fi
    # 备用：使用Python
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c "
import urllib.request
import urllib.error
import socket
import sys

try:
    req = urllib.request.Request('$url')
    with urllib.request.urlopen(req, timeout=$timeout) as response:
        if response.getcode() == 200:
            sys.exit(0)
        else:
            sys.exit(1)
except Exception as e:
    print(f'Python健康检查失败: {e}', file=sys.stderr)
    sys.exit(1)
"
        return $?
    else
        log_error "没有可用的HTTP客户端工具 (curl, wget, python3)"
        return 1
    fi
}

# =============================================================================
# 详细健康检查
# =============================================================================
detailed_health_check() {
    local url=$1
    local timeout=$2
    
    if command -v curl >/dev/null 2>&1; then
        # 获取详细的健康检查信息
        response=$(curl -s --max-time "$timeout" "$url" 2>/dev/null)
        http_code=$(curl -s -w "%{http_code}" -o /dev/null --max-time "$timeout" "$url" 2>/dev/null || echo "000")
        
        if [ "$http_code" = "200" ]; then
            # 尝试解析JSON响应
            if echo "$response" | python3 -c "import sys, json; json.load(sys.stdin)" 2>/dev/null; then
                log_success "健康检查通过，响应格式正确"
                # 可以在这里添加更多的响应内容验证
                return 0
            else
                log_error "健康检查响应格式无效"
                return 1
            fi
        else
            log_error "健康检查失败，HTTP状态码: $http_code"
            return 1
        fi
    else
        # 回退到基础检查
        return $(basic_health_check "$url" "$timeout")
    fi
}

# =============================================================================
# 进程检查
# =============================================================================
check_process() {
    # 检查Python进程是否运行
    if pgrep -f "uvicorn" >/dev/null 2>&1; then
        log_info "Uvicorn进程正在运行"
        return 0
    else
        log_error "Uvicorn进程未运行"
        return 1
    fi
}

# =============================================================================
# 端口检查
# =============================================================================
check_port() {
    local port=${1:-8000}
    
    if command -v netstat >/dev/null 2>&1; then
        if netstat -ln | grep ":$port " >/dev/null 2>&1; then
            log_info "端口 $port 正在监听"
            return 0
        else
            log_error "端口 $port 未在监听"
            return 1
        fi
    elif command -v ss >/dev/null 2>&1; then
        if ss -ln | grep ":$port " >/dev/null 2>&1; then
            log_info "端口 $port 正在监听"
            return 0
        else
            log_error "端口 $port 未在监听"
            return 1
        fi
    else
        log_info "无法检查端口状态（缺少netstat/ss工具）"
        return 0
    fi
}

# =============================================================================
# 内存检查
# =============================================================================
check_memory() {
    if command -v free >/dev/null 2>&1; then
        # 获取内存使用情况
        mem_usage=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
        log_info "内存使用率: ${mem_usage}%"
        
        # 如果内存使用率超过90%，发出警告
        if [ "$(echo "$mem_usage > 90" | bc 2>/dev/null || echo 0)" = "1" ]; then
            log_error "内存使用率过高: ${mem_usage}%"
            return 1
        fi
    fi
    return 0
}

# =============================================================================
# 磁盘空间检查
# =============================================================================
check_disk_space() {
    # 检查根目录磁盘使用情况
    if command -v df >/dev/null 2>&1; then
        disk_usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
        log_info "磁盘使用率: ${disk_usage}%"
        
        # 如果磁盘使用率超过90%，发出警告
        if [ "$disk_usage" -gt 90 ]; then
            log_error "磁盘空间不足: ${disk_usage}%"
            return 1
        fi
    fi
    return 0
}

# =============================================================================
# 主健康检查函数
# =============================================================================
health_check() {
    local check_type=${1:-"basic"}
    local retries=0
    
    log_info "开始健康检查 (类型: $check_type)"
    
    while [ $retries -lt $MAX_RETRIES ]; do
        case $check_type in
            "basic")
                if basic_health_check "$HEALTH_CHECK_URL" "$TIMEOUT"; then
                    log_success "基础健康检查通过"
                    return 0
                fi
                ;;
            "detailed")
                if detailed_health_check "$HEALTH_CHECK_URL" "$TIMEOUT"; then
                    log_success "详细健康检查通过"
                    return 0
                fi
                ;;
            "full")
                # 执行完整检查
                local all_passed=true
                
                # HTTP健康检查
                if ! detailed_health_check "$HEALTH_CHECK_URL" "$TIMEOUT"; then
                    all_passed=false
                fi
                
                # 进程检查
                if ! check_process; then
                    all_passed=false
                fi
                
                # 端口检查
                if ! check_port 8000; then
                    all_passed=false
                fi
                
                # 系统资源检查
                check_memory || true  # 不因为内存警告而失败
                check_disk_space || true  # 不因为磁盘警告而失败
                
                if [ "$all_passed" = "true" ]; then
                    log_success "完整健康检查通过"
                    return 0
                fi
                ;;
            *)
                log_error "未知的检查类型: $check_type"
                return 1
                ;;
        esac
        
        retries=$((retries + 1))
        if [ $retries -lt $MAX_RETRIES ]; then
            log_info "健康检查失败，等待 ${RETRY_INTERVAL}s 后重试 ($retries/$MAX_RETRIES)"
            sleep $RETRY_INTERVAL
        fi
    done
    
    log_error "健康检查失败，已达到最大重试次数"
    return 1
}

# =============================================================================
# 脚本入口
# =============================================================================
main() {
    local check_type=${1:-"basic"}
    
    # 从环境变量读取配置
    HEALTH_CHECK_URL=${HEALTH_CHECK_URL:-"http://localhost:8000/health"}
    TIMEOUT=${HEALTH_CHECK_TIMEOUT:-10}
    MAX_RETRIES=${HEALTH_CHECK_MAX_RETRIES:-3}
    RETRY_INTERVAL=${HEALTH_CHECK_RETRY_INTERVAL:-2}
    
    # 执行健康检查
    if health_check "$check_type"; then
        exit 0
    else
        exit 1
    fi
}

# 如果脚本被直接执行
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi