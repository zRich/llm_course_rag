#!/bin/bash

# RAG系统服务启动脚本
# 用于快速启动和管理Docker服务

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker未安装，请先安装Docker Desktop"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker未运行，请启动Docker Desktop"
        exit 1
    fi
    
    print_success "Docker检查通过"
}

# 检查docker-compose是否可用
check_compose() {
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose不可用，请确保Docker Desktop版本支持compose命令"
        exit 1
    fi
    
    print_success "Docker Compose检查通过"
}

# 检查环境变量文件
check_env() {
    if [ ! -f ".env" ]; then
        print_warning ".env文件不存在，从.env.example复制"
        cp .env.example .env
        print_info "已创建.env文件，请根据需要修改配置"
    else
        print_success "环境变量文件检查通过"
    fi
}

# 启动服务
start_services() {
    print_info "启动RAG系统服务..."
    
    # 拉取最新镜像
    print_info "拉取Docker镜像..."
    docker compose pull
    
    # 启动服务
    print_info "启动服务容器..."
    docker compose up -d
    
    print_success "服务启动完成"
}

# 等待服务就绪
wait_for_services() {
    print_info "等待服务启动完成..."
    
    # 等待30秒让服务启动
    sleep 30
    
    print_info "检查服务健康状态..."
    
    # 检查服务状态
    docker compose ps
    
    print_success "服务状态检查完成"
}

# 运行连接测试
run_tests() {
    print_info "运行服务连接测试..."
    
    if [ -f "test_connections.py" ]; then
        # 检查Python虚拟环境
        if [ -d ".venv" ]; then
            print_info "使用虚拟环境运行测试"
            source .venv/bin/activate
            python test_connections.py
        else
            print_warning "未找到虚拟环境，使用系统Python运行测试"
            python3 test_connections.py
        fi
    else
        print_warning "未找到测试脚本，跳过连接测试"
    fi
}

# 显示服务信息
show_info() {
    echo
    print_success "🎉 RAG系统服务启动完成！"
    echo
    echo "📊 服务访问地址:"
    echo "   PostgreSQL:     localhost:5432"
    echo "   Qdrant API:      http://localhost:6333"
    echo "   Qdrant Dashboard: http://localhost:6333/dashboard"
    echo "   Redis:           localhost:6379"
    echo "   MinIO API:       http://localhost:9000"
    echo "   MinIO Console:   http://localhost:9001"
    echo
    echo "🔑 默认登录信息:"
    echo "   PostgreSQL: rag_user / rag_password"
    echo "   Redis:      redis_password"
    echo "   MinIO:      minio_admin / minio_password"
    echo
    echo "🛠️  常用命令:"
    echo "   查看服务状态:   docker compose ps"
    echo "   查看服务日志:   docker compose logs -f"
    echo "   停止服务:      docker compose stop"
    echo "   重启服务:      docker compose restart"
    echo "   完全清理:      docker compose down -v"
    echo
}

# 停止服务
stop_services() {
    print_info "停止RAG系统服务..."
    docker compose stop
    print_success "服务已停止"
}

# 重启服务
restart_services() {
    print_info "重启RAG系统服务..."
    docker compose restart
    print_success "服务已重启"
}

# 清理服务
clean_services() {
    print_warning "这将删除所有容器和数据卷，数据将丢失！"
    read -p "确定要继续吗？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "清理RAG系统服务..."
        docker compose down -v
        print_success "服务已清理"
    else
        print_info "取消清理操作"
    fi
}

# 显示帮助信息
show_help() {
    echo "RAG系统服务管理脚本"
    echo
    echo "用法: $0 [命令]"
    echo
    echo "命令:"
    echo "  start     启动所有服务（默认）"
    echo "  stop      停止所有服务"
    echo "  restart   重启所有服务"
    echo "  status    查看服务状态"
    echo "  logs      查看服务日志"
    echo "  test      运行连接测试"
    echo "  clean     清理所有服务和数据"
    echo "  help      显示此帮助信息"
    echo
}

# 主函数
main() {
    local command=${1:-start}
    
    case $command in
        start)
            print_info "🚀 启动RAG系统服务"
            check_docker
            check_compose
            check_env
            start_services
            wait_for_services
            run_tests
            show_info
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        status)
            print_info "服务状态:"
            docker compose ps
            ;;
        logs)
            print_info "服务日志:"
            docker compose logs -f
            ;;
        test)
            run_tests
            ;;
        clean)
            clean_services
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"