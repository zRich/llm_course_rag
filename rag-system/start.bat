@echo off
REM RAG系统服务启动脚本 (Windows版本)
REM 用于快速启动和管理Docker服务

setlocal enabledelayedexpansion

REM 设置编码为UTF-8
chcp 65001 >nul

REM 颜色定义（Windows cmd颜色代码）
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM 打印带颜色的消息
:print_info
echo %BLUE%[INFO]%NC% %~1
goto :eof

:print_success
echo %GREEN%[SUCCESS]%NC% %~1
goto :eof

:print_warning
echo %YELLOW%[WARNING]%NC% %~1
goto :eof

:print_error
echo %RED%[ERROR]%NC% %~1
goto :eof

REM 检查Docker是否安装
:check_docker
docker --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker未安装，请先安装Docker Desktop"
    exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker未运行，请启动Docker Desktop"
    exit /b 1
)

call :print_success "Docker检查通过"
goto :eof

REM 检查docker-compose是否可用
:check_compose
docker compose version >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker Compose不可用，请确保Docker Desktop版本支持compose命令"
    exit /b 1
)

call :print_success "Docker Compose检查通过"
goto :eof

REM 检查环境变量文件
:check_env
if not exist ".env" (
    call :print_warning ".env文件不存在，从.env.example复制"
    copy ".env.example" ".env" >nul
    call :print_info "已创建.env文件，请根据需要修改配置"
) else (
    call :print_success "环境变量文件检查通过"
)
goto :eof

REM 启动服务
:start_services
call :print_info "启动RAG系统服务..."

REM 拉取最新镜像
call :print_info "拉取Docker镜像..."
docker compose pull

REM 启动服务
call :print_info "启动服务容器..."
docker compose up -d

call :print_success "服务启动完成"
goto :eof

REM 等待服务就绪
:wait_for_services
call :print_info "等待服务启动完成..."

REM 等待30秒让服务启动
timeout /t 30 /nobreak >nul

call :print_info "检查服务健康状态..."

REM 检查服务状态
docker compose ps

call :print_success "服务状态检查完成"
goto :eof

REM 运行连接测试
:run_tests
call :print_info "运行服务连接测试..."

if exist "test_connections.py" (
    REM 检查Python虚拟环境
    if exist ".venv" (
        call :print_info "使用虚拟环境运行测试"
        call .venv\Scripts\activate.bat
        python test_connections.py
    ) else (
        call :print_warning "未找到虚拟环境，使用系统Python运行测试"
        python test_connections.py
    )
) else (
    call :print_warning "未找到测试脚本，跳过连接测试"
)
goto :eof

REM 显示服务信息
:show_info
echo.
call :print_success "🎉 RAG系统服务启动完成！"
echo.
echo 📊 服务访问地址:
echo    PostgreSQL:      localhost:5432
echo    Qdrant API:      http://localhost:6333
echo    Qdrant Dashboard: http://localhost:6333/dashboard
echo    Redis:           localhost:6379
echo    MinIO API:       http://localhost:9000
echo    MinIO Console:   http://localhost:9001
echo.
echo 🔑 默认登录信息:
echo    PostgreSQL: rag_user / rag_password
echo    Redis:      redis_password
echo    MinIO:      minio_admin / minio_password
echo.
echo 🛠️ 常用命令:
echo    查看服务状态:   docker compose ps
echo    查看服务日志:   docker compose logs -f
echo    停止服务:      docker compose stop
echo    重启服务:      docker compose restart
echo    完全清理:      docker compose down -v
echo.
goto :eof

REM 停止服务
:stop_services
call :print_info "停止RAG系统服务..."
docker compose stop
call :print_success "服务已停止"
goto :eof

REM 重启服务
:restart_services
call :print_info "重启RAG系统服务..."
docker compose restart
call :print_success "服务已重启"
goto :eof

REM 清理服务
:clean_services
call :print_warning "这将删除所有容器和数据卷，数据将丢失！"
set /p "confirm=确定要继续吗？(y/N): "
if /i "!confirm!"=="y" (
    call :print_info "清理RAG系统服务..."
    docker compose down -v
    call :print_success "服务已清理"
) else (
    call :print_info "取消清理操作"
)
goto :eof

REM 显示帮助信息
:show_help
echo RAG系统服务管理脚本
echo.
echo 用法: %~nx0 [命令]
echo.
echo 命令:
echo   start     启动所有服务（默认）
echo   stop      停止所有服务
echo   restart   重启所有服务
echo   status    查看服务状态
echo   logs      查看服务日志
echo   test      运行连接测试
echo   clean     清理所有服务和数据
echo   help      显示此帮助信息
echo.
goto :eof

REM 主函数
:main
set "command=%~1"
if "%command%"=="" set "command=start"

if "%command%"=="start" (
    call :print_info "🚀 启动RAG系统服务"
    call :check_docker
    if errorlevel 1 exit /b 1
    call :check_compose
    if errorlevel 1 exit /b 1
    call :check_env
    call :start_services
    call :wait_for_services
    call :run_tests
    call :show_info
) else if "%command%"=="stop" (
    call :stop_services
) else if "%command%"=="restart" (
    call :restart_services
) else if "%command%"=="status" (
    call :print_info "服务状态:"
    docker compose ps
) else if "%command%"=="logs" (
    call :print_info "服务日志:"
    docker compose logs -f
) else if "%command%"=="test" (
    call :run_tests
) else if "%command%"=="clean" (
    call :clean_services
) else if "%command%"=="help" (
    call :show_help
) else if "%command%"=="--help" (
    call :show_help
) else if "%command%"=="-h" (
    call :show_help
) else (
    call :print_error "未知命令: %command%"
    call :show_help
    exit /b 1
)

goto :eof

REM 调用主函数
call :main %*