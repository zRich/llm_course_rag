# Lesson 02 环境设置检查清单 - 容器化与依赖服务启动

## 课前环境准备检查

### 系统要求验证
- [ ] **操作系统兼容性**
  ```bash
  # 检查操作系统版本
  uname -a
  # macOS: 10.15+ / Windows: 10+ / Linux: Ubuntu 18.04+
  ```

- [ ] **系统资源检查**
  ```bash
  # 检查可用内存（至少8GB推荐）
  free -h  # Linux
  # 或 Activity Monitor (macOS) / Task Manager (Windows)
  
  # 检查磁盘空间（至少20GB可用）
  df -h
  ```

- [ ] **网络连接验证**
  ```bash
  # 测试网络连接
  ping -c 4 google.com
  ping -c 4 docker.io
  ```

### Docker环境检查

- [ ] **Docker Desktop安装验证**
  ```bash
  # 检查Docker版本（要求20.10+）
  docker --version
  
  # 检查Docker Compose版本（要求2.0+）
  docker-compose --version
  ```

- [ ] **Docker服务状态检查**
  ```bash
  # 检查Docker守护进程状态
  docker info
  
  # 验证Docker可以运行容器
  docker run hello-world
  ```

- [ ] **Docker权限验证**
  ```bash
  # 确保当前用户可以运行Docker命令（无需sudo）
  docker ps
  
  # 如果需要sudo，请将用户添加到docker组
  # sudo usermod -aG docker $USER
  ```

---

## 基础容器化Exercise检查

### Dockerfile编写验证

- [ ] **项目结构创建**
  ```bash
  # 创建Exercise目录
  mkdir -p ~/rag-lesson02
  cd ~/rag-lesson02
  
  # 创建必要的目录结构
  mkdir -p src logs uploads
  touch Dockerfile requirements.txt src/main.py
  ```

- [ ] **requirements.txt配置**
  ```bash
  # 验证依赖文件内容
  cat requirements.txt
  # 应包含: fastapi, uvicorn, psycopg2-binary, redis等
  ```

- [ ] **Dockerfile语法检查**
  ```bash
  # 使用hadolint检查Dockerfile最佳实践（可选）
  # docker run --rm -i hadolint/hadolint < Dockerfile
  
  # 基础语法验证
  docker build --dry-run -t test-build .
  ```

### 镜像构建验证

- [ ] **构建应用镜像**
  ```bash
  # 构建镜像
  docker build -t rag-app:lesson02 .
  
  # 验证镜像创建成功
  docker images | grep rag-app
  ```

- [ ] **镜像大小检查**
  ```bash
  # 检查镜像大小（应该合理，不超过1GB）
  docker images rag-app:lesson02 --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
  ```

- [ ] **镜像安全扫描**（可选）
  ```bash
  # 使用Docker Scout扫描安全漏洞
  docker scout cves rag-app:lesson02
  ```

### 单容器运行验证

- [ ] **容器启动测试**
  ```bash
  # 启动容器
  docker run -d --name rag-test -p 8000:8000 rag-app:lesson02
  
  # 检查容器状态
  docker ps | grep rag-test
  ```

- [ ] **应用访问验证**
  ```bash
  # 测试应用响应
  curl http://localhost:8000/
  curl http://localhost:8000/health
  
  # 或在浏览器中访问 http://localhost:8000/docs
  ```

- [ ] **容器日志检查**
  ```bash
  # 查看容器日志
  docker logs rag-test
  
  # 实时查看日志
  docker logs -f rag-test
  ```

- [ ] **容器资源监控**
  ```bash
  # 检查容器资源使用情况
  docker stats rag-test --no-stream
  ```

---

## Docker Compose多服务验证

### 配置文件检查

- [ ] **docker-compose.yml语法验证**
  ```bash
  # 验证配置文件语法
  docker-compose config
  
  # 检查服务定义
  docker-compose config --services
  ```

- [ ] **环境变量配置**
  ```bash
  # 创建环境变量文件
  cp .env.example .env
  
  # 验证环境变量加载
  docker-compose config | grep -E "DATABASE_URL|REDIS_URL"
  ```

### 服务栈启动验证

- [ ] **清理之前的容器**
  ```bash
  # 停止并删除测试容器
  docker stop rag-test
  docker rm rag-test
  ```

- [ ] **启动完整服务栈**
  ```bash
  # 启动所有服务
  docker-compose up -d
  
  # 检查所有服务状态
  docker-compose ps
  ```

- [ ] **服务健康检查**
  ```bash
  # 等待服务完全启动（约30-60秒）
  sleep 60
  
  # 检查各服务状态
  docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
  ```

### 服务连通性验证

- [ ] **FastAPI应用测试**
  ```bash
  # 测试应用健康检查
  curl http://localhost:8000/health
  
  # 测试API文档访问
  curl -I http://localhost:8000/docs
  ```

- [ ] **PostgreSQL连接测试**
  ```bash
  # 测试数据库连接
  docker-compose exec postgres pg_isready -U rag -d rag_db
  
  # 连接数据库并查看版本
  docker-compose exec postgres psql -U rag -d rag_db -c "SELECT version();"
  ```

- [ ] **Redis连接测试**
  ```bash
  # 测试Redis连接
  docker-compose exec redis redis-cli ping
  
  # 测试Redis基本操作
  docker-compose exec redis redis-cli set test "hello"
  docker-compose exec redis redis-cli get test
  ```

- [ ] **Qdrant API测试**
  ```bash
  # 测试Qdrant API
  curl http://localhost:6333/
  curl http://localhost:6333/collections
  ```

- [ ] **MinIO服务测试**
  ```bash
  # 测试MinIO API
  curl http://localhost:9000/minio/health/live
  
  # 访问MinIO控制台
  echo "访问 http://localhost:9001"
  echo "用户名: minioadmin"
  echo "密码: minioadmin"
  ```

### 服务间通信验证

- [ ] **应用到数据库连接**
  ```bash
  # 进入应用容器测试网络连通性
  docker-compose exec app ping postgres
  docker-compose exec app nslookup postgres
  ```

- [ ] **应用到缓存连接**
  ```bash
  # 测试Redis连接
  docker-compose exec app ping redis
  ```

- [ ] **环境变量验证**
  ```bash
  # 检查应用容器中的环境变量
  docker-compose exec app env | grep -E "DATABASE_URL|REDIS_URL"
  ```

---

## 数据持久化验证

### 数据卷检查

- [ ] **数据卷创建验证**
  ```bash
  # 查看创建的数据卷
  docker volume ls | grep lesson02
  ```

- [ ] **数据持久化测试**
  ```bash
  # 在数据库中创建测试数据
  docker-compose exec postgres psql -U rag -d rag_db -c "
  CREATE TABLE IF NOT EXISTS test_persistence (
    id SERIAL PRIMARY KEY,
    message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  INSERT INTO test_persistence (message) VALUES ('persistence test');
  "
  ```

- [ ] **服务重启后数据验证**
  ```bash
  # 重启服务栈
  docker-compose restart
  
  # 等待服务启动
  sleep 30
  
  # 验证数据是否还存在
  docker-compose exec postgres psql -U rag -d rag_db -c "
  SELECT * FROM test_persistence;
  "
  ```

---

## 故障排查验证

### 日志分析能力

- [ ] **查看服务日志**
  ```bash
  # 查看所有服务日志
  docker-compose logs
  
  # 查看特定服务日志
  docker-compose logs app
  docker-compose logs postgres
  ```

- [ ] **实时日志监控**
  ```bash
  # 实时查看应用日志
  docker-compose logs -f app
  ```

### 网络诊断能力

- [ ] **网络连接测试**
  ```bash
  # 查看Docker网络
  docker network ls
  
  # 检查网络详情
  docker network inspect lesson02_rag-network
  ```

- [ ] **端口占用检查**
  ```bash
  # 检查端口占用情况
  netstat -tulpn | grep -E "8000|5432|6379|6333|9000"
  # 或使用 lsof -i :8000
  ```

### 容器调试能力

- [ ] **进入容器调试**
  ```bash
  # 进入应用容器
  docker-compose exec app bash
  
  # 在容器内检查文件系统
  ls -la /app
  ps aux
  ```

- [ ] **资源使用监控**
  ```bash
  # 监控容器资源使用
  docker-compose exec app top
  docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
  ```

---

## 性能和安全检查

### 性能基准测试

- [ ] **应用响应时间测试**
  ```bash
  # 使用curl测试响应时间
  time curl http://localhost:8000/health
  
  # 使用ab进行简单压力测试（如果安装了apache2-utils）
  # ab -n 100 -c 10 http://localhost:8000/
  ```

- [ ] **数据库连接性能**
  ```bash
  # 测试数据库查询性能
  time docker-compose exec postgres psql -U rag -d rag_db -c "SELECT 1;"
  ```

### 安全配置检查

- [ ] **容器用户权限**
  ```bash
  # 检查容器是否以非root用户运行
  docker-compose exec app whoami
  docker-compose exec app id
  ```

- [ ] **网络安全配置**
  ```bash
  # 检查网络隔离
  docker network inspect lesson02_rag-network
  ```

- [ ] **敏感信息保护**
  ```bash
  # 确保没有硬编码密码
  grep -r "password" docker-compose.yml || echo "No hardcoded passwords found"
  ```

---

## 清理和重置

### 环境清理

- [ ] **停止所有服务**
  ```bash
  # 停止服务栈
  docker-compose down
  ```

- [ ] **清理容器和镜像**（可选）
  ```bash
  # 删除所有容器
  docker-compose down --remove-orphans
  
  # 删除数据卷（注意：会丢失数据）
  docker-compose down -v
  
  # 清理未使用的镜像
  docker image prune -f
  ```

- [ ] **验证清理结果**
  ```bash
  # 检查是否还有相关容器运行
  docker ps -a | grep lesson02
  
  # 检查数据卷是否已删除
  docker volume ls | grep lesson02
  ```

---

## 检查清单总结

### 必须通过的检查项
- [ ] Docker环境正常运行
- [ ] 能够成功构建应用镜像
- [ ] 单容器可以正常启动和访问
- [ ] 多服务栈可以正常启动
- [ ] 所有服务间网络连通正常
- [ ] 数据持久化功能正常
- [ ] 基本故障排查能力具备

### 推荐完成的检查项
- [ ] 性能基准测试完成
- [ ] 安全配置检查通过
- [ ] 监控和日志分析能力具备
- [ ] 环境清理和重置能力具备

### 故障排查参考

**常见问题及解决方案**：

1. **端口冲突**
   ```bash
   # 查找占用端口的进程
   lsof -i :8000
   # 修改docker-compose.yml中的端口映射
   ```

2. **内存不足**
   ```bash
   # 检查系统内存使用
   free -h
   # 减少并发服务或增加系统内存
   ```

3. **网络连接失败**
   ```bash
   # 重启Docker服务
   sudo systemctl restart docker  # Linux
   # 或重启Docker Desktop
   ```

4. **镜像拉取失败**
   ```bash
   # 配置镜像加速器
   # 或使用代理设置
   ```

---

**检查完成标志**：当所有必须检查项都通过时，说明容器化环境已经正确配置，可以进行后续的开发和学习。