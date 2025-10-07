# 实验4：工程化部署实验（Lesson 15-18）

## 实验概述

本实验是RAG实战课程的第四个综合实验，专注于将前三个实验构建的RAG系统进行工程化改造和生产环境部署。学生将学习如何通过数据管道工程化、API网关集成、容器化部署和CI/CD流程来构建一个可在生产环境稳定运行的企业级RAG系统。

## 实验目标

- 构建完整的数据摄取和处理管道
- 实现企业级API网关和服务治理
- 掌握容器化部署和编排技术
- 建立完整的CI/CD开发运维流程
- 实现生产环境的监控和日志系统

## 涉及课程

- **Lesson 15**：Ingestion数据管道
- **Lesson 16**：API网关与服务治理
- **Lesson 17**：容器化与编排
- **Lesson 18**：CI/CD与自动化部署

## 技术栈

### 核心技术组件
- **容器化**：Docker、Docker Compose
- **编排工具**：Kubernetes、Helm
- **API网关**：Kong、Traefik、Nginx
- **消息队列**：RabbitMQ、Apache Kafka
- **工作流引擎**：Apache Airflow、Celery
- **CI/CD**：GitHub Actions、GitLab CI、Jenkins
- **监控系统**：Prometheus、Grafana、ELK Stack
- **服务网格**：Istio（可选）

### 继承技术栈
- 实验1-3的所有技术组件

## 前置条件

- 完成实验1、2、3
- 熟悉Docker和容器化概念
- 了解微服务架构原理
- 具备基础的DevOps知识
- 理解API设计和网关概念

## 实验步骤

### 第一阶段：数据摄取管道工程化（Lesson 15）

1. **数据管道架构设计**
   ```python
   # 数据管道配置
   class IngestionPipeline:
       def __init__(self, config: PipelineConfig):
           self.extractors = self.init_extractors(config.sources)
           self.transformers = self.init_transformers(config.transforms)
           self.loaders = self.init_loaders(config.targets)
           self.scheduler = self.init_scheduler(config.schedule)
       
       async def run_pipeline(self, pipeline_id: str):
           """执行完整的ETL管道"""
           try:
               # Extract
               raw_data = await self.extract_data()
               
               # Transform
               processed_data = await self.transform_data(raw_data)
               
               # Load
               await self.load_data(processed_data)
               
               # Update metadata
               await self.update_pipeline_metadata(pipeline_id)
               
           except Exception as e:
               await self.handle_pipeline_error(pipeline_id, e)
   ```

2. **多源数据连接器**
   ```python
   from abc import ABC, abstractmethod
   
   class DataConnector(ABC):
       @abstractmethod
       async def extract(self) -> List[Document]:
           pass
   
   class FileSystemConnector(DataConnector):
       async def extract(self) -> List[Document]:
           # 文件系统数据提取
           pass
   
   class DatabaseConnector(DataConnector):
       async def extract(self) -> List[Document]:
           # 数据库数据提取
           pass
   
   class APIConnector(DataConnector):
       async def extract(self) -> List[Document]:
           # API数据提取
           pass
   
   class WebCrawlerConnector(DataConnector):
       async def extract(self) -> List[Document]:
           # 网页爬虫数据提取
           pass
   ```

3. **流式数据处理**
   ```python
   import asyncio
   from kafka import KafkaConsumer, KafkaProducer
   
   class StreamProcessor:
       def __init__(self, kafka_config: dict):
           self.consumer = KafkaConsumer(**kafka_config)
           self.producer = KafkaProducer(**kafka_config)
       
       async def process_stream(self):
           """处理实时数据流"""
           async for message in self.consumer:
               try:
                   # 解析消息
                   document = self.parse_message(message.value)
                   
                   # 处理文档
                   processed_doc = await self.process_document(document)
                   
                   # 向量化
                   embeddings = await self.vectorize_document(processed_doc)
                   
                   # 存储到向量数据库
                   await self.store_embeddings(embeddings)
                   
                   # 发送处理完成消息
                   await self.send_completion_message(processed_doc.id)
                   
               except Exception as e:
                   await self.handle_processing_error(message, e)
   ```

4. **数据质量监控**
   ```python
   class DataQualityMonitor:
       def __init__(self):
           self.quality_rules = self.load_quality_rules()
           self.metrics_collector = MetricsCollector()
       
       async def validate_data_quality(self, data: List[Document]) -> QualityReport:
           """数据质量验证"""
           report = QualityReport()
           
           for rule in self.quality_rules:
               violations = await rule.check(data)
               report.add_violations(violations)
           
           # 收集质量指标
           await self.metrics_collector.record_quality_metrics(report)
           
           return report
   ```

### 第二阶段：API网关与服务治理（Lesson 16）

1. **API网关配置**
   ```yaml
   # kong.yml - Kong API网关配置
   _format_version: "3.0"
   
   services:
   - name: rag-service
     url: http://rag-backend:8000
     plugins:
     - name: rate-limiting
       config:
         minute: 100
         hour: 1000
     - name: key-auth
     - name: prometheus
   
   routes:
   - name: rag-query
     service: rag-service
     paths:
     - /api/v1/query
     methods:
     - POST
   
   - name: rag-documents
     service: rag-service
     paths:
     - /api/v1/documents
     methods:
     - GET
     - POST
     - PUT
     - DELETE
   ```

2. **服务发现和负载均衡**
   ```python
   from consul import Consul
   import random
   
   class ServiceDiscovery:
       def __init__(self, consul_host: str = 'localhost', consul_port: int = 8500):
           self.consul = Consul(host=consul_host, port=consul_port)
       
       async def register_service(self, service_name: str, service_id: str, 
                                address: str, port: int):
           """注册服务"""
           self.consul.agent.service.register(
               name=service_name,
               service_id=service_id,
               address=address,
               port=port,
               check=Consul.Check.http(f"http://{address}:{port}/health", 
                                     interval="10s")
           )
       
       async def discover_service(self, service_name: str) -> str:
           """服务发现"""
           services = self.consul.health.service(service_name, passing=True)[1]
           if not services:
               raise ServiceNotAvailableError(f"No healthy {service_name} instances")
           
           # 简单的随机负载均衡
           service = random.choice(services)
           return f"http://{service['Service']['Address']}:{service['Service']['Port']}"
   ```

3. **API版本管理**
   ```python
   from fastapi import FastAPI, APIRouter
   from fastapi.middleware.cors import CORSMiddleware
   
   app = FastAPI(title="RAG System API", version="2.0.0")
   
   # API版本路由
   v1_router = APIRouter(prefix="/api/v1", tags=["v1"])
   v2_router = APIRouter(prefix="/api/v2", tags=["v2"])
   
   @v1_router.post("/query")
   async def query_v1(request: QueryRequestV1) -> QueryResponseV1:
       """V1版本的查询接口"""
       return await process_query_v1(request)
   
   @v2_router.post("/query")
   async def query_v2(request: QueryRequestV2) -> QueryResponseV2:
       """V2版本的查询接口，支持更多功能"""
       return await process_query_v2(request)
   
   app.include_router(v1_router)
   app.include_router(v2_router)
   ```

4. **API安全和认证**
   ```python
   from fastapi import Depends, HTTPException, status
   from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
   import jwt
   
   security = HTTPBearer()
   
   class AuthManager:
       def __init__(self, secret_key: str):
           self.secret_key = secret_key
       
       def create_access_token(self, user_id: str, permissions: List[str]) -> str:
           """创建访问令牌"""
           payload = {
               "user_id": user_id,
               "permissions": permissions,
               "exp": datetime.utcnow() + timedelta(hours=24)
           }
           return jwt.encode(payload, self.secret_key, algorithm="HS256")
       
       def verify_token(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
           """验证访问令牌"""
           try:
               payload = jwt.decode(credentials.credentials, self.secret_key, algorithms=["HS256"])
               return payload
           except jwt.ExpiredSignatureError:
               raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
           except jwt.JWTError:
               raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
   ```

### 第三阶段：容器化与编排（Lesson 17）

1. **Docker容器化**
   ```dockerfile
   # Dockerfile
   FROM python:3.12-slim
   
   WORKDIR /app
   
   # 安装系统依赖
   RUN apt-get update && apt-get install -y \
       gcc \
       g++ \
       && rm -rf /var/lib/apt/lists/*
   
   # 复制依赖文件
   COPY pyproject.toml uv.lock ./
   
   # 安装uv和依赖
   RUN pip install uv
   RUN uv sync --frozen
   
   # 复制应用代码
   COPY . .
   
   # 设置环境变量
   ENV PYTHONPATH=/app
   ENV PYTHONUNBUFFERED=1
   
   # 健康检查
   HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
       CMD curl -f http://localhost:8000/health || exit 1
   
   # 暴露端口
   EXPOSE 8000
   
   # 启动命令
   CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Docker Compose编排**
   ```yaml
   # docker-compose.yml
   version: '3.8'
   
   services:
     rag-backend:
       build: .
       ports:
         - "8000:8000"
       environment:
         - DATABASE_URL=postgresql://user:password@postgres:5432/ragdb
         - REDIS_URL=redis://redis:6379
         - QDRANT_URL=http://qdrant:6333
       depends_on:
         - postgres
         - redis
         - qdrant
       volumes:
         - ./logs:/app/logs
       networks:
         - rag-network
   
     postgres:
       image: postgres:15
       environment:
         POSTGRES_DB: ragdb
         POSTGRES_USER: user
         POSTGRES_PASSWORD: password
       volumes:
         - postgres_data:/var/lib/postgresql/data
       networks:
         - rag-network
   
     redis:
       image: redis:7-alpine
       volumes:
         - redis_data:/data
       networks:
         - rag-network
   
     qdrant:
       image: qdrant/qdrant:latest
       ports:
         - "6333:6333"
       volumes:
         - qdrant_data:/qdrant/storage
       networks:
         - rag-network
   
     nginx:
       image: nginx:alpine
       ports:
         - "80:80"
         - "443:443"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf
         - ./ssl:/etc/nginx/ssl
       depends_on:
         - rag-backend
       networks:
         - rag-network
   
   volumes:
     postgres_data:
     redis_data:
     qdrant_data:
   
   networks:
     rag-network:
       driver: bridge
   ```

3. **Kubernetes部署**
   ```yaml
   # k8s/deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: rag-backend
     labels:
       app: rag-backend
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: rag-backend
     template:
       metadata:
         labels:
           app: rag-backend
       spec:
         containers:
         - name: rag-backend
           image: rag-system:latest
           ports:
           - containerPort: 8000
           env:
           - name: DATABASE_URL
             valueFrom:
               secretKeyRef:
                 name: rag-secrets
                 key: database-url
           - name: REDIS_URL
             value: "redis://redis-service:6379"
           resources:
             requests:
               memory: "512Mi"
               cpu: "250m"
             limits:
               memory: "1Gi"
               cpu: "500m"
           livenessProbe:
             httpGet:
               path: /health
               port: 8000
             initialDelaySeconds: 30
             periodSeconds: 10
           readinessProbe:
             httpGet:
               path: /ready
               port: 8000
             initialDelaySeconds: 5
             periodSeconds: 5
   
   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: rag-backend-service
   spec:
     selector:
       app: rag-backend
     ports:
     - protocol: TCP
       port: 80
       targetPort: 8000
     type: LoadBalancer
   ```

4. **Helm Chart管理**
   ```yaml
   # helm/rag-system/values.yaml
   replicaCount: 3
   
   image:
     repository: rag-system
     tag: latest
     pullPolicy: IfNotPresent
   
   service:
     type: LoadBalancer
     port: 80
     targetPort: 8000
   
   ingress:
     enabled: true
     className: nginx
     annotations:
       cert-manager.io/cluster-issuer: letsencrypt-prod
     hosts:
       - host: rag.example.com
         paths:
           - path: /
             pathType: Prefix
     tls:
       - secretName: rag-tls
         hosts:
           - rag.example.com
   
   resources:
     limits:
       cpu: 500m
       memory: 1Gi
     requests:
       cpu: 250m
       memory: 512Mi
   
   autoscaling:
     enabled: true
     minReplicas: 2
     maxReplicas: 10
     targetCPUUtilizationPercentage: 80
   ```

### 第四阶段：CI/CD与自动化部署（Lesson 18）

1. **GitHub Actions工作流**
   ```yaml
   # .github/workflows/ci-cd.yml
   name: CI/CD Pipeline
   
   on:
     push:
       branches: [ main, develop ]
     pull_request:
       branches: [ main ]
   
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
       - uses: actions/checkout@v3
       
       - name: Set up Python
         uses: actions/setup-python@v4
         with:
           python-version: '3.12'
       
       - name: Install uv
         run: pip install uv
       
       - name: Install dependencies
         run: uv sync
       
       - name: Run tests
         run: uv run pytest tests/ --cov=src --cov-report=xml
       
       - name: Upload coverage
         uses: codecov/codecov-action@v3
         with:
           file: ./coverage.xml
   
     build:
       needs: test
       runs-on: ubuntu-latest
       if: github.ref == 'refs/heads/main'
       steps:
       - uses: actions/checkout@v3
       
       - name: Set up Docker Buildx
         uses: docker/setup-buildx-action@v2
       
       - name: Login to Container Registry
         uses: docker/login-action@v2
         with:
           registry: ghcr.io
           username: ${{ github.actor }}
           password: ${{ secrets.GITHUB_TOKEN }}
       
       - name: Build and push
         uses: docker/build-push-action@v4
         with:
           context: .
           push: true
           tags: |
             ghcr.io/${{ github.repository }}:latest
             ghcr.io/${{ github.repository }}:${{ github.sha }}
   
     deploy:
       needs: build
       runs-on: ubuntu-latest
       if: github.ref == 'refs/heads/main'
       steps:
       - uses: actions/checkout@v3
       
       - name: Deploy to Kubernetes
         uses: azure/k8s-deploy@v1
         with:
           manifests: |
             k8s/deployment.yaml
             k8s/service.yaml
           images: |
             ghcr.io/${{ github.repository }}:${{ github.sha }}
           kubeconfig: ${{ secrets.KUBE_CONFIG }}
   ```

2. **自动化测试策略**
   ```python
   # tests/integration/test_deployment.py
   import pytest
   import requests
   import time
   
   class TestDeployment:
       @pytest.fixture(scope="class")
       def deployment_url(self):
           """获取部署环境URL"""
           return os.getenv("DEPLOYMENT_URL", "http://localhost:8000")
       
       def test_health_check(self, deployment_url):
           """健康检查测试"""
           response = requests.get(f"{deployment_url}/health")
           assert response.status_code == 200
           assert response.json()["status"] == "healthy"
       
       def test_api_functionality(self, deployment_url):
           """API功能测试"""
           query_data = {
               "query": "What is RAG?",
               "top_k": 5
           }
           response = requests.post(f"{deployment_url}/api/v1/query", json=query_data)
           assert response.status_code == 200
           assert "answer" in response.json()
       
       def test_load_performance(self, deployment_url):
           """负载性能测试"""
           import concurrent.futures
           
           def make_request():
               response = requests.post(f"{deployment_url}/api/v1/query", 
                                      json={"query": "test query"})
               return response.status_code == 200
           
           with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
               futures = [executor.submit(make_request) for _ in range(50)]
               results = [future.result() for future in futures]
           
           success_rate = sum(results) / len(results)
           assert success_rate >= 0.95  # 95%成功率
   ```

3. **蓝绿部署策略**
   ```python
   # scripts/blue_green_deploy.py
   import subprocess
   import time
   import requests
   
   class BlueGreenDeployer:
       def __init__(self, config):
           self.config = config
           self.current_env = self.get_current_environment()
       
       def deploy(self, new_image_tag: str):
           """执行蓝绿部署"""
           target_env = "green" if self.current_env == "blue" else "blue"
           
           try:
               # 1. 部署到目标环境
               self.deploy_to_environment(target_env, new_image_tag)
               
               # 2. 健康检查
               if not self.health_check(target_env):
                   raise DeploymentError("Health check failed")
               
               # 3. 运行烟雾测试
               if not self.smoke_test(target_env):
                   raise DeploymentError("Smoke test failed")
               
               # 4. 切换流量
               self.switch_traffic(target_env)
               
               # 5. 验证切换
               time.sleep(30)
               if not self.verify_traffic_switch(target_env):
                   self.rollback()
                   raise DeploymentError("Traffic switch verification failed")
               
               # 6. 清理旧环境
               self.cleanup_old_environment(self.current_env)
               
               print(f"Successfully deployed to {target_env}")
               
           except Exception as e:
               print(f"Deployment failed: {e}")
               self.rollback()
               raise
   ```

## 实验任务

### 任务1：数据管道工程化

**目标**：构建完整的数据摄取和处理管道

**具体要求**：
1. 支持至少4种不同的数据源
2. 实现实时和批处理两种模式
3. 包含数据质量监控和告警
4. 提供管道状态监控界面

**评估指标**：
- 数据处理吞吐量
- 数据质量得分
- 管道可用性
- 错误恢复时间

### 任务2：API网关集成

**目标**：实现企业级API网关和服务治理

**具体要求**：
1. 配置API网关（Kong/Traefik）
2. 实现服务发现和负载均衡
3. 添加认证、授权、限流功能
4. 支持API版本管理

**评估指标**：
- API响应时间
- 服务可用性
- 安全性测试通过率
- 负载均衡效果

### 任务3：容器化部署

**目标**：实现完整的容器化部署方案

**具体要求**：
1. 创建优化的Docker镜像
2. 编写Docker Compose配置
3. 实现Kubernetes部署
4. 配置Helm Chart

**评估指标**：
- 镜像大小优化
- 容器启动时间
- 资源使用效率
- 部署成功率

### 任务4：CI/CD流程

**目标**：建立完整的CI/CD自动化流程

**具体要求**：
1. 配置自动化测试流水线
2. 实现自动化构建和部署
3. 添加部署策略（蓝绿/滚动）
4. 建立监控和告警机制

**评估指标**：
- 部署频率
- 部署成功率
- 回滚时间
- 测试覆盖率

## 部署架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        Load Balancer                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                     API Gateway                             │
│                   (Kong/Traefik)                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                 Kubernetes Cluster                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ RAG Service │  │ RAG Service │  │ RAG Service │         │
│  │   Pod 1     │  │   Pod 2     │  │   Pod 3     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ PostgreSQL  │  │   Redis     │  │   Qdrant    │         │
│  │  Cluster    │  │  Cluster    │  │  Cluster    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## 监控和日志

### 监控配置
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'rag-backend'
    static_configs:
      - targets: ['rag-backend:8000']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

### 日志聚合
```yaml
# logging/fluentd.conf
<source>
  @type forward
  port 24224
  bind 0.0.0.0
</source>

<match rag.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name rag-logs
  type_name _doc
  
  <buffer>
    @type file
    path /var/log/fluentd-buffers/rag.buffer
    flush_mode interval
    flush_interval 10s
  </buffer>
</match>
```

## 评估标准

### 工程化实现（40分）

- [ ] 数据管道完整性（10分）
- [ ] API网关配置（10分）
- [ ] 容器化质量（10分）
- [ ] CI/CD流程（10分）

### 部署质量（30分）

- [ ] 部署成功率（10分）
- [ ] 性能表现（10分）
- [ ] 监控完整性（10分）

### 运维能力（30分）

- [ ] 故障恢复能力（10分）
- [ ] 扩展性设计（10分）
- [ ] 安全性配置（10分）

## 常见问题

### 部署问题

**Q: 容器启动失败？**
A: 检查资源限制、环境变量配置、依赖服务状态。

**Q: 服务发现不工作？**
A: 验证网络配置、服务注册、健康检查设置。

**Q: 负载均衡不均匀？**
A: 检查负载均衡算法、服务权重、健康状态。

### 性能问题

**Q: 部署后性能下降？**
A: 分析资源使用、网络延迟、配置差异。

**Q: 扩容不生效？**
A: 检查HPA配置、指标收集、扩容策略。

## 参考资源

- [Docker最佳实践](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes部署指南](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [Kong API网关文档](https://docs.konghq.com/)
- [GitHub Actions文档](https://docs.github.com/en/actions)
- [Prometheus监控指南](https://prometheus.io/docs/guides/)

## 实验时间安排

- **理论学习**：4-5小时（DevOps和容器化理论）
- **数据管道开发**：8-10小时（ETL管道实现）
- **API网关配置**：6-8小时（网关和服务治理）
- **容器化实现**：8-10小时（Docker和K8s部署）
- **CI/CD配置**：6-8小时（自动化流水线）
- **监控配置**：4-6小时（监控和日志系统）
- **测试和优化**：6-8小时（部署测试和调优）
- **文档编写**：3-4小时

**总计**：45-59小时

## 提交要求

1. **完整的部署包**：包含所有配置文件和脚本
2. **CI/CD流水线**：可运行的自动化部署流程
3. **监控仪表板**：实时监控和告警系统
4. **部署文档**：详细的部署和运维指南
5. **性能测试报告**：生产环境性能基准

## 后续实验预告

完成本实验后，学生将进入最后的实验5：系统可靠性实验，学习如何通过故障注入、自动恢复和灾难恢复来确保RAG系统在生产环境中的高可用性。