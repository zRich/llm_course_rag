# 第17节课：结构化数据接入系统

本系统实现了完整的结构化数据接入功能，支持从多种数据源（API、数据库）获取数据，并与RAG系统集成。

## 系统架构

```
结构化数据接入系统
├── 数据连接器层
│   ├── DataConnector (基类)
│   ├── APIConnector (REST API连接器)
│   └── DatabaseConnector (数据库连接器)
├── 同步管理层
│   └── SyncManager (同步管理器)
├── 数据转换层
│   └── DataTransformer (数据转换器)
├── API接口层
│   └── FastAPI REST接口
└── RAG系统集成
    └── RAGIntegrationManager
```

## 核心功能

### 1. 数据连接器

#### APIConnector - REST API连接器
- 支持HTTP/HTTPS协议
- 自动重试机制
- 速率限制
- 分页数据获取
- 增量同步支持

#### DatabaseConnector - 数据库连接器
- 支持PostgreSQL、MySQL等主流数据库
- 连接池管理
- SQL查询执行
- 事务支持
- 增量同步支持

### 2. 同步管理器
- 全量同步和增量同步
- 同步状态跟踪
- 错误处理和重试
- 同步历史记录
- 并发同步控制

### 3. 数据转换
- 数据格式标准化
- 数据清洗和验证
- 字段映射
- 数据类型转换

### 4. RAG系统集成
- 自动将结构化数据转换为文档格式
- 与现有RAG系统无缝集成
- 支持增量更新
- 元数据保留

## 快速开始

### 1. 安装依赖

```bash
cd lesson17_structured_data
pip install -r requirements.txt
```

### 2. 启动API服务

```bash
python api.py
```

服务将在 `http://localhost:8001` 启动

### 3. 使用API接口

#### 创建API连接器

```bash
curl -X POST "http://localhost:8001/connectors" \
  -H "Content-Type: application/json" \
  -d '{
    "connector_id": "my_api",
    "connector_type": "api",
    "config": {
      "base_url": "https://jsonplaceholder.typicode.com",
      "headers": {"Content-Type": "application/json"}
    }
  }'
```

#### 测试连接器

```bash
curl -X POST "http://localhost:8001/connectors/my_api/test"
```

#### 开始同步

```bash
curl -X POST "http://localhost:8001/sync/start" \
  -H "Content-Type: application/json" \
  -d '{
    "connector_id": "my_api",
    "sync_type": "full",
    "endpoint": "/posts"
  }'
```

#### 查看同步状态

```bash
curl "http://localhost:8001/sync/status/{sync_id}"
```

### 4. 数据库连接器示例

```bash
curl -X POST "http://localhost:8001/connectors" \
  -H "Content-Type: application/json" \
  -d '{
    "connector_id": "my_db",
    "connector_type": "database",
    "config": {
      "host": "localhost",
      "port": 5432,
      "database": "mydb",
      "username": "user",
      "password": "password"
    }
  }'
```

## 与RAG系统集成

### 1. 启动集成的RAG系统

```bash
# 在rag-system根目录下
cd ..
python -m src.api.health
```

集成后的RAG系统将在 `http://localhost:8000` 启动，结构化数据API在 `/structured-data` 路径下可用。

### 2. 运行集成演示

```bash
python integration_demo.py
```

这将演示：
- API数据源集成
- 数据库数据源集成
- 数据转换为文档格式
- 自动索引到RAG系统

## 测试

### 运行所有测试

```bash
python -m pytest tests/ -v
```

### 运行特定测试

```bash
# API连接器测试
python -m pytest tests/test_api_connector.py -v

# 数据库连接器测试
python -m pytest tests/test_database_connector.py -v

# 同步管理器测试
python -m pytest tests/test_sync_manager.py -v

# API接口测试
python -m pytest tests/test_api.py -v
```

## 配置说明

### API连接器配置

```json
{
  "base_url": "https://api.example.com",
  "headers": {
    "Authorization": "Bearer your-token",
    "Content-Type": "application/json"
  },
  "timeout": 30,
  "max_retries": 3,
  "rate_limit": {
    "requests_per_second": 10
  }
}
```

### 数据库连接器配置

```json
{
  "host": "localhost",
  "port": 5432,
  "database": "mydb",
  "username": "user",
  "password": "password",
  "ssl_mode": "prefer",
  "pool_size": 5,
  "max_overflow": 10
}
```

## 高级功能

### 1. 增量同步

增量同步基于时间戳字段，自动检测数据变更：

```python
# 配置增量同步字段
connector_config = {
    "incremental_field": "updated_at",
    "incremental_field_type": "timestamp"
}
```

### 2. 数据转换

支持自定义数据转换规则：

```python
# 字段映射
field_mapping = {
    "user_name": "name",
    "user_email": "email",
    "created_time": "created_at"
}

# 数据过滤
filter_rules = {
    "status": "active",
    "type": ["user", "admin"]
}
```

### 3. 错误处理

系统提供完善的错误处理机制：
- 连接错误自动重试
- 数据验证失败记录
- 同步失败回滚
- 详细错误日志

### 4. 监控和日志

- 同步进度跟踪
- 性能指标收集
- 详细操作日志
- 错误告警

## 故障排除

### 常见问题

1. **连接失败**
   - 检查网络连接
   - 验证认证信息
   - 确认防火墙设置

2. **同步缓慢**
   - 调整批次大小
   - 优化查询语句
   - 检查网络带宽

3. **数据不一致**
   - 检查增量字段配置
   - 验证时区设置
   - 确认数据源时间戳

### 日志查看

```bash
# 查看API日志
tail -f logs/api.log

# 查看同步日志
tail -f logs/sync.log
```

## 扩展开发

### 添加新的连接器类型

1. 继承 `DataConnector` 基类
2. 实现必要的抽象方法
3. 注册到连接器工厂
4. 添加相应测试

```python
class CustomConnector(DataConnector):
    async def connect(self) -> bool:
        # 实现连接逻辑
        pass
    
    async def fetch_data(self, **kwargs) -> List[Dict[str, Any]]:
        # 实现数据获取逻辑
        pass
```

### 自定义数据转换器

```python
class CustomTransformer(DataTransformer):
    def transform_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        # 实现自定义转换逻辑
        return transformed_record
```

## 性能优化

1. **批量处理**：使用适当的批次大小
2. **并发控制**：合理设置并发数量
3. **连接池**：复用数据库连接
4. **缓存机制**：缓存频繁访问的数据
5. **索引优化**：为增量字段创建索引

## 安全考虑

1. **认证信息**：使用环境变量存储敏感信息
2. **网络安全**：使用HTTPS和SSL连接
3. **访问控制**：限制API访问权限
4. **数据加密**：对敏感数据进行加密
5. **审计日志**：记录所有操作日志

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 添加测试
5. 提交 Pull Request

## 许可证

MIT License