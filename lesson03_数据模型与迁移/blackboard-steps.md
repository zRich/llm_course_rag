# Lesson 03 黑板演示步骤

## 演示环境准备

### 开始前检查
- [ ] PostgreSQL容器服务已启动
- [ ] IDE已打开项目目录
- [ ] 终端已准备就绪
- [ ] 投屏设备工作正常

### 环境验证命令
```bash
# 检查PostgreSQL服务状态
docker-compose ps

# 验证数据库连接
psql -h localhost -U postgres -d rag_db -c "SELECT version();"
```

## 第一部分：SQLModel基础演示（30分钟）

### 演示1：创建用户模型（10分钟）

#### 步骤1：创建模型文件
```bash
# 在IDE中创建目录结构
mkdir -p src/models
touch src/models/__init__.py
touch src/models/user.py
```

#### 步骤2：编写基础模型
**在黑板/投屏上逐行展示：**
```python
# src/models/user.py
from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID, uuid4

# 第一步：定义基础模型
class UserBase(SQLModel):
    """用户基础模型 - 共享字段定义"""
    email: str = Field(unique=True, index=True, description="用户邮箱")
    username: str = Field(unique=True, index=True, description="用户名")
    is_active: bool = Field(default=True, description="是否激活")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
```

**讲解要点**：
- 解释`SQLModel`的继承关系
- 强调`Field`的约束参数
- 说明`default_factory`的用法

#### 步骤3：定义数据库表模型
```python
# 继续在user.py中添加
class User(UserBase, table=True):
    """用户表模型"""
    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    password_hash: str = Field(description="密码哈希")
    
    class Config:
        """模型配置"""
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "username": "testuser",
                "is_active": True
            }
        }
```

**演示重点**：
- `table=True`的作用
- UUID主键的优势
- 配置示例的重要性

### 演示2：数据库连接配置（10分钟）

#### 步骤1：创建连接文件
```bash
mkdir -p src/database
touch src/database/__init__.py
touch src/database/connection.py
```

#### 步骤2：配置数据库引擎
**在投屏上展示完整配置：**
```python
# src/database/connection.py
from sqlmodel import create_engine, SQLModel, Session
from sqlalchemy.pool import StaticPool
import os
from typing import Generator

# 数据库配置
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:password@localhost:5432/rag_db"
)

# 创建数据库引擎
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,      # 连接前检查
    pool_recycle=300,        # 连接回收时间（秒）
    pool_size=5,             # 连接池大小
    max_overflow=10,         # 最大溢出连接
    echo=True                # 开发环境显示SQL
)

def get_session() -> Generator[Session, None, None]:
    """获取数据库会话"""
    with Session(engine) as session:
        yield session

def create_tables():
    """创建所有数据库表"""
    SQLModel.metadata.create_all(engine)

def drop_tables():
    """删除所有数据库表（仅用于测试）"""
    SQLModel.metadata.drop_all(engine)
```

**演示要点**：
- 逐个解释连接池参数
- 强调环境变量的使用
- 展示会话管理的最佳实践

#### 步骤3：测试数据库连接
```python
# 在connection.py末尾添加测试函数
if __name__ == "__main__":
    # 测试数据库连接
    try:
        with Session(engine) as session:
            result = session.exec("SELECT 1").first()
            print(f"数据库连接成功: {result}")
    except Exception as e:
        print(f"数据库连接失败: {e}")
```

**在终端演示执行：**
```bash
cd src/database
python connection.py
```

### 演示3：模型关系设计（10分钟）

#### 步骤1：创建文档模型
```bash
touch src/models/document.py
```

#### 步骤2：定义文档模型
**在投屏上逐步构建：**
```python
# src/models/document.py
from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List, TYPE_CHECKING
from datetime import datetime
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from .user import User

class DocumentBase(SQLModel):
    """文档基础模型"""
    title: str = Field(index=True, description="文档标题")
    content: str = Field(description="文档内容")
    file_path: Optional[str] = Field(default=None, description="文件路径")
    file_size: Optional[int] = Field(default=None, description="文件大小(字节)")
    mime_type: Optional[str] = Field(default=None, description="MIME类型")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default=None)

class Document(DocumentBase, table=True):
    """文档表模型"""
    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id", description="所属用户ID")
    
    # 关系定义
    user: Optional["User"] = Relationship(back_populates="documents")
```

**讲解重点**：
- `TYPE_CHECKING`的作用
- 外键约束的定义
- 关系字段的配置

#### 步骤3：更新用户模型关系
```python
# 更新src/models/user.py，在User类中添加
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .document import Document

class User(UserBase, table=True):
    """用户表模型"""
    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    password_hash: str = Field(description="密码哈希")
    
    # 关系定义
    documents: List["Document"] = Relationship(back_populates="user")
```

## 第二部分：数据迁移演示（30分钟）

### 演示4：Alembic初始化（10分钟）

#### 步骤1：安装和初始化
```bash
# 在项目根目录执行
pip install alembic

# 初始化Alembic
alembic init migrations
```

**展示生成的文件结构：**
```
migrations/
├── versions/
├── env.py
├── script.py.mako
└── README
alembic.ini
```

#### 步骤2：配置Alembic
**编辑alembic.ini文件：**
```ini
# alembic.ini
[alembic]
script_location = migrations
prepend_sys_path = .
version_path_separator = os

# 数据库连接配置
sqlalchemy.url = postgresql://postgres:password@localhost:5432/rag_db

[post_write_hooks]

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

#### 步骤3：配置env.py
**编辑migrations/env.py：**
```python
# migrations/env.py
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 导入模型
from src.models.user import User
from src.models.document import Document
from sqlmodel import SQLModel

# Alembic配置对象
config = context.config

# 设置日志
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 设置MetaData
target_metadata = SQLModel.metadata

def run_migrations_offline() -> None:
    """离线模式运行迁移"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """在线模式运行迁移"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### 演示5：创建和执行迁移（10分钟）

#### 步骤1：生成初始迁移
```bash
# 创建初始迁移
alembic revision --autogenerate -m "Initial migration: add users and documents tables"
```

**展示生成的迁移文件：**
```python
# migrations/versions/xxx_initial_migration.py
"""Initial migration: add users and documents tables

Revision ID: abc123
Revises: 
Create Date: 2024-01-01 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel

# revision identifiers
revision = 'abc123'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('user',
    sa.Column('email', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('username', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.Column('password_hash', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_email'), 'user', ['email'], unique=True)
    op.create_index(op.f('ix_user_username'), 'user', ['username'], unique=True)
    
    op.create_table('document',
    sa.Column('title', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('content', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('file_path', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('file_size', sa.Integer(), nullable=True),
    sa.Column('mime_type', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.Column('id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.Column('user_id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_document_title'), 'document', ['title'], unique=False)
    # ### end Alembic commands ###

def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_document_title'), table_name='document')
    op.drop_table('document')
    op.drop_index(op.f('ix_user_username'), table_name='user')
    op.drop_index(op.f('ix_user_email'), table_name='user')
    op.drop_table('user')
    # ### end Alembic commands ###
```

#### 步骤2：执行迁移
```bash
# 查看当前迁移状态
alembic current

# 执行迁移
alembic upgrade head

# 验证迁移结果
alembic current
alembic history --verbose
```

#### 步骤3：验证数据库结构
```bash
# 连接数据库查看表结构
psql -h localhost -U postgres -d rag_db

# 在psql中执行
\dt  -- 查看所有表
\d user  -- 查看user表结构
\d document  -- 查看document表结构
```

### 演示6：迁移管理操作（10分钟）

#### 步骤1：添加新字段迁移
```python
# 修改src/models/user.py，添加新字段
class User(UserBase, table=True):
    """用户表模型"""
    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    password_hash: str = Field(description="密码哈希")
    last_login: Optional[datetime] = Field(default=None, description="最后登录时间")
    
    # 关系定义
    documents: List["Document"] = Relationship(back_populates="user")
```

```bash
# 生成新的迁移
alembic revision --autogenerate -m "Add last_login field to user table"

# 执行迁移
alembic upgrade head
```

#### 步骤2：迁移回滚演示
```bash
# 查看迁移历史
alembic history

# 回滚到上一个版本
alembic downgrade -1

# 验证回滚结果
alembic current

# 重新升级到最新版本
alembic upgrade head
```

#### 步骤3：迁移脚本手动编辑
**展示如何手动编辑迁移脚本：**
```python
# 在迁移文件中添加数据迁移逻辑
def upgrade() -> None:
    # 结构变更
    op.add_column('user', sa.Column('last_login', sa.DateTime(), nullable=True))
    
    # 数据迁移
    connection = op.get_bind()
    connection.execute(
        "UPDATE user SET last_login = created_at WHERE last_login IS NULL"
    )

def downgrade() -> None:
    op.drop_column('user', 'last_login')
```

## Exercise指导

### Exercise 1: SQLModel基础模型设计
**任务**：设计用户和文档基础模型
**时间**：15分钟
**指导要点**：
- 强调字段类型的正确性
- 检查约束条件的设置
- 验证模型定义的完整性

### Exercise 2: 数据库连接配置
**任务**：配置PostgreSQL连接池
**时间**：10分钟
**指导要点**：
- 连接字符串的格式
- 连接池参数的含义
- 环境变量的使用

### Exercise 3: 关系模型设计
**任务**：实现用户-文档关联关系
**时间**：15分钟
**指导要点**：
- 外键约束的定义
- 双向关系的建立
- 类型检查的处理

### Exercise 4: 迁移脚本编写
**任务**：使用Alembic创建和执行迁移
**时间**：20分钟
**指导要点**：
- Alembic配置的正确性
- 迁移文件的生成和编辑
- 迁移执行和验证

## 常见问题处理

### 问题1：模型导入错误
**现象**：循环导入或模块找不到
**解决方案**：
```python
# 使用TYPE_CHECKING避免循环导入
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .other_model import OtherModel
```

### 问题2：数据库连接失败
**现象**：连接超时或认证失败
**检查步骤**：
1. 验证PostgreSQL服务状态
2. 检查连接字符串格式
3. 确认用户权限设置

### 问题3：迁移执行失败
**现象**：表已存在或字段冲突
**解决方案**：
1. 检查数据库当前状态
2. 手动调整迁移脚本
3. 使用`--fake`标记跳过问题迁移

### 问题4：关系定义错误
**现象**：外键约束失败
**解决方案**：
1. 检查外键字段类型匹配
2. 确认关系方向正确
3. 验证表创建顺序

## 演示总结检查清单

### 技术要点覆盖
- [ ] SQLModel基础概念已讲解
- [ ] 数据模型设计原则已说明
- [ ] 数据库连接配置已演示
- [ ] 关系映射已正确实现
- [ ] Alembic迁移已成功执行

### 实践操作验证
- [ ] 所有代码都能正常运行
- [ ] 数据库表结构正确创建
- [ ] 迁移脚本可以正常执行
- [ ] 关系查询功能正常

### 学生理解检查
- [ ] 核心概念理解到位
- [ ] 实践操作能够跟上
- [ ] 常见问题得到解答
- [ ] Exercise任务明确清晰