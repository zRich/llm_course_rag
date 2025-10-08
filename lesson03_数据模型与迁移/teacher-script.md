# Lesson 03 教师讲稿：数据模型与迁移

## 课程时间轴（120分钟）

### 第一阶段：导入与概念讲解（30分钟）
- **0-10分钟**: 课程导入，回顾上节课内容
- **10-25分钟**: SQLModel核心概念讲解
- **25-30分钟**: 数据模型设计原则

### 第二阶段：实践演示（60分钟）
- **30-50分钟**: Exercise 1 & 2 - 基础模型设计与数据库连接
- **50-70分钟**: Exercise 3 & 4 - 关系模型与迁移脚本
- **70-90分钟**: 学生实践时间

### 第三阶段：总结与答疑（30分钟）
- **90-105分钟**: 成果展示与问题解答
- **105-115分钟**: Lab 03任务说明
- **115-120分钟**: 课程总结与下节预告

## 教师讲稿（可直接照读）

### 开场白（5分钟）
"同学们好，欢迎来到第三课：数据模型与迁移。今天我们将学习如何为RAG系统设计高效的数据模型，并掌握数据库迁移的最佳实践。

**今天的核心目标**：
1. 掌握SQLModel的使用方法
2. 设计RAG系统的核心数据模型
3. 实现自动化的数据库迁移
4. 建立数据访问层的基础架构

**今天的产出**：
- 完整的数据模型定义
- 可执行的迁移脚本
- 数据库连接配置
- 为下节课API开发做好数据层准备"

### 核心概念讲解（20分钟）

#### SQLModel概述
"首先，让我们了解SQLModel。SQLModel是FastAPI作者开发的现代Python SQL工具包，它有三个核心特点：

1. **类型安全**：完整的类型提示支持
2. **自动补全**：IDE友好的开发体验  
3. **运行时验证**：结合了Pydantic的数据验证能力

**为什么选择SQLModel？**
- 统一了Pydantic和SQLAlchemy的优点
- 与FastAPI无缝集成
- 现代Python开发的最佳实践"

#### 数据模型设计原则
"在设计数据模型时，我们需要遵循以下原则：

1. **单一职责**：每个模型只负责一个业务实体
2. **关系清晰**：明确定义实体间的关联关系
3. **扩展性**：考虑未来的业务扩展需求
4. **性能优化**：合理设计索引和查询路径"

### Exercise 1: SQLModel基础模型设计（20分钟）

#### 演示脚本
"现在让我们开始第一个Exercise：设计用户和文档的基础模型。

**步骤1：创建基础用户模型**
```python
# src/models/user.py
from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID, uuid4

class UserBase(SQLModel):
    email: str = Field(unique=True, index=True)
    username: str = Field(unique=True, index=True)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class User(UserBase, table=True):
    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    password_hash: str
```

**关键点解释**：
- `UserBase`：共享字段的基类
- `table=True`：标记为数据库表
- `Field`：定义字段约束和索引
- `UUID`：使用UUID作为主键的优势"

#### 学生实践指导
"请同学们按照以下步骤完成用户模型的设计：

1. 创建`src/models/user.py`文件
2. 定义`UserBase`和`User`类
3. 添加必要的字段约束
4. 验证模型定义的正确性

**常见问题**：
- 忘记导入必要的模块
- 字段类型定义不正确
- 缺少必要的约束条件"

### Exercise 2: 数据库连接配置（15分钟）

#### 演示脚本
"接下来配置数据库连接。我们使用连接池来提高性能：

```python
# src/database/connection.py
from sqlmodel import create_engine, SQLModel, Session
from sqlalchemy.pool import StaticPool
import os

# 数据库URL配置
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:password@localhost:5432/rag_db"
)

# 创建引擎
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=True  # 开发环境显示SQL
)

# 创建会话
def get_session():
    with Session(engine) as session:
        yield session

# 创建所有表
def create_tables():
    SQLModel.metadata.create_all(engine)
```

**配置说明**：
- `pool_pre_ping`：连接健康检查
- `pool_recycle`：连接回收时间
- `echo=True`：开发环境显示SQL语句"

### Exercise 3: 关系模型设计（15分钟）

#### 演示脚本
"现在设计文档模型和用户-文档关系：

```python
# src/models/document.py
from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from datetime import datetime
from uuid import UUID, uuid4

class DocumentBase(SQLModel):
    title: str = Field(index=True)
    content: str
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

class Document(DocumentBase, table=True):
    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id")
    
    # 关系定义
    user: Optional["User"] = Relationship(back_populates="documents")

# 更新User模型
class User(UserBase, table=True):
    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    password_hash: str
    
    # 关系定义
    documents: List["Document"] = Relationship(back_populates="user")
```

**关系设计要点**：
- 外键约束的正确定义
- 双向关系的建立
- 懒加载与预加载的选择"

### Exercise 4: 迁移脚本编写（10分钟）

#### 演示脚本
"最后，我们使用Alembic创建迁移脚本：

**步骤1：初始化Alembic**
```bash
# 在项目根目录执行
alembic init migrations
```

**步骤2：配置alembic.ini**
```ini
# alembic.ini
sqlalchemy.url = postgresql://postgres:password@localhost:5432/rag_db
```

**步骤3：创建迁移**
```bash
# 创建初始迁移
alembic revision --autogenerate -m "Initial migration"

# 执行迁移
alembic upgrade head
```

**步骤4：验证迁移**
```bash
# 检查迁移状态
alembic current

# 查看迁移历史
alembic history
```"

## 课堂提问与Exercise

### 关键概念问题
1. **Q**: "SQLModel与传统ORM的主要区别是什么？"
   **A**: "SQLModel结合了Pydantic的类型验证和SQLAlchemy的ORM功能，提供了更好的类型安全和IDE支持。"

2. **Q**: "为什么使用UUID作为主键？"
   **A**: "UUID提供了全局唯一性，避免了分布式环境下的ID冲突，同时增强了安全性。"

3. **Q**: "数据库连接池的作用是什么？"
   **A**: "连接池复用数据库连接，减少连接建立和销毁的开销，提高应用性能。"

### 实践Exercise问题
1. **Exercise**: "请设计一个评论模型，包含用户和文档的关联关系"
2. **Exercise**: "如何为文档标题字段添加全文搜索索引？"
3. **Exercise**: "设计一个软删除机制，不物理删除数据"

## 常见误区与应对话术

### 常见错误1：字段类型定义错误
**现象**: 学生使用Python基础类型而不是SQLModel字段类型
**话术**: "注意区分Python类型和SQLModel字段类型。我们需要使用`Field()`来定义数据库字段的约束条件。"

### 常见错误2：关系定义混乱
**现象**: 外键和关系定义不匹配
**话术**: "外键定义了数据库层面的约束，而Relationship定义了ORM层面的关联。两者需要保持一致。"

### 常见错误3：迁移脚本执行失败
**现象**: 数据库连接或权限问题
**话术**: "检查数据库连接字符串和用户权限。确保PostgreSQL服务正在运行，并且用户有创建表的权限。"

## 黑板/投屏操作步骤

### 演示1：模型定义展示
1. 打开IDE，创建新文件`src/models/user.py`
2. 逐行输入用户模型代码，解释每个字段的作用
3. 展示IDE的类型提示和自动补全功能
4. 运行代码验证语法正确性

### 演示2：数据库连接测试
1. 创建`src/database/connection.py`文件
2. 配置数据库连接字符串
3. 测试数据库连接是否成功
4. 展示连接池的配置参数

### 演示3：迁移脚本执行
1. 在终端中执行Alembic命令
2. 展示生成的迁移文件内容
3. 执行迁移并验证数据库表结构
4. 演示迁移的回滚操作

### 演示4：错误处理展示
1. 故意制造常见错误（如字段类型错误）
2. 展示错误信息的解读方法
3. 演示调试和修复过程
4. 强调代码规范的重要性

## 课后反思记录

### 教学效果评估
- [ ] 学生对SQLModel概念的理解程度
- [ ] 数据模型设计的完成质量
- [ ] 迁移脚本执行的成功率
- [ ] 课堂互动和提问的积极性

### 需要改进的地方
- [ ] 概念讲解的深度和广度
- [ ] 实践Exercise的难度设置
- [ ] 时间分配的合理性
- [ ] 错误处理的覆盖面

### 学生反馈收集
- [ ] 课程内容的难易程度
- [ ] 实践Exercise的实用性
- [ ] 教学节奏的适宜性
- [ ] 需要补充的知识点

## Lab 03任务说明

### 任务目标
设计并实现RAG系统的完整数据模型，包括：
- 用户管理模型
- 文档存储模型
- 向量数据模型
- 知识库组织模型

### 技术要求
- 使用SQLModel定义所有模型
- 实现完整的关系映射
- 添加必要的索引和约束
- 编写完整的迁移脚本

### 提交标准
- 代码结构清晰，符合规范
- 模型设计合理，关系明确
- 迁移脚本可正常执行
- 包含完整的测试用例

### 评估重点
- 数据模型的设计质量
- 代码的可维护性
- 性能优化的考虑
- 文档的完整性