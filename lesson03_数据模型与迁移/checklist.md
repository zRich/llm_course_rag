# Lesson 03 检查清单

## 课前准备检查清单

### 环境准备
- [ ] PostgreSQL容器服务已启动并可连接
- [ ] Python开发环境已配置（Python 3.8+）
- [ ] 必要的Python包已安装：
  - [ ] sqlmodel
  - [ ] psycopg2-binary
  - [ ] alembic
  - [ ] fastapi
  - [ ] uvicorn
- [ ] IDE已打开项目目录
- [ ] 投屏设备测试正常
- [ ] 网络连接稳定

### 教学材料准备
- [ ] 教师讲稿已熟悉
- [ ] 黑板演示步骤已准备
- [ ] 代码示例已测试验证
- [ ] 学生Exercise指导已更新
- [ ] Exercise模板文件已准备

### 数据库环境验证
```bash
# 执行以下命令验证环境
docker-compose ps
psql -h localhost -U postgres -d rag_db -c "SELECT version();"
```

## 课堂教学检查清单

### 第一阶段：概念讲解（30分钟）
- [ ] 课程目标和产出已明确说明
- [ ] SQLModel核心概念已清晰讲解
- [ ] 数据模型设计原则已阐述
- [ ] 与上节课内容的连接已建立
- [ ] 学生理解程度已确认

### 第二阶段：实践演示（60分钟）

#### Exercise 1: SQLModel基础模型设计（15分钟）
- [ ] 用户模型结构已正确演示
- [ ] 字段类型和约束已详细说明
- [ ] Field参数的作用已解释
- [ ] 学生实践指导已提供
- [ ] 常见错误已预防和纠正

#### Exercise 2: 数据库连接配置（10分钟）
- [ ] 连接字符串格式已正确展示
- [ ] 连接池参数已详细解释
- [ ] 环境变量使用已演示
- [ ] 连接测试已成功执行
- [ ] 错误处理机制已说明

#### Exercise 3: 关系模型设计（15分钟）
- [ ] 文档模型已正确定义
- [ ] 外键约束已正确设置
- [ ] 双向关系已正确建立
- [ ] TYPE_CHECKING使用已说明
- [ ] 关系查询已演示

#### Exercise 4: 迁移脚本编写（20分钟）
- [ ] Alembic初始化已完成
- [ ] 配置文件已正确设置
- [ ] 迁移脚本已成功生成
- [ ] 迁移执行已验证成功
- [ ] 回滚操作已演示

### 第三阶段：总结答疑（30分钟）
- [ ] 学生Exercise完成情况已检查
- [ ] 主要问题已收集和解答
- [ ] Lab 03任务已详细说明
- [ ] 下节课内容已预告
- [ ] 课程总结已完成

## 学生Exercise检查清单

### Exercise 1检查要点
- [ ] `src/models/user.py`文件已创建
- [ ] `UserBase`类定义正确
- [ ] `User`表模型定义完整
- [ ] 字段类型和约束正确
- [ ] 代码语法无错误

### Exercise 2检查要点
- [ ] `src/database/connection.py`文件已创建
- [ ] 数据库连接字符串正确
- [ ] 连接池配置合理
- [ ] 会话管理函数正确
- [ ] 连接测试成功

### Exercise 3检查要点
- [ ] `src/models/document.py`文件已创建
- [ ] 文档模型定义完整
- [ ] 外键关系正确设置
- [ ] 双向关系正确建立
- [ ] 导入语句正确

### Exercise 4检查要点
- [ ] Alembic已正确初始化
- [ ] `alembic.ini`配置正确
- [ ] `migrations/env.py`已正确配置
- [ ] 初始迁移已成功生成
- [ ] 迁移已成功执行到数据库

## 技术验证检查清单

### 代码质量检查
- [ ] 所有Python文件语法正确
- [ ] 类型注解完整准确
- [ ] 字段定义符合规范
- [ ] 导入语句正确无循环
- [ ] 代码结构清晰合理

### 数据库验证检查
```sql
-- 在PostgreSQL中执行以下检查
\dt  -- 验证表已创建
\d user  -- 检查user表结构
\d document  -- 检查document表结构
SELECT * FROM alembic_version;  -- 检查迁移版本
```

### 功能测试检查
```python
# 测试模型创建
from src.models.user import User
from src.models.document import Document
from src.database.connection import get_session

# 测试数据库连接
with next(get_session()) as session:
    # 创建测试用户
    user = User(
        email="test@example.com",
        username="testuser",
        password_hash="hashed_password"
    )
    session.add(user)
    session.commit()
    
    # 创建测试文档
    document = Document(
        title="Test Document",
        content="Test content",
        user_id=user.id
    )
    session.add(document)
    session.commit()
    
    # 验证关系查询
    user_with_docs = session.get(User, user.id)
    assert len(user_with_docs.documents) == 1
```

## Lab 03验收检查清单

### 项目结构检查
- [ ] `src/models/`目录结构完整
- [ ] `src/database/`配置文件完整
- [ ] `migrations/`迁移文件完整
- [ ] `alembic.ini`配置正确
- [ ] 测试文件已创建

### 数据模型检查
- [ ] 用户模型设计合理
- [ ] 文档模型设计完整
- [ ] 向量数据模型已实现
- [ ] 知识库组织模型已设计
- [ ] 模型关系映射正确

### 迁移脚本检查
- [ ] 所有迁移脚本可正常执行
- [ ] 迁移历史记录完整
- [ ] 回滚功能正常工作
- [ ] 数据完整性约束正确

### 代码规范检查
- [ ] 代码风格符合PEP 8
- [ ] 类型注解完整
- [ ] 文档字符串完整
- [ ] 错误处理完善
- [ ] 测试覆盖充分

## 常见问题检查清单

### 环境问题
- [ ] PostgreSQL服务启动失败
  - 检查Docker容器状态
  - 验证端口占用情况
  - 检查配置文件正确性

- [ ] Python包安装失败
  - 检查Python版本兼容性
  - 验证虚拟环境激活
  - 检查网络连接状态

### 代码问题
- [ ] 模型导入错误
  - 检查文件路径正确性
  - 验证__init__.py文件存在
  - 解决循环导入问题

- [ ] 数据库连接失败
  - 验证连接字符串格式
  - 检查数据库用户权限
  - 确认防火墙设置

### 迁移问题
- [ ] Alembic初始化失败
  - 检查项目目录结构
  - 验证配置文件路径
  - 确认数据库连接正常

- [ ] 迁移执行失败
  - 检查数据库当前状态
  - 验证迁移脚本语法
  - 解决表结构冲突

## 课后反思检查清单

### 教学效果评估
- [ ] 学生对核心概念的理解程度如何？
- [ ] 实践Exercise的完成质量如何？
- [ ] 课堂互动和参与度如何？
- [ ] 时间分配是否合理？
- [ ] 教学节奏是否适宜？

### 需要改进的方面
- [ ] 哪些概念需要更详细的解释？
- [ ] 哪些实践环节需要调整难度？
- [ ] 哪些演示步骤需要优化？
- [ ] 哪些错误处理需要补充？

### 学生反馈收集
- [ ] 课程内容难度是否适中？
- [ ] 实践Exercise是否有实用价值？
- [ ] 教学材料是否清晰易懂？
- [ ] 还需要补充哪些知识点？

### 下次课程准备
- [ ] 根据反馈调整教学内容
- [ ] 准备针对性的补充材料
- [ ] 优化实践Exercise设计
- [ ] 完善错误处理指导