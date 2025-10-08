"""
数据库连接模板 - Exercise 2
请根据注释提示完成数据库连接配置
"""

# TODO: 导入必要的模块
# 提示：需要从sqlmodel导入create_engine, SQLModel, Session
# 提示：需要导入os模块用于环境变量
# 提示：需要从typing导入Generator

from sqlmodel import _____, _____, _____
import _____
from typing import _____

# TODO: 配置数据库连接URL
# 提示：使用os.getenv获取环境变量，提供默认值
DATABASE_URL = os.getenv(
    "_____",  # 环境变量名
    "_____"   # 默认连接字符串：postgresql://postgres:password@localhost:5432/rag_db
)

# TODO: 创建数据库引擎
# 提示：使用create_engine函数，配置连接池参数
engine = create_engine(
    _____,                    # 数据库URL
    pool_pre_ping=_____,      # 连接前检查，设置为True
    pool_recycle=_____,       # 连接回收时间，设置为300秒
    pool_size=_____,          # 连接池大小，设置为5
    max_overflow=_____,       # 最大溢出连接，设置为10
    echo=_____                # 是否显示SQL，开发环境设置为True
)

def get_session() -> Generator[Session, None, None]:
    """获取数据库会话"""
    # TODO: 创建会话并使用yield返回
    # 提示：使用with语句管理Session生命周期
    with _____(engine) as session:
        yield _____

def create_tables():
    """创建所有数据库表"""
    # TODO: 使用SQLModel.metadata.create_all创建表
    # 提示：需要传入engine参数
    SQLModel.metadata._____(_____) 

def test_connection() -> bool:
    """测试数据库连接"""
    try:
        # TODO: 测试数据库连接
        # 提示：使用get_session()获取会话，执行简单查询
        with next(_____()) as session:
            result = session.exec("_____").first()  # 执行SELECT 1查询
            print(f"数据库连接成功: {result}")
            return _____
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return _____

# TODO: 添加连接池配置类（扩展Exercise）
class DatabaseConfig:
    """数据库配置类"""
    
    def __init__(self):
        # TODO: 从环境变量读取配置
        self.host = os.getenv("DB_HOST", "_____")      # 默认localhost
        self.port = int(os.getenv("DB_PORT", "_____"))  # 默认5432
        self.user = os.getenv("DB_USER", "_____")      # 默认postgres
        self.password = os.getenv("DB_PASSWORD", "_____")  # 默认password
        self.database = os.getenv("DB_NAME", "_____")  # 默认rag_db
    
    @property
    def url(self) -> str:
        """构建数据库连接URL"""
        # TODO: 构建PostgreSQL连接字符串
        return f"postgresql://{_____}:{_____}@{_____}:{_____}/{_____}"

if __name__ == "__main__":
    # TODO: 测试脚本
    print("=== 数据库连接测试 ===")
    
    # 显示配置信息
    print(f"数据库URL: {DATABASE_URL}")
    
    # 测试连接
    if _____():  # 调用test_connection函数
        print("✅ 数据库连接成功")
    else:
        print("❌ 数据库连接失败")

# Exercise检查清单：
# □ 所有TODO项目都已完成
# □ 导入语句正确
# □ 数据库URL配置正确
# □ 引擎参数设置正确
# □ 会话管理函数正确
# □ 测试连接功能正常
# □ 扩展Exercise（DatabaseConfig类）已完成