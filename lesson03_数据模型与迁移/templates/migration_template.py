"""
数据迁移脚本模板 - Exercise 4
请根据注释提示完成Alembic迁移脚本的配置和使用
"""

# TODO: 导入必要的模块
from alembic import _____
from alembic.config import _____
from sqlmodel import _____
import _____

# TODO: 定义迁移配置
def get_alembic_config() -> Config:
    """获取Alembic配置"""
    # TODO: 创建配置对象
    # 提示：使用Config类，传入alembic.ini文件路径
    config = _____(_____)  # "alembic.ini"
    
    # TODO: 设置数据库URL
    # 提示：从环境变量获取DATABASE_URL
    database_url = os.getenv("_____", "postgresql://postgres:password@localhost:5432/rag_db")
    config.set_main_option("_____", database_url)  # "sqlalchemy.url"
    
    return config

def init_alembic():
    """初始化Alembic"""
    try:
        # TODO: 执行alembic init命令
        # 提示：使用command.init函数，传入配置和目录名
        config = get_alembic_config()
        command._____(config, "_____")  # "alembic"
        print("✅ Alembic初始化成功")
    except Exception as e:
        print(f"❌ Alembic初始化失败: {e}")

def create_migration(message: str):
    """创建新的迁移文件"""
    try:
        # TODO: 执行alembic revision命令
        # 提示：使用command.revision函数，设置autogenerate=True
        config = get_alembic_config()
        command._____(
            config,
            message=_____,
            autogenerate=_____  # 设置为True
        )
        print(f"✅ 迁移文件创建成功: {message}")
    except Exception as e:
        print(f"❌ 迁移文件创建失败: {e}")

def upgrade_database(revision: str = "_____"):  # "head"
    """升级数据库到指定版本"""
    try:
        # TODO: 执行alembic upgrade命令
        # 提示：使用command.upgrade函数
        config = get_alembic_config()
        command._____(config, _____)
        print(f"✅ 数据库升级成功: {revision}")
    except Exception as e:
        print(f"❌ 数据库升级失败: {e}")

def downgrade_database(revision: str):
    """降级数据库到指定版本"""
    try:
        # TODO: 执行alembic downgrade命令
        # 提示：使用command.downgrade函数
        config = get_alembic_config()
        command._____(config, _____)
        print(f"✅ 数据库降级成功: {revision}")
    except Exception as e:
        print(f"❌ 数据库降级失败: {e}")

def show_current_revision():
    """显示当前数据库版本"""
    try:
        # TODO: 执行alembic current命令
        # 提示：使用command.current函数
        config = get_alembic_config()
        command._____(config)
    except Exception as e:
        print(f"❌ 获取当前版本失败: {e}")

def show_migration_history():
    """显示迁移历史"""
    try:
        # TODO: 执行alembic history命令
        # 提示：使用command.history函数
        config = get_alembic_config()
        command._____(config)
    except Exception as e:
        print(f"❌ 获取迁移历史失败: {e}")

# TODO: 扩展Exercise - 创建迁移管理类
class MigrationManager:
    """迁移管理器"""
    
    def __init__(self):
        # TODO: 初始化配置
        self.config = _____()
    
    def create_initial_migration(self):
        """创建初始迁移"""
        # TODO: 创建初始迁移
        # 提示：调用create_migration函数，消息为"Initial migration"
        _____("_____")
    
    def create_user_table_migration(self):
        """创建用户表迁移"""
        # TODO: 创建用户表迁移
        _____("_____")  # "Add user table"
    
    def create_document_table_migration(self):
        """创建文档表迁移"""
        # TODO: 创建文档表迁移
        _____("_____")  # "Add document table"
    
    def apply_all_migrations(self):
        """应用所有迁移"""
        # TODO: 升级到最新版本
        _____()
    
    def rollback_last_migration(self):
        """回滚最后一次迁移"""
        # TODO: 降级一个版本
        # 提示：使用"-1"作为revision参数
        _____("_____")

def main():
    """主函数 - 演示迁移操作"""
    print("=== 数据迁移演示 ===")
    
    # TODO: 创建迁移管理器实例
    manager = _____()
    
    # 显示当前状态
    print("\n1. 当前数据库版本:")
    _____()
    
    # 显示迁移历史
    print("\n2. 迁移历史:")
    _____()
    
    # 创建新迁移（示例）
    print("\n3. 创建新迁移:")
    # manager.create_user_table_migration()
    
    # 应用迁移
    print("\n4. 应用迁移:")
    # manager.apply_all_migrations()

if __name__ == "__main__":
    main()

# Exercise检查清单：
# □ 所有TODO项目都已完成
# □ 导入语句正确
# □ Alembic配置函数正确
# □ 迁移创建函数正确
# □ 数据库升级/降级函数正确
# □ 版本查询函数正确
# □ 扩展Exercise（MigrationManager类）已完成
# □ 主函数演示逻辑正确
# □ 代码可以正常运行