"""Alembic环境配置"""
import asyncio
from logging.config import fileConfig
from typing import Any, Dict

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import SQLModel

# 导入所有模型以确保它们被注册到SQLModel.metadata
from src.models import *  # noqa: F403, F401
from src.database.config import db_config

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = SQLModel.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_url() -> str:
    """获取数据库连接URL"""
    # 优先使用环境变量中的数据库URL
    url = db_config.get_alembic_url()
    if url:
        return url
    
    # 如果没有环境变量，使用配置文件中的URL
    return config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """运行迁移"""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        # 包含对象名称以便更好地跟踪变更
        include_object=include_object,
        # 渲染项目以便更好地处理枚举等类型
        render_item=render_item,
    )

    with context.begin_transaction():
        context.run_migrations()


def include_object(object, name, type_, reflected, compare_to):
    """决定是否包含对象在迁移中"""
    # 排除某些表或对象
    if type_ == "table" and name in ["spatial_ref_sys"]:
        return False
    return True


def render_item(type_, obj, autogen_context):
    """渲染项目"""
    # 自定义渲染逻辑
    if type_ == "type" and hasattr(obj, "__class__"):
        # 处理枚举类型
        if "Enum" in str(obj.__class__):
            return f"sa.{obj.__class__.__name__}({', '.join(repr(v.value) for v in obj.__class__)})"
    
    # 默认渲染
    return False


async def run_async_migrations() -> None:
    """异步运行迁移"""
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    
    connectable = AsyncEngine(
        engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
            future=True,
        )
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    # 检查是否需要异步运行
    if db_config.use_async:
        asyncio.run(run_async_migrations())
    else:
        # 同步运行
        configuration = config.get_section(config.config_ini_section)
        configuration["sqlalchemy.url"] = get_url()
        
        connectable = engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

        with connectable.connect() as connection:
            do_run_migrations(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()