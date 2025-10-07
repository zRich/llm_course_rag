from typing import Dict, List, Any, Optional, Iterator
from datetime import datetime
import logging
from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

from data_connector import DataConnector

logger = logging.getLogger(__name__)

class DatabaseConnector(DataConnector):
    """
    数据库连接器
    支持MySQL、PostgreSQL等关系型数据库
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据库连接器
        
        Args:
            config: 数据库配置参数
                - db_type: 数据库类型 (mysql, postgresql)
                - host: 数据库主机
                - port: 数据库端口
                - database: 数据库名
                - username: 用户名
                - password: 密码
                - table: 目标表名
                - timestamp_column: 时间戳字段名（用于增量同步）
        """
        super().__init__(config)
        self.engine = None
        self.metadata = None
        
    def connect(self) -> bool:
        """
        建立数据库连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            if not self.validate_config(self.config):
                return False
                
            # 构建数据库连接字符串
            db_type = self.config['db_type']
            
            if db_type == 'sqlite':
                database = self.config['database']
                connection_string = f"sqlite:///{database}"
            elif db_type == 'mysql':
                host = self.config['host']
                port = self.config['port']
                database = self.config['database']
                username = self.config['username']
                password = self.config['password']
                connection_string = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}"
            elif db_type == 'postgresql':
                host = self.config['host']
                port = self.config['port']
                database = self.config['database']
                username = self.config['username']
                password = self.config['password']
                connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            else:
                logger.error(f"Unsupported database type: {db_type}")
                return False
            
            # 创建数据库引擎
            self.engine = create_engine(
                connection_string,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # 测试连接
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.metadata = MetaData()
            self.is_connected = True
            if db_type == 'sqlite':
                logger.info(f"Successfully connected to {db_type} database: {database}")
            else:
                logger.info(f"Successfully connected to {db_type} database: {host}:{port}/{database}")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Database connection failed: {str(e)}")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error during database connection: {str(e)}")
            self.is_connected = False
            return False
    
    def disconnect(self) -> bool:
        """
        断开数据库连接
        
        Returns:
            bool: 断开是否成功
        """
        try:
            if self.engine:
                self.engine.dispose()
                self.engine = None
            self.metadata = None
            self.is_connected = False
            logger.info("Database connection closed")
            return True
        except Exception as e:
            logger.error(f"Error closing database connection: {str(e)}")
            return False
    
    def test_connection(self) -> bool:
        """
        测试数据库连接
        
        Returns:
            bool: 连接是否正常
        """
        try:
            if not self.engine:
                return False
            
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                return result.fetchone() is not None
                
        except SQLAlchemyError as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return False
    
    def get_schema(self) -> Dict[str, Any]:
        """
        获取数据库表结构信息
        
        Returns:
            Dict[str, Any]: 表结构信息
        """
        try:
            if not self.is_connected:
                raise Exception("Database not connected")
            
            table_name = self.config['table']
            inspector = inspect(self.engine)
            
            # 获取表信息
            columns = inspector.get_columns(table_name)
            indexes = inspector.get_indexes(table_name)
            primary_keys = inspector.get_pk_constraint(table_name)
            
            schema_info = {
                "table_name": table_name,
                "columns": [
                    {
                        "name": col['name'],
                        "type": str(col['type']),
                        "nullable": col['nullable'],
                        "default": col.get('default')
                    }
                    for col in columns
                ],
                "primary_keys": primary_keys['constrained_columns'],
                "indexes": [
                    {
                        "name": idx['name'],
                        "columns": idx['column_names'],
                        "unique": idx['unique']
                    }
                    for idx in indexes
                ]
            }
            
            logger.info(f"Retrieved schema for table: {table_name}")
            return schema_info
            
        except Exception as e:
            logger.error(f"Error getting database schema: {str(e)}")
            return {}
    
    def fetch_data(self, 
                   query: Optional[str] = None,
                   limit: Optional[int] = None,
                   offset: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """
        获取数据库数据
        
        Args:
            query: SQL查询语句，如果为None则查询整个表
            limit: 限制返回记录数
            offset: 偏移量
            
        Returns:
            Iterator[Dict[str, Any]]: 数据记录迭代器
        """
        try:
            if not self.is_connected:
                raise Exception("Database not connected")
            
            # 构建查询语句
            if query is None:
                table_name = self.config['table']
                query = f"SELECT * FROM {table_name}"
            
            # 添加限制和偏移
            if limit is not None:
                query += f" LIMIT {limit}"
            if offset is not None:
                query += f" OFFSET {offset}"
            
            logger.info(f"Executing query: {query}")
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                columns = result.keys()
                
                for row in result:
                    yield dict(zip(columns, row))
                    
        except SQLAlchemyError as e:
            logger.error(f"Database query failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during data fetch: {str(e)}")
            raise
    
    def fetch_incremental_data(self, 
                              last_sync_time: datetime,
                              query: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """
        获取增量数据
        
        Args:
            last_sync_time: 上次同步时间
            query: 额外的查询条件
            
        Returns:
            Iterator[Dict[str, Any]]: 增量数据记录迭代器
        """
        try:
            if not self.is_connected:
                raise Exception("Database not connected")
            
            table_name = self.config['table']
            timestamp_column = self.config.get('timestamp_column', 'updated_at')
            
            # 构建增量查询
            base_query = f"SELECT * FROM {table_name} WHERE {timestamp_column} > '{last_sync_time}'"
            
            if query:
                base_query += f" AND ({query})"
            
            base_query += f" ORDER BY {timestamp_column}"
            
            logger.info(f"Executing incremental query: {base_query}")
            
            with self.engine.connect() as conn:
                result = conn.execute(text(base_query))
                columns = result.keys()
                
                for row in result:
                    yield dict(zip(columns, row))
                    
        except SQLAlchemyError as e:
            logger.error(f"Incremental data fetch failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during incremental fetch: {str(e)}")
            raise
    
    def get_total_count(self, query: Optional[str] = None) -> int:
        """
        获取数据总数
        
        Args:
            query: 查询条件
            
        Returns:
            int: 数据总数
        """
        try:
            if not self.is_connected:
                raise Exception("Database not connected")
            
            table_name = self.config['table']
            
            if query:
                count_query = f"SELECT COUNT(*) FROM {table_name} WHERE {query}"
            else:
                count_query = f"SELECT COUNT(*) FROM {table_name}"
            
            with self.engine.connect() as conn:
                result = conn.execute(text(count_query))
                count = result.scalar()
                
            logger.info(f"Total count: {count}")
            return count
            
        except SQLAlchemyError as e:
            logger.error(f"Count query failed: {str(e)}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error during count: {str(e)}")
            return 0
    
    def get_required_config_fields(self) -> List[str]:
        """
        获取必需的配置字段
        
        Returns:
            List[str]: 必需的配置字段列表
        """
        return [
            'db_type', 'host', 'port', 'database', 
            'username', 'password', 'table'
        ]
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """
        验证数据库连接器配置
        
        Args:
            config: 配置字典
            
        Returns:
            bool: 配置是否有效
        """
        # 检查数据库类型
        supported_db_types = ['mysql', 'postgresql', 'sqlite']
        db_type = config.get('db_type')
        if db_type not in supported_db_types:
            logger.error(f"Unsupported database type: {db_type}")
            return False
        
        # 检查基本必需字段
        basic_required = ['db_type', 'database', 'table']
        if not all(field in config for field in basic_required):
            missing = [field for field in basic_required if field not in config]
            logger.error(f"Missing required fields: {missing}")
            return False
        
        # SQLite只需要database和table字段
        if db_type == 'sqlite':
            return True
        
        # MySQL和PostgreSQL需要额外的连接字段
        network_required = ['host', 'port', 'username', 'password']
        if not all(field in config for field in network_required):
            missing = [field for field in network_required if field not in config]
            logger.error(f"Missing required network fields for {db_type}: {missing}")
            return False
        
        # 检查端口号
        try:
            port = int(config.get('port', 0))
            if port <= 0 or port > 65535:
                logger.error(f"Invalid port number: {port}")
                return False
        except (ValueError, TypeError):
            logger.error(f"Invalid port format: {config.get('port')}")
            return False
        
        return True
    
    def execute_custom_query(self, query: str) -> pd.DataFrame:
        """
        执行自定义SQL查询并返回DataFrame
        
        Args:
            query: SQL查询语句
            
        Returns:
            pd.DataFrame: 查询结果
        """
        try:
            if not self.is_connected:
                raise Exception("Database not connected")
            
            logger.info(f"Executing custom query: {query}")
            df = pd.read_sql(query, self.engine)
            logger.info(f"Query returned {len(df)} rows")
            return df
            
        except SQLAlchemyError as e:
            logger.error(f"Custom query failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during custom query: {str(e)}")
            raise