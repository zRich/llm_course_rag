-- 数据库初始化脚本
-- 创建RAG系统所需的数据库和用户

-- 确保数据库存在
CREATE DATABASE IF NOT EXISTS rag_db;

-- 创建用户（如果不存在）
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = 'rag_user') THEN

      CREATE ROLE rag_user LOGIN PASSWORD 'rag_password';
   END IF;
END
$do$;

-- 授予权限
GRANT ALL PRIVILEGES ON DATABASE rag_db TO rag_user;

-- 连接到rag_db数据库
\c rag_db;

-- 授予schema权限
GRANT ALL ON SCHEMA public TO rag_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO rag_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO rag_user;

-- 设置默认权限
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO rag_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO rag_user;

-- 创建扩展（如果需要）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- 输出初始化完成信息
SELECT 'Database initialization completed successfully!' as status;