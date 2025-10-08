# Lesson 07 环境与依赖准备说明

## 环境准备概览

本课程需要在已有的RAG系统基础上添加关键词检索功能，主要涉及PostgreSQL全文检索配置和jieba中文分词库的安装使用。

### 前置条件检查
- ✅ 已完成Lesson 06的环境配置
- ✅ PostgreSQL数据库正常运行
- ✅ Python环境和基础依赖已安装
- ✅ documents表已创建并包含测试数据

## 系统要求

### 硬件要求
- **内存**：≥ 4GB可用内存 (推荐8GB)
- **存储**：≥ 2GB可用空间
- **CPU**：支持多核处理 (jieba分词可并行)
- **网络**：能够访问PyPI和GitHub

### 软件要求
- **操作系统**：macOS 10.15+, Ubuntu 18.04+, Windows 10+
- **Python**：3.8, 3.9, 3.10 (推荐3.9)
- **PostgreSQL**：12.0+ (推荐14.0+)
- **Git**：用于获取课程材料

## 依赖库安装

### 1. 核心依赖安装

#### 安装jieba分词库
```bash
# 安装jieba
pip install jieba>=0.42.1

# 验证安装
python -c "import jieba; print(jieba.__version__)"
```

#### 更新requirements.txt
```txt
# 在现有requirements.txt基础上添加
jieba>=0.42.1
pytest>=7.0.0
psycopg2-binary>=2.9.0
numpy>=1.21.0
```

#### 批量安装
```bash
# 安装所有依赖
pip install -r requirements.txt

# 或者单独安装新增依赖
pip install jieba pytest
```

### 2. 可选依赖安装

#### 性能分析工具
```bash
# 用于性能测试和分析
pip install memory-profiler
pip install line-profiler
```

#### 开发工具
```bash
# 代码质量检查
pip install pylint black

# 类型检查
pip install mypy
```

## PostgreSQL配置

### 1. 中文全文检索配置

#### 检查当前配置
```sql
-- 连接到数据库
psql -U your_username -d your_database

-- 查看可用的文本搜索配置
SELECT cfgname FROM pg_ts_config;

-- 查看默认配置
SHOW default_text_search_config;
```

#### 配置中文全文检索
```sql
-- 方案1：使用simple配置 (推荐，兼容性好)
ALTER DATABASE your_database SET default_text_search_config = 'simple';

-- 方案2：如果有zhparser扩展 (可选，效果更好)
-- CREATE EXTENSION IF NOT EXISTS zhparser;
-- CREATE TEXT SEARCH CONFIGURATION chinese (PARSER = zhparser);
-- ALTER TEXT SEARCH CONFIGURATION chinese ADD MAPPING FOR n,v,a,i,e,l WITH simple;
```

### 2. 数据库表结构扩展

#### 添加全文检索字段
```sql
-- 为documents表添加tsvector字段
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_tsvector tsvector;

-- 创建GIN索引以提升检索性能
CREATE INDEX IF NOT EXISTS idx_documents_content_tsvector 
ON documents USING GIN(content_tsvector);

-- 更新现有数据的tsvector字段
UPDATE documents 
SET content_tsvector = to_tsvector('simple', content) 
WHERE content_tsvector IS NULL;
```

#### 创建触发器自动更新tsvector
```sql
-- 创建触发器函数
CREATE OR REPLACE FUNCTION update_content_tsvector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsvector := to_tsvector('simple', NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 创建触发器
DROP TRIGGER IF EXISTS trigger_update_content_tsvector ON documents;
CREATE TRIGGER trigger_update_content_tsvector
    BEFORE INSERT OR UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_content_tsvector();
```

### 3. 数据库连接配置

#### 更新配置文件
```python
# config/database.py
import os
from typing import Dict, Any

DATABASE_CONFIG: Dict[str, Any] = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'rag_system'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
    'options': '-c default_text_search_config=simple'  # 新增配置
}

# 连接池配置
POOL_CONFIG: Dict[str, Any] = {
    'minconn': 1,
    'maxconn': 10,
    'host': DATABASE_CONFIG['host'],
    'port': DATABASE_CONFIG['port'],
    'database': DATABASE_CONFIG['database'],
    'user': DATABASE_CONFIG['user'],
    'password': DATABASE_CONFIG['password'],
    'options': DATABASE_CONFIG['options']
}
```

## 环境变量配置

### 1. 更新.env文件
```bash
# 在现有.env文件基础上添加
# 数据库配置 (如果还没有)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_system
DB_USER=postgres
DB_PASSWORD=your_password

# 搜索配置
CHUNK_SIZE=500
MAX_QUERY_LENGTH=200
DEFAULT_SEARCH_LIMIT=10
SEARCH_TIMEOUT=30

# jieba配置
JIEBA_DICT_PATH=./data/jieba_dict/
JIEBA_USER_DICT=./data/jieba_dict/user_dict.txt
JIEBA_STOP_WORDS=./data/stop_words.txt

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=./logs/search.log
```

### 2. 创建配置加载器
```python
# config/search.py
import os
from typing import List, Optional

class SearchConfig:
    """搜索相关配置"""
    
    # 基础配置
    CHUNK_SIZE: int = int(os.getenv('CHUNK_SIZE', 500))
    MAX_QUERY_LENGTH: int = int(os.getenv('MAX_QUERY_LENGTH', 200))
    DEFAULT_SEARCH_LIMIT: int = int(os.getenv('DEFAULT_SEARCH_LIMIT', 10))
    SEARCH_TIMEOUT: int = int(os.getenv('SEARCH_TIMEOUT', 30))
    
    # jieba配置
    JIEBA_DICT_PATH: str = os.getenv('JIEBA_DICT_PATH', './data/jieba_dict/')
    JIEBA_USER_DICT: Optional[str] = os.getenv('JIEBA_USER_DICT')
    JIEBA_STOP_WORDS: str = os.getenv('JIEBA_STOP_WORDS', './data/stop_words.txt')
    
    # 停用词列表
    DEFAULT_STOP_WORDS: List[str] = [
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
        '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
        '你', '会', '着', '没有', '看', '好', '自己', '这'
    ]
```

## 目录结构准备

### 1. 创建必要目录
```bash
# 在项目根目录下创建
mkdir -p data/jieba_dict
mkdir -p data/stop_words
mkdir -p logs
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p examples/queries
mkdir -p cache/jieba
```

### 2. 准备数据文件

#### 创建停用词文件
```bash
# data/stop_words.txt
cat > data/stop_words.txt << 'EOF'
的
了
在
是
我
有
和
就
不
人
都
一
一个
上
也
很
到
说
要
去
你
会
着
没有
看
好
自己
这
EOF
```

#### 创建测试查询文件
```bash
# examples/queries/test_queries.txt
cat > examples/queries/test_queries.txt << 'EOF'
Python编程
机器学习算法
数据库设计
Web开发
人工智能
深度学习
自然语言处理
计算机视觉
软件工程
系统架构
EOF
```

### 3. 项目结构确认
```
your_project/
├── config/
│   ├── __init__.py
│   ├── database.py
│   └── search.py
├── data/
│   ├── jieba_dict/
│   ├── stop_words.txt
│   └── documents/
├── examples/
│   └── queries/
│       └── test_queries.txt
├── logs/
├── tests/
│   ├── unit/
│   └── integration/
├── cache/
│   └── jieba/
├── .env
├── requirements.txt
└── README.md
```

## 功能验证

### 1. 数据库连接测试
```python
# test_db_connection.py
import psycopg2
from config.database import DATABASE_CONFIG

def test_database_connection():
    """测试数据库连接"""
    try:
        conn = psycopg2.connect(**DATABASE_CONFIG)
        cursor = conn.cursor()
        
        # 测试基本查询
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"PostgreSQL版本: {version[0]}")
        
        # 测试全文检索配置
        cursor.execute("SHOW default_text_search_config;")
        config = cursor.fetchone()
        print(f"全文检索配置: {config[0]}")
        
        # 测试documents表
        cursor.execute("SELECT COUNT(*) FROM documents;")
        count = cursor.fetchone()
        print(f"文档数量: {count[0]}")
        
        cursor.close()
        conn.close()
        print("✅ 数据库连接测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 数据库连接测试失败: {e}")
        return False

if __name__ == "__main__":
    test_database_connection()
```

### 2. jieba分词测试
```python
# test_jieba.py
import jieba
import time
from config.search import SearchConfig

def test_jieba_installation():
    """测试jieba安装和基本功能"""
    try:
        # 测试基本分词
        text = "我爱自然语言处理和机器学习"
        
        # 精确模式
        words_precise = jieba.cut(text, cut_all=False)
        print(f"精确模式: {' / '.join(words_precise)}")
        
        # 全模式
        words_full = jieba.cut(text, cut_all=True)
        print(f"全模式: {' / '.join(words_full)}")
        
        # 搜索引擎模式
        words_search = jieba.cut_for_search(text)
        print(f"搜索模式: {' / '.join(words_search)}")
        
        # 性能测试
        start_time = time.time()
        for _ in range(1000):
            list(jieba.cut(text))
        end_time = time.time()
        
        print(f"性能测试: 1000次分词耗时 {end_time - start_time:.3f}秒")
        print("✅ jieba分词测试通过")
        return True
        
    except Exception as e:
        print(f"❌ jieba分词测试失败: {e}")
        return False

if __name__ == "__main__":
    test_jieba_installation()
```

### 3. 全文检索测试
```python
# test_fulltext_search.py
import psycopg2
from config.database import DATABASE_CONFIG

def test_fulltext_search():
    """测试PostgreSQL全文检索功能"""
    try:
        conn = psycopg2.connect(**DATABASE_CONFIG)
        cursor = conn.cursor()
        
        # 测试tsvector字段是否存在
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'documents' 
            AND column_name = 'content_tsvector';
        """)
        
        if not cursor.fetchone():
            print("❌ content_tsvector字段不存在，请先运行数据库配置脚本")
            return False
        
        # 测试全文检索查询
        test_query = "Python"
        cursor.execute("""
            SELECT id, content, ts_rank(content_tsvector, query) as rank
            FROM documents, to_tsquery('simple', %s) query
            WHERE content_tsvector @@ query
            ORDER BY rank DESC
            LIMIT 5;
        """, (test_query,))
        
        results = cursor.fetchall()
        print(f"测试查询 '{test_query}' 返回 {len(results)} 个结果")
        
        for i, (doc_id, content, rank) in enumerate(results, 1):
            print(f"{i}. 文档ID: {doc_id}, 相关性: {rank:.4f}")
            print(f"   内容预览: {content[:100]}...")
        
        cursor.close()
        conn.close()
        print("✅ 全文检索测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 全文检索测试失败: {e}")
        return False

if __name__ == "__main__":
    test_fulltext_search()
```

### 4. 综合环境测试
```python
# test_environment.py
import sys
import os
from test_db_connection import test_database_connection
from test_jieba import test_jieba_installation
from test_fulltext_search import test_fulltext_search

def test_python_version():
    """测试Python版本"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python版本不支持: {version.major}.{version.minor}.{version.micro}")
        return False

def test_required_packages():
    """测试必需包的安装"""
    required_packages = ['psycopg2', 'jieba', 'pytest', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 未安装")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def test_directory_structure():
    """测试目录结构"""
    required_dirs = [
        'config', 'data', 'logs', 'tests', 'examples', 'cache'
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ 目录存在")
        else:
            print(f"❌ {dir_name}/ 目录不存在")
            missing_dirs.append(dir_name)
    
    return len(missing_dirs) == 0

def main():
    """运行所有环境测试"""
    print("=" * 50)
    print("Lesson 07 环境检查")
    print("=" * 50)
    
    tests = [
        ("Python版本检查", test_python_version),
        ("必需包检查", test_required_packages),
        ("目录结构检查", test_directory_structure),
        ("数据库连接检查", test_database_connection),
        ("jieba分词检查", test_jieba_installation),
        ("全文检索检查", test_fulltext_search),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"请参考SETUP.md解决 {test_name} 问题")
    
    print("\n" + "=" * 50)
    print(f"环境检查结果: {passed}/{total} 项通过")
    
    if passed == total:
        print("🎉 环境准备完成，可以开始Lesson 07学习！")
    else:
        print("⚠️  请解决上述问题后再开始学习")
    
    return passed == total

if __name__ == "__main__":
    main()
```

## 常见问题解决

### 1. jieba安装问题

#### 问题：pip install jieba失败
```bash
# 解决方案1：升级pip
python -m pip install --upgrade pip
pip install jieba

# 解决方案2：使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple jieba

# 解决方案3：从源码安装
git clone https://github.com/fxsjy/jieba.git
cd jieba
python setup.py install
```

#### 问题：jieba分词速度慢
```python
# 解决方案：启用并行分词
import jieba
jieba.enable_parallel(4)  # 使用4个进程

# 或者预加载词典
jieba.initialize()
```

### 2. PostgreSQL配置问题

#### 问题：全文检索不支持中文
```sql
-- 解决方案1：使用simple配置
ALTER DATABASE your_database SET default_text_search_config = 'simple';

-- 解决方案2：安装zhparser扩展 (Ubuntu/Debian)
sudo apt-get install postgresql-14-zhparser
-- 然后在数据库中创建扩展
CREATE EXTENSION zhparser;
```

#### 问题：GIN索引创建失败
```sql
-- 检查PostgreSQL版本
SELECT version();

-- 确保版本 >= 12.0，然后重新创建索引
DROP INDEX IF EXISTS idx_documents_content_tsvector;
CREATE INDEX idx_documents_content_tsvector 
ON documents USING GIN(content_tsvector);
```

### 3. 权限问题

#### 问题：数据库连接被拒绝
```bash
# 检查PostgreSQL服务状态
sudo systemctl status postgresql

# 启动PostgreSQL服务
sudo systemctl start postgresql

# 检查pg_hba.conf配置
sudo nano /etc/postgresql/14/main/pg_hba.conf
# 确保有类似这样的行：
# local   all             all                                     trust
```

#### 问题：文件权限不足
```bash
# 修复目录权限
chmod -R 755 data/
chmod -R 755 logs/
chmod -R 755 cache/

# 修复文件权限
chmod 644 .env
chmod 644 requirements.txt
```

### 4. 内存问题

#### 问题：jieba占用内存过多
```python
# 解决方案：使用精确模式，避免全模式
import jieba
words = jieba.cut(text, cut_all=False)  # 精确模式

# 清理jieba缓存
jieba.dt.cache_file = None
```

#### 问题：PostgreSQL内存不足
```sql
-- 调整PostgreSQL内存参数 (需要重启)
-- 在postgresql.conf中设置：
shared_buffers = 256MB
work_mem = 4MB
maintenance_work_mem = 64MB
```

## 环境准备检查清单

### 安装检查
- [ ] Python 3.8+ 已安装
- [ ] PostgreSQL 12.0+ 已安装并运行
- [ ] jieba >= 0.42.1 已安装
- [ ] psycopg2-binary >= 2.9.0 已安装
- [ ] pytest >= 7.0.0 已安装

### 配置检查
- [ ] .env文件已更新
- [ ] 数据库连接配置正确
- [ ] default_text_search_config已设置
- [ ] documents表已添加content_tsvector字段
- [ ] GIN索引已创建

### 目录检查
- [ ] data/目录已创建
- [ ] logs/目录已创建
- [ ] tests/目录已创建
- [ ] examples/目录已创建
- [ ] cache/目录已创建

### 功能检查
- [ ] 数据库连接测试通过
- [ ] jieba分词测试通过
- [ ] 全文检索测试通过
- [ ] 综合环境测试通过

### 性能检查
- [ ] 单次分词耗时 < 10ms
- [ ] 单次检索耗时 < 200ms
- [ ] 内存使用 < 512MB
- [ ] 并发连接数 ≤ 10

完成以上所有检查项后，环境准备就绪，可以开始Lesson 07的学习！