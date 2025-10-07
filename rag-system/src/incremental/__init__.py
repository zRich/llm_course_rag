"""增量更新模块

提供增量索引更新、变更检测、冲突解决等功能
"""

from .indexer import IncrementalIndexer, IndexEntry, IndexStats
from .change_detector import ChangeDetector
from .conflict_resolver import ConflictResolver
from .version_manager import VersionManager
from .monitoring import get_monitoring_manager
from .config import IncrementalConfig
from .integration import IncrementalIntegration

__all__ = [
    'IncrementalIndexer',
    'IndexEntry', 
    'IndexStats',
    'ChangeDetector',
    'ConflictResolver',
    'VersionManager',
    'get_monitoring_manager',
    'IncrementalConfig',
    'IncrementalIntegration'
]