#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
版本管理器 - VersionManager

实现文档版本控制和追踪功能
支持版本创建、查询、比较和回滚
提供完整的版本历史管理

作者: RAG系统开发团队
日期: 2024-01-15
"""

import json
import os
import shutil
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum


class VersionStatus(Enum):
    """版本状态枚举"""
    ACTIVE = "active"          # 活跃版本
    ARCHIVED = "archived"      # 已归档
    DELETED = "deleted"        # 已删除
    CONFLICTED = "conflicted"  # 冲突状态


@dataclass
class DocumentVersion:
    """文档版本信息"""
    doc_id: str
    version: int
    hash: str
    timestamp: str
    file_path: str
    size: int
    status: str = VersionStatus.ACTIVE.value
    parent_version: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DocumentVersion':
        """从字典创建实例"""
        return cls(**data)
    
    def __str__(self) -> str:
        return f"Version {self.version} of {self.doc_id} ({self.status})"


@dataclass
class VersionDiff:
    """版本差异信息"""
    doc_id: str
    old_version: int
    new_version: int
    change_type: str  # content, metadata, status
    changes: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VersionDiff':
        """从字典创建实例"""
        return cls(**data)


class VersionManager:
    """版本管理器
    
    功能:
    1. 创建和管理文档版本
    2. 版本查询和比较
    3. 版本回滚和恢复
    4. 版本历史追踪
    5. 版本清理和归档
    """
    
    def __init__(self, 
                 version_file: str = "versions.json",
                 version_storage_dir: str = "version_storage",
                 max_versions_per_doc: int = 10,
                 auto_cleanup: bool = True):
        """
        初始化版本管理器
        
        Args:
            version_file: 版本信息文件路径
            version_storage_dir: 版本存储目录
            max_versions_per_doc: 每个文档最大版本数
            auto_cleanup: 是否自动清理旧版本
        """
        self.version_file = version_file
        self.version_storage_dir = version_storage_dir
        self.max_versions_per_doc = max_versions_per_doc
        self.auto_cleanup = auto_cleanup
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 确保存储目录存在
        os.makedirs(self.version_storage_dir, exist_ok=True)
        
        # 加载版本信息
        self.versions: Dict[str, List[DocumentVersion]] = self._load_versions()
        
        # 统计信息
        self.stats = {
            'total_documents': 0,
            'total_versions': 0,
            'active_versions': 0,
            'archived_versions': 0,
            'storage_size': 0,
            'last_cleanup': None
        }
        self._update_stats()
    
    def create_version(self, 
                      doc_id: str, 
                      file_path: str, 
                      hash_value: str,
                      metadata: Optional[Dict[str, Any]] = None) -> DocumentVersion:
        """
        创建新版本
        
        Args:
            doc_id: 文档ID
            file_path: 文件路径
            hash_value: 文件哈希值
            metadata: 附加元数据
            
        Returns:
            创建的版本对象
        """
        # 获取文件信息
        try:
            file_size = os.path.getsize(file_path)
        except OSError:
            file_size = 0
        
        # 获取当前版本号
        current_versions = self.versions.get(doc_id, [])
        next_version = len(current_versions) + 1
        
        # 获取父版本
        parent_version = None
        if current_versions:
            # 找到最新的活跃版本
            active_versions = [v for v in current_versions if v.status == VersionStatus.ACTIVE.value]
            if active_versions:
                parent_version = max(v.version for v in active_versions)
        
        # 创建版本对象
        version = DocumentVersion(
            doc_id=doc_id,
            version=next_version,
            hash=hash_value,
            timestamp=datetime.now().isoformat(),
            file_path=file_path,
            size=file_size,
            parent_version=parent_version,
            metadata=metadata or {}
        )
        
        # 存储版本文件副本
        version_file_path = self._get_version_file_path(doc_id, next_version)
        try:
            os.makedirs(os.path.dirname(version_file_path), exist_ok=True)
            shutil.copy2(file_path, version_file_path)
            self.logger.info(f"版本文件已保存: {version_file_path}")
        except (IOError, OSError) as e:
            self.logger.error(f"保存版本文件失败 {version_file_path}: {e}")
        
        # 添加到版本列表
        if doc_id not in self.versions:
            self.versions[doc_id] = []
        self.versions[doc_id].append(version)
        
        # 自动清理旧版本
        if self.auto_cleanup:
            self._cleanup_old_versions(doc_id)
        
        # 保存版本信息
        self._save_versions()
        self._update_stats()
        
        self.logger.info(f"创建版本: {version}")
        return version
    
    def get_version(self, doc_id: str, version: Optional[int] = None) -> Optional[DocumentVersion]:
        """
        获取指定版本
        
        Args:
            doc_id: 文档ID
            version: 版本号，None表示获取最新版本
            
        Returns:
            版本对象，如果不存在返回None
        """
        if doc_id not in self.versions:
            return None
        
        doc_versions = self.versions[doc_id]
        
        if version is None:
            # 获取最新的活跃版本
            active_versions = [v for v in doc_versions if v.status == VersionStatus.ACTIVE.value]
            if not active_versions:
                return None
            return max(active_versions, key=lambda x: x.version)
        else:
            # 获取指定版本
            for v in doc_versions:
                if v.version == version:
                    return v
            return None
    
    def get_version_history(self, doc_id: str, limit: Optional[int] = None) -> List[DocumentVersion]:
        """
        获取版本历史
        
        Args:
            doc_id: 文档ID
            limit: 返回数量限制
            
        Returns:
            版本历史列表，按版本号倒序
        """
        if doc_id not in self.versions:
            return []
        
        versions = sorted(self.versions[doc_id], key=lambda x: x.version, reverse=True)
        
        if limit:
            versions = versions[:limit]
        
        return versions
    
    def compare_versions(self, doc_id: str, version1: int, version2: int) -> Optional[VersionDiff]:
        """
        比较两个版本
        
        Args:
            doc_id: 文档ID
            version1: 版本1
            version2: 版本2
            
        Returns:
            版本差异对象
        """
        v1 = self.get_version(doc_id, version1)
        v2 = self.get_version(doc_id, version2)
        
        if not v1 or not v2:
            return None
        
        changes = {}
        
        # 比较基本属性
        if v1.hash != v2.hash:
            changes['hash'] = {'old': v1.hash, 'new': v2.hash}
        
        if v1.size != v2.size:
            changes['size'] = {'old': v1.size, 'new': v2.size}
        
        if v1.file_path != v2.file_path:
            changes['file_path'] = {'old': v1.file_path, 'new': v2.file_path}
        
        if v1.status != v2.status:
            changes['status'] = {'old': v1.status, 'new': v2.status}
        
        # 比较元数据
        if v1.metadata != v2.metadata:
            changes['metadata'] = {'old': v1.metadata, 'new': v2.metadata}
        
        # 确定变更类型
        change_type = "content" if 'hash' in changes else "metadata"
        
        return VersionDiff(
            doc_id=doc_id,
            old_version=version1,
            new_version=version2,
            change_type=change_type,
            changes=changes,
            timestamp=datetime.now().isoformat()
        )
    
    def rollback_to_version(self, doc_id: str, target_version: int) -> bool:
        """
        回滚到指定版本
        
        Args:
            doc_id: 文档ID
            target_version: 目标版本号
            
        Returns:
            是否成功回滚
        """
        target = self.get_version(doc_id, target_version)
        if not target:
            self.logger.error(f"目标版本不存在: {doc_id} v{target_version}")
            return False
        
        # 获取版本文件路径
        version_file_path = self._get_version_file_path(doc_id, target_version)
        if not os.path.exists(version_file_path):
            self.logger.error(f"版本文件不存在: {version_file_path}")
            return False
        
        try:
            # 恢复文件
            shutil.copy2(version_file_path, target.file_path)
            
            # 创建新版本记录回滚操作
            rollback_metadata = {
                'rollback_from': self.get_version(doc_id).version if self.get_version(doc_id) else None,
                'rollback_to': target_version,
                'rollback_reason': 'manual_rollback'
            }
            
            self.create_version(
                doc_id=doc_id,
                file_path=target.file_path,
                hash_value=target.hash,
                metadata=rollback_metadata
            )
            
            self.logger.info(f"成功回滚 {doc_id} 到版本 {target_version}")
            return True
            
        except (IOError, OSError) as e:
            self.logger.error(f"回滚失败 {doc_id} v{target_version}: {e}")
            return False
    
    def archive_version(self, doc_id: str, version: int) -> bool:
        """
        归档版本
        
        Args:
            doc_id: 文档ID
            version: 版本号
            
        Returns:
            是否成功归档
        """
        target = self.get_version(doc_id, version)
        if not target:
            return False
        
        target.status = VersionStatus.ARCHIVED.value
        self._save_versions()
        self._update_stats()
        
        self.logger.info(f"版本已归档: {doc_id} v{version}")
        return True
    
    def delete_version(self, doc_id: str, version: int, permanent: bool = False) -> bool:
        """
        删除版本
        
        Args:
            doc_id: 文档ID
            version: 版本号
            permanent: 是否永久删除
            
        Returns:
            是否成功删除
        """
        target = self.get_version(doc_id, version)
        if not target:
            return False
        
        if permanent:
            # 永久删除
            self.versions[doc_id] = [v for v in self.versions[doc_id] if v.version != version]
            
            # 删除版本文件
            version_file_path = self._get_version_file_path(doc_id, version)
            try:
                if os.path.exists(version_file_path):
                    os.remove(version_file_path)
            except OSError as e:
                self.logger.error(f"删除版本文件失败 {version_file_path}: {e}")
            
            self.logger.info(f"版本已永久删除: {doc_id} v{version}")
        else:
            # 标记为删除
            target.status = VersionStatus.DELETED.value
            self.logger.info(f"版本已标记删除: {doc_id} v{version}")
        
        self._save_versions()
        self._update_stats()
        return True
    
    def get_document_list(self) -> List[str]:
        """
        获取所有文档ID列表
        
        Returns:
            文档ID列表
        """
        return list(self.versions.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return self.stats.copy()
    
    def cleanup_old_versions(self, days: int = 30):
        """
        清理旧版本
        
        Args:
            days: 保留天数
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()
        
        cleaned_count = 0
        
        for doc_id in list(self.versions.keys()):
            doc_versions = self.versions[doc_id]
            
            # 保留最新版本和最近的版本
            versions_to_keep = []
            versions_to_clean = []
            
            # 按版本号排序
            sorted_versions = sorted(doc_versions, key=lambda x: x.version, reverse=True)
            
            for i, version in enumerate(sorted_versions):
                # 总是保留最新版本
                if i == 0:
                    versions_to_keep.append(version)
                # 保留最近的版本
                elif version.timestamp > cutoff_str:
                    versions_to_keep.append(version)
                # 保留活跃状态的版本
                elif version.status == VersionStatus.ACTIVE.value:
                    versions_to_keep.append(version)
                else:
                    versions_to_clean.append(version)
            
            # 执行清理
            for version in versions_to_clean:
                self.delete_version(doc_id, version.version, permanent=True)
                cleaned_count += 1
        
        self.stats['last_cleanup'] = datetime.now().isoformat()
        self.logger.info(f"清理了{cleaned_count}个旧版本")
    
    def _cleanup_old_versions(self, doc_id: str):
        """
        清理指定文档的旧版本
        
        Args:
            doc_id: 文档ID
        """
        if doc_id not in self.versions:
            return
        
        doc_versions = self.versions[doc_id]
        active_versions = [v for v in doc_versions if v.status == VersionStatus.ACTIVE.value]
        
        if len(active_versions) <= self.max_versions_per_doc:
            return
        
        # 按版本号排序，保留最新的版本
        sorted_versions = sorted(active_versions, key=lambda x: x.version, reverse=True)
        versions_to_archive = sorted_versions[self.max_versions_per_doc:]
        
        for version in versions_to_archive:
            self.archive_version(doc_id, version.version)
    
    def _get_version_file_path(self, doc_id: str, version: int) -> str:
        """
        获取版本文件存储路径
        
        Args:
            doc_id: 文档ID
            version: 版本号
            
        Returns:
            版本文件路径
        """
        # 使用文档ID的哈希值作为子目录，避免单个目录文件过多
        import hashlib
        doc_hash = hashlib.md5(doc_id.encode()).hexdigest()[:8]
        subdir = os.path.join(self.version_storage_dir, doc_hash)
        
        # 文件名格式: doc_id_v{version}.ext
        original_ext = os.path.splitext(doc_id)[1] if '.' in doc_id else '.txt'
        filename = f"{doc_id.replace('/', '_')}_v{version}{original_ext}"
        
        return os.path.join(subdir, filename)
    
    def _update_stats(self):
        """
        更新统计信息
        """
        total_documents = len(self.versions)
        total_versions = sum(len(versions) for versions in self.versions.values())
        active_versions = 0
        archived_versions = 0
        
        for versions in self.versions.values():
            for version in versions:
                if version.status == VersionStatus.ACTIVE.value:
                    active_versions += 1
                elif version.status == VersionStatus.ARCHIVED.value:
                    archived_versions += 1
        
        # 计算存储大小
        storage_size = 0
        try:
            for root, dirs, files in os.walk(self.version_storage_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    storage_size += os.path.getsize(file_path)
        except OSError:
            pass
        
        self.stats.update({
            'total_documents': total_documents,
            'total_versions': total_versions,
            'active_versions': active_versions,
            'archived_versions': archived_versions,
            'storage_size': storage_size
        })
    
    def _load_versions(self) -> Dict[str, List[DocumentVersion]]:
        """
        加载版本信息文件
        
        Returns:
            版本信息字典
        """
        if not os.path.exists(self.version_file):
            return {}
        
        try:
            with open(self.version_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                versions = {}
                for doc_id, version_list in data.items():
                    versions[doc_id] = [
                        DocumentVersion.from_dict(version_data)
                        for version_data in version_list
                    ]
                
                return versions
                
        except (IOError, json.JSONDecodeError) as e:
            self.logger.error(f"加载版本信息文件失败: {e}")
            return {}
    
    def _save_versions(self):
        """
        保存版本信息文件
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.version_file) if os.path.dirname(self.version_file) else '.', exist_ok=True)
            
            data = {}
            for doc_id, version_list in self.versions.items():
                data[doc_id] = [version.to_dict() for version in version_list]
            
            with open(self.version_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except IOError as e:
            self.logger.error(f"保存版本信息文件失败: {e}")


if __name__ == "__main__":
    # 测试代码
    import tempfile
    import shutil
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建临时测试目录
    test_dir = tempfile.mkdtemp()
    print(f"测试目录: {test_dir}")
    
    try:
        # 创建测试文件
        test_file = os.path.join(test_dir, "test_document.txt")
        with open(test_file, 'w') as f:
            f.write("这是测试文档的初始内容")
        
        # 初始化版本管理器
        version_manager = VersionManager(
            version_file=os.path.join(test_dir, "versions.json"),
            version_storage_dir=os.path.join(test_dir, "versions")
        )
        
        # 创建第一个版本
        print("\n=== 创建第一个版本 ===")
        v1 = version_manager.create_version(
            doc_id="test_document.txt",
            file_path=test_file,
            hash_value="hash1",
            metadata={"author": "测试用户", "description": "初始版本"}
        )
        print(f"创建版本: {v1}")
        
        # 修改文件并创建第二个版本
        with open(test_file, 'w') as f:
            f.write("这是测试文档的修改内容")
        
        print("\n=== 创建第二个版本 ===")
        v2 = version_manager.create_version(
            doc_id="test_document.txt",
            file_path=test_file,
            hash_value="hash2",
            metadata={"author": "测试用户", "description": "修改版本"}
        )
        print(f"创建版本: {v2}")
        
        # 获取版本历史
        print("\n=== 版本历史 ===")
        history = version_manager.get_version_history("test_document.txt")
        for version in history:
            print(f"版本 {version.version}: {version.timestamp} - {version.metadata.get('description', '')}")
        
        # 比较版本
        print("\n=== 版本比较 ===")
        diff = version_manager.compare_versions("test_document.txt", 1, 2)
        if diff:
            print(f"版本差异: {diff.changes}")
        
        # 回滚测试
        print("\n=== 回滚测试 ===")
        success = version_manager.rollback_to_version("test_document.txt", 1)
        print(f"回滚结果: {success}")
        
        # 显示统计信息
        print("\n=== 统计信息 ===")
        stats = version_manager.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
    finally:
        # 清理测试目录
        shutil.rmtree(test_dir)
        print(f"\n清理测试目录: {test_dir}")