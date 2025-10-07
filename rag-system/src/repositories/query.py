"""查询仓库"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from ..models.query import (
    QueryHistory,
    QueryHistoryCreate,
    QueryHistoryUpdate,
    QueryStatus,
    QueryType,
    SystemConfig,
    SystemConfigCreate,
    SystemConfigUpdate
)
from .base import BaseRepository


class QueryHistoryRepository(BaseRepository[QueryHistory, QueryHistoryCreate, QueryHistoryUpdate]):
    """查询历史仓库类"""
    
    def __init__(self):
        super().__init__(QueryHistory)
    
    # 查询历史方法
    def get_by_user(
        self, 
        session: Session, 
        *, 
        user_id: UUID, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[QueryHistory]:
        """获取用户的查询历史"""
        return self.get_multi(
            session,
            skip=skip,
            limit=limit,
            filters={"user_id": user_id},
            order_by="-created_at"
        )
    
    def get_by_session(
        self, 
        session: Session, 
        *, 
        session_id: str, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[QueryHistory]:
        """获取会话的查询历史"""
        return self.get_multi(
            session,
            skip=skip,
            limit=limit,
            filters={"session_id": session_id},
            order_by="created_at"
        )
    
    def get_by_status(
        self, 
        session: Session, 
        *, 
        status: QueryStatus, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[QueryHistory]:
        """根据状态获取查询历史"""
        return self.get_multi(
            session,
            skip=skip,
            limit=limit,
            filters={"status": status},
            order_by="-created_at"
        )
    
    def get_by_type(
        self, 
        session: Session, 
        *, 
        query_type: QueryType, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[QueryHistory]:
        """根据类型获取查询历史"""
        return self.get_multi(
            session,
            skip=skip,
            limit=limit,
            filters={"query_type": query_type},
            order_by="-created_at"
        )
    
    def search_queries(
        self, 
        session: Session, 
        *, 
        query: str, 
        user_id: Optional[UUID] = None,
        query_type: Optional[QueryType] = None,
        skip: int = 0, 
        limit: int = 100
    ) -> List[QueryHistory]:
        """搜索查询历史"""
        conditions = [
            or_(
                QueryHistory.query_text.like(f"%{query}%"),
                QueryHistory.response_text.like(f"%{query}%")
            )
        ]
        
        if user_id:
            conditions.append(QueryHistory.user_id == user_id)
        
        if query_type:
            conditions.append(QueryHistory.query_type == query_type)
        
        stmt = select(QueryHistory).where(
            and_(*conditions)
        ).offset(skip).limit(limit).order_by(desc(QueryHistory.created_at))
        
        return session.exec(stmt).all()
    
    def get_recent_queries(
        self, 
        session: Session, 
        *, 
        user_id: Optional[UUID] = None,
        hours: int = 24,
        limit: int = 50
    ) -> List[QueryHistory]:
        """获取最近的查询"""
        since_time = datetime.utcnow() - timedelta(hours=hours)
        
        conditions = [QueryHistory.created_at >= since_time]
        if user_id:
            conditions.append(QueryHistory.user_id == user_id)
        
        stmt = select(QueryHistory).where(
            and_(*conditions)
        ).limit(limit).order_by(desc(QueryHistory.created_at))
        
        return session.exec(stmt).all()
    
    def get_popular_queries(
        self, 
        session: Session, 
        *, 
        days: int = 7,
        limit: int = 20
    ) -> List[Dict]:
        """获取热门查询"""
        since_time = datetime.utcnow() - timedelta(days=days)
        
        stmt = select(
            QueryHistory.query_text,
            func.count(QueryHistory.id).label('count')
        ).where(
            and_(
                QueryHistory.created_at >= since_time,
                QueryHistory.status == QueryStatus.SUCCESS
            )
        ).group_by(
            QueryHistory.query_text
        ).order_by(
            desc('count')
        ).limit(limit)
        
        result = session.exec(stmt).all()
        return [{'query': row[0], 'count': row[1]} for row in result]
    
    def get_failed_queries(
        self, 
        session: Session, 
        *, 
        user_id: Optional[UUID] = None,
        hours: int = 24
    ) -> List[QueryHistory]:
        """获取失败的查询"""
        since_time = datetime.utcnow() - timedelta(hours=hours)
        
        conditions = [
            QueryHistory.status == QueryStatus.FAILED,
            QueryHistory.created_at >= since_time
        ]
        
        if user_id:
            conditions.append(QueryHistory.user_id == user_id)
        
        stmt = select(QueryHistory).where(
            and_(*conditions)
        ).order_by(desc(QueryHistory.created_at))
        
        return session.exec(stmt).all()
    
    def update_response(
        self, 
        session: Session, 
        *, 
        query_history: QueryHistory, 
        response_text: str,
        response_time: float,
        status: QueryStatus = QueryStatus.SUCCESS,
        error_message: Optional[str] = None
    ) -> QueryHistory:
        """更新查询响应"""
        query_history.response_text = response_text
        query_history.response_time = response_time
        query_history.status = status
        
        if error_message:
            query_history.error_message = error_message
        
        session.add(query_history)
        session.commit()
        session.refresh(query_history)
        return query_history
    
    def get_query_stats(
        self, 
        session: Session, 
        *, 
        user_id: Optional[UUID] = None,
        days: int = 30
    ) -> Dict:
        """获取查询统计信息"""
        since_time = datetime.utcnow() - timedelta(days=days)
        
        base_query = select(QueryHistory).where(QueryHistory.created_at >= since_time)
        if user_id:
            base_query = base_query.where(QueryHistory.user_id == user_id)
        
        # 总查询数
        total_count = session.exec(
            select(func.count()).select_from(base_query.subquery())
        ).one()
        
        # 按状态统计
        status_stats = {}
        for status in QueryStatus:
            count_query = base_query.where(QueryHistory.status == status)
            count = session.exec(
                select(func.count()).select_from(count_query.subquery())
            ).one()
            status_stats[status.value] = count
        
        # 按类型统计
        type_stats = {}
        for query_type in QueryType:
            count_query = base_query.where(QueryHistory.query_type == query_type)
            count = session.exec(
                select(func.count()).select_from(count_query.subquery())
            ).one()
            type_stats[query_type.value] = count
        
        # 平均响应时间
        avg_response_time = session.exec(
            select(func.avg(QueryHistory.response_time)).select_from(
                base_query.where(QueryHistory.status == QueryStatus.SUCCESS).subquery()
            )
        ).one() or 0
        
        return {
            "total": total_count,
            "by_status": status_stats,
            "by_type": type_stats,
            "avg_response_time": float(avg_response_time)
        }
    
    # 异步方法
    async def aget_by_user(
        self, 
        session: AsyncSession, 
        *, 
        user_id: UUID, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[QueryHistory]:
        """异步获取用户的查询历史"""
        return await self.aget_multi(
            session,
            skip=skip,
            limit=limit,
            filters={"user_id": user_id},
            order_by="-created_at"
        )
    
    async def aget_by_session(
        self, 
        session: AsyncSession, 
        *, 
        session_id: str, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[QueryHistory]:
        """异步获取会话的查询历史"""
        return await self.aget_multi(
            session,
            skip=skip,
            limit=limit,
            filters={"session_id": session_id},
            order_by="created_at"
        )
    
    async def asearch_queries(
        self, 
        session: AsyncSession, 
        *, 
        query: str, 
        user_id: Optional[UUID] = None,
        query_type: Optional[QueryType] = None,
        skip: int = 0, 
        limit: int = 100
    ) -> List[QueryHistory]:
        """异步搜索查询历史"""
        conditions = [
            or_(
                QueryHistory.query_text.like(f"%{query}%"),
                QueryHistory.response_text.like(f"%{query}%")
            )
        ]
        
        if user_id:
            conditions.append(QueryHistory.user_id == user_id)
        
        if query_type:
            conditions.append(QueryHistory.query_type == query_type)
        
        stmt = select(QueryHistory).where(
            and_(*conditions)
        ).offset(skip).limit(limit).order_by(desc(QueryHistory.created_at))
        
        result = await session.exec(stmt)
        return result.all()
    
    async def aupdate_response(
        self, 
        session: AsyncSession, 
        *, 
        query_history: QueryHistory, 
        response_text: str,
        response_time: float,
        status: QueryStatus = QueryStatus.SUCCESS,
        error_message: Optional[str] = None
    ) -> QueryHistory:
        """异步更新查询响应"""
        query_history.response_text = response_text
        query_history.response_time = response_time
        query_history.status = status
        
        if error_message:
            query_history.error_message = error_message
        
        session.add(query_history)
        await session.commit()
        await session.refresh(query_history)
        return query_history


class SystemConfigRepository(BaseRepository[SystemConfig, SystemConfigCreate, SystemConfigUpdate]):
    """系统配置仓库类"""
    
    def __init__(self):
        super().__init__(SystemConfig)
    
    # 配置管理方法
    def get_by_key(self, session: Session, *, key: str) -> Optional[SystemConfig]:
        """根据键获取配置"""
        query = select(SystemConfig).where(SystemConfig.key == key)
        return session.exec(query).first()
    
    def get_by_category(
        self, 
        session: Session, 
        *, 
        category: str, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[SystemConfig]:
        """根据分类获取配置"""
        return self.get_multi(
            session,
            skip=skip,
            limit=limit,
            filters={"category": category},
            order_by="key"
        )
    
    def get_public_configs(self, session: Session) -> List[SystemConfig]:
        """获取公开配置"""
        return self.get_multi(
            session,
            filters={"is_public": True},
            order_by="category,key"
        )
    
    def get_private_configs(self, session: Session) -> List[SystemConfig]:
        """获取私有配置"""
        return self.get_multi(
            session,
            filters={"is_public": False},
            order_by="category,key"
        )
    
    def search_configs(
        self, 
        session: Session, 
        *, 
        query: str, 
        category: Optional[str] = None,
        skip: int = 0, 
        limit: int = 100
    ) -> List[SystemConfig]:
        """搜索配置"""
        conditions = [
            or_(
                SystemConfig.key.like(f"%{query}%"),
                SystemConfig.description.like(f"%{query}%")
            )
        ]
        
        if category:
            conditions.append(SystemConfig.category == category)
        
        stmt = select(SystemConfig).where(
            and_(*conditions)
        ).offset(skip).limit(limit).order_by(
            SystemConfig.category, 
            SystemConfig.key
        )
        
        return session.exec(stmt).all()
    
    def set_config(
        self, 
        session: Session, 
        *, 
        key: str, 
        value: str,
        category: str = "general",
        description: Optional[str] = None,
        is_public: bool = True
    ) -> SystemConfig:
        """设置配置值"""
        config = self.get_by_key(session, key=key)
        
        if config:
            # 更新现有配置
            config.value = value
            if description is not None:
                config.description = description
            config.is_public = is_public
            config.updated_at = datetime.utcnow()
        else:
            # 创建新配置
            config_data = SystemConfigCreate(
                key=key,
                value=value,
                category=category,
                description=description,
                is_public=is_public
            )
            config = self.create(session, obj_in=config_data)
        
        return config
    
    def get_config_value(self, session: Session, *, key: str, default: Optional[str] = None) -> Optional[str]:
        """获取配置值"""
        config = self.get_by_key(session, key=key)
        return config.value if config else default
    
    def delete_config(self, session: Session, *, key: str) -> bool:
        """删除配置"""
        config = self.get_by_key(session, key=key)
        if config:
            self.remove(session, id=config.id)
            return True
        return False
    
    def get_config_categories(self, session: Session) -> List[str]:
        """获取所有配置分类"""
        stmt = select(SystemConfig.category).distinct().order_by(SystemConfig.category)
        result = session.exec(stmt)
        return [row for row in result.all()]
    
    def get_configs_dict(
        self, 
        session: Session, 
        *, 
        category: Optional[str] = None,
        public_only: bool = False
    ) -> Dict[str, str]:
        """获取配置字典"""
        conditions = []
        
        if category:
            conditions.append(SystemConfig.category == category)
        
        if public_only:
            conditions.append(SystemConfig.is_public == True)
        
        if conditions:
            stmt = select(SystemConfig).where(and_(*conditions))
        else:
            stmt = select(SystemConfig)
        
        configs = session.exec(stmt).all()
        return {config.key: config.value for config in configs}
    
    # 异步方法
    async def aget_by_key(self, session: AsyncSession, *, key: str) -> Optional[SystemConfig]:
        """异步根据键获取配置"""
        query = select(SystemConfig).where(SystemConfig.key == key)
        result = await session.exec(query)
        return result.first()
    
    async def aget_by_category(
        self, 
        session: AsyncSession, 
        *, 
        category: str, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[SystemConfig]:
        """异步根据分类获取配置"""
        return await self.aget_multi(
            session,
            skip=skip,
            limit=limit,
            filters={"category": category},
            order_by="key"
        )
    
    async def aset_config(
        self, 
        session: AsyncSession, 
        *, 
        key: str, 
        value: str,
        category: str = "general",
        description: Optional[str] = None,
        is_public: bool = True
    ) -> SystemConfig:
        """异步设置配置值"""
        config = await self.aget_by_key(session, key=key)
        
        if config:
            # 更新现有配置
            config.value = value
            if description is not None:
                config.description = description
            config.is_public = is_public
            config.updated_at = datetime.utcnow()
            
            session.add(config)
            await session.commit()
            await session.refresh(config)
        else:
            # 创建新配置
            config_data = SystemConfigCreate(
                key=key,
                value=value,
                category=category,
                description=description,
                is_public=is_public
            )
            config = await self.acreate(session, obj_in=config_data)
        
        return config
    
    async def aget_config_value(
        self, 
        session: AsyncSession, 
        *, 
        key: str, 
        default: Optional[str] = None
    ) -> Optional[str]:
        """异步获取配置值"""
        config = await self.aget_by_key(session, key=key)
        return config.value if config else default
    
    async def adelete_config(self, session: AsyncSession, *, key: str) -> bool:
        """异步删除配置"""
        config = await self.aget_by_key(session, key=key)
        if config:
            await self.aremove(session, id=config.id)
            return True
        return False


# 创建全局仓库实例
query_history_repository = QueryHistoryRepository()
system_config_repository = SystemConfigRepository()