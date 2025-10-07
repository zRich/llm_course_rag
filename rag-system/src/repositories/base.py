"""基础仓库类"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from uuid import UUID

from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlmodel import SQLModel

from ..models.base import BaseModel

# 类型变量
ModelType = TypeVar("ModelType", bound=BaseModel)
CreateSchemaType = TypeVar("CreateSchemaType", bound=SQLModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=SQLModel)


class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType], ABC):
    """基础仓库类
    
    提供通用的CRUD操作和查询方法
    """
    
    def __init__(self, model: Type[ModelType]):
        """
        Args:
            model: SQLModel模型类
        """
        self.model = model
    
    # 同步方法
    def create(self, session: Session, *, obj_in: CreateSchemaType) -> ModelType:
        """创建对象"""
        obj_data = obj_in.model_dump(exclude_unset=True)
        db_obj = self.model(**obj_data)
        session.add(db_obj)
        session.commit()
        session.refresh(db_obj)
        return db_obj
    
    def get(self, session: Session, id: Union[UUID, str, int]) -> Optional[ModelType]:
        """根据ID获取对象"""
        return session.get(self.model, id)
    
    def get_multi(
        self,
        session: Session,
        *,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None
    ) -> List[ModelType]:
        """获取多个对象"""
        query = select(self.model)
        
        # 应用过滤条件
        if filters:
            conditions = []
            for key, value in filters.items():
                if hasattr(self.model, key):
                    attr = getattr(self.model, key)
                    if isinstance(value, list):
                        conditions.append(attr.in_(value))
                    elif isinstance(value, dict):
                        # 支持范围查询
                        if "gte" in value:
                            conditions.append(attr >= value["gte"])
                        if "lte" in value:
                            conditions.append(attr <= value["lte"])
                        if "gt" in value:
                            conditions.append(attr > value["gt"])
                        if "lt" in value:
                            conditions.append(attr < value["lt"])
                        if "like" in value:
                            conditions.append(attr.like(f"%{value['like']}%"))
                    else:
                        conditions.append(attr == value)
            
            if conditions:
                query = query.where(and_(*conditions))
        
        # 排序
        if order_by:
            if order_by.startswith("-"):
                # 降序
                attr_name = order_by[1:]
                if hasattr(self.model, attr_name):
                    query = query.order_by(getattr(self.model, attr_name).desc())
            else:
                # 升序
                if hasattr(self.model, order_by):
                    query = query.order_by(getattr(self.model, order_by))
        
        # 分页
        query = query.offset(skip).limit(limit)
        
        return session.exec(query).all()
    
    def update(
        self,
        session: Session,
        *,
        db_obj: ModelType,
        obj_in: Union[UpdateSchemaType, Dict[str, Any]]
    ) -> ModelType:
        """更新对象"""
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.model_dump(exclude_unset=True)
        
        for field, value in update_data.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        
        session.add(db_obj)
        session.commit()
        session.refresh(db_obj)
        return db_obj
    
    def delete(self, session: Session, *, id: Union[UUID, str, int]) -> Optional[ModelType]:
        """删除对象"""
        obj = session.get(self.model, id)
        if obj:
            session.delete(obj)
            session.commit()
        return obj
    
    def count(
        self,
        session: Session,
        *,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """统计对象数量"""
        query = select(func.count()).select_from(self.model)
        
        if filters:
            conditions = []
            for key, value in filters.items():
                if hasattr(self.model, key):
                    attr = getattr(self.model, key)
                    if isinstance(value, list):
                        conditions.append(attr.in_(value))
                    else:
                        conditions.append(attr == value)
            
            if conditions:
                query = query.where(and_(*conditions))
        
        return session.exec(query).one()
    
    def exists(
        self,
        session: Session,
        *,
        filters: Dict[str, Any]
    ) -> bool:
        """检查对象是否存在"""
        conditions = []
        for key, value in filters.items():
            if hasattr(self.model, key):
                attr = getattr(self.model, key)
                conditions.append(attr == value)
        
        if not conditions:
            return False
        
        query = select(self.model).where(and_(*conditions))
        result = session.exec(query).first()
        return result is not None
    
    # 异步方法
    async def acreate(self, session: AsyncSession, *, obj_in: CreateSchemaType) -> ModelType:
        """异步创建对象"""
        obj_data = obj_in.model_dump(exclude_unset=True)
        db_obj = self.model(**obj_data)
        session.add(db_obj)
        await session.commit()
        await session.refresh(db_obj)
        return db_obj
    
    async def aget(self, session: AsyncSession, id: Union[UUID, str, int]) -> Optional[ModelType]:
        """异步根据ID获取对象"""
        return await session.get(self.model, id)
    
    async def aget_multi(
        self,
        session: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None
    ) -> List[ModelType]:
        """异步获取多个对象"""
        query = select(self.model)
        
        # 应用过滤条件
        if filters:
            conditions = []
            for key, value in filters.items():
                if hasattr(self.model, key):
                    attr = getattr(self.model, key)
                    if isinstance(value, list):
                        conditions.append(attr.in_(value))
                    elif isinstance(value, dict):
                        # 支持范围查询
                        if "gte" in value:
                            conditions.append(attr >= value["gte"])
                        if "lte" in value:
                            conditions.append(attr <= value["lte"])
                        if "gt" in value:
                            conditions.append(attr > value["gt"])
                        if "lt" in value:
                            conditions.append(attr < value["lt"])
                        if "like" in value:
                            conditions.append(attr.like(f"%{value['like']}%"))
                    else:
                        conditions.append(attr == value)
            
            if conditions:
                query = query.where(and_(*conditions))
        
        # 排序
        if order_by:
            if order_by.startswith("-"):
                # 降序
                attr_name = order_by[1:]
                if hasattr(self.model, attr_name):
                    query = query.order_by(getattr(self.model, attr_name).desc())
            else:
                # 升序
                if hasattr(self.model, order_by):
                    query = query.order_by(getattr(self.model, order_by))
        
        # 分页
        query = query.offset(skip).limit(limit)
        
        result = await session.exec(query)
        return result.all()
    
    async def aupdate(
        self,
        session: AsyncSession,
        *,
        db_obj: ModelType,
        obj_in: Union[UpdateSchemaType, Dict[str, Any]]
    ) -> ModelType:
        """异步更新对象"""
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.model_dump(exclude_unset=True)
        
        for field, value in update_data.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        
        session.add(db_obj)
        await session.commit()
        await session.refresh(db_obj)
        return db_obj
    
    async def adelete(self, session: AsyncSession, *, id: Union[UUID, str, int]) -> Optional[ModelType]:
        """异步删除对象"""
        obj = await session.get(self.model, id)
        if obj:
            await session.delete(obj)
            await session.commit()
        return obj
    
    async def acount(
        self,
        session: AsyncSession,
        *,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """异步统计对象数量"""
        query = select(func.count()).select_from(self.model)
        
        if filters:
            conditions = []
            for key, value in filters.items():
                if hasattr(self.model, key):
                    attr = getattr(self.model, key)
                    if isinstance(value, list):
                        conditions.append(attr.in_(value))
                    else:
                        conditions.append(attr == value)
            
            if conditions:
                query = query.where(and_(*conditions))
        
        result = await session.exec(query)
        return result.one()
    
    async def aexists(
        self,
        session: AsyncSession,
        *,
        filters: Dict[str, Any]
    ) -> bool:
        """异步检查对象是否存在"""
        conditions = []
        for key, value in filters.items():
            if hasattr(self.model, key):
                attr = getattr(self.model, key)
                conditions.append(attr == value)
        
        if not conditions:
            return False
        
        query = select(self.model).where(and_(*conditions))
        result = await session.exec(query)
        obj = result.first()
        return obj is not None
    
    # 批量操作
    async def abulk_create(
        self,
        session: AsyncSession,
        *,
        objs_in: List[CreateSchemaType]
    ) -> List[ModelType]:
        """异步批量创建对象"""
        db_objs = []
        for obj_in in objs_in:
            obj_data = obj_in.model_dump(exclude_unset=True)
            db_obj = self.model(**obj_data)
            db_objs.append(db_obj)
            session.add(db_obj)
        
        await session.commit()
        
        # 刷新所有对象
        for db_obj in db_objs:
            await session.refresh(db_obj)
        
        return db_objs
    
    async def abulk_update(
        self,
        session: AsyncSession,
        *,
        filters: Dict[str, Any],
        update_data: Dict[str, Any]
    ) -> int:
        """异步批量更新对象"""
        conditions = []
        for key, value in filters.items():
            if hasattr(self.model, key):
                attr = getattr(self.model, key)
                conditions.append(attr == value)
        
        if not conditions:
            return 0
        
        stmt = update(self.model).where(and_(*conditions)).values(**update_data)
        result = await session.exec(stmt)
        await session.commit()
        return result.rowcount
    
    async def abulk_delete(
        self,
        session: AsyncSession,
        *,
        filters: Dict[str, Any]
    ) -> int:
        """异步批量删除对象"""
        conditions = []
        for key, value in filters.items():
            if hasattr(self.model, key):
                attr = getattr(self.model, key)
                conditions.append(attr == value)
        
        if not conditions:
            return 0
        
        stmt = delete(self.model).where(and_(*conditions))
        result = await session.exec(stmt)
        await session.commit()
        return result.rowcount