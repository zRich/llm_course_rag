"""用户仓库"""
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from werkzeug.security import check_password_hash, generate_password_hash

from ..models.user import (
    User,
    UserCreate,
    UserRole,
    UserStatus,
    UserUpdate
)
from .base import BaseRepository


class UserRepository(BaseRepository[User, UserCreate, UserUpdate]):
    """用户仓库类"""
    
    def __init__(self):
        super().__init__(User)
    
    # 用户认证相关方法
    def get_by_username(self, session: Session, *, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        query = select(User).where(User.username == username)
        return session.exec(query).first()
    
    def get_by_email(self, session: Session, *, email: str) -> Optional[User]:
        """根据邮箱获取用户"""
        query = select(User).where(User.email == email)
        return session.exec(query).first()
    
    def get_by_username_or_email(
        self, 
        session: Session, 
        *, 
        identifier: str
    ) -> Optional[User]:
        """根据用户名或邮箱获取用户"""
        query = select(User).where(
            or_(User.username == identifier, User.email == identifier)
        )
        return session.exec(query).first()
    
    def authenticate(
        self, 
        session: Session, 
        *, 
        username: str, 
        password: str
    ) -> Optional[User]:
        """用户认证"""
        user = self.get_by_username_or_email(session, identifier=username)
        if not user:
            return None
        if not self.verify_password(password, user.password_hash):
            return None
        return user
    
    def create_user(
        self, 
        session: Session, 
        *, 
        obj_in: UserCreate
    ) -> User:
        """创建用户"""
        # 检查用户名是否已存在
        if self.get_by_username(session, username=obj_in.username):
            raise ValueError(f"用户名 {obj_in.username} 已存在")
        
        # 检查邮箱是否已存在
        if self.get_by_email(session, email=obj_in.email):
            raise ValueError(f"邮箱 {obj_in.email} 已存在")
        
        # 创建用户数据
        user_data = obj_in.model_dump(exclude={"password"})
        user_data["password_hash"] = self.get_password_hash(obj_in.password)
        
        db_user = User(**user_data)
        session.add(db_user)
        session.commit()
        session.refresh(db_user)
        return db_user
    
    def update_password(
        self, 
        session: Session, 
        *, 
        user: User, 
        new_password: str
    ) -> User:
        """更新用户密码"""
        user.password_hash = self.get_password_hash(new_password)
        user.password_changed_at = datetime.utcnow()
        session.add(user)
        session.commit()
        session.refresh(user)
        return user
    
    def update_last_login(self, session: Session, *, user: User) -> User:
        """更新最后登录时间"""
        user.last_login_at = datetime.utcnow()
        session.add(user)
        session.commit()
        session.refresh(user)
        return user
    
    def activate_user(self, session: Session, *, user: User) -> User:
        """激活用户"""
        user.status = UserStatus.ACTIVE
        session.add(user)
        session.commit()
        session.refresh(user)
        return user
    
    def deactivate_user(self, session: Session, *, user: User) -> User:
        """停用用户"""
        user.status = UserStatus.INACTIVE
        session.add(user)
        session.commit()
        session.refresh(user)
        return user
    
    def get_active_users(
        self, 
        session: Session, 
        *, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[User]:
        """获取活跃用户列表"""
        return self.get_multi(
            session,
            skip=skip,
            limit=limit,
            filters={"status": UserStatus.ACTIVE},
            order_by="-created_at"
        )
    
    def get_users_by_role(
        self, 
        session: Session, 
        *, 
        role: UserRole, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[User]:
        """根据角色获取用户列表"""
        return self.get_multi(
            session,
            skip=skip,
            limit=limit,
            filters={"role": role},
            order_by="-created_at"
        )
    
    def search_users(
        self, 
        session: Session, 
        *, 
        query: str, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[User]:
        """搜索用户"""
        stmt = select(User).where(
            or_(
                User.username.like(f"%{query}%"),
                User.email.like(f"%{query}%"),
                User.full_name.like(f"%{query}%")
            )
        ).offset(skip).limit(limit)
        
        return session.exec(stmt).all()
    
    # 异步方法
    async def aget_by_username(self, session: AsyncSession, *, username: str) -> Optional[User]:
        """异步根据用户名获取用户"""
        query = select(User).where(User.username == username)
        result = await session.exec(query)
        return result.first()
    
    async def aget_by_email(self, session: AsyncSession, *, email: str) -> Optional[User]:
        """异步根据邮箱获取用户"""
        query = select(User).where(User.email == email)
        result = await session.exec(query)
        return result.first()
    
    async def aget_by_username_or_email(
        self, 
        session: AsyncSession, 
        *, 
        identifier: str
    ) -> Optional[User]:
        """异步根据用户名或邮箱获取用户"""
        query = select(User).where(
            or_(User.username == identifier, User.email == identifier)
        )
        result = await session.exec(query)
        return result.first()
    
    async def aauthenticate(
        self, 
        session: AsyncSession, 
        *, 
        username: str, 
        password: str
    ) -> Optional[User]:
        """异步用户认证"""
        user = await self.aget_by_username_or_email(session, identifier=username)
        if not user:
            return None
        if not self.verify_password(password, user.password_hash):
            return None
        return user
    
    async def acreate_user(
        self, 
        session: AsyncSession, 
        *, 
        obj_in: UserCreate
    ) -> User:
        """异步创建用户"""
        # 检查用户名是否已存在
        existing_user = await self.aget_by_username(session, username=obj_in.username)
        if existing_user:
            raise ValueError(f"用户名 {obj_in.username} 已存在")
        
        # 检查邮箱是否已存在
        existing_email = await self.aget_by_email(session, email=obj_in.email)
        if existing_email:
            raise ValueError(f"邮箱 {obj_in.email} 已存在")
        
        # 创建用户数据
        user_data = obj_in.model_dump(exclude={"password"})
        user_data["password_hash"] = self.get_password_hash(obj_in.password)
        
        db_user = User(**user_data)
        session.add(db_user)
        await session.commit()
        await session.refresh(db_user)
        return db_user
    
    async def aupdate_password(
        self, 
        session: AsyncSession, 
        *, 
        user: User, 
        new_password: str
    ) -> User:
        """异步更新用户密码"""
        user.password_hash = self.get_password_hash(new_password)
        user.password_changed_at = datetime.utcnow()
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user
    
    async def aupdate_last_login(self, session: AsyncSession, *, user: User) -> User:
        """异步更新最后登录时间"""
        user.last_login_at = datetime.utcnow()
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user
    
    async def aactivate_user(self, session: AsyncSession, *, user: User) -> User:
        """异步激活用户"""
        user.status = UserStatus.ACTIVE
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user
    
    async def adeactivate_user(self, session: AsyncSession, *, user: User) -> User:
        """异步停用用户"""
        user.status = UserStatus.INACTIVE
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user
    
    async def aget_active_users(
        self, 
        session: AsyncSession, 
        *, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[User]:
        """异步获取活跃用户列表"""
        return await self.aget_multi(
            session,
            skip=skip,
            limit=limit,
            filters={"status": UserStatus.ACTIVE},
            order_by="-created_at"
        )
    
    async def aget_users_by_role(
        self, 
        session: AsyncSession, 
        *, 
        role: UserRole, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[User]:
        """异步根据角色获取用户列表"""
        return await self.aget_multi(
            session,
            skip=skip,
            limit=limit,
            filters={"role": role},
            order_by="-created_at"
        )
    
    async def asearch_users(
        self, 
        session: AsyncSession, 
        *, 
        query: str, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[User]:
        """异步搜索用户"""
        stmt = select(User).where(
            or_(
                User.username.like(f"%{query}%"),
                User.email.like(f"%{query}%"),
                User.full_name.like(f"%{query}%")
            )
        ).offset(skip).limit(limit)
        
        result = await session.exec(stmt)
        return result.all()
    
    # 工具方法
    @staticmethod
    def get_password_hash(password: str) -> str:
        """生成密码哈希"""
        return generate_password_hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        return check_password_hash(hashed_password, plain_password)
    
    def is_active(self, user: User) -> bool:
        """检查用户是否活跃"""
        return user.status == UserStatus.ACTIVE
    
    def is_admin(self, user: User) -> bool:
        """检查用户是否为管理员"""
        return user.role == UserRole.ADMIN
    
    def can_manage_users(self, user: User) -> bool:
        """检查用户是否可以管理其他用户"""
        return user.role in [UserRole.ADMIN, UserRole.MANAGER]


# 创建全局用户仓库实例
user_repository = UserRepository()