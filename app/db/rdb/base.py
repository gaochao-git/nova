from typing import Any, Dict, List, Optional, Type
from app.db.base import Database
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class RelationalDatabase(Database):
    """关系型数据库基类"""
    
    def __init__(self, url: str):
        self.url = url
        self.engine = None
        self.session_factory = None
        
    async def connect(self) -> None:
        """连接数据库"""
        self.engine = create_async_engine(self.url, echo=True)
        self.session_factory = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        
    async def disconnect(self) -> None:
        """断开连接"""
        if self.engine:
            await self.engine.dispose()
            
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            async with self.session_factory() as session:
                await session.execute("SELECT 1")
            return True
        except Exception as e:
            print(f"Database health check failed: {str(e)}")
            return False
            
    async def get_session(self) -> AsyncSession:
        """获取会话"""
        return self.session_factory()
    
    async def init_tables(self) -> None:
        """初始化表"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
