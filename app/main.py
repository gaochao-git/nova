from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config.logger import setup_logger
from app.routers import db_qa
# 设置日志
setup_logger()

app = FastAPI()

# 添加 CORS 中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有请求头
)

# 注册路由
app.include_router(db_qa.router)