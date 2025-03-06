from pydantic_settings import BaseSettings
from pathlib import Path
import os
from dotenv import load_dotenv
from enum import Enum
from typing import Optional

# 强制重新加载环境变量
load_dotenv(override=True)

class StorageType(str, Enum):
    LOCAL = "local"
    S3 = "s3"
    OSS = "oss"

class VectorStoreType(str, Enum):
    FAISS = "faiss"
    MILVUS = "milvus"
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    POSTGRESQL = "postgresql"

class Settings(BaseSettings):
    """应用配置，自动从环境变量加载"""
    
    # 基础配置
    PROJECT_NAME: str
    VERSION: str
    API_PREFIX: str
    
    # 文档目录配置
    DOCS_BASE_DIR: str = "db_docks"  # 默认值为 db_docks
    
    # LLM 配置
    LLM_TYPE: str
    
    # Local LLM 配置
    OLLAMA_API_URL: str
    OLLAMA_MODEL_NAME: str
    
    # API LLM 配置
    API_KEY: str
    API_URL: str
    API_MODEL_NAME: str
    
    # Embedding 配置
    EMBEDDING_TYPE: str
    EMBEDDING_API_URL: str
    EMBEDDING_API_KEY: str
    EMBEDDING_API_MODEL_NAME: str
    EMBEDDING_LOCAL_MODEL_PATH: Optional[str]
    
    # 向量数据库配置
    VECTOR_STORE: VectorStoreType
    
    # FAISS 配置
    FAISS_STORE_TYPE: StorageType
    FAISS_STORE_PATH: str
    FAISS_STORE_URL: Optional[str]
    
    # Milvus 配置
    MILVUS_HOST: str
    MILVUS_PORT: int
    MILVUS_USERNAME: str
    MILVUS_PASSWORD: str
    MILVUS_SECURE: bool
    
    # Qdrant 配置
    QDRANT_URL: str
    QDRANT_API_KEY: str
    
    # Elasticsearch 配置
    ES_HOST: str = "localhost"
    ES_PORT: int = 9200
    ES_USERNAME: Optional[str] = None
    ES_PASSWORD: Optional[str] = None
    
    # 文档处理配置
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 默认100MB
    REGEX_STORE_PATH: str = "/tmp/nova/regex"  # 正则匹配缓存目录
    BM25_STORE_PATH: str = "/tmp/nova/bm25"  # BM25缓存目录
    
    # 日志配置
    LOG_LEVEL: str
    LOG_PATH: str
    LOG_FILENAME: str
    LOG_MAX_BYTES: int
    LOG_BACKUP_COUNT: int
    LOG_FORMAT: str
    
    # 系统配置
    USER_AGENT: str = None
    
    # 文档基础路径
    DOC_BASE_PATH: str = "db_docks"
    
    @property
    def faiss_path(self) -> str:
        """获取 FAISS 存储路径"""
        if self.FAISS_STORE_TYPE == StorageType.LOCAL:
            return self.FAISS_STORE_PATH
        return self.FAISS_STORE_URL
    
    def validate_embedding_config(self) -> None:
        """验证 Embedding 配置"""
        if self.EMBEDDING_TYPE == "api":
            if not self.EMBEDDING_API_KEY:
                raise ValueError("使用 API 类型的 Embedding 时必须配置 EMBEDDING_API_KEY")
            if not self.EMBEDDING_API_URL:
                raise ValueError("使用 API 类型的 Embedding 时必须配置 EMBEDDING_API_URL")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 设置 USER_AGENT
        self.USER_AGENT = f"{self.PROJECT_NAME}/{self.VERSION}"
        # 验证配置
        self.validate_embedding_config()
        # 确保必要的目录存在
        if (self.VECTOR_STORE == VectorStoreType.FAISS and 
            self.FAISS_STORE_TYPE == StorageType.LOCAL):
            Path(self.faiss_path).mkdir(parents=True, exist_ok=True)
        Path(self.LOG_PATH).mkdir(parents=True, exist_ok=True)

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True

# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 获取环境变量中的模型路径
model_path = os.getenv("EMBEDDING_LOCAL_MODEL_PATH")

# 如果是绝对路径，直接使用；如果是文件名，则在models目录下查找
if os.path.isabs(model_path):
    EMBEDDING_LOCAL_MODEL_PATH = model_path
elif "/" not in model_path and "\\" not in model_path:  # 确保是单个文件名
    EMBEDDING_LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "app/core/llm/models", model_path)
else:
    raise ValueError("EMBEDDING_LOCAL_MODEL_PATH 只支持绝对路径或单个文件名")

settings = Settings() 