from typing import Optional
import logging
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from app.config.settings import settings, EMBEDDING_LOCAL_MODEL_PATH

logger = logging.getLogger(__name__)

class EmbeddingFactory:
    @staticmethod
    def create_embeddings(embedding_type: str = None):
        """创建 Embedding 实例"""
        try:
            embedding_type = embedding_type or settings.EMBEDDING_TYPE
            
            if embedding_type == "api":
                logger.info("使用 DashScope Embedding API")
                if not settings.EMBEDDING_API_KEY:
                    raise ValueError("未配置 EMBEDDING_API_KEY")
                    
                return DashScopeEmbeddings(
                    dashscope_api_key=settings.EMBEDDING_API_KEY,
                    model=settings.EMBEDDING_API_MODEL_NAME,
                )
            elif embedding_type == "local":
                logger.info("使用本地 Embedding 模型")
                model_path = EMBEDDING_LOCAL_MODEL_PATH
                if not model_path:
                    raise ValueError("未配置 EMBEDDING_LOCAL_MODEL_PATH")
                    
                return HuggingFaceEmbeddings(
                    model_name=model_path,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            else:
                raise ValueError(f"不支持的 Embedding 类型: {embedding_type}")
                
        except Exception as e:
            logger.error(f"创建 Embedding 实例失败: {str(e)}", exc_info=True)
            raise 