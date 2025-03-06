from typing import List, Optional, Dict
import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from app.config.settings import settings
from .loader import DocumentLoader
from .embedding_factory import EmbeddingFactory
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

logger = logging.getLogger(__name__)

class FaissService:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, embedding_type: str = None):
        if self._initialized:
            return
            
        try:
            logger.info("Initializing FaissService...")
            self.embeddings = EmbeddingFactory.create_embeddings(embedding_type)
            self.vector_stores: Dict[str, FAISS] = {}  # 每个数据库类型对应一个向量存储
            self.batch_size = 50  # 每批处理的文档数量，增加到50以提高效率
            self.max_workers = 10  # 最大并行工作线程数
            
            # 使用配置文件中的路径
            self.index_path = Path(settings.FAISS_STORE_PATH)
            self.index_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"向量存储路径: {self.index_path}")
            
            self.loader = DocumentLoader()
            
            # 检查并加载所有已存在的向量存储
            self._load_existing_vector_stores()
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"初始化 FaissService 失败: {str(e)}", exc_info=True)
            raise

    def _load_existing_vector_stores(self):
        """加载所有已存在的向量存储"""
        try:
            # 遍历向量存储目录
            for db_dir in self.index_path.iterdir():
                if db_dir.is_dir():  # 只处理目录
                    db_type = db_dir.name
                    try:
                        self._load_vector_store(db_type)
                    except Exception as e:
                        logger.error(f"加载{db_type}数据库的向量存储失败: {str(e)}")
        except Exception as e:
            logger.error(f"加载向量存储失败: {str(e)}")

    def _load_vector_store(self, db_type: str) -> bool:
        """加载指定数据库的向量存储"""
        try:
            # 为每个数据库创建独立的存储目录
            db_index_path = self.index_path / db_type
            logger.info(f"尝试加载向量存储: {db_type}, 路径: {db_index_path}")
            
            index_faiss = db_index_path / "index.faiss"
            index_pkl = db_index_path / "index.pkl"
            logger.info(f"检查文件是否存在: {index_faiss} ({index_faiss.exists()}), {index_pkl} ({index_pkl.exists()})")
            
            if index_faiss.exists() and index_pkl.exists():
                logger.info(f"找到向量存储文件: {index_faiss}")
                try:
                    self.vector_stores[db_type] = FAISS.load_local(
                        folder_path=str(db_index_path),
                        embeddings=self.embeddings,
                        index_name="index",
                        allow_dangerous_deserialization=True
                    )
                    logger.info(f"成功加载向量存储: {db_type}")
                    return True
                except Exception as e:
                    logger.error(f"加载向量存储文件失败: {str(e)}")
                    return False
            logger.warning(f"向量存储文件不完整或不存在: {db_index_path}")
            return False
        except Exception as e:
            logger.error(f"加载向量存储失败: {str(e)}")
            return False

    async def get_relevant_context(self, query: str, db_type: Optional[str] = None, k: int = 3) -> Optional[str]:
        """获取相关上下文"""
        try:
            if not self._initialized:
                await self.initialize()
            if db_type:
                # 只从指定数据库搜索
                if db_type in self.vector_stores:
                    docs = self.vector_stores[db_type].similarity_search(query, k=k)
                    logger.info(f"在 {db_type} 中找到 {len(docs)} 条相关内容")
                    # 修改返回格式，包含元数据
                    contexts = []
                    for doc in docs:
                        metadata_str = f"[来源: {doc.metadata.get('source', 'unknown')}, 页码: {doc.metadata.get('page', 'unknown')}]"
                        contexts.append(f"{metadata_str}\n{doc.page_content}")
                    return "\n\n".join(contexts)
                logger.warning(f"未找到 {db_type} 的向量存储")
                return None
            else:
                # 如果未指定数据库，则搜索所有数据库
                logger.info(f"当前已加载的向量存储: {list(self.vector_stores.keys())}")
                all_contexts = []
                for db_name, vector_store in self.vector_stores.items():
                    logger.info(f"从 {db_name} 数据库搜索相关内容")
                    docs = vector_store.similarity_search(query, k=k)
                    logger.info(f"在 {db_name} 中找到 {len(docs)} 条相关内容")
                    db_contexts = []
                    for doc in docs:
                        metadata_str = f"[来源: {doc.metadata.get('source', 'unknown')}, 页码: {doc.metadata.get('page', 'unknown')}]"
                        db_contexts.append(f"{metadata_str}\n{doc.page_content}")
                    context = f"\n{db_name} 相关内容:\n" + "\n\n".join(db_contexts)
                    all_contexts.append(context)
                return "\n\n".join(all_contexts) if all_contexts else None

        except Exception as e:
            logger.error(f"搜索相关上下文时出错: {str(e)}")
            return None

    def similarity_search_with_score(self, query: str, k: int = 3):
        """带相似度分数的搜索"""
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search_with_score(query, k=k) 