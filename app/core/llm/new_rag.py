from typing import List, Optional
import logging
from pydantic import BaseModel
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class SearchResult(BaseModel):
    """搜索结果模型"""
    content: str
    db_type: str
    file_path: str
    page: int
    score: float

class RAGService:
    def __init__(
        self,
        host: str = "82.156.146.51",
        port: str = "19530",
        collection_name: str = "db_docs",
        db_name: str = "nova",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        dim: int = 384,
        search_params: dict = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
    ):
        """初始化 RAG 服务"""
        self.collection_name = collection_name
        self.search_params = search_params
        
        logger.info("Starting RAG service initialization...")
        
        try:
            # 使用 MilvusClient
            self.client = MilvusClient(
                uri=f"http://{host}:{port}",
                db_name=db_name
            )
            logger.info(f"Successfully connected to Milvus at {host}:{port}")
            
            # 检查集合是否存在
            collections = self.client.list_collections()
            if collection_name not in collections:
                raise Exception(f"Collection {collection_name} does not exist")
            
            # 加载集合
            self.client.load_collection(collection_name)
            logger.info(f"Collection {collection_name} loaded successfully")
            
            # 初始化 embedding 模型
            logger.info(f"Loading embedding model {embedding_model_name}...")
            self.embedding_model = SentenceTransformer(embedding_model_name)
            logger.info(f"Embedding model {embedding_model_name} loaded successfully")
            
            logger.info("RAG service initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {str(e)}")
            raise

    async def get_search_context(
        self,
        vector_query: str,
        db_type: str,
        scalar_query: Optional[str] = None,
        top_k: int = 5
    ) -> List[str]:
        """获取搜索上下文"""
        try:
            # 生成查询向量
            query_vector = self.embedding_model.encode(vector_query).tolist()
            logger.info(f"Generated query vector for: {vector_query}")
            
            # 构建查询条件
            conditions = []
            
            # 添加数据库类型过滤
            if db_type:
                conditions.append(f"db_type == '{db_type}'")
            
            # 添加文本内容过滤
            if scalar_query:
                conditions.append(f'text_content like "%{scalar_query}%"')
            
            # 构建最终的过滤条件
            filter = " && ".join(conditions) if conditions else None
            logger.info(f"Search filter: {filter}")
            
            # 执行向量搜索
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                anns_field="text_embedding",
                search_params=self.search_params,
                limit=top_k,
                filter=filter,
                output_fields=["text_content", "db_type", "file_path", "page_number"]
            )
            # 格式化结果
            contexts = []
            for hits in results:
                for hit in hits:
                    # 从 entity 字段中获取数据
                    content = hit['entity'].get("text_content")
                    file_path = hit['entity'].get("file_path")
                    page = hit['entity'].get("page_number")
                    score = hit.get("distance")
                    citation = f"[来源: {file_path}, 第{page}页]"
                    formatted_content = f"{content}\n{citation}"
                    contexts.append(formatted_content)
            
            return contexts
                
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise

    def __del__(self):
        """析构函数，确保关闭连接"""
        try:
            if hasattr(self, 'client'):
                self.client.close()
                logger.info("Successfully closed Milvus client")
        except Exception as e:
            logger.error(f"Failed to close Milvus client: {str(e)}") 