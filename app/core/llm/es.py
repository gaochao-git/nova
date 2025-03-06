from typing import List, Optional
import logging
from dataclasses import dataclass
from elasticsearch import Elasticsearch
from app.config.settings import settings
from .loader import DocumentLoader

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    db_type: str
    file_path: str
    content: str
    score: float
    page: Optional[int] = None

class ElasticsearchService:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
        
    def __init__(self):
        # 初始化ES客户端
        self.es = Elasticsearch(
            f"http://{settings.ES_HOST}:{settings.ES_PORT}",
            basic_auth=(settings.ES_USERNAME, settings.ES_PASSWORD) if settings.ES_USERNAME else None
        )
        
        self.loader = DocumentLoader()
        self.batch_size = 50
        self.max_workers = 10
        self._initialized = True
        
    async def search(self, query: str, db_type: Optional[str] = None, k: int = 3) -> List[SearchResult]:
        """搜索文档"""
        try:
            if db_type:
                # 只在指定数据库中搜索
                index_name = f"db_docs_{db_type.lower()}"
                if not self.es.indices.exists(index=index_name):
                    logger.warning(f"索引不存在: {index_name}")
                    return []
                indices = [index_name]
            else:
                # 搜索所有数据库
                indices = [idx for idx in self.es.indices.get_alias().keys() 
                          if idx.startswith("db_docs_")]
            
            if not indices:
                logger.warning("没有可用的索引")
                return []
            
            all_results = []
            # 对每个索引单独搜索，确保每个数据库都返回k个结果
            for index in indices:
                # 构建搜索查询
                search_body = {
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["content^3", "file_name"],
                            "type": "best_fields",
                            "operator": "or"
                        }
                    },
                    "size": k,  # 每个数据库返回k个结果
                    "_source": ["content", "file_path", "db_type", "page_number"],
                    "highlight": {
                        "fields": {
                            "content": {}
                        }
                    }
                }
                
                # 执行搜索
                response = self.es.search(
                    index=index,
                    body=search_body
                )
                
                # 处理结果
                for hit in response["hits"]["hits"]:
                    source = hit["_source"]
                    all_results.append(SearchResult(
                        db_type=source["db_type"],
                        file_path=source["file_path"],
                        content=source["content"],
                        score=hit["_score"],
                        page=source.get("page_number", 1)
                    ))
            
            return all_results
            
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            return []
