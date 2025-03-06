from typing import List, Optional
from pathlib import Path
import logging
from app.core.llm.faiss import FaissService
from app.core.llm.es import ElasticsearchService

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, faiss_service: FaissService, es_service: ElasticsearchService):
        self.faiss_service = faiss_service
        self.es_service = es_service
    
    async def get_vector_search_context(self, query: str, db_types: List[str]) -> List[str]:
        """执行向量搜索"""
        contexts = []
        for db_type in db_types:
            context = await self.faiss_service.get_relevant_context(
                query=query,
                db_type=db_type,
                k=15
            )
            if context:
                formatted_context = f"\n-----------------{db_type} 数据库相关内容-----------------:\n{context}"
            else:
                formatted_context = f"\n-----------------{db_type} 数据库-----------------:\n未找到相关内容"
            contexts.append(formatted_context)
        return contexts

    async def get_es_search_context(self, query: str, db_types: List[str]) -> List[str]:
        """执行 ES 搜索"""
        logger.info(f"6666,{query},{db_types}")
        contexts = []
        results = await self.es_service.search(
            query=query,
            db_type=None,
            k=15
        )
        
        db_results = {}
        for r in results:
            if r.db_type not in db_results:
                db_results[r.db_type] = []
            citation = f"[来源: {r.file_path},第{r.page}页]"
            content_with_citation = f"{r.content}\n{citation}"
            db_results[r.db_type].append(content_with_citation)
        
        for db_type in db_types:
            if db_type in db_results and db_results[db_type]:
                context = "\n\n".join(db_results[db_type])
                formatted_context = f"\n-----------------{db_type} 数据库相关内容-----------------:\n{context}"
            else:
                formatted_context = f"\n-----------------{db_type} 数据库相关内容-----------------:\n未找到相关内容"
            contexts.append(formatted_context)
        return contexts

    async def get_hybrid_search_context(self, query: str, db_types: List[str]) -> List[str]:
        """执行混合搜索"""
        contexts = []
        for db_type in db_types:
            db_contexts = []
            
            vector_context = await self.faiss_service.get_relevant_context(
                query=query,
                db_type=db_type,
                k=5
            )
            if vector_context:
                db_contexts.append(f"【向量搜索结果】\n{vector_context}")
            
            es_results = await self.es_service.search(
                query=query,
                db_type=db_type,
                k=5
            )
            if es_results:
                es_context = "\n".join([f"{r.content}\n[来源: {r.file_path},第{r.page}页]" for r in es_results])
                db_contexts.append(f"【ES搜索结果】\n{es_context}")
            
            if db_contexts:
                combined_db_context = "\n\n".join(db_contexts)
                formatted_context = f"\n-----------------{db_type} 数据库相关内容-----------------:\n{combined_db_context}"
            else:
                formatted_context = f"\n-----------------{db_type} 数据库相关内容-----------------:\n未找到相关内容"
            contexts.append(formatted_context)
        return contexts

    async def get_db_types(self, specified_db_types: Optional[List[str]] = None) -> List[str]:
        """获取数据库类型列表"""
        if specified_db_types:
            return specified_db_types
        return []

    async def get_search_context(self, query: str, db_types: List[str], algorithm: str) -> str:
        """根据搜索算法获取上下文"""
        if algorithm == "vector":
            contexts = await self.get_vector_search_context(query, db_types)
        elif algorithm == "es":
            contexts = await self.get_es_search_context(query, db_types)
        else:  # hybrid
            contexts = await self.get_hybrid_search_context(query, db_types)
        
        return "\n\n".join(contexts) if contexts else "未找到相关文档内容" 