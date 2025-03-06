from typing import List, Dict, Optional, Any, Tuple
from fastapi import APIRouter, HTTPException
from pathlib import Path
import logging
from elasticsearch import AsyncElasticsearch
from app.config.settings import settings
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from enum import Enum
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.llm.embedding_factory import EmbeddingFactory
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/es2faiss", tags=["es2faiss"])

# 创建全局线程池
thread_pool = ThreadPoolExecutor(max_workers=4)

class ProcessType(str, Enum):
    APPEND = "append"
    OVERWRITE = "overwrite"

class Es2FaissRequest(BaseModel):
    db_type: Optional[str] = None
    process_type: ProcessType

class Es2FaissService:
    def __init__(self):
        self.es_client = None
        self.faiss_base_path = Path(settings.FAISS_STORE_PATH)
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.embedding = EmbeddingFactory.create_embeddings()

    async def ensure_es_client(self):
        """确保ES客户端可用，如果不可用则重新创建"""
        try:
            if self.es_client is None:
                self.es_client = AsyncElasticsearch(
                    f"http://{settings.ES_HOST}:{settings.ES_PORT}",
                    basic_auth=(settings.ES_USERNAME, settings.ES_PASSWORD) if settings.ES_USERNAME else None
                )
            # 测试连接是否有效
            if not await self.es_client.ping():
                if self.es_client:
                    await self.es_client.close()
                self.es_client = None
                await asyncio.sleep(2)  # 失败后等待2秒再重试
                return await self.ensure_es_client()
        except Exception as e:
            logger.error(f"ES连接失败: {str(e)}")
            if self.es_client:
                await self.es_client.close()
                self.es_client = None
            await asyncio.sleep(2)  # 失败后等待2秒
            raise HTTPException(status_code=500, detail=f"ES连接失败: {str(e)}")

    async def get_es_indices(self, db_type: Optional[str] = None) -> List[str]:
        """获取要处理的ES索引列表"""
        await self.ensure_es_client()
        try:
            if db_type:
                index_name = f"db_docs_{db_type.lower()}"
                if await self.es_client.indices.exists(index=index_name):
                    return [index_name]
                else:
                    raise HTTPException(status_code=404, detail=f"索引不存在: {index_name}")
            else:
                # 获取所有索引
                indices = await self.es_client.indices.get_alias()
                # 只返回 db_docs_ 开头的索引
                db_indices = [name for name in indices.keys() if name.startswith("db_docs_")]
                if not db_indices:
                    raise HTTPException(status_code=404, detail="未找到任何数据库文档索引")
                return db_indices
        except Exception as e:
            logger.error(f"获取ES索引失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def get_faiss_dir(self, index_name: str) -> Path:
        """获取FAISS索引目录"""
        db_type = index_name.replace("db_docs_", "")
        faiss_dir = self.faiss_base_path / db_type
        faiss_dir.mkdir(parents=True, exist_ok=True)
        return faiss_dir

    def create_chunks(self, text: str) -> List[str]:
        """将文本分割成块"""
        return self.text_splitter.split_text(text)

    async def get_total_docs(self, index_name: str) -> int:
        """获取索引中的文档总数"""
        await self.ensure_es_client()
        count_result = await self.es_client.count(index=index_name)
        return count_result["count"]

    def prepare_faiss_dir(self, faiss_dir: Path, process_type: ProcessType):
        """准备FAISS目录"""
        if process_type == ProcessType.OVERWRITE:
            for file in faiss_dir.glob("index.*"):
                file.unlink()

    def init_vector_store(self, faiss_dir: Path, process_type: ProcessType):
        """初始化向量库"""
        if process_type == ProcessType.APPEND and (faiss_dir / "index.faiss").exists():
            return FAISS.load_local(
                folder_path=str(faiss_dir),
                embeddings=self.embedding,
                index_name="index",
                allow_dangerous_deserialization=True
            )
        return None

    async def get_next_batch(self, scroll_id: str) -> Tuple[str, List[Dict]]:
        """获取下一批文档"""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                await self.ensure_es_client()
                results = await self.es_client.scroll(
                    scroll_id=scroll_id,
                    scroll="10m",
                    request_timeout=600
                )
                return results["_scroll_id"], results["hits"]["hits"]
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"获取下一批文档失败，{retry_delay}秒后重试 (尝试 {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    continue
                raise

    def process_batch(self, batch_documents: List[Dict]) -> List[Document]:
        """处理一批文档"""
        batch_docs = []
        for doc in batch_documents:
            source = doc["_source"]
            content = source.get("content", "")
            chunks = self.create_chunks(content)
            
            for i, chunk in enumerate(chunks):
                metadata = {
                    "es_doc_id": doc["_id"],
                    "file_path": source.get("file_path", ""),
                    "page_number": source.get("page_number", 0),
                    "chunk_index": i,
                    "chunk_start": i * self.chunk_size,
                    "chunk_end": (i + 1) * self.chunk_size
                }
                batch_docs.append(Document(page_content=chunk, metadata=metadata))
        return batch_docs

    async def create_vector_store(self, docs: List[Document], embedding) -> FAISS:
        """在线程池中创建向量库"""
        loop = asyncio.get_running_loop()
        batch_size = 100  # 每批处理100个文档
        total_batches = (len(docs) + batch_size - 1) // batch_size
        vector_store = None

        with tqdm(total=len(docs), desc="创建向量库进度") as pbar:
            for i in range(0, len(docs), batch_size):
                batch_docs = docs[i:i + batch_size]
                def _process_batch():
                    nonlocal vector_store
                    if vector_store is None:
                        # 第一批创建向量库
                        vector_store = FAISS.from_documents(batch_docs, embedding)
                    else:
                        # 后续批次添加到向量库
                        vector_store.add_documents(batch_docs)
                    pbar.update(len(batch_docs))
                    return vector_store
                
                vector_store = await loop.run_in_executor(thread_pool, _process_batch)
                
        return vector_store

    async def add_documents_to_store(self, vector_store: FAISS, docs: List[Document]):
        """在线程池中添加文档到向量库"""
        loop = asyncio.get_running_loop()
        batch_size = 100  # 每批处理100个文档
        
        with tqdm(total=len(docs), desc="添加文档进度") as pbar:
            for i in range(0, len(docs), batch_size):
                batch_docs = docs[i:i + batch_size]
                def _add_batch():
                    vector_store.add_documents(batch_docs)
                    pbar.update(len(batch_docs))
                    return vector_store
                
                vector_store = await loop.run_in_executor(thread_pool, _add_batch)

    async def save_vector_store(self, vector_store: FAISS, faiss_dir: Path):
        """在线程池中保存向量库"""
        loop = asyncio.get_running_loop()
        def _save():
            vector_store.save_local(str(faiss_dir), "index")
        await loop.run_in_executor(thread_pool, _save)

    async def load_all_documents(self, index_name: str) -> List[Dict]:
        """从ES加载所有文档"""
        await self.ensure_es_client()
        documents = []
        
        # 开始第一次搜索
        results = await self.es_client.search(
            index=index_name,
            body={"query": {"match_all": {}}, "sort": ["_doc"]},
            scroll="10m",
            size=100,
            request_timeout=600
        )
        
        scroll_id = results["_scroll_id"]
        batch = results["hits"]["hits"]
        documents.extend(batch)

        # 使用scroll API获取剩余文档
        while len(batch) > 0:
            results = await self.es_client.scroll(
                scroll_id=scroll_id,
                scroll="10m",
                request_timeout=600
            )
            scroll_id = results["_scroll_id"]
            batch = results["hits"]["hits"]
            documents.extend(batch)
            
        return documents

    async def process_index(self, index_name: str, process_type: ProcessType) -> Dict[str, Any]:
        """异步处理单个ES索引的入口方法"""
        try:
            # 1. 先异步加载所有文档
            documents = await self.load_all_documents(index_name)
            
            # 2. 初始化
            faiss_dir = self.get_faiss_dir(index_name)
            total_documents = 0
            total_chunks = 0
            vector_store = None
            all_docs = []

            # 3. 准备FAISS目录和向量库
            self.prepare_faiss_dir(faiss_dir, process_type)
            if process_type == ProcessType.APPEND and (faiss_dir / "index.faiss").exists():
                def _load():
                    return FAISS.load_local(
                        folder_path=str(faiss_dir),
                        embeddings=self.embedding,
                        index_name="index",
                        allow_dangerous_deserialization=True
                    )
                vector_store = await asyncio.get_running_loop().run_in_executor(thread_pool, _load)
                logger.info(f"{index_name} 已加载现有向量库")

            # 4. 处理所有文档
            for doc in documents:
                source = doc["_source"]
                content = source.get("content", "")
                chunks = self.create_chunks(content)
                
                for i, chunk in enumerate(chunks):
                    metadata = {
                        "es_doc_id": doc["_id"],
                        "file_path": source.get("file_path", ""),
                        "page_number": source.get("page_number", 0),
                        "chunk_index": i,
                        "chunk_start": i * self.chunk_size,
                        "chunk_end": (i + 1) * self.chunk_size
                    }
                    all_docs.append(Document(page_content=chunk, metadata=metadata))
                
                total_documents += 1
                total_chunks += len(chunks)
                if total_documents % 100 == 0:
                    logger.info(f"{index_name}已处理 {total_documents} 个文档，共 {total_chunks} 个文本块")

            # 5. 创建或更新向量库
            if all_docs:
                if vector_store is None:
                    logger.info(f"{index_name} 开始创建向量库...")
                    vector_store = await self.create_vector_store(all_docs, self.embedding)
                    logger.info(f"{index_name} 向量库创建完成")
                else:
                    logger.info(f"{index_name} 开始更新向量库...")
                    vector_store = await self.add_documents_to_store(vector_store, all_docs)
                    logger.info(f"{index_name} 向量库更新完成")

            # 6. 保存向量库
            if vector_store:
                logger.info(f"{index_name} 开始保存向量库...")
                await self.save_vector_store(vector_store, faiss_dir)
                logger.info(f"{index_name} 向量库保存完成")

            return {
                "index_name": index_name,
                "status": "success",
                "documents_processed": total_documents,
                "chunks_created": total_chunks
            }

        except Exception as e:
            logger.error(f"处理索引 {index_name} 失败: {str(e)}")
            return {
                "index_name": index_name,
                "status": "error",
                "error": str(e)
            }

    async def close(self):
        """关闭ES连接"""
        if self.es_client:
            await self.es_client.close()
            self.es_client = None

async def get_indices(db_type: Optional[str] = None) -> List[str]:
    """获取要处理的索引列表"""
    service = Es2FaissService()
    try:
        return await service.get_es_indices(db_type)
    finally:
        await service.close()

async def process_single_index(index_name: str, process_type: ProcessType) -> Dict[str, Any]:
    """处理单个索引的独立任务"""
    service = Es2FaissService()
    try:
        return await service.process_index(index_name, process_type)
    finally:
        await service.close()

@router.post("")
async def es_to_faiss(request: Es2FaissRequest) -> Dict[str, Any]:
    """
    将ES文档转换为FAISS向量索引
    :param request: {
        "db_type": "mysql",  # 可选，不指定则处理所有索引
        "process_type": "append"  # append或overwrite
    }
    """
    try:
        # 获取要处理的索引列表
        indices = await get_indices(request.db_type)
        
        # 为每个索引创建独立的任务，每个任务使用自己的service实例
        tasks = [
            process_single_index(index_name, request.process_type)
            for index_name in indices
        ]
        results = await asyncio.gather(*tasks)
        
        return {
            "status": "success",
            "results": results
        }
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 