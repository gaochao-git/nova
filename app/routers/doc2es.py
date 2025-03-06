from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException, Body
from pathlib import Path
import logging
from elasticsearch import AsyncElasticsearch
from app.config.settings import settings
from langchain_community.document_loaders import PyPDFLoader
import asyncio
from datetime import datetime
from pydantic import BaseModel
from enum import Enum

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/doc2es", tags=["doc2es"])

class ProcessType(str, Enum):
    OVERWRITE = "overwrite"  # 覆盖：删除现有索引后重新创建
    APPEND = "append"      # 追加：保留现有索引，添加新文档

class IndexRequest(BaseModel):
    db_type: Optional[str] = None
    process_type: ProcessType = ProcessType.APPEND  # 默认为追加模式

class Doc2ESService:
    def __init__(self):
        self.es_client = AsyncElasticsearch(
            f"http://{settings.ES_HOST}:{settings.ES_PORT}",
            basic_auth=(settings.ES_USERNAME, settings.ES_PASSWORD) if settings.ES_USERNAME else None
        )
        self.db_docs_dir = Path(settings.DOCS_BASE_DIR)
        self.batch_size = 100  # 批量写入ES的文档数

    async def _delete_index_if_exists(self, index_name: str):
        """如果索引存在则删除"""
        if await self.es_client.indices.exists(index=index_name):
            logger.info(f"删除现有索引: {index_name}")
            await self.es_client.indices.delete(index=index_name)

    async def _create_index_if_not_exists(self, index_name: str):
        """创建ES索引（如果不存在）"""
        if not await self.es_client.indices.exists(index=index_name):
            await self.es_client.indices.create(
                index=index_name,
                body={
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                        "analysis": {
                            "analyzer": {
                                "text_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "standard",
                                    "filter": ["lowercase", "stop", "snowball"]
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": {
                            "content": {"type": "text", "analyzer": "text_analyzer"},
                            "page_number": {"type": "integer"},
                            "file_path": {"type": "keyword"},
                            "db_type": {"type": "keyword"},
                            "title": {"type": "text", "analyzer": "text_analyzer"},
                            "created_at": {"type": "date"}
                        }
                    }
                }
            )
            logger.info(f"创建索引 {index_name}")

    async def _index_documents_batch(self, docs: List[Dict], index_name: str):
        """批量索引文档到ES"""
        if not docs:
            return

        bulk_data = []
        for doc in docs:
            bulk_data.extend([
                {"index": {"_index": index_name}},
                {**doc, "created_at": datetime.now().isoformat()}
            ])

        try:
            # 不再每次都强制刷新
            response = await self.es_client.bulk(body=bulk_data, refresh=False)
            if response.get("errors"):
                error_details = [item for item in response["items"] if item.get("index", {}).get("error")]
                logger.error(f"批量索引部分失败: {len(error_details)} 个文档索引失败")
                for error in error_details:
                    logger.error(f"索引错误: {error.get('index', {}).get('error')}")
            else:
                logger.info(f"成功索引 {len(docs)} 个文档到 {index_name}")
        except Exception as e:
            logger.error(f"批量索引失败: {str(e)}")

    async def process_pdf_file(self, pdf_path: Path, db_type: str):
        """处理单个PDF文件"""
        try:
            logger.info(f"开始处理PDF文件: {pdf_path}")
            loop = asyncio.get_event_loop()
            
            def load_pdf():
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()    # 这个位置5000页需要4分钟左右
                logger.info(f"PDF加载完成: {pdf_path}, 共 {len(pages)} 页")
                return pages

            pages = await loop.run_in_executor(None, load_pdf)
            
            if not pages:
                logger.warning(f"未能从PDF加载文档: {pdf_path}")
                return 0

            index_name = f"db_docs_{db_type.lower()}"
            await self._create_index_if_not_exists(index_name)
            
            es_docs = []
            total_pages = len(pages)
            success_count = 0
            
            logger.info(f"开始处理文档页面, 文件: {pdf_path.name}, 总页数: {total_pages}")
            
            for i, page in enumerate(pages, 1):
                try:
                    es_doc = {
                        "content": page.page_content,
                        "page_number": page.metadata.get("page", 0),
                        "file_path": str(pdf_path),
                        "db_type": db_type,
                        "title": pdf_path.stem
                    }
                    es_docs.append(es_doc)
                    success_count += 1
                    
                    # 达到批次大小时写入
                    if len(es_docs) >= self.batch_size:
                        await self._index_documents_batch(es_docs, index_name)
                        logger.info(f"文件进度: {pdf_path.name} - {i}/{total_pages} 页 ({(i/total_pages*100):.1f}%)")
                        es_docs = []
                except Exception as e:
                    logger.error(f"处理页面失败 - 文件: {pdf_path.name}, 页码: {i}, 错误: {str(e)}")
                    continue

            # 处理剩余的文档
            if es_docs:
                try:
                    await self._index_documents_batch(es_docs, index_name)
                    success_count += len(es_docs)
                except Exception as e:
                    logger.error(f"处理剩余页面失败 - 文件: {pdf_path.name}, 错误: {str(e)}")

            # 最后刷新一次索引
            await self.es_client.indices.refresh(index=index_name)
            logger.info(f"文件处理完成: {pdf_path.name} - 总页数: {total_pages}, 成功索引: {success_count} 页")
            return success_count

        except Exception as e:
            logger.error(f"处理PDF文件失败 {pdf_path}: {str(e)}")
            return 0

    async def _get_index_doc_count(self, index_name: str) -> int:
        """获取索引中的文档数量"""
        try:
            if not await self.es_client.indices.exists(index=index_name):
                return 0
            stats = await self.es_client.indices.stats(index=index_name)
            return stats["indices"][index_name]["total"]["docs"]["count"]
        except Exception as e:
            logger.error(f"获取索引 {index_name} 文档数量失败: {str(e)}")
            return 0

    async def initialize(self) -> bool:
        """初始化ES连接和索引"""
        try:
            # 检查ES连接
            if not await self.es_client.ping():
                logger.error("无法连接到Elasticsearch服务器")
                return False
                
            logger.info("成功连接到Elasticsearch服务器")
            return True
            
        except Exception as e:
            logger.error(f"初始化Elasticsearch服务失败: {str(e)}", exc_info=True)
            return False

    async def index_documents(self, db_type: Optional[str] = None, process_type: ProcessType = ProcessType.APPEND) -> Dict[str, int]:
        """索引指定数据库或所有数据库文档"""
        results = {}
        
        try:
            # 初始化ES连接
            if not await self.initialize():
                raise HTTPException(status_code=500, detail="无法连接到Elasticsearch服务器")

            # 如果指定了数据库类型，只处理该类型
            if db_type:
                logger.info(f"开始处理数据库类型: {db_type} (处理方式: {process_type})")
                db_dir = self.db_docs_dir / db_type
                if not db_dir.exists() or not db_dir.is_dir():
                    raise HTTPException(status_code=404, detail=f"数据库目录不存在: {db_type}")
                
                # 如果是覆盖模式，先删除现有索引
                index_name = f"db_docs_{db_type.lower()}"
                if process_type == ProcessType.OVERWRITE:
                    await self._delete_index_if_exists(index_name)
                
                total_docs = 0
                pdf_files = list(db_dir.rglob("*.pdf"))
                if not pdf_files:
                    raise HTTPException(status_code=404, detail=f"未找到PDF文件: {db_type}")
                
                logger.info(f"找到 {len(pdf_files)} 个PDF文件待处理")
                for i, pdf_file in enumerate(pdf_files, 1):
                    logger.info(f"处理文件 {i}/{len(pdf_files)}: {pdf_file.name}")
                    docs_count = await self.process_pdf_file(pdf_file, db_type)
                    total_docs += docs_count
                    logger.info(f"文件处理完成 ({i}/{len(pdf_files)}): {pdf_file.name}, 索引了 {docs_count} 页")
                    logger.info(f"总进度: {i}/{len(pdf_files)} 文件 ({i/len(pdf_files)*100:.1f}%)")
                
                results[db_type] = total_docs
                logger.info(f"数据库 {db_type} 处理完成, 共索引 {total_docs} 页")
                return results

            # 否则处理所有数据库目录
            logger.info(f"开始处理所有数据库文档 (处理方式: {process_type})")
            db_dirs = [d for d in self.db_docs_dir.iterdir() if d.is_dir()]
            logger.info(f"找到 {len(db_dirs)} 个数据库目录")
            
            for i, db_dir in enumerate(db_dirs, 1):
                if not db_dir.is_dir():
                    continue
                    
                db_type = db_dir.name
                logger.info(f"处理数据库 ({i}/{len(db_dirs)}): {db_type}")
                
                # 如果是覆盖模式，先删除现有索引
                index_name = f"db_docs_{db_type.lower()}"
                if process_type == ProcessType.OVERWRITE:
                    await self._delete_index_if_exists(index_name)
                
                total_docs = 0
                pdf_files = list(db_dir.rglob("*.pdf"))
                if pdf_files:
                    logger.info(f"数据库 {db_type} 找到 {len(pdf_files)} 个PDF文件")
                    for j, pdf_file in enumerate(pdf_files, 1):
                        logger.info(f"处理文件 {j}/{len(pdf_files)}: {pdf_file.name}")
                        docs_count = await self.process_pdf_file(pdf_file, db_type)
                        total_docs += docs_count
                        logger.info(f"文件处理完成 ({j}/{len(pdf_files)}): {pdf_file.name}, 索引了 {docs_count} 页")
                        logger.info(f"{db_type} 进度: {j}/{len(pdf_files)} 文件 ({j/len(pdf_files)*100:.1f}%)")
                else:
                    logger.warning(f"数据库 {db_type} 未找到PDF文件")
                
                results[db_type] = total_docs
                logger.info(f"数据库 {db_type} 处理完成, 共索引 {total_docs} 页")
                logger.info(f"总进度: {i}/{len(db_dirs)} 数据库 ({i/len(db_dirs)*100:.1f}%)")
            
            total_pages = sum(results.values())
            logger.info(f"所有数据库处理完成, 共索引 {total_pages} 页")
            return results
            
        except Exception as e:
            logger.error(f"索引文档失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def close(self):
        """关闭ES连接"""
        await self.es_client.close()

doc2es_service = Doc2ESService()

@router.post("/index")
async def index_documents(request: IndexRequest = Body(...)) -> Dict[str, int]:
    """
    索引数据库文档到ES
    :param request: 包含可选的 db_type 参数和处理类型参数
                   db_type: 不指定则索引所有数据库
                   process_type: 'overwrite' 覆盖现有索引，'append' 追加到现有索引
    :return: 每个数据库类型索引的文档数量
    """
    try:
        return await doc2es_service.index_documents(request.db_type, request.process_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/list_indices")
async def list_indices() -> List[str]:
    """列出所有db_docs相关的索引"""
    try:
        indices = await doc2es_service.es_client.indices.get("db_docs_*")
        return list(indices.keys())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/delete_index")
async def delete_index(index_name: Optional[str] = Body(None)):
    """
    删除指定的索引或所有索引
    :param index_name: 要删除的索引名称，如果不指定则删除所有db_docs_开头的索引
    """
    try:
        if index_name:
            if not index_name.startswith("db_docs_"):
                raise HTTPException(status_code=400, detail="只能删除db_docs_开头的索引")
            
            await doc2es_service.es_client.indices.delete(index_name)
            return {"message": f"索引 {index_name} 已删除"}
        else:
            # 删除所有db_docs_开头的索引
            indices = await doc2es_service.es_client.indices.get("db_docs_*")
            for index_name in indices:
                await doc2es_service.es_client.indices.delete(index_name)
            return {"message": f"已删除所有数据库索引: {list(indices.keys())}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 