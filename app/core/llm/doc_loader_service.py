from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
import logging
import sys
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config.settings import settings
from app.core.llm.loader import DocumentLoader

logger = logging.getLogger(__name__)

class DocLoaderService:
    _instance = None
    _initialized = False
    _docs_by_index: Dict[str, List[Document]] = {}
    _metadata_by_index: Dict[str, List[Dict]] = {}
    _loaded_files: Set[str] = set()  # 记录已加载的文件路径
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
            
        try:
            logger.info("初始化 DocLoaderService...")
            self.loader = DocumentLoader()
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            self._initialized = True
            
        except Exception as e:
            logger.error(f"初始化 DocLoaderService 失败: {str(e)}", exc_info=True)
            raise

    async def load_documents(self, base_path: Path, force: bool = False) -> List[str]:
        """
        加载文档并按索引目录组织
        :param base_path: 文档根目录
        :param force: 是否强制重新加载
        :return: 已加载的文件列表
        """
        try:
            if not base_path.exists():
                logger.warning(f"目录不存在: {base_path}")
                return []
            
            # 获取索引名（目录名）
            index_name = base_path.name
            logger.info(f"处理索引目录: {index_name}")
            
            # 如果不是强制加载且已有文档，则跳过
            if not force and self._loaded_files and any(str(base_path) in f for f in self._loaded_files):
                logger.info("已有加载的文档且非强制加载，跳过")
                return list(self._loaded_files)
            
            # 强制加载时清空该索引的缓存
            if force and index_name in self._docs_by_index:
                # 清除该索引相关的文件记录
                self._loaded_files = {f for f in self._loaded_files if str(base_path) not in f}
                # 清除该索引的文档
                self._docs_by_index.pop(index_name, None)
                self._metadata_by_index.pop(index_name, None)
            
            loaded_files = []
            
            # 初始化索引的文档列表
            if index_name not in self._docs_by_index:
                self._docs_by_index[index_name] = []
                self._metadata_by_index[index_name] = []
            
            # 遍历目录下的所有文件
            for file_path in base_path.rglob("*"):
                if not file_path.is_file():
                    continue
                    
                try:
                    # 如果不是强制加载且文件已处理过，则跳过
                    if not force and str(file_path) in self._loaded_files:
                        continue
                    
                    # 根据文件类型选择加载方法
                    file_type = file_path.suffix.lower()
                    if file_type == '.pdf':
                        docs = self.loader.load_and_split(str(file_path), "pdf")
                    elif file_type in ['.docx', '.doc']:
                        docs = self.loader.load_and_split(str(file_path), "docx")
                    elif file_type in ['.txt', '.md', '.py', '.java', '.cpp', '.h', '.c', '.json', '.xml', '.html', '.css', '.js']:
                        docs = self.loader.load_and_split(str(file_path), "text")
                    else:
                        logger.warning(f"不支持的文件类型: {file_path}")
                        continue
                        
                    if not docs:
                        continue
                        
                    # 处理每个文档片段
                    for doc in docs:
                        # 更新元数据
                        metadata = {
                            "title": file_path.stem,
                            "file_path": str(file_path),
                            "file_name": file_path.name,
                            "file_type": file_path.suffix,
                            "index_name": index_name
                        }
                        doc.metadata.update(metadata)
                        
                        # 添加到对应索引的列表中
                        self._docs_by_index[index_name].append(doc)
                        self._metadata_by_index[index_name].append(metadata)
                        
                    self._loaded_files.add(str(file_path))
                    loaded_files.append(str(file_path))
                    logger.info(f"成功加载文件: {file_path}, 生成 {len(docs)} 个片段")
                    
                except Exception as e:
                    logger.error(f"处理文件失败 {file_path}: {str(e)}")
                    continue
            
            # 输出索引的统计信息
            doc_count = len(self._docs_by_index.get(index_name, []))
            logger.info(f"索引 {index_name} 完成: {doc_count} 个文档片段, {len(loaded_files)} 个文件")
            
            return loaded_files
            
        except Exception as e:
            logger.error(f"加载文档失败: {str(e)}")
            return []

    def get_docs(self, index_names: Optional[List[str]] = None) -> Tuple[List[Document], List[Dict]]:
        """
        获取指定索引的文档和元数据
        :param index_names: 索引名称列表，为None时返回所有文档
        :return: (文档列表, 元数据列表)
        """
        if not self._docs_by_index:
            logger.warning("没有加载任何文档")
            return [], []
            
        if not index_names:
            # 返回所有文档
            all_docs = []
            all_metadata = []
            for index_name in self._docs_by_index:
                all_docs.extend(self._docs_by_index[index_name])
                all_metadata.extend(self._metadata_by_index[index_name])
            return all_docs, all_metadata
            
        # 返回指定索引的文档
        filtered_docs = []
        filtered_metadata = []
        for index_name in index_names:
            if index_name in self._docs_by_index:
                filtered_docs.extend(self._docs_by_index[index_name])
                filtered_metadata.extend(self._metadata_by_index[index_name])
            else:
                logger.warning(f"索引不存在: {index_name}")
                
        return filtered_docs, filtered_metadata

    def get_docs_by_index(self, index_name: str) -> Tuple[List[Document], List[Dict]]:
        """
        获取指定索引的文档和元数据
        :param index_name: 索引名称
        :return: (文档列表, 元数据列表)
        """
        if index_name not in self._docs_by_index:
            logger.warning(f"索引不存在: {index_name}")
            return [], []
        return self._docs_by_index[index_name], self._metadata_by_index[index_name]

    def get_available_indices(self) -> List[str]:
        """获取所有可用的索引名称"""
        return list(self._docs_by_index.keys())

    def get_memory_stats(self) -> Dict:
        """获取内存使用统计"""
        stats = {
            "total_docs": sum(len(docs) for docs in self._docs_by_index.values()),
            "docs_by_index": {index: len(docs) for index, docs in self._docs_by_index.items()},
            "total_files": len(self._loaded_files),
            "memory_usage_bytes": (
                sys.getsizeof(self._docs_by_index) + 
                sys.getsizeof(self._metadata_by_index) +
                sum(sys.getsizeof(doc) for docs in self._docs_by_index.values() for doc in docs) +
                sum(sys.getsizeof(meta) for metas in self._metadata_by_index.values() for meta in metas)
            )
        }
        stats["memory_usage_mb"] = stats["memory_usage_bytes"] / (1024 * 1024)
        return stats

    def clear_cache(self, index_names: Optional[List[str]] = None):
        """
        清空文档缓存
        :param index_names: 要清除的索引名称列表，为None时清除所有缓存
        """
        if index_names:
            for index_name in index_names:
                if index_name in self._docs_by_index:
                    # 记录要删除的文件路径
                    files_to_remove = {meta["file_path"] for meta in self._metadata_by_index[index_name]}
                    self._loaded_files -= files_to_remove
                    # 删除索引数据
                    del self._docs_by_index[index_name]
                    del self._metadata_by_index[index_name]
                    logger.info(f"已清除索引 {index_name} 的缓存")
        else:
            self._docs_by_index.clear()
            self._metadata_by_index.clear()
            self._loaded_files.clear()
            logger.info("已清除所有缓存") 