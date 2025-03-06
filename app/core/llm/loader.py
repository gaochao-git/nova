from typing import List, Optional
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader, Docx2txtLoader
from langchain.docstore.document import Document
from app.config.settings import settings
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self):
        self.chunk_size = getattr(settings, 'CHUNK_SIZE', 1000)
        self.chunk_overlap = getattr(settings, 'CHUNK_OVERLAP', 200)
        self.max_file_size = getattr(settings, 'MAX_FILE_SIZE', 100 * 1024 * 1024)  # 默认100MB
        
        logger.info(f"DocumentLoader配置: chunk_size={self.chunk_size}, "
                   f"chunk_overlap={self.chunk_overlap}, "
                   f"max_file_size={self.max_file_size / 1024 / 1024}MB")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

    def _check_file_size(self, file_path: str) -> bool:
        """检查文件大小是否超过限制"""
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / 1024 / 1024
        logger.info(f"文件大小: {file_size_mb:.2f}MB ({file_size} bytes): {file_path}")
        
        if file_size > self.max_file_size:
            logger.warning(f"文件过大 ({file_size_mb:.2f}MB > {self.max_file_size / 1024 / 1024}MB): {file_path}")
            return False
        return True

    def _load_pdf(self, file_path: str) -> List[Document]:
        """分块加载PDF文件"""
        try:
            if not self._check_file_size(file_path):
                return []

            logger.info(f"开始加载PDF文件: {file_path}")
            loader = PyPDFLoader(file_path)  # 这个位置很快，30M用不了1秒
            pages = loader.load_and_split()  # 这个位置很耗时3分钟左右处理5000个页面
            
            # 使用tqdm显示分割进度
            documents = []
            for page in tqdm(pages, desc="处理PDF页面"):
                chunks = self.text_splitter.split_documents([page])
                documents.extend(chunks)
            
            logger.info(f"成功加载PDF文件，共 {len(documents)} 个文本块")
            return documents
            
        except Exception as e:
            logger.error(f"加载PDF文件失败: {str(e)}")
            return []

    def _load_docx(self, file_path: str) -> List[Document]:
        """加载Word文档"""
        try:
            if not self._check_file_size(file_path):
                return []

            logger.info(f"开始加载Word文档: {file_path}")
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"成功加载Word文档，共 {len(chunks)} 个文本块")
            return chunks
            
        except Exception as e:
            logger.error(f"加载Word文档失败: {str(e)}")
            return []

    def _load_text(self, file_path: str) -> List[Document]:
        """加载文本文件"""
        try:
            if not self._check_file_size(file_path):
                return []

            logger.info(f"开始加载文本文件: {file_path}")
            loader = TextLoader(file_path)
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"成功加载文本文件，共 {len(chunks)} 个文本块")
            return chunks
            
        except Exception as e:
            logger.error(f"加载文本文件失败: {str(e)}")
            return []

    def load_and_split(self, source: str, file_type: str = "text") -> List[Document]:
        """加载并分割文档"""
        try:
            if not os.path.exists(source):
                logger.error(f"文件不存在: {source}")
                return []
                
            if file_type.lower() == "pdf":
                return self._load_pdf(source)
            elif file_type.lower() == "docx":
                return self._load_docx(source)
            elif file_type.lower() == "text":
                return self._load_text(source)
            else:
                logger.error(f"不支持的文件类型: {file_type}")
                return []
                
        except Exception as e:
            logger.error(f"处理文档失败: {str(e)}")
            return [] 