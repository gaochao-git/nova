#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pymilvus import utility
from pymilvus import (
    connections,
    Collection,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def setup_logging(log_file: str = "doc2milvus.log"):
    """设置日志配置"""
    # 创建logger
    logger = logging.getLogger('doc2milvus')
    logger.setLevel(logging.INFO)
    
    # 清理已存在的处理器
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 创建文件处理器
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logging()

class MilvusService:
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        collection_name: str = "db_docs",
        db_name: str = "nova",
    ):
        self.collection_name = collection_name
        self.db_name = db_name
        self.host = host
        self.port = port
        
        # 建立连接
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                db_name=self.db_name
            )
            
            # 打印所有集合
            collections = utility.list_collections()
            logger.info(f"Available collections in database {self.db_name}:")
            for coll in collections:
                logger.info(f"- {coll}")
            
            # 获取collection
            self.collection = Collection(self.collection_name)
            logger.info(f"Successfully connected to Milvus server at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus server: {str(e)}")
            raise

    def insert_data(self, data: Dict[str, Any]):
        """插入数据到Milvus"""
        try:
            # 确保连接存在
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                db_name=self.db_name
            )
            
            # 检查并获取collection
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
                logger.debug(f"Collection {self.collection_name} loaded successfully")
            else:
                error_msg = f"Collection {self.collection_name} does not exist in database {self.db_name}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # 执行插入
            self.collection.insert(data)
            self.collection.flush()
            logger.debug(f"Successfully inserted {len(data)} records")
        except Exception as e:
            logger.error(f"Failed to insert data: {str(e)}")
            raise

    def disconnect(self):
        """断开与Milvus服务器的连接"""
        try:
            connections.disconnect("default")
            logger.info("Successfully disconnected from Milvus server")
        except Exception as e:
            logger.error(f"Failed to disconnect from Milvus server: {str(e)}")

class PDFProcessor:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 200,
        chunk_overlap: int = 50,
    ):
        self.model = SentenceTransformer(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def load_pdf(self, pdf_path: str) -> List[dict]:
        """加载并处理PDF文件，按页面处理"""
        logger.info(f"开始加载{pdf_path}")
        loader = PyPDFLoader(pdf_path)
        logger.info(f"加载完成{pdf_path}")
        pages = loader.load()
        logger.info(f"共加载{len(pages)}页")
        
        all_pages = []
        for page in pages:
            all_pages.append({
                "text_content": page.page_content,
                "page_number": page.metadata["page"],
                "file_path": pdf_path
            })
        return all_pages

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本的向量嵌入"""
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

def process_single_page(args):
    """处理单个页面的函数"""
    page, processor = args
    try:
        # 从文件路径中提取数据库类型（最后一层目录名）
        db_type = os.path.basename(os.path.dirname(page["file_path"]))
        
        # 生成embedding
        embedding = processor.get_embeddings([page["text_content"]])[0]
        return {
            "text_content": page["text_content"],
            "text_embedding": embedding,
            "file_path": page["file_path"],
            "page_number": page["page_number"],
            "db_type": db_type
        }
    except Exception as e:
        logger.error(f"Error processing page {page['page_number']}: {str(e)}")
        raise

def process_single_pdf(args):
    """处理单个PDF文件的函数"""
    pdf_path, milvus_service = args
    try:
        logger.info(f"Start processing {pdf_path}")
        processor = PDFProcessor()

        # 加载PDF，按页面处理
        pages = processor.load_pdf(pdf_path)
        if not pages:
            logger.warning(f"No content extracted from {pdf_path}")
            return f"No content extracted from {pdf_path}"

        total_pages = len(pages)
        logger.info(f"Found {total_pages} pages in {os.path.basename(pdf_path)}")

        # 使用线程池并发处理页面
        processed_pages = []
        with ThreadPoolExecutor(max_workers=50) as executor:  # 50个并发
            # 准备任务列表
            page_tasks = [(page, processor) for page in pages]
            
            # 提交所有任务并使用tqdm显示进度
            futures = [executor.submit(process_single_page, args) for args in page_tasks]
            
            # 收集结果
            for future in tqdm(as_completed(futures), 
                             total=len(pages), 
                             desc=f"Processing pages in {os.path.basename(pdf_path)}"):
                try:
                    result = future.result()
                    processed_pages.append(result)
                except Exception as e:
                    logger.error(f"Failed to process page in {pdf_path}: {str(e)}")
                    continue

        # 批量插入所有处理完的页面
        if processed_pages:
            try:
                # 一次插入一批数据
                data_to_insert = []
                for page in processed_pages:
                    data_to_insert.append({
                        "text_content": str(page["text_content"]),  # 确保是字符串
                        "text_embedding": page["text_embedding"],
                        "file_path": str(page["file_path"]),       # 确保是字符串
                        "page_number": int(page["page_number"]),   # 确保是整数
                        "db_type": str(page["db_type"])           # 新增字段
                    })
                
                # 执行插入
                milvus_service.insert_data(data_to_insert)
                logger.info(f"Successfully inserted {len(processed_pages)} pages for {os.path.basename(pdf_path)}")
            except Exception as e:
                logger.error(f"Failed to batch insert pages for {pdf_path}: {str(e)}")
                raise

        logger.info(f"Successfully processed {pdf_path}")
        return f"Successfully processed {pdf_path}"
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        return f"Error processing {pdf_path}: {str(e)}"

def process_pdfs(pdf_dir: str, milvus_host: str = "localhost", milvus_port: str = "19530", db_name: str = "nova", max_workers: int = 4):
    """使用线程池处理多个PDF文件"""
    # 获取所有PDF文件
    pdf_files = [
        os.path.join(pdf_dir, f) 
        for f in os.listdir(pdf_dir) 
        if f.lower().endswith('.pdf')
    ]
    
    if not pdf_files:
        logger.warning("No PDF files found in the specified directory.")
        return

    logger.info(f"Found {len(pdf_files)} PDF files to process")
    logger.info(f"Using {max_workers} worker threads")

    try:
        # 创建一个Milvus服务实例
        milvus_service = MilvusService(
            host=milvus_host,
            port=milvus_port,
            db_name=db_name
        )

        # 创建参数列表
        args_list = [(pdf_file, milvus_service) for pdf_file in pdf_files]

        # 使用线程池处理文件
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务并使用tqdm显示进度
            futures = [executor.submit(process_single_pdf, args) for args in args_list]
            
            # 使用tqdm显示进度
            for future in tqdm(as_completed(futures), total=len(pdf_files), desc="Processing PDFs"):
                result = future.result()
                logger.info(result)

        # 所有文件处理完成后断开连接
        milvus_service.disconnect()
        logger.info("All files processed, disconnected from Milvus server")

    except Exception as e:
        logger.error(f"Error in process_pdfs: {str(e)}")
        try:
            milvus_service.disconnect()
        except:
            pass
        raise

def main():
    parser = argparse.ArgumentParser(description="Process PDFs and insert into Milvus")
    parser.add_argument("pdf_dir", help="Directory containing PDF files")
    parser.add_argument("--host", default="82.156.146.51", help="Milvus host")
    parser.add_argument("--port", default="19530", help="Milvus port")
    parser.add_argument("--db", default="nova", help="Milvus database name")
    parser.add_argument("--workers", type=int, default=10, help="Number of worker threads")
    parser.add_argument("--log-file", default="doc2milvus.log", help="Log file path")
    
    args = parser.parse_args()
    
    
    # 设置日志文件
    global logger
    logger = setup_logging(args.log_file)
    
    logger.info("Starting PDF processing")
    logger.info(f"Configuration: host={args.host}, port={args.port}, db={args.db}, workers={args.workers}")
    
    process_pdfs(args.pdf_dir, args.host, args.port, args.db, args.workers)
    
    logger.info("PDF processing completed")

if __name__ == "__main__":
    main()
