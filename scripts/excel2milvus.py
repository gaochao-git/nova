#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import pandas as pd
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pymilvus import utility
from pymilvus import (
    connections,
    Collection,
)
from sentence_transformers import SentenceTransformer

def setup_logging(log_file: str = "excel2milvus.log"):
    """设置日志配置"""
    # 创建logger
    logger = logging.getLogger('excel2milvus')
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
        collection_name: str = "zjtj",
        db_name: str = "tj",
    ):
        self.collection_name = collection_name
        self.db_name = db_name
        self.host = host
        self.port = port
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
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

    def insert_data(self, data: Dict[str, List]):
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
            logger.debug(f"Successfully inserted {len(data.get('id', []))} records")
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

class ExcelProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def process_row(self, row: pd.Series, embedding_model) -> Dict:
        """处理单行数据"""
        try:
            # 生成向量嵌入
            article_title_embedding = embedding_model.encode(str(row['article_title']))
            keywords_embedding = embedding_model.encode(str(row['keywords']))
            abstract_embedding = embedding_model.encode(str(row['abstract']))
            
            # 返回处理后的行数据，确保向量是numpy数组格式
            return {
                "article_title_embedding": article_title_embedding,
                "keywords_embedding": keywords_embedding,
                "abstract_embedding": abstract_embedding,
                "article_title_text": str(row['article_title']),
                "keywords_text": str(row['keywords']),
                "abstract_text": str(row['abstract']),
                "article_id": str(row['article_id']),
                "scholar_id": str(row['scholar_id']),
                "year": int(row['year']) if pd.notna(row['year']) else 0
            }
        except Exception as e:
            logger.error(f"Error processing row: {str(e)}")
            raise

    def process_excel(self, excel_path: str, batch_size: int = 1000, max_workers: int = 4):
        """处理Excel文件并写入Milvus"""
        logger.info(f"开始处理Excel文件: {excel_path}")
        
        try:
            # 读取Excel文件
            df = pd.read_excel(excel_path)
            logger.info(f"成功加载Excel文件，共{len(df)}行")
            
            # 填充空值
            df['article_title'] = df['article_title'].fillna('')
            df['keywords'] = df['keywords'].fillna('')
            df['abstract'] = df['abstract'].fillna('')
            df['article_id'] = df['article_id'].fillna('')
            df['scholar_id'] = df['scholar_id'].fillna('')
            df['year'] = df['year'].fillna(0)
            
            processed_rows = []
            
            # 使用线程池并行处理行数据
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for _, row in df.iterrows():
                    future = executor.submit(self.process_row, row, self.embedding_model)
                    futures.append(future)
                
                # 使用tqdm显示进度
                for future in tqdm(as_completed(futures), total=len(futures), 
                                 desc="Processing rows"):
                    try:
                        result = future.result()
                        processed_rows.append(result)
                        
                        # 当处理的行数达到batch_size时，生成批次数据并写入
                        if len(processed_rows) >= batch_size:
                            yield processed_rows
                            processed_rows = []
                            
                    except Exception as e:
                        logger.error(f"Error processing row: {str(e)}")
            
            # 处理最后一个不完整的批次
            if processed_rows:
                yield processed_rows
            
        except Exception as e:
            logger.error(f"处理Excel文件时出错: {str(e)}")
            raise

def process_excel_file(excel_path: str, milvus_host: str = "localhost", milvus_port: str = "19530", 
                      batch_size: int = 1000, max_workers: int = 4):
    """处理单个Excel文件并插入到Milvus"""
    try:
        # 创建处理器实例
        processor = ExcelProcessor()
        milvus_service = MilvusService(host=milvus_host, port=milvus_port)
        
        # 处理Excel文件并批量写入
        for batch_data in processor.process_excel(excel_path, batch_size, max_workers):
            try:
                milvus_service.insert_data(batch_data)
                logger.info(f"Successfully inserted batch of {len(batch_data)} records")
            except Exception as e:
                logger.error(f"Failed to insert batch: {str(e)}")
        
        # 断开连接
        milvus_service.disconnect()
        
        logger.info(f"Successfully processed {excel_path}")
        return f"Successfully processed {excel_path}"
    
    except Exception as e:
        logger.error(f"Error processing {excel_path}: {str(e)}")
        return f"Error processing {excel_path}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Process Excel and insert into Milvus")
    parser.add_argument("excel_path", help="Path to Excel file")
    parser.add_argument("--host", default="localhost", help="Milvus host")
    parser.add_argument("--port", default="19530", help="Milvus port")
    parser.add_argument("--log-file", default="excel2milvus.log", help="Log file path")
    
    args = parser.parse_args()
    
    # 设置日志文件
    global logger
    logger = setup_logging(args.log_file)
    
    logger.info("Starting Excel processing")
    logger.info(f"Configuration: host={args.host}, port={args.port}")
    
    process_excel_file(args.excel_path, args.host, args.port)
    
    logger.info("Excel processing completed")

if __name__ == "__main__":
    main() 