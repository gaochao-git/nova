#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection, utility
from pymilvus import AnnSearchRequest
from sentence_transformers import SentenceTransformer

def setup_logging(log_file: str = "search_articles.log"):
    """设置日志配置"""
    logger = logging.getLogger('search_articles')
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logging()

class ArticleSearcher:
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        collection_name: str = "articles",
        db_name: str = "tj",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.collection_name = collection_name
        self.db_name = db_name
        self.embedding_model = SentenceTransformer(model_name)
        
        try:
            connections.connect(
                alias="default",
                host=host,
                port=port,
                db_name=db_name
            )
            self.collection = Collection(collection_name)
            self.collection.load()
            logger.info(f"Successfully connected to Milvus and loaded collection {collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize: {str(e)}")
            raise

    def search(
        self,
        title: Optional[str] = None,
        keywords: Optional[str] = None,
        abstract: Optional[str] = None,
        top_k: int = 5,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        多字段联合向量搜索
        """
        try:
            results = []
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            output_fields = ["article_title_text", "keywords_text", "abstract_text", "year"]
            
            # 处理标题搜索
            if title:
                title_embedding = self.embedding_model.encode(title)
                title_results = self.collection.search(
                    data=[title_embedding],
                    anns_field="article_title_embedding",
                    param=search_params,
                    limit=top_k,
                    output_fields=output_fields
                )
                print("Title search results:", title_results)
                results.append(title_results)
            
            # 处理关键词搜索
            if keywords:
                keywords_embedding = self.embedding_model.encode(keywords)
                keywords_results = self.collection.search(
                    data=[keywords_embedding],
                    anns_field="keywords_embedding",
                    param=search_params,
                    limit=top_k,
                    output_fields=output_fields
                )
                print("Keywords search results:", keywords_results)
                results.append(keywords_results)
            
            # 处理摘要搜索
            if abstract:
                abstract_embedding = self.embedding_model.encode(abstract)
                abstract_results = self.collection.search(
                    data=[abstract_embedding],
                    anns_field="abstract_embedding",
                    param=search_params,
                    limit=top_k,
                    output_fields=output_fields
                )
                print("Abstract search results:", abstract_results)
                results.append(abstract_results)
            
            if not results:
                logger.warning("No search fields provided")
                return []
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise

    def disconnect(self):
        """断开连接"""
        try:
            connections.disconnect("default")
            logger.info("Successfully disconnected from Milvus")
        except Exception as e:
            logger.error(f"Failed to disconnect: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Search articles in Milvus")
    parser.add_argument("--title", help="Title to search for")
    parser.add_argument("--keywords", help="Keywords to search for")
    parser.add_argument("--abstract", help="Abstract to search for")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--threshold", type=float, default=0.5, 
                       help="Similarity score threshold")
    parser.add_argument("--host", default="localhost", help="Milvus host")
    parser.add_argument("--port", default="19530", help="Milvus port")
    parser.add_argument("--collection", default="zjtj", help="Collection name")
    parser.add_argument("--db", default="tj", help="Database name")
    
    args = parser.parse_args()
    
    try:
        searcher = ArticleSearcher(
            host=args.host,
            port=args.port,
            collection_name=args.collection,
            db_name=args.db
        )
        
        results = searcher.search(
            title=args.title,
            keywords=args.keywords,
            abstract=args.abstract,
            top_k=args.top_k,
            score_threshold=args.threshold
        )
        
        print("\nRaw Search Results:")
        print(results)
        
        searcher.disconnect()
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 