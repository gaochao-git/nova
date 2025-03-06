from typing import List, Dict, Any, Optional
from pydantic import BaseModel, HttpUrl
from enum import Enum

class FileType(str, Enum):
    TEXT = "text"
    PDF = "pdf"
    URL = "url"

class AddDocumentRequest(BaseModel):
    source: str  # URL、文本内容或文件路径
    type: FileType = FileType.TEXT

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class SearchResult(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    status: str
    query: str
    results: List[SearchResult]

class DocumentResponse(BaseModel):
    status: str
    message: str 