from pydantic import BaseModel
from typing import Optional, List

class ChatInputs(BaseModel):
    use_rag: bool = False
    knowledge_bases: Optional[List[str]] = None

class ChatRequest(BaseModel):
    inputs: ChatInputs
    query: str
    response_mode: str = "blocking"
    conversation_id: Optional[str] = None
    user: Optional[str] = ""

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[dict]] = None
    conversation_id: Optional[str] = None
    used_rag: bool = False 