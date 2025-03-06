from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from datetime import datetime

class ChatRequest(BaseModel):
    content: str
    conversation_id: Optional[str] = None
    use_rag: bool = False

class ChatResponse(BaseModel):
    conversation_id: str
    response: str

class MessageSchema(BaseModel):
    role: str
    content: str
    timestamp: str

class ConversationSchema(BaseModel):
    conversation_id: str
    message_count: int
    last_message: Optional[str] = None

class ConversationListResponse(BaseModel):
    conversations: List[ConversationSchema]

class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    messages: List[MessageSchema]
