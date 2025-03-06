from fastapi import APIRouter, HTTPException
import logging
from app.core.llm.chat import ChatService
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationListResponse,
    ConversationHistoryResponse
)
import uuid
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])
chat_service = ChatService()

@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"收到聊天请求: content={request.content}, use_rag={request.use_rag}")
        conversation_id = request.conversation_id or str(uuid.uuid4())
        logger.info(f"会话ID: {conversation_id}")
        
        response = await chat_service.get_response(
            conversation_id=conversation_id,
            message=request.content,
            use_rag=request.use_rag
        )
        
        logger.info(f"返回响应: {response}")
        return ChatResponse(
            conversation_id=conversation_id,
            response=response
        )
    except Exception as e:
        logger.error(f"处理聊天请求时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/chat/conversations")
async def list_conversations(conversation_id: Optional[str] = None):
    """列出对话列表或获取特定对话详情"""
    try:
        if conversation_id:
            messages = chat_service.get_conversation_history(conversation_id)
            if not messages:
                raise HTTPException(status_code=404, detail="对话不存在")
            return ConversationHistoryResponse(
                conversation_id=conversation_id,
                messages=messages
            )
        else:
            conversations = chat_service.list_conversations()
            return ConversationListResponse(conversations=conversations)
    except Exception as e:
        logger.error(f"获取对话信息时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 