from typing import Optional, List, Dict
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
import logging
import uuid
import time
import json
from app.core.llm.chat import ChatService
from app.core.llm.new_rag import RAGService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["db_qa"])

class RAGConfig(BaseModel):
    """RAG 相关配置"""
    enabled: bool = False
    db_types: Optional[List[str]] = None
    vectorQuery: Optional[str] = None  # 向量搜索关键词
    scalarQuery: Optional[str] = None  # 标量搜索关键词

class DBQARequest(BaseModel):
    """数据库问答请求"""
    question: str
    prompt: Optional[str] = None
    conversation_id: Optional[str] = None
    rag_config: Optional[RAGConfig] = RAGConfig()

def get_services():
    rag_service = RAGService()
    chat_service = ChatService(temperature=0.0)
    return {"rag_service": rag_service, "chat_service": chat_service}

class SSEFormatter:
    @staticmethod
    def get_default_metadata() -> Dict:
        """获取默认的元数据"""
        return {
            "metadata": {
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "prompt_price": "0.0",
                    "completion_price": "0.0",
                    "total_price": "0.0",
                    "currency": "USD",
                    "latency": 0.0
                }
            },
            "files": None
        }

    @staticmethod
    def format_message(
        event: str,
        answer: str = "",
        metadata: Optional[Dict] = None,
        message_info: Optional[Dict] = None
    ) -> str:
        """格式化 SSE 消息"""
        data = {
            "event": event,
            "conversation_id": message_info["conversation_id"],
            "message_id": message_info["message_id"],
            "created_at": message_info["created_at"],
            "task_id": message_info["task_id"],
            "id": message_info["message_id"],
            "answer": answer,
            "from_variable_selector": None
        }
        if metadata:
            data.update(metadata)
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    @staticmethod
    def format_end_message(message_info: Dict) -> str:
        """格式化结束消息"""
        return SSEFormatter.format_message(
            "message_end",
            "",
            metadata=SSEFormatter.get_default_metadata(),
            message_info=message_info
        )

    @staticmethod
    def format_error_message(error: str, message_info: Dict) -> str:
        """格式化错误消息"""
        return SSEFormatter.format_message("error", str(error), message_info=message_info)

@router.post("/api/db/qa/stream")
async def db_qa_stream(
    request: DBQARequest,
    services: dict = Depends(get_services)
):
    """基于数据库文档的流式问答"""
    try:
        rag_service = services["rag_service"]
        chat_service = services["chat_service"]
        
        logger.info(
            f"收到流式问答请求: question={request.question}, "
            f"rag_config={request.rag_config}"
        )
        
        # 获取会话相关ID
        conversation_id = request.conversation_id or str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())
        created_at = int(time.time())
        
        message_info = {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "created_at": created_at,
            "task_id": task_id
        }
        
        async def generate_response():
            try:
                logger.info("Starting generate_response")
                if request.rag_config.enabled:
                    logger.info("RAG is enabled")
                    
                    # 构建搜索查询
                    vector_query = request.rag_config.vectorQuery or request.question
                    scalar_query = request.rag_config.scalarQuery
                    logger.info(f"Search queries - vector: {vector_query}, scalar: {scalar_query}")
                    
                    # 获取所有请求的数据库的上下文
                    all_contexts = []
                    db_types_list = []
                    
                    # 遍历请求中的数据库类型
                    yield SSEFormatter.format_message("message", "开始检索相关文档...", message_info=message_info)
                    for db_type in request.rag_config.db_types:
                        # 获取特定数据库的搜索上下文
                        contexts = await rag_service.get_search_context(
                            vector_query=vector_query,
                            scalar_query=scalar_query,
                            db_type=db_type
                        )
                        # 无论是否找到内容，都添加对应的提示
                        formatted_context = f"------{db_type}数据库相关内容------\n"
                        if contexts:  # 如果找到了相关内容
                            formatted_context += "\n\n".join(contexts)
                            db_types_list.append(db_type)
                        else:  # 如果没有找到相关内容
                            formatted_context += "未找到相关内容"
                        
                        all_contexts.append(formatted_context)
                    
                    # 合并所有上下文
                    combined_context = "\n\n".join(all_contexts)
                    logger.info(f"Found content for databases: {db_types_list}")
                    
                    # 构建提示词
                    prompt = (request.prompt or PROMPT_TEMPLATE).format(
                        context=combined_context,
                        question=request.question,
                        db_types=", ".join(db_types_list) if db_types_list else "未找到相关数据库"
                    )
                    logger.info("Prompt constructed")
                else:
                    logger.info("RAG is disabled, using default prompt")
                    prompt = f"""你是一个专业的数据库助手。请回答以下问题：\n\n问题：{request.question}\n\n请基于你的专业知识提供准确、清晰的回答。"""
                logger.info(f"Prompt: {prompt}")
                # 流式回答内容
                logger.info("Starting stream response generation")
                yield SSEFormatter.format_message("message", "正在思考中...", message_info=message_info)
                async for chunk in chat_service.get_stream_response(
                    conversation_id=conversation_id,
                    message=prompt
                ):
                    yield SSEFormatter.format_message("message", chunk, message_info=message_info)

                # 发送结束消息
                logger.info("Sending end message")
                yield SSEFormatter.format_end_message(message_info)

            except Exception as e:
                logger.error(f"生成流式响应时出错: {str(e)}", exc_info=True)
                yield SSEFormatter.format_error_message(str(e), message_info)
        
        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked",
            }
        )
        
    except Exception as e:
        logger.error(f"处理流式问答请求时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 
        