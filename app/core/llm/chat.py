from typing import Dict, Optional, List
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from app.config.settings import settings
from .factory import LLMFactory
from datetime import datetime
import logging
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, llm_type: str = None, temperature: float = 0.7):
        self.chat_model = LLMFactory.create_llm(
            llm_type=llm_type,
            temperature=temperature
        )
        self.conversations: Dict[str, List] = {}  # 存储会话历史

    def get_or_create_conversation(self, conversation_id: str) -> ConversationChain:
        if conversation_id not in self.conversations:
            memory = ConversationBufferMemory()
            self.conversations[conversation_id] = ConversationChain(
                llm=self.chat_model,
                memory=memory,
                verbose=True
            )
        return self.conversations[conversation_id]

    def get_conversation_history(self, conversation_id: str) -> Optional[List[Dict]]:
        """获取会话历史"""
        if conversation_id not in self.conversations:
            return None
        memory = self.conversations[conversation_id].memory
        return [
            {
                "role": "human" if i % 2 == 0 else "assistant",
                "content": msg.content,
                "timestamp": datetime.now().isoformat()
            }
            for i, msg in enumerate(memory.chat_memory.messages)
        ]

    def list_conversations(self) -> List[Dict]:
        """列出所有会话"""
        return [
            {
                "conversation_id": conv_id,
                "message_count": len(conv.memory.chat_memory.messages),
                "last_message": conv.memory.chat_memory.messages[-1].content if conv.memory.chat_memory.messages else None
            }
            for conv_id, conv in self.conversations.items()
        ]

    def remove_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return True
        return False

    def clear_conversations(self):
        """清空所有会话"""
        self.conversations.clear()

    async def get_response(self, conversation_id: str, message: str) -> str:
        conversation = self.get_or_create_conversation(conversation_id)
        logger.info(f"最终提问: {message}")
        response = conversation.predict(input=message)
        logger.info(f"AI 回复: {response}")
        
        return response 

    async def get_stream_response(self, conversation_id: str, message: str):
        """获取流式响应"""
        try:
            # 获取或初始化会话历史
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
            
            # 添加用户新消息
            self.conversations[conversation_id].append(HumanMessage(content=message))
            logger.info(f"历史对话: {len(self.conversations[conversation_id])}")
            # 构建完整的消息历史
            messages = self.conversations[conversation_id]
            
            # 使用 LLM 的流式接口
            response_content = ""
            async for chunk in self.chat_model.astream(messages):
                if hasattr(chunk, 'content'):
                    response_content += chunk.content
                    yield chunk.content
                else:
                    response_content += chunk
                    yield chunk
            
            # 保存AI回复到会话历史
            self.conversations[conversation_id].append(AIMessage(content=response_content))
                    
        except Exception as e:
            logger.error(f"获取流式响应时出错: {str(e)}", exc_info=True)
            yield f"错误: {str(e)}" 