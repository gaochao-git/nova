from typing import List, Optional, Dict, Any, AsyncGenerator, Union
from pydantic import BaseModel
from app.config.settings import settings
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

class Message(BaseModel):
    """聊天消息"""
    role: str
    content: str

class LLMResponse(BaseModel):
    """LLM 响应"""
    text: str
    raw_response: Optional[Dict[str, Any]] = None

class BaseLLM:
    """基础 LLM 类"""
    
    def __init__(
        self,
        api_key: str = settings.LLM_API_KEY,
        api_base: str = settings.LLM_API_URL,
        model: str = settings.LLM_MODEL_NAME,
        temperature: float = settings.LLM_TEMPERATURE,
        streaming: bool = False,
        callbacks: Optional[List[Any]] = None,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.temperature = temperature
        self.streaming = streaming
        self.callbacks = callbacks or []
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
    
    async def generate(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """生成回复"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[msg.dict() for msg in messages],
                temperature=temperature or self.temperature,
                stream=stream
            )
            
            if stream:
                async def stream_generator():
                    async for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                return stream_generator()
            else:
                return LLMResponse(
                    text=response.choices[0].message.content,
                    raw_response=response.dict()
                )
                
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")
            
    async def get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入向量"""
        raise NotImplementedError
