from typing import Optional
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from app.config.settings import settings

class LLMFactory:
    @staticmethod
    def create_llm(llm_type: str = None, temperature: float = 0.7):
        llm_type = llm_type or settings.LLM_TYPE
        
        if llm_type == "local":
            return ChatOllama(
                model=settings.OLLAMA_MODEL_NAME,
                base_url=settings.OLLAMA_API_URL,
                temperature=temperature
            )
        elif llm_type == "api":
            return ChatOpenAI(
                temperature=temperature,
                model_name=settings.API_MODEL_NAME,
                openai_api_key=settings.API_KEY,
                openai_api_base=settings.API_URL
            )
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}") 