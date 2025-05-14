from typing import AsyncGenerator, Optional
from llama_cpp import Llama
from torch.cuda import empty_cache
from app.config.settings import Settings
from app.api.schemas.chat import ChatResponse


class LLMService:
    _instance: Optional["LLMService"] = None
    
    def __init__(self, settings: Settings):
        if LLMService._instance is not None:
            raise RuntimeError("Use get_instance() instead")
        
        self.llm = Llama(
            model_path=settings.LLM.model_path,
            n_gpu_layers=settings.LLM.n_gpu_layers,
            n_ctx=settings.LLM.n_ctx,
            verbose=True
        )
        
    @classmethod
    def get_instance(cls, settings: Settings) -> "LLMService":
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance
    
    async def stream_response(self, request) -> AsyncGenerator[ChatResponse, None]:
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": request.message}],
            stream=True,
            temperature=request.temperature
        )
        
        for chunk in response:
            content = chunk["choices"][0]["delta"].get("content", "")
            yield ChatResponse(content=content, is_final=False)
        
        yield ChatResponse(content="", is_final=True)
    
    async def process_request(self, request) -> ChatResponse:
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": request.message}],
            temperature=request.temperature
        )
        
        empty_cache()
        
        return ChatResponse(
            content=response["choices"][0]["message"]["content"],
            is_final=True
        )
