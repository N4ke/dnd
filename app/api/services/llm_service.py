from typing import AsyncGenerator, Optional
from llama_cpp import Llama
from app.config.settings import Settings
from app.api.schemas.chat import ChatResponse
import asyncio
from concurrent.futures import ThreadPoolExecutor


class LLMService:
    _instance: Optional["LLMService"] = None
    
    def __init__(self, settings: Settings):
        if LLMService._instance is not None:
            raise RuntimeError("Use get_instance() instead")
        
        self.llm = Llama(
            model_path=settings.LLM.model_path,
            n_gpu_layers=settings.LLM.n_gpu_layers,
            n_ctx=settings.LLM.n_ctx,
            verbose=False
        )
        
        self.executor = ThreadPoolExecutor(max_workers=1)
            
    async def _run_in_executor(self, func, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, lambda: func(*args, **kwargs))
        
    @classmethod
    def get_instance(cls, settings: Settings) -> "LLMService":
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance
    
    async def process_request(self, request) -> ChatResponse:
        response = await self._run_in_executor(
            self.llm.create_chat_completion,
            messages=[{"role": "user", "content": request.message}],
            temperature=request.temperature
        )
        
        return ChatResponse(
            content=response["choices"][0]["message"]["content"],
            is_final=True
        )
