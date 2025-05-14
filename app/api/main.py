from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.rag.rag_manager import RAGOrchestrator
from app.api.routers import chat, context, rag
from app.api.services.llm_service import LLMService
from app.config.settings import settings
from redis.asyncio import Redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    from langchain_huggingface import HuggingFaceEmbeddings
    
    app.state.redis = Redis.from_url(settings.REDIS_URL)
    app.state.llm = LLMService.get_instance(settings)
    
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        cache_folder=str(settings.MODELS_DIR / "embeddings")
    )
    
    app.state.rag = RAGOrchestrator()
    for rag_name, rag_config in settings.RAG_SYSTEMS.items():
        app.state.rag.add_system(rag_config, embeddings)
    
    yield
    
    await app.state.redis.close()
    
    del app.state.llm

app = FastAPI(
    title="D&D AI Master",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(context.router)
app.include_router(rag.router)

@app.get("/")
async def root():
    return {"message": "D&D AI Master is running"}
