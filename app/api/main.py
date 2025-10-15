from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.rag.rag_manager import RAGOrchestrator
from app.api.routers import chat, context, rag
from app.api.services.llm_service import LLMService
from app.config.settings import settings
from app.rag.utils.get_embeddings import get_embeddings
from redis.asyncio import Redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = Redis.from_url(settings.REDIS_URL)
    app.state.llm = LLMService.get_instance(settings)
    
    embeddings = get_embeddings()
    
    app.state.rag = RAGOrchestrator()
    rules_config = settings.RAG_SYSTEMS["rules"]
    app.state.rag.add_system(rules_config, embeddings)
    
    yield
    
    await app.state.redis.close()
    
    from pymilvus import connections
    connections.disconnect("default")
    
    del app.state.llm.llm

app = FastAPI(
    title="D&D AI Assistent",
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
    return {"message": "D&D AI Assistent is running"}
