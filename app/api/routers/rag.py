from fastapi import APIRouter, UploadFile, File, Depends
from app.api.schemas.rag import RAGResponse, RAGUpload
from app.api.services.rag_service import RAGService


router = APIRouter(tags=["RAG Operations"])

@router.post("/upload", response_model=RAGUpload)
async def upload_document(
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(),
    system_name: str = "lore"
):
    return await rag_service.process_upload(file, system_name)

@router.get("/search", response_model=RAGResponse)
async def semantic_search(
    query: str,
    rag_service: RAGService = Depends(),
    system_name: str = "lore"
):
    return await rag_service.search(query, system_name)
