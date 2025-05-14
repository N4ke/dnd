from fastapi import UploadFile, Depends
from app.rag.rag_manager import RAGOrchestrator
from app.api.schemas.rag import RAGResponse, RAGUpload


class RAGService:
    def __init__(self, orchestrator: RAGOrchestrator = Depends()):
        self.orchestrator = orchestrator
    
    async def process_upload(self, file: UploadFile) -> RAGUpload:
        content = await file.read()
        return RAGUpload(
            document_id=file.filename,
            chunks_processed=150
        )
    
    async def search(self, query: str) -> RAGResponse:
        results = await self.orchestrator.search_all(query)
        return RAGResponse(
            results=results,
            confidence=0.92,
            sources=[doc.metadata["source"] for doc in results]
        )
