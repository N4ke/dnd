from fastapi import UploadFile, Depends, Request
import os
import tempfile
from app.rag.rag_manager import RAGOrchestrator
from app.api.schemas.rag import RAGResponse, RAGUpload


def get_rag_orchestrator(request: Request) -> RAGOrchestrator:
    return request.app.state.rag

class RAGService:
    def __init__(self, orchestrator: RAGOrchestrator = Depends(get_rag_orchestrator)):
        self.orchestrator = orchestrator
    
    async def process_upload(
        self,
        file: UploadFile,
        system_name: str
    ) -> RAGUpload:
        rag_system = self.orchestrator.get_system(system_name)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            chunks = rag_system.ingest(tmp_path)
        finally:
            os.unlink(tmp_path)

        return RAGUpload(
            document_id=file.filename,
            chunks_processed=chunks
        )
    
    def search(self, query: str, system_name: str) -> RAGResponse:
        rag_system = self.orchestrator.get_system(system_name)
        results = rag_system.retrieve(query)
        return RAGResponse(
            results=[doc.page_content for doc in results],
            sources=[doc.metadata["source"] for doc in results]
        )
