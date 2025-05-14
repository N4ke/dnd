from app.api.schemas.base import APIBase


class RAGResponse(APIBase):
    results: list
    confidence: float
    sources: list

class RAGUpload(APIBase):
    document_id: str
    chunks_processed: int
