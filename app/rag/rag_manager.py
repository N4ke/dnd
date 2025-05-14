from typing import Dict, List
from langchain_core.documents import Document
from app.config.schemas import RAGConfig
from app.rag.document_processor import DocumentProcessor
from app.rag.vector_store import MilvusManager


class RAGSystem:
    def __init__(self, config: RAGConfig, embeddings):
        self.config = config
        self.embeddings = embeddings
        self.processor = DocumentProcessor(config)
        self.vector_db = MilvusManager(config, embeddings)
    
    def ingest(self, file_path: str):
        docs = self.processor.process_file(file_path)
        self.vector_db.upsert_documents(docs)
    
    def retrieve(self, query: str, **filters) -> List[Document]:
        return self.vector_db.query(query, filters)


class RAGOrchestrator:
    def __init__(self):
        self.systems: Dict[str, RAGSystem] = {}
    
    def add_system(self, config: RAGConfig, embeddings):
        self.systems[config.index_name] = RAGSystem(config, embeddings)
    
    def get_system(self, name: str) -> RAGSystem:
        return self.systems[name]

    async def search_all(self, query: str) -> List[Document]:
        results = []
        for system in self.systems.values():
            results.extend(await system.retrieve(query))
        return results