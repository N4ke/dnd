from langchain_milvus import Milvus
from langchain_core.documents import Document
from typing import List, Dict, Any
from app.config.schemas import RAGConfig


class MilvusManager:
    def __init__(self, config: RAGConfig, embeddings):
        self.config = config
        self.vector_store = Milvus(
            embedding_function=embeddings,
            connection_args=config.connection_args,
            collection_name=config.index_name,
            index_params=config.index_params,
            drop_old=config.drop_old,
            consistency_level="Session"
        )

    def upsert_documents(self, documents: List[Document], batch_size: int=64):
        for i in range(0, len(documents), batch_size):
            chunk = documents[i : i + batch_size]
            self.vector_store.add_documents(documents=chunk, batch_size=batch_size)
    
    def query(self, text: str, filters: Dict[str, Any] = None) -> List[Document]:
        return self.vector_store.similarity_search(
            query=text,
            k=self.config.top_k,
            filter=filters
        )