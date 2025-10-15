from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import SpacyTextSplitter
from app.config.schemas import RAGConfig
from app.rag.utils.metadata import apply_metadata_rules


class DocumentProcessor:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.splitter = self._create_splitter()
    
    def _create_splitter(self):
        return SpacyTextSplitter(
            pipeline="ru_core_news_sm",
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separator=" "
        )
    
    def process_file(self, file_path: str) -> List[Document]:
        loader = self._get_loader(file_path)
        docs = loader.load()
        chunks = self.splitter.split_documents(docs)
        return apply_metadata_rules(chunks, self.config)

    def _get_loader(self, file_path: str):
        if file_path.endswith(".pdf"):
            return PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            return TextLoader(file_path)
        raise ValueError(f"Unsupported format: {file_path}")