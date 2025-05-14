import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List
from langchain_core.documents import Document
from dotenv import load_dotenv
from app.config.settings import Settings, settings
from app.rag.document_processor import DocumentProcessor
from app.rag.vector_store import MilvusManager


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MilvusDataloader:
    def __init__(self, config: Settings, rag_system_name: str = "core_rules"):
        self.settings = config
        self.rag_config = config.RAG_SYSTEMS[rag_system_name]
        self.processor = DocumentProcessor(self.rag_config)
        self.vector_db = MilvusManager(
            embeddings=self._get_embeddings(),
            config=self.rag_config
        )
    
    def _get_embeddings(self):
        from langchain_huggingface import HuggingFaceEmbeddings
        
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cuda"},
            cache_folder=str(settings.MODELS_DIR) + "/embeddings"
        )
        
    def _process_documents(self, data_dir: Path) -> List[Document]:
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} not founded")
        
        processed = []
        for ext in ["pdf", "txt", "md"]:
            for file_path in data_dir.glob(f"**/*.{ext}"):
                try:
                    docs = self.processor.process_file(str(file_path))
                    processed.extend(docs)
                    logger.info(f"Processed {file_path}: {len(docs)} chunks")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
        
        return processed
    
    def run(self, data_dir: Path):
        logger.info(f"Loading data for '{self.rag_config.namespace}' namespace")
        
        documents = self._process_documents(data_dir)
        
        self.vector_db.upsert_documents(documents)
        
        logger.info(f"Successfully loaded {len(documents)} documents")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loading data to Pinecone databse")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/raw/rules"
    )
    parser.add_argument(
        "--rag-system",
        type=str,
        default="core_rules",
        choices=["core_rules", "world_lore"]
    )
    args = parser.parse_args()
    
    start_time = datetime.now()
    logger.info(f"Started loading at {start_time}")
    
    try:
        loader = MilvusDataloader(settings, args.rag_system)
        loader.run(Path(args.data_dir))
    except Exception as e:
        logger.error(f"Loading failed: {str(e)}")
        raise
    
    duration = datetime.now() - start_time
    logger.info(f"Loading completed in {duration.total_seconds():.2f} seconds")