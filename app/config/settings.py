from typing import Dict
from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from pathlib import Path
from app.config.schemas import RAGConfig, LLMConfig

class Settings(BaseSettings):
    REDIS_URL: str = "redis://localhost:6379/0"

    # Directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DATABASE_DIR: Path = BASE_DIR / "database"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Models
    MAIN_LLM: str = "saiga_nemo_12b.Q4_K_M_gguf"
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    MAX_HISTORY_TOKENS: int = 20000
    SESSION_TTL: int = 3600

    RAG_SYSTEMS: Dict[str, RAGConfig] = {
        "rules": RAGConfig(
            connection_args={"uri": str(DATABASE_DIR / "rules.db")},
            index_name="rules",
            embedding_model=str(MODELS_DIR /"embeddings/paraphrase-multilingual-mpnet-base-v2"),
            embedding_dim=768
        ),
        "lore": RAGConfig(
            connection_args={"uri": str(DATABASE_DIR / "lore.db")},
            index_name="lore",
            embedding_model=str(MODELS_DIR / "embeddings/paraphrase-multilingual-mpnet-base-v2"),
            embedding_dim=768
        )
    }
    
    LLM: LLMConfig = LLMConfig(
        model_path=str(MODELS_DIR / "llms/saiga_nemo_12b.Q4_K_M.gguf"),
        n_ctx=4096,
        n_gpu_layers=20
    )
    
    model_config = ConfigDict(
        env_file=".env",
        env_nested_delimiter="__"
    )

settings = Settings()