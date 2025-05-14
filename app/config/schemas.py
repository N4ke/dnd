from pydantic import BaseModel, Field
from typing import Dict, List, Any


class RAGConfig(BaseModel):
    connection_args: Dict[str, Any]
    index_name: str
    embedding_model: str
    embedding_dim: int
    index_params: Dict[str, Any] = Field(
        default_factory=lambda: {"index_type": "HNSW", "metric_type": "cosine"}
    )
    drop_old: bool = False
    metric: str = "cosine"
    chunk_size: int = 800
    chunk_overlap: int = 200
    metadata_rules: Dict = {}
    document_types: List[str] = ["pdf", "md", "txt"]
    top_k: int = 3


class LLMConfig(BaseModel):
    model_path: str
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    main_gpu: int = 0
    n_batch: int = 512
    temperature: float = 0.7
