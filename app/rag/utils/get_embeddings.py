from app.config.settings import Settings, settings


def get_embeddings():
        from langchain_huggingface import HuggingFaceEmbeddings
        
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cuda"},
            cache_folder=str(settings.EMBEDDINGS_DIR)
        )
