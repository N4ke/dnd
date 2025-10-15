from typing import List
from langchain_core.documents import Document


def apply_metadata_rules(docs: List[Document], config) -> List[Document]:
    allowed_metadata_keys = {
        "source",
        "page",
        "rag_system",
    }
    
    for doc in docs:
        filtered_metadata = {
            k: v for k, v in doc.metadata.items()
            if k in allowed_metadata_keys
        }
        
        filtered_metadata.update({
            "rag_system": config.index_name,
            **config.metadata_rules
        })
        
        doc.metadata = filtered_metadata
        
    return docs