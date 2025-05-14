from typing import List
from langchain_core.documents import Document


def apply_metadata_rules(docs: List[Document], config) -> List[Document]:
    for doc in docs:
        doc.metadata.update({
            "rag_system": config.index_name,
            **config.metadata_rules
        })
    return docs