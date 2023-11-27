from typing import Any, List

from sentence_transformers import SentenceTransformer

def sbert_embeddings(model_name: str, words: List[str]) -> Any : 
    model = SentenceTransformer(model_name)
    embeddings = model.encode(words)
    return embeddings