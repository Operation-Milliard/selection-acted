from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str) -> SentenceTransformer:
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    print(f"Embedding {len(texts)} text(s)")
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def select_top_k_chunks_with_embeddings(
    model: SentenceTransformer,
    question: str,
    chunks: list[dict],
    chunk_embeddings: np.ndarray,
    top_k: int,
) -> list[dict]:
    if not chunks:
        return []
    question_embedding = embed_texts(model, [f"query: {question}"])[0]
    scores = chunk_embeddings @ question_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "score": float(scores[idx]),
            **chunks[idx],
        }
        for idx in top_indices
    ]
