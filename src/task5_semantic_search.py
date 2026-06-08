"""
Task 5 — Semantic Search Module.

Dense retrieval sử dụng cosine similarity trên ChromaDB
với embedding model sentence-transformers/all-MiniLM-L6-v2 (384 dim).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.task4_chunking_indexing import (
    EMBEDDING_MODEL,
    get_chroma_collection,
    _get_embedding_model,
)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity (cosine).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # Cosine similarity (0..1, càng cao càng liên quan)
            'metadata': dict
        }
        Sorted by score descending.
    """
    model = _get_embedding_model()
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    collection = get_chroma_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results and results["documents"] and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, dists):
            # ChromaDB cosine distance = 1 - cosine_similarity
            similarity = 1.0 - dist
            output.append({
                "content": doc,
                "score": round(similarity, 4),
                "metadata": meta,
            })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] [{r['metadata'].get('source', '?')}] {r['content'][:100]}...")
