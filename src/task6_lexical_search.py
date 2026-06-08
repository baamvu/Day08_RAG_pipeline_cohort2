"""
Task 6 — Lexical Search Module (BM25).

BM25 (Best Matching 25) hoạt động dựa trên:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

Cài đặt:
    pip install rank-bm25
"""

import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.task4_chunking_indexing import chunk_documents, load_documents

_bm25 = None
_corpus: list[dict] = []


def _ensure_index():
    """Build BM25 index lazily từ standardized markdown files."""
    global _bm25, _corpus
    if _bm25 is not None:
        return

    docs = load_documents()
    _corpus = chunk_documents(docs)

    tokenized = [doc["content"].lower().split() for doc in _corpus]
    _bm25 = BM25Okapi(tokenized)


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    global _bm25, _corpus
    _corpus = corpus
    tokenized = [doc["content"].lower().split() for doc in _corpus]
    _bm25 = BM25Okapi(tokenized)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    _ensure_index()

    tokenized_query = query.lower().split()
    scores = _bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": _corpus[idx]["content"],
                "score": round(float(scores[idx]), 4),
                "metadata": _corpus[idx]["metadata"],
            })
    return results


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] [{r['metadata'].get('source', '?')}] {r['content'][:100]}...")
