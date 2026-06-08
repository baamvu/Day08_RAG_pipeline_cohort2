"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search
    2. Merge kết quả bằng RRF (Reciprocal Rank Fusion)
    3. Rerank bằng Cross-Encoder
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank, rerank_rrf
from src.task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank (Cross-Encoder) → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Semantic + Lexical search
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # Step 2: Merge bằng RRF
    merged = []
    if dense_results or sparse_results:
        non_empty = [r for r in [dense_results, sparse_results] if r]
        merged = rerank_rrf(non_empty, top_k=top_k * 2)

    for item in merged:
        item["source"] = "hybrid"

    # Step 3: Rerank bằng Cross-Encoder
    final_results = []
    if merged:
        if use_reranking:
            final_results = rerank(query, merged, top_k=top_k)
        else:
            final_results = merged[:top_k]
        # Ensure source tag after rerank
        for item in final_results:
            if "source" not in item:
                item["source"] = "hybrid"

    # Step 4: Check threshold → fallback PageIndex
    best_score = final_results[0]["score"] if final_results else 0
    if not final_results or best_score < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            src = r.get("source", "?")
            print(f"  {i}. [{r['score']:.3f}] [{src}] {r['content'][:80]}...")
