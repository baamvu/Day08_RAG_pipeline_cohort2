"""
Task 7 — Reranking Module.

Phương pháp chọn: RRF (Reciprocal Rank Fusion)
    - Không cần API key hay model nặng
    - Gộp kết quả từ nhiều ranker (semantic + lexical) bằng công thức: RRF(d) = Σ 1/(k + rank_r(d))
    - k=60 (từ paper Cormack et al. 2009) giúp cân bằng giữa top-1 và các rank thấp hơn
    - Đặc biệt hiệu quả khi 2 ranker có cách scoring khác nhau (cosine vs BM25)

Cũng implement Cross-Encoder reranking (cần sentence-transformers) và MMR cho đa dạng.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Cross-Encoder Reranking (local, không cần API key)
# =============================================================================

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    return _cross_encoder


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank sử dụng cross-encoder model (local).
    Cross-encoder score query-document pair trực tiếp → chính xác hơn bi-encoder.
    """
    if not candidates:
        return []

    model = _get_cross_encoder()
    pairs = [(query, c["content"]) for c in candidates]
    scores = model.predict(pairs)

    for c, score in zip(candidates, scores):
        c["score"] = float(score)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


# =============================================================================
# MMR (Maximal Marginal Relevance)
# =============================================================================

def _cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))
    Chọn candidates vừa relevant vừa diverse, giảm trùng lặp.
    """
    if not candidates:
        return []

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            relevance = _cosine_sim(query_embedding, candidates[idx].get("embedding", [0]))

            max_sim = 0
            for sel_idx in selected:
                sim = _cosine_sim(
                    candidates[idx].get("embedding", [0]),
                    candidates[sel_idx].get("embedding", [0]),
                )
                max_sim = max(max_sim, sim)

            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


# =============================================================================
# RRF (Reciprocal Rank Fusion)
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    RRF(d) = Σ 1 / (k + rank_r(d))
    Gộp kết quả từ nhiều ranker, mỗi ranker contribute theo reciprocal rank.
    k=60 là smoothing constant từ paper gốc (Cormack et al. 2009).
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in content_map:
                content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = round(score, 6)
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank candidates sử dụng Cross-Encoder.
    Re-score và re-order dựa trên relevance với query.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k reranked candidates, mỗi item có 'score' là rerank score.
    """
    return rerank_cross_encoder(query, candidates, top_k)


if __name__ == "__main__":
    dummy = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
        {"content": "Python programming language tutorial", "score": 0.5, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy, top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
