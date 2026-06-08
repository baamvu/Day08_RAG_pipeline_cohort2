"""
Task 10 — Generation Có Citation.

Sắp xếp chunks tránh "lost in the middle", inject vào prompt,
và yêu cầu LLM trả lời có citation.

Lựa chọn:
    - top_k=5: đủ evidence mà không quá dài gây lost in the middle
    - top_p=0.9: đủ diverse nhưng không quá random
    - temperature=0.3: RAG cần factual, ít sáng tạo
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION
# =============================================================================

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: [1, 3, 5, 4, 2] — best first, worst in middle, second-best last.

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    reordered = []
    # Take odd-indexed items (0, 2, 4, ...) — important ones first
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])
    # Take even-indexed items (1, 3, 5, ...) in reverse — less important in middle
    for i in range(len(chunks) - 1 - (len(chunks) % 2 == 0), 0, -2):
        reordered.append(chunks[i])

    return reordered


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    # Step 5: Call LLM (OpenAI-compatible: Mimo, OpenAI, etc.)
    from openai import OpenAI

    mimo_key = os.getenv("MIMO_API_KEY", "")
    mimo_base = os.getenv("MIMO_BASE_URL", "")
    mimo_model = os.getenv("MIMO_MODEL", "mimo-v2.5-pro")

    openai_key = os.getenv("OPENAI_API_KEY", "")

    if mimo_key and mimo_base:
        client = OpenAI(api_key=mimo_key, base_url=mimo_base)
        model = mimo_model
    elif openai_key and openai_key != "sk-xxx":
        client = OpenAI(api_key=openai_key)
        model = "gpt-4o-mini"
    else:
        raise ValueError("No API key set. Add MIMO_API_KEY + MIMO_BASE_URL or OPENAI_API_KEY to .env")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    answer = response.choices[0].message.content or ""

    # Step 6: Return
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
