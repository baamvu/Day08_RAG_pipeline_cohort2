"""
Task 8 — PageIndex Vectorless RAG.

PageIndex dùng LLM reasoning trên document tree structure thay vì vector similarity.
Workflow: upload PDF → build tree → submit query → get retrieval results.

Cài đặt:
    pip install pageindex

API Key: https://pageindex.ai/ → Developer → API Keys
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOC_IDS_FILE = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"


def _get_client():
    from pageindex import PageIndexClient
    if not PAGEINDEX_API_KEY:
        raise ValueError("PAGEINDEX_API_KEY not set. Get it at https://pageindex.ai/")
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def upload_documents():
    """Upload PDF files từ data/landing/legal/ lên PageIndex."""
    import json

    client = _get_client()
    pdf_files = sorted(LEGAL_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/landing/legal/")
        return

    doc_ids = {}

    # Load existing doc_ids if available
    if DOC_IDS_FILE.exists():
        doc_ids = json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))

    for pdf in pdf_files:
        if pdf.name in doc_ids:
            print(f"  [SKIP] Already uploaded: {pdf.name} -> {doc_ids[pdf.name]}")
            continue

        print(f"  [UPLOAD] {pdf.name} ...")
        try:
            result = client.submit_document(str(pdf))
            doc_id = result.get("doc_id", "")
            doc_ids[pdf.name] = doc_id
            print(f"  [OK] doc_id: {doc_id}")
        except Exception as e:
            print(f"  [ERROR] {pdf.name}: {e}")

    # Save doc_ids for later use
    DOC_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    DOC_IDS_FILE.write_text(json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved {len(doc_ids)} doc_ids to {DOC_IDS_FILE}")


def _wait_for_ready(client, doc_id: str, max_wait: int = 120):
    """Wait until document is ready for retrieval."""
    for i in range(max_wait // 5):
        if client.is_retrieval_ready(doc_id):
            return True
        print(f"    Waiting for doc {doc_id[:8]}... ({(i+1)*5}s)")
        time.sleep(5)
    return False


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Fallback khi hybrid search không có kết quả phù hợp.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'
        }
    """
    import json

    if not PAGEINDEX_API_KEY:
        return []

    if not DOC_IDS_FILE.exists():
        return []

    doc_ids = json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
    if not doc_ids:
        return []

    client = _get_client()
    all_results = []

    for filename, doc_id in doc_ids.items():
        try:
            submit_result = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = submit_result.get("retrieval_id", "")
            if not retrieval_id:
                continue

            # Poll for results
            for _ in range(24):  # max 120s
                result = client.get_retrieval(retrieval_id)
                status = result.get("status", "")
                if status == "completed":
                    break
                if status == "failed":
                    break
                time.sleep(5)
            else:
                continue

            if status != "completed":
                continue

            # Extract results
            sections = result.get("sections", result.get("results", []))
            if isinstance(sections, list):
                for section in sections:
                    content = section.get("content", section.get("text", section.get("section_content", "")))
                    score = section.get("score", section.get("relevance_score", 0.5))
                    if content:
                        all_results.append({
                            "content": str(content)[:2000],
                            "score": float(score) if score else 0.5,
                            "metadata": {
                                "source": filename.replace(".pdf", ""),
                                "doc_id": doc_id,
                                "retrieval_id": retrieval_id,
                            },
                            "source": "pageindex",
                        })
        except Exception as e:
            print(f"  [WARN] Query failed for {filename}: {e}")
            continue

    # Sort by score and return top_k
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("[WARN] PAGEINDEX_API_KEY not set.")
        print("  1. Go to https://pageindex.ai/")
        print("  2. Sign in with GitHub")
        print("  3. Go to Developer -> API Keys -> Create key")
        print("  4. Add to .env: PAGEINDEX_API_KEY=pi_xxx")
    else:
        print("Upload documents to PageIndex...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        if results:
            for r in results:
                print(f"  [{r['score']:.3f}] [{r['source']}] {r['content'][:100]}...")
        else:
            print("  No results (docs may still be processing)")
