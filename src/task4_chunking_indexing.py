"""
Task 4 — Chunking & Indexing vào Vector Store.

Lựa chọn:
    - Chunking: 3 strategies
        1. RecursiveCharacterTextSplitter (mặc định, an toàn)
        2. MarkdownHeaderTextSplitter (tốt cho file có heading rõ)
        3. SemanticChunker (nâng cao, dùng embedding để tách)
        -> Chọn RecursiveCharacterTextSplitter làm mặc định vì ổn định nhất.
    - Embedding: sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ, nhanh)
    - Vector Store: ChromaDB (local, không cần Docker; Weaviate không hỗ trợ Windows embedded)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "drug_law_docs"

# =============================================================================
# CONFIGURATION
# =============================================================================

# RecursiveCharacterTextSplitter: an toàn, phổ biến, hoạt động tốt với mọi loại text.
# 500 chars ~ 1 đoạn văn bản đủ ngữ nghĩa, không quá lớn cho embedding model.
# Overlap 50 (10%) giúp giữ context khi chunk bị cắt ở ranh giới câu.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# all-MiniLM-L12-v2: 384 dim, 12 layers (vs L6's 6 layers), ~130MB.
# Better English benchmarks (+0.85), same dimension, slightly slower but more accurate.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
EMBEDDING_DIM = 384

# ChromaDB: local vector store, không cần Docker. Weaviate embedded không hỗ trợ Windows.
VECTOR_STORE = "chromadb"


# =============================================================================
# LOAD DOCUMENTS
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str, ...}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if len(content) < 50:
            continue
        rel_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = "legal" if "legal" in str(rel_path) else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.stem,
                "type": doc_type,
                "filename": md_file.name,
                "path": str(rel_path),
            }
        })
    return documents


# =============================================================================
# CHUNKING STRATEGIES
# =============================================================================

def _chunk_recursive(documents: list[dict]) -> list[dict]:
    """Strategy 1: RecursiveCharacterTextSplitter — an toàn, mặc định."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, text in enumerate(splits):
            text = text.strip()
            if text:
                chunks.append({
                    "content": text,
                    "metadata": {**doc["metadata"], "chunk_index": i, "chunk_total": len(splits)},
                })
    return chunks


def _chunk_markdown_header(documents: list[dict]) -> list[dict]:
    """Strategy 2: MarkdownHeaderTextSplitter — tách theo heading, giữ cấu trúc."""
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    headers_to_split = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split)
    fallback = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        try:
            md_docs = md_splitter.split_text(doc["content"])
        except Exception:
            md_docs = [{"page_content": doc["content"], "metadata": {}}]

        for md_doc in md_docs:
            text = md_doc.page_content if hasattr(md_doc, "page_content") else str(md_doc)
            meta = md_doc.metadata if hasattr(md_doc, "metadata") else {}

            if len(text) <= CHUNK_SIZE:
                chunks.append({
                    "content": text.strip(),
                    "metadata": {**doc["metadata"], **meta, "chunk_index": len(chunks)},
                })
            else:
                sub_splits = fallback.split_text(text)
                for j, sub in enumerate(sub_splits):
                    sub = sub.strip()
                    if sub:
                        chunks.append({
                            "content": sub,
                            "metadata": {**doc["metadata"], **meta, "chunk_index": j},
                        })
    return chunks


def _chunk_semantic(documents: list[dict]) -> list[dict]:
    """Strategy 3: SemanticChunker — dùng embedding để tìm ranh giới tự nhiên."""
    from langchain_text_splitters import SemanticChunker
    from langchain_core.embeddings import Embeddings
    from sentence_transformers import SentenceTransformer

    class STEmbeddings(Embeddings):
        def __init__(self):
            self.model = SentenceTransformer(EMBEDDING_MODEL)

        def embed_documents(self, texts):
            return self.model.encode(texts, normalize_embeddings=True).tolist()

        def embed_query(self, text):
            return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    embedder = STEmbeddings()
    splitter = SemanticChunker(
        embeddings=embedder,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=75,
    )

    chunks = []
    for doc in documents:
        try:
            splits = splitter.split_text(doc["content"])
        except Exception:
            splits = [doc["content"]]

        for i, text in enumerate(splits):
            text = text.strip()
            if text:
                chunks.append({
                    "content": text,
                    "metadata": {**doc["metadata"], "chunk_index": i, "chunk_total": len(splits)},
                })
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    strategies = {
        "recursive": _chunk_recursive,
        "markdown_header": _chunk_markdown_header,
        "semantic": _chunk_semantic,
    }

    strategy_fn = strategies.get(CHUNKING_METHOD)
    if strategy_fn is None:
        raise ValueError(f"Unknown chunking method: {CHUNKING_METHOD}")

    return strategy_fn(documents)


# =============================================================================
# EMBEDDING
# =============================================================================

_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng all-MiniLM-L6-v2.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    model = _get_embedding_model()
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


# =============================================================================
# VECTOR STORE (ChromaDB)
# =============================================================================

def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks vào ChromaDB với pre-computed embeddings."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        ids = [f"chunk_{start + i}" for i in range(len(batch))]
        documents = [c["content"] for c in batch]
        embeddings = [c["embedding"] for c in batch]
        metadatas = [
            {
                "source": c["metadata"].get("source", ""),
                "type": c["metadata"].get("type", ""),
                "filename": c["metadata"].get("filename", ""),
                "chunk_index": c["metadata"].get("chunk_index", 0),
            }
            for c in batch
        ]
        collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    print(f"  Indexed {len(chunks)} chunks into '{COLLECTION_NAME}'")


def get_chroma_collection():
    """Get existing ChromaDB collection for use in other tasks."""
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline():
    """Chạy toàn bộ pipeline: load -> chunk -> embed -> index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents")
    for d in docs:
        print(f"     - {d['metadata']['source']} ({d['metadata']['type']}, {len(d['content'])} chars)")

    chunks = chunk_documents(docs)
    print(f"\n[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"\n[OK] Embedded {len(chunks)} chunks (dim={len(chunks[0]['embedding'])})")

    index_to_vectorstore(chunks)
    print(f"\n[DONE] Pipeline complete. DB at: {CHROMA_DIR}")


if __name__ == "__main__":
    run_pipeline()
