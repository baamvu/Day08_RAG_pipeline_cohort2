"""
Group Project — RAG Chatbot with Supervisor-Workers Multi-Agent Pattern

Architecture:
    User Query → Supervisor Agent
                    │
                    ├→ Worker 1: Retriever (semantic + lexical → RRF merge)
                    │
                    ├→ Worker 2: Reranker (cross-encoder reranking)
                    │
                    ├→ [If score < threshold] PageIndex fallback
                    │
                    └→ Worker 3: Generator (LLM call with context)

Chạy:
    streamlit run group_project/app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.multi_agent.supervisor import Supervisor

st.set_page_config(page_title="DrugLaw Chatbot", page_icon="🤖", layout="wide")

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider("Top K chunks", 1, 20, 5)
    score_threshold = st.slider("Score Threshold", 0.0, 1.0, 0.3, 0.05)
    use_reranking = st.checkbox("Cross-Encoder Reranking", value=True)

    st.divider()

    mimo_key = os.getenv("MIMO_API_KEY", "")
    mimo_url = os.getenv("MIMO_BASE_URL", "")
    if mimo_key and mimo_url:
        st.success("Mimo API: Connected")
    else:
        st.error("Mimo API: Not configured")

    st.divider()

    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.header("📊 Index Stats")
    try:
        from src.task4_chunking_indexing import get_chroma_collection
        col = get_chroma_collection()
        st.metric("Chunks Indexed", col.count())
    except Exception:
        st.warning("Chưa index data")

    st.divider()
    st.header("🤖 Multi-Agent Architecture")
    st.markdown("""
    **Supervisor** orchestrates:
    1. 🔍 **RetrieverWorker** — semantic + lexical → RRF
    2. 🔄 **RerankerWorker** — cross-encoder rerank
    3. 📑 **PageIndex** — fallback if score < threshold
    4. ✍️ **GeneratorWorker** — LLM generation
    """)

# =============================================================================
# SESSION STATE
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "supervisor" not in st.session_state:
    st.session_state.supervisor = Supervisor(
        top_k=top_k,
        score_threshold=score_threshold,
        use_reranking=use_reranking,
    )

# Update supervisor settings if changed
supervisor = st.session_state.supervisor
supervisor.top_k = top_k
supervisor.score_threshold = score_threshold
supervisor.use_reranking = use_reranking

# =============================================================================
# HEADER
# =============================================================================
st.title("🤖 DrugLaw Chatbot")
st.caption("Supervisor-Workers Multi-Agent RAG | Hybrid Search + Reranking + Generation")

# =============================================================================
# DISPLAY CHAT HISTORY
# =============================================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])} chunks)"):
                for i, src in enumerate(msg["sources"], 1):
                    s = src.get("metadata", {}).get("source", "?")
                    t = src.get("metadata", {}).get("type", "?")
                    score = src.get("score", 0)
                    emoji = "📜" if t == "legal" else "📰"
                    st.markdown(f"**{emoji} [{i}] {s}** (score: {score:.4f})")
                    st.caption(src["content"][:200] + "...")
        if msg.get("trace"):
            with st.expander("🔍 Agent Execution Trace"):
                for step in msg["trace"]:
                    st.json(step)

# =============================================================================
# CHAT INPUT
# =============================================================================
query = st.chat_input("Hỏi về pháp luật ma tuý, nghệ sĩ liên quan...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Supervisor đang điều phối workers..."):
            result = supervisor.run(query)

        st.markdown(result.answer)

        # Show sources
        if result.sources:
            with st.expander(f"📚 Sources ({len(result.sources)} chunks)"):
                for i, src in enumerate(result.sources, 1):
                    s = src.get("metadata", {}).get("source", "?")
                    t = src.get("metadata", {}).get("type", "?")
                    score = src.get("score", 0)
                    emoji = "📜" if t == "legal" else "📰"
                    st.markdown(f"**{emoji} [{i}] {s}** (score: {score:.4f})")
                    st.caption(src["content"][:200] + "...")

        # Show execution trace
        if result.worker_trace:
            with st.expander("🔍 Agent Execution Trace"):
                for step in result.worker_trace:
                    st.json(step)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result.answer,
            "sources": result.sources,
            "trace": result.worker_trace,
        })
