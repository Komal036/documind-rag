"""
DocuMind — Streamlit Frontend
------------------------------
A clean, enterprise-grade UI for uploading documents and asking questions.

Run with:
    streamlit run frontend/app.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_TIMEOUT = 120  # seconds (model loading can be slow on first call)

# ── Page setup ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DocuMind — Enterprise RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A5F; }
    .sub-header  { font-size: 1rem; color: #6B7280; margin-bottom: 1.5rem; }
    .answer-box  { background: #1E293B; border-left: 4px solid #60A5FA;
                   padding: 1rem 1.2rem; border-radius: 6px; color: #F1F5F9; }
    .source-card { background: #F9FAFB; border: 1px solid #E5E7EB;
                   padding: 0.8rem; border-radius: 6px; margin-bottom: 0.5rem; }
    .badge-green { background: #D1FAE5; color: #065F46;
                   padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
    .badge-blue  { background: #DBEAFE; color: #1E40AF;
                   padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── API helpers ───────────────────────────────────────────────────────

def api_health() -> bool:
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def api_stats() -> dict:
    try:
        r = httpx.get(f"{API_BASE}/api/v1/stats", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_ingest(file_bytes: bytes, filename: str) -> dict:
    try:
        r = httpx.post(
            f"{API_BASE}/api/v1/ingest",
            files={"file": (filename, file_bytes)},
            timeout=API_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        return {"error": e.response.json().get("detail", str(e))}
    except Exception as e:
        return {"error": str(e)}


def api_query(question: str, use_reranker: bool, top_k: int, top_n: int) -> dict:
    try:
        payload = {
            "question": question,
            "use_reranker": use_reranker,
            "top_k": top_k,
            "top_n": top_n,
        }
        r = httpx.post(
            f"{API_BASE}/api/v1/query",
            json=payload,
            timeout=API_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        return {"error": e.response.json().get("detail", str(e))}
    except Exception as e:
        return {"error": str(e)}


def api_delete(filename: str) -> dict:
    try:
        r = httpx.delete(f"{API_BASE}/api/v1/documents/{filename}", timeout=15)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        return {"error": e.response.json().get("detail", str(e))}
    except Exception as e:
        return {"error": str(e)}


# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 DocuMind")
    st.markdown("*Enterprise RAG System*")
    st.divider()

    # API health indicator
    healthy = api_health()
    status_icon = "🟢" if healthy else "🔴"
    status_text = "API Connected" if healthy else "API Offline"
    st.markdown(f"{status_icon} **{status_text}**")

    if not healthy:
        st.error("Cannot reach the API. Make sure the backend is running:\n```\nuvicorn src.main:app --reload\n```")

    st.divider()

    # Stats
    if healthy:
        stats = api_stats()
        if "error" not in stats:
            st.markdown("### 📊 Index Stats")
            st.metric("Indexed Chunks", stats.get("total_chunks", 0))
            files = stats.get("indexed_files", [])
            st.metric("Documents", len(files))
            if files:
                st.markdown("**Files:**")
                for f in files:
                    st.markdown(f"- `{f}`")
        st.divider()

    # Advanced settings
    st.markdown("### ⚙️ Query Settings")
    use_reranker = st.toggle("Cross-Encoder Re-ranking", value=True,
                              help="More accurate but slightly slower")
    top_k = st.slider("Retrieval candidates (top_k)", 5, 50, 20)
    top_n = st.slider("Chunks sent to LLM (top_n)", 1, 10, 5)

    st.divider()
    st.markdown(
        "<small>Built with LangChain · sentence-transformers · ChromaDB · FastAPI · Streamlit</small>",
        unsafe_allow_html=True,
    )


# ── Main area ─────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🧠 DocuMind</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Enterprise document Q&A — Upload documents and ask anything.</div>',
    unsafe_allow_html=True,
)

tab_query, tab_upload, tab_manage = st.tabs(["💬 Ask a Question", "📄 Upload Document", "🗂️ Manage Index"])


# ── Tab 1: Query ──────────────────────────────────────────────────────

with tab_query:
    st.markdown("### Ask a question about your documents")

    # Example questions
    with st.expander("💡 Example questions"):
        examples = [
            "What is the remote work policy?",
            "How many days can employees work from home?",
            "What is the expense reimbursement limit?",
            "Who is eligible for remote work?",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state["question_input"] = ex

    question = st.text_area(
        "Your question",
        value=st.session_state.get("question_input", ""),
        placeholder="e.g. What is the vacation policy?",
        height=80,
        key="question_area",
    )

    if st.button("🔍 Search & Answer", type="primary", disabled=not healthy or not question.strip()):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving relevant passages and generating answer…"):
                t0 = time.time()
                result = api_query(question, use_reranker, top_k, top_n)
                elapsed = time.time() - t0

            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                # Answer
                st.markdown("### 📝 Answer")
                st.markdown(result["answer"])

                # Meta
                cols = st.columns(4)
                cols[0].metric("Response time", f"{elapsed:.1f}s")
                cols[1].metric("Chunks used", result.get("chunks_used", 0))
                cols[2].markdown(
                    f'<span class="badge-blue">Model: {result["model"]}</span>',
                    unsafe_allow_html=True,
                )
                cols[3].markdown(
                    f'<span class="badge-green">Provider: {result["provider"]}</span>',
                    unsafe_allow_html=True,
                )

                # Sources
                if result.get("sources"):
                    st.markdown("### 📚 Sources")
                    for src in result["sources"]:
                        with st.expander(
                            f"[{src['citation_number']}] {src['filename']} — page {src['page']} "
                            f"(score: {src['relevance_score']:.3f})"
                        ):
                            st.markdown(f"**Preview:** {src['preview']}")
                            st.markdown(f"**Chunk ID:** `{src['chunk_id']}`")


# ── Tab 2: Upload ─────────────────────────────────────────────────────

with tab_upload:
    st.markdown("### Upload a document to index")
    st.info("Supported formats: **PDF, TXT, DOCX, MD** — Max 50 MB per file")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "docx", "md"],
        help="The document will be chunked, embedded, and stored in ChromaDB.",
    )

    if uploaded_file and st.button("📥 Ingest Document", type="primary", disabled=not healthy):
        with st.spinner(f"Processing '{uploaded_file.name}'…"):
            result = api_ingest(uploaded_file.read(), uploaded_file.name)

        if "error" in result:
            st.error(f"❌ Ingestion failed: {result['error']}")
        else:
            st.success(f"✅ {result['message']}")
            cols = st.columns(3)
            cols[0].metric("Chunks created", result["chunk_count"])
            cols[1].metric("Status", result["status"].replace("_", " ").title())
            cols[2].markdown(f"**SHA-256:** `{result['sha256'][:16]}…`")
            st.balloons()


# ── Tab 3: Manage ─────────────────────────────────────────────────────

with tab_manage:
    st.markdown("### Manage indexed documents")

    if not healthy:
        st.warning("API is not available.")
    else:
        stats = api_stats()
        files = stats.get("indexed_files", [])

        if not files:
            st.info("No documents are currently indexed. Upload one in the **Upload Document** tab.")
        else:
            st.markdown(f"**{len(files)} document(s) indexed:**")

            for fname in files:
                col1, col2 = st.columns([4, 1])
                col1.markdown(f"📄 `{fname}`")
                if col2.button("🗑️ Delete", key=f"del_{fname}"):
                    result = api_delete(fname)
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.success(result["message"])
                        st.rerun()

        st.divider()
        st.markdown("**Full stats:**")
        if "error" not in stats:
            st.json(stats)
