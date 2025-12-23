import os
import sys
import time
import json
import streamlit as st

# Add project root to path
CURRENTDIR = os.path.dirname(__file__)
PROJECTROOT = os.path.abspath(os.path.join(CURRENTDIR, '..'))
if PROJECTROOT not in sys.path:
    sys.path.append(PROJECTROOT)

# ✅ SAFE IMPORTS
from ingestion.ingestion import (
    load_uploaded_pdfs,
    semantic_chunk_docs,
    enrich_chunks_with_metadata,
    build_vectorstore
)
from ingestion.retrieval import HybridRetriever

st.set_page_config(page_title="GenAI Assignment - Internal Analyst", layout="wide")
st.title("🔍 GenAI Assignment - Internal AI Analyst (Supervisor Agent Only)")

# ========== Sidebar: Upload PDFs & Build Index ==========
st.sidebar.header("1. Upload PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Upload policy manuals / financial PDFs",
    type=['pdf'],
    accept_multiple_files=True
)

if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'docs' not in st.session_state:
    st.session_state.docs = None

if st.sidebar.button("Build Index"):
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF.")
    else:
        with st.spinner("Reading, chunking, and indexing PDFs..."):
            raw_docs = load_uploaded_pdfs(uploaded_files)
            chunks = semantic_chunk_docs(raw_docs)
            if not chunks:
                st.sidebar.error("No text chunks could be created from the uploaded PDFs.")
            else:
                enriched = enrich_chunks_with_metadata(chunks)
                vectordb = build_vectorstore(enriched)
                st.session_state.vectordb = vectordb
                st.session_state.docs = enriched
                st.sidebar.success(f"✅ Indexed {len(enriched)} chunks from {len(uploaded_files)} files.")

# ========== Main UI - SUPERVISOR ONLY ==========
st.header("2. Ask a Question")
question = st.text_input("Your question about the uploaded documents:")

if st.button("🚀 Run Supervisor Agent", type="primary"):
    if not st.session_state.vectordb or st.session_state.docs is None:
        st.error("❌ Please upload PDFs and click 'Build Index' first.")
    elif len(st.session_state.docs) == 0:
        st.error("❌ Index has 0 chunks. Check your PDFs.")
    elif not question:
        st.error("❌ Please enter a question.")
    else:
        vectordb = st.session_state.vectordb
        docs = st.session_state.docs
        t0 = time.time()
        final_json_str = ""

        with st.spinner("🔍 Supervisor Agent analyzing..."):
            try:
                # ALWAYS invoke supervisor agent - NO FALLBACKS
                from agents.supervisor_agent import run_supervisor_pipeline
                final_json_str = run_supervisor_pipeline(question, vectordb, docs)
            except Exception as e:
                # Azure filter error - show exactly what happened
                final_json_str = json.dumps({
                    "error": "Azure filter blocked response",
                    "details": str(e)[:200],
                    "question": question
                })

        latency = time.time() - t0

        # ========== Results Display ==========
        st.subheader("📊 Raw Supervisor Output")
        st.code(final_json_str, language="json")

        try:
            parsed = json.loads(final_json_str)
        except:
            parsed = {"answer": final_json_str, "context": []}

        # Clean layout
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("💡 Answer")
            answer = parsed.get('answer', parsed.get('response', 'No answer generated.'))
            st.markdown(answer)
        with col2:
            st.subheader("📊 Metrics")
            st.metric("Confidence", parsed.get('confidencescore', 0.0))
            st.metric("Latency", f"{latency:.2f}s")

        # Evidence & Chunks
        st.subheader("📄 Top Evidence")
        chunks = parsed.get('topchunks', parsed.get('context', []))
        if chunks:
            for i, chunk in enumerate(chunks[:5]):
                with st.expander(f"Chunk {i+1} (Score: {chunk.get('score', 0):.3f})"):
                    st.write(f"**Source:** `{chunk.get('source', 'N/A')}`")
                    st.text_area("Content", chunk.get('text', ''), height=120)

        st.subheader("❓ Analysis")
        st.write(parsed.get('missinginformation', 'None identified.'))

        with st.expander("🔧 Debug"):
            st.metric("Total chunks in index", len(docs))
            st.json({"diagnostics": parsed.get('diagnostics', {})})

st.sidebar.markdown("---")
st.sidebar.info("""
**Supervisor Agent Only Mode**
- Always uses multi-agent pipeline
- Direct Azure OpenAI calls
- No fallback pipelines
- Shows raw Azure filter errors if blocked
""")
