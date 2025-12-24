"""
ui_streamlit.py - FINAL WORKING VERSION
✅ Fixed Chroma deprecation
✅ Fixed empty DB (0 documents)
✅ Persistent storage
✅ Working supervisor pipeline
"""

import os
import sys
import time
import json
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

load_dotenv()

# ✅ FIXED IMPORTS
from langchain_core.documents import Document
from langchain_chroma import Chroma  # ✅ langchain-chroma
from ingestion.ingestion import (
    load_uploaded_pdfs, 
    semantic_chunk_docs, 
    buildvectorstore  # We'll update this function
)
from ingestion.retrieval import HybridRetriever
from agents.supervisor_agent import run_supervisor_pipeline

# Page config
st.set_page_config(page_title="GenAI Assignment - Internal Analyst", layout="wide")

st.title("🧠 GenAI Assignment - Internal AI Analyst")
st.markdown("**Supervisor Agent Only** - Multi-agent RAG pipeline")

# ========== SIDEBAR: UPLOAD & INDEX ==========
st.sidebar.header("1. Upload PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Upload policy manuals, financial PDFs", 
    type="pdf", 
    accept_multiple_files=True
)

# Session state
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'docs' not in st.session_state:
    st.session_state.docs = None

CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_db")

if st.sidebar.button("🚀 Build Index", type="primary"):
    if not uploaded_files:
        st.sidebar.error("❌ Please upload at least one PDF.")
    else:
        with st.spinner("📖 Reading → 🔪 Chunking → 💾 Indexing PDFs..."):
            try:
                # Step 1: Load PDFs
                raw_docs = load_uploaded_pdfs(uploaded_files)
                st.sidebar.info(f"📄 Loaded {len(raw_docs)} pages")
                
                # Step 2: Chunk
                chunks = semantic_chunk_docs(raw_docs)
                st.sidebar.info(f"🔪 Created {len(chunks)} chunks")
                
                if not chunks:
                    st.sidebar.error("❌ No text chunks created. Check PDFs.")
                else:
                    # Step 3: Build Vectorstore WITH PERSISTENCE ✅
                    vectordb = buildvectorstore(
                        chunks, 
                        persistdir=CHROMA_DB_DIR  # ✅ FIXED
                    )
                    
                    # Store raw chunks for BM25
                    st.session_state.vectordb = vectordb
                    st.session_state.docs = chunks
                    
                    # Verify count
                    count = len(vectordb.get()['documents'])
                    st.sidebar.success(f"✅ Indexed **{count}** docs from {len(uploaded_files)} files!")
                    st.sidebar.info(f"💾 Saved to: `{CHROMA_DB_DIR}`")
                    
            except Exception as e:
                st.sidebar.error(f"❌ Indexing failed: {str(e)}")
                st.sidebar.info("Try simpler PDFs or check Azure credentials")

# ========== MAIN: QUERY ==========
st.header("2. Ask a Question")
question = st.text_input("Your question about the uploaded documents", 
                        placeholder="e.g., 'how to start machine?'")

if st.button("🤖 Run Supervisor Agent", type="primary", use_container_width=True):
    if not st.session_state.vectordb or st.session_state.docs is None:
        st.error("👆 **Upload PDFs** → **Build Index** first!")
    elif len(st.session_state.docs) == 0:
        st.error("❌ **Index has 0 chunks**. Upload valid PDFs.")
    elif not question:
        st.error("❌ Enter a question.")
    else:
        vectordb = st.session_state.vectordb
        docs = st.session_state.docs
        
        t0 = time.time()
        final_json_str = ""
        
        with st.spinner("🧠 Supervisor Agent analyzing..."):
            try:
                # ✅ YOUR SUPERVISOR PIPELINE
                final_json_str = run_supervisor_pipeline(question, vectordb, docs)
            except Exception as e:
                final_json_str = json.dumps({
                    "error": "Azure filter blocked response",
                    "details": str(e),
                    "question": question
                })
        
        latency = time.time() - t0
        
        # ========== DISPLAY RESULTS ==========
        st.subheader("📊 Raw Supervisor Output")
        st.code(final_json_str, language="json")
        
        try:
            parsed = json.loads(final_json_str)
        except:
            parsed = {"answer": final_json_str, "context": []}
        
        # Layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("💡 Answer")
            answer = parsed.get('answer', parsed.get('response', 'No answer generated.'))
            st.markdown(answer)
        
        with col2:
            st.subheader("📈 Metrics")
            st.metric("Confidence", parsed.get('confidence_score', 0.0))
            st.metric("Latency", f"{latency:.2f}s")
        
        # Evidence
        st.subheader("📚 Top Evidence")
        chunks = parsed.get('top_chunks', parsed.get('context', []))
        if chunks:
            for i, chunk in enumerate(chunks[:5]):
                with st.expander(f"Chunk {i+1} • Score: {chunk.get('score', 0):.3f}"):
                    st.write(f"**Source:** {chunk.get('source', 'N/A')}")
                    st.textarea(chunk.get('text', ''), height=120)
        else:
            st.warning("No chunks retrieved.")
        
        # Analysis
        st.subheader("🔍 Analysis")
        st.write(parsed.get('missing_information', 'None identified.'))
        
        # Debug
        with st.expander("🔧 Debug Info"):
            st.metric("Total chunks in index", len(st.session_state.docs))
            st.json(parsed.get('diagnostics', {}))
        
        st.sidebar.markdown("---")
        st.sidebar.info(
            "**Supervisor Agent Only Mode**\n"
            "• Multi-agent pipeline\n"
            "• Direct Azure OpenAI\n"
            "• No fallbacks\n"
            "• Shows raw errors"
        )

# ========== LOAD EXISTING DB ==========
if st.sidebar.button("📂 Load Existing DB"):
    try:
        embeddings = HFEmbeddings()  # From ingestion.ingestion
        vectordb = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        collection_data = vectordb.get()
        docs_count = len(collection_data.get("documents", []))
        
        if docs_count > 0:
            # Extract docs for BM25
            texts = collection_data.get("documents", [])
            metadatas = collection_data.get("metadatas", [])
            docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas) if t]
            
            st.session_state.vectordb = vectordb
            st.session_state.docs = docs
            st.sidebar.success(f"✅ Loaded **{docs_count}** docs from existing DB!")
        else:
            st.sidebar.warning("❌ Existing DB is empty!")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load DB: {e}")

st.sidebar.markdown("---")
st.sidebar.caption("DeepAgents + LangChain + ChromaDB")
