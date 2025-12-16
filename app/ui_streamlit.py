# app/ui_streamlit.py

import os
import sys
import time
import json

import streamlit as st

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ingestion.ingestion import (
    load_uploaded_pdfs,
    semantic_chunk_docs,
    enrich_chunks_with_metadata,
    build_vectorstore,
)
from ingestion.retrieval import HybridRetriever
from agents.supervisor_agent import model, ANSWER_AGENT_PROMPT  # reuse same model/prompt


st.set_page_config(page_title="GenAI Assignment – Internal Analyst", layout="wide")
st.title("GenAI Assignment – Internal AI Analyst (DeepAgents + RAG)")

# ---------- Sidebar: upload PDFs & build index ----------

st.sidebar.header("1. Upload PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Upload policy / manuals / financial PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "docs" not in st.session_state:
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
                vectordb = build_vectorstore(enriched, persist_dir=None)
                st.session_state.vectordb = vectordb
                st.session_state.docs = enriched
                st.sidebar.success(
                    f"Indexed {len(enriched)} chunks from {len(uploaded_files)} files."
                )

# ---------- Main: ask a question and show answer ----------

st.header("2. Ask a Question")
question = st.text_input("Your question about the uploaded documents")

def generate_answer_with_context(question: str, top_chunks):
    """Direct, simple answer model call using the same JSON prompt."""
    payload = {
        "question": question,
        "context": [
            {
                "source": c.get("source"),
                "text": c.get("text", ""),
                "score": c.get("score", 0.0),
            }
            for c in top_chunks
        ],
    }
    messages = [
        {"role": "system", "content": ANSWER_AGENT_PROMPT},
        {"role": "user", "content": json.dumps(payload)},
    ]
    resp = model.invoke(messages)
    return resp.content

if st.button("Run Analysis"):
    if not st.session_state.vectordb or st.session_state.docs is None:
        st.error("Please upload PDFs and click 'Build Index' first. Index is empty.")
    elif len(st.session_state.docs) == 0:
        st.error("Index has 0 chunks. Check your PDFs and try again.")
    elif not question:
        st.error("Please enter a question.")
    else:
        vectordb = st.session_state.vectordb
        docs = st.session_state.docs

        t0 = time.time()
        retriever = HybridRetriever(vectordb, docs)
        top_chunks, diag = retriever.retrieve(question, top_k=5, strategy="hybrid")
        latency = time.time() - t0

        final_json_str = generate_answer_with_context(question, top_chunks)

        st.subheader("Answer (raw JSON)")
        st.code(final_json_str, language="json")

        parsed = None
        try:
            parsed = json.loads(final_json_str)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            st.subheader("Answer (parsed)")
            st.write(parsed.get("answer", ""))

            st.subheader("Evidence Used")
            ev = parsed.get("evidence_used", [])
            st.json(ev if isinstance(ev, list) else [])

            st.subheader("Top 5 Chunks (with text)")
            top_chunks_json = parsed.get("top_chunks", [])
            if not isinstance(top_chunks_json, list):
                top_chunks_json = []
            for i, c in enumerate(top_chunks_json[:5]):
                st.markdown(f"**Chunk {i+1}**")
                st.write(f"Source (doc id): {c.get('source')}")
                st.write(f"Score: {c.get('score')}")
                st.text_area(
                    label="Text",
                    value=c.get("text", ""),
                    height=150,
                    key=f"chunk_{i}",
                )

            st.subheader("Missing Information")
            st.write(parsed.get("missing_information", ""))

            st.subheader("Confidence Score")
            st.write(parsed.get("confidence_score", 0.0))

        with st.expander("Evaluation & Retrieval Diagnostics"):
            st.markdown("**Latency (seconds)**")
            st.metric("Last query latency", f"{latency:.2f}")
            st.markdown("**Raw Retrieval Diagnostics:**")
            st.json(diag)
