# GenAI Assignment – Internal AI Analyst (DeepAgents + RAG)

End-to-end RAG system with DeepAgents multi-agent workflow for analyzing policy documents, product manuals, and financial reports.

## Features

- **Hybrid Retrieval**: Sentence-Transformers + Chroma + BM25 + re-ranking
- **3-Agent DeepAgents Workflow**: Query Analyzer → Retrieval Agent → Answer Agent  
- **Streamlit UI**: Upload PDFs → Build Index → Ask Questions → Answer + Evidence + Top-5 Chunks + Metrics
- **Evaluation**: Hallucination rate, precision/recall, latency, embedding diagnostics

## Project Structure

genai-analyst/
├── agents/ # DeepAgents supervisor + 3 subagents
├── ingestion/ # PDF loading, semantic chunking, HybridRetriever
├── prompts/ # Classifier, answer generation, summarization prompts
├── app/ # ui_streamlit.py
├── eval/ # evaluation.py + notebook
├── data/ # PDFs + chroma_db
├── requirements.txt
└── README.md

text

## Quick Start

Setup
python -m venv .venv
.venv\Scripts\activate # Windows
pip install -r requirements.txt

Run UI
streamlit run app/ui_streamlit.py

text

1. Upload 20+ PDFs (25+ pages) in sidebar
2. Click "Build Index" 
3. Ask questions → See answer + evidence + top-5 chunks + metrics

## Key Components

- **RAG**: HF embeddings → Chroma → HybridRetriever (vector+BM25+re-rank)
- **Agents**: `supervisor_agent.py` → query-analyzer → retrieval-agent → answer-agent
- **Retrieval Tool**: `retrieval_tool_hybrid_top5` returns top-5 chunks + diagnostics
- **UI Output**: Answer → Evidence → Top-5 chunks (doc_id+score+text) → Latency/Diagnostics

## Evaluation

Run `notebooks/eval_notebook.ipynb` for:
- Hallucination rate (10 questions)
- Retrieval precision/recall 
- Latency per query
- Embedding cosine similarity diagnostics

## Deliverables ✓

✅ All 3 prompts (classifier/answer/summarization)  
✅ RAG pipeline (semantic chunking + hybrid retrieval)  
✅ DeepAgents 3-agent workflow + supervisor  
✅ Streamlit UI with top-5 chunks + diagnostics  
✅ Evaluation module (hallucination + metrics)  
✅ Dependencies + setup instructions

**Datasets**: EURLEX, SEC EDGAR, appliance manuals (links in assignment PDF)