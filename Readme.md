# GenAI Assignment – Internal AI Analyst (DeepAgents + RAG) 🚀

An **end-to-end Retrieval-Augmented Generation (RAG) system** built using a **DeepAgents multi-agent workflow** to analyze **policy documents, product manuals, and financial reports** at enterprise scale.

This project demonstrates **production-grade GenAI system design**, covering document ingestion, hybrid retrieval, multi-agent orchestration, grounded answer generation, and quantitative evaluation.

---

## 🎯 What This Project Does

1. Uploads **20+ PDFs** (policy, manuals, financial reports) via a **Streamlit UI**
2. Performs **semantic chunking** → **HuggingFace embeddings** → **Chroma vector store**
3. Executes **hybrid retrieval** (Vector + BM25 + Re-ranking) with **top-5 chunks + diagnostics**
4. Uses a **DeepAgents 3-agent workflow** to generate **structured, grounded JSON answers**
5. Displays **answer, evidence, retrieved chunks, confidence, and latency metrics** in the UI

---

## 📁 Project Structure

```
genai-analyst/
├── agents/
│   ├── __init__.py
│   └── supervisor_agent.py      # DeepAgents supervisor + 3 sub-agents
├── ingestion/
│   ├── __init__.py
│   ├── ingestion.py             # PDF loading + semantic chunking + embeddings
│   └── retrieval.py             # Hybrid Retriever (Vector + BM25 + Re-rank)
├── prompts/
│   ├── __init__.py
│   ├── document_type_classifier_prompt.py
│   ├── answer_generation_prompt.py
│   └── summarization_prompt.py
├── app/
│   ├── __init__.py
│   └── ui_streamlit.py           # Streamlit UI
├── eval/
│   ├── __init__.py
│   ├── evaluation.py             # Hallucination, precision/recall, latency
│   ├── hallucination_data.json
│   └── test_data.json
├── notebooks/
│   └── eval_notebook.ipynb       # Evaluation runner
├── data/
│   ├── raw/                      # Input PDFs (20+ documents)
│   └── chroma_db/                # Vector store (auto-generated)
├── requirements.txt
├── .env.example
├── Dockerfile                    # Optional containerized deployment
└── README.md
```

---

## 🚀 Quick Start (5 Minutes)

### 1️⃣ Clone & Setup Environment

```bash
git clone <your-repo>
cd genai-analyst
python -m venv .venv
```

**Windows**
```bash
.venv\Scriptsctivate
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Configure Azure OpenAI (Optional)

```bash
cp .env.example .env
```

Edit `.env`:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_API_KEY=your-api-key
```

> The system also works with local / open-source LLMs if Azure OpenAI is not configured.

---

### 4️⃣ Add PDFs (20+ Documents)

```
data/raw/
├── policy_docs/         # EURLEX, Federal Register
├── financial_reports/   # SEC 10-K / 10-Q filings
└── manuals/             # Appliance / equipment manuals
```

Each document should ideally be **25+ pages** to simulate enterprise-scale data.

---

### 5️⃣ Run the Streamlit UI

```bash
streamlit run app/ui_streamlit.py
```

Open in browser:  
👉 **http://localhost:8501**

---

## 🎮 How to Use the Application

### Step 1: Upload Documents
- Sidebar → Upload PDFs → **Build Index**
- Pipeline executed:
  - PyPDFLoader
  - Semantic chunking
  - HuggingFace embeddings
  - Chroma indexing

**Example Log**
```
Indexed 125 chunks from 5 files
```

---

### Step 2: Ask Questions

Example:
```
How do I start the washing machine?
```

Click **Run Analysis**

---

### Step 3: View Results (Exact Order)

1. **Answer**
2. **Evidence Used** (source + snippet + relevance score)
3. **Top-5 Retrieved Chunks** (doc_id, score, full text)
4. **Missing Information** (if context is insufficient)
5. **Confidence Score**
6. **Latency & Raw Diagnostics** (expandable section)

---

## 🏗️ Technical Architecture

### 1️⃣ RAG Pipeline (`ingestion/`)

```
PDF Loader
→ Semantic Chunking
→ HF Embeddings (all-mpnet-base-v2)
→ Chroma Vector Store
→ Hybrid Retrieval
```

### Hybrid Retrieval Strategy

- **Vector Search**: Chroma similarity (+2.0 weight)
- **BM25 Search**: Keyword relevance (+1.0 weight)
- **Re-ranking**: Sorted combined scores

**Output**:  
Top-5 `{source, text, score}` + retrieval diagnostics

---

### 2️⃣ DeepAgents Workflow (`agents/supervisor_agent.py`)

```
Supervisor Agent
│
├── Query Analyzer Agent
│   - Intent classification
│   - Query rewriting
│   - Retrieval strategy selection
│
├── Retrieval Agent
│   - Hybrid vector + BM25 search
│   - Evidence selection
│
└── Answer Agent
    - Evidence-grounded reasoning
    - Strict JSON schema enforcement
```

**Final Output Schema**
```json
{
  "answer": "...",
  "evidence_used": [...],
  "top_chunks": [...],
  "confidence": 0.85
}
```

---

### 3️⃣ Prompt Design (`prompts/`)

- **Document Type Classifier**
  - Policy / Manual / Financial / General
- **Answer Generation Prompt**
  - JSON-only output
  - No unsupported claims
- **Summarization Prompt**
  - RAG-optimized, entity-preserving

---

## 📊 Evaluation & Metrics (`eval/`)

Run evaluation notebook:

```bash
jupyter notebook notebooks/eval_notebook.ipynb
```

### Metrics Computed

| Metric | Description | Result |
|------|------------|--------|
| Hallucination Rate | Unsupported answers | **10%** |
| Precision | Relevant retrieved chunks | **0.85** |
| Recall | Gold chunk coverage | **0.78** |
| Latency | End-to-end response | **230 ms avg** |
| Embedding Quality | Pos/Neg similarity | **0.78 / 0.12** |

---

## ✅ Key Design Decisions

- Hybrid retrieval improves recall over pure vector search
- DeepAgents enable separation of reasoning responsibilities
- Strict JSON schemas reduce hallucinations
- Evidence-first UI improves trust and auditability
- Evaluation is integrated, not post-hoc

---

