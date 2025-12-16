# evaluation/eval.py
import os
import sys
import json
import time
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

# Ensure project root on path
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
from agents.supervisor_agent import run_supervisor_pipeline

import numpy as np

# ---------- Helpers ----------

def cosine_sim(a, b) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def load_test_questions() -> List[Dict[str, Any]]:
    """
    Replace this with loading from a JSON/CSV of labeled Q-A pairs.
    Minimal example structure:
    [
      {
        "question": "...",
        "expected_answer_substring": "...",
        "expected_sources": ["doc1.pdf", "doc2.pdf"]
      },
      ...
    ]
    """
    return [
        {
            "question": "What is the main policy for employee leave?",
            "expected_answer_substring": "paid annual leave",
            "expected_sources": ["policy_manual.pdf"],
        },
        {
            "question": "Summarize the 2023 revenue performance.",
            "expected_answer_substring": "2023 revenue",
            "expected_sources": ["financial_report_2023.pdf"],
        },
    ]

def build_index_for_eval(pdf_folder: str):
    """
    Simple ingestion for offline eval: load all PDFs from a folder.
    """
    from pathlib import Path
    pdf_paths = list(Path(pdf_folder).glob("*.pdf"))
    uploaded_files = []
    # Emulate Streamlit UploadedFile using binary reads
    for p in pdf_paths:
        # Wrap path in a simple object with .read() for compatibility
        class _F:
            def __init__(self, path): self._p = path
            def read(self): return open(self._p, "rb").read()
        uploaded_files.append(_F(str(p)))

    raw_docs = load_uploaded_pdfs(uploaded_files)
    chunks = semantic_chunk_docs(raw_docs)
    enriched = enrich_chunks_with_metadata(chunks)
    vectordb = build_vectorstore(enriched, persist_dir=None)
    return vectordb, enriched

# ---------- Evaluation metrics ----------

def evaluate_latency(question: str, vectordb, docs) -> float:
    t0 = time.perf_counter()
    _ = run_supervisor_pipeline(question, vectordb, docs)
    t1 = time.perf_counter()
    return t1 - t0

def evaluate_single_case(
    case: Dict[str, Any],
    vectordb,
    docs,
) -> Dict[str, Any]:
    """
    Returns metrics for one question:
    - hallucination_flag
    - retrieval_precision
    - retrieval_recall
    - latency
    - avg_query_cosine
    """
    question = case["question"]
    expected_answer_substring = case["expected_answer_substring"].lower()
    expected_sources = set(case["expected_sources"])

    # 1) Run full DeepAgents pipeline
    latency = evaluate_latency(question, vectordb, docs)
    answer_json_str = run_supervisor_pipeline(question, vectordb, docs)

    try:
        parsed = json.loads(answer_json_str)
    except Exception:
        parsed = {}
    answer_text = parsed.get("answer", "").lower()
    top_chunks = parsed.get("top_chunks", [])

    # 2) Hallucination rate: simple string containment check
    hallucination_flag = expected_answer_substring not in answer_text

    # 3) Retrieval precision/recall: compare sources
    retrieved_sources = [c.get("source") for c in top_chunks if c.get("source")]
    retrieved_sources_set = set(retrieved_sources)

    true_positives = len(retrieved_sources_set & expected_sources)
    false_positives = len(retrieved_sources_set - expected_sources)
    false_negatives = len(expected_sources - retrieved_sources_set)

    precision = true_positives / (true_positives + false_positives + 1e-12)
    recall = true_positives / (true_positives + false_negatives + 1e-12)

    # 4) Embedding effectiveness: cosine sim between query embedding and chunk embeddings
    # Here we approximate by calling vectordb._collection.get; adjust to your API.
    # If not easily accessible, you can skip or customize.
    query_embedding = vectordb._embedding_function.embed_query(question)
    cosines = []
    for c in top_chunks:
        doc = c.get("doc_obj")
        if doc is None:
            continue
        emb = vectordb._embedding_function.embed_query(doc.page_content[:512])
        cosines.append(cosine_sim(query_embedding, emb))
    avg_cosine = float(np.mean(cosines)) if cosines else 0.0

    return {
        "question": question,
        "hallucination_flag": hallucination_flag,
        "precision": precision,
        "recall": recall,
        "latency_sec": latency,
        "avg_query_cosine": avg_cosine,
    }

def evaluate_dataset(pdf_folder: str):
    print(f"Building index from PDFs in: {pdf_folder}")
    vectordb, docs = build_index_for_eval(pdf_folder)

    cases = load_test_questions()
    results = []
    for case in cases:
        print(f"Evaluating question: {case['question']}")
        res = evaluate_single_case(case, vectordb, docs)
        results.append(res)
        print(" ->", res)

    # Aggregate metrics
    hallucination_rate = sum(r["hallucination_flag"] for r in results) / len(results)
    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_latency = sum(r["latency_sec"] for r in results) / len(results)
    avg_cosine = sum(r["avg_query_cosine"] for r in results) / len(results)

    print("\n==== Aggregate Evaluation ====")
    print(f"Hallucination rate: {hallucination_rate:.2f}")
    print(f"Avg retrieval precision: {avg_precision:.2f}")
    print(f"Avg retrieval recall: {avg_recall:.2f}")
    print(f"Avg latency (s): {avg_latency:.2f}")
    print(f"Avg query-chunk cosine similarity: {avg_cosine:.2f}")

    # Optionally, write to JSON file
    out = {
        "per_case": results,
        "aggregate": {
            "hallucination_rate": hallucination_rate,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_latency_sec": avg_latency,
            "avg_query_cosine": avg_cosine,
        },
    }
    os.makedirs("evaluation_outputs", exist_ok=True)
    with open("evaluation_outputs/results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSaved detailed results to evaluation_outputs/results.json")


if __name__ == "__main__":
    # Point this to a folder containing your 20+ PDFs
    PDF_FOLDER = os.getenv("EVAL_PDF_FOLDER", "data/eval_pdfs")
    evaluate_dataset(PDF_FOLDER)
