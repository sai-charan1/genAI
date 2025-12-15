# notebooks/eval_notebook.ipynb or eval/run_eval.py
import json
from ingestion.ingestion import load_uploaded_pdfs, semantic_chunk_docs, build_vectorstore
from eval.evaluation import compute_hallucination_rate, retrieval_precision_recall, measure_latency

# Load your indexed data (same as UI)
# vectordb, docs = ... (from your session_state or saved index)

# 1. Hallucination
with open("eval/hallucination_data.json") as f:
    hallucination_spec = json.load(f)
hallucination_metrics = compute_hallucination_rate(hallucination_spec, vectordb, docs)
print("Hallucination:", hallucination_metrics)

# 2. Precision/Recall  
with open("eval/test_data.json") as f:
    qa_pairs = json.load(f)
retrieval_metrics = retrieval_precision_recall(qa_pairs, vectordb, docs)
print("Retrieval:", retrieval_metrics)

# 3. Latency
questions = ["How to start?", "Safety checks?", "Warranty?"]
latency_metrics = measure_latency(questions, vectordb, docs)
print("Latency:", latency_metrics)
