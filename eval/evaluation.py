# eval/evaluation.py
import json
import time
from typing import List, Dict, Any
import numpy as np

from ingestion.retrieval import HybridRetriever
from agents.supervisor_agent import generate_answer_with_context  # your helper


def compute_hallucination_rate(
    qa_spec: List[Dict[str, Any]], vectordb, docs, top_k: int = 5
) -> Dict[str, Any]:
    """Hallucination rate over 10 questions (assignment requirement)"""
    retriever = HybridRetriever(vectordb, docs)
    hallucinated = 0
    total = 0

    for item in qa_spec:
        q = item["question"]
        allowed_sources = set(item.get("allowed_sources", []))
        required_terms = set(item.get("gold_answer", "").lower().split(","))

        # Retrieve + generate answer
        top_chunks, _ = retriever.retrieve(q, top_k=top_k)
        answer_json = generate_answer_with_context(q, top_chunks)
        
        try:
            parsed = json.loads(answer_json)
            ans_text = (parsed.get("answer", "") or "").lower()
            evidence = parsed.get("evidence_used", [])
            cited_sources = {e.get("source", "") for e in evidence}
            
            # Hallucination if: bad sources OR missing required terms
            bad_sources = bool(allowed_sources) and not cited_sources.issubset(allowed_sources)
            missing_terms = len(required_terms - set(ans_text.split())) > 0
            
            if bad_sources or missing_terms:
                hallucinated += 1
        except:
            hallucinated += 1  # JSON parse fail = hallucinated
        
        total += 1

    return {
        "hallucination_rate": float(hallucinated / total),
        "total_questions": total,
        "hallucinated": hallucinated
    }


def retrieval_precision_recall(
    qa_pairs: List[Dict[str, Any]], vectordb, docs, top_k: int = 5
) -> Dict[str, Any]:
    """Retrieval precision/recall from labeled QA pairs"""
    retriever = HybridRetriever(vectordb, docs)
    precisions, recalls = [], []

    for item in qa_pairs:
        q = item["question"]
        gold_sources = set(item["gold_sources"])
        
        top_chunks, _ = retriever.retrieve(q, top_k=top_k)
        retrieved_sources = {c.get("source", "") for c in top_chunks}
        
        tp = len(retrieved_sources & gold_sources)
        precision = tp / len(retrieved_sources) if retrieved_sources else 0
        recall = tp / len(gold_sources) if gold_sources else 0
        
        precisions.append(precision)
        recalls.append(recall)

    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "avg_precision": float(np.mean(precisions)),
        "avg_recall": float(np.mean(recalls))
    }


def measure_latency(questions: List[str], vectordb, docs) -> Dict[str, Any]:
    """Average latency over questions"""
    retriever = HybridRetriever(vectordb, docs)
    times = []
    
    for q in questions:
        t0 = time.time()
        top_chunks, _ = retriever.retrieve(q, top_k=5)
        generate_answer_with_context(q, top_chunks)
        times.append(time.time() - t0)
    
    return {
        "avg_latency": float(np.mean(times)),
        "p95_latency": float(np.percentile(times, 95)),
        "times": times
    }
