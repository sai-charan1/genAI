# eval/evaluation.py

import time
from typing import List, Dict, Any

import numpy as np

from agents.supervisor_agent import supervisor
from ingestion.retrieval import HybridRetriever


def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def measure_latency(questions: List[str], vectordb, docs) -> Dict[str, Any]:
    times = []
    for q in questions:
        t0 = time.time()
        _ = supervisor.invoke(
            {
                "messages": [{"role": "user", "content": q}],
                "extra": {"vectorstore": vectordb, "docs": docs},
            }
        )
        times.append(time.time() - t0)
    return {"avg_latency": float(sum(times) / len(times)), "all": times}


def retrieval_precision_recall(
    qa_pairs: List[Dict[str, Any]], vectordb, docs, top_k: int = 5
) -> Dict[str, Any]:
    """
    qa_pairs: list of {"question": str, "gold_sources": [source_ids]}
    Returns macro precision and recall based on whether the retriever returns chunks
    whose 'source' is in gold_sources.
    """
    retriever = HybridRetriever(vectordb, docs)
    precisions = []
    recalls = []

    for item in qa_pairs:
        q = item["question"]
        gold_sources = set(item["gold_sources"])
        top_chunks, _ = retriever.retrieve(q, top_k=top_k)
        retrieved_sources = [c.get("source") for c in top_chunks if c.get("source")]

        if not retrieved_sources or not gold_sources:
            continue

        retrieved_set = set(retrieved_sources)
        tp = len(retrieved_set & gold_sources)
        precision = tp / len(retrieved_set)
        recall = tp / len(gold_sources)

        precisions.append(precision)
        recalls.append(recall)

    return {
        "precision": float(sum(precisions) / len(precisions)) if precisions else 0.0,
        "recall": float(sum(recalls) / len(recalls)) if recalls else 0.0,
    }


def embedding_diagnostics(positive_pairs, negative_pairs, embed_fn) -> Dict[str, Any]:
    """
    positive_pairs / negative_pairs: list of (text1, text2)
    embed_fn: function that maps text -> embedding vector (from your HF model)
    """
    pos_scores = []
    neg_scores = []

    for a, b in positive_pairs:
        ea = embed_fn(a)
        eb = embed_fn(b)
        pos_scores.append(cosine_sim(ea, eb))

    for a, b in negative_pairs:
        ea = embed_fn(a)
        eb = embed_fn(b)
        neg_scores.append(cosine_sim(ea, eb))

    return {
        "positive_mean": float(np.mean(pos_scores)) if pos_scores else 0.0,
        "negative_mean": float(np.mean(neg_scores)) if neg_scores else 0.0,
        "positive_samples": pos_scores,
        "negative_samples": neg_scores,
    }
