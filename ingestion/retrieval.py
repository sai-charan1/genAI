# ingestion/retrieval.py - FINAL 768-DIM HYBRID RETRIEVER

from typing import List, Dict, Any, Literal, Tuple
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
import numpy as np
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def cosine_sim(a, b) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# ✅ MATCH YOUR CHROMADB (768 dims)
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"  # 768 dimensions ✅
)

class HybridRetriever:
    """
    Supports 3 retrieval strategies:
    - "vector": pure vector similarity
    - "bm25": pure BM25
    - "hybrid": combined + simple re-ranking
    """
    
    def __init__(self, vectordb: Chroma, docs):
        self.vectordb = vectordb
        self.docs = docs or []
        
        # ✅ PROVEN EMBEDDING FIX
        try:
            self.vectordb._client.set_embedding_function(EMBEDDINGS)
            print("[DEBUG] ✅ 768-dim embeddings set via client")
        except Exception as e:
            print(f"[DEBUG] ⚠️ Embeddings setup: {e}")
        
        # BM25 setup (unchanged)
        tokenized = [(d.page_content or "").split() for d in self.docs]
        if len(tokenized) == 0:
            self.bm25 = None
            return
        tokenized = [t if len(t) > 0 else [""] for t in tokenized]
        self.bm25 = BM25Okapi(tokenized)

    def _vector_search(self, query: str, top_k: int):
        return self.vectordb.similarity_search(query, k=top_k)

    def _bm25_search(self, query: str, top_k: int):
        tokenized_query = (query or "").split()
        scores = self.bm25.get_scores(tokenized_query)
        if len(scores) == 0:
            return [], np.array([], dtype=int), scores
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.docs[i] for i in top_idx if i < len(self.docs)], top_idx, scores

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        strategy: Literal["vector", "bm25", "hybrid"] = "hybrid",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not self.docs or self.bm25 is None:
            return [], {
                "vector_count": 0,
                "bm25_top_indices": [],
                "combined_scores": {},
                "note": "No documents available in retriever.",
            }

        combined_scores = {}
        vector_results = []
        bm25_top_idx = np.array([], dtype=int)

        if strategy in ("vector", "hybrid"):
            vector_results = self._vector_search(query, top_k)

        if strategy in ("bm25", "hybrid"):
            bm25_results, bm25_top_idx, bm25_scores = self._bm25_search(query, top_k)
        else:
            bm25_results, bm25_scores = [], []

        if strategy == "vector":
            candidates = vector_results
        elif strategy == "bm25":
            candidates = bm25_results
        else:
            combined = {}
            for i, d in enumerate(vector_results):
                text = getattr(d, "page_content", "") if hasattr(d, "page_content") else str(d)
                src = d.metadata.get("source") if hasattr(d, "metadata") else f"doc_v_{i}"
                key = f"{src}::{hash(text) % (10**8)}"
                combined.setdefault(key, {"doc": d, "score": 0.0})
                combined[key]["score"] += 2.0
            for i, d in enumerate(bm25_results):
                text = getattr(d, "page_content", "") if hasattr(d, "page_content") else str(d)
                src = d.metadata.get("source") if hasattr(d, "metadata") else f"doc_bm_{i}"
                key = f"{src}::{hash(text) % (10**8)}"
                combined.setdefault(key, {"doc": d, "score": 0.0})
                combined[key]["score"] += 1.0
            combined_scores = {k: v["score"] for k, v in combined.items()}
            ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
            candidates = [item["doc"] for item in ranked[:top_k]]

        output = []
        for doc in candidates[:top_k]:
            text = getattr(doc, "page_content", str(doc))
            source = getattr(doc, "metadata", {}).get("source", None) if hasattr(doc, "metadata") else None
            score = 1.0
            output.append(
                {
                    "source": source,
                    "text": text,
                    "doc_obj": doc,
                    "score": float(score),
                }
            )

        diagnostics = {
            "vector_count": len(vector_results),
            "bm25_top_indices": bm25_top_idx.tolist() if hasattr(bm25_top_idx, "tolist") else [],
            "combined_scores": combined_scores,
            "strategy": strategy,
        }
        return output, diagnostics
