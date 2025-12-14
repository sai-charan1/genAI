# ingestion/retrieval.py

from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
import numpy as np
from dotenv import load_dotenv
load_dotenv()


def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class HybridRetriever:
    def __init__(self, vectordb: Chroma, docs):
        """
        vectordb : Chroma instance
        docs : list of langchain Documents (the same ones created during ingestion)
        """
        self.vectordb = vectordb
        self.docs = docs or []

        tokenized = [(d.page_content or "").split() for d in self.docs]

        if len(tokenized) == 0:
            self.bm25 = None
            self.doc_embeddings = None
            return

        tokenized = [t if len(t) > 0 else [""] for t in tokenized]
        self.bm25 = BM25Okapi(tokenized)
        self.doc_embeddings = None

    def retrieve(self, query, top_k=5):
        """
        Returns (output_list, diagnostics)

        output_list: list of dicts {source, text, doc_obj, score}
        diagnostics: dict with vector_count, bm25_top_indices, combined_scores
        """
        if not self.docs or self.bm25 is None:
            return [], {
                "vector_count": 0,
                "bm25_top_indices": [],
                "combined_scores": {},
                "note": "No documents available in retriever.",
            }

        try:
            vector_results = self.vectordb.similarity_search(query, k=top_k)
        except TypeError:
            vector_results = self.vectordb.similarity_search(query, k=top_k)

        tokenized_query = (query or "").split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        if len(bm25_scores) == 0:
            bm25_top_idx = np.array([], dtype=int)
            bm25_results = []
        else:
            bm25_top_idx = np.argsort(bm25_scores)[::-1][:top_k]
            bm25_results = [self.docs[i] for i in bm25_top_idx if i < len(self.docs)]

        combined = {}
        for i, d in enumerate(vector_results):
            text = getattr(d, "page_content", "") if hasattr(d, "page_content") else str(d)
            src = d.metadata.get("source") if hasattr(d, "metadata") else f"doc_{i}"
            key = f"{src}::{hash(text) % (10**8)}"
            combined.setdefault(key, {"doc": d, "score": 0.0})
            combined[key]["score"] += 2.0

        for i, d in enumerate(bm25_results):
            text = getattr(d, "page_content", "") if hasattr(d, "page_content") else str(d)
            src = d.metadata.get("source") if hasattr(d, "metadata") else f"doc_bm_{i}"
            key = f"{src}::{hash(text) % (10**8)}"
            combined.setdefault(key, {"doc": d, "score": 0.0})
            combined[key]["score"] += 1.0

        ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)

        output = []
        for item in ranked[:top_k]:
            doc = item["doc"]
            score = item["score"]
            text = getattr(doc, "page_content", str(doc))
            source = getattr(doc, "metadata", {}).get("source", None) if hasattr(doc, "metadata") else None
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
            "combined_scores": {k: v["score"] for k, v in combined.items()},
        }

        return output, diagnostics
