# ingestion/ingestion.py

import os
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from tempfile import NamedTemporaryFile

from langchain_openai import AzureChatOpenAI
from prompts.document_type_classifier_prompt import DOCUMENT_TYPE_CLASSIFIER_PROMPT
from prompts.summarization_prompt import SUMMARIZATION_PROMPT


def load_uploaded_pdfs(uploaded_files) -> List:
    """Load pages from uploaded PDF files into a list of LangChain Documents."""
    docs = []
    for f in uploaded_files:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())
        os.remove(tmp_path)
    return docs


def semantic_chunk_docs(docs, chunk_size: int = 1200, chunk_overlap: int = 150):
    """Split documents into semantic-ish chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


class HFEmbeddings:
    """Wrapper so Chroma can call .embed_documents and .embed_query using HF model."""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


# ---- LLMs for classification and summarization ----

_classifier_llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0,
    max_retries=5,
)
_summarizer_llm = _classifier_llm


def classify_chunk(text: str) -> Dict[str, Any]:
    """Call LLM with DOCUMENT_TYPE_CLASSIFIER_PROMPT and return JSON classification."""
    messages = [
        {"role": "system", "content": DOCUMENT_TYPE_CLASSIFIER_PROMPT},
        {"role": "user", "content": text},
    ]
    resp = _classifier_llm.invoke(messages)
    try:
        return json.loads(resp.content)
    except Exception:
        return {"classification": "General Info", "confidence": 0.0}


def summarize_for_rag(text: str) -> str:
    """Call LLM with SUMMARIZATION_PROMPT and return plain-text summary."""
    messages = [
        {"role": "system", "content": SUMMARIZATION_PROMPT},
        {"role": "user", "content": text},
    ]
    resp = _summarizer_llm.invoke(messages)
    return resp.content.strip()


def enrich_chunks_with_metadata(chunks):
    """
    For each chunk: classify doc type, generate RAG summary,
    and attach to metadata for filtering/analysis.
    """
    enriched = []
    for doc in chunks:
        text = doc.page_content or ""
        # You can skip classification/summarization during debugging to save tokens.
        # cls = classify_chunk(text)
        # summary = summarize_for_rag(text)
        cls = {"classification": "General Info", "confidence": 0.0}
        summary = text[:400]

        meta = dict(doc.metadata or {})
        meta["doc_type"] = cls.get("classification", "General Info")
        meta["doc_type_confidence"] = cls.get("confidence", 0.0)
        meta["rag_summary"] = summary
        doc.metadata = meta
        enriched.append(doc)
    return enriched


def buildvectorstore(chunks, persistdir: str = "data/chroma_db"):
    """Build Chroma with persistence and error handling."""
    print(f"📊 Building vectorstore with {len(chunks)} chunks...")
    
    try:
        # Skip LLM enrichment if failing
        embeddings = HFEmbeddings()
        vectordb = Chroma.from_documents(
            documents=chunks,  # Use raw chunks, skip enrichment
            embedding=embeddings,
            persist_directory=persistdir  # ✅ PERSIST!
        )
        print(f"✅ Stored {len(chunks)} docs in {persistdir}")
        return vectordb
    except Exception as e:
        print(f"❌ Vectorstore failed: {e}")
        # Fallback: empty store
        return Chroma(embedding_function=HFEmbeddings(), persist_directory=persistdir)


import json  # keep at bottom to avoid circular for classify_chunk
