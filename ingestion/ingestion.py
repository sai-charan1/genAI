# ingestion/ingestion.py

import os
from typing import List

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from tempfile import NamedTemporaryFile


def load_uploaded_pdfs(uploaded_files) -> List:
    """
    Load pages from uploaded PDF files (Streamlit UploadedFile objects)
    into a list of LangChain Documents.
    """
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
    """
    Split documents into semantic-ish chunks using RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


class HFEmbeddings:
    """
    Minimal wrapper so Chroma can call .embed_documents and .embed_query
    using a Hugging Face sentence-transformers model.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


def build_vectorstore(chunks, persist_dir: str = None):
    """
    Build a Chroma vectorstore from chunks using HF sentence-transformers embeddings.
    """
    embeddings = HFEmbeddings()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return vectordb
