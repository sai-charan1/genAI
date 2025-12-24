# agents/supervisor_agent.py - FINAL WORKING VERSION

import os
import json
from typing import Dict, Any, List
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ingestion.retrieval import HybridRetriever
from prompts.answer_generation_prompt import ANSWER_GENERATION_PROMPT

model = init_chat_model("azure_openai:gpt-4o-mini",
                        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                        api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
                        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'))

QUERY_ANALYZER_PROMPT = """
You are a Query Analyzer Agent for a document-grounded AI system.

Input:
- question: the user question as a string.

You must decide and RETURN STRICT JSON ONLY:

{
  "intent": "factual | reasoning | comparison | multi-hop | missing_data",
  "retrieval_strategy": "vector | bm25 | hybrid",
  "top_k": 5,
  "query": ""
}

Guidelines:
- Use "multi-hop" if the answer needs multiple pieces of evidence.
- Use "missing_data" if the question clearly asks for info unlikely to be
  in the documents.
- Prefer "hybrid" for long or ambiguous questions, "vector" for semantic,
  "bm25" for very keyword / ID heavy queries.

Do NOT output anything except that JSON object.
""".strip()

RETRIEVAL_AGENT_PROMPT = """
You are the Retrieval Agent.

Input:
- plan: JSON with {intent, retrieval_strategy, top_k, query}

You MUST call retrieval_tool with the plan JSON and RETURN STRICT JSON ONLY:

{
  "top_chunks": [...],          
  "contradictions": "",         
  "diagnostics": {}             
}
""".strip()

ANSWER_AGENT_PROMPT = ANSWER_GENERATION_PROMPT

class RetrievalMiddleware(AgentMiddleware):
    def __init__(self, vectorstore, docs):
        self.vectorstore = vectorstore
        self.docs = docs
        self.retrieval_tool = self._create_retrieval_tool()
    
    def _create_retrieval_tool(self):
        @tool
        def retrieval_tool(plan: Dict[str, Any]) -> Dict[str, Any]:
            """
            Execute retrieval according to the plan. Input is JSON plan.
            """
            print(f"[DEBUG] 🛠️ TOOL START - plan: {plan}")
            strategy = plan.get("retrieval_strategy", "hybrid")
            top_k = int(plan.get("top_k", 5))
            query = plan.get("query", "")
            
            # ✅ 768-DIM EMBEDDINGS
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            retriever = HybridRetriever(self.vectorstore, self.docs)
            
            # ✅ PROVEN EMBEDDING FIX
            try:
                retriever.vectordb._client.set_embedding_function(embeddings)
            except:
                pass  # Use retriever's built-in embeddings
            
            top_chunks, diagnostics = retriever.retrieve(query, top_k=top_k, strategy=strategy)
            
            print(f"[DEBUG] Retrieved {len(top_chunks)} chunks")
            # ✅ FIXED: Handle HybridRetriever dict format
            simple_chunks = [
                {
                    "source": c.get('source', 'Unknown'),
                    "text": c.get('text', '')[:1000],
                    "score": c.get('score', 0.0)
                }
                for c in top_chunks if c and c.get('text', '').strip()
            ]
            print(f"[DEBUG] ✅ Formatted {len(simple_chunks)} chunks")
            return {"top_chunks": simple_chunks, "diagnostics": diagnostics}
        return retrieval_tool
    
    @property
    def tools(self):
        return [self.retrieval_tool]

def get_or_create_supervisor(vectorstore, docs):
    retrieval_middleware = RetrievalMiddleware(vectorstore, docs)
    subagents = [
        {
            "name": "query-analyzer",
            "description": "Understands the user question and returns a retrieval plan JSON.",
            "system_prompt": QUERY_ANALYZER_PROMPT,
            "tools": [],
        },
        {
            "name": "retrieval-agent",
            "description": "Executes retrieval plan via retrieval_tool and surfaces contradictions.",
            "system_prompt": RETRIEVAL_AGENT_PROMPT,
            "tools": retrieval_middleware.tools,
            "middleware": []
        },
        {
            "name": "answer-agent",
            "description": "Uses question + top_chunks to produce final JSON answer with citations.",
            "system_prompt": ANSWER_AGENT_PROMPT,
            "tools": [],
        },
    ]
    supervisor = create_deep_agent(
        model=model,
        system_prompt=(
            "You are an internal AI analyst.\n"
            "You orchestrate three sub-agents using tasks.\n\n"
            "Workflow:\n"
            "1) Call task(name='query-analyzer') with {question} to get a JSON plan.\n"
            "2) Call task(name='retrieval-agent') with {plan}; it must\n"
            "   use retrieval_tool to fetch top_chunks + diagnostics.\n"
            "3) Call task(name='answer-agent') with {question, context=top_chunks}.\n\n"
            "The answer-agent MUST output STRICT JSON ONLY with keys:\n"
            "  answer, evidence_used, top_chunks, missing_information, confidence_score, diagnostics\n\n"
            "Return ONLY that final JSON object as the user-visible output. No prose."
        ),
        subagents=subagents,
    )
    return supervisor

def run_supervisor_pipeline(question: str, vectorstore, docs: List[Dict[str, Any]]) -> str:
    supervisor = get_or_create_supervisor(vectorstore, docs)
    state = {"messages": [{"role": "user", "content": question}]}
    result = supervisor.invoke(state)
    
    if hasattr(result, "content"):
        return result.content
    if isinstance(result, dict):
        msgs = result.get("messages")
        if isinstance(msgs, list) and msgs:
            last = msgs[-1]
            if hasattr(last, "content"):
                return last.content
            if isinstance(last, dict) and "content" in last:
                return last["content"]
    return str(result)
