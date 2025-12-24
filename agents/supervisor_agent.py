# agents/supervisor_agent.py - FINAL FIXED VERSION

import os
import json
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool

from langchain_chroma import Chroma  # ✅ FIXED
from ingestion.retrieval import HybridRetriever
from prompts.answer_generation_prompt import ANSWER_GENERATION_PROMPT

# ---------- Shared LLM ----------
model = init_chat_model(
    "azure_openai:gpt-4o-mini",
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
)

# ---------- Sub-agent Prompts ----------
QUERY_ANALYZER_PROMPT = """
You are a Query Analyzer Agent for a document-grounded AI system.

Input: question (string)

RETURN STRICT JSON ONLY:
{
  "intent": "factual | reasoning | comparison | multi-hop | missing_data",
  "retrieval_strategy": "vector | bm25 | hybrid",
  "top_k": 5,
  "query": ""
}

Guidelines:
- "multi-hop": needs multiple evidence pieces
- "missing_data": info unlikely in documents
- "hybrid": long/ambiguous questions
- "vector": semantic questions
- "bm25": keyword/ID heavy queries

No prose. JSON only.
""".strip()

RETRIEVAL_AGENT_PROMPT = """
You are the Retrieval Agent.

You get: plan (JSON with intent, retrieval_strategy, top_k, query)

You MUST:
1. Call retrieval_tool(plan)
2. PARSE the JSON STRING response as dict
3. RETURN STRICT JSON ONLY:

{
  "top_chunks": [...],          // list of {source, text, score}
  "contradictions": "",         // contradictions or ""
  "diagnostics": {}             // from tool
}

JSON only. No explanations.
""".strip()

ANSWER_AGENT_PROMPT = ANSWER_GENERATION_PROMPT

# ---------- FIXED Middleware ----------
class RetrievalMiddleware(AgentMiddleware):
    def __init__(self, vectorstore, docs):
        self.vectorstore = vectorstore
        self.docs = docs
        
    @property
    def tools(self):
        @tool
        def retrieval_tool(plan: Dict[str, Any]) -> str:  # ✅ str return type
            """
            Execute retrieval according to plan:
            - retrieval_strategy: "vector", "bm25", or "hybrid"
            - top_k: int
            - query: str
            Returns JSON string: {"top_chunks": [...], "diagnostics": {}}
            """
            strategy = plan.get("retrieval_strategy", "hybrid")
            top_k = int(plan.get("top_k", 5))
            query = plan.get("query", "")

            retriever = HybridRetriever(self.vectorstore, self.docs)
            top_chunks, diagnostics = retriever.retrieve(
                query, top_k=top_k, strategy=strategy
            )

            print(f"[DEBUG] Retrieved chunks: {len(top_chunks)}")
            print(f"[DEBUG] Diagnostics: {diagnostics}")

            simple_chunks = [
                {
                    "source": c.get("source"),
                    "text": c.get("text", ""),
                    "score": float(c.get("score", 0.0)),
                }
                for c in top_chunks
            ]
            
            # ✅ CRITICAL FIX: RETURN JSON STRING
            result = {
                "top_chunks": simple_chunks, 
                "diagnostics": diagnostics
            }
            return json.dumps(result)

        return [retrieval_tool]

# ---------- Supervisor Factory ----------
def get_or_create_supervisor(vectorstore, docs):
    """Creates supervisor with fixed middleware."""
    retrieval_middleware = RetrievalMiddleware(vectorstore, docs)

    subagents = [
        {
            "name": "query-analyzer",
            "description": "Creates retrieval plan JSON from question.",
            "system_prompt": QUERY_ANALYZER_PROMPT,
            "tools": [],
        },
        {
            "name": "retrieval-agent",
            "description": "Executes retrieval and returns chunks JSON.",
            "system_prompt": RETRIEVAL_AGENT_PROMPT,
            "tools": [],  # From middleware
            "middleware": [retrieval_middleware]
        },
        {
            "name": "answer-agent",
            "description": "Generates final JSON answer with citations.",
            "system_prompt": ANSWER_AGENT_PROMPT,
            "tools": [],
        },
    ]

    supervisor = create_deep_agent(
        model=model,
        system_prompt=(
    "You orchestrate 3 sub-agents:\n\n"
    "1. task('query-analyzer', question) → plan JSON\n"
    "2. task('retrieval-agent', plan) → retrieval JSON with top_chunks\n"
    "3. task('answer-agent', question=question, context=retrieval.top_chunks) → final JSON\n\n"
    
    "**CRITICAL:** Pass the FULL retrieval JSON response as `context` to answer-agent.\n"
    "**Final output MUST be JSON ONLY with keys:**\n"
    "answer, evidence_used, top_chunks, missing_information, confidence_score, diagnostics"),

        subagents=subagents,
    )
    return supervisor

# ---------- Streamlit Helper ----------
def run_supervisor_pipeline(question: str, vectorstore, docs: List[Dict[str, Any]]) -> str:
    """Invoke supervisor and return JSON string."""
    supervisor = get_or_create_supervisor(vectorstore, docs)
    state = {"messages": [{"role": "user", "content": question}]}
    result = supervisor.invoke(state)

    # Extract content
    if hasattr(result, "content"):
        return result.content
    if isinstance(result, dict) and "messages" in result:
        msgs = result["messages"]
        if isinstance(msgs, list) and msgs:
            last = msgs[-1]
            return last.content if hasattr(last, "content") else last.get("content", str(result))
    return str(result)
