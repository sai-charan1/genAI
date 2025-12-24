# agents/supervisor_agent.py - COMPLETE FIXED VERSION

import os
import json
from typing import Dict, Any, List, Annotated
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from ingestion.retrieval import HybridRetriever
from prompts.answer_generation_prompt import ANSWER_GENERATION_PROMPT

# ========== FIXED MODEL ==========
model = init_chat_model("azure_openai:gpt-4o-mini",
                        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                        api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
                        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'))

# ========== FIXED PROMPTS ==========
QUERY_ANALYZER_PROMPT = """
You are a Query Analyzer Agent for a document-grounded AI system.

Input: question (string)

Return STRICT JSON ONLY:
{
  "intent": "factual | reasoning | comparison | multi-hop | missing_data",
  "retrieval_strategy": "vector | bm25 | hybrid", 
  "top_k": 5,
  "query": "optimized search query"
}

Guidelines:
- Use "hybrid" for most queries
- "missing_data" only if clearly not in docs
- Optimize query for retrieval
""".strip()

RETRIEVAL_AGENT_PROMPT = """
You are the Retrieval Agent. Input contains JSON plan.

MANDATORY: Call retrieval_tool('{"retrieval_strategy": "...", "top_k": 5, "query": "..."}')

Parse tool response and return STRICT JSON ONLY:
{
  "top_chunks": [...],      // list of {source, text, score}
  "contradictions": "",     // describe contradictions or ""
  "diagnostics": {}         // pass-through diagnostics
}
""".strip()

ANSWER_AGENT_PROMPT = ANSWER_GENERATION_PROMPT

# ========== FIXED RETRIEVAL TOOLS ==========
def create_retrieval_tools(vectorstore, docs) -> List:
    """Creates retrieval tools for DeepAgents subagents"""
    
    @tool
    def retrieval_tool(plan_str: str) -> str:
        """
        Retrieves document chunks based on retrieval plan.
        
        Args:
            plan_str: JSON string like '{"retrieval_strategy": "hybrid", "top_k": 3, "query": "..."}'
        """
        print(f"[DEBUG] 🛠️ retrieval_tool called: {plan_str[:80]}...")
        
        try:
            plan = json.loads(plan_str)
            strategy = plan.get("retrieval_strategy", "hybrid")
            top_k = int(plan.get("top_k", 5))
            query = plan.get("query", "")
        except Exception as e:
            print(f"[ERROR] Plan parse failed: {e}")
            return json.dumps({"top_chunks": [], "diagnostics": {"error": f"Parse error: {e}"}})
        
        print(f"[DEBUG] Executing {strategy} x{top_k}: '{query[:50]}...'")
        
        retriever = HybridRetriever(vectorstore, docs)
        top_chunks, diagnostics = retriever.retrieve(query, top_k=top_k, strategy=strategy)
        
        simple_chunks = []
        for i, c in enumerate(top_chunks):
            if c and hasattr(c, 'page_content') and c.page_content.strip():
                simple_chunks.append({
                    "source": getattr(c, 'metadata', {}).get('source', f'chunk_{i}'),
                    "text": c.page_content[:1500],
                    "score": float(getattr(c, 'score', 0.0))
                })
        
        result = {"top_chunks": simple_chunks, "diagnostics": diagnostics or {}}
        print(f"[DEBUG] ✅ RETURNING {len(simple_chunks)} chunks")
        return json.dumps(result)
    
    return [retrieval_tool]

# ========== FIXED SUPERVISOR ==========
def get_or_create_supervisor(vectorstore, docs):
    """
    Creates the supervisor agent with proper DeepAgents subagent pattern.
    """
    retrieval_tools = create_retrieval_tools(vectorstore, docs)
    
    subagents = [
        {
            "name": "query-analyzer",
            "description": "Creates JSON retrieval plan from question",
            "system_prompt": QUERY_ANALYZER_PROMPT,
            "tools": []
        },
        {
            "name": "retrieval-agent",
            "description": "Executes retrieval plan using retrieval_tool",
            "system_prompt": RETRIEVAL_AGENT_PROMPT,
            "tools": retrieval_tools  # ✅ Direct tool binding
        },
        {
            "name": "answer-agent",
            "description": "Generates final JSON answer with citations",
            "system_prompt": ANSWER_AGENT_PROMPT,
            "tools": []
        }
    ]
    
    supervisor = create_deep_agent(
        model=model,
        system_prompt="""
You are an internal AI analyst orchestrating 3 sub-agents using task() calls.

STRICT WORKFLOW:
1. Call task("query-analyzer") with {question} → Get JSON plan
2. Call task("retrieval-agent") with {plan} → Get top_chunks JSON  
3. Call task("answer-agent") with {question, context=top_chunks} → Get FINAL JSON

Return ONLY the final JSON from answer-agent. No explanations.""",
        subagents=subagents
    )
    return supervisor

# ========== STREAMLIT HELPER ==========
def run_supervisor_pipeline(question: str, vectorstore, docs: List[Document]) -> str:
    """
    Invoke supervisor and return JSON string for UI.
    """
    supervisor = get_or_create_supervisor(vectorstore, docs)
    state = {"messages": [{"role": "user", "content": question}]}
    result = supervisor.invoke(state)
    
    # Extract content safely
    if hasattr(result, "content"):
        return result.content
    if isinstance(result, dict) and "messages" in result:
        msgs = result["messages"]
        if msgs and hasattr(msgs[-1], "content"):
            return msgs[-1].content
    return str(result)
