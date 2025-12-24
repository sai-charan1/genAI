import os
import json
from typing import Dict, Any, List
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from deepagents import create_deep_agent
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from ingestion.retrieval import HybridRetriever
from prompts.answer_generation_prompt import ANSWER_GENERATION_PROMPT

model = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    temperature=0.1
)

QUERY_ANALYZER_PROMPT = """
You are a Query Analyzer Agent for a document-grounded AI system.
Input - question the user question as a string. 
You must decide and RETURN STRICT JSON ONLY
{
  "intent": "factual | reasoning | comparison | multi-hop | missing_data",
  "retrieval_strategy": "vector | bm25 | hybrid",
  "top_k": 5,
  "query": ""
}
""".strip()

RETRIEVAL_AGENT_PROMPT = """
You are the Retrieval Agent. 
Input - plan JSON with {intent, retrieval_strategy, top_k, query}

You MUST:
1. Call retrieval_tool('{"retrieval_strategy": "...", "top_k": 5, "query": "..."}')
2. Take its output and RETURN STRICT JSON ONLY:
{
  "top_chunks": [...], // list of {source, text, score}
  "contradictions": "",
  "diagnostics": {}
}
""".strip()

ANSWER_AGENT_PROMPT = ANSWER_GENERATION_PROMPT

def create_retrieval_tools(vectorstore, docs):
    """🔥 FIXED: Closure captures vectorstore/docs"""
    
    @tool
    def retrieval_tool(plan_json: str) -> str:
        """Execute retrieval - plan_json is stringified plan"""
        print(f"[DEBUG] 🛠️ TOOL START - plan: {plan_json[:80]}...")
        
        plan = json.loads(plan_json)
        strategy = plan.get("retrieval_strategy", "hybrid")
        top_k = int(plan.get("top_k", 5))
        query = plan.get("query", "")
        
        print(f"[DEBUG] Strategy: {strategy}, top_k: {top_k}, query: '{query[:50]}'")
        
        valid_docs = [d for d in docs if d.page_content and len(d.page_content.strip()) > 5]
        print(f"[DEBUG] Valid docs passed to retriever: {len(valid_docs)}")
        
        retriever = HybridRetriever(vectorstore, valid_docs)
        raw_chunks, diagnostics = retriever.retrieve(query, top_k=top_k, strategy=strategy)
        
        print(f"[DEBUG] Raw chunks from retriever: {len(raw_chunks)}")
        
        simple_chunks = []
        for i, chunk in enumerate(raw_chunks):
            print(f"[DEBUG] Chunk {i} keys: {list(chunk.keys())}")
            print(f"[DEBUG] Chunk {i} text len: {len(chunk.get('text', ''))}")
            
            if isinstance(chunk, dict) and 'text' in chunk:
                text = chunk['text']
                if text:
                    simple_chunks.append({
                        "source": chunk.get('source', 'unknown'),
                        "text": text[:1500],
                        "score": float(chunk.get('score', 0.0))
                    })
                    print(f"[DEBUG] ✅ ADDED chunk {i}: {len(text)} chars")
        
        print(f"[DEBUG] TOTAL simple_chunks created: {len(simple_chunks)}")
        
        result = {
            "top_chunks": simple_chunks,
            "diagnostics": diagnostics or {}
        }
        print(f"[DEBUG] TOOL END - returning {len(simple_chunks)} chunks")
        return json.dumps(result)
    
    return [retrieval_tool]

def get_or_create_supervisor(vectorstore, docs):
    """NO MIDDLEWARE - DIRECT TOOLS ONLY"""
    tools = create_retrieval_tools(vectorstore, docs)
    
    subagents = [
        {
            "name": "query-analyzer",
            "description": "Creates JSON retrieval plan",
            "system_prompt": QUERY_ANALYZER_PROMPT,
            "tools": []
        },
        {
            "name": "retrieval-agent",
            "description": "Calls retrieval_tool with plan JSON",
            "system_prompt": RETRIEVAL_AGENT_PROMPT,
            "tools": tools
        },
        {
            "name": "answer-agent", 
            "description": "Generates final JSON answer",
            "system_prompt": ANSWER_AGENT_PROMPT,
            "tools": []
        }
    ]
    
    supervisor = create_deep_agent(
        model=model,
        system_prompt="""
You orchestrate 3 subagents using task():

1. task("query-analyzer") with question → JSON plan  
2. task("retrieval-agent") with plan → top_chunks JSON
3. task("answer-agent") with question + top_chunks → FINAL JSON

Return ONLY final JSON object.""",
        subagents=subagents
    )
    return supervisor

def run_supervisor_pipeline(question: str, vectorstore, docs: List[Document]) -> str:
    supervisor = get_or_create_supervisor(vectorstore, docs)
    state = {"messages": [{"role": "user", "content": question}]}
    result = supervisor.invoke(state)
    return result["messages"][-1].content if isinstance(result, dict) else str(result)
