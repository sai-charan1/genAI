# agents/supervisor_agent.py

import os
import json
from typing import Dict, Any, List

from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_chroma import Chroma
from ingestion.retrieval import HybridRetriever
from prompts.answer_generation_prompt import ANSWER_GENERATION_PROMPT


# ---------- Shared LLM for all agents ----------

# model = AzureChatOpenAI(
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     temperature=0,
#     max_retries=5,
# )
model = init_chat_model("azure_openai:gpt-4o-mini",
						api_key=os.getenv('AZURE_OPENAI_API_KEY'),
						api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
						azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'))


# ---------- Sub‑agent prompts ----------

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
- You also have access to a tool: retrieval_tool_dynamic(plan, vectorstore, docs)
**MAKESURE TO CALL THE RETRIVER TOOL DYNAMIC TO GET THE ANSWER FOR THE QUERY THAT IS GIVEN**
You MUST:
1) Call retrieval_tool_dynamic(plan, vectorstore, docs).
2) Take its output and RETURN STRICT JSON ONLY:

{
  "top_chunks": [...],          // list of {source, text, score}
  "contradictions": "",         // describe contradictions if any, else ""
  "diagnostics": {}             // pass-through diagnostics from the tool
}

Do NOT output anything except that JSON object.
""".strip()

# ANSWER_GENERATION_PROMPT already enforces correct schema and types
ANSWER_AGENT_PROMPT = ANSWER_GENERATION_PROMPT


# ---------- Middleware Definition ----------

class RetrievalMiddleware(AgentMiddleware):
    def __init__(self, vectorstore, docs):
        self.vectorstore = vectorstore
        self.docs = docs
        
    @property
    def tools(self):
        @tool
        def retrieval_tool(plan: Dict[str, Any]) -> Dict[str, Any]:
            """
            Execute retrieval according to the plan:
            - plan["retrieval_strategy"]: "vector", "bm25", or "hybrid"
            - plan["top_k"]: int
            - plan["query"]: str
            Returns: {top_chunks, diagnostics}
            """
            strategy = plan.get("retrieval_strategy", "hybrid")
            top_k = int(plan.get("top_k", 5))
            query = plan.get("query", "")

            # Use self.vectorstore and self.docs injected via __init__
            retriever = HybridRetriever(self.vectorstore, self.docs)
            top_chunks, diagnostics = retriever.retrieve(query, top_k=top_k, strategy=strategy)

            # Debug: see how many chunks we actually retrieved
            print("[DEBUG] Retrieved chunks:", len(top_chunks))
            print("[DEBUG] Diagnostics:", diagnostics)

            simple_chunks = [
                {
                    "source": c.get("source"),
                    "text": c.get("text", ""),
                    "score": c.get("score", 0.0),
                }
                for c in top_chunks
            ]
            return {"top_chunks": simple_chunks, "diagnostics": diagnostics}
        
        return [retrieval_tool]


# ---------- Supervisor DeepAgent Factory ----------

def get_or_create_supervisor(vectorstore, docs):
    """
    Creates the supervisor agent with the RetrievalMiddleware injected.
    """
    
    # Instantiate middleware with captured state
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
            "tools": [], # Tools are provided by the middleware
            "middleware": [retrieval_middleware]
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


# ---------- Helper for Streamlit ----------

def run_supervisor_pipeline(
    question: str,
    vectorstore,
    docs: List[Dict[str, Any]],
) -> str:
    """
    Invoke the DeepAgent supervisor and return final output as a string.
    This string should be a JSON object, but the UI will treat it as raw text
    if parsing fails.
    """
    # Create the supervisor with current vectorstore/docs
    supervisor = get_or_create_supervisor(vectorstore, docs)
    
    state = {
        "messages": [{"role": "user", "content": question}],
    }

    result = supervisor.invoke(state)

    # Case 1: AIMessage-like
    if hasattr(result, "content"):
        return result.content

    # Case 2: dict with messages
    if isinstance(result, dict):
        msgs = result.get("messages")
        if isinstance(msgs, list) and msgs:
            last = msgs[-1]
            if hasattr(last, "content"):
                return last.content
            if isinstance(last, dict) and "content" in last:
                return last["content"]

    # Fallback: string representation
    return str(result)