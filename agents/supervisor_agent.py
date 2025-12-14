# agents/supervisor_agent.py

import os
from typing import Dict, Any, List

from dotenv import load_dotenv
load_dotenv()

from deepagents import create_deep_agent
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool

from ingestion.retrieval import HybridRetriever
from prompts.answer_generation_prompt import ANSWER_GENERATION_PROMPT


# ---------- shared LLM for planning / retrieval-agent ----------
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0,
)


QUERY_ANALYZER_PROMPT = """
You are a Query Analyzer Agent for a document-grounded AI system.
Decide:
- intent: factual, reasoning, comparison, multi-hop, missing_data
- retrieval_strategy: vector, bm25, hybrid
- top_k: integer
- query: possibly rewritten query (same topic)

Return STRICT JSON ONLY:
{"intent": "", "retrieval_strategy": "", "top_k": 5, "query": ""}
""".strip()

RETRIEVAL_AGENT_PROMPT = """
You are the Retrieval Agent.
You will be given a retrieval plan JSON and access to a retrieval tool.
Call the tool with the query to fetch top chunks.

Return STRICT JSON ONLY:
{"top_chunks": [...], "contradictions": "", "diagnostics": {}}
""".strip()

ANSWER_AGENT_PROMPT = ANSWER_GENERATION_PROMPT


@tool
def retrieval_tool_hybrid_top5(query: str, vectorstore, docs) -> Dict[str, Any]:
    """
    Retrieve the top 5 most relevant chunks for a query using HybridRetriever
    over the given vectorstore and docs. Returns:
      - top_chunks: list of {source, text, score}
      - diagnostics: retrieval diagnostic information.
    """
    retriever = HybridRetriever(vectorstore, docs)
    top_chunks, diagnostics = retriever.retrieve(query, top_k=5)
    return {"top_chunks": top_chunks, "diagnostics": diagnostics}


subagents = [
    {
        "name": "query-analyzer",
        "description": "Understands the user question and returns a retrieval plan JSON.",
        "system_prompt": QUERY_ANALYZER_PROMPT,
        "tools": [],
    },
    {
        "name": "retrieval-agent",
        "description": "Uses retrieval_tool_hybrid_top5 to get the top 5 chunks for the query.",
        "system_prompt": RETRIEVAL_AGENT_PROMPT,
        "tools": [retrieval_tool_hybrid_top5],
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
        "Always follow this sequence:\n"
        "1) task(name='query-analyzer') -> plan JSON.\n"
        "2) task(name='retrieval-agent') -> call retrieval_tool_hybrid_top5(plan.query) -> top_chunks.\n"
        "3) task(name='answer-agent') with {question, context=top_chunks} -> final JSON answer.\n"
        "Return ONLY the final JSON answer to the user, with no additional text."
    ),
    subagents=subagents,
)


# ---------- helper to call the answer model explicitly with context ----------
import json

answer_model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0,
)


def generate_answer_with_context(question: str, top_chunks: List[Dict[str, Any]]) -> str:
    """
    Call the answer-generation model with question + context and expect strict JSON
    with: answer, evidence_used, top_chunks, missing_information, confidence_score.
    """
    payload = {
        "question": question,
        "context": [
            {
                "source": c.get("source"),
                "text": c.get("text", ""),
                "score": c.get("score", 0.0),
            }
            for c in top_chunks
        ],
    }
    messages = [
        {"role": "system", "content": ANSWER_AGENT_PROMPT},
        {"role": "user", "content": json.dumps(payload)},
    ]
    resp = answer_model.invoke(messages)
    return resp.content