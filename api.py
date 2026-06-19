import os
import sys
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))

from src.utils import load_config, setup_cuda
from src.retriever import EvidenceAwareRetriever
from src.agent_pipeline import RAGMultiAgentSystem

app = FastAPI(title="Evidence-Aware RAG API")

# Allow CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global retriever instance
retriever = None

class QueryRequest(BaseModel):
    question: str
    num_documents: Optional[int] = 5

class RetrievalResponse(BaseModel):
    question: str
    query_used: str
    original_documents: List[str]
    filtered_documents: List[str]
    independence_score: float
    utility_score: float
    stability_score: float
    overall_quality: float
    final_answer: str
    metadata: dict

@app.on_event("startup")
def load_models():
    """Initialize the RAG pipeline on startup."""
    global retriever, agent_system
    
    print("[API] Loading configuration...")
    config = load_config("configs/config.yaml")
    setup_cuda(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"[API] Initializing retriever on {device}...")
    retriever = EvidenceAwareRetriever(
        embedder_name=config['models']['embedder']['name'],
        reranker_name=config['models']['reranker']['name'],
        nli_model_name=config['models']['nli']['name'],
        independence_config=config.get('independence', {}),
        utility_config=config.get('utility', {}),
        search_policy_config=config.get('search_policy', {}),
        stability_config=config.get('stability', {}),
        device=device,
        top_k=config['retrieval']['top_k'],
        use_stability=True,
    )
    
    print("[API] Running in Open-Domain Wikipedia Mode (No Static Database)")
    print("[API] Initializing LangGraph Multi-Agent System...")
    agent_system = RAGMultiAgentSystem(retriever_pipeline=retriever)
    
    print("[API] Ready!")

@app.post("/api/query", response_model=RetrievalResponse)
async def query_pipeline(request: QueryRequest):
    """Run the pipeline for a given query."""
    if not agent_system:
        return {"error": "Agent system not initialized"}
        
    result = agent_system.run(request.question)
    
    return RetrievalResponse(
        question=result.get('question', request.question),
        query_used=result.get('query_used', request.question),
        original_documents=result.get('original_documents', []),
        filtered_documents=result.get('filtered_documents', []),
        independence_score=float(result.get('independence_score', 0)),
        utility_score=float(result.get('utility_score', 0)),
        stability_score=float(result.get('stability_score', 0)),
        overall_quality=float(result.get('overall_quality', 0)),
        final_answer=result.get('final_answer', ''),
        metadata={
            "num_original": int(result.get('metadata', {}).get('num_original', 0)),
            "num_filtered": int(result.get('metadata', {}).get('num_filtered', 0))
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
