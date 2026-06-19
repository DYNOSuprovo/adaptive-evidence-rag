import os
import sys
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

sys.path.insert(0, os.path.dirname(__file__))

from src.utils import load_config, setup_cuda
from src.retriever import EvidenceAwareRetriever

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
    metadata: dict

@app.on_event("startup")
def load_models():
    """Initialize the RAG pipeline on startup."""
    global retriever
    
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
        use_stability=False,
    )
    
    # Load sample corpus (same as app.py)
    SAMPLE_CORPUS = [
        "Artificial intelligence (AI) is the simulation of human intelligence processes by computer systems, including learning, reasoning, and self-correction.",
        "Machine learning is a subset of artificial intelligence that enables systems to automatically learn and improve from experience without being explicitly programmed.",
        "Deep learning is a branch of machine learning that uses artificial neural networks with multiple layers (hence 'deep') to model and understand complex patterns in data.",
        "Natural language processing (NLP) allows computers to understand, interpret, and generate human language in a valuable way.",
        "Computer vision is a field of AI that trains computers to interpret and understand the visual world through digital images and videos.",
        "Reinforcement learning is an area of machine learning where an agent learns to make decisions by performing actions and receiving rewards or penalties.",
        "Transfer learning is a technique where a model trained on one task is repurposed as the starting point for a model on a second, related task.",
        "Generative AI refers to artificial intelligence systems capable of creating new content including text, images, audio, and video.",
        "Large language models (LLMs) are neural networks trained on massive text datasets that can generate human-like text and perform a wide range of language tasks.",
        "Retrieval-Augmented Generation (RAG) combines information retrieval with language generation to produce more accurate and grounded responses.",
        "The transformer architecture, introduced in the 'Attention Is All You Need' paper, revolutionized NLP by enabling parallel processing of sequences.",
        "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model developed by Google for understanding the context of words in search queries.",
        "GPT (Generative Pre-trained Transformer) is a family of large language models developed by OpenAI, trained using unsupervised learning on large text corpora.",
        "Convolutional Neural Networks (CNNs) are deep learning models primarily used for image recognition and classification tasks.",
        "Recurrent Neural Networks (RNNs) are designed to recognize patterns in sequences of data, such as text, genomes, handwriting, or spoken words.",
        "Attention mechanisms allow neural networks to focus on relevant parts of the input when producing output, dramatically improving performance on sequence tasks.",
        "Supervised learning uses labeled training data to learn a mapping from inputs to outputs, commonly used for classification and regression tasks.",
        "Unsupervised learning finds hidden patterns in data without labeled examples, commonly used for clustering, dimensionality reduction, and anomaly detection.",
        "Semi-supervised learning uses a combination of labeled and unlabeled data during training, often when labeled data is scarce.",
        "Federated learning is a machine learning approach where a model is trained across decentralized devices holding local data samples, without exchanging them.",
        "Neural Architecture Search (NAS) automates the design of artificial neural networks, using algorithms to discover optimal network architectures.",
        "Explainable AI (XAI) refers to methods and techniques that make AI system results understandable to humans.",
        "The Turing test, proposed by Alan Turing in 1950, is a measure of machine intelligence based on whether a machine can exhibit behavior indistinguishable from a human.",
        "Quantum computing uses quantum-mechanical phenomena such as superposition and entanglement to perform computations far faster than classical computers for certain problems.",
        "Blockchain is a decentralized, distributed ledger technology that records transactions across many computers so that records cannot be altered retroactively.",
        "Climate change refers to long-term shifts in temperatures and weather patterns, primarily driven by human activities since the 1800s, especially the burning of fossil fuels.",
        "The human genome contains approximately 3 billion DNA base pairs and about 20,000–25,000 protein-coding genes.",
        "Vaccines work by stimulating the body's immune system to recognize and fight specific pathogens, providing immunity without causing the disease.",
        "The speed of light in a vacuum is approximately 299,792,458 meters per second, a fundamental constant in physics.",
        "Photosynthesis is the process by which plants convert carbon dioxide and water into glucose and oxygen using sunlight.",
    ]
    print(f"[API] Indexing {len(SAMPLE_CORPUS)} documents...")
    retriever.index_documents(SAMPLE_CORPUS)
    print("[API] Ready!")

@app.post("/api/query", response_model=RetrievalResponse)
async def query_pipeline(request: QueryRequest):
    """Run the pipeline for a given query."""
    if not retriever:
        return {"error": "Retriever not initialized"}
        
    result = retriever.run_pipeline(request.question)
    
    return RetrievalResponse(
        question=result.question,
        query_used=result.query_used,
        original_documents=result.original_documents,
        filtered_documents=result.filtered_documents,
        independence_score=float(result.independence_score),
        utility_score=float(result.utility_score),
        stability_score=float(result.stability_score),
        overall_quality=float(result.overall_quality),
        metadata={
            "num_original": int(result.metadata.get('num_original', 0)),
            "num_filtered": int(result.metadata.get('num_filtered', 0))
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
