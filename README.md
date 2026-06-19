<div align="center">
  <h1>🛡️ Adaptive Evidence-Aware RAG</h1>
  <p><em>Beyond Agreement Counting: Evaluating Evidence Independence, Utility, Search Quality, and Behavioral Stability in Retrieval-Augmented Generation</em></p>

  [![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.103.1-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
  [![React](https://img.shields.io/badge/React-18.2.0-61DAFB.svg?logo=react)](https://react.dev/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

## 📖 1. Project Overview

This project implements a next-generation **Evidence-Aware RAG System** that goes beyond traditional retrieval by evaluating the quality of evidence across four key dimensions:

1. **Evidence Independence** - Detecting replicated narratives vs. truly independent sources
2. **Retrieval Utility Learning** - Rewarding evidence that provides new information, confidence gains, or contradiction discovery
3. **Search Policy Learning** - Learning which search strategies consistently produce high-quality evidence
4. **Behavioral Stability** - Measuring robustness across equivalent query variations

---

## 💡 2. The Core Insight

Current RAG systems think: *"3 websites agree = strong evidence"*

We discovered that's not enough. What matters is whether the evidence is truly **independent**, **useful**, and **stable**.

**Example Problem:**
- Blog A copies Blog B
- Blog C copies Blog B
- Current systems see: "3 sources agree!"
- Reality: "1 source + 2 copies"

Our architecture leverages **Live Wikipedia Retrieval**, NLI (Natural Language Inference) models (DeBERTa), and LLM-driven paraphrasing to ensure that answers are built on robust, independent, and contradiction-free facts.

---

## 🏗️ 3. Architecture

The system transitions from traditional vector-database RAG into a dynamic, **Open-Domain Live Web RAG**.

```mermaid
graph TD
    User([User Query]) --> SP[Search Policy Optimizer]
    SP --> |Optimized Query| LR[Live Wikipedia Retriever]
    
    subgraph Multi-Query Behavioral Stability
        BS[Stability Checker / LLM]
        SP --> |Paraphrased Query 1| BS
        SP --> |Paraphrased Query 2| BS
        SP --> |Paraphrased Query 3| BS
        BS --> |Parallel Searches| LR
    end
    
    LR --> |Raw Evidence Docs| ES[Evidence Scoring Pipeline]
    
    subgraph Evidence Scoring Pipeline
        ES --> EI[Independence Scorer]
        EI --> |Cross-Encoder NLI| EI_Score{Are docs independent?}
        EI_Score -- No --> Drop[Discard Redundant Docs]
        EI_Score -- Yes --> RU[Utility Scorer]
        
        RU --> |Contradiction Check| RU_Score{Does it add utility?}
        RU_Score -- No --> Drop
        RU_Score -- Yes --> Keep[High-Quality Evidence]
    end
    
    Keep --> LLM[Qwen Synthesizer LLM]
    LLM --> Final([Robust Hallucination-Free Answer])
    
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef highlight fill:#d4edda,stroke:#28a745,stroke-width:2px;
    classDef drop fill:#f8d7da,stroke:#dc3545,stroke-width:2px;
    
    class Keep highlight;
    class Drop drop;
```

---

## 📊 4. Experimental Results

Our system drastically reduces hallucination rates while maintaining high retrieval utility across diverse datasets.

### Module Performance (Independence vs Utility vs Stability)
![Module Performance](logs/figures/module_performance_bar.png)

### Ablation Study: Standard RAG vs Evidence-Aware RAG
![Ablation Study](logs/figures/ablation_study.png)

---

## 🚀 5. Quick Start (Open-Domain Mode)

### Installation

```bash
# Clone the repository
git clone https://github.com/DYNOSuprovo/adaptive-evidence-rag.git
cd adaptive-evidence-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\Activate.ps1   # Windows PowerShell

# Install dependencies
pip install -e .
```

### Full-Stack Usage (React + FastAPI)

The fastest way to test the hallucination-free generation is by running the built-in React UI and FastAPI backend.

1. **Start the API** (Runs the Live Wikipedia + Qwen Backend):
```bash
python api.py
```

2. **Start the React Frontend**:
```bash
cd frontend
npm install
npm run dev
```

Navigate to `http://localhost:5173/` and ask the system open-domain questions like:
- *"Who painted the Mona Lisa?"*
- *"When did World War 1 start?"*
- *"What is a black hole?"*

### Python API Usage

```python
from src.retriever import EvidenceAwareRetriever

# Initialize the retriever with all modules active!
retriever = EvidenceAwareRetriever(
    embedder_name="BAAI/bge-large-en-v1.5",
    use_independence=True,
    use_utility=True,
    use_search_policy=True,
    use_stability=True,  # Activates Multi-Query Paraphrasing
)

# Run full pipeline (Dynamically fetches from Live Wikipedia!)
result = retriever.run_pipeline("Who invented the transformer architecture?", check_stability=True)

print(f"Independence: {result.independence_score:.4f}")
print(f"Utility:      {result.utility_score:.4f}")
print(f"Stability:    {result.stability_score:.4f}")
print(f"Final Answer: {result.final_answer}")
```

---

## 🛠️ 6. Models & Resources

### Embedding & NLI Models

| Model | Use Case | Link |
|-------|----------|------|
| **BAAI/bge-small-en-v1.5** | Primary embeddings | [HuggingFace](https://huggingface.co/BAAI/bge-small-en-v1.5) |
| **cross-encoder/ms-marco-MiniLM-L-6-v2** | Cross-encoder Reranking | [HuggingFace](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) |
| **cross-encoder/nli-deberta-v3-xsmall** | Independence & Utility Scoring | [HuggingFace](https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall) |

### Generator LLMs

| Model | Size | Link |
|-------|------|------|
| **Qwen/Qwen1.5-0.5B-Chat** | 0.5B (Local fine-tuned LoRA) | [HuggingFace](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat) |

---

## 📂 7. Project Structure

```
adaptive-evidence-rag/
|-- api.py                        # FastAPI Backend
|-- frontend/                     # React User Interface
|-- configs/                      # Configuration files
|-- notebooks/                    # Jupyter notebooks for model training and metric evaluation
|-- src/
|   |-- evidence_independence.py  # Module 1: Detects copied narratives
|   |-- retrieval_utility.py      # Module 2: Detects useless/contradictory info
|   |-- search_policy.py          # Module 3: Keyword optimization
|   |-- behavioral_stability.py   # Module 4: Multi-query paraphrasing
|   |-- retriever.py              # Core Evidence-Aware Retriever
|   |-- agent_pipeline.py         # Multi-Agent LangGraph Router
|-- scratch/                      # Test scripts
|-- requirements.txt
|-- README.md
```

---

## 🏆 8. Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **Answer Accuracy** | > 85% | Exact match & F1 score |
| **Independence Score** | > 0.8 | Measures lack of semantic redundancy |
| **Utility Score** | > 0.7 | Penalizes redundant evidence |
| **Stability Score** | > 0.9 | Consistency across equivalent query paraphrases |
| **Hallucination Rate** | < 5% | Unsupported claims passed to final generation |

---

## 📝 9. License

MIT License - See LICENSE file for details.
