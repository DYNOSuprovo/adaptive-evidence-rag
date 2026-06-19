"""
Interactive Pipeline Test Script
================================
Run your Adaptive Evidence-Aware RAG pipeline interactively from the terminal.

Usage:
    python test_pipeline.py
    python test_pipeline.py --question "What is machine learning?"
"""

import os
import sys
import json
import argparse
import torch
from typing import List

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.utils import load_config, setup_cuda
from src.retriever import EvidenceAwareRetriever


# ──────────────────────────────────────────────
# Sample knowledge corpus for demonstration
# ──────────────────────────────────────────────
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
] * 1  # Repeat to make a larger corpus


def print_header():
    """Print a styled header."""
    print()
    print("+" + "-" * 62 + "+")
    print("|" + " Adaptive Evidence-Aware RAG - Pipeline Tester ".center(62) + "|")
    print("+" + "-" * 62 + "+")
    print("|" + " Modules: Independence | Utility | Search Policy | Stability ".center(62) + "|")
    print("+" + "-" * 62 + "+")
    print()


def print_divider(char="-", width=64):
    print(char * width)


def display_result(result):
    """Display a retrieval result in a clean, readable format."""
    print()
    print_divider("-")
    print(f"  ? Question: {result.question}")
    print(f"  > Query Used: {result.query_used}")
    print_divider()

    # Scores
    print(f"  [SCORES]")
    print(f"     Independence:  {result.independence_score:.4f}")
    print(f"     Utility:       {result.utility_score:.4f}")
    print(f"     Stability:     {result.stability_score:.4f}")
    print(f"     * Overall:     {result.overall_quality:.4f}")
    print_divider()

    # Documents
    print(f"  [RETRIEVED]: {result.metadata.get('num_original', '?')} docs -> "
          f"FILTERED: {result.metadata.get('num_filtered', '?')} docs")
    print_divider()

    for i, doc in enumerate(result.filtered_documents, 1):
        # Truncate long docs for display
        display_doc = doc[:120] + "..." if len(doc) > 120 else doc
        print(f"  [{i}] {display_doc}")

    print_divider("-")
    print()


def run_interactive(retriever):
    """Run an interactive question-answer loop."""
    print("  Type a question and press Enter. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("  ? Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break

        print("  Running pipeline...")
        result = retriever.run_pipeline(question)
        display_result(result)


def main():
    parser = argparse.ArgumentParser(description="Test the Evidence-Aware RAG Pipeline")
    parser.add_argument("--question", "-q", type=str, default=None,
                        help="Single question to test (skips interactive mode)")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--stability", action="store_true",
                        help="Enable stability checking (slower)")
    args = parser.parse_args()

    print_header()

    # Load config
    config = load_config(args.config)
    setup_cuda(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}\n")

    # Initialize retriever
    print("  Loading models...")
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
        use_stability=args.stability,
    )

    # Index sample documents
    print("  Indexing sample knowledge corpus...")
    retriever.index_documents(SAMPLE_CORPUS)
    print(f"  Indexed {len(SAMPLE_CORPUS)} documents.\n")

    if args.question:
        # Single question mode
        print(f"  Running pipeline for: \"{args.question}\"")
        result = retriever.run_pipeline(args.question)
        display_result(result)
    else:
        # Interactive mode
        run_interactive(retriever)


if __name__ == "__main__":
    main()
