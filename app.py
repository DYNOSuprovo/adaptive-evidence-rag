"""
Gradio Web UI for Adaptive Evidence-Aware RAG
==============================================
A beautiful, interactive web interface for querying the RAG pipeline.

Usage:
    python app.py
    python app.py --port 7860
"""

import os
import sys
import json
import argparse
import torch
import gradio as gr
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from src.utils import load_config, setup_cuda
from src.retriever import EvidenceAwareRetriever


# ──────────────────────────────────────────────
# Sample knowledge corpus
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
] * 1


# ──────────────────────────────────────────────
# Global retriever (initialized on startup)
# ──────────────────────────────────────────────
retriever = None


def initialize_retriever():
    """Initialize the retriever on startup."""
    global retriever

    config = load_config("configs/config.yaml")
    setup_cuda(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    retriever.index_documents(SAMPLE_CORPUS)
    print(f"[App] Retriever initialized with {len(SAMPLE_CORPUS)} documents on {device}")


def query_pipeline(question: str) -> Tuple[str, str, str]:
    """Run the RAG pipeline and return formatted results."""
    if not question or not question.strip():
        return "", "", ""

    result = retriever.run_pipeline(question.strip())

    # ── Format Scores ──
    def score_bar(score, label):
        pct = int(score * 100)
        filled = "█" * (pct // 5)
        empty = "░" * (20 - pct // 5)
        return f"**{label}**: `{filled}{empty}` **{score:.3f}**"

    scores_md = "\n\n".join([
        score_bar(result.independence_score, "🔗 Independence"),
        score_bar(result.utility_score, "⚡ Utility"),
        score_bar(result.stability_score, "🛡️ Stability"),
        "---",
        f"### ⭐ Overall Quality: **{result.overall_quality:.3f}**",
    ])

    # ── Format Documents ──
    docs_md = ""
    if result.filtered_documents:
        for i, doc in enumerate(result.filtered_documents, 1):
            docs_md += f"**[{i}]** {doc}\n\n---\n\n"
    else:
        docs_md = "*No documents passed the quality filters.*"

    # ── Format Metadata ──
    meta = result.metadata
    meta_md = f"""| Metric | Value |
|--------|-------|
| Query Used | `{result.query_used}` |
| Documents Retrieved | {meta.get('num_original', 'N/A')} |
| Documents After Filtering | {meta.get('num_filtered', 'N/A')} |
| Compression Ratio | {meta.get('num_filtered', 0) / max(meta.get('num_original', 1), 1):.1%} |
"""

    return scores_md, docs_md, meta_md


# ──────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────
custom_css = """
.gradio-container {
    max-width: 960px !important;
    margin: auto !important;
}
.header-text {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2em;
    font-weight: 800;
    margin-bottom: 0.2em;
}
.sub-text {
    text-align: center;
    color: #666;
    font-size: 1em;
    margin-bottom: 1.5em;
}
"""


def build_ui():
    """Build the Gradio interface."""

    with gr.Blocks(
        title="Adaptive Evidence-Aware RAG",
    ) as demo:

        gr.HTML("""
        <div class="header-text">🧠 Adaptive Evidence-Aware RAG</div>
        <div class="sub-text">
            Ask a question and watch the pipeline retrieve, filter, and score evidence in real-time.<br>
            <b>Modules:</b> Evidence Independence · Retrieval Utility · Search Policy · Behavioral Stability
        </div>
        """)

        with gr.Row():
            question_input = gr.Textbox(
                label="Ask a Question",
                placeholder="e.g. What is the difference between machine learning and deep learning?",
                lines=2,
                elem_id="question-input",
            )

        with gr.Row():
            submit_btn = gr.Button("🔍 Run Pipeline", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="lg")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Quality Scores")
                scores_output = gr.Markdown(elem_id="scores-output")

            with gr.Column(scale=1):
                gr.Markdown("### 📋 Pipeline Metadata")
                meta_output = gr.Markdown(elem_id="meta-output")

        gr.Markdown("### 📄 Retrieved Evidence (Post-Filtering)")
        docs_output = gr.Markdown(elem_id="docs-output")

        # ── Example Questions ──
        gr.Markdown("### 💡 Try These Examples")
        gr.Examples(
            examples=[
                ["What is machine learning?"],
                ["How does deep learning differ from traditional AI?"],
                ["Explain the transformer architecture"],
                ["What is retrieval-augmented generation?"],
                ["How do vaccines work?"],
                ["What is quantum computing?"],
                ["Explain the concept of transfer learning"],
                ["What is climate change?"],
            ],
            inputs=[question_input],
        )

        # ── Wiring ──
        submit_btn.click(
            fn=query_pipeline,
            inputs=[question_input],
            outputs=[scores_output, docs_output, meta_output],
        )

        question_input.submit(
            fn=query_pipeline,
            inputs=[question_input],
            outputs=[scores_output, docs_output, meta_output],
        )

        clear_btn.click(
            fn=lambda: ("", "", "", ""),
            outputs=[question_input, scores_output, docs_output, meta_output],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch the Evidence-Aware RAG Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    print("Initializing pipeline...")
    initialize_retriever()

    print("Launching Gradio UI...")
    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
