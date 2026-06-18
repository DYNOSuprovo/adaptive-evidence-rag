"""
Adaptive Evidence-Aware RAG — Command-Line Interface

Provides subcommands for running queries, evaluation, training,
document indexing, and an interactive demo.

Usage:
    evidence-rag query "Who invented the transformer architecture?"
    evidence-rag evaluate --config configs/config.yaml
    evidence-rag train --epochs 5
    evidence-rag demo
    evidence-rag info
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Optional

import contextlib
try:
    import huggingface_hub.utils
    import huggingface_hub.file_download
    
    @contextlib.contextmanager
    def dummy_weak_file_lock(*args, **kwargs):
        yield None

    huggingface_hub.utils.WeakFileLock = dummy_weak_file_lock
    huggingface_hub.file_download.WeakFileLock = dummy_weak_file_lock
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Lazy imports — heavyweight ML libs are loaded only when a subcommand needs
# them so that ``evidence-rag --help`` stays instant.
# ---------------------------------------------------------------------------


def _get_project_root() -> Path:
    """Return the project root (directory containing pyproject.toml)."""
    current = Path(__file__).resolve().parent.parent
    if (current / "pyproject.toml").exists():
        return current
    # Fallback: current working directory
    return Path.cwd()


# ── Helpers ─────────────────────────────────────────────────────────────────

def _load_config(config_path: str) -> dict:
    """Load YAML config, falling back to project-root default."""
    from src.utils import load_config
    root = _get_project_root()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    return load_config(str(path))


def _coloured(text: str, colour: str) -> str:
    """Wrap *text* in ANSI colour codes if the terminal supports it."""
    colours = {
        "green": "\033[92m",
        "cyan": "\033[96m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "reset": "\033[0m",
    }
    try:
        import colorama
        colorama.init()
    except ImportError:
        pass

    if not sys.stdout.isatty():
        return text
    return f"{colours.get(colour, '')}{text}{colours['reset']}"


def _print_banner():
    """Print a nice project banner."""
    banner = r"""
    +==============================================================+
    |   Adaptive Evidence-Aware RAG System                         |
    |   Beyond Agreement Counting -- Truly Independent Evidence    |
    +==============================================================+
    """
    print(_coloured(banner, "cyan"))


# ── Subcommand: info ────────────────────────────────────────────────────────

def cmd_info(args: argparse.Namespace) -> None:
    """Show system & project information."""
    _print_banner()
    import torch
    from src import __version__

    config = _load_config(args.config)

    print(_coloured("System Information", "bold"))
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  CUDA avail  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Memory  : {mem:.1f} GB")
    print(f"  Package ver : {__version__}")
    print()

    print(_coloured("Configuration", "bold"))
    print(f"  Config file : {args.config}")
    print(f"  Embedder    : {config['models']['embedder']['name']}")
    print(f"  Reranker    : {config['models']['reranker']['name']}")
    print(f"  NLI model   : {config['models']['nli']['name']}")
    print(f"  Generator   : {config['models']['generator']['name']}")
    print(f"  Vector store: {config['retrieval']['vector_store']}")
    print()

    print(_coloured("Modules", "bold"))
    for module in ("independence", "utility", "search_policy", "stability"):
        enabled = config.get(module, {}).get("enabled", False)
        status = _coloured("[ON] enabled", "green") if enabled else _coloured("[--] disabled", "dim")
        print(f"  {module:<18s} {status}")


# ── Subcommand: query ──────────────────────────────────────────────────────

def cmd_query(args: argparse.Namespace) -> None:
    """Run a single question through the full evidence-aware pipeline."""
    _print_banner()
    import torch
    from src.retriever import EvidenceAwareRetriever

    config = _load_config(args.config)
    device = "cuda" if torch.cuda.is_available() and config.get("device", {}).get("use_cuda", True) else "cpu"

    print(_coloured(f"Device: {device}", "dim"))
    print(_coloured(f"Question: {args.question}\n", "bold"))

    # Build retriever with config-driven module flags
    retriever = EvidenceAwareRetriever(
        embedder_name=config["models"]["embedder"]["name"],
        reranker_name=config["models"]["reranker"]["name"],
        nli_model_name=config["models"]["nli"]["name"],
        independence_config=config.get("independence", {}),
        utility_config=config.get("utility", {}),
        search_policy_config=config.get("search_policy", {}),
        stability_config=config.get("stability", {}),
        device=device,
        top_k=config["retrieval"]["top_k"],
        use_independence=config.get("independence", {}).get("enabled", True),
        use_utility=config.get("utility", {}).get("enabled", True),
        use_search_policy=config.get("search_policy", {}).get("enabled", True),
        use_stability=args.check_stability,
    )

    # Index a demo corpus if no external corpus is provided
    if args.corpus:
        corpus_path = Path(args.corpus)
        if corpus_path.suffix == ".jsonl":
            docs = []
            with open(corpus_path) as fh:
                for line in fh:
                    obj = json.loads(line)
                    docs.append(obj.get("text", obj.get("content", str(obj))))
        elif corpus_path.suffix == ".json":
            with open(corpus_path) as fh:
                data = json.load(fh)
            docs = data if isinstance(data, list) else [str(d) for d in data.values()]
        elif corpus_path.suffix == ".txt":
            docs = corpus_path.read_text(encoding="utf-8").splitlines()
        else:
            print(_coloured(f"Unsupported corpus format: {corpus_path.suffix}", "red"))
            sys.exit(1)
        print(f"Loaded {len(docs)} documents from {corpus_path.name}")
    else:
        # Minimal built-in demo corpus
        docs = _demo_corpus()
        print(_coloured(f"Using built-in demo corpus ({len(docs)} documents)", "dim"))

    retriever.index_documents(docs)

    # Run pipeline
    result = retriever.run_pipeline(args.question, check_stability=args.check_stability)

    # Print results
    print("\n" + "=" * 60)
    print(_coloured("RESULTS", "bold"))
    print("=" * 60)
    print(f"  Query used          : {result.query_used}")
    print(f"  Original docs       : {len(result.original_documents)}")
    print(f"  Filtered docs       : {len(result.filtered_documents)}")
    print(f"  Independence score  : {result.independence_score:.4f}")
    print(f"  Utility score       : {result.utility_score:.4f}")
    print(f"  Stability score     : {result.stability_score:.4f}")
    print(f"  Overall quality     : {result.overall_quality:.4f}")

    print(f"\n{_coloured('Top Evidence:', 'bold')}")
    for i, doc in enumerate(result.filtered_documents[:5], 1):
        print(f"  [{i}] {doc[:120]}{'...' if len(doc) > 120 else ''}")

    # Optionally dump full JSON
    if args.output:
        out = {
            "question": result.question,
            "query_used": result.query_used,
            "independence_score": result.independence_score,
            "utility_score": result.utility_score,
            "stability_score": result.stability_score,
            "overall_quality": result.overall_quality,
            "filtered_documents": result.filtered_documents,
            "metadata": result.metadata,
        }
        with open(args.output, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nFull results saved to {args.output}")


# ── Subcommand: evaluate ───────────────────────────────────────────────────

def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation on benchmark datasets."""
    _print_banner()
    from src.evaluate import run_full_evaluation

    results = run_full_evaluation(
        config_path=args.config,
        output_path=args.output,
        max_samples_per_dataset=args.max_samples,
    )
    print(_coloured("\nEvaluation complete.", "green"))


# ── Subcommand: train ──────────────────────────────────────────────────────

def cmd_train(args: argparse.Namespace) -> None:
    """Train the search-policy reward model."""
    _print_banner()
    import torch
    from src.search_policy import SearchPolicyLearner, SearchEpisode

    config = _load_config(args.config)
    device = "cuda" if torch.cuda.is_available() and config.get("device", {}).get("use_cuda", True) else "cpu"

    print(_coloured(f"Device: {device}", "dim"))
    print(f"Training for {args.epochs} epochs...\n")

    learner = SearchPolicyLearner(
        embedder_name=config["models"]["embedder"]["name"],
        device=device,
        learning_mode=config.get("search_policy", {}).get("learning_algorithm", "reward_model"),
    )

    # If a pre-trained model checkpoint exists, load it
    model_dir = Path(args.model_dir)
    if (model_dir / "reward_model.pt").exists():
        print(f"Resuming from checkpoint in {model_dir}")
        learner.load_model(str(model_dir))

    # If episode data exists, load it
    if args.episodes:
        ep_path = Path(args.episodes)
        with open(ep_path) as fh:
            raw_episodes = json.load(fh)
        for ep_dict in raw_episodes:
            learner.store_episode(SearchEpisode(**ep_dict))
        print(f"Loaded {len(raw_episodes)} episodes from {ep_path.name}")
    elif not learner.episodes:
        # Generate synthetic training data as a starting point
        print(_coloured("No episodes provided -- generating synthetic training data...", "yellow"))
        _generate_synthetic_episodes(learner, config, device)

    losses = learner.train_reward_model(epochs=args.epochs)

    # Save model
    model_dir.mkdir(parents=True, exist_ok=True)
    learner.save_model(str(model_dir))
    print(_coloured(f"\nModel saved to {model_dir}", "green"))

    if losses:
        print(f"Final loss: {losses[-1]:.6f}")


def _generate_synthetic_episodes(learner, config, device):
    """Create simple synthetic episodes so the reward model has *something* to train on."""
    from src.search_policy import SearchEpisode
    import numpy as np

    questions = [
        "What is machine learning?",
        "Who invented the telephone?",
        "How does photosynthesis work?",
        "What is quantum computing?",
        "Who wrote Romeo and Juliet?",
        "What causes climate change?",
        "How do vaccines work?",
        "What is blockchain?",
        "Who was the first person on the moon?",
        "What is artificial intelligence?",
    ]

    for q in questions:
        # Good query variant
        learner.store_episode(SearchEpisode(
            question=q,
            query=f"{q} explained in detail",
            answer_correctness=0.8 + np.random.random() * 0.2,
            evidence_quality=0.7 + np.random.random() * 0.3,
            retrieval_efficiency=0.6 + np.random.random() * 0.4,
        ))
        # Mediocre query variant
        words = q.split()
        learner.store_episode(SearchEpisode(
            question=q,
            query=" ".join(words[:3]),
            answer_correctness=0.3 + np.random.random() * 0.3,
            evidence_quality=0.2 + np.random.random() * 0.3,
            retrieval_efficiency=0.2 + np.random.random() * 0.3,
        ))

    print(f"  Generated {len(learner.episodes)} synthetic episodes")


# ── Subcommand: index ──────────────────────────────────────────────────────

def cmd_index(args: argparse.Namespace) -> None:
    """Index a corpus of documents for later retrieval."""
    _print_banner()
    import torch
    from src.retriever import EvidenceAwareRetriever

    config = _load_config(args.config)
    device = "cuda" if torch.cuda.is_available() and config.get("device", {}).get("use_cuda", True) else "cpu"

    input_path = Path(args.input)
    if not input_path.exists():
        print(_coloured(f"File not found: {input_path}", "red"))
        sys.exit(1)

    # Load documents
    docs = []
    if input_path.suffix == ".jsonl":
        with open(input_path) as fh:
            for line in fh:
                obj = json.loads(line)
                docs.append(obj.get("text", obj.get("content", str(obj))))
    elif input_path.suffix == ".json":
        with open(input_path) as fh:
            data = json.load(fh)
        docs = data if isinstance(data, list) else [str(d) for d in data.values()]
    elif input_path.suffix == ".txt":
        docs = input_path.read_text(encoding="utf-8").splitlines()
        docs = [d.strip() for d in docs if d.strip()]
    else:
        print(_coloured(f"Unsupported format: {input_path.suffix}. Use .jsonl, .json, or .txt", "red"))
        sys.exit(1)

    print(f"Loaded {len(docs)} documents from {input_path.name}")

    retriever = EvidenceAwareRetriever(
        embedder_name=config["models"]["embedder"]["name"],
        reranker_name=config["models"]["reranker"]["name"],
        device=device,
        top_k=config["retrieval"]["top_k"],
        use_independence=False,
        use_utility=False,
        use_search_policy=False,
        use_stability=False,
    )

    retriever.index_documents(docs)
    print(_coloured(f"\nIndexed {len(docs)} documents successfully.", "green"))
    print("Note: In-memory index is ephemeral. Use the `query` subcommand with --corpus to query.")


# ── Subcommand: demo ───────────────────────────────────────────────────────

def cmd_demo(args: argparse.Namespace) -> None:
    """Launch an interactive REPL for querying the pipeline."""
    _print_banner()
    import torch
    from src.retriever import EvidenceAwareRetriever

    config = _load_config(args.config)
    device = "cuda" if torch.cuda.is_available() and config.get("device", {}).get("use_cuda", True) else "cpu"

    print(_coloured(f"Device: {device}", "dim"))
    print("Loading models -- this may take a moment...\n")

    retriever = EvidenceAwareRetriever(
        embedder_name=config["models"]["embedder"]["name"],
        reranker_name=config["models"]["reranker"]["name"],
        nli_model_name=config["models"]["nli"]["name"],
        independence_config=config.get("independence", {}),
        utility_config=config.get("utility", {}),
        search_policy_config=config.get("search_policy", {}),
        stability_config=config.get("stability", {}),
        device=device,
        top_k=config["retrieval"]["top_k"],
        use_independence=config.get("independence", {}).get("enabled", True),
        use_utility=config.get("utility", {}).get("enabled", True),
        use_search_policy=config.get("search_policy", {}).get("enabled", True),
        use_stability=False,  # too slow for interactive use
    )

    # Index demo corpus
    docs = _demo_corpus()
    retriever.index_documents(docs)
    print(_coloured(f"Indexed {len(docs)} demo documents.\n", "dim"))

    print(_coloured("Interactive Demo -- type a question and press Enter.", "bold"))
    print(_coloured("Commands: :quit :help :stability on/off\n", "dim"))

    check_stability = False

    while True:
        try:
            question = input(_coloured("> ", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in (":quit", ":q", ":exit", "exit", "quit"):
            print("Goodbye!")
            break
        if question.lower() == ":help":
            print("  Type a question to query the pipeline.")
            print("  :stability on/off -- toggle stability checking")
            print("  :quit -- exit")
            continue
        if question.lower().startswith(":stability"):
            toggle = question.split()[-1].lower()
            check_stability = toggle in ("on", "true", "1", "yes")
            print(f"  Stability checking: {'on' if check_stability else 'off'}")
            continue

        result = retriever.run_pipeline(question, check_stability=check_stability)

        print(f"\n  {'Independence':<20s}: {result.independence_score:.4f}")
        print(f"  {'Utility':<20s}: {result.utility_score:.4f}")
        print(f"  {'Stability':<20s}: {result.stability_score:.4f}")
        print(f"  {'Overall Quality':<20s}: {result.overall_quality:.4f}")
        print(f"  {'Docs (orig->filt)':<20s}: {len(result.original_documents)} -> {len(result.filtered_documents)}")

        if result.filtered_documents:
            print(f"\n  {_coloured('Top Evidence:', 'bold')}")
            for i, doc in enumerate(result.filtered_documents[:3], 1):
                print(f"    [{i}] {doc[:100]}{'...' if len(doc) > 100 else ''}")
        print()


# ── Demo corpus ─────────────────────────────────────────────────────────────

def _demo_corpus():
    """Return a small built-in corpus for demos and quick tests."""
    base = [
        "Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
        "Deep learning uses neural networks with multiple layers to model complex patterns in large datasets.",
        "Natural language processing allows computers to understand, interpret, and generate human language.",
        "Computer vision enables machines to interpret and make decisions based on visual data from the world.",
        "Reinforcement learning trains agents to make sequences of decisions through trial and error rewards.",
        "The transformer architecture was introduced in 2017 by Vaswani et al. in the paper 'Attention Is All You Need'.",
        "BERT (Bidirectional Encoder Representations from Transformers) revolutionized NLP through pre-training.",
        "GPT models are autoregressive language models that generate text by predicting the next token.",
        "Retrieval-augmented generation combines document retrieval with language model generation for factual answers.",
        "Evidence independence measures whether retrieved sources provide genuinely independent corroboration.",
        "Quantum computing uses quantum bits (qubits) that can exist in superposition of states simultaneously.",
        "Climate change refers to long-term shifts in global temperatures and weather patterns driven by human activities.",
        "Vaccines work by training the immune system to recognize and fight specific pathogens without causing disease.",
        "Blockchain is a distributed ledger technology that records transactions across a network of computers.",
        "The telephone was invented by Alexander Graham Bell, who patented it in 1876.",
        "William Shakespeare wrote Romeo and Juliet, believed to have been written between 1591 and 1596.",
        "Neil Armstrong was the first person to walk on the Moon on July 20, 1969 during the Apollo 11 mission.",
        "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
        "The theory of general relativity, published by Albert Einstein in 1915, describes gravity as spacetime curvature.",
    ]
    # Repeat for a larger corpus with slight variety
    return base * 50


# ── Argument Parser ─────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evidence-rag",
        description="Adaptive Evidence-Aware RAG System -- CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              evidence-rag info
              evidence-rag query "Who invented the transformer architecture?"
              evidence-rag evaluate --max-samples 100
              evidence-rag train --epochs 5
              evidence-rag demo
        """),
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to YAML config file (default: configs/config.yaml)",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── info ──
    sub.add_parser("info", help="Show system & project information")

    # ── query ──
    p_query = sub.add_parser("query", help="Run a single question through the pipeline")
    p_query.add_argument("question", help="The question to answer")
    p_query.add_argument("--corpus", default=None, help="Path to corpus file (.jsonl, .json, .txt)")
    p_query.add_argument("--output", "-o", default=None, help="Save full results to JSON file")
    p_query.add_argument("--check-stability", action="store_true", help="Enable stability checking (slow)")

    # ── evaluate ──
    p_eval = sub.add_parser("evaluate", help="Run evaluation on benchmark datasets")
    p_eval.add_argument("--output", "-o", default="logs/evaluation_results.json", help="Output file path")
    p_eval.add_argument("--max-samples", type=int, default=500, help="Max samples per dataset")

    # ── train ──
    p_train = sub.add_parser("train", help="Train the search-policy reward model")
    p_train.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    p_train.add_argument("--model-dir", default="models/search_policy", help="Directory to save/load model")
    p_train.add_argument("--episodes", default=None, help="Path to episodes JSON file")

    # ── index ──
    p_index = sub.add_parser("index", help="Index a corpus of documents")
    p_index.add_argument("--input", "-i", required=True, help="Input corpus file (.jsonl, .json, .txt)")

    # ── demo ──
    sub.add_parser("demo", help="Launch interactive question-answering demo")

    return parser


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "info": cmd_info,
        "query": cmd_query,
        "evaluate": cmd_evaluate,
        "train": cmd_train,
        "index": cmd_index,
        "demo": cmd_demo,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
