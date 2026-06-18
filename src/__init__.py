# Adaptive Evidence-Aware Retrieval for Robust RAG Systems

__version__ = "0.1.0"


def __getattr__(name):
    """Lazy imports — heavy ML modules load only when accessed."""
    _exports = {
        "IndependenceScorer": "src.evidence_independence",
        "UtilityScorer": "src.retrieval_utility",
        "SearchPolicyLearner": "src.search_policy",
        "StabilityChecker": "src.behavioral_stability",
        "EvidenceAwareRetriever": "src.retriever",
    }
    if name in _exports:
        import importlib
        module = importlib.import_module(_exports[name])
        return getattr(module, name)
    raise AttributeError(f"module 'src' has no attribute {name!r}")


__all__ = [
    "__version__",
    "IndependenceScorer",
    "UtilityScorer",
    "SearchPolicyLearner",
    "StabilityChecker",
    "EvidenceAwareRetriever",
]
