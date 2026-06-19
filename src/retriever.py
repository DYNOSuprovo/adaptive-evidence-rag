"""
Core Retriever Module

Integrates all 4 modules into a single retrieval pipeline:
1. Retrieve documents using embedding search
2. Score independence (Module 1)
3. Score utility (Module 2)
4. Apply search policy (Module 3)
5. Check stability (Module 4)
6. Return verified, high-quality evidence
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.evidence_independence import IndependenceScorer, IndependenceResult
from src.retrieval_utility import UtilityScorer, UtilityResult
from src.search_policy import SearchPolicyLearner, QueryVariant
from src.behavioral_stability import StabilityChecker, StabilityResult


@dataclass
class RetrievalResult:
    """Complete result from the evidence-aware retrieval pipeline."""
    question: str
    query_used: str
    original_documents: List[str]
    filtered_documents: List[str]
    independence_results: List[IndependenceResult]
    utility_results: List[UtilityResult]
    stability_result: Optional[StabilityResult]
    independence_score: float
    utility_score: float
    stability_score: float
    overall_quality: float
    metadata: Dict[str, Any]


class EvidenceAwareRetriever:
    """
    Main retriever that integrates all 4 modules.
    
    Pipeline:
    1. Generate/Select query (using search policy)
    2. Retrieve documents
    3. Score independence
    4. Score utility
    5. Filter documents
    6. Check stability (optional)
    """
    
    def __init__(
        self,
        embedder_name: str = "BAAI/bge-large-en-v1.5",
        reranker_name: str = "BAAI/bge-reranker-v2-m3",
        nli_model_name: str = "microsoft/deberta-v3-large",
        independence_config: Dict = None,
        utility_config: Dict = None,
        search_policy_config: Dict = None,
        stability_config: Dict = None,
        device: str = None,
        top_k: int = 10,
        use_independence: bool = True,
        use_utility: bool = True,
        use_search_policy: bool = True,
        use_stability: bool = False  # Expensive, off by default
    ):
        """
        Initialize the Evidence-Aware Retriever.
        
        Args:
            embedder_name: Embedding model
            reranker_name: Reranking model
            nli_model_name: NLI model for entailment/contradiction checks
            independence_config: Config for Module 1
            utility_config: Config for Module 2
            search_policy_config: Config for Module 3
            stability_config: Config for Module 4
            device: torch device
            top_k: Number of documents to retrieve
            use_*: Whether to enable each module
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.top_k = top_k
        self.use_independence = use_independence
        self.use_utility = use_utility
        self.use_search_policy = use_search_policy
        self.use_stability = use_stability
        
        # Load embedder
        print(f"[Retriever] Loading embedder: {embedder_name}")
        self.embedder = SentenceTransformer(embedder_name, device=self.device)
        
        # Load reranker
        if reranker_name:
            print(f"[Retriever] Loading reranker: {reranker_name}")
            self.reranker = CrossEncoder(reranker_name, device=self.device)
        else:
            self.reranker = None
        
        # Initialize modules
        import inspect

        if use_independence:
            print("[Retriever] Initializing Independence Scorer")
            independence_config = independence_config or {}
            # Propagate NLI model name if not explicitly set
            independence_cfg = independence_config.copy()
            if "nli_model_name" not in independence_cfg:
                independence_cfg["nli_model_name"] = nli_model_name
            # Filter config to only pass keys accepted by IndependenceScorer.__init__
            valid_keys = inspect.signature(IndependenceScorer.__init__).parameters.keys()
            filtered_config = {k: v for k, v in independence_cfg.items() if k in valid_keys}
            self.independence_scorer = IndependenceScorer(
                embedder_name=embedder_name,
                device=self.device,
                **filtered_config
            )
        
        if use_utility:
            print("[Retriever] Initializing Utility Scorer")
            utility_config = utility_config or {}
            utility_cfg = utility_config.copy()
            if "nli_model_name" not in utility_cfg:
                utility_cfg["nli_model_name"] = nli_model_name
            valid_keys = inspect.signature(UtilityScorer.__init__).parameters.keys()
            filtered_config = {k: v for k, v in utility_cfg.items() if k in valid_keys}
            self.utility_scorer = UtilityScorer(
                embedder_name=embedder_name,
                device=self.device,
                **filtered_config
            )
        
        if use_search_policy:
            print("[Retriever] Initializing Search Policy Learner")
            search_policy_config = search_policy_config or {}
            valid_keys = inspect.signature(SearchPolicyLearner.__init__).parameters.keys()
            filtered_config = {k: v for k, v in search_policy_config.items() if k in valid_keys}
            self.search_policy = SearchPolicyLearner(
                embedder_name=embedder_name,
                device=self.device,
                **filtered_config
            )
        
        if use_stability:
            print("[Retriever] Initializing Stability Checker")
            stability_config = stability_config or {}
            valid_keys = inspect.signature(StabilityChecker.__init__).parameters.keys()
            filtered_config = {k: v for k, v in stability_config.items() if k in valid_keys}
            self.stability_checker = StabilityChecker(
                embedder_name=embedder_name,
                device=self.device,
                **filtered_config
            )
        
        # Document store (in-memory for now)
        self.documents: List[str] = []
        self.document_embeddings: np.ndarray = None
    
    def index_documents(self, documents: List[str]):
        """
        Index documents for retrieval.
        
        Args:
            documents: List of documents to index
        """
        print(f"[Retriever] Indexing {len(documents)} documents")
        self.documents = documents
        
        # Compute embeddings in batches
        self.document_embeddings = self.embedder.encode(
            documents,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        print(f"[Retriever] Indexed documents with embedding shape: {self.document_embeddings.shape}")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = None,
        filter_score: float = 0.0
    ) -> List[Tuple[int, float, str]]:
        """
        Basic retrieval using embedding similarity.
        
        Args:
            query: Search query
            top_k: Number of results
            filter_score: Minimum similarity score
            
        Returns:
            List of (doc_index, score, text) tuples
        """
        top_k = top_k or self.top_k
        
        # If no static documents are loaded, use Live Wikipedia!
        if getattr(self, 'document_embeddings', None) is None or len(self.documents) == 0:
            print(f"[Retriever] Dynamic Wikipedia Search for: {query}")
            import wikipedia
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')
            wikipedia.set_user_agent("AdaptiveEvidenceRAG/1.0 (test@example.com)")
            
            try:
                search_results = wikipedia.search(query, results=top_k)
                wiki_docs = []
                for title in search_results:
                    try:
                        # Get a clean summary
                        summary = wikipedia.summary(title, sentences=6, auto_suggest=False)
                        if summary and len(summary) > 50:
                            wiki_docs.append(summary)
                    except:
                        continue
                
                if wiki_docs:
                    # Dynamically encode and score them just for this query
                    wiki_embeddings = self.embedder.encode(wiki_docs, normalize_embeddings=True, convert_to_numpy=True)
                    query_embedding = self.embedder.encode(query, normalize_embeddings=True, convert_to_numpy=True)
                    similarities = np.dot(wiki_embeddings, query_embedding)
                    
                    # Sort and return
                    results = []
                    for idx in np.argsort(similarities)[::-1]:
                        if similarities[idx] >= filter_score:
                            results.append((int(idx), float(similarities[idx]), wiki_docs[idx]))
                    return results[:top_k]
            except Exception as e:
                print(f"[Retriever] Wikipedia search failed: {e}")
                
            return []
        
        # Original static logic
        # Encode query
        query_embedding = self.embedder.encode(
            query, 
            normalize_embeddings=True, 
            convert_to_numpy=True
        )
        
        # Compute similarities
        similarities = np.dot(self.document_embeddings, query_embedding)
        
        # Filter by score
        valid_indices = np.where(similarities >= filter_score)[0]
        
        # Get top-k
        top_indices = valid_indices[np.argsort(similarities[valid_indices])[-top_k:][::-1]]
        
        results = []
        for idx in top_indices:
            results.append((int(idx), float(similarities[idx]), self.documents[idx]))
        
        return results
    
    def rerank(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: Query string
            documents: Documents to rerank
            
        Returns:
            List of (original_index, score) sorted by score
        """
        if self.reranker is None or not documents:
            return list(enumerate([0.5] * len(documents)))
        
        pairs = [[query, doc] for doc in documents]
        scores = self.reranker.predict(pairs)
        
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        return indexed_scores
    
    def retrieve_with_policy(
        self, 
        question: str,
        num_variants: int = 3
    ) -> Tuple[str, List[Tuple[int, float, str]]]:
        """
        Retrieve using learned search policy to select best query.
        
        Args:
            question: User question
            num_variants: Number of query variants to try
            
        Returns:
            (best_query, retrieval_results)
        """
        if not self.use_search_policy:
            results = self.retrieve(question)
            return question, results
        
        # Generate query variants
        variants = self.search_policy.generate_query_variants(question, num_variants)
        
        # Try each variant
        all_results = []
        for variant in variants:
            results = self.retrieve(variant.query, top_k=self.top_k // 2)
            all_results.extend(results)
        
        # Also try original question
        orig_results = self.retrieve(question, top_k=self.top_k)
        all_results.extend(orig_results)
        
        # Deduplicate by document index
        seen_indices = set()
        unique_results = []
        for idx, score, text in all_results:
            if idx not in seen_indices:
                seen_indices.add(idx)
                unique_results.append((idx, score, text))
        
        # Sort by score
        unique_results.sort(key=lambda x: x[1], reverse=True)
        
        # Select best query (for tracking)
        best_variant = self.search_policy.select_best_query(question, variants)
        
        return best_variant.query, unique_results[:self.top_k]
    
    def run_pipeline(
        self, 
        question: str,
        check_stability: bool = None
    ) -> RetrievalResult:
        """
        Run the full evidence-aware retrieval pipeline.
        
        Args:
            question: User question
            check_stability: Whether to check stability (overrides default)
            
        Returns:
            RetrievalResult with all scores and filtered documents
        """
        check_stability = check_stability if check_stability is not None else self.use_stability
        
        # Step 1: Retrieve with search policy
        if self.use_search_policy:
            query_used, retrieved = self.retrieve_with_policy(question)
        else:
            query_used = question
            retrieved = self.retrieve(question)
        
        doc_indices = [r[0] for r in retrieved]
        doc_scores = [r[1] for r in retrieved]
        documents = [r[2] for r in retrieved]
        
        if not documents:
            return RetrievalResult(
                question=question,
                query_used=query_used,
                original_documents=[],
                filtered_documents=[],
                independence_results=[],
                utility_results=[],
                stability_result=None,
                independence_score=0,
                utility_score=0,
                stability_score=0,
                overall_quality=0,
                metadata={}
            )
        
        # Step 2: Score independence
        independence_results = None
        independence_aggregate = {}
        if self.use_independence:
            doc_embeddings = self.document_embeddings[doc_indices]
            independence_results = self.independence_scorer.score_independence(
                documents, 
                embeddings=doc_embeddings
            )
            independence_aggregate = self.independence_scorer.get_aggregate_score(
                independence_results
            )
            
            # Filter redundant documents
            documents, keep_indices = self.independence_scorer.filter_redundant_documents(
                documents,
                results=independence_results
            )
            independence_results = [independence_results[i] for i in keep_indices]
            doc_embeddings = doc_embeddings[keep_indices]
        
        # Step 3: Score utility
        utility_results = None
        utility_aggregate = {}
        if self.use_utility:
            utility_results = self.utility_scorer.score_utility(
                question,
                documents
            )
            utility_aggregate = self.utility_scorer.get_aggregate_metrics(utility_results)
            
            # Filter low-utility documents
            documents, keep_indices = self.utility_scorer.filter_by_utility(
                documents,
                results=utility_results,
                top_k=min(5, len(documents))
            )
            utility_results = [utility_results[i] for i in keep_indices]
        
        # Step 4: Check stability (expensive)
        stability_result = None
        if check_stability:
            stability_result = self.stability_checker.check_stability(
                question=question,
                paraphrases=[]  # Will be generated
            )
        
        # Compute aggregate scores
        independence_score = independence_aggregate.get('mean_independence', 0.5)
        utility_score = utility_aggregate.get('mean_utility', 0.5)
        stability_score = stability_result.stability_score if stability_result else 0.5
        
        # Overall quality score
        overall_quality = (
            0.35 * independence_score +
            0.35 * utility_score +
            0.30 * stability_score
        )
        
        result = RetrievalResult(
            question=question,
            query_used=query_used,
            original_documents=[r[2] for r in retrieved],
            filtered_documents=documents,
            independence_results=independence_results or [],
            utility_results=utility_results or [],
            stability_result=stability_result,
            independence_score=independence_score,
            utility_score=utility_score,
            stability_score=stability_score,
            overall_quality=overall_quality,
            metadata={
                'num_original': len(retrieved),
                'num_filtered': len(documents),
                'independence_aggregate': independence_aggregate,
                'utility_aggregate': utility_aggregate
            }
        )
        
        return result
    
    def batch_retrieve(
        self, 
        questions: List[str],
        check_stability: bool = None
    ) -> List[RetrievalResult]:
        """
        Run pipeline on multiple questions.
        
        Args:
            questions: List of questions
            check_stability: Whether to check stability
            
        Returns:
            List of RetrievalResult
        """
        results = []
        for i, question in enumerate(questions):
            print(f"[Retriever] Processing question {i+1}/{len(questions)}: {question[:50]}...")
            result = self.run_pipeline(question, check_stability)
            results.append(result)
        
        return results
    
    def get_pipeline_stats(self, results: List[RetrievalResult]) -> Dict[str, float]:
        """
        Get aggregate statistics from multiple retrieval results.
        """
        if not results:
            return {}
        
        return {
            'mean_independence': np.mean([r.independence_score for r in results]),
            'mean_utility': np.mean([r.utility_score for r in results]),
            'mean_stability': np.mean([r.stability_score for r in results]),
            'mean_overall_quality': np.mean([r.overall_quality for r in results]),
            'avg_compression': np.mean([
                len(r.filtered_documents) / max(len(r.original_documents), 1) 
                for r in results
            ]),
            'total_questions': len(results)
        }
