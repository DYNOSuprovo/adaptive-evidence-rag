"""
Module 1: Evidence Independence Scorer

Detects whether retrieved documents are genuinely independent sources
or just repeating the same narrative.

Key Idea:
- Current RAG: "3 websites agree = strong evidence"
- Our approach: "Are these actually independent sources or copies?"

Example:
    Blog A copies Blog B
    Blog C copies Blog B
    Current system: "3 sources agree!"
    Our system: "Only 1 independent source exists"
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F


@dataclass
class IndependenceResult:
    """Result of independence analysis."""
    doc_id: int
    independence_score: float  # 0 to 1, higher = more independent
    cluster_id: int
    cluster_size: int
    is_duplicate: bool
    redundancy_score: float  # 0 to 1, higher = more redundant
    source_diversity: float  # Average distance to other clusters


class IndependenceScorer:
    """
    Scores the independence of retrieved evidence documents.
    
    Uses a combination of:
    1. Embedding similarity analysis
    2. NLI-based entailment detection
    3. Source clustering
    """
    
    def __init__(
        self,
        embedder_name: str = "BAAI/bge-large-en-v1.5",
        nli_model_name: str = "microsoft/deberta-v3-large",
        similarity_threshold: float = 0.92,
        nli_threshold: float = 0.85,
        device: str = None,
        use_nli: bool = True,
        batch_size: int = 16
    ):
        """
        Initialize the Independence Scorer.
        
        Args:
            embedder_name: HuggingFace model name for embeddings
            nli_model_name: HuggingFace model name for NLI
            similarity_threshold: Threshold for cosine similarity (above = duplicate)
            nli_threshold: Threshold for NLI entailment (above = entailment)
            device: torch device
            use_nli: Whether to use NLI model for verification
            batch_size: Batch size for encoding
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.similarity_threshold = similarity_threshold
        self.nli_threshold = nli_threshold
        self.use_nli = use_nli
        self.batch_size = batch_size
        
        # Load embedding model
        print(f"[IndependenceScorer] Loading embedder: {embedder_name}")
        self.embedder = SentenceTransformer(embedder_name, device=self.device)
        
        # Load NLI model for entailment detection
        if self.use_nli:
            print(f"[IndependenceScorer] Loading NLI model: {nli_model_name}")
            self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(
                nli_model_name,
                num_labels=3,  # entailment, neutral, contradiction
                ignore_mismatched_sizes=True
            ).to(self.device)
            self.nli_model.eval()
    
    def compute_embeddings(self, documents: List[str]) -> np.ndarray:
        """
        Compute embeddings for a list of documents.
        
        Args:
            documents: List of document texts
            
        Returns:
            Document embeddings (N x D)
        """
        embeddings = self.embedder.encode(
            documents,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.
        
        Args:
            embeddings: Document embeddings
            
        Returns:
            Similarity matrix (N x N)
        """
        return np.dot(embeddings, embeddings.T)
    
    def detect_nli_entailment(
        self, 
        premise: str, 
        hypothesis: str
    ) -> Tuple[str, float]:
        """
        Check if hypothesis is entailed by premise using NLI.
        
        Args:
            premise: Source text
            hypothesis: Text to check
            
        Returns:
            (label, confidence) where label is 'entailment', 'neutral', or 'contradiction'
        """
        if not self.use_nli:
            return 'neutral', 0.5
        
        inputs = self.nli_tokenizer(
            premise,
            hypothesis,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            
        # Resolve label mapping dynamically from model config if available
        if hasattr(self.nli_model, 'config') and hasattr(self.nli_model.config, 'id2label') and self.nli_model.config.id2label:
            label_map = {int(k): v.lower() for k, v in self.nli_model.config.id2label.items()}
        else:
            # Fallback (DeBERTa-MNLI labels: 0=contradiction, 1=neutral, 2=entailment)
            label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
            
        pred_idx = probs.argmax().item()
        pred_label = label_map.get(pred_idx, 'neutral')
        confidence = probs.max().item()
        
        return pred_label, confidence
    
    def cluster_documents(
        self, 
        embeddings: np.ndarray, 
        documents: List[str] = None
    ) -> List[List[int]]:
        """
        Cluster documents based on similarity.
        
        Args:
            embeddings: Document embeddings
            documents: Original documents (optional, for NLI verification)
            
        Returns:
            List of clusters (each cluster is list of indices)
        """
        n = len(embeddings)
        assigned = [-1] * n
        clusters = []
        current_cluster = 0
        
        for i in range(n):
            if assigned[i] != -1:
                continue
                
            # Start new cluster
            cluster = [i]
            assigned[i] = current_cluster
            
            for j in range(i + 1, n):
                if assigned[j] != -1:
                    continue
                
                # Check embedding similarity
                sim = np.dot(embeddings[i], embeddings[j])
                
                if sim > self.similarity_threshold:
                    # If NLI is enabled, verify with entailment
                    if self.use_nli and documents:
                        label, conf = self.detect_nli_entailment(documents[i], documents[j])
                        if label == 'entailment' and conf > self.nli_threshold:
                            cluster.append(j)
                            assigned[j] = current_cluster
                    else:
                        cluster.append(j)
                        assigned[j] = current_cluster
            
            clusters.append(cluster)
            current_cluster += 1
        
        return clusters
    
    def score_independence(
        self, 
        documents: List[str],
        embeddings: np.ndarray = None,
        source_urls: List[str] = None
    ) -> List[IndependenceResult]:
        """
        Main method: Score independence of each document.
        
        Args:
            documents: List of document texts
            embeddings: Pre-computed embeddings (optional)
            source_urls: Source URLs for domain-based analysis (optional)
            
        Returns:
            List of IndependenceResult objects
        """
        if embeddings is None:
            embeddings = self.compute_embeddings(documents)
        
        # Cluster documents
        clusters = self.cluster_documents(embeddings, documents)
        
        results = []
        for doc_idx, doc in enumerate(documents):
            # Find which cluster this doc belongs to
            cluster_id = None
            cluster_size = 0
            for cid, cluster in enumerate(clusters):
                if doc_idx in cluster:
                    cluster_id = cid
                    cluster_size = len(cluster)
                    break
            
            # Calculate independence score
            # 1 / cluster_size penalizes being in large clusters
            independence_score = 1.0 / max(cluster_size, 1)
            
            # Calculate redundancy score
            redundancy_score = (cluster_size - 1) / max(len(documents) - 1, 1)
            
            # Calculate source diversity (avg distance to other clusters)
            other_cluster_docs = []
            for cid, cluster in enumerate(clusters):
                if cid != cluster_id:
                    other_cluster_docs.extend(cluster)
            
            if other_cluster_docs:
                distances = []
                for other_idx in other_cluster_docs:
                    sim = np.dot(embeddings[doc_idx], embeddings[other_idx])
                    distances.append(1 - sim)
                source_diversity = np.mean(distances)
            else:
                source_diversity = 0.0
            
            # Is this a duplicate?
            is_duplicate = cluster_size > 1 and self._is_least_representative(
                doc_idx, cluster_id, clusters, embeddings
            )
            
            result = IndependenceResult(
                doc_id=doc_idx,
                independence_score=independence_score,
                cluster_id=cluster_id,
                cluster_size=cluster_size,
                is_duplicate=is_duplicate,
                redundancy_score=redundancy_score,
                source_diversity=source_diversity
            )
            results.append(result)
        
        return results
    
    def _is_least_representative(
        self, 
        doc_idx: int, 
        cluster_id: int, 
        clusters: List[List[int]], 
        embeddings: np.ndarray
    ) -> bool:
        """
        Check if a document is the least representative in its cluster.
        Used to mark duplicates.
        """
        cluster = clusters[cluster_id]
        if len(cluster) <= 1:
            return False
        
        # Compute cluster centroid
        cluster_embeddings = embeddings[cluster]
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Find index in cluster closest to the centroid (with tie-breaker choosing the first)
        dists = [np.linalg.norm(embeddings[i] - centroid) for i in cluster]
        best_idx_in_cluster = np.argmin(dists)
        representative_doc_idx = cluster[best_idx_in_cluster]
        
        # If this is not the chosen representative, it is a duplicate
        return doc_idx != representative_doc_idx
    
    def filter_redundant_documents(
        self,
        documents: List[str],
        results: List[IndependenceResult] = None,
        embeddings: np.ndarray = None,
        keep_ratio: float = 0.7
    ) -> Tuple[List[str], List[int]]:
        """
        Filter out redundant documents, keeping the most independent ones.
        
        Args:
            documents: List of documents
            results: Independence results (will compute if None)
            embeddings: Pre-computed embeddings
            keep_ratio: Ratio of documents to keep
            
        Returns:
            (filtered_documents, kept_indices)
        """
        if results is None:
            results = self.score_independence(documents, embeddings)
        
        # Exclude duplicates first to eliminate redundancy
        non_duplicate_indices = [i for i, r in enumerate(results) if not r.is_duplicate]
        
        # Sort non-duplicates by independence score (descending)
        sorted_results = sorted(
            [(i, results[i]) for i in non_duplicate_indices], 
            key=lambda x: x[1].independence_score, 
            reverse=True
        )
        
        # Keep top non-duplicate documents up to keep_ratio fraction
        num_keep = max(1, int(len(documents) * keep_ratio))
        kept_indices = [idx for idx, _ in sorted_results[:num_keep]]
        kept_indices.sort()  # Maintain original order
        
        filtered_docs = [documents[i] for i in kept_indices]
        
        return filtered_docs, kept_indices
    
    def get_aggregate_score(self, results: List[IndependenceResult]) -> Dict[str, float]:
        """
        Compute aggregate independence metrics.
        
        Returns:
            Dictionary of aggregate scores
        """
        scores = [r.independence_score for r in results]
        redundancy = [r.redundancy_score for r in results]
        diversity = [r.source_diversity for r in results]
        
        return {
            'mean_independence': np.mean(scores),
            'min_independence': np.min(scores),
            'mean_redundancy': np.mean(redundancy),
            'mean_diversity': np.mean(diversity),
            'num_clusters': len(set(r.cluster_id for r in results)),
            'duplicate_count': sum(1 for r in results if r.is_duplicate)
        }
