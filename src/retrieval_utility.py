"""
Module 2: Retrieval Utility Scorer

Rewards evidence that provides new information, confidence gains, or contradiction discovery.
Penalizes evidence that is redundant and adds no value.

Key Idea:
- Current RAG: "More documents = Better"
- Our approach: "More USEFUL documents = Better"

Utility Formula:
    U = 0.4 * Novelty + 0.4 * Confidence_Gain + 0.2 * Contradiction_Detection

Good Evidence:
    Gives new information
    Increases confidence
    Finds contradictions
    Helps make better decisions

Bad Evidence:
    Repeats same thing
    Adds no value
    Wastes context window
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_distances


@dataclass
class UtilityResult:
    """Result of utility analysis for a single document."""
    doc_id: int
    novelty: float              # 0 to 1, higher = more novel
    confidence_gain: float      # Change in answer confidence
    contradiction_detected: bool
    contradiction_score: float  # 0 to 1
    utility_score: float        # Weighted combination
    information_added: float    # Estimated new information
    is_useful: bool             # Above threshold?


@dataclass
class ConfidenceReading:
    """Store confidence before and after adding evidence."""
    before: float
    after: float
    question: str
    answer: str


class UtilityScorer:
    """
    Scores the utility of retrieved evidence documents.
    
    Uses a weighted combination of:
    1. Novelty - Does it contain new information?
    2. Confidence Gain - Does it increase answer confidence?
    3. Contradiction Detection - Does it reveal contradictions?
    """
    
    def __init__(
        self,
        embedder_name: str = "BAAI/bge-large-en-v1.5",
        nli_model_name: str = "microsoft/deberta-v3-large",
        generator_name: str = None,  # For confidence measurement
        novelty_weight: float = 0.4,
        confidence_gain_weight: float = 0.4,
        contradiction_weight: float = 0.2,
        min_utility_threshold: float = 0.3,
        device: str = None,
        batch_size: int = 16,
        use_generator_for_confidence: bool = False
    ):
        """
        Initialize the Utility Scorer.
        
        Args:
            embedder_name: Model for computing novelty embeddings
            nli_model_name: Model for contradiction detection
            generator_name: LLM for confidence measurement (optional)
            novelty_weight: Weight for novelty in utility formula
            confidence_gain_weight: Weight for confidence gain
            contradiction_weight: Weight for contradiction detection
            min_utility_threshold: Minimum score to consider useful
            device: torch device
            batch_size: Batch size for encoding
            use_generator_for_confidence: Whether to use LLM for confidence
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.novelty_weight = novelty_weight
        self.confidence_gain_weight = confidence_gain_weight
        self.contradiction_weight = contradiction_weight
        self.min_utility_threshold = min_utility_threshold
        self.batch_size = batch_size
        self.use_generator_for_confidence = use_generator_for_confidence
        
        # Load embedding model for novelty
        print(f"[UtilityScorer] Loading embedder: {embedder_name}")
        self.embedder = SentenceTransformer(embedder_name, device=self.device)
        
        # Load NLI model for contradiction detection
        print(f"[UtilityScorer] Loading NLI model: {nli_model_name}")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            nli_model_name,
            num_labels=3,
            ignore_mismatched_sizes=True
        ).to(self.device)
        self.nli_model.eval()
        
        # Optional: Load generator for confidence measurement
        if self.use_generator_for_confidence and generator_name:
            print(f"[UtilityScorer] Loading generator: {generator_name}")
            self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_name)
            self.generator = AutoModelForCausalLM.from_pretrained(
                generator_name,
                torch_dtype=torch.float16 if 'cuda' in self.device else torch.float32,
                device_map='auto' if 'cuda' in self.device else None
            )
            self.generator.eval()
    
    def compute_novelty(
        self, 
        document: str, 
        existing_documents: List[str]
    ) -> float:
        """
        Compute novelty of a document relative to existing documents.
        
        Args:
            document: New document to evaluate
            existing_documents: Already seen documents
            
        Returns:
            Novelty score (0 to 1, higher = more novel)
        """
        if not existing_documents:
            return 1.0  # First document is maximally novel
        
        # Compute embeddings
        doc_embedding = self.embedder.encode(
            document, 
            normalize_embeddings=True, 
            convert_to_numpy=True
        )
        
        existing_embeddings = self.embedder.encode(
            existing_documents,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Compute distances to all existing documents
        distances = cosine_distances([doc_embedding], existing_embeddings)[0]
        
        # Novelty = max distance (most different from existing)
        novelty = float(np.max(distances))
        
        return novelty
    
    def compute_novelty_batch(
        self, 
        documents: List[str]
    ) -> List[float]:
        """
        Compute novelty for all documents in a batch.
        Novelty is calculated relative to all previous documents.
        
        Args:
            documents: List of documents in retrieval order
            
        Returns:
            List of novelty scores
        """
        if not documents:
            return []
        
        embeddings = self.embedder.encode(
            documents,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        novelty_scores = []
        for i in range(len(documents)):
            if i == 0:
                novelty_scores.append(1.0)
            else:
                # Distance to all previous documents
                distances = cosine_distances([embeddings[i]], embeddings[:i])[0]
                novelty = float(np.max(distances))
                novelty_scores.append(novelty)
        
        return novelty_scores
    
    def detect_contradiction(
        self, 
        evidence1: str, 
        evidence2: str
    ) -> Tuple[bool, float]:
        """
        Detect contradiction between two pieces of evidence.
        
        Args:
            evidence1: First evidence
            evidence2: Second evidence
            
        Returns:
            (is_contradiction, confidence)
        """
        # Use NLI to check for contradiction
        inputs = self.nli_tokenizer(
            evidence1,
            evidence2,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
        
        # Resolve index for contradiction dynamically from model config
        contradiction_idx = 0
        if hasattr(self.nli_model, 'config') and hasattr(self.nli_model.config, 'id2label') and self.nli_model.config.id2label:
            for idx, label in self.nli_model.config.id2label.items():
                if label.lower() == 'contradiction':
                    contradiction_idx = int(idx)
                    break
                    
        contradiction_prob = probs[0][contradiction_idx].item()
        is_contradiction = contradiction_prob > 0.5
        
        return is_contradiction, contradiction_prob
    
    def detect_contradictions_batch(
        self, 
        documents: List[str]
    ) -> List[Tuple[int, int, float]]:
        """
        Detect all contradictions in a batch of documents.
        
        Returns:
            List of (idx1, idx2, contradiction_score) tuples
        """
        contradictions = []
        
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                is_contra, score = self.detect_contradiction(documents[i], documents[j])
                if is_contra:
                    contradictions.append((i, j, score))
        
        return contradictions
    
    def measure_confidence(
        self,
        question: str,
        evidence: List[str],
        use_generator: bool = False
    ) -> float:
        """
        Measure confidence in the answer given evidence.
        
        Simple version: Use NLI self-consistency
        Advanced version: Use generator token probabilities
        
        Args:
            question: The question being answered
            evidence: List of evidence documents
            use_generator: Whether to use LLM for confidence
            
        Returns:
            Confidence score (0 to 1)
        """
        if not evidence:
            return 0.0
        
        if use_generator and self.use_generator_for_confidence:
            return self._measure_confidence_with_llm(question, evidence)
        else:
            return self._measure_confidence_with_consistency(evidence)
    
    def _measure_confidence_with_consistency(self, evidence: List[str]) -> float:
        """
        Measure confidence by checking consistency of evidence.
        High consistency = high confidence.
        """
        if len(evidence) <= 1:
            return 0.5  # Neutral confidence for single evidence
        
        # Compute pairwise entailment scores
        consistent_count = 0
        total_pairs = 0
        
        for i in range(len(evidence)):
            for j in range(i + 1, len(evidence)):
                inputs = self.nli_tokenizer(
                    evidence[i],
                    evidence[j],
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.nli_model(**inputs)
                    probs = F.softmax(outputs.logits, dim=-1)
                
                # entailment prob
                entail_prob = probs[0][2].item()
                if entail_prob > 0.5:
                    consistent_count += 1
                total_pairs += 1
        
        confidence = consistent_count / total_pairs if total_pairs > 0 else 0.5
        return confidence
    
    def _measure_confidence_with_llm(
        self, 
        question: str, 
        evidence: List[str]
    ) -> float:
        """
        Measure confidence using LLM token probabilities.
        Requires generator model.
        """
        if not self.use_generator_for_confidence:
            return self._measure_confidence_with_consistency(evidence)
        
        # Format prompt with evidence
        evidence_text = "\n\n".join([f"Evidence {i+1}: {e}" for i, e in enumerate(evidence)])
        prompt = f"Answer the following question based on the evidence.\n\nQuestion: {question}\n\n{evidence_text}\n\nAnswer:"
        
        inputs = self.generator_tokenizer(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.generator(**inputs)
            # Use average log probability as confidence proxy
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            # Average max probability across tokens
            confidence = probs.max(dim=-1).values.mean().item()
        
        return confidence
    
    def compute_confidence_gain(
        self,
        question: str,
        evidence_before: List[str],
        new_evidence: str,
        use_generator: bool = False
    ) -> float:
        """
        Measure how much confidence improves by adding new evidence.
        
        Args:
            question: The question
            evidence_before: Evidence before adding new document
            new_evidence: New evidence document
            use_generator: Use LLM for confidence
            
        Returns:
            Confidence gain (can be negative)
        """
        conf_before = self.measure_confidence(question, evidence_before, use_generator)
        conf_after = self.measure_confidence(
            question, 
            evidence_before + [new_evidence], 
            use_generator
        )
        
        gain = conf_after - conf_before
        return gain
    
    def score_utility(
        self,
        question: str,
        documents: List[str],
        measure_confidence: bool = True
    ) -> List[UtilityResult]:
        """
        Main method: Compute utility score for each document.
        
        Args:
            question: User question
            documents: Retrieved documents
            measure_confidence: Whether to measure confidence gain
            
        Returns:
            List of UtilityResult objects
        """
        if not documents:
            return []
        
        # Compute novelty for all documents
        novelty_scores = self.compute_novelty_batch(documents)
        
        # Detect contradictions
        contradictions = self.detect_contradictions_batch(documents)
        contra_dict = {}
        for i, j, score in contradictions:
            contra_dict[i] = max(contra_dict.get(i, 0), score)
            contra_dict[j] = max(contra_dict.get(j, 0), score)
        
        # Compute confidence gains
        confidence_gains = []
        if measure_confidence:
            current_evidence = []
            for i, doc in enumerate(documents):
                gain = self.compute_confidence_gain(question, current_evidence, doc)
                confidence_gains.append(gain)
                current_evidence.append(doc)
        else:
            confidence_gains = [0.0] * len(documents)
        
        # Combine into utility scores
        results = []
        for i in range(len(documents)):
            novelty = novelty_scores[i]
            conf_gain = max(0, confidence_gains[i])  # Clip negative gains
            contra_score = contra_dict.get(i, 0.0)
            contra_detected = contra_score > 0.5
            
            # Weighted utility
            utility = (
                self.novelty_weight * novelty +
                self.confidence_gain_weight * conf_gain +
                self.contradiction_weight * contra_score
            )
            
            # Information added estimate
            info_added = novelty * (1 + conf_gain)
            
            result = UtilityResult(
                doc_id=i,
                novelty=novelty,
                confidence_gain=conf_gain,
                contradiction_detected=contra_detected,
                contradiction_score=contra_score,
                utility_score=utility,
                information_added=info_added,
                is_useful=utility >= self.min_utility_threshold
            )
            results.append(result)
        
        return results
    
    def filter_by_utility(
        self,
        documents: List[str],
        results: List[UtilityResult] = None,
        question: str = None,
        top_k: int = None,
        threshold: float = None
    ) -> Tuple[List[str], List[int]]:
        """
        Filter documents to keep only the most useful ones.
        
        Args:
            documents: All documents
            results: Pre-computed utility results
            question: Question (for computing results if not provided)
            top_k: Keep top K by utility
            threshold: Keep documents above threshold
            
        Returns:
            (filtered_documents, kept_indices)
        """
        if results is None:
            if question is None:
                raise ValueError("Either results or question must be provided")
            results = self.score_utility(question, documents)
        
        # Sort by utility score
        indexed_results = list(enumerate(results))
        indexed_results.sort(key=lambda x: x[1].utility_score, reverse=True)
        
        # Apply filter
        if top_k:
            indexed_results = indexed_results[:top_k]
        elif threshold:
            indexed_results = [(i, r) for i, r in indexed_results if r.utility_score >= threshold]
        
        kept_indices = [i for i, _ in indexed_results]
        kept_indices.sort()  # Maintain original order
        
        filtered_docs = [documents[i] for i in kept_indices]
        
        return filtered_docs, kept_indices
    
    def get_aggregate_metrics(self, results: List[UtilityResult]) -> Dict[str, float]:
        """
        Compute aggregate utility metrics.
        
        Returns:
            Dictionary of aggregate metrics
        """
        if not results:
            return {}
        
        return {
            'mean_utility': np.mean([r.utility_score for r in results]),
            'mean_novelty': np.mean([r.novelty for r in results]),
            'mean_confidence_gain': np.mean([r.confidence_gain for r in results]),
            'contradiction_rate': np.mean([1 if r.contradiction_detected else 0 for r in results]),
            'useful_doc_ratio': np.mean([1 if r.is_useful else 0 for r in results]),
            'total_information': np.sum([r.information_added for r in results])
        }
