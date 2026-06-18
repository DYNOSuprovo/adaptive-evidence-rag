"""
Module 4: Behavioral Stability Checker

Tests whether the system produces consistent results across equivalent query variations.
If small wording changes completely change the conclusion, the system is unstable.

Key Idea:
- Good system: "Who invented Python?" and "Python creator?" -> Same answer
- Bad system: Different questions -> Different answers every time

Stability Formula:
    S = 0.4 * Answer_Similarity + 0.3 * Evidence_Overlap + 0.3 * Confidence_Consistency

This is one of the most publishable components because it measures robustness.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from scipy import stats


@dataclass
class StabilityResult:
    """Result of stability analysis."""
    original_question: str
    paraphrases: List[str]
    answers: List[str]
    answer_similarities: np.ndarray      # Pairwise answer similarity matrix
    evidence_lists: List[List[str]]      # Retrieved evidence for each variant
    evidence_overlaps: np.ndarray        # Pairwise evidence overlap matrix
    confidence_scores: List[float]
    stability_score: float               # 0 to 1, higher = more stable
    answer_consistency: float            # Average answer similarity
    evidence_overlap_score: float        # Average evidence overlap
    confidence_variance: float           # Lower is better
    is_stable: bool                      # Above threshold?


class StabilityChecker:
    """
    Checks behavioral stability across equivalent query variations.
    
    Process:
    1. Generate paraphrases of original question
    2. Run retrieval + answering for each
    3. Measure consistency across variants
    """
    
    def __init__(
        self,
        embedder_name: str = "BAAI/bge-large-en-v1.5",
        paraphrase_model_name: str = None,
        num_paraphrases: int = 10,
        answer_similarity_weight: float = 0.4,
        evidence_overlap_weight: float = 0.3,
        confidence_consistency_weight: float = 0.3,
        min_stability_score: float = 0.7,
        paraphrase_temperature: float = 0.8,
        device: str = None
    ):
        """
        Initialize Stability Checker.
        
        Args:
            embedder_name: Model for computing similarities
            paraphrase_model_name: Model for generating paraphrases
            num_paraphrases: Number of paraphrases to generate
            answer_similarity_weight: Weight for answer similarity
            evidence_overlap_weight: Weight for evidence overlap
            confidence_consistency_weight: Weight for confidence consistency
            min_stability_score: Threshold for stability
            paraphrase_temperature: Temperature for paraphrase generation
            device: torch device
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_paraphrases = num_paraphrases
        self.answer_similarity_weight = answer_similarity_weight
        self.evidence_overlap_weight = evidence_overlap_weight
        self.confidence_consistency_weight = confidence_consistency_weight
        self.min_stability_score = min_stability_score
        self.paraphrase_temperature = paraphrase_temperature
        
        # Load embedder
        print(f"[StabilityChecker] Loading embedder: {embedder_name}")
        self.embedder = SentenceTransformer(embedder_name, device=self.device)
        
        # Load paraphrase model
        if paraphrase_model_name:
            print(f"[StabilityChecker] Loading paraphrase model: {paraphrase_model_name}")
            self.para_tokenizer = AutoTokenizer.from_pretrained(paraphrase_model_name)
            self.para_model = AutoModelForCausalLM.from_pretrained(
                paraphrase_model_name,
                torch_dtype=torch.float16 if 'cuda' in self.device else torch.float32,
                device_map='auto' if 'cuda' in self.device else None
            )
        else:
            self.para_tokenizer = None
            self.para_model = None
    
    def generate_paraphrases(
        self, 
        question: str, 
        num_paraphrases: int = None
    ) -> List[str]:
        """
        Generate semantically equivalent paraphrases of a question.
        
        Args:
            question: Original question
            num_paraphrases: Number of paraphrases
            
        Returns:
            List of paraphrased questions
        """
        num_paraphrases = num_paraphrases or self.num_paraphrases
        
        if self.para_model is None:
            # Use template-based paraphrasing
            return self._generate_paraphrases_template(question, num_paraphrases)
        
        # Use LLM for paraphrasing
        prompt = f"""Generate {num_paraphrases} different ways to ask the following question. Each should have the same meaning but use different words and structure.

Original: {question}

Variations:
1."""
        
        inputs = self.para_tokenizer(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.para_model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=self.paraphrase_temperature,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.9
            )
        
        generated = self.para_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse paraphrases
        paraphrases = []
        lines = generated.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                para = line.lstrip('0123456789.- ').strip()
                # Remove quotes if present
                para = para.strip('"\'')
                if para and len(para) > 10 and para != question:
                    paraphrases.append(para)
        
        # If LLM failed, use templates
        if len(paraphrases) < num_paraphrases:
            templates = self._generate_paraphrases_template(question, num_paraphrases)
            existing = set(paraphrases)
            for t in templates:
                if t not in existing:
                    paraphrases.append(t)
        
        return paraphrases[:num_paraphrases]
    
    def _generate_paraphrases_template(
        self, 
        question: str, 
        num: int
    ) -> List[str]:
        """
        Generate paraphrases using predefined templates.
        """
        templates = [
            # Direct variations
            lambda q: q,
            lambda q: f"Can you tell me {q.lower()}?",
            lambda q: f"I need to know {q.lower()}.",
            lambda q: f"What is the answer to: {q}",
            
            # Structural changes
            lambda q: f"Tell me about {q.lower().replace('what is ', '').replace('who is ', '').replace('what are ', '')}",
            lambda q: f"Explain {q.lower().replace('what is ', '').replace('who is ', '').replace('what are ', '')}",
            lambda q: f"Could you explain {q.lower().replace('what is ', '').replace('who is ', '').replace('what are ', '')}?",
            
            # Add context
            lambda q: f"In your knowledge, {q.lower()}?",
            lambda q: f"Based on your understanding, {q.lower()}?",
            lambda q: f"I'm curious: {q}",
            
            # Question form changes
            lambda q: f"Do you know {q.lower()}?",
            lambda q: f"I'd like to understand {q.lower().replace('what is ', '').replace('who is ', '').replace('what are ', '')}",
        ]
        
        paraphrases = []
        for i in range(min(num, len(templates))):
            try:
                para = templates[i](question)
                if para and para != question:
                    paraphrases.append(para)
            except:
                continue
        
        return paraphrases
    
    def compute_answer_similarity(
        self, 
        answers: List[str]
    ) -> np.ndarray:
        """
        Compute pairwise similarity between answers.
        
        Args:
            answers: List of answer strings
            
        Returns:
            Similarity matrix (N x N)
        """
        if not answers:
            return np.array([[0]])
        
        embeddings = self.embedder.encode(answers, normalize_embeddings=True)
        
        # Cosine similarity matrix
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        return similarity_matrix
    
    def compute_evidence_overlap(
        self, 
        evidence_lists: List[List[str]]
    ) -> np.ndarray:
        """
        Compute pairwise evidence overlap using Jaccard similarity.
        
        Args:
            evidence_lists: List of evidence lists (one per query variant)
            
        Returns:
            Overlap matrix (N x N) with Jaccard similarities
        """
        n = len(evidence_lists)
        overlap_matrix = np.eye(n)  # Diagonal = 1
        
        for i in range(n):
            for j in range(i + 1, n):
                # Compute embedding-based overlap
                if not evidence_lists[i] or not evidence_lists[j]:
                    overlap = 0.0
                else:
                    emb_i = self.embedder.encode(evidence_lists[i], normalize_embeddings=True)
                    emb_j = self.embedder.encode(evidence_lists[j], normalize_embeddings=True)
                    
                    # Compute max similarity for each document in i to j
                    similarities = np.dot(emb_i, emb_j.T)
                    
                    # Count matches above threshold
                    threshold = 0.85
                    matches_i = np.sum(np.max(similarities, axis=1) > threshold)
                    matches_j = np.sum(np.max(similarities, axis=0) > threshold)
                    
                    # Jaccard-like overlap
                    total_unique = len(evidence_lists[i]) + len(evidence_lists[j])
                    overlap = (matches_i + matches_j) / total_unique if total_unique > 0 else 0
                
                overlap_matrix[i, j] = overlap
                overlap_matrix[j, i] = overlap
        
        return overlap_matrix
    
    def compute_confidence_consistency(
        self, 
        confidence_scores: List[float]
    ) -> float:
        """
        Measure consistency of confidence scores.
        
        Lower variance = higher consistency.
        
        Args:
            confidence_scores: List of confidence scores
            
        Returns:
            Consistency score (0 to 1)
        """
        if not confidence_scores or len(confidence_scores) < 2:
            return 1.0  # Perfect consistency with single score
        
        # Convert to numpy array
        scores = np.array(confidence_scores)
        
        # Calculate coefficient of variation (CV)
        mean = np.mean(scores)
        std = np.std(scores)
        
        if mean == 0:
            return 0.0
        
        cv = std / mean
        
        # Convert to consistency score (1 - normalized CV)
        # CV > 0.5 is considered unstable
        consistency = max(0, 1 - cv)
        
        return consistency
    
    def check_stability(
        self,
        question: str,
        answers: List[str] = None,
        evidence_lists: List[List[str]] = None,
        confidence_scores: List[float] = None,
        generate_paraphrases: bool = True,
        paraphrases: List[str] = None
    ) -> StabilityResult:
        """
        Main method: Check stability of the system for a given question.
        
        Args:
            question: Original question
            answers: Answers from different query variants (if already computed)
            evidence_lists: Retrieved evidence lists (if already computed)
            confidence_scores: Confidence scores (if already computed)
            generate_paraphrases: Whether to generate paraphrases
            paraphrases: Pre-generated paraphrases
            
        Returns:
            StabilityResult object
        """
        # Generate paraphrases if not provided
        if paraphrases is None and generate_paraphrases:
            paraphrases = self.generate_paraphrases(question)
        elif paraphrases is None:
            paraphrases = []
        
        all_questions = [question] + paraphrases
        
        # Placeholders if not provided
        if answers is None:
            answers = [""] * len(all_questions)
        if evidence_lists is None:
            evidence_lists = [[] for _ in all_questions]
        if confidence_scores is None:
            confidence_scores = [0.5] * len(all_questions)
        
        # Compute metrics
        answer_sim_matrix = self.compute_answer_similarity(answers)
        evidence_overlap_matrix = self.compute_evidence_overlap(evidence_lists)
        
        # Aggregate metrics
        # Average off-diagonal similarity
        n = len(answers)
        if n > 1:
            mask = ~np.eye(n, dtype=bool)
            answer_consistency = float(answer_sim_matrix[mask].mean())
            evidence_overlap_score = float(evidence_overlap_matrix[mask].mean())
        else:
            answer_consistency = 1.0
            evidence_overlap_score = 1.0
        
        confidence_variance = float(np.var(confidence_scores))
        confidence_consistency = self.compute_confidence_consistency(confidence_scores)
        
        # Weighted stability score
        stability_score = (
            self.answer_similarity_weight * answer_consistency +
            self.evidence_overlap_weight * evidence_overlap_score +
            self.confidence_consistency_weight * confidence_consistency
        )
        
        result = StabilityResult(
            original_question=question,
            paraphrases=paraphrases,
            answers=answers,
            answer_similarities=answer_sim_matrix,
            evidence_lists=evidence_lists,
            evidence_overlaps=evidence_overlap_matrix,
            confidence_scores=confidence_scores,
            stability_score=stability_score,
            answer_consistency=answer_consistency,
            evidence_overlap_score=evidence_overlap_score,
            confidence_variance=confidence_variance,
            is_stable=stability_score >= self.min_stability_score
        )
        
        return result
    
    def batch_check_stability(
        self,
        questions: List[str],
        answers_batch: List[List[str]] = None,
        evidence_batch: List[List[List[str]]] = None,
        confidence_batch: List[List[float]] = None
    ) -> List[StabilityResult]:
        """
        Check stability for multiple questions.
        
        Args:
            questions: List of questions
            answers_batch: Answers for each question (list of lists)
            evidence_batch: Evidence for each question
            confidence_batch: Confidence scores for each question
            
        Returns:
            List of StabilityResult objects
        """
        results = []
        
        for i, question in enumerate(questions):
            answers = answers_batch[i] if answers_batch else None
            evidence = evidence_batch[i] if evidence_batch else None
            confidences = confidence_batch[i] if confidence_batch else None
            
            result = self.check_stability(
                question=question,
                answers=answers,
                evidence_lists=evidence,
                confidence_scores=confidences
            )
            results.append(result)
        
        return results
    
    def get_stability_report(self, results: List[StabilityResult]) -> Dict[str, float]:
        """
        Generate aggregate stability report.
        
        Args:
            results: List of stability results
            
        Returns:
            Dictionary of aggregate metrics
        """
        if not results:
            return {}
        
        stability_scores = [r.stability_score for r in results]
        answer_consistencies = [r.answer_consistency for r in results]
        evidence_overlaps = [r.evidence_overlap_score for r in results]
        confidence_variances = [r.confidence_variance for r in results]
        
        stable_count = sum(1 for r in results if r.is_stable)
        
        return {
            'mean_stability': float(np.mean(stability_scores)),
            'min_stability': float(np.min(stability_scores)),
            'max_stability': float(np.max(stability_scores)),
            'stability_std': float(np.std(stability_scores)),
            'mean_answer_consistency': float(np.mean(answer_consistencies)),
            'mean_evidence_overlap': float(np.mean(evidence_overlaps)),
            'mean_confidence_variance': float(np.mean(confidence_variances)),
            'stable_ratio': stable_count / len(results),
            'total_questions': len(results)
        }
