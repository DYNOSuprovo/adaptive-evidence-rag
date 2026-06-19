"""
Module 3: Search Policy Learning

Learns which search strategies consistently produce high-quality evidence.
Instead of treating retrieval as one-time random search, we learn from history.

Key Idea:
- Current RAG: Question -> Random search -> Results
- Our approach: Question -> Learned search strategy -> Better results

Learning Process:
    Question + Query Strategy + Evidence Found + Final Outcome
    -> Learn which strategies produce best evidence

Example:
    Query 1: "AI paper" -> Bad results
    Query 2: "AI paper benchmark results methodology" -> Excellent results
    System learns: Second style works better
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import json
import os
from collections import defaultdict


@dataclass
class SearchEpisode:
    """Single search episode for learning."""
    question: str
    query: str
    query_embedding: np.ndarray = None
    documents: List[str] = None
    doc_scores: List[float] = None
    answer_correctness: float = 0.0  # Did we get the right answer?
    evidence_quality: float = 0.0    # Quality of retrieved evidence
    retrieval_efficiency: float = 0.0  # Good evidence / Total retrieved
    reward: float = 0.0


@dataclass
class QueryVariant:
    """A generated query variant."""
    query: str
    score: float = 0.0
    embedding: np.ndarray = None


class QueryRewardModel(nn.Module):
    """
    Neural network that predicts the quality of a search query.
    Input: Query embedding + Question embedding
    Output: Expected reward (quality score)
    """
    
    def __init__(self, embedding_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, question_emb: torch.Tensor, query_emb: torch.Tensor) -> torch.Tensor:
        """
        Predict reward for a query given a question.
        
        Args:
            question_emb: Question embedding (batch, embed_dim)
            query_emb: Query embedding (batch, embed_dim)
            
        Returns:
            Predicted reward (batch, 1)
        """
        x = torch.cat([question_emb, query_emb], dim=-1)
        return self.network(x)


class SearchPolicyLearner:
    """
    Learns optimal search policies from retrieval history.
    
    Two modes:
    1. Simple: Store history and rank query patterns
    2. Advanced: Train a neural reward model to predict query quality
    """
    
    def __init__(
        self,
        embedder_name: str = "BAAI/bge-large-en-v1.5",
        query_generator_name: str = None,
        learning_mode: str = "reward_model",  # 'simple', 'reward_model', 'rl'
        num_variants: int = 5,
        reward_weights: Dict[str, float] = None,
        device: str = None,
        lr: float = 1e-4,
        batch_size: int = 16
    ):
        """
        Initialize Search Policy Learner.
        
        Args:
            embedder_name: Model for computing embeddings
            query_generator_name: LLM for generating query variants
            learning_mode: 'simple', 'reward_model', or 'rl'
            num_variants: Number of query variants to generate
            reward_weights: Weights for reward computation
            device: torch device
            lr: Learning rate for reward model
            batch_size: Batch size for training
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_mode = learning_mode
        self.num_variants = num_variants
        self.lr = lr
        self.batch_size = batch_size
        
        self.reward_weights = reward_weights or {
            'answer_correctness': 0.5,
            'evidence_quality': 0.3,
            'retrieval_efficiency': 0.2
        }
        
        # Load embedder
        print(f"[SearchPolicy] Loading embedder: {embedder_name}")
        self.embedder = SentenceTransformer(embedder_name, device=self.device)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Initialize reward model (for advanced mode)
        if learning_mode in ['reward_model', 'rl']:
            print("[SearchPolicy] Initializing reward model")
            self.reward_model = QueryRewardModel(
                embedding_dim=self.embedding_dim
            ).to(self.device)
            self.optimizer = optim.Adam(self.reward_model.parameters(), lr=lr)
            self.criterion = nn.MSELoss()
        
        # Initialize query generator
        if query_generator_name:
            print(f"[SearchPolicy] Loading query generator: {query_generator_name}")
            self.query_tokenizer = AutoTokenizer.from_pretrained(query_generator_name)
            self.query_generator = AutoModelForCausalLM.from_pretrained(
                query_generator_name,
                torch_dtype=torch.float16 if 'cuda' in self.device else torch.float32,
                device_map='auto' if 'cuda' in self.device else None
            )
        else:
            self.query_tokenizer = None
            self.query_generator = None
        
        # Episode history (for simple mode)
        self.episodes: List[SearchEpisode] = []
        self.query_success: Dict[str, List[float]] = defaultdict(list)
    
    def generate_query_variants(
        self, 
        question: str, 
        num_variants: int = None
    ) -> List[QueryVariant]:
        """
        Generate multiple query variants from a question.
        
        Args:
            question: Original question
            num_variants: Number of variants to generate
            
        Returns:
            List of QueryVariant objects
        """
        num_variants = num_variants or self.num_variants
        
        if self.query_generator is None:
            # Fallback: Use simple paraphrasing strategies
            return self._generate_variants_heuristic(question, num_variants)
        
        # Use LLM to generate variants
        prompt = f"""Generate {num_variants} different search queries for the following question. Each query should be optimized for finding relevant information.

Question: {question}

Search Queries:
1."""
        
        inputs = self.query_tokenizer(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.query_generator.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.8,
                num_return_sequences=1,
                do_sample=True
            )
        
        generated = self.query_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse generated queries
        queries = []
        lines = generated.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                query = line.lstrip('0123456789.- ').strip()
                if query and len(query) > 5:
                    queries.append(query)
        
        # If LLM failed, fall back to heuristics
        if len(queries) < num_variants:
            heuristic = self._generate_variants_heuristic(question, num_variants)
            existing = [q.query for q in queries]
            for h in heuristic:
                if h.query not in existing:
                    queries.append(h.query)
        
        # Create variants with embeddings
        variants = []
        query_texts = [question] + queries[:num_variants]
        embeddings = self.embedder.encode(query_texts, normalize_embeddings=True)
        
        for i, query in enumerate(queries[:num_variants]):
            variants.append(QueryVariant(
                query=query,
                embedding=embeddings[i + 1]
            ))
        
        return variants
    
    def _generate_variants_heuristic(
        self, 
        question: str, 
        num_variants: int
    ) -> List[QueryVariant]:
        """
        Generate query variants using heuristic transformations.
        """
        variants = []
        
        # Strategy 1: Original question
        variants.append(question)
        
        # Strategy 2: Remove question words
        stop_words = {'what', 'who', 'when', 'where', 'why', 'how', 'which', 'is', 'are', 'the', 'a', 'an'}
        words = question.lower().split()
        cleaned = ' '.join([w for w in words if w not in stop_words])
        if cleaned:
            variants.append(cleaned)
        else:
            variants.append(question)
        
        # Strategy 3: Extract key terms
        words = [w for w in question.split() if len(w) > 3]
        variants.append(' '.join(words[:6]))
        
        # Strategy 4: Add context words
        variants.append(f"{question} explained")
        
        # Strategy 5: Reorder
        words = question.split()
        if len(words) > 4:
            variants.append(' '.join(words[2:] + words[:2]))
        else:
            variants.append(f"about {question}")
        
        # Compute embeddings
        embeddings = self.embedder.encode(variants, normalize_embeddings=True)
        
        query_variants = []
        for i, query in enumerate(variants[:num_variants]):
            query_variants.append(QueryVariant(
                query=query,
                embedding=embeddings[i]
            ))
        
        return query_variants
    
    def compute_reward(self, episode: SearchEpisode) -> float:
        """
        Compute reward for a search episode.
        
        Reward = w1 * correctness + w2 * evidence_quality + w3 * efficiency
        """
        reward = (
            self.reward_weights['answer_correctness'] * episode.answer_correctness +
            self.reward_weights['evidence_quality'] * episode.evidence_quality +
            self.reward_weights['retrieval_efficiency'] * episode.retrieval_efficiency
        )
        return reward
    
    def store_episode(self, episode: SearchEpisode):
        """
        Store a search episode for learning.
        """
        episode.reward = self.compute_reward(episode)
        self.episodes.append(episode)
        
        # Track query success (for simple mode)
        self.query_success[episode.query].append(episode.reward)
    
    def rank_query_patterns(self, question: str) -> List[Tuple[str, float]]:
        """
        Rank query patterns by historical success.
        
        Args:
            question: Current question
            
        Returns:
            List of (query_pattern, avg_reward) sorted by reward
        """
        if not self.query_success:
            return []
        
        # Compute average reward per query
        ranked = []
        for query, rewards in self.query_success.items():
            avg_reward = np.mean(rewards)
            ranked.append((query, avg_reward))
        
        # Sort by reward (descending)
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def predict_query_quality(
        self, 
        question: str, 
        query: str
    ) -> float:
        """
        Predict the quality of a query using the reward model.
        
        Args:
            question: Question embedding
            query: Query to evaluate
            
        Returns:
            Predicted quality score (0 to 1)
        """
        if self.learning_mode == 'simple':
            # Use historical average
            rewards = self.query_success.get(query, [0.5])
            return np.mean(rewards)
        
        # Use neural reward model
        question_emb = self.embedder.encode(question, normalize_embeddings=True)
        query_emb = self.embedder.encode(query, normalize_embeddings=True)
        
        question_tensor = torch.tensor(question_emb).unsqueeze(0).float().to(self.device)
        query_tensor = torch.tensor(query_emb).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            predicted_reward = self.reward_model(question_tensor, query_tensor)
        
        return predicted_reward.item()
    
    def select_best_query(
        self, 
        question: str, 
        variants: List[QueryVariant] = None
    ) -> QueryVariant:
        """
        Select the best query from variants.
        
        Args:
            question: Original question
            variants: Pre-generated variants (optional)
            
        Returns:
            Best QueryVariant
        """
        if variants is None:
            variants = self.generate_query_variants(question)
        
        # Score each variant
        for variant in variants:
            variant.score = self.predict_query_quality(question, variant.query)
        
        # Return best
        best = max(variants, key=lambda v: v.score)
        return best
    
    def train_reward_model(
        self, 
        epochs: int = 10, 
        batch_size: int = None
    ) -> List[float]:
        """
        Train the reward model on collected episodes.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training losses
        """
        if self.learning_mode == 'simple':
            print("[SearchPolicy] Simple mode - no training needed")
            return []
        
        if len(self.episodes) < 10:
            print(f"[SearchPolicy] Not enough episodes ({len(self.episodes)}). Need at least 10.")
            return []
        
        batch_size = batch_size or self.batch_size
        
        # Prepare training data
        questions = []
        queries = []
        rewards = []
        
        for ep in self.episodes:
            if ep.query_embedding is not None:
                q_emb = ep.query_embedding
            else:
                q_emb = self.embedder.encode(ep.question, normalize_embeddings=True)
            
            if ep.query_embedding is not None:
                query_emb = ep.query_embedding
            else:
                query_emb = self.embedder.encode(ep.query, normalize_embeddings=True)
            
            questions.append(q_emb)
            queries.append(query_emb)
            rewards.append(ep.reward)
        
        # Convert to tensors
        X_question = torch.tensor(np.array(questions)).float().to(self.device)
        X_query = torch.tensor(np.array(queries)).float().to(self.device)
        Y_reward = torch.tensor(rewards).float().unsqueeze(1).to(self.device)
        
        # Training loop
        self.reward_model.train()
        losses = []
        
        dataset = torch.utils.data.TensorDataset(X_question, X_query, Y_reward)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_q, batch_query, batch_reward in loader:
                self.optimizer.zero_grad()
                
                pred = self.reward_model(batch_q, batch_query)
                loss = self.criterion(pred, batch_reward)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print(f"[SearchPolicy] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.reward_model.eval()
        return losses
    
    def save_model(self, path: str):
        """Save reward model and episode history."""
        os.makedirs(path, exist_ok=True)
        
        if self.learning_mode in ['reward_model', 'rl']:
            torch.save({
                'reward_model': self.reward_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, os.path.join(path, 'reward_model.pt'))
        
        # Save episodes
        episodes_data = []
        for ep in self.episodes:
            ep_dict = asdict(ep)
            # Convert numpy arrays to lists for JSON
            for key in ['query_embedding', 'doc_scores']:
                if ep_dict.get(key) is not None:
                    ep_dict[key] = ep_dict[key].tolist() if hasattr(ep_dict[key], 'tolist') else ep_dict[key]
            if ep_dict.get('documents') is not None:
                pass  # Keep as is
            episodes_data.append(ep_dict)
        
        with open(os.path.join(path, 'episodes.json'), 'w') as f:
            json.dump(episodes_data, f, indent=2)
        
        print(f"[SearchPolicy] Model saved to {path}")
    
    def load_model(self, path: str):
        """Load reward model and episode history."""
        if self.learning_mode in ['reward_model', 'rl']:
            checkpoint = torch.load(
                os.path.join(path, 'reward_model.pt'),
                map_location=self.device
            )
            self.reward_model.load_state_dict(checkpoint['reward_model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load episodes
        with open(os.path.join(path, 'episodes.json'), 'r') as f:
            episodes_data = json.load(f)
        
        self.episodes = []
        for ep_dict in episodes_data:
            if ep_dict.get('query_embedding') is not None:
                ep_dict['query_embedding'] = np.array(ep_dict['query_embedding'])
            if ep_dict.get('doc_scores') is not None:
                ep_dict['doc_scores'] = [float(s) for s in ep_dict['doc_scores']]
            self.episodes.append(SearchEpisode(**ep_dict))
        
        print(f"[SearchPolicy] Model loaded from {path}, {len(self.episodes)} episodes")
    
    def get_policy_stats(self) -> Dict[str, Any]:
        """Get statistics about learned policy."""
        if not self.episodes:
            return {"status": "No episodes collected"}
        
        rewards = [ep.reward for ep in self.episodes]
        
        return {
            "total_episodes": len(self.episodes),
            "mean_reward": float(np.mean(rewards)),
            "max_reward": float(np.max(rewards)),
            "min_reward": float(np.min(rewards)),
            "unique_queries": len(self.query_success),
            "best_queries": self.rank_query_patterns("")[:5],
            "learning_mode": self.learning_mode
        }
