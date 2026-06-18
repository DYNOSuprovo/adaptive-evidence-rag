"""
Neural Network Models for Training

This module defines PyTorch models that can be trained to improve:
1. Independence scoring
2. Utility prediction
3. Query reward prediction
4. End-to-end retrieval quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Any, Optional


class EvidenceQualityModel(nn.Module):
    """
    End-to-end model that predicts evidence quality from
    question + evidence pairs.
    
    Input: Question embedding + Evidence embedding
    Output: Quality scores (independence, utility, stability)
    """
    
    def __init__(
        self,
        embedding_dim: int = 1024,
        hidden_dim: int = 512,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Independence head
        self.independence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Utility head
        self.utility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Redundancy head (for detecting duplicates)
        self.redundancy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        question_emb: torch.Tensor, 
        evidence_emb: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            question_emb: (batch, embed_dim)
            evidence_emb: (batch, embed_dim)
            
        Returns:
            Dictionary of quality scores
        """
        # Concatenate
        x = torch.cat([question_emb, evidence_emb], dim=-1)
        
        # Encode
        features = self.encoder(x)
        
        # Predict scores
        independence = self.independence_head(features)
        utility = self.utility_head(features)
        redundancy = self.redundancy_head(features)
        
        return {
            'independence': independence,
            'utility': utility,
            'redundancy': redundancy,
            'quality': (independence + utility) / 2
        }


class DocumentPairClassifier(nn.Module):
    """
    Classifies pairs of documents as:
    - Independent (different sources)
    - Related (same topic, different info)
    - Duplicate (copied content)
    
    This is trained on the FEVER dataset with NLI labels.
    """
    
    def __init__(
        self,
        embedding_dim: int = 1024,
        hidden_dim: int = 512,
        num_classes: int = 3  # independent, related, duplicate
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(
        self, 
        doc1_emb: torch.Tensor, 
        doc2_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Classify document pair.
        
        Args:
            doc1_emb: (batch, embed_dim)
            doc2_emb: (batch, embed_dim)
            
        Returns:
            Logits (batch, num_classes)
        """
        x = torch.cat([doc1_emb, doc2_emb], dim=-1)
        return self.network(x)


class UtilityPredictor(nn.Module):
    """
    Predicts the utility of a retrieved document given:
    - Question embedding
    - Document embedding
    - Existing context embedding (optional)
    
    Output: Utility score (0 to 1)
    """
    
    def __init__(
        self,
        embedding_dim: int = 1024,
        hidden_dim: int = 512,
        use_context: bool = True
    ):
        super().__init__()
        
        self.use_context = use_context
        
        input_dim = embedding_dim * 3 if use_context else embedding_dim * 2
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 4)  # 4 utility components
        )
    
    def forward(
        self,
        question_emb: torch.Tensor,
        doc_emb: torch.Tensor,
        context_emb: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict utility components.
        
        Args:
            question_emb: (batch, embed_dim)
            doc_emb: (batch, embed_dim)
            context_emb: (batch, embed_dim) or None
            
        Returns:
            Dictionary with utility components and overall score
        """
        if self.use_context and context_emb is not None:
            x = torch.cat([question_emb, doc_emb, context_emb], dim=-1)
        else:
            x = torch.cat([question_emb, doc_emb], dim=-1)
        
        logits = self.network(x)
        
        # Split into components
        novelty = torch.sigmoid(logits[:, 0:1])
        confidence_gain = torch.sigmoid(logits[:, 1:2])
        contradiction = torch.sigmoid(logits[:, 2:3])
        overall = torch.sigmoid(logits[:, 3:4])
        
        return {
            'novelty': novelty,
            'confidence_gain': confidence_gain,
            'contradiction': contradiction,
            'overall': overall
        }


class ContrastiveEvidenceLoss(nn.Module):
    """
    Contrastive loss for learning better evidence representations.
    
    Pulls together:
    - Question and supporting evidence
    - Evidence from same source cluster
    
    Pushes apart:
    - Question and non-supporting evidence
    - Evidence from different independent sources
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        question_emb: torch.Tensor,
        pos_evidence_emb: torch.Tensor,
        neg_evidence_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            question_emb: (batch, embed_dim)
            pos_evidence_emb: (batch, embed_dim) - supporting evidence
            neg_evidence_emb: (batch, embed_dim) - non-supporting evidence
            
        Returns:
            Contrastive loss
        """
        # Normalize
        question_emb = F.normalize(question_emb, p=2, dim=-1)
        pos_evidence_emb = F.normalize(pos_evidence_emb, p=2, dim=-1)
        neg_evidence_emb = F.normalize(neg_evidence_emb, p=2, dim=-1)
        
        # Positive similarity
        pos_sim = torch.sum(question_emb * pos_evidence_emb, dim=-1) / self.temperature
        
        # Negative similarity
        neg_sim = torch.sum(question_emb * neg_evidence_emb, dim=-1) / self.temperature
        
        # InfoNCE loss
        logits = torch.stack([pos_sim, neg_sim], dim=-1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


class MultiTaskEvidenceTrainer:
    """
    Trainer that jointly trains all models with multi-task learning.
    """
    
    def __init__(
        self,
        quality_model: EvidenceQualityModel,
        pair_classifier: DocumentPairClassifier,
        utility_predictor: UtilityPredictor,
        device: str = 'cuda',
        lr: float = 1e-4,
        weight_decay: float = 0.01
    ):
        self.device = device
        self.quality_model = quality_model.to(device)
        self.pair_classifier = pair_classifier.to(device)
        self.utility_predictor = utility_predictor.to(device)
        
        # Combined optimizer
        self.optimizer = torch.optim.AdamW(
            list(quality_model.parameters()) +
            list(pair_classifier.parameters()) +
            list(utility_predictor.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.contrastive_loss = ContrastiveEvidenceLoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Dictionary containing:
                - question_emb
                - evidence_emb
                - pos_evidence_emb
                - neg_evidence_emb
                - doc1_emb, doc2_emb
                - pair_labels
                - utility_targets
                
        Returns:
            Dictionary of losses
        """
        self.optimizer.zero_grad()
        
        total_loss = 0
        losses = {}
        
        # 1. Quality prediction loss
        if 'question_emb' in batch and 'evidence_emb' in batch:
            quality_out = self.quality_model(
                batch['question_emb'].to(self.device),
                batch['evidence_emb'].to(self.device)
            )
            
            if 'independence_targets' in batch:
                ind_loss = self.regression_loss(
                    quality_out['independence'],
                    batch['independence_targets'].to(self.device).unsqueeze(1)
                )
                losses['independence'] = ind_loss.item()
                total_loss += ind_loss
            
            if 'utility_targets' in batch:
                util_loss = self.regression_loss(
                    quality_out['utility'],
                    batch['utility_targets'].to(self.device).unsqueeze(1)
                )
                losses['utility'] = util_loss.item()
                total_loss += util_loss
        
        # 2. Pair classification loss
        if 'doc1_emb' in batch and 'doc2_emb' in batch:
            pair_logits = self.pair_classifier(
                batch['doc1_emb'].to(self.device),
                batch['doc2_emb'].to(self.device)
            )
            
            if 'pair_labels' in batch:
                cls_loss = self.classification_loss(
                    pair_logits,
                    batch['pair_labels'].to(self.device)
                )
                losses['classification'] = cls_loss.item()
                total_loss += cls_loss
        
        # 3. Contrastive loss
        if all(k in batch for k in ['question_emb', 'pos_evidence_emb', 'neg_evidence_emb']):
            cont_loss = self.contrastive_loss(
                batch['question_emb'].to(self.device),
                batch['pos_evidence_emb'].to(self.device),
                batch['neg_evidence_emb'].to(self.device)
            )
            losses['contrastive'] = cont_loss.item()
            total_loss += cont_loss
        
        # 4. Utility prediction loss
        if 'question_emb' in batch and 'doc_emb' in batch:
            util_out = self.utility_predictor(
                batch['question_emb'].to(self.device),
                batch['doc_emb'].to(self.device),
                batch.get('context_emb', None)
            )
            
            if 'utility_components' in batch:
                util_pred_loss = self.regression_loss(
                    util_out['overall'],
                    batch['utility_components'].to(self.device)[:, 3:4]
                )
                losses['utility_pred'] = util_pred_loss.item()
                total_loss += util_pred_loss
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.quality_model.parameters()) +
            list(self.pair_classifier.parameters()) +
            list(self.utility_predictor.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        
        losses['total'] = total_loss.item()
        
        return losses
