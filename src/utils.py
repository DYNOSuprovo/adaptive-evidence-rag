"""
Utility functions for the Adaptive Evidence-Aware RAG System.
"""

import os
import yaml
import torch
import random
import numpy as np
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config: Dict[str, Any] = None) -> torch.device:
    """Get the appropriate device (CUDA or CPU)."""
    if config and not config.get('device', {}).get('use_cuda', True):
        return torch.device('cpu')
    
    if torch.cuda.is_available():
        device_id = config.get('device', {}).get('cuda_visible_devices', '0') if config else '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        return torch.device(f'cuda:{device_id}')
    return torch.device('cpu')


def setup_cuda(config: Dict[str, Any] = None):
    """Setup CUDA environment and print GPU info."""
    if torch.cuda.is_available():
        device_id = config.get('device', {}).get('cuda_visible_devices', '0') if config else '0'
        gpu_name = torch.cuda.get_device_name(int(device_id))
        gpu_memory = torch.cuda.get_device_properties(int(device_id)).total_memory / 1e9
        
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        
        # Enable TF32 for better performance on Ampere GPUs (Causes segfault on GTX 1650 Ti)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        print("No GPU available. Using CPU.")
        print(f"CPU Count: {os.cpu_count()}")


def batch_encode(texts: List[str], model: SentenceTransformer, batch_size: int = 32, 
                 show_progress: bool = True, device: str = None) -> np.ndarray:
    """Encode texts in batches with progress bar."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        device=device or model.device,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embeddings


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    # Compute similarity matrix
    similarity_matrix = np.dot(normalized, normalized.T)
    return similarity_matrix


def cluster_documents(embeddings: np.ndarray, threshold: float = 0.92, 
                      method: str = "agglomerative") -> List[List[int]]:
    """
    Cluster documents based on embedding similarity.
    
    Returns:
        List of clusters, where each cluster is a list of document indices.
    """
    from sklearn.cluster import AgglomerativeClustering
    
    # Compute distance matrix
    similarity_matrix = compute_similarity_matrix(embeddings)
    distance_matrix = 1 - similarity_matrix
    
    # Ensure diagonal is 0
    np.fill_diagonal(distance_matrix, 0)
    
    # Clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - threshold,
        metric='precomputed',
        linkage='average'
    )
    
    labels = clustering.fit_predict(distance_matrix)
    
    # Group by cluster
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)
    
    return list(clusters.values())


def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with given variables."""
    return template.format(**kwargs)


def save_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON."""
    import json
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def load_checkpoint(model, checkpoint_path: str):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def save_checkpoint(model, optimizer, epoch: int, loss: float, path: str):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved to {path}")


# Prompt Templates
VERIFICATION_QUESTION_TEMPLATE = """Given the following claim, generate specific verification questions that can be used to fact-check it.

Claim: {claim}

Generate {num_questions} verification questions:"""

PARAPHRASE_TEMPLATE = """Generate {num_paraphrases} different ways to ask the following question. Each paraphrase should have the same meaning but use different words.

Original Question: {question}

Paraphrases:
1."""

EVIDENCE_SYNTHESIS_TEMPLATE = """You are a helpful assistant. Answer the question based on the provided evidence.

Question: {question}

Evidence:
{evidence}

Provide a concise, accurate answer based only on the evidence above. Cite the evidence numbers in your answer."""

INDEPENDENCE_ANALYSIS_TEMPLATE = """Analyze whether the following two pieces of evidence are independent sources or if one is derived from the other.

Evidence 1: {evidence1}

Evidence 2: {evidence2}

Are these independent? Answer with a score from 0 to 1, where 0 means they are copies of each other and 1 means they are completely independent.

Score:"""
