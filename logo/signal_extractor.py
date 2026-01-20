"""
Signal Extractor for LOGO

Implements signal extraction methods from LoRA projection outputs:
- L2 norm-based signal
- Entropy-based signal
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Literal


def compute_norm_signal(projection_output: torch.Tensor) -> float:
    """
    Compute L2 norm-based signal from LoRA projection output.
    
    Formula: s = ||o||_2
    
    Intuition: Larger norm indicates stronger activation and greater influence.
    
    Args:
        projection_output: Tensor of shape (batch_size, seq_len, hidden_dim)
        
    Returns:
        Signal score (float)
    """
    # Use mean of L2 norms across batch and sequence
    return torch.norm(projection_output, p=2, dim=-1).mean().item()


def compute_entropy_signal(projection_output: torch.Tensor) -> float:
    """
    Compute entropy-based signal from LoRA projection output.
    
    Formula: s = 1 / entropy(softmax(o))
    
    Intuition: Lower entropy means more confident/focused response.
    
    Args:
        projection_output: Tensor of shape (batch_size, seq_len, hidden_dim)
        
    Returns:
        Signal score (float) - larger is better
    """
    # Apply softmax to get probability distribution
    probs = F.softmax(projection_output, dim=-1)
    
    # Compute entropy: H = -sum(p * log(p))
    log_probs = torch.log(probs + 1e-8)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    
    # Return inverse of entropy (lower entropy = higher signal)
    return 1.0 / (entropy.item() + 1e-8)


def normalize_weights(signals: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize signal scores to get merging weights.
    
    Formula: w_i = s_i / sum(s_j)
    
    Args:
        signals: Dict mapping adapter names to signal scores
        
    Returns:
        Dict mapping adapter names to normalized weights (sum to 1.0)
    """
    total = sum(signals.values())
    if total == 0:
        # Uniform weights if all signals are zero
        n = len(signals)
        return {name: 1.0 / n for name in signals}
    
    return {name: score / total for name, score in signals.items()}


def select_top_k(
    signals: Dict[str, float], 
    k: int
) -> List[str]:
    """
    Select top-K adapters based on signal scores.
    
    Args:
        signals: Dict mapping adapter names to signal scores
        k: Number of adapters to select
        
    Returns:
        List of selected adapter names (sorted by score, descending)
    """
    sorted_adapters = sorted(signals.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_adapters[:k]]


def compute_signal(
    projection_output: torch.Tensor,
    signal_type: Literal['norm', 'entropy'] = 'norm'
) -> float:
    """
    Compute signal score using specified method.
    
    Args:
        projection_output: Tensor from LoRA projection
        signal_type: 'norm' for L2 norm, 'entropy' for inverse entropy
        
    Returns:
        Signal score
    """
    if signal_type == 'norm':
        return compute_norm_signal(projection_output)
    elif signal_type == 'entropy':
        return compute_entropy_signal(projection_output)
    else:
        raise ValueError(f"Unknown signal type: {signal_type}. Use 'norm' or 'entropy'.")
