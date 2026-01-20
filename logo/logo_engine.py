"""
LOGO Engine: LoRA on the Go for InternVL2

Core engine class that implements dynamic LoRA selection and merging
using Swift's native multi-adapter support.

Based on the paper: "LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging"

NOTE: Swift/PEFT's set_active_adapters/deactivate_adapter are no-ops for PeftModel.
      We use add_weighted_adapter for merging and PeftModel.set_adapter for activation.
"""

import os
import json
import torch
from typing import Dict, List, Optional, Literal, Tuple, Any
from dataclasses import dataclass

from swift.tuners import Swift
from swift.utils import get_logger

from .signal_extractor import compute_signal, normalize_weights, select_top_k

logger = get_logger()


@dataclass
class LoRAConfig:
    """Configuration for a single LoRA adapter"""
    lora_name: str
    lora_path: str
    description: str = ""


class LOGOEngine:
    """
    LOGO: LoRA on the Go Engine
    
    Implements dynamic LoRA selection and merging at inference time.
    Uses Swift library's native multi-adapter support.
    
    Since Swift/PEFT's set_active_adapters has no effect on PeftModel,
    we use the following approach:
    1. Load all LoRAs
    2. For signal extraction: Use embedding similarity or other lightweight methods
    3. For merging: Use add_weighted_adapter to create merged adapter
    
    Args:
        base_model: The base InternVL2 model
        lora_configs: List of LoRA configurations
        top_k: Number of top adapters to select (default: 5)
        signal_type: Signal computation method ('norm' or 'entropy' or 'embedding')
    """
    
    def __init__(
        self,
        base_model,
        lora_configs: List[LoRAConfig],
        top_k: int = 5,
        signal_type: Literal['norm', 'entropy', 'embedding'] = 'embedding'
    ):
        self.model = base_model
        self.lora_configs = lora_configs
        self.top_k = min(top_k, len(lora_configs))
        self.signal_type = signal_type
        
        self.adapter_names: List[str] = []
        self.adapter_embeddings: Dict[str, torch.Tensor] = {}
        self.is_loaded = False
        self.merge_count = 0  # For unique merged adapter names
        
    def load_all_loras(self) -> None:
        """
        Load all LoRA adapters using Swift.from_pretrained.
        """
        logger.info(f"Loading {len(self.lora_configs)} LoRA adapters...")
        
        for i, cfg in enumerate(self.lora_configs):
            if not os.path.exists(cfg.lora_path):
                logger.warning(f"LoRA path not found: {cfg.lora_path}, skipping...")
                continue
                
            logger.info(f"[{i+1}/{len(self.lora_configs)}] Loading {cfg.lora_name}")
            
            try:
                self.model = Swift.from_pretrained(
                    self.model,
                    cfg.lora_path,
                    adapter_name=cfg.lora_name,
                    inference_mode=True
                )
                self.adapter_names.append(cfg.lora_name)
                
                # Load precomputed embedding if available
                embedding_path = getattr(cfg, 'embedding_path', None)
                if embedding_path and os.path.exists(embedding_path):
                    import numpy as np
                    self.adapter_embeddings[cfg.lora_name] = torch.from_numpy(
                        np.load(embedding_path)
                    ).float()
                    
            except Exception as e:
                logger.error(f"Failed to load {cfg.lora_name}: {e}")
                continue
            
        self.is_loaded = True
        logger.info(f"Successfully loaded {len(self.adapter_names)} LoRA adapters")
        
    def compute_embedding_similarity(
        self,
        query_embedding: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute similarity between query embedding and adapter embeddings.
        This is a lightweight alternative to forward-pass signal extraction.
        
        Args:
            query_embedding: Embedding of the input query
            
        Returns:
            Dict mapping adapter names to similarity scores
        """
        signals = {}
        query_embedding = query_embedding.float()
        
        for name, adapter_emb in self.adapter_embeddings.items():
            # Cosine similarity
            adapter_emb = adapter_emb.to(query_embedding.device)
            similarity = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                adapter_emb.unsqueeze(0)
            ).item()
            signals[name] = max(0, similarity)  # Ensure non-negative
            
        # For adapters without embeddings, assign average score
        if signals:
            avg_score = sum(signals.values()) / len(signals)
        else:
            avg_score = 1.0
            
        for name in self.adapter_names:
            if name not in signals:
                signals[name] = avg_score
                
        return signals
    
    def compute_uniform_signals(self) -> Dict[str, float]:
        """
        Return uniform signals for all adapters.
        Used as baseline or when no embedding is available.
        """
        return {name: 1.0 for name in self.adapter_names}
    
    def select_and_merge(
        self,
        signals: Dict[str, float],
        combination_type: str = 'linear'
    ) -> Tuple[List[str], List[float]]:
        """
        Select top-K LoRAs and merge them with signal-based weights.
        
        Args:
            signals: Dict mapping adapter names to signal scores
            combination_type: Merging method ('linear', 'svd', 'cat')
            
        Returns:
            Tuple of (selected adapter names, their weights)
        """
        # Select top-K adapters
        available_signals = {k: v for k, v in signals.items() if k in self.adapter_names}
        selected_names = select_top_k(available_signals, self.top_k)
        
        if not selected_names:
            logger.warning("No adapters selected!")
            return [], []
        
        # Get signals for selected adapters only
        selected_signals = {name: signals[name] for name in selected_names}
        
        # Normalize to get weights
        weights_dict = normalize_weights(selected_signals)
        weights = [weights_dict[name] for name in selected_names]
        
        logger.info(f"Selected LoRAs: {list(zip(selected_names, [f'{w:.3f}' for w in weights]))}")
        
        # Create unique merged adapter name
        self.merge_count += 1
        merged_name = f'logo_merged_{self.merge_count}'
        
        # Merge using Swift's add_weighted_adapter
        try:
            self.model.add_weighted_adapter(
                adapters=selected_names,
                weights=weights,
                adapter_name=merged_name,
                combination_type=combination_type
            )
            
            # Set the merged adapter as active
            # Note: For PeftModel, we use set_adapter
            if hasattr(self.model, 'set_adapter'):
                self.model.set_adapter(merged_name)
            elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'set_adapter'):
                self.model.base_model.set_adapter(merged_name)
                
            logger.info(f"Created and activated merged adapter: {merged_name}")
            
        except Exception as e:
            logger.error(f"Failed to merge adapters: {e}")
            # Fallback: just use the top-1 adapter
            if selected_names:
                top_adapter = selected_names[0]
                logger.info(f"Falling back to single adapter: {top_adapter}")
                if hasattr(self.model, 'set_adapter'):
                    self.model.set_adapter(top_adapter)
                    
        return selected_names, weights
    
    def get_current_model(self):
        """Get the current model (with merged/selected adapters active)."""
        return self.model


def load_lora_configs(config_paths: List[str]) -> List[LoRAConfig]:
    """
    Load LoRA configurations from JSON files.
    
    Args:
        config_paths: List of paths to config JSON files
        
    Returns:
        List of LoRAConfig objects
    """
    configs = []
    
    for path in config_paths:
        if not os.path.exists(path):
            logger.warning(f"Config file not found: {path}")
            continue
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        for item in data:
            cfg = LoRAConfig(
                lora_name=item['lora_name'],
                lora_path=item['lora_path'],
                description=item.get('description', '')
            )
            # Store embedding path as attribute
            if 'embedding_path' in item:
                cfg.embedding_path = item['embedding_path']
            configs.append(cfg)
            
    logger.info(f"Loaded {len(configs)} LoRA configs from {len(config_paths)} files")
    return configs
