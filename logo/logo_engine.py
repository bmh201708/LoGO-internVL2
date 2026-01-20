"""
LOGO Engine: LoRA on the Go for InternVL2

Core engine class that implements dynamic LoRA selection and merging
using Swift's native multi-adapter support.

Based on the paper: "LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging"

Key Components:
1. Signal Extraction: Extract signals from LoRA projection outputs (Section 3.2)
2. Adapter Selection: Top-K selection based on signal scores
3. Adapter Merging: Weighted merging using add_weighted_adapter (Section 3.3)
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Literal, Tuple, Any
from dataclasses import dataclass

from swift.tuners import Swift
from swift.utils import get_logger

from .signal_extractor import (
    LOGOSignalExtractor,
    EmbeddingSignalExtractor,
    normalize_weights,
    select_top_k,
    compute_uniform_signals
)

logger = get_logger()


@dataclass
class LoRAConfig:
    """Configuration for a single LoRA adapter"""
    lora_name: str
    lora_path: str
    description: str = ""
    embedding_path: Optional[str] = None  # Path to precomputed embedding


class LOGOEngine:
    """
    LOGO: LoRA on the Go Engine
    
    Implements dynamic LoRA selection and merging at inference time.
    Uses Swift library's native multi-adapter support.
    
    The workflow (following Algorithm 1 in the paper):
    1. Load all LoRA adapters
    2. For each input, extract signals from LoRA projections (probe pass)
    3. Select top-K adapters based on signal scores
    4. Merge selected adapters with signal-based weights
    5. Generate output using merged adapter
    
    Args:
        base_model: The base InternVL2 model
        lora_configs: List of LoRA configurations
        top_k: Number of top adapters to select (default: 5, paper uses 20)
        signal_type: Signal computation method ('norm', 'entropy', 'embedding', 'uniform')
        target_block_idx: Transformer block for signal extraction (-1 = last block)
        token_position: Token position for signal computation ('last', 'first', 'mean')
        combination_type: Merging method ('linear', 'svd', 'cat')
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        lora_configs: List[LoRAConfig],
        top_k: int = 5,
        signal_type: Literal['norm', 'entropy', 'embedding', 'uniform'] = 'norm',
        target_block_idx: int = -1,
        token_position: Literal['last', 'first', 'mean'] = 'last',
        combination_type: str = 'linear',
        use_baseline_calibration: bool = True
    ):
        self.model = base_model
        self.lora_configs = lora_configs
        self.top_k = min(top_k, len(lora_configs))
        self.signal_type = signal_type
        self.target_block_idx = target_block_idx
        self.token_position = token_position
        self.combination_type = combination_type
        self.use_baseline_calibration = use_baseline_calibration
        
        self.adapter_names: List[str] = []
        self.adapter_embeddings: Dict[str, torch.Tensor] = {}
        self.is_loaded = False
        self.merge_count = 0  # For unique merged adapter names
        self.current_merged_adapter: Optional[str] = None
        
        # Signal extractors (initialized after loading adapters)
        self._signal_extractor: Optional[LOGOSignalExtractor] = None
        self._embedding_extractor: Optional[EmbeddingSignalExtractor] = None
        
        # Tokenizer for baseline computation
        self._tokenizer = None
        
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
                embedding_path = cfg.embedding_path
                if embedding_path and os.path.exists(embedding_path):
                    import numpy as np
                    self.adapter_embeddings[cfg.lora_name] = torch.from_numpy(
                        np.load(embedding_path)
                    ).float()
                    logger.info(f"  Loaded embedding for {cfg.lora_name}")
                    
            except Exception as e:
                logger.error(f"Failed to load {cfg.lora_name}: {e}")
                continue
            
        self.is_loaded = True
        logger.info(f"Successfully loaded {len(self.adapter_names)} LoRA adapters")
        
        # Initialize signal extractors
        self._init_signal_extractors()
    
    def _init_signal_extractors(self) -> None:
        """Initialize signal extractors after loading adapters."""
        if not self.adapter_names:
            return
        
        # Initialize LoRA-based signal extractor
        if self.signal_type in ['norm', 'entropy']:
            self._signal_extractor = LOGOSignalExtractor(
                model=self.model,
                adapter_names=self.adapter_names,
                signal_type=self.signal_type,
                target_block_idx=self.target_block_idx,
                token_position=self.token_position,
                use_baseline_calibration=self.use_baseline_calibration
            )
            logger.info(f"Initialized LOGOSignalExtractor with signal_type={self.signal_type}, calibration={self.use_baseline_calibration}")
        
        # Initialize embedding-based extractor if embeddings available
        if self.adapter_embeddings:
            self._embedding_extractor = EmbeddingSignalExtractor(self.adapter_embeddings)
            logger.info(f"Initialized EmbeddingSignalExtractor with {len(self.adapter_embeddings)} embeddings")
    
    def compute_baseline(self, tokenizer=None) -> Dict[str, float]:
        """
        Compute baseline signals for calibration.
        
        Should be called after load_all_loras() and before extract_signals().
        
        Args:
            tokenizer: Tokenizer for encoding baseline queries
            
        Returns:
            Dict of baseline signals
        """
        if self._signal_extractor is None:
            logger.warning("Signal extractor not initialized. Call load_all_loras() first.")
            return {}
        
        if tokenizer is not None:
            self._tokenizer = tokenizer
        
        return self._signal_extractor.compute_baseline(tokenizer=self._tokenizer)
    
    def set_tokenizer(self, tokenizer) -> None:
        """Set tokenizer for baseline computation."""
        self._tokenizer = tokenizer
    
    def extract_signals(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        query_embedding: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Extract signals from all adapters.
        
        Implements the signal extraction from Section 3.2 of the paper.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask  
            inputs_embeds: Optional input embeddings
            pixel_values: Optional pixel values for vision models
            query_embedding: Optional query embedding for embedding-based signal
            **kwargs: Additional model inputs
            
        Returns:
            Dict mapping adapter names to signal scores
        """
        if not self.adapter_names:
            logger.warning("No adapters loaded!")
            return {}
        
        # Uniform signals (baseline)
        if self.signal_type == 'uniform':
            return compute_uniform_signals(self.adapter_names)
        
        # Embedding-based signals
        if self.signal_type == 'embedding':
            if query_embedding is not None and self._embedding_extractor:
                return self._embedding_extractor.extract_signals(query_embedding)
            else:
                logger.warning("No query embedding provided or extractor not available. Using uniform signals.")
                return compute_uniform_signals(self.adapter_names)
        
        # LoRA projection-based signals (norm or entropy)
        if self._signal_extractor is not None:
            try:
                signals = self._signal_extractor.extract_signals(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    pixel_values=pixel_values,
                    **kwargs
                )
                return signals
            except Exception as e:
                logger.error(f"Signal extraction failed: {e}. Using uniform signals.")
                return compute_uniform_signals(self.adapter_names)
        
        # Fallback to uniform signals
        return compute_uniform_signals(self.adapter_names)
    
    def select_and_merge(
        self,
        signals: Dict[str, float],
        combination_type: Optional[str] = None
    ) -> Tuple[List[str], List[float]]:
        """
        Select top-K LoRAs and compute their weights.
        
        NOTE: This method now uses the Mixture mode from the paper (Section 3.3),
        which computes weighted sum of LoRA outputs at runtime, rather than
        merging parameters beforehand.
        
        Implements Section 3.2 (selection) and Section 3.3 (merging) of the paper.
        
        Args:
            signals: Dict mapping adapter names to signal scores
            combination_type: Merging method (for backward compatibility, ignored in mixture mode)
            
        Returns:
            Tuple of (selected adapter names, their weights)
        """
        # Select top-K adapters (Equation 4)
        available_signals = {k: v for k, v in signals.items() if k in self.adapter_names}
        selected_names = select_top_k(available_signals, self.top_k)
        
        if not selected_names:
            logger.warning("No adapters selected!")
            return [], []
        
        # Get signals for selected adapters only
        selected_signals = {name: signals[name] for name in selected_names}
        
        # Normalize to get weights (Equation 5)
        weights_dict = normalize_weights(selected_signals)
        weights = [weights_dict[name] for name in selected_names]
        
        logger.info(f"Selected LoRAs: {list(zip(selected_names, [f'{w:.3f}' for w in weights]))}")
        
        # Store current selection for mixture mode
        self._current_selected_names = selected_names
        self._current_weights = weights
        
        return selected_names, weights
    
    def get_lora_mapping(
        self,
        selected_names: List[str],
        weights: List[float],
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Create lora_mapping tensor for Mixture mode inference.
        
        The lora_mapping tensor has shape (batch_size, num_adapters) where:
        - Each row corresponds to a sample in the batch
        - Each column corresponds to an adapter (in the order of self.adapter_names)
        - Values are the weights for selected adapters, 0 for non-selected
        
        Args:
            selected_names: List of selected adapter names
            weights: Corresponding weights for selected adapters
            batch_size: Batch size
            device: Device for the tensor
            
        Returns:
            lora_mapping tensor of shape (batch_size, num_adapters)
        """
        if device is None:
            device = next(self.model.parameters()).device
        
        num_adapters = len(self.adapter_names)
        lora_mapping = torch.zeros(batch_size, num_adapters, device=device)
        
        # Fill in the weights for selected adapters
        for name, weight in zip(selected_names, weights):
            if name in self.adapter_names:
                idx = self.adapter_names.index(name)
                lora_mapping[:, idx] = weight
        
        return lora_mapping
    
    def select_and_merge_legacy(
        self,
        signals: Dict[str, float],
        combination_type: Optional[str] = None
    ) -> Tuple[List[str], List[float]]:
        """
        Legacy method: Parameter-level merging using add_weighted_adapter.
        
        This creates a new merged adapter by combining LoRA parameters.
        Use this for debugging or comparison with the mixture mode.
        
        Args:
            signals: Dict mapping adapter names to signal scores
            combination_type: Merging method ('linear', 'svd', 'cat')
            
        Returns:
            Tuple of (selected adapter names, their weights)
        """
        if combination_type is None:
            combination_type = self.combination_type
            
        # Select top-K adapters (Equation 4)
        available_signals = {k: v for k, v in signals.items() if k in self.adapter_names}
        selected_names = select_top_k(available_signals, self.top_k)
        
        if not selected_names:
            logger.warning("No adapters selected!")
            return [], []
        
        # Get signals for selected adapters only
        selected_signals = {name: signals[name] for name in selected_names}
        
        # Normalize to get weights (Equation 5)
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
            if hasattr(self.model, 'set_adapter'):
                self.model.set_adapter(merged_name)
            elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'set_adapter'):
                self.model.base_model.set_adapter(merged_name)
                
            self.current_merged_adapter = merged_name
            logger.info(f"Created and activated merged adapter: {merged_name}")
            
        except Exception as e:
            logger.error(f"Failed to merge adapters: {e}")
            # Fallback: just use the top-1 adapter
            if selected_names:
                top_adapter = selected_names[0]
                logger.info(f"Falling back to single adapter: {top_adapter}")
                if hasattr(self.model, 'set_adapter'):
                    self.model.set_adapter(top_adapter)
                self.current_merged_adapter = top_adapter
                    
        return selected_names, weights
    
    def process_input(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        query_embedding: Optional[torch.Tensor] = None,
        debug_signals: bool = False,
        **kwargs
    ) -> Tuple[List[str], List[float]]:
        """
        Full LOGO pipeline: extract signals, select, and merge.
        
        This is the main entry point for processing an input with LOGO.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            inputs_embeds: Optional input embeddings
            pixel_values: Optional pixel values
            query_embedding: Optional query embedding
            debug_signals: If True, print all raw signals for debugging
            **kwargs: Additional model inputs
            
        Returns:
            Tuple of (selected adapter names, their weights)
        """
        # Step 1: Extract signals
        signals = self.extract_signals(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            query_embedding=query_embedding,
            **kwargs
        )
        
        # Debug: print all raw signals
        if debug_signals:
            logger.info("=" * 60)
            logger.info("DEBUG: All raw signals (sorted by value):")
            sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
            for name, sig in sorted_signals:
                prefix = "  [APP]" if name.startswith("app_") else "  [CAT]"
                logger.info(f"{prefix} {name}: {sig:.6f}")
            logger.info("=" * 60)
        
        # Step 2 & 3: Select and merge
        selected_names, weights = self.select_and_merge(signals)
        
        return selected_names, weights
    
    def merge_with_weights(
        self,
        adapter_names: List[str],
        weights: List[float],
        combination_type: Optional[str] = None
    ) -> str:
        """
        Directly merge adapters with given weights (no re-selection).
        
        Args:
            adapter_names: List of adapter names to merge
            weights: Corresponding weights (should already be normalized)
            combination_type: Merging method ('linear', 'svd', 'cat')
            
        Returns:
            Name of the merged adapter
        """
        if combination_type is None:
            combination_type = self.combination_type
            
        if not adapter_names:
            logger.warning("No adapters to merge!")
            return None
            
        # Create unique merged adapter name
        self.merge_count += 1
        merged_name = f'logo_merged_{self.merge_count}'
        
        logger.info(f"Merging adapters: {list(zip(adapter_names, [f'{w:.3f}' for w in weights]))}")
        
        # Merge using Swift's add_weighted_adapter
        try:
            self.model.add_weighted_adapter(
                adapters=adapter_names,
                weights=weights,
                adapter_name=merged_name,
                combination_type=combination_type
            )
            
            # Set the merged adapter as active
            if hasattr(self.model, 'set_adapter'):
                self.model.set_adapter(merged_name)
            elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'set_adapter'):
                self.model.base_model.set_adapter(merged_name)
                
            self.current_merged_adapter = merged_name
            logger.info(f"Created and activated merged adapter: {merged_name}")
            
        except Exception as e:
            logger.error(f"Failed to merge adapters: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: just use the top-1 adapter
            if adapter_names:
                top_adapter = adapter_names[0]
                logger.info(f"Falling back to single adapter: {top_adapter}")
                if hasattr(self.model, 'set_adapter'):
                    self.model.set_adapter(top_adapter)
                self.current_merged_adapter = top_adapter
                    
        return merged_name
    
    def reset_merged_adapter(self) -> None:
        """
        Reset to default state after processing an input.
        
        This should be called between processing different inputs
        to ensure clean state.
        """
        # Note: We don't delete the merged adapter as it might cause issues
        # Just track that we need a new merge for the next input
        self.current_merged_adapter = None
    
    def delete_merged_adapters(self) -> None:
        """
        Delete all merged adapters to free memory.
        
        Call this periodically if memory is a concern.
        """
        if hasattr(self.model, 'delete_adapter'):
            for i in range(1, self.merge_count + 1):
                adapter_name = f'logo_merged_{i}'
                try:
                    self.model.delete_adapter(adapter_name)
                except:
                    pass
        self.merge_count = 0
        self.current_merged_adapter = None
    
    def get_current_model(self) -> nn.Module:
        """Get the current model (with merged/selected adapters active)."""
        return self.model
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """Get information about loaded adapters."""
        info = {
            'loaded_adapters': self.adapter_names,
            'num_adapters': len(self.adapter_names),
            'top_k': self.top_k,
            'signal_type': self.signal_type,
            'target_block_idx': self.target_block_idx,
            'token_position': self.token_position,
            'combination_type': self.combination_type,
            'use_baseline_calibration': self.use_baseline_calibration,
            'has_embeddings': list(self.adapter_embeddings.keys()),
            'current_merged_adapter': self.current_merged_adapter,
            'merge_count': self.merge_count
        }
        
        # Add baseline info if available
        if self._signal_extractor is not None:
            baseline = self._signal_extractor.get_baseline_signals()
            info['baseline_computed'] = baseline is not None
            if baseline:
                info['baseline_signals'] = baseline
        
        return info


def load_lora_configs(config_paths: List[str]) -> List[LoRAConfig]:
    """
    Load LoRA configurations from JSON files.
    
    Expected JSON format:
    [
        {
            "lora_name": "adapter_name",
            "lora_path": "/path/to/lora",
            "description": "optional description",
            "embedding_path": "/path/to/embedding.npy"  // optional
        },
        ...
    ]
    
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
                description=item.get('description', ''),
                embedding_path=item.get('embedding_path', None)
            )
            configs.append(cfg)
            
    logger.info(f"Loaded {len(configs)} LoRA configs from {len(config_paths)} files")
    return configs


def create_logo_engine(
    model: nn.Module,
    config_paths: List[str],
    top_k: int = 5,
    signal_type: str = 'norm',
    **kwargs
) -> LOGOEngine:
    """
    Convenience function to create and initialize a LOGOEngine.
    
    Args:
        model: Base model
        config_paths: Paths to LoRA config JSON files
        top_k: Number of adapters to select
        signal_type: Signal computation method
        **kwargs: Additional arguments for LOGOEngine
        
    Returns:
        Initialized LOGOEngine with all adapters loaded
    """
    lora_configs = load_lora_configs(config_paths)
    
    engine = LOGOEngine(
        base_model=model,
        lora_configs=lora_configs,
        top_k=top_k,
        signal_type=signal_type,
        **kwargs
    )
    
    engine.load_all_loras()
    
    return engine
