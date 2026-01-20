"""
Signal Extractor for LOGO (LoRA on the Go)

Implements signal extraction methods from LoRA projection outputs as described in:
"LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging"

Methods:
- L2 norm-based signal (Equation 2)
- Entropy-based signal (Equation 3)
- Embedding similarity-based signal (lightweight alternative)

Usage:
    extractor = LOGOSignalExtractor(model, adapter_names, target_block_idx=-1)
    signals = extractor.extract_signals(input_ids, attention_mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Literal, Optional, Tuple, Any
from contextlib import contextmanager

from swift.utils import get_logger

logger = get_logger()


def compute_norm_signal(projection_output: torch.Tensor) -> float:
    """
    Compute L2 norm-based signal from LoRA projection output.
    
    Formula (Equation 2): s = ||o||_2
    
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
    
    Formula (Equation 3): s = 1 / entropy(softmax(o))
    
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
    
    Formula (Equation 5): w_i = s_i / sum(s_j)
    
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
    Select top-K adapters based on signal scores (Section 3.2).
    
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


class LOGOSignalExtractor:
    """
    Signal extractor for LOGO that extracts signals from LoRA projection outputs.
    
    Implements the signal extraction method from Section 3.2 of the paper:
    - Performs a single forward pass with all adapters attached
    - Extracts projection outputs from designated transformer block
    - Computes signal scores (norm or entropy) for each adapter
    
    Enhanced with baseline calibration:
    - Computes baseline signals using neutral queries
    - Supports ratio-based calibration (signal / baseline) for better accuracy
    
    Args:
        model: The model with LoRA adapters attached (PeftModel)
        adapter_names: List of adapter names to extract signals for
        signal_type: 'norm' or 'entropy'
        target_block_idx: Which transformer block to extract from (-1 = last block)
        token_position: Which token to use ('last', 'first', 'mean')
        use_baseline_calibration: Whether to use baseline calibration (default: True)
    """
    
    # Default neutral queries for baseline computation
    DEFAULT_BASELINE_QUERIES = [
        "Hello",
        "The",
        "What is this?",
        "Please help me",
        "I need",
    ]
    
    def __init__(
        self,
        model: nn.Module,
        adapter_names: List[str],
        signal_type: Literal['norm', 'entropy'] = 'norm',
        target_block_idx: int = -1,
        token_position: Literal['last', 'first', 'mean'] = 'last',
        use_baseline_calibration: bool = True
    ):
        self.model = model
        self.adapter_names = adapter_names
        self.signal_type = signal_type
        self.target_block_idx = target_block_idx
        self.token_position = token_position
        self.use_baseline_calibration = use_baseline_calibration
        
        # Storage for captured projections
        self._lora_projections: Dict[str, Dict[str, torch.Tensor]] = {}
        self._hooks: List[Any] = []
        
        # Baseline signals for calibration
        self._baseline_signals: Optional[Dict[str, float]] = None
        self._baseline_computed: bool = False
        
    def _get_base_model(self) -> nn.Module:
        """Get the underlying base model from PeftModel wrapper."""
        model = self.model
        
        # Navigate through possible wrappers
        if hasattr(model, 'base_model'):
            model = model.base_model
        if hasattr(model, 'model'):
            model = model.model
            
        return model
    
    def _get_tokenizer(self):
        """Try to get tokenizer from model."""
        # Try common locations for tokenizer
        if hasattr(self.model, 'tokenizer'):
            return self.model.tokenizer
        if hasattr(self.model, 'get_tokenizer'):
            return self.model.get_tokenizer()
        
        # Try to find it in config
        base_model = self._get_base_model()
        if hasattr(base_model, 'config') and hasattr(base_model.config, '_name_or_path'):
            try:
                from transformers import AutoTokenizer
                return AutoTokenizer.from_pretrained(
                    base_model.config._name_or_path, 
                    trust_remote_code=True
                )
            except:
                pass
        
        return None
    
    def compute_baseline(
        self,
        tokenizer=None,
        baseline_queries: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute baseline signals using neutral queries.
        
        Baseline signals represent the "average" activation of each adapter
        on generic/neutral inputs. This is used for ratio-based calibration.
        
        Args:
            tokenizer: Tokenizer to encode queries (if None, tries to get from model)
            baseline_queries: List of neutral query strings (if None, uses defaults)
            
        Returns:
            Dict mapping adapter names to baseline signal scores
        """
        if tokenizer is None:
            tokenizer = self._get_tokenizer()
            
        if tokenizer is None:
            logger.warning("No tokenizer available. Using uniform baseline.")
            self._baseline_signals = {name: 1.0 for name in self.adapter_names}
            self._baseline_computed = True
            return self._baseline_signals
        
        queries = baseline_queries or self.DEFAULT_BASELINE_QUERIES
        
        logger.info(f"Computing baseline signals using {len(queries)} neutral queries...")
        
        # Collect signals for each query
        all_signals: Dict[str, List[float]] = {name: [] for name in self.adapter_names}
        
        for query in queries:
            inputs = tokenizer(query, return_tensors="pt")
            input_ids = inputs['input_ids'].to(self.model.device)
            attention_mask = inputs['attention_mask'].to(self.model.device)
            
            # Extract raw signals (without calibration)
            old_calibration = self.use_baseline_calibration
            self.use_baseline_calibration = False
            
            signals = self.extract_signals(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            self.use_baseline_calibration = old_calibration
            
            for name, sig in signals.items():
                all_signals[name].append(sig)
        
        # Average baseline signals
        self._baseline_signals = {
            name: sum(sigs) / len(sigs) if sigs else 1.0
            for name, sigs in all_signals.items()
        }
        self._baseline_computed = True
        
        logger.info("Baseline signals computed:")
        for name, baseline in sorted(self._baseline_signals.items()):
            logger.info(f"  {name}: {baseline:.4f}")
        
        return self._baseline_signals
    
    def get_baseline_signals(self) -> Optional[Dict[str, float]]:
        """Get the computed baseline signals."""
        return self._baseline_signals
    
    def set_baseline_signals(self, baseline: Dict[str, float]) -> None:
        """
        Manually set baseline signals.
        
        Useful when you want to use precomputed baselines or custom values.
        
        Args:
            baseline: Dict mapping adapter names to baseline values
        """
        self._baseline_signals = baseline.copy()
        self._baseline_computed = True
        logger.info(f"Baseline signals set manually for {len(baseline)} adapters")
    
    def calibrate_signals(
        self,
        raw_signals: Dict[str, float],
        method: Literal['ratio', 'subtract'] = 'ratio'
    ) -> Dict[str, float]:
        """
        Apply baseline calibration to raw signals.
        
        Args:
            raw_signals: Dict of raw signal scores
            method: 'ratio' (signal / baseline) or 'subtract' (signal - baseline)
            
        Returns:
            Dict of calibrated signal scores
        """
        if self._baseline_signals is None:
            logger.warning("Baseline not computed. Returning raw signals.")
            return raw_signals
        
        calibrated = {}
        for name, signal in raw_signals.items():
            baseline = self._baseline_signals.get(name, 1.0)
            
            if method == 'ratio':
                # Ratio-based: signal / baseline
                # Higher ratio means more relevant than baseline
                calibrated[name] = signal / baseline if baseline > 0 else signal
            elif method == 'subtract':
                # Subtraction-based: signal - baseline  
                # Positive means more relevant than baseline
                calibrated[name] = max(0, signal - baseline)
            else:
                raise ValueError(f"Unknown calibration method: {method}")
        
        return calibrated
    
    def _find_lora_layers(self) -> Dict[str, nn.Module]:
        """
        Find all LoRA layers in the model.
        
        Returns:
            Dict mapping module names to LoRA layer modules
        """
        lora_layers = {}
        
        for name, module in self.model.named_modules():
            # Check if this is a LoRA layer (has lora_A and lora_B)
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_layers[name] = module
                
        return lora_layers
    
    def _find_target_block_lora_layers(self) -> Dict[str, nn.Module]:
        """
        Find LoRA layers in the target transformer block.
        
        According to the paper (Section 3.2), we extract signals from the 
        query projection (W_Q) of the target block B_T.
        
        Returns:
            Dict mapping module names to LoRA layer modules in target block
        """
        all_lora_layers = self._find_lora_layers()
        
        # Filter to find layers in transformer blocks
        # Common patterns: layers.N., blocks.N., h.N., etc.
        block_patterns = ['layers.', 'blocks.', 'h.', 'transformer.h.']
        
        # Group by block number
        blocks: Dict[int, Dict[str, nn.Module]] = {}
        
        for name, module in all_lora_layers.items():
            block_idx = None
            for pattern in block_patterns:
                if pattern in name:
                    # Extract block number
                    try:
                        start = name.find(pattern) + len(pattern)
                        end = name.find('.', start)
                        block_idx = int(name[start:end])
                        break
                    except (ValueError, IndexError):
                        continue
            
            if block_idx is not None:
                if block_idx not in blocks:
                    blocks[block_idx] = {}
                blocks[block_idx][name] = module
        
        if not blocks:
            logger.warning("Could not identify transformer block structure. Using all LoRA layers.")
            return all_lora_layers
        
        # Get target block
        sorted_block_idxs = sorted(blocks.keys())
        if self.target_block_idx < 0:
            target_idx = sorted_block_idxs[self.target_block_idx]
        else:
            target_idx = self.target_block_idx if self.target_block_idx in sorted_block_idxs else sorted_block_idxs[-1]
        
        logger.info(f"Extracting signals from transformer block {target_idx}")
        
        # Filter to Q projection layers
        # Priority: separate Q layers > fused QKV layers > all attention layers
        target_layers = blocks.get(target_idx, {})
        
        # First, try to find separate Q projection layers
        q_only_layers = {k: v for k, v in target_layers.items() 
                         if any(q in k.lower() for q in ['q_proj', 'query', 'self_attn.q'])
                         and 'qkv' not in k.lower() and 'wqkv' not in k.lower()}
        
        if q_only_layers:
            logger.info(f"Found {len(q_only_layers)} separate Q projection layers")
            return q_only_layers
        
        # Second, try fused QKV layers (we'll extract Q portion in the hook)
        qkv_layers = {k: v for k, v in target_layers.items() 
                      if any(q in k.lower() for q in ['wqkv', 'qkv', 'wq'])}
        
        if qkv_layers:
            logger.info(f"Found {len(qkv_layers)} fused QKV layers (will extract Q portion)")
            return qkv_layers
        
        # Fallback: return all attention-related layers in target block
        attn_layers = {k: v for k, v in target_layers.items()
                       if any(a in k.lower() for a in ['attention', 'attn', 'self_attn'])}
        
        if attn_layers:
            logger.info(f"Found {len(attn_layers)} attention layers (fallback)")
            return attn_layers
        
        # Last resort: return all layers in target block
        logger.warning(f"No Q/attention layers found, using all {len(target_layers)} layers in block")
        return target_layers
    
    def _get_q_projection_slice(self, layer: nn.Module, layer_name: str) -> Optional[slice]:
        """
        Determine the slice for Q projection in fused QKV layers.
        
        For InternLM2/InternVL2, wqkv outputs [Q, K, V] concatenated:
        - Q: first hidden_size dimensions
        - K: next kv_dim dimensions  
        - V: last kv_dim dimensions
        
        Returns:
            slice object for Q projection, or None if not a fused layer
        """
        # Check if this is a fused QKV layer
        if 'wqkv' in layer_name.lower() or 'qkv' in layer_name.lower():
            # Get the output dimension of lora_B to determine Q size
            # For fused layers, we need to extract only the Q portion
            base_layer = layer.get_base_layer() if hasattr(layer, 'get_base_layer') else layer
            
            if hasattr(base_layer, 'out_features'):
                out_features = base_layer.out_features
                # For InternLM2: out_features = hidden_size + 2 * kv_dim
                # Q portion is the first hidden_size dimensions
                # Heuristic: Q is typically 1/2 to 2/3 of total output for grouped query attention
                # For InternVL2-2B: hidden=2048, out_features=4096 (Q=2048, K=1024, V=1024)
                q_dim = out_features // 2  # Q is first half for standard MHA
                return slice(0, q_dim)
            
        return None  # Not a fused layer, use full output
    
    def _register_hooks(self, target_layers: Dict[str, nn.Module]) -> None:
        """
        Register forward hooks to capture LoRA projection outputs.
        
        For each adapter, we capture the LoRA output: o = B(A(x)) * scaling
        
        For fused QKV layers (wqkv), we extract only the Q projection portion
        as recommended in the paper (Section 3.2).
        """
        self._lora_projections = {name: {} for name in self.adapter_names}
        self._hooks = []
        
        for layer_name, layer in target_layers.items():
            # Determine if we need to slice for Q projection
            q_slice = self._get_q_projection_slice(layer, layer_name)
            is_fused_qkv = q_slice is not None
            
            if is_fused_qkv:
                logger.info(f"  Layer '{layer_name}' is fused QKV, extracting Q portion only (dims {q_slice.start}:{q_slice.stop})")
            
            def create_hook(ln, q_sl, is_fused):
                def hook(module, input, output):
                    # Capture LoRA projections for each adapter
                    if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
                        return
                    
                    x = input[0] if isinstance(input, tuple) else input
                    
                    for adapter_name in self.adapter_names:
                        if adapter_name not in module.lora_A:
                            continue
                        
                        lora_A = module.lora_A[adapter_name]
                        lora_B = module.lora_B[adapter_name]
                        scaling = module.scaling.get(adapter_name, 1.0)
                        
                        # Compute LoRA projection: o = B(A(x)) * scaling
                        with torch.no_grad():
                            x_input = x.to(lora_A.weight.dtype)
                            lora_output = lora_B(lora_A(x_input)) * scaling
                            
                            # For fused QKV layers, extract only Q projection
                            if is_fused and q_sl is not None:
                                lora_output = lora_output[..., q_sl]
                            
                            # Store for this adapter
                            if ln not in self._lora_projections[adapter_name]:
                                self._lora_projections[adapter_name][ln] = []
                            self._lora_projections[adapter_name][ln].append(lora_output.detach())
                
                return hook
            
            handle = layer.register_forward_hook(create_hook(layer_name, q_slice, is_fused_qkv))
            self._hooks.append(handle)
    
    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
    
    @contextmanager
    def _hook_context(self, target_layers: Dict[str, nn.Module]):
        """Context manager for hook registration."""
        self._register_hooks(target_layers)
        try:
            yield
        finally:
            self._remove_hooks()
    
    def _select_token(self, projection: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Select the token position for signal computation.
        
        According to the paper (Appendix D.1), the last token is used by default.
        
        Args:
            projection: Tensor of shape (batch_size, seq_len, hidden_dim)
            attention_mask: Optional attention mask
            
        Returns:
            Tensor of shape (batch_size, hidden_dim)
        """
        if self.token_position == 'last':
            if attention_mask is not None:
                # Find the last non-padding token
                seq_lens = attention_mask.sum(dim=1) - 1
                batch_size = projection.shape[0]
                return projection[torch.arange(batch_size), seq_lens.long()]
            else:
                return projection[:, -1, :]
        elif self.token_position == 'first':
            return projection[:, 0, :]
        elif self.token_position == 'mean':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                return (projection * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                return projection.mean(dim=1)
        else:
            raise ValueError(f"Unknown token_position: {self.token_position}")
    
    def extract_signals(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Extract signals from all adapters using a single forward pass.
        
        This implements the probe pass from Algorithm 1 in the paper.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            inputs_embeds: Optional input embeddings (for multimodal)
            pixel_values: Optional pixel values for vision models
            **kwargs: Additional model inputs
            
        Returns:
            Dict mapping adapter names to signal scores
        """
        # Find target LoRA layers
        target_layers = self._find_target_block_lora_layers()
        
        if not target_layers:
            logger.warning("No LoRA layers found. Returning uniform signals.")
            return {name: 1.0 for name in self.adapter_names}
        
        logger.info(f"Found {len(target_layers)} LoRA layers in target block")
        
        # Prepare model inputs
        # Note: Cannot specify both input_ids and inputs_embeds at the same time
        # For vision-language models, inputs_embeds already contains the full input
        # including image features, so we should NOT pass input_ids when inputs_embeds exists
        model_inputs = {}
        if inputs_embeds is not None:
            # Use inputs_embeds (already contains image features)
            model_inputs['inputs_embeds'] = inputs_embeds
            # Do NOT add input_ids when using inputs_embeds
        elif input_ids is not None:
            model_inputs['input_ids'] = input_ids
        
        if attention_mask is not None:
            model_inputs['attention_mask'] = attention_mask
        if pixel_values is not None:
            model_inputs['pixel_values'] = pixel_values
        model_inputs.update(kwargs)
        
        # Run forward pass with hooks
        with self._hook_context(target_layers):
            with torch.no_grad():
                try:
                    # Run forward pass
                    _ = self.model(**model_inputs)
                except Exception as e:
                    logger.error(f"Forward pass failed: {e}")
                    return {name: 1.0 for name in self.adapter_names}
        
        # Compute signals for each adapter
        signals = {}
        
        for adapter_name in self.adapter_names:
            adapter_projections = self._lora_projections.get(adapter_name, {})
            
            if not adapter_projections:
                logger.warning(f"No projections captured for adapter: {adapter_name}")
                signals[adapter_name] = 0.0
                continue
            
            # Aggregate projections from all layers
            all_projections = []
            for layer_name, proj_list in adapter_projections.items():
                for proj in proj_list:
                    # Select token position
                    selected = self._select_token(proj, attention_mask)
                    all_projections.append(selected)
            
            if not all_projections:
                signals[adapter_name] = 0.0
                continue
            
            # Stack and compute signal
            stacked = torch.stack(all_projections, dim=0).mean(dim=0)  # Average across layers
            signals[adapter_name] = compute_signal(stacked, self.signal_type)
        
        # Clear stored projections
        self._lora_projections = {}
        
        # Apply baseline calibration if enabled
        if self.use_baseline_calibration and self._baseline_computed and self._baseline_signals:
            signals = self.calibrate_signals(signals, method='ratio')
        
        return signals
    
    def extract_signals_manual(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Extract signals by manually computing LoRA projections.
        
        This is a more direct implementation that doesn't require a full forward pass.
        Useful when hidden states are already available.
        
        Args:
            hidden_states: Hidden states from target block, shape (batch, seq_len, hidden_dim)
            attention_mask: Optional attention mask
            
        Returns:
            Dict mapping adapter names to signal scores
        """
        target_layers = self._find_target_block_lora_layers()
        
        if not target_layers:
            return {name: 1.0 for name in self.adapter_names}
        
        signals = {}
        
        for adapter_name in self.adapter_names:
            adapter_signals = []
            
            for layer_name, layer in target_layers.items():
                if not hasattr(layer, 'lora_A') or adapter_name not in layer.lora_A:
                    continue
                
                lora_A = layer.lora_A[adapter_name]
                lora_B = layer.lora_B[adapter_name]
                scaling = layer.scaling.get(adapter_name, 1.0)
                
                # Compute LoRA projection
                with torch.no_grad():
                    x = hidden_states.to(lora_A.weight.dtype)
                    lora_output = lora_B(lora_A(x)) * scaling
                    
                    # Select token position
                    selected = self._select_token(lora_output, attention_mask)
                    adapter_signals.append(compute_signal(selected, self.signal_type))
            
            if adapter_signals:
                signals[adapter_name] = sum(adapter_signals) / len(adapter_signals)
            else:
                signals[adapter_name] = 0.0
        
        return signals


class EmbeddingSignalExtractor:
    """
    Lightweight signal extractor based on embedding similarity.
    
    This is a faster alternative that doesn't require a forward pass.
    It computes cosine similarity between query embedding and precomputed
    adapter embeddings.
    
    Args:
        adapter_embeddings: Dict mapping adapter names to their embeddings
    """
    
    def __init__(self, adapter_embeddings: Dict[str, torch.Tensor]):
        self.adapter_embeddings = adapter_embeddings
    
    def extract_signals(
        self,
        query_embedding: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute similarity-based signals.
        
        Args:
            query_embedding: Embedding of the input query
            
        Returns:
            Dict mapping adapter names to similarity scores
        """
        signals = {}
        query_embedding = query_embedding.float()
        
        if query_embedding.dim() > 1:
            query_embedding = query_embedding.mean(dim=0)  # Average over sequence
        
        for name, adapter_emb in self.adapter_embeddings.items():
            adapter_emb = adapter_emb.to(query_embedding.device).float()
            
            if adapter_emb.dim() > 1:
                adapter_emb = adapter_emb.mean(dim=0)
            
            # Cosine similarity
            similarity = F.cosine_similarity(
                query_embedding.unsqueeze(0),
                adapter_emb.unsqueeze(0)
            ).item()
            
            signals[name] = max(0, similarity)  # Ensure non-negative
        
        return signals


def compute_uniform_signals(adapter_names: List[str]) -> Dict[str, float]:
    """
    Return uniform signals for all adapters.
    Used as baseline or when no signal extraction is available.
    
    Args:
        adapter_names: List of adapter names
        
    Returns:
        Dict with uniform signal of 1.0 for each adapter
    """
    return {name: 1.0 for name in adapter_names}
