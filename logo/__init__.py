# LOGO: LoRA on the Go for InternVL2
# 
# Implementation based on: "LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging"
# Paper: https://arxiv.org/abs/2511.07129
#
# This module provides dynamic LoRA selection and merging for InternVL2.

from .signal_extractor import (
    LOGOSignalExtractor,
    EmbeddingSignalExtractor,
    compute_signal,
    compute_norm_signal,
    compute_entropy_signal,
    normalize_weights,
    select_top_k,
    compute_uniform_signals
)

from .logo_engine import (
    LOGOEngine,
    LoRAConfig,
    load_lora_configs,
    create_logo_engine
)

__all__ = [
    # Signal extraction
    'LOGOSignalExtractor',
    'EmbeddingSignalExtractor',
    'compute_signal',
    'compute_norm_signal',
    'compute_entropy_signal',
    'normalize_weights',
    'select_top_k',
    'compute_uniform_signals',
    # Engine
    'LOGOEngine',
    'LoRAConfig',
    'load_lora_configs',
    'create_logo_engine',
]
