#!/usr/bin/env python3
"""
Model factory for creating different model architectures.
"""

import torch.nn as nn
from .residual import EmitterEncoder
from .ft_transformer import FTTransformer
from .deep_ft_transformer import DeepFTTransformer

def get_model(model_type: str, in_dim: int, emb_dim: int, **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('residual', 'ft_transformer', 'deep_ft')
        in_dim: Input dimension
        emb_dim: Embedding dimension
        **kwargs: Additional model parameters
        
    Returns:
        Model instance
    """
    if model_type == 'residual':
        return EmitterEncoder(in_dim, emb_dim, **kwargs)
    elif model_type == 'ft_transformer':
        return FTTransformer(in_dim, emb_dim, **kwargs)
    elif model_type == 'deep_ft':
        return DeepFTTransformer(in_dim, emb_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Available models
AVAILABLE_MODELS = {
    'residual': EmitterEncoder,
    'ft_transformer': FTTransformer,
    'deep_ft': DeepFTTransformer
} 