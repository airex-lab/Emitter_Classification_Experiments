#!/usr/bin/env python3
"""
Model architectures for emitter classification experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ResidualBlock(nn.Module):
    """Residual block with dropout for regularization."""
    
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return self.relu(out + residual)

class EmitterEncoder(nn.Module):
    """Residual network encoder for emitter classification."""
    
    def __init__(self, in_dim: int, emb_dim: int, hidden_dim: int = 64, 
                 num_blocks: int = 2, dropout: float = 0.3):
        """
        Args:
            in_dim: Input feature dimension
            emb_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension
            num_blocks: Number of residual blocks
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = [nn.Linear(in_dim, hidden_dim)]
        
        # Add residual blocks
        for _ in range(num_blocks):
            layers.append(ResidualBlock(hidden_dim, dropout))
            
        layers.append(nn.Linear(hidden_dim, emb_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=1)

class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer for tabular data."""
    
    def __init__(self, num_feats: int, dim: int, heads: int = 8, 
                 layers: int = 3, dropout: float = 0.2):
        """
        Args:
            num_feats: Number of input features
            dim: Embedding dimension
            heads: Number of attention heads
            layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Feature tokenizer
        self.token = nn.Parameter(torch.randn(num_feats, dim))
        self.cls = nn.Parameter(torch.randn(1, dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        
        # Output normalization
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [batch_size, num_feats]
            
        Returns:
            Normalized embeddings [batch_size, dim]
        """
        B = x.size(0)
        
        # Create feature tokens
        tokens = self.token * x.unsqueeze(-1)  # [B, F, D]
        
        # Add CLS token
        tokens = torch.cat([self.cls.expand(B, -1, -1), tokens], dim=1)
        
        # Pass through transformer
        out = self.encoder(tokens)[:, 0]  # Take CLS token
        
        # Normalize output
        return F.normalize(self.norm(out), p=2, dim=1)

class DeepFTTransformer(nn.Module):
    """Deep FT-Transformer with more layers for complex patterns."""
    
    def __init__(self, num_feats: int, dim: int = 192, heads: int = 8, 
                 layers: int = 6, dropout: float = 0.2):
        """
        Args:
            num_feats: Number of input features
            dim: Embedding dimension
            heads: Number of attention heads
            layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Feature tokenizer
        self.token = nn.Parameter(torch.randn(num_feats, dim))
        self.cls = nn.Parameter(torch.randn(1, dim))
        
        # Deep transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        
        # Output normalization
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [batch_size, num_feats]
            
        Returns:
            Normalized embeddings [batch_size, dim]
        """
        B = x.size(0)
        
        # Create feature tokens
        tokens = self.token * x.unsqueeze(-1)  # [B, F, D]
        
        # Add CLS token
        tokens = torch.cat([self.cls.expand(B, -1, -1), tokens], dim=1)
        
        # Pass through transformer
        out = self.encoder(tokens)[:, 0]  # Take CLS token
        
        # Normalize output
        return F.normalize(self.norm(out), p=2, dim=1)

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