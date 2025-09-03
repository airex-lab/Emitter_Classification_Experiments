#!/usr/bin/env python3
"""
Feature Tokenizer Transformer for tabular data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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